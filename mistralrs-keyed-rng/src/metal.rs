use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex},
};

use candle_core::{DType, Device, Error, MetalStorage, Result, Shape, Storage, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::{MTLCompileOptions, MTLComputePipelineState, MTLMathMode, MTLSize};

use crate::{Purpose, SequenceKey};

pub const INVALID_TOKEN: u32 = u32::MAX;
const RECORD_WIDTH: usize = 3;
const EVENT_WIDTH: usize = 4;
const STATE_WIDTH: usize = 3;
const THREADS: usize = 64;
const SELECT_THREADS: usize = 256;
const SIMD_WIDTH: usize = 32;
const LOGITS_TILE: usize = 1024;
const LOGIT_TILE_WIDTH: usize = 4;
const TOPK_BLOCK: usize = 1024;
pub const MAX_PARTIAL_TOP_K: usize = 128;
pub const MAX_SAMPLING_BATCH: usize = 128;
const SOURCE: &str = concat!(
    include_str!("threefry.metal"),
    "\n",
    include_str!("selection.metal"),
    "\n",
    include_str!("topk.metal"),
    "\n",
    include_str!("sampling.metal")
);
type Pipelines = HashMap<(u64, &'static str), ComputePipeline>;
static PIPELINES: LazyLock<Mutex<Pipelines>> = LazyLock::new(|| Mutex::new(HashMap::new()));

fn pipeline(device: &Device, name: &'static str) -> Result<ComputePipeline> {
    let device = device.as_metal_device()?;
    let mut pipelines = PIPELINES.lock().map_err(Error::msg)?;
    let key = (device.device().registry_id(), name);
    if let Some(pipeline) = pipelines.get(&key) {
        return Ok(pipeline.clone());
    }
    let opts = MTLCompileOptions::new();
    opts.setMathMode(MTLMathMode::Safe);
    let library = device
        .device()
        .new_library_with_source(SOURCE, Some(&opts))
        .map_err(Error::wrap)?;
    let function = library.get_function(name, None).map_err(Error::wrap)?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(Error::wrap)?;
    if matches!(
        name,
        "rng_select" | "rng_logits" | "logits_tiles" | "logits_finish" | "candidates_topk_blocks"
    ) && pipeline.as_ref().threadExecutionWidth() != SIMD_WIDTH
    {
        candle_core::bail!("keyed Metal sampling requires 32-lane SIMD groups");
    }
    pipelines.insert(key, pipeline.clone());
    Ok(pipeline)
}

fn allocate(device: &Device, shape: impl Into<Shape>, dtype: DType) -> Result<Tensor> {
    let shape = shape.into();
    let dev = device.as_metal_device()?;
    let buffer = dev.new_buffer(shape.elem_count(), dtype, "keyed-rng")?;
    Ok(Tensor::from((
        Storage::Metal(MetalStorage::new(
            buffer,
            dev.clone(),
            shape.elem_count(),
            dtype,
        )),
        shape,
    )))
}

fn allocate_records(device: &Device, rows: usize) -> Result<Tensor> {
    let dev = device.as_metal_device()?;
    let count = rows * RECORD_WIDTH;
    let buffer = dev.allocate_buffer(count * DType::U32.size_in_bytes())?;
    Ok(Tensor::from((
        Storage::Metal(MetalStorage::new(buffer, dev.clone(), count, DType::U32)),
        (rows, RECORD_WIDTH),
    )))
}

#[inline(never)]
fn readback_boundary(device: &Device) -> Result<()> {
    device.as_metal_device()?.flush_and_wait_current()
}

fn dispatch(
    name: &'static str,
    inputs: &[&Tensor],
    outputs: &[&Tensor],
    ints: &[u32],
    floats: &[f32],
    threads: usize,
) -> Result<()> {
    let device = inputs[0].device();
    let pipeline = pipeline(device, name)?;
    let mut bindings = Vec::new();
    for tensor in inputs.iter().chain(outputs) {
        if !tensor.is_contiguous() || !device.same_device(tensor.device()) {
            candle_core::bail!(
                "keyed Metal RNG requires contiguous tensors on the same Candle device"
            );
        }
        let (storage, layout) = tensor.storage_and_layout();
        let Storage::Metal(storage) = &*storage else {
            candle_core::bail!("keyed RNG requires Metal storage");
        };
        bindings.push((
            storage.buffer().clone(),
            layout.start_offset() * tensor.dtype().size_in_bytes(),
        ));
    }
    let dev = device.as_metal_device()?;
    let encoder = dev.command_encoder()?;
    let encoder: &candle_metal_kernels::metal::ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    for (index, (buffer, offset)) in bindings.iter().enumerate() {
        if index < inputs.len() {
            encoder.set_input_buffer(index, Some(buffer), *offset);
        } else {
            encoder.set_input_buffer(index, Some(buffer), *offset);
            encoder.set_output_buffer(index, Some(buffer), *offset);
        }
    }
    let index = bindings.len();
    if !ints.is_empty() {
        encoder.set_bytes_directly(index, std::mem::size_of_val(ints), ints.as_ptr().cast());
    }
    if !floats.is_empty() {
        encoder.set_bytes_directly(
            index + 1,
            std::mem::size_of_val(floats),
            floats.as_ptr().cast(),
        );
    }
    encoder.dispatch_threads(
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads.min(
                if matches!(
                    name,
                    "rng_select"
                        | "rng_logits"
                        | "logits_tiles"
                        | "logits_finish"
                        | "candidates_topk_blocks"
                ) {
                    SELECT_THREADS
                } else {
                    THREADS
                },
            ),
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn probe(events: &Tensor) -> Result<Tensor> {
    let (rows, width) = events.dims2()?;
    if width != EVENT_WIDTH || events.dtype() != DType::U32 || rows == 0 {
        candle_core::bail!("RNG events must be nonempty u32 [rows, 4]");
    }
    let output = allocate(events.device(), (rows, EVENT_WIDTH), DType::U32)?;
    dispatch("rng_probe", &[events], &[&output], &[], &[], rows)?;
    Ok(output)
}

pub fn candidates(probs: &Tensor, sorted: bool) -> Result<(Tensor, Tensor)> {
    let (rows, width) = probs.dims2()?;
    if probs.dtype() != DType::F32
        || rows == 0
        || width == 0
        || probs.elem_count() > u32::MAX as usize / 2
    {
        candle_core::bail!("invalid Metal candidate dimensions");
    }
    let mut values = allocate(probs.device(), probs.shape(), DType::F32)?;
    let mut ids = allocate(probs.device(), probs.shape(), DType::U32)?;
    dispatch(
        "candidates_init",
        &[probs],
        &[&values, &ids],
        &[width as u32],
        &[],
        probs.elem_count(),
    )?;
    if sorted {
        let mut next_values = allocate(probs.device(), probs.shape(), DType::F32)?;
        let mut next_ids = allocate(probs.device(), probs.shape(), DType::U32)?;
        let mut span = 1;
        while span < width {
            dispatch(
                "candidates_merge",
                &[&values, &ids],
                &[&next_values, &next_ids],
                &[width as u32, span as u32],
                &[],
                probs.elem_count(),
            )?;
            std::mem::swap(&mut values, &mut next_values);
            std::mem::swap(&mut ids, &mut next_ids);
            span *= 2;
        }
    }
    Ok((values, ids))
}

pub fn top_candidates(probs: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    let (rows, width) = probs.dims2()?;
    if probs.dtype() != DType::F32
        || rows == 0
        || width == 0
        || k == 0
        || k > MAX_PARTIAL_TOP_K
        || probs.elem_count() > u32::MAX as usize / 2
    {
        candle_core::bail!("invalid partial Metal top-k inputs");
    }
    let k = k.min(width);
    let mut groups = width.div_ceil(TOPK_BLOCK);
    let mut values = allocate(probs.device(), (rows, groups * k), DType::F32)?;
    let mut ids = allocate(probs.device(), values.shape(), DType::U32)?;
    dispatch(
        "candidates_topk_blocks",
        &[probs],
        &[&values, &ids],
        &[width as u32, k as u32, groups as u32, 0],
        &[],
        rows * groups * SELECT_THREADS,
    )?;
    while groups > 1 {
        let pairs = groups.div_ceil(2);
        let next_values = allocate(probs.device(), (rows, pairs * k), DType::F32)?;
        let next_ids = allocate(probs.device(), next_values.shape(), DType::U32)?;
        dispatch(
            "candidates_topk_merge",
            &[&values, &ids],
            &[&next_values, &next_ids],
            &[k as u32, groups as u32, 0, 0],
            &[],
            rows * pairs * k * 2,
        )?;
        values = next_values;
        ids = next_ids;
        groups = pairs;
    }
    Ok((values, ids))
}

#[derive(Clone, Copy, Debug)]
pub struct Filter {
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub greedy: bool,
}

pub struct LogitsSampling {
    pub key: SequenceKey,
    pub attempt: u32,
    pub temperature: f32,
    pub frequency: f32,
    pub presence: f32,
    pub repetition: f32,
    pub min_p: f32,
    pub greedy: bool,
}

pub struct Selection {
    records: Tensor,
    tokens: Tensor,
}

impl Selection {
    fn into_rows(self) -> Result<Vec<Self>> {
        (0..self.records.dim(0)?)
            .map(|row| {
                Ok(Self {
                    records: self.records.narrow(0, row, 1)?,
                    tokens: self.tokens.narrow(0, row, 1)?,
                })
            })
            .collect()
    }

    pub fn tokens(&self) -> &Tensor {
        &self.tokens
    }

    pub fn readback(&self) -> Result<Vec<(u32, f32)>> {
        readback_boundary(self.records.device())?;
        self.read_completed().into_iter().collect()
    }

    pub fn readback_batch(selections: &[&Self]) -> Result<Vec<Result<(u32, f32)>>> {
        let Some(first) = selections.first() else {
            return Ok(Vec::new());
        };
        if selections.iter().any(|selection| {
            !first
                .records
                .device()
                .same_device(selection.records.device())
        }) {
            candle_core::bail!("keyed readback requires one Candle device");
        }
        readback_boundary(first.records.device())?;
        Ok(selections
            .iter()
            .flat_map(|selection| selection.read_completed())
            .collect())
    }

    fn read_completed(&self) -> Vec<Result<(u32, f32)>> {
        let (storage, layout) = self.records.storage_and_layout();
        let Storage::Metal(storage) = &*storage else {
            unreachable!()
        };
        // These shared buffers are immutable after the producer completes at readback_boundary.
        let words = unsafe {
            std::slice::from_raw_parts(
                storage
                    .buffer()
                    .contents()
                    .cast::<u32>()
                    .add(layout.start_offset()),
                self.records.elem_count(),
            )
        };
        words
            .chunks_exact(RECORD_WIDTH)
            .map(|row| {
                if row[2] != 0 || row[0] == INVALID_TOKEN {
                    candle_core::bail!("invalid or empty keyed sampling distribution");
                }
                Ok((row[0], f32::from_bits(row[1])))
            })
            .collect()
    }
}

/// Candidate rows must be sorted by descending weight when nucleus filtering is enabled.
pub fn select(
    weights: &Tensor,
    ids: &Tensor,
    reporting: &Tensor,
    events: &Tensor,
    filter: Filter,
) -> Result<Selection> {
    let (rows, width) = weights.dims2()?;
    if rows == 0
        || width == 0
        || width > u32::MAX as usize
        || weights.dtype() != DType::F32
        || reporting.dtype() != DType::F32
        || ids.dtype() != DType::U32
        || events.dtype() != DType::U32
        || ids.dims() != weights.dims()
        || reporting.dims() != weights.dims()
        || events.dims() != [rows, EVENT_WIDTH]
        || !filter.top_p.is_finite()
        || !filter.min_p.is_finite()
    {
        candle_core::bail!("invalid keyed sampling tensors or filters");
    }
    let records = allocate_records(weights.device(), rows)?;
    let tokens = allocate(weights.device(), (rows, 1), DType::U32)?;
    dispatch(
        "rng_select",
        &[weights, ids, reporting, events],
        &[&records, &tokens],
        &[
            width as u32,
            filter.top_k.min(width) as u32,
            u32::from(filter.greedy),
            0,
        ],
        &[filter.top_p, filter.min_p],
        rows * SELECT_THREADS,
    )?;
    Ok(Selection { records, tokens })
}

pub struct DeviceHistory {
    history: Tensor,
    counts: Tensor,
    state: Tensor,
    last_token: Option<Tensor>,
    committed_len: usize,
    capacity: usize,
    vocab: usize,
    stop_tokens: (Vec<u32>, Tensor),
    batch: Option<(Arc<HistoryBatch>, usize)>,
}

struct HistoryBatch {
    counts: Tensor,
    state: Tensor,
    rows: usize,
}

impl DeviceHistory {
    fn batch_tensors(histories: &mut [&mut Self]) -> Result<(Tensor, Tensor)> {
        if let Some((batch, _)) = &histories[0].batch {
            if batch.rows == histories.len()
                && histories.iter().enumerate().all(|(row, history)| {
                    history
                        .batch
                        .as_ref()
                        .is_some_and(|(other, index)| *index == row && Arc::ptr_eq(batch, other))
                })
            {
                return Ok((batch.counts.clone(), batch.state.clone()));
            }
        }
        let counts = Tensor::cat(&histories.iter().map(|h| &h.counts).collect::<Vec<_>>(), 0)?;
        let state = Tensor::cat(&histories.iter().map(|h| &h.state).collect::<Vec<_>>(), 0)?;
        let batch = Arc::new(HistoryBatch {
            counts,
            state,
            rows: histories.len(),
        });
        for (row, history) in histories.iter_mut().enumerate() {
            history.counts = batch.counts.narrow(0, row * 2, 2)?;
            history.state = batch.state.narrow(0, row * STATE_WIDTH, STATE_WIDTH)?;
            history.batch = Some((batch.clone(), row));
        }
        Ok((batch.counts.clone(), batch.state.clone()))
    }

    pub fn sample_batch(
        logits: &Tensor,
        histories: &mut [&mut Self],
        params: &[LogitsSampling],
        filter: Filter,
    ) -> Result<Vec<Selection>> {
        let (rows, width) = logits.dims2()?;
        if rows == 0
            || rows > MAX_SAMPLING_BATCH
            || rows != histories.len()
            || rows != params.len()
            || logits.dtype() != DType::F32
            || logits.elem_count() > u32::MAX as usize / 2
            || histories
                .iter()
                .any(|h| h.vocab != width || !h.state.device().same_device(logits.device()))
            || params.iter().any(|p| {
                !p.temperature.is_finite()
                    || p.temperature <= 0.0
                    || !p.repetition.is_finite()
                    || p.repetition <= 0.0
            })
            || !filter.top_p.is_finite()
            || !filter.min_p.is_finite()
        {
            candle_core::bail!("invalid batched Metal sampling inputs");
        }
        let (counts, state) = Self::batch_tensors(histories)?;
        let floats = params
            .iter()
            .flat_map(|p| [p.temperature.recip(), p.frequency, p.presence, p.repetition])
            .collect::<Vec<_>>();
        let unfiltered = filter.top_k == 0
            && !(filter.top_p > 0.0 && filter.top_p < 1.0)
            && !(filter.min_p > 0.0 && filter.min_p < 1.0);
        if filter.greedy || unfiltered {
            let tile_count = width.div_ceil(LOGITS_TILE);
            let weights = allocate(logits.device(), (rows, width), DType::F32)?;
            let tiles = allocate(
                logits.device(),
                (rows * tile_count, LOGIT_TILE_WIDTH),
                DType::F32,
            )?;
            let records = allocate_records(logits.device(), rows)?;
            let tokens = allocate(logits.device(), (rows, 1), DType::U32)?;
            let ints = params
                .iter()
                .flat_map(|p| {
                    let key = p.key.words(Purpose::Generation);
                    [
                        width as u32,
                        u32::from(filter.greedy),
                        key[0],
                        key[1],
                        p.attempt,
                    ]
                })
                .collect::<Vec<_>>();
            dispatch(
                "logits_tiles",
                &[logits, &counts],
                &[&weights, &tiles],
                &[width as u32],
                &floats,
                rows * tile_count * SELECT_THREADS,
            )?;
            dispatch(
                "logits_finish",
                &[&weights, &tiles, &state],
                &[&records, &tokens],
                &ints,
                &[],
                rows * SELECT_THREADS,
            )?;
            return Selection { records, tokens }.into_rows();
        }
        let scores = allocate(logits.device(), (rows, width), DType::F32)?;
        let events = allocate(logits.device(), (rows, EVENT_WIDTH), DType::U32)?;
        let ints = params
            .iter()
            .flat_map(|p| {
                let key = p.key.words(Purpose::Generation);
                [width as u32, key[0], key[1], p.attempt]
            })
            .collect::<Vec<_>>();
        dispatch(
            "batch_logits",
            &[logits, &counts, &state],
            &[&scores, &events],
            &ints,
            &floats,
            rows * width,
        )?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let sorted = filter.top_k > 0 || (filter.top_p > 0.0 && filter.top_p < 1.0);
        let (weights, ids) = if filter.top_k > 0 && filter.top_k <= MAX_PARTIAL_TOP_K {
            top_candidates(&probs, filter.top_k)?
        } else {
            candidates(&probs, sorted)?
        };
        select(&weights, &ids, &weights, &events, filter)?.into_rows()
    }

    pub fn sample_logits(&self, logits: &Tensor, params: LogitsSampling) -> Result<Selection> {
        if logits.dims() != [self.vocab]
            || logits.dtype() != DType::F32
            || !params.temperature.is_finite()
            || params.temperature <= 0.0
            || !params.repetition.is_finite()
            || params.repetition <= 0.0
            || !params.min_p.is_finite()
        {
            candle_core::bail!("invalid fused Metal sampling inputs");
        }
        let records = allocate_records(logits.device(), 1)?;
        let tokens = allocate(logits.device(), (1, 1), DType::U32)?;
        let key = params.key.words(Purpose::Generation);
        if params.greedy || !(params.min_p > 0.0 && params.min_p < 1.0) {
            let tile_count = self.vocab.div_ceil(LOGITS_TILE);
            let weights = allocate(logits.device(), self.vocab, DType::F32)?;
            let tiles = allocate(logits.device(), (tile_count, LOGIT_TILE_WIDTH), DType::F32)?;
            dispatch(
                "logits_tiles",
                &[logits, &self.counts],
                &[&weights, &tiles],
                &[self.vocab as u32],
                &[
                    params.temperature.recip(),
                    params.frequency,
                    params.presence,
                    params.repetition,
                ],
                tile_count * SELECT_THREADS,
            )?;
            dispatch(
                "logits_finish",
                &[&weights, &tiles, &self.state],
                &[&records, &tokens],
                &[
                    self.vocab as u32,
                    u32::from(params.greedy),
                    key[0],
                    key[1],
                    params.attempt,
                ],
                &[],
                SELECT_THREADS,
            )?;
            return Ok(Selection { records, tokens });
        }
        dispatch(
            "rng_logits",
            &[logits, &self.counts, &self.state],
            &[&records, &tokens],
            &[
                self.vocab as u32,
                u32::from(params.greedy),
                key[0],
                key[1],
                params.attempt,
            ],
            &[
                params.temperature.recip(),
                params.frequency,
                params.presence,
                params.repetition,
                params.min_p,
            ],
            SELECT_THREADS,
        )?;
        Ok(Selection { records, tokens })
    }

    pub fn new(
        tokens: &[u32],
        prompt_len: usize,
        capacity: usize,
        vocab: usize,
        device: &Device,
    ) -> Result<Self> {
        if prompt_len > tokens.len()
            || tokens.len() > capacity
            || capacity > u32::MAX as usize
            || vocab == 0
            || vocab > u32::MAX as usize
        {
            candle_core::bail!("invalid device history dimensions");
        }
        let source = Tensor::new(tokens, device)?;
        let history = Tensor::zeros(capacity, DType::U32, device)?;
        let counts = Tensor::zeros((2, vocab), DType::U32, device)?;
        let state = allocate(device, STATE_WIDTH, DType::U32)?;
        dispatch(
            "history_init",
            &[&source],
            &[&history, &counts, &state],
            &[tokens.len() as u32, prompt_len as u32, vocab as u32, 0],
            &[],
            1,
        )?;
        Ok(Self {
            history,
            counts,
            state,
            last_token: None,
            committed_len: tokens.len(),
            capacity,
            vocab,
            stop_tokens: (Vec::new(), Tensor::new(&[INVALID_TOKEN], device)?),
            batch: None,
        })
    }

    pub fn matches(&self, len: usize, vocab: usize, device: &Device) -> bool {
        self.committed_len == len
            && self.vocab == vocab
            && self.history.device().same_device(device)
    }

    pub fn next_input(&self, len: usize, device: &Device) -> Option<&Tensor> {
        (self.committed_len == len && self.history.device().same_device(device))
            .then_some(self.last_token.as_ref())
            .flatten()
    }

    pub fn state(&self) -> &Tensor {
        &self.state
    }

    pub fn tokens(&self) -> Result<Tensor> {
        self.history.narrow(0, 0, self.committed_len)
    }

    pub fn event(&self, key: SequenceKey, purpose: Purpose, attempt: u32) -> Result<Tensor> {
        let words = key.words(purpose);
        let output = allocate(self.state.device(), (1, EVENT_WIDTH), DType::U32)?;
        dispatch(
            "rng_event",
            &[&self.state],
            &[&output],
            &[words[0], words[1], attempt, 0],
            &[],
            1,
        )?;
        Ok(output)
    }

    pub fn penalties(
        &self,
        logits: &Tensor,
        frequency: f32,
        presence: f32,
        repetition: f32,
    ) -> Result<Tensor> {
        if logits.dims() != [self.vocab]
            || logits.dtype() != DType::F32
            || !repetition.is_finite()
            || repetition <= 0.0
        {
            candle_core::bail!("invalid device history penalty inputs");
        }
        let output = allocate(logits.device(), self.vocab, DType::F32)?;
        dispatch(
            "history_penalties",
            &[logits, &self.counts],
            &[&output],
            &[self.vocab as u32],
            &[frequency, presence, repetition, 0.0],
            self.vocab,
        )?;
        Ok(output)
    }

    pub fn commit(&mut self, selection: &Selection, stop_tokens: &Tensor) -> Result<()> {
        if selection.records.dims() != [1, RECORD_WIDTH]
            || stop_tokens.rank() != 1
            || stop_tokens.dtype() != DType::U32
            || self.committed_len >= self.capacity
        {
            candle_core::bail!("invalid device history commit");
        }
        dispatch(
            "history_commit",
            &[&selection.records, stop_tokens],
            &[&self.history, &self.counts, &self.state],
            &[
                self.vocab as u32,
                self.capacity as u32,
                stop_tokens.elem_count() as u32,
                0,
            ],
            &[],
            1,
        )?;
        self.last_token = Some(selection.tokens.clone());
        self.committed_len += 1;
        Ok(())
    }

    pub fn commit_with_stop_tokens(&mut self, selection: &Selection, tokens: &[u32]) -> Result<()> {
        if self.stop_tokens.0 != tokens {
            let values = if tokens.is_empty() {
                &[INVALID_TOKEN]
            } else {
                tokens
            };
            self.stop_tokens = (tokens.to_vec(), Tensor::new(values, self.state.device())?);
        }
        let stops = self.stop_tokens.1.clone();
        self.commit(selection, &stops)
    }
}
