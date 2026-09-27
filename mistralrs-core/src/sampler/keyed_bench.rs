use std::{hint::black_box, time::Instant};

use mistralrs_keyed_rng::{
    metal::{DeviceHistory, Selection},
    Purpose, SequenceKey,
};
use rand::SeedableRng;
use rayon::iter::IntoParallelIterator;

use super::*;

const PROMPT_LEN: usize = 128;
const WARMUP_STEPS: usize = 8;
const DEFAULT_STEPS: usize = 32;
const DEFAULT_REPEATS: usize = 5;
const SEED: u64 = 42;
const MODES: [&str; 6] = [
    "cpu_legacy",
    "cpu_keyed",
    "readback_cpu_legacy",
    "readback_cpu_keyed",
    "metal_compact",
    "metal_queued",
];

struct Case {
    device: Device,
    cpu: Tensor,
    metal: Tensor,
    sampler: Sampler,
    batch: usize,
    vocab: usize,
}

impl Case {
    fn new(device: &Device, vocab: usize, batch: usize, filter: &str) -> Result<Self> {
        let logits = (0..vocab * batch)
            .map(|i| {
                let mut x = (i as u32).wrapping_add(0x9e37_79b9);
                x = (x ^ (x >> 16)).wrapping_mul(0x85eb_ca6b);
                x = (x ^ (x >> 13)).wrapping_mul(0xc2b2_ae35);
                x ^= x >> 16;
                (x >> 8) as f32 * (16.0 / 16_777_216.0) - 8.0
            })
            .collect::<Vec<_>>();
        let cpu = Tensor::from_vec(logits, (batch, vocab), &Device::Cpu)?;
        let metal = cpu.to_device(device)?;
        let sampler = Sampler::new(
            (filter != "greedy").then_some(0.8),
            0,
            None,
            None,
            None,
            None,
            None,
            if filter == "topk40_p90" { 40 } else { -1 },
            if filter == "topk40_p90" { 0.9 } else { 1.0 },
            0.0,
            HashMap::new(),
            vec![],
        )
        .map_err(Error::msg)?;
        Ok(Self {
            device: device.clone(),
            cpu,
            metal,
            sampler,
            batch,
            vocab,
        })
    }

    fn run(&self, mode: &str, steps: usize, audit: bool) -> Result<(f64, Vec<Vec<u32>>)> {
        let prompt = (0..PROMPT_LEN as u32).collect::<Vec<_>>();
        let mut histories = (0..self.batch)
            .map(|_| {
                DeviceHistory::new(
                    &prompt,
                    PROMPT_LEN,
                    PROMPT_LEN + steps + 1,
                    self.vocab,
                    &self.device,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut contexts = vec![prompt; self.batch];
        let keys = (0..self.batch)
            .map(|row| SequenceKey::new(SEED + row as u64))
            .collect::<Vec<_>>();
        let rngs = (0..self.batch)
            .map(|row| Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(SEED + row as u64))))
            .collect::<Vec<_>>();
        let metal_rows = self.metal.chunk(self.batch, 0)?;
        let metal_rows = metal_rows
            .iter()
            .map(|row| row.squeeze(0))
            .collect::<Result<Vec<_>>>()?;
        self.device.synchronize()?;
        if audit {
            eprintln!("AUDIT_BEGIN mode={mode} steps={steps} batch={}", self.batch);
        }
        let start = Instant::now();
        for position in 0..steps {
            if mode.starts_with("metal_") {
                let selections = if self.batch > 1 {
                    let params = keys
                        .iter()
                        .map(|key| self.sampler.keyed_metal_params(*key))
                        .collect::<Vec<_>>();
                    let mut histories = histories.iter_mut().collect::<Vec<_>>();
                    DeviceHistory::sample_batch(
                        &Tensor::stack(&metal_rows, 0)?,
                        &mut histories,
                        &params,
                        self.sampler.keyed_metal_filter(),
                    )?
                } else {
                    let mut selections = Vec::with_capacity(self.batch);
                    for row in 0..self.batch {
                        let (selection, _) = self.sampler.sample_keyed_metal(
                            &metal_rows[row],
                            &histories[row],
                            keyed::KeyedMetalContext {
                                key: keys[row],
                                attempt: 0,
                                return_logprobs: false,
                            },
                        )?;
                        selections.push(selection);
                    }
                    selections
                };
                for (history, selection) in histories.iter_mut().zip(&selections) {
                    history.commit_with_stop_tokens(selection, &[])?;
                }
                if mode == "metal_compact" {
                    let refs = selections.iter().collect::<Vec<_>>();
                    let tokens = Selection::readback_batch(&refs)?;
                    for (context, token) in contexts.iter_mut().zip(tokens) {
                        context.push(token?.0);
                    }
                }
            } else {
                let logits = if mode.starts_with("readback_") {
                    self.metal.to_device(&Device::Cpu)?
                } else {
                    self.cpu.clone()
                };
                let rows = logits.chunk(self.batch, 0)?;
                let tokens = (0..self.batch)
                    .into_par_iter()
                    .map(|row| {
                        let logits = rows[row].squeeze(0)?;
                        let sampled = if mode.ends_with("_keyed") {
                            self.sampler.sample_keyed_cpu(
                                logits,
                                KeyedSampleContext {
                                    tokens: &contexts[row],
                                    prompt_len: PROMPT_LEN,
                                    return_logprobs: false,
                                    uniform: keys[row].uniform(
                                        Purpose::Generation,
                                        position as u32,
                                        0,
                                    ),
                                },
                            )?
                        } else {
                            self.sampler.sample(
                                logits,
                                &contexts[row],
                                PROMPT_LEN,
                                false,
                                rngs[row].clone(),
                                false,
                                self.batch > 1,
                            )?
                        };
                        Ok(sampled.token)
                    })
                    .collect::<Result<Vec<_>>>()?;
                for (context, token) in contexts.iter_mut().zip(tokens) {
                    context.push(token);
                }
            }
        }
        self.device.synchronize()?;
        let elapsed_us = start.elapsed().as_secs_f64() * 1e6 / steps as f64;
        if audit {
            eprintln!("AUDIT_END mode={mode}");
        }
        if mode.starts_with("metal_") {
            for (row, history) in histories.iter().enumerate() {
                assert_eq!(history.state().to_vec1::<u32>()?[1], steps as u32);
                let tokens = history.tokens()?.to_vec1::<u32>()?;
                assert_eq!(tokens.len(), PROMPT_LEN + steps);
                if mode == "metal_compact" {
                    assert_eq!(tokens, contexts[row]);
                }
                contexts[row] = tokens;
            }
        }
        assert!(contexts.iter().all(|tokens| tokens[PROMPT_LEN..]
            .iter()
            .all(|&token| (token as usize) < self.vocab)));
        black_box(&contexts);
        Ok((elapsed_us, contexts))
    }
}

fn setting(name: &str, default: usize) -> usize {
    std::env::var(name)
        .map(|value| value.parse().unwrap())
        .unwrap_or(default)
}

#[test]
#[ignore = "manual hardware benchmark; includes GPU completion and validates device history"]
fn benchmark_keyed_sampling() -> Result<()> {
    let device = Device::new_metal(0)?;
    let steps = setting("KEYED_BENCH_STEPS", DEFAULT_STEPS);
    let repeats = setting("KEYED_BENCH_REPEATS", DEFAULT_REPEATS);
    let mode_filter = std::env::var("KEYED_BENCH_MODE").ok();
    let vocab_filter = std::env::var("KEYED_BENCH_VOCAB").ok();
    let batch_filter = std::env::var("KEYED_BENCH_BATCH").ok();
    let sampling_filter = std::env::var("KEYED_BENCH_FILTER").ok();
    let audit = std::env::var_os("KEYED_BENCH_AUDIT").is_some();
    let quick = std::env::var_os("KEYED_BENCH_QUICK").is_some();
    println!("vocab,batch,filter,mode,repeat,steps,us_per_batch_step");
    for vocab in if quick {
        vec![32_768]
    } else {
        vec![32_768, 131_072]
    } {
        if vocab_filter
            .as_ref()
            .is_some_and(|value| value != &vocab.to_string())
        {
            continue;
        }
        for batch in if quick {
            vec![setting("KEYED_BENCH_BATCH", 1)]
        } else {
            vec![1, 8]
        } {
            if batch_filter
                .as_ref()
                .is_some_and(|value| value != &batch.to_string())
            {
                continue;
            }
            for filter in if quick {
                vec!["topk40_p90"]
            } else {
                vec!["greedy", "categorical", "topk40_p90"]
            } {
                if sampling_filter
                    .as_ref()
                    .is_some_and(|value| value.split(',').all(|entry| entry != filter))
                {
                    continue;
                }
                let case = Case::new(&device, vocab, batch, filter)?;
                let modes = MODES
                    .iter()
                    .copied()
                    .filter(|mode| mode_filter.as_ref().is_none_or(|value| value == mode))
                    .collect::<Vec<_>>();
                for mode in &modes {
                    case.run(mode, WARMUP_STEPS, false)?;
                }
                for repeat in 0..repeats {
                    let mut observed = HashMap::new();
                    for offset in 0..modes.len() {
                        let mode = modes[(offset + repeat) % modes.len()];
                        let (micros, tokens) = case.run(mode, steps, audit)?;
                        println!("{vocab},{batch},{filter},{mode},{repeat},{steps},{micros:.3}");
                        observed.insert(mode, tokens);
                    }
                    for (left, right) in [
                        ("cpu_keyed", "readback_cpu_keyed"),
                        ("metal_compact", "metal_queued"),
                    ] {
                        if let (Some(a), Some(b)) = (observed.get(left), observed.get(right)) {
                            assert!(a == b, "keyed replay differs: {left} vs {right}");
                        }
                    }
                    if let (Some(cpu), Some(metal)) =
                        (observed.get("cpu_keyed"), observed.get("metal_compact"))
                    {
                        let differences = cpu
                            .iter()
                            .zip(metal)
                            .map(|(cpu, metal)| {
                                cpu[PROMPT_LEN..]
                                    .iter()
                                    .zip(&metal[PROMPT_LEN..])
                                    .filter(|(a, b)| a != b)
                                    .count()
                            })
                            .sum::<usize>();
                        println!("KEYED_CPU_METAL_DIFFERENCES vocab={vocab} batch={batch} filter={filter} repeat={repeat} differing={differences} total={}", steps * batch);
                    }
                }
            }
        }
    }
    Ok(())
}
