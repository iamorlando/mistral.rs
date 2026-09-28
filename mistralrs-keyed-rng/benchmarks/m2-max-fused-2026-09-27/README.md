# Metal batch residency and fused greedy sampling

A model-pipeline gate downloaded batched Metal logits to CPU before the keyed
GPU sampler could consume them. The gate preserved batched causal outputs only
for CUDA. Both cache-backend branches now preserve Metal outputs when every
sequence qualifies for keyed Metal sampling. Earlier sampler-only batch tests
bypassed this boundary and did not prove that full-model batched decoding used
the GPU sampler.

This fixes a missing integration in the keyed implementation. The default Metal
batch fallback already copied logits to CPU; the previous keyed integration
failed to replace that behavior. The fixed path performs logits adjustment,
selection, normalization and accepted-token history updates on Metal.

## Complete model: eight completion choices

Apple M2 Max, 64 GiB, release build with `metal,accelerate`, f32
`SmolLM2-135M-Instruct` revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, all
30 transformer layers on Metal, paged attention off. Each request has eight
completion choices sharing one prompt, with 128 generated tokens per choice.
Each process runs two warmups and five measured iterations at each context
depth. Five rounds alternate the order of reference and keyed Metal. No
compilation, debugger audit or other benchmark from this task ran concurrently.

These are aggregate streamed decode tokens per second after the first token;
model loading and prefill are excluded. The table averages the five process
means. Each process's spread, command and executable checksum are preserved in
[model-b8/model-results.json](model-b8/model-results.json) and adjacent logs.

| Context | Reference Metal | Fixed keyed Metal | Throughput gain |
| --- | ---: | ---: | ---: |
| 128 | 1181.58 tokens/s | 1325.10 tokens/s | +12.15% |
| 512 | 1035.88 tokens/s | 1148.66 tokens/s | +10.89% |

Every keyed process exceeded every reference process at both context depths:
1322.1-1333.5 versus 1160.5-1197.1 tokens/s at depth 128, and 1144.4-1155.8
versus 1017.6-1046.6 at depth 512. This is a measured end-to-end decode gain for
this workload, not an extrapolation from isolated sampling kernels.

## Matched CPU model comparison

A separate release-build comparison used the same binary, checkpoint, batch
eight, generation length, two warmups and five measured iterations per depth.
CPU uses the Accelerate-enabled backend with all layers on CPU; Metal uses all
30 layers on GPU. CPU retains its default f16 KV storage with f32 accumulation;
Metal KV storage follows the f32 model tensors. This is one process per mode,
not the five-round study above.

| Context | CPU + Accelerate | Reference Metal | Fixed keyed Metal | Fixed / CPU |
| --- | ---: | ---: | ---: | ---: |
| 128 | 161.5 tokens/s | 1177.4 | 1326.1 | 8.21x |
| 512 | 161.1 tokens/s | 1038.9 | 1149.7 | 7.14x |

These ratios compare full model decode, including its sampling and streaming
boundaries. They do not imply that the new sampler alone delivers the entire
GPU-versus-CPU gain; reference Metal already runs model layers on the GPU.
[Commands, measurements and binary checksum](model-b8-cpu/model-results.json)
and adjacent logs preserve the CPU comparison.

## Runtime transfer and dispatch evidence

Separate LLDB runs trace Candle's actual Metal storage-to-CPU copy boundary and
the keyed shared-result read boundary during eight model steps. Copy element
counts are read from the live storage object. The audit also records keyed
kernel dispatch names at runtime, proving that the model reaches the GPU sampler.
Debugger runs are excluded from throughput measurements.

| Model path | Steps | Candle copies | Payload per step | Shared-result boundaries |
| --- | ---: | ---: | ---: | ---: |
| Reference, batch 8 | 8 | 8 | 393,216 f32 = 1.5 MiB | 0 |
| Fixed keyed, batch 8 | 8 | 0 | 8 selected records = 96 bytes | 8 |
| Reference, batch 1 | 8 | 8 | 4 f32 = 16 bytes | 0 |
| Fixed keyed, batch 1 | 8 | 0 | 1 selected record = 12 bytes | 8 |

The eight-row keyed run dispatched `logits_tiles` eight times, `logits_finish`
eight times, and `history_commit` 64 times, plus eight initial history kernels.
Selection and normalization are batched; history commits remain one per row.
The single-row ordinary greedy run dispatched only `argmax_tiles` and
`argmax_finish_commit` per token, plus initial history setup.

There is still one compact host read/synchronization boundary per model batch
step for streaming, stopping and scheduling. No vocabulary tensor crosses that
boundary in the fixed eligible path. This is not zero host synchronization or
a fully GPU-resident serving scheduler. The queued sampler microbenchmark has
no intermediate host read, but serving does not queue whole model steps that way.

Audits and adjacent full transcripts: [batch 8 reference](model-readback-legacy-b8.json),
[batch 8 keyed](model-readback-keyed-b8.json), [batch 1 reference](model-readback-legacy.json),
[batch 1 keyed](model-readback-keyed.json), and [isolated top-1 dispatch audit](readback-top1-b1.json).
These instrument tensor and selected-result reads, not every driver operation.

## Sampler changes

- Reuse tile and probability scratch allocations across decoding steps while
  keeping selected-result buffers immutable for queued consumers.
- Reduce tile maxima and masses with parallel SIMD/threadgroup reductions;
  select categorical tiles in parallel rather than scanning all tile summaries
  in one thread.
- Skip penalty-count loads when the corresponding penalties are disabled.
- For single-sequence greedy generation without probability reporting/ranking,
  compute argmax and commit accepted-token history in two GPU kernels. Skip
  probability normalization and use the selected-result token as the next GPU
  input view. Ranking and requested log probabilities keep normalized results.

## Single-stream model result

Five rotating release-build rounds compared reference Metal, the saved keyed
executable at `43f689162d864d560a607192a056bf6efedc7367`, and the fused keyed
sampler. These ran before the batch-preservation correction; single-stream
forward behavior is unaffected by that correction.

| Context | Reference Metal | Previous keyed | Fused keyed | Gain vs reference |
| --- | ---: | ---: | ---: | ---: |
| 128 | 278.50 tokens/s | 277.86 | 279.54 | +0.37% |
| 512 | 257.28 tokens/s | 256.86 | 258.28 | +0.39% |

These differences do not establish a meaningful single-stream model speedup.
The reference already reads compact top-1 candidates for a single sequence;
its transfer savings are much smaller than in the batch-eight case. Raw rounds
and both binary checksums are in [model/model-results.json](model/model-results.json).

## Complete isolated sampling matrix

Median microseconds per batch step including every row and GPU completion.
The optimized test profile has debug assertions; these are not release-model
measurements. Each of five rotating repetitions contains 32 steps after eight
warmup steps. Inputs are fixed resident f32 logits, without a model forward
pass. Temperature is 0.8 for `top1`, categorical and top-k/top-p; `greedy` means
absent sampler temperature. Penalties and requested log probabilities are off.
Single-row greedy/top-1 uses the fused commit path. Batched sampling retains
normalization and separate history commits, matching the multiple-choice path.

| Vocabulary | Batch | Sampling | CPU resident | Reference Metal | Keyed compact | Keyed queued |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 32,768 | 1 | greedy | 102.7 | 296.1 | 228.1 | 29.7 |
| 32,768 | 1 | top1 | 184.5 | 235.8 | 207.4 | 17.9 |
| 32,768 | 1 | categorical | 107.8 | 278.5 | 236.4 | 40.3 |
| 32,768 | 1 | topk40_p90 | 219.7 | 317.6 | 404.8 | 111.1 |
| 32,768 | 8 | greedy | 211.3 | 457.3 | 402.2 | 96.4 |
| 32,768 | 8 | top1 | 329.8 | 592.1 | 414.3 | 93.9 |
| 32,768 | 8 | categorical | 232.2 | 474.0 | 434.6 | 95.0 |
| 32,768 | 8 | topk40_p90 | 367.3 | 637.0 | 819.7 | 308.8 |
| 131,072 | 1 | greedy | 420.0 | 651.8 | 230.0 | 19.8 |
| 131,072 | 1 | top1 | 718.4 | 250.5 | 217.6 | 19.8 |
| 131,072 | 1 | categorical | 430.5 | 654.3 | 263.4 | 47.2 |
| 131,072 | 1 | topk40_p90 | 888.1 | 403.7 | 546.5 | 203.8 |
| 131,072 | 8 | greedy | 562.4 | 1370.1 | 542.2 | 106.9 |
| 131,072 | 8 | top1 | 984.1 | 1768.7 | 540.7 | 106.1 |
| 131,072 | 8 | categorical | 655.5 | 1572.3 | 539.0 | 123.5 |
| 131,072 | 8 | topk40_p90 | 1153.0 | 1977.9 | 1693.1 | 872.4 |

Single-row compact top-1 sampling takes 12-13% less time than reference Metal in
this matrix. Top-k 40 / top-p 0.9 remains slower for both single-row vocabulary
sizes and for the 32K eight-row case. This change does not establish a speedup
for every sampling configuration. Queued numbers exclude intermediate host
reads and are not serving throughput. CPU-resident sampler timing excludes
model inference and cannot establish CPU-versus-GPU model performance.

All seven modes, including keyed CPU and forced logits-copy controls, are in
[timings.csv](timings.csv): 560 observations, [112 summaries](summary.csv), and
[validation output](timings.log). Integer RNG agreement, compact/queued token
agreement, CPU agreement with/without readback, and committed device histories
are checked. Cross-device categorical tokens can differ with floating-point
reductions; this is reported rather than treated as bitwise sampling equivalence.

## Validation and reproduction

The native Metal suite covers penalties, tied maxima, invalid distributions,
queued immutable outputs, stop tokens, capacity and history state. Core tests
cover batched sampling, normalized ranking probabilities and full reporting.
The CLI regression test verifies that batch measurements wait for every choice.
The actual batch-eight model readback audit exercises the forward boundary that
sampler-only tests missed. Both cache-backend branches were corrected, but the
recorded model runs use paged attention off.

Build release for model timing and a debug-information binary for LLDB auditing:

```sh
cargo build --release -p mistralrs-cli --features metal,accelerate
python3 scripts/benchmark_keyed_model.py /path/to/release/mistralrs /path/to/model --features metal,accelerate --batch-size 8 --rounds 5 --modes metal_legacy metal_keyed --output /tmp/keyed-model-b8
cargo build -p mistralrs-cli --features metal,accelerate
python3 scripts/audit_keyed_readbacks.py /path/to/debug/mistralrs /path/to/candle/candle-core --model /path/to/model --batch 8 --dispatches --output /tmp/keyed-model-b8-readbacks.txt
```

See the [benchmark instructions](../README.md) for the complete sampling matrix,
CPU comparison, controls and debugger requirements. [metadata.json](metadata.json)
records pinned sources, source hashes, hardware and build profiles;
[validation.json](validation.json) records completed checks. Builds were made
with uncommitted changes over the listed base revision, so source and binary
hashes identify the measured artifacts. Model weights and binaries are not committed.
