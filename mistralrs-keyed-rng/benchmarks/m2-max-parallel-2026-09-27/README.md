# Parallel Metal sampling on Apple M2 Max

Correction: the original explanation of the model readback was wrong. The
existing Metal path already performs GPU top-k reduction for eligible requests.
The CLI requests top-k 1 with effective temperature 1, so this model benchmark
reads four packed f32 values (16 bytes), not the 49,152-element logits vector.
The old audit counted calls without checking tensor sizes. The timings below
remain the recorded measurements, but they compare against an existing GPU
top-k path, and the matrix omitted that path as a separate baseline. The keyed
implementation also failed to specialize top-k 1 and unnecessarily ran a full
softmax and block sort. See the [corrected top-1 report](../m2-max-top1-2026-09-27/README.md).

The severe slowdown in the earlier report was introduced by the original keyed
selector, which scanned a vocabulary in one GPU thread per sequence. It was not
evidence that Mistral's existing Metal model execution lacked parallelism.

The replacement distributes logits work across GPU tiles, uses SIMD reductions
and prefix sums, computes exact partial top-k with block sorting and a pruned
merge tree, and dispatches compatible batches together. Counts, generated
positions, penalties, RNG, filtering, selection, and accepted-token feedback stay
on the GPU. CPU keyed sampling also avoids unnecessary full-vocabulary sorting.

## Complete model decoding

Primary comparison: the same optimized binary with `metal,accelerate`, f32
weights, all 30 layers explicitly on the chosen device, no paged attention, greedy
sampling, two warmups and five measured iterations. Each request generates 128
tokens. Throughput excludes loading, prefill, and the first generated token, but
includes actual model execution, sampling, host scheduling and streaming.

Model: `HuggingFaceTB/SmolLM2-135M-Instruct`, revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, vocabulary 49,152.
The checkpoint checksum and source hashes are in [metadata.json](metadata.json).

| Context depth | CPU + Accelerate | Existing Metal | Keyed Metal | Keyed Metal / CPU |
| --- | ---: | ---: | ---: | ---: |
| 128 | 69.3 +/- 1.0 tokens/s | 273.9 +/- 0.3 | 264.9 +/- 0.9 | 3.82x |
| 512 | 68.9 +/- 0.8 tokens/s | 247.1 +/- 0.3 | 240.3 +/- 0.4 | 3.49x |

Spread is the CLI's population standard deviation across five iterations.
The new path is substantially faster than optimized CPU inference, but is
**2.8-3.3% slower than existing Metal** on this small greedy checkpoint. These
results do not establish an end-to-end gain over the existing Metal backend.
The absolute GPU/CPU model speedup largely comes from Mistral's existing GPU
model kernels, not from this sampler change.

Commands and raw results are in [accelerate/model-results.json](accelerate/model-results.json)
and its adjacent logs. [preliminary-metal-only](preliminary-metal-only/) retains
earlier runs without Accelerate (CPU about 16.5 tokens/s). Those runs used the
initial parallel single-sequence implementation before batch optimization and
are not the primary CPU comparison. The final timings were run without compiler
or debugger activity from this task.

## Sampling matrix

Median microseconds per batch step, including every row. Five repetitions of
32 steps after eight warmup steps; mode order rotates. CPU rows use Rayon.
Both backends use the same `metal,accelerate` test binary with opt-level 3,
debug information and debug assertions. GPU completion is included in timing.
The harness uses fixed synthetic logits; it contains no model forward pass.

| Vocabulary | Batch | Sampling | CPU | Logits readback + CPU | GPU + result read | GPU queued |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 32,768 | 1 | greedy | 104.4 | 275.5 | 251.5 | 37.9 |
| 32,768 | 1 | categorical | 108.0 | 274.9 | 255.3 | 43.8 |
| 32,768 | 1 | topk40_p90 | 222.5 | 390.4 | 380.0 | 110.3 |
| 32,768 | 8 | greedy | 201.5 | 437.7 | 440.7 | 97.1 |
| 32,768 | 8 | categorical | 237.1 | 479.1 | 460.1 | 100.8 |
| 32,768 | 8 | topk40_p90 | 370.0 | 626.8 | 777.6 | 307.0 |
| 131,072 | 1 | greedy | 428.4 | 632.6 | 328.0 | 112.7 |
| 131,072 | 1 | categorical | 429.9 | 682.2 | 358.4 | 115.9 |
| 131,072 | 1 | topk40_p90 | 892.3 | 1453.5 | 523.4 | 203.9 |
| 131,072 | 8 | greedy | 554.3 | 1353.6 | 670.4 | 167.9 |
| 131,072 | 8 | categorical | 587.5 | 1386.9 | 702.1 | 176.9 |
| 131,072 | 8 | topk40_p90 | 1181.8 | 2047.4 | 1617.3 | 860.8 |

`GPU queued` is 1.2-4.4x faster than the existing CPU sampler on resident CPU
logits across this matrix. It queues selection and feedback without intermediate
host reads; the current serving scheduler does not queue whole model decode
steps this way.

`GPU + result read` includes the serving-style host boundary. Comparing it with
`Logits readback + CPU` describes cases that use CPU sampling; it does not cover
the existing eligible GPU top-k path omitted from this matrix. At 128K
vocabulary it is 1.27-2.78x faster across these cases. At 32K the improvements are
small, batch-eight greedy is approximately tied, and batch-eight top-k is about
24% slower. GPU dispatch/synchronization and top-k costs still matter at these
sizes. Moving numeric work onto the GPU does not guarantee a win in every case.

All 360 observations are in [timings.csv](timings.csv), all 72 mode summaries
including minima/maxima in [summary.csv](summary.csv), and correctness/parity
output in [timings.log](timings.log). CPU and GPU RNG words match exactly;
floating-point reductions can still choose different categorical tokens.

## Runtime readback audit

LLDB breakpoints count both Candle's real tensor-copy boundary and the keyed
sampler's wait before accessing shared result records. Debugger runs are separate
from timing runs. Initialization, warmup and final diagnostic reads are excluded
from the sampler counts.

| Workload | Steps | Tensor copies | Shared result reads |
| --- | ---: | ---: | ---: |
| Sampler, full logits + CPU, batch 1 or 8 | 4 | 4 | 0 |
| Sampler, compact GPU, batch 1 or 8 | 4 | 0 | 4 |
| Sampler, queued GPU, batch 1 or 8 | 4 | 0 | 0 |
| Actual model, existing Metal | 8 | 8 | 0 |
| Actual model, keyed Metal | 8 | 0 | 8 |

The model audit covers complete inference requests after warmup. It observes no
Candle tensor readback during keyed model generation, and one 12-byte selected
result record per token. Existing Metal copies four packed f32 values
(16 bytes) per token in this model benchmark. Shared memory eliminates a staging copy, **not the
host read or synchronization**. The host still consumes tokens for output,
stop strings and scheduling. GPU history commits are queued before reporting,
and the next text-model token input uses the selected device tensor.

Transcripts and structured counts: [batch 1](readbacks-batch1.json),
[batch 8](readbacks-batch8.json), [model keyed](model-readbacks-keyed.json),
[model existing](model-readbacks-legacy.json). These audits cover the instrumented
Candle and keyed readback boundaries, not every driver operation. Custom host
logits processors, DRY, grammar checks, full reporting probabilities and generic
speculative verification retain the documented integration boundaries.

## Correctness and reproduction

The GPU tests include exact RNG reference bits, probability/CDF checks, penalties,
invalid distributions, stable top-k through 131,072 candidates, concurrent shared
record reads, batch reorder/split/join, EOS and accepted-token device feedback.
Serving tests cover retries, row ownership and invalid-row history disposal.

See [benchmark instructions](../README.md) for commands. Explicit layer placement
avoids this host's automatic CPU-memory-capacity detection issue. Global CLI
`--seed` is omitted because CPU `set_seed` is unsupported and greedy sampling
requires no random draw. No model files are committed.
