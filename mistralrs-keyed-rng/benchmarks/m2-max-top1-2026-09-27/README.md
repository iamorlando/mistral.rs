# Top-1 correction and existing Metal baseline

The [batch residency and fused greedy report](../m2-max-fused-2026-09-27/README.md)
supersedes this performance result. It fixes a further full-model batch transfer
missed by the isolated sampler tests below and records release-build speedups.

The earlier explanation overstated what this work removes. Existing Mistral
Metal already reduces eligible single-sequence top-k requests on the GPU. The
complete-model benchmark uses top-k 1 and reads four packed f32 values per token,
not a full vocabulary. Counting readback calls without their sizes concealed
that distinction. The old sampling matrix also omitted this existing GPU path.

The keyed implementation had a separate performance defect: it treated top-k 1
with a nonzero temperature as general stochastic top-k. It materialized a full
probability vector and ran the block sort to retain one candidate. Top-k 1 now
uses the parallel argmax path, preserves temperature for the full-distribution
log probability, and skips the random draw and categorical tile scan. This
recovers about 2% model throughput relative to the previous keyed implementation.
It does **not** demonstrate a meaningful model-throughput gain over existing Metal.

## Source and readback evidence

- The CLI's deterministic parameters specify top-k 1. Request construction in
  [`add_request.rs`](../../../mistralrs-core/src/engine/add_request.rs) supplies
  effective temperature 1 when the request temperature is absent.
- [`Sampler::sample_topk_on_device_metal`](../../../mistralrs-core/src/sampler.rs)
  calls the existing GPU `metal_topk_logits_packed` and reads `2 * k + 2` f32 values.
- [`coalesce_batch_logits_to_cpu`](../../../mistralrs-core/src/pipeline/sampling.rs)
  retains a single sequence's device logits but moves multiple rows to CPU for
  the existing fallback. The new matrix includes both cases as `metal_legacy`.
- [`keyed.rs`](../../../mistralrs-core/src/sampler/keyed.rs) now recognizes top-k 1
  in both direct and batched selection. The regression test covers tied maxima,
  top-p values, full-distribution log probability, reporting, and two-row batches.

The size-aware LLDB audit checks actual Candle storage element counts at its
Metal-to-CPU boundary and counts the keyed shared-result boundary separately.
These debugger runs are separate from all timing runs.

| Workload | Steps | Candle copies | Elements per copy | Shared-result boundaries |
| --- | ---: | ---: | ---: | ---: |
| Forced logits readback, batch 1 | 4 | 4 | 32,768 f32 | 0 |
| Existing Metal top-k 40, batch 1 | 4 | 4 | 82 f32 | 0 |
| Existing batch fallback, batch 8 | 4 | 4 | 262,144 f32 | 0 |
| Keyed compact, batch 1 or 8 | 4 | 0 | 0 | 4 |
| Keyed queued, batch 1 or 8 | 4 | 0 | 0 | 0 |
| Complete model, existing Metal top-k 1 | 8 | 8 | 4 f32 | 0 |
| Complete model, keyed top-k 1 | 8 | 0 | 0 | 8 |

Each keyed shared result contains three u32 words, or 12 bytes per sequence.
For the model case, the existing path reads 16 bytes per token and the keyed
path reads 12. **Both serving paths still synchronize with the host once per
generated token.** The keyed path avoids Candle's staging copy, but it does not
eliminate the host read or the scheduler's per-token dependency. The queued
sampler has no intermediate readback; the complete-model serving loop does not
use that execution pattern across model steps.

Structured audits and adjacent full transcripts: [batch 1](readback-audit-b1.json),
[batch 8](readback-audit-b8.json), [model existing](model-readback-legacy.json),
[model keyed](model-readback-keyed.json). These cover the instrumented tensor and
shared-result boundaries, not every driver operation.

## Matched complete-model comparison

Apple M2 Max, 64 GiB, f32 `SmolLM2-135M-Instruct` at revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, all 30 layers on Metal, no paged
attention, 128 generated tokens, two warmups and five measured iterations per
context depth. Three rounds rotate the order of existing Metal, the saved prior
keyed executable, and the corrected keyed executable. Compiler and debugger
activity from this task were excluded from these timings.

Both executables use `metal,accelerate` in the workspace's optimized dev profile
with debug information and debug assertions. These are **not release-build
measurements**. Means below aggregate the three rounds' reported token rates;
each individual round and its spread is retained in the raw results.

| Context depth | Existing Metal | Keyed before | Keyed corrected | Change vs before | Change vs existing Metal |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 207.33 tokens/s | 202.23 | 206.73 | +2.23% | -0.29% |
| 512 | 190.80 tokens/s | 185.67 | 189.47 | +2.05% | -0.70% |

The correction removes most of the measured regression. The remaining difference
is small; these runs do not establish a speedup over existing Metal. Absolute
rates differ from the earlier session, so the matched rotating rounds are the
comparison used here. CPU model inference was not remeasured in this experiment.

Commands, executable checksums, all rounds and raw adjacent logs are in
[model/model-results.json](model/model-results.json). The saved prior executable
contains the implementation at `a76966891ffbec92d43c2b63d4cab53af6f00350`.
Its embedded build revision predates that implementation because the earlier
build was made with uncommitted source changes; the executable checksum identifies
the measured artifact. Source hashes and checkpoint metadata are in
[metadata.json](metadata.json).

## Sampling matrix with the missing baseline restored

Median microseconds per batch step, including every row and GPU completion.
Five repetitions of 32 steps follow eight warmup steps, with rotating mode order.
The test profile is optimized with debug assertions. Fixed synthetic resident
logits are used; there is no model forward pass. `greedy` means absent sampler
temperature, while `top1` means top-k 1 with temperature 0.8, matching the
important dispatch distinction exposed by the CLI benchmark.

| Vocabulary | Batch | Sampling | CPU resident | Existing Metal path | Keyed compact | Keyed queued |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 32,768 | 1 | greedy | 108.8 | 356.1 | 326.2 | 45.0 |
| 32,768 | 1 | top1 | 193.9 | 256.0 | 291.8 | 33.2 |
| 32,768 | 1 | categorical | 113.1 | 308.2 | 280.3 | 45.0 |
| 32,768 | 1 | topk40_p90 | 229.5 | 373.3 | 457.3 | 142.5 |
| 32,768 | 8 | greedy | 238.8 | 503.9 | 516.1 | 105.9 |
| 32,768 | 8 | top1 | 345.9 | 663.3 | 537.9 | 100.6 |
| 32,768 | 8 | categorical | 254.1 | 534.6 | 547.2 | 112.0 |
| 32,768 | 8 | topk40_p90 | 383.3 | 667.7 | 951.2 | 399.1 |
| 131,072 | 1 | greedy | 424.8 | 693.9 | 360.7 | 62.9 |
| 131,072 | 1 | top1 | 749.6 | 272.1 | 376.3 | 66.9 |
| 131,072 | 1 | categorical | 452.4 | 689.3 | 391.9 | 124.8 |
| 131,072 | 1 | topk40_p90 | 922.3 | 484.9 | 672.2 | 246.5 |
| 131,072 | 8 | greedy | 608.0 | 1301.8 | 853.6 | 166.3 |
| 131,072 | 8 | top1 | 1064.4 | 1744.1 | 859.9 | 195.4 |
| 131,072 | 8 | categorical | 633.2 | 1343.5 | 845.3 | 198.0 |
| 131,072 | 8 | topk40_p90 | 1274.6 | 2031.5 | 1923.3 | 1059.6 |

The keyed compact path improves several cases where the existing sampler falls
back to CPU. It remains slower than the existing single-row GPU top-k path in
this matrix, and batch-eight top-k at 32K also regresses. Queued timings show what
the sampler can do without intermediate host reads; they are not a claim about
current serving throughput. The substantial end-to-end performance objective
has not been demonstrated by this change.

All seven modes, including forced CPU readback controls and keyed CPU, are in
[timings.csv](timings.csv): 560 observations and [112 summaries](summary.csv).
[timings.log](timings.log) includes completed-work checks, device-history checks,
compact/queued replay equality, and reported CPU/Metal token differences.
Floating-point reduction differences can change categorical tokens even when
the integer RNG words match.

## Validation and reproduction

The native Metal tests passed, including RNG bits, stable top-k, tiled logits,
penalties, invalid distributions, concurrent reads and device history. All four
core keyed Metal integration tests passed, including the new top-1 regression.
CPU and Metal/Accelerate `cargo check`, targeted Clippy and formatting checks
passed; [validation.json](validation.json) records the checks.

Use the [benchmark instructions](../README.md) for builds, matrix runs and
readback audits. To reproduce the rotating model comparison, preserve the prior
executable before building the corrected version, then pass `--rounds 3
--modes metal_legacy metal_keyed_before metal_keyed --baseline-binary /path/to/prior`
to `scripts/benchmark_keyed_model.py`. No model weights or executable binaries
are committed.
