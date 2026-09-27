# M2 Max sampling benchmark, 2026-09-27

The queued device API performs no intermediate tensor readback. The current
serving sampling path still reads one compact batch of token records every step.
Metal is not generally faster than CPU: its full-vocabulary greedy and categorical
selectors are severe regressions. Queued top-k sampling wins at batch 1, while
batch-8 top-k sampling remains slower than the existing CPU sampler.

These are sampler/history microbenchmarks, not model-generation throughput.
See [methodology and reproduction commands](../README.md),
[hardware/build metadata](metadata.json), [all 360 measurements](timings.csv),
and [per-mode medians and ranges](summary.csv).

## Completed-work latency

Median milliseconds per batch step, lower is better. Each row contains all tokens
in that batch. Five repetitions of 32 steps, after 8 warmup steps per mode.
CPU and GPU code run in the same optimized test binary. Logits are synthetic f32
and resident before timing. CPU rows execute through Rayon; Metal uses the current
per-sequence production submission path. GPU execution is drained inside timing.

| Vocabulary | Batch | Filter | CPU legacy | CPU keyed | Readback + CPU legacy | Readback + CPU keyed | Metal compact | Metal queued |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 1 | greedy | 0.111 | 0.112 | 0.320 | 0.318 | 7.872 | 7.493 |
| 32,768 | 1 | categorical | 0.118 | 1.180 | 0.351 | 1.697 | 12.913 | 12.545 |
| 32,768 | 1 | topk40_p90 | 0.233 | 1.298 | 0.503 | 1.846 | 0.633 | 0.183 |
| 32,768 | 8 | greedy | 0.238 | 0.241 | 0.539 | 0.517 | 59.983 | 59.281 |
| 32,768 | 8 | categorical | 0.263 | 1.435 | 0.598 | 2.022 | 99.085 | 98.520 |
| 32,768 | 8 | topk40_p90 | 0.378 | 1.539 | 0.756 | 2.165 | 1.818 | 1.250 |
| 131,072 | 1 | greedy | 0.438 | 0.431 | 0.693 | 0.683 | 31.237 | 30.715 |
| 131,072 | 1 | categorical | 0.449 | 5.185 | 0.809 | 5.760 | 53.319 | 52.856 |
| 131,072 | 1 | topk40_p90 | 0.931 | 5.804 | 1.362 | 6.261 | 1.230 | 0.443 |
| 131,072 | 8 | greedy | 0.624 | 0.611 | 1.344 | 1.317 | 246.274 | 245.363 |
| 131,072 | 8 | categorical | 0.655 | 5.882 | 1.424 | 6.475 | 413.513 | 412.822 |
| 131,072 | 8 | topk40_p90 | 1.223 | 6.415 | 1.986 | 7.207 | 4.160 | 3.161 |

For batch 1, top-k 40 / top-p 0.9:

- 32K queued Metal: 0.183 ms versus 0.233 ms for CPU-resident legacy sampling
  (1.27x faster). Compact Metal takes 0.633 ms versus 0.503 ms for full logits
  readback plus legacy CPU sampling (1.26x slower).
- 128K queued Metal: 0.443 ms versus 0.931 ms for CPU-resident legacy sampling
  (2.10x faster). Compact Metal takes 1.230 ms versus 1.362 ms for full logits
  readback plus legacy CPU sampling (1.11x faster).

The queued numbers cannot be substituted for serving latency: the current server
consumes selected tokens before scheduling the next model step. Comparisons only
against the keyed CPU sampler overstate the improvement, because that CPU path
currently performs extra full-vocabulary sorting. Timing variation and device
load can affect small differences; this is one machine, not a broad hardware study.

## Observed readbacks

LLDB breakpoints in the pinned Candle `MetalStorage::to_cpu` implementation counted
actual tensor readbacks during four steps. The breakpoint resolved to ten compiled
locations, covering the instantiated scalar types, including f32 logits and u32
records. Initialization, warmup and final validation are outside the counted region.

| Mode | Batch 1: calls in 4 steps | Batch 8: calls in 4 steps | Bytes per step at 32K vocabulary |
| --- | ---: | ---: | --- |
| CPU-resident legacy/keyed | 0 | 0 | 0 |
| Full logits readback + CPU legacy/keyed | 4 | 4 | 131,072 / 1,048,576 |
| Metal compact | 4 | 4 | 12 / 96 |
| Metal queued | 0 | 0 | 0 intermediate |

The byte counts follow the actual buffer shapes: f32 `[batch, vocabulary]` versus
u32 `[batch, 3]`. The three record fields are token ID, logprob bits and status.
Counts were measured; byte counts were derived from those shapes. Final history
validation reads state/history after the measured loop. A terminal GPU completion
wait remains inside timing, so zero readbacks does not mean zero synchronization
or zero CPU/driver overhead. Full logprobs, grammar and fallback paths are outside
this audit and can add readbacks.

Raw debugger transcripts and machine-readable counts:
[batch 1 transcript](readbacks-batch1.txt), [batch 1 counts](readbacks-batch1.json),
[batch 8 transcript](readbacks-batch8.txt), [batch 8 counts](readbacks-batch8.json).
Debugger timings in those transcripts are not used in the performance table.

## Validation and limitations

Compact and queued Metal histories agree exactly, and CPU keyed histories agree
with/without the initial logits readback. All committed lengths, generated positions
and token ranges were checked. At 128K vocabulary, batch 8, unrestricted categorical
sampling, CPU and Metal choose different tokens for 5 of the 256 fixed events in
each repetition. Other measured fixtures agree. Exact RNG-word and uniform-bit
parity still passes; matching RNG bits does not guarantee matching token choices
when probability and CDF arithmetic differ across devices.

The initial run stopped on an overly strict cross-device token-equality assertion
at that categorical case. The harness was corrected to assert same-device replay
and report cross-device differences separately. The ten completed cases were
retained; both remaining cases were run with five fresh repetitions. No sampler
or kernel implementation was changed between those measurements.

Validation completed:

- `cargo check -p mistralrs-core --features metal --tests`
- `cargo test -p mistralrs-keyed-rng --features metal`: 2 CPU and 6 Metal tests,
  including 4,119 exact CPU/Metal RNG cases.
- `MISTRALRS_SAMPLING_RNG=keyed-threefry2x32-v1 cargo test -p mistralrs-core --features metal keyed_metal`:
  both serving integration tests pass.
- Benchmark replay/history checks, batch-1 and batch-8 LLDB audits, Rust formatting,
  and Python syntax checks pass.

## Optimization priorities

Source inspection identifies the primary candidates for follow-up work:

1. `rng_select` uses one GPU thread per sequence and serially scans the whole
   vocabulary for greedy/unrestricted categorical sampling. Parallel reductions
   and CDF operations are needed before these paths are competitive.
2. Top-k uses a full-vocabulary merge sort, and serving submits each sequence
   separately. Partial selection and batch-wide kernels could reduce work.
3. The keyed CPU path sorts unnecessarily for unrestricted sampling and sorts the
   whole reporting order for top-k; it needs the same efficient baseline treatment.
4. Removing the remaining serving readback requires scheduler changes beyond the
   device sampler. The current queued benchmark already shows that eliminating
   readbacks alone does not solve the full-vocabulary kernel cost.
