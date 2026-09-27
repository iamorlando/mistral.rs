# Sampling benchmarks

Run the real CPU and Metal sampler implementations with synthetic, resident f32
logits, then validate the generated device histories:

```sh
cargo test -p mistralrs-core --features metal benchmark_keyed_sampling -- --ignored --nocapture --test-threads=1
```

The workspace test profile uses `opt-level = 3`, debug information and debug
assertions. CPU and Metal use the same binary. The default matrix uses vocabularies
32,768 and 131,072, batches 1 and 8, and greedy, categorical, and top-k 40 / top-p
0.9 sampling. Sampling temperature is 0.8. Penalties, bias, grammar, DRY and full
logprobs are disabled. Each mode gets 8 warmup steps, then five repetitions of 32
steps. Mode order rotates between repetitions. CPU batch rows use Rayon.

CSV timings are microseconds per **batch step**, including all rows. Metal is
synchronized before timing and at the end, so these measure completed work.
Compilation, initial logits uploads, prompt/history initialization and post-run
validation are excluded. Selection, allocations, key/event derivation, history
updates and per-step stop-token uploads are included. Inputs stay fixed while
generated positions and histories advance. Queued commits feed subsequent device
events and history, without a model forward pass.

| Mode | Work measured |
| --- | --- |
| `cpu_legacy` | Existing Isaac64 CPU sampler on CPU-resident logits |
| `cpu_keyed` | Keyed CPU sampler on CPU-resident logits |
| `readback_cpu_legacy` | One full logits-batch readback per step, then existing CPU sampler |
| `readback_cpu_keyed` | One full logits-batch readback per step, then keyed CPU sampler |
| `metal_compact` | Production Metal sampler, one compact batch readback per step, device history commit |
| `metal_queued` | Same Metal sampler/history operations queued across steps, no intermediate readback |

`metal_compact` exercises the sampler and commit methods used by serving, but
excludes the rest of the scheduler. `metal_queued` measures a low-level capability
that the serving scheduler does not currently use across model steps. None of
these numbers measures model generation, tokenization, streaming, complete server
throughput, or CPU-versus-GPU model inference. CPU modes do not upload the next
model input, and Metal modes do not construct a model forward input. No model
weights are needed.

The harness checks token validity, committed device state, exact compact/queued
Metal agreement, and exact CPU agreement with/without logits readback. It reports
CPU/Metal token differences separately: cross-device token equality is not
promised when floating-point probabilities or accumulation differ.

Useful overrides: `KEYED_BENCH_STEPS`, `KEYED_BENCH_REPEATS`,
`KEYED_BENCH_MODE` (one table entry), `KEYED_BENCH_VOCAB`, `KEYED_BENCH_BATCH`,
`KEYED_BENCH_FILTER` (comma-separated filter names), and `KEYED_BENCH_QUICK=1`
(32K, batch 1 unless overridden, top-k 40 / top-p 0.9 only).

## Runtime readback audit

Build the test binary with debug information, then run the separate debugger
audit. Cargo prints the test executable path; the second argument below is the
`candle-core` directory in the pinned Candle checkout:

```sh
cargo test -p mistralrs-core --features metal benchmark_keyed_sampling --no-run
python3 scripts/audit_keyed_readbacks.py /path/to/mistralrs_core-test-binary /path/to/candle/candle-core --output /tmp/keyed-readbacks.txt
```

Add `--batch 8` to verify that compact results still use one readback per batch
step. A machine-readable JSON summary is saved next to the debugger transcript.

The script sets a source breakpoint inside Candle's actual Metal storage `to_cpu`
implementation, where it allocates a CPU staging buffer, blits the source buffer,
waits for completion and reads its contents. It automatically continues each
breakpoint and counts hits between explicit begin/end markers around four
sampling steps. Initialization, warmup and final validation lie outside these
markers. The full-logits and compact modes are positive controls for the same
breakpoint. Debugger timings must not be used as performance measurements.

This measures tensor readback, not all host/driver activity. A terminal
`Device::synchronize()` remains necessary for completed-work timing. Metal command
buffer submission, allocation and driver work can also involve CPU overhead.
