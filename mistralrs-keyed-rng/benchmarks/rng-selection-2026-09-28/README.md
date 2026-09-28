# Explicit RNG selection validation

The RNG choice is now an engine default with a request-local override. The CLI
uses `--sampling-rng isaac64` or `--sampling-rng keyed-threefry2x32-v1`; TOML uses
`[global].sampling_rng`. Rust model builders and the Python Runner expose the
same choice. Chat completions, completions and Responses requests accept an
optional `sampling_rng` override. An omitted override inherits the engine value.
The former process-global environment switch is removed.

## Real model path audit

Separate debugger runs used the CLI option with the pinned f32
SmolLM2-135M-Instruct checkpoint, 30 Metal layers, paged attention off, eight
completion choices and eight generated tokens. The engine startup logs record
the configured RNG. These are readback/dispatch checks, not timing measurements.

| CLI RNG | Candle copies | Elements per copy | Compact result boundaries |
| --- | ---: | ---: | ---: |
| `isaac64` | 8 | 393,216 f32 | 0 |
| `keyed-threefry2x32-v1` | 0 | 0 | 8 |

The keyed audit also observes eight `logits_tiles`, eight `logits_finish`, 64
`history_commit` and eight initial history dispatches. Each compact result batch
is 96 bytes. This confirms explicit configuration reaches the GPU path measured
in the [earlier performance report](../m2-max-fused-2026-09-27/README.md).

Structured results and adjacent debugger transcripts:
[keyed](model-keyed-b8.json), [Isaac64](model-isaac64-b8.json).

## Live request overrides

The [HTTP validation](http-validation.json) starts one localhost server for each
engine default. For seeds 42 and 123, it submits a completion with the request
RNG omitted, explicitly Isaac64, and explicitly keyed Threefry. All 12 requests
use the same prompt and sampling parameters. It verifies:

- Omitting the request RNG reproduces the configured engine algorithm.
- Either explicit request choice reproduces that algorithm under both engine defaults.
- The two algorithms produce distinct continuations on this fixture.
- Unknown request RNG values are rejected with HTTP 400 or 422.

The server processes are stopped after each test. This is an inheritance and
replay check, not a throughput measurement. Reproduce with:

```sh
python3 mistralrs-keyed-rng/benchmarks/rng-selection-2026-09-28/validate_http.py /path/to/mistralrs /path/to/model /tmp/rng-http-validation
```

[Python validation](python-validation.json) imports the built native extension,
constructs both request classes with each RNG choice and inheritance, rejects an
invalid choice, and verifies the Runner exposes the Isaac64 default. Reproduce
with `python3 validate_python.py /path/to/libmistralrs.dylib /tmp/python-validation.json`
from this report directory.

[validation.json](validation.json) records the compile, test and formatting
checks. [metadata.json](metadata.json) identifies the binaries and source hashes.
The model revision is `12fd25f77366fa6b3b4b768ec3050bf629380bac`. No model weights or
binaries are committed.
