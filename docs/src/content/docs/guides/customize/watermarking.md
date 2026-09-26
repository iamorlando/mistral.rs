---
title: Text watermarking
description: Select library watermark schemes through Rust, Python, and HTTP, with CPU and GPU sampling integration.
---

mistral.rs delegates watermark algorithms to the standalone Rust `llm-watermarking`
library. Six token schemes are available through Rust, Python, and HTTP generation:
SynthID-Text, KGW, Unigram, exponential-race, inverse-transform, and MPAC. SemStamp
is available through embedding helpers; it cannot be attached to token generation.

Watermarking is opt-in. Omit `watermark` or set it to `null` for ordinary sampling.
All four generation endpoints accept the same configuration: `/v1/chat/completions`,
`/v1/completions`, `/v1/responses`, and `/v1/messages`. Interactive slash commands
do not configure watermarks. Image and audio generation are outside this feature.

For local development, place `llm_watermarking` beside this repository. The workspace
uses `llm-watermarking = { path = "../llm_watermarking", version = "0.1.0" }`.
Both projects use the same pinned Candle Git revision. Mistral enables the library's
`candle` feature and forwards its `cuda` and `metal` build features.

## Configuration

Every scheme requires a secret 32-byte key encoded as 64 hexadecimal characters.
Generate it once and retain it with the tokenizer revision, scheme parameters,
vocabulary size, and implementation version:

```bash
export MISTRALRS_WATERMARK_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
```

The examples read this variable; the server does not automatically read or enforce
it. Rust debug formatting redacts keys. Serialized configurations and request-body
logs contain the key, so store and transmit them accordingly.

| `scheme` | Parameters beyond `key` | Detection evidence |
| --- | --- | --- |
| `synthid` | `ngram_len=5`, `depth=30` | `tokens_scored`, `mean_g_value` |
| `kgw` | Required `vocab_size`; `context_width=1`, `green_fraction=0.5`, `delta=2`, `ignore_repeated_ngrams=true` | Green counts, expected/observed rate, nominal z-score |
| `unigram` | Required `vocab_size`; `green_fraction=0.5`, `delta=2`, `ignore_repeated_tokens=true` | Green counts, expected/observed rate, nominal z-score |
| `exponential` | Required `vocab_size`; `sequence_len=1024`, `start_position=0` | `tokens_scored`, `mean_cost` (lower is stronger) |
| `inverse_transform` | Required `vocab_size`; `sequence_len=1024`, `start_position=0` | `tokens_scored`, `mean_cost` (lower is stronger) |
| `mpac` | Required `vocab_size`, nonempty `payload`; `radix=2`, `context_width=1`, `delta=2`, `ignore_repeated_ngrams=true` | Decoded payload, per-symbol votes, winning fraction |
| `semstamp` | Required `embedding_dim`; `num_hyperplanes=8`, `green_fraction=0.25`, `margin=0.02`, `max_attempts=100`, `ignore_repeated_transitions=true` | Valid sentence transitions and nominal z-score |

`vocab_size` is the model's logits width, including padded output slots. It need
not equal the tokenizer's reported vocabulary size. A mismatch is rejected before
building vocabulary-sized watermark tables. MPAC payload entries are radix-r
symbols; they are bits only when `radix=2`. Positions for the two keyed samplers
count generated tokens, excluding the prompt, and wrap at `sequence_len`.

Native library validation checks parameter ranges. Unknown schemes and unknown
fields are rejected. Existing HTTP configurations without `scheme` still select
SynthID. The existing Rust/Python `SynthIdTextWatermarkConfig` remains supported.

## Examples for every scheme

`examples/watermarking/` contains a JSON file for each of the seven schemes. The
six generation configurations use the [Qwen3-4B output vocabulary of 151936](https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json).
The SemStamp file uses three-dimensional synthetic embeddings for a small demo.
All runners replace the fixture key with `MISTRALRS_WATERMARK_KEY`.

```bash
cargo run --release -p mistralrs --features metal --example watermarking -- examples/watermarking/kgw.json
python examples/python/watermarking.py kgw
python examples/server/watermarking.py kgw --endpoint chat/completions
```

Replace `kgw` with `synthid`, `unigram`, `exponential`, `inverse_transform`, or
`mpac` to run each token scheme. For NVIDIA builds use `--features cuda`; omit the
feature for CPU builds. Python uses the backend compiled into its extension.
The examples set `top_k=40`, enabling the supported GPU top-k insertion point.
The HTTP runner also supports `completions`, `responses`, and `messages`.

## Rust

```rust
use mistralrs::{RequestBuilder, TextMessageRole, Watermark, WatermarkConfig};

let config: WatermarkConfig = serde_json::from_value(serde_json::json!({
    "scheme": "kgw",
    "key": std::env::var("MISTRALRS_WATERMARK_KEY")?,
    "vocab_size": 151936,
    "delta": 2.0
}))?;
let detector = Watermark::new(&config)?;
let request = RequestBuilder::new()
    .add_message(TextMessageRole::User, "Write a long story about a lunar garden.")
    .set_sampler_temperature(0.8)
    .set_sampler_topk(40)
    .set_sampler_watermark(config);
// Send the request with the model, then score its original token IDs.
let evidence = detector.detect(&tokens, prompt_len, &eos_token_ids)?;
```

`SamplingParams.watermark` accepts `WatermarkConfig` directly. Existing SynthID
configurations convert with `.into()`; `set_sampler_watermark` accepts either type.
`Watermark::apply_tensor` delegates dense probability rows to the library without
reading their values back. `WatermarkTensor::Probabilities` retains categorical
selection; `WatermarkTensor::SelectionScores` requires argmax. Its input contract
is finite, nonnegative weights with positive mass. The output remains on the input
device. The host supplies already-filtered weights and retains sampling ownership.

## Python

```python
import os
from mistralrs import ChatCompletionRequest, WatermarkConfig

watermark = WatermarkConfig(
    os.environ["MISTRALRS_WATERMARK_KEY"],
    scheme="mpac",
    vocab_size=151936,
    payload=[1, 0, 1, 1],
)
request = ChatCompletionRequest(
    model="Qwen/Qwen3-4B",
    messages="Write a long story about a lunar garden.",
    temperature=0.8,
    top_k=40,
    max_tokens=512,
    watermark=watermark,
)
evidence = watermark.detect(tokens, prompt_len=prompt_len, eos_token_ids=eos_ids)
```

The generic detector returns a dictionary with scheme-specific fields and a
`kind` discriminator. The older `SynthIdTextWatermarkConfig.detect` continues
returning `(mean_g_value, tokens_scored)`.

## HTTP

For an OpenAI client, put the configuration in `extra_body`:

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-4B",
    messages=[{"role": "user", "content": "Write a long story about a lunar garden."}],
    temperature=0.8,
    max_tokens=512,
    extra_body={
        "top_k": 40,
        "watermark": {
            "scheme": "unigram",
            "key": os.environ["MISTRALRS_WATERMARK_KEY"],
            "vocab_size": 151936,
        },
    },
)
```

SemStamp configurations deserialize but generation rejects them with an explicit
sentence-embedding requirement. There is no HTTP sentence-rejection or watermark
detection endpoint.

## SemStamp

SemStamp accepts sentence embeddings from a fixed encoder. It needs complete
candidate sentences, embedding, acceptance testing, and retry/commit handling.
Those operations are not part of mistral's token sampler. The adapter exposes the
library's semantic watermark without constructing another generation pipeline.

Rust callers use `Watermark::semstamp()` for acceptance, signatures, and the
library's `prepare_tensor` operation. `detect_embeddings` scores host embeddings;
`detect_embeddings_tensor` scores Candle embeddings on their existing device.
Python exposes `accepts_embedding(previous, candidate)` and
`detect_embeddings(embeddings, prompt_len=0, device_name="cpu")`; the latter also
accepts `metal` and `cuda` when compiled in. Python lists are uploaded when a GPU
is requested; detection reads only the library's validation status and signatures.

```bash
MISTRALRS_WATERMARK_DEVICE=metal cargo run --release -p mistralrs --features metal --example watermarking -- examples/watermarking/semstamp.json
python examples/python/watermarking.py semstamp --device metal
```

These examples use synthetic vectors, not sentence generation or a robustness
benchmark. Real generation and detection must use the same embedding model.

## Sampling and device behavior

The order remains penalties/processors, temperature and softmax, top-k/top-p/min-p
filters, watermarking, and token selection. Excluded tokens remain excluded.
Reported logprobs refer to the unwatermarked, pre-filter model probabilities.
Greedy sampling has no watermark choice.

CPU generation supports all six token schemes, including speculative probabilities.
The position-keyed samplers produce a point mass for a fixed key, position, and
model distribution; speculative acceptance uses that distribution rather than
treating selection scores as probabilities. Context and position are derived from
sequence history, so clones and discarded branches do not advance shared state.

CUDA and Metal single-sequence top-k sampling run the watermark transformation
on the GPU. Existing top-k kernels identify candidates; Candle prepares filtered
weights; the library applies its algorithm. One compact readback supplies the
existing host draw and original reporting probabilities. No full probability
vector is downloaded, and the library's strict scalar validation readback is
avoided by its trusted tensor API.

This path currently requires active temperature, `top_k` from 1 to 128, no requested
top logprobs, no speculative sampling, and no multiple-sequence mode. Existing
GPU penalty support is reused; custom logits processors and active DRY use the
CPU path. Metal logits bias also uses the CPU path. Other request shapes retain
existing host sampling and CPU watermarking. CUDA fused batch/resident and sparse
speculative verification remain ineligible for watermarked requests.

See [GPU integration boundaries](/guides/customize/watermarking-gpu/) for exact
library and host API gaps. This feature does not replace mistral's GPU backend.

## Detection and format

Prefer original token IDs. Detokenizing and re-encoding can change IDs, and
completion-only detection loses initial context for context-dependent schemes.
Provide the prompt length and EOS IDs when available. For position-keyed sampling,
use the generation's starting position or adjust it for a known cropped prefix.

Scores are uncalibrated evidence, not probabilities of AI authorship. Counts,
mean g-values, alignment costs, and decoded payload votes are different statistics.
Repetitions, short samples, low-entropy output, editing, and retokenization affect
detection. Calibrate each scheme with representative marked and unmarked data.
Possession of a key permits generating its signal; a watermark is not a signature.

SynthID retains the previous mistral domain `mistralrs-synthid-text-v1` followed
by a zero byte. Its digest is SHA256(domain || decoded key || LE32(ngram_len) ||
LE32(context tokens) || LE32(candidate)); layer bits are read least significant bit
first within each byte. Other schemes use the library's versioned default domains.
See `llm_watermarking/docs/formats.md` for their precise formats. Matching only the
algorithm name does not establish byte compatibility with another implementation.
None of these independent keys detect private Claude or Gemini watermarks.

## Prior implementation discussions

Before implementation, on 2026-09-25, the upstream
`EricLBuehler/mistral.rs` and fork `iamorlando/mistral.rs` GitHub issue/PR searches
were checked for `watermark`, `watermarking`, and `SynthID`, including closed items
and comment searches. Local tracked code and available Git history were also
checked, starting at commit `2370966bb`.

No relevant text-watermarking discussion, abandoned implementation, or recorded
maintainer rationale was found. Upstream [PR #2030](https://github.com/EricLBuehler/mistral.rs/pull/2030) mentions a CUDA memory-pool
watermark, which is unrelated. There is consequently no evidenced reason to
attribute the previous absence to a maintainer decision.

The implementation concerns identified in the code were sampler ordering,
speculative verification, GPU paths that bypass ordinary sampling, repeated
contexts, and a matching detector. These are engineering findings from this
change, not historical reasons given by maintainers. Anthropic's undisclosed
production parameters also prevent an exact replica of its deployed watermark.
