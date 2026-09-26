---
title: Text watermarking
description: Opt-in SynthID-Text tournament sampling, detection, and implementation limits.
---

mistral.rs supports opt-in **SynthID-Text tournament sampling** for generated text.
[Anthropic identifies its text watermark as a version of SynthID-Text](https://www.anthropic.com/news/claude-text-watermark).
This implementation uses the published two-candidate tournament algorithm with an
independent keyed hash. It does not reproduce Anthropic's private keys, parameters,
or detector, and does not detect Claude or Gemini watermarks.

Watermarking is disabled unless a request supplies `watermark`. It works through
the Rust SDK, Python bindings, and the HTTP `/v1/chat/completions`, `/v1/completions`,
`/v1/responses`, and `/v1/messages` endpoints. It applies to text token selection;
it does not watermark images, audio, or files with C2PA metadata. Interactive CLI
slash commands do not expose watermark configuration.

## Configure a key

Create a random 32-byte deployment key once and store it for both generation and
detection. For example:

```bash
export MISTRALRS_WATERMARK_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
```

This variable is read by the examples below; the server does not automatically
read it or enforce watermarking. A serving application can attach the configuration
to each request. Keep the key in trusted application code and use protected
transport when submitting it over HTTP. Request-body logs and serialized request
archives contain the key; Rust `Debug` output redacts it. There is no built-in
key, key registry, key rotation service, or per-user identifier.

| Field | Default | Accepted values |
| --- | --- | --- |
| `key` | Required | 64 hexadecimal characters encoding 32 random bytes |
| `ngram_len` | `5` | `2` through `32`, including the candidate token |
| `depth` | `30` | `1` through `256` tournament layers |

Retain the key, parameters, tokenizer revision, and implementation version to
reproduce detection. Unknown configuration fields and invalid parameter values
are rejected. Omit `watermark` or set it to `null` to keep ordinary sampling.

## HTTP generation

The OpenAI Python client can send the extension using `extra_body`:

```python
import os
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="unused")
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Write a long story about a lunar garden."}],
    temperature=0.8,
    max_tokens=512,
    extra_body={
        "top_k": 40,
        "watermark": {"key": os.environ["MISTRALRS_WATERMARK_KEY"]},
    },
)
print(response.choices[0].message.content)
```

The same `watermark` JSON object is a top-level field on the other supported HTTP
generation endpoints. It remains active during streaming and server-executed tool
continuations. It does not change the response schema or insert extra characters.

## Rust generation

```rust
use mistralrs::{RequestBuilder, SynthIdTextWatermarkConfig, TextMessageRole};

let config = SynthIdTextWatermarkConfig::new(
    std::env::var("MISTRALRS_WATERMARK_KEY")?,
)?;
let request = RequestBuilder::new()
    .add_message(TextMessageRole::User, "Write a long story about a lunar garden.")
    .set_sampler_temperature(0.8)
    .set_sampler_topk(40)
    .set_sampler_max_len(512)
    .set_sampler_watermark(config);
let response = model.send_chat_request(request).await?;
```

`SamplingParams.watermark` also accepts the configuration directly. Set a
non-greedy sampler: `RequestBuilder` starts with `top_k = 1`, so setting only a
temperature and watermark does not create a watermark signal.

The runnable Rust example generates text and scores it with the same tokenizer:

```bash
cargo run --release -p mistralrs --example watermarking
```

## Python generation and detection

```python
import os
from mistralrs import ChatCompletionRequest, SynthIdTextWatermarkConfig

watermark = SynthIdTextWatermarkConfig(os.environ["MISTRALRS_WATERMARK_KEY"])
request = ChatCompletionRequest(
    model="default",
    messages=[{"role": "user", "content": "Write a long story about a lunar garden."}],
    temperature=0.8,
    top_k=40,
    max_tokens=512,
    watermark=watermark,
)
response = runner.send_chat_completion_request(request)

# Use the exact tokenizer revision that generated this completion.
tokens = tokenizer.encode(response.choices[0].message.content, add_special_tokens=False).ids
mean_g_value, tokens_scored = watermark.detect(tokens)
print(mean_g_value, tokens_scored)
```

Here `runner` is an initialized mistral.rs `Runner` and `tokenizer` is the matching
`tokenizers.Tokenizer`. Prefer original generated token IDs when available:
decoding and re-encoding may change token boundaries. To score full prompt plus
completion IDs, pass `prompt_len` and the model's `eos_token_ids`. Detection stops
before the first generated EOS. To score completion-only IDs, use `prompt_len=0`;
the first `ngram_len - 1` tokens are then context only.

Rust exposes the same detector:

```rust
use mistralrs::SynthIdTextWatermark;

let detector = SynthIdTextWatermark::new(&config)?;
let evidence = detector.detect(&completion_token_ids, 0, &eos_token_ids)?;
println!("{:?}, {}", evidence.mean_g_value, evidence.tokens_scored);
```

The score is the mean of keyed binary g-values over eligible tokens and layers.
An independent unwatermarked sample has an expected score near `0.5`; marked text
typically scores higher. **This is evidence, not a probability of AI authorship.**
An empty or too-short sample yields `None` and zero scored tokens. Detection
excludes every repeated context, including repeats beyond the generation history
window, so repeated text cannot inflate the evidence count.

Choose decision thresholds using held-out marked and unmarked text representative
of the deployment, separated by sample length. Measure false positives and false
negatives with the intended key, model, tokenizer, sampling settings, and text
domains. The [DeepMind reference implementation](https://github.com/google-deepmind/synthid-text)
also calls for calibrated thresholds for mean-based detection. No universal
threshold or Bayesian detector is supplied here.

## Algorithm and sampling integration

The [SynthID-Text paper](https://www.nature.com/articles/s41586-024-08025-4) describes
multilayer tournaments where two independent candidate tokens compete according
to a pseudorandom binary score, with random tie breaking. For each layer, this
implementation computes that tournament's exact probability distribution:

```text
G = sum over tokens of p[token] * g[token]
p_next[token] = p[token] * (1 + g[token] - G)
```

This avoids drawing an exponentially large candidate set. The final token is
drawn using the request's normal RNG. Tournament sampling preserves the model
distribution in expectation over independent pseudorandom g-values, while a
fixed key and context change the conditional distribution.

The application order is penalties, custom logits processors, temperature and
softmax, top-k/top-p/min-p filtering, normalization, watermarking, and sampling.
Zero-probability tokens stay excluded. Grammar masking therefore remains binding.
Reported logprobs retain the existing pre-filter, pre-watermark model-probability
semantics; they are not probabilities under the keyed tournament distribution.

The version-1 g-function is fully specified as:

```text
digest = SHA256(
    ASCII("mistralrs-synthid-text-v1") || 0x00 || decoded_32_byte_key ||
    LE32(ngram_len) || LE32(context_token_1) || ... ||
    LE32(context_token_(ngram_len - 1)) || LE32(candidate_token)
)
g[layer] = (digest[layer / 8] >> (layer % 8)) & 1
```

All fields have fixed widths for a configuration. This is an independent hash
instantiation of the published method; the public Transformers sampling-table
hash and detectors are not byte-compatible with it. No Python or PyTorch runtime
is needed for generation or Rust detection.

Generation skips incomplete contexts and contexts seen in the previous 1,024
generation positions of that sequence. Prompt-only occurrences are not counted
as previous generation positions. History is derived from the supplied token
prefix, so sequence clones, discarded speculative branches, and grammar retries
do not mutate shared watermark state. Standard draft sampling and speculative
target probabilities include the watermark transformation. Sparse proposals
retain their supplied probabilities for the acceptance calculation.

## Limits and performance

Greedy decoding (`temperature=0` or `top_k=1`) has no choice to watermark.
Short samples, tightly constrained text, repeated passages, heavy editing,
retokenization, and paraphrasing can weaken detection. Low evidence does not
establish that text is human-written. Anyone holding the key can generate or
imitate the signal; this is not a cryptographic signature or proof of authorship.

Watermarking currently runs on the CPU with work proportional to the surviving
vocabulary size times `depth`. CUDA batch/resident sampling and CUDA/Metal top-k
fast paths are bypassed for marked requests; GPU inference itself remains
available. This can increase latency, particularly with an unfiltered vocabulary.
Benchmark your model and batch sizes before deployment. No model-quality or GPU
performance parity with Anthropic is claimed.

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
