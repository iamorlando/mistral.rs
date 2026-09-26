---
title: Watermark GPU integration boundaries
description: Device-resident integration, remaining library API gaps, and host sampling boundaries.
---

The watermark adapter uses the same Candle package as mistral. GPU algorithms
remain in `llm-watermarking`; no watermark kernel or GPU runtime is implemented
in mistral. The library's CUDA feature uses NVRTC for its own metadata kernels.

## Connected path

The single-sequence CUDA/Metal top-k path calls the existing
`cuda_topk_logits_f32_packed` or `metal_topk_logits_packed` operation. It retains
actual vocabulary token IDs, original logits, and the existing softmax normalizer.
Candle prepares the same top-p/min-p masks as the host sampler, then the library's
prepared operation runs with `apply_trusted`. The host selects a token after its
existing compact readback. Keyed sampler outputs use argmax; probability-transform
outputs use the existing weighted categorical draw.

Metal tensor views with a nonzero storage offset are copied on the GPU because
the existing top-k helper accepts a base buffer without an offset. This performs
no host transfer and preserves the view's contents.

The readback contains three arrays of K values: watermark weights/scores, token
IDs, and original reporting probabilities. Carrying original probabilities
preserves logprob semantics. There is one readback, not a CPU round trip for the
watermark operation, and no vocabulary-sized probability download. The adapter
uses no library strict-validation scalar readback.

## Library improvement: indexed candidates

The current library accepts only dense `[vocab_size]` rows. Mistral consequently
scatters its K candidates into a zero-filled vocabulary row on the same device,
then gathers the K results. This is correct but allocates vocabulary-sized work
and prevents efficient use of compact candidate buffers.

A useful library extension would accept:

- Device-resident candidate token IDs and their filtered weights.
- The original vocabulary size and the same key/context/position parameters.
- A result in the same candidate order, on the same device, with no readback.

The partition and hash formats must continue using actual vocabulary token IDs
and the full vocabulary definition. Hashing candidate ranks would break detection.
The library should own this indexed algorithm support; mistral should not recreate
its hashes, partitions, or tournament calculations.

## Host boundary: fused CUDA samplers

The current CUDA batch/resident and sparse speculative paths combine filtering,
selection, or acceptance inside existing kernels. They do not expose the filtered
probability tensor at the needed point. Removing their watermark exclusion would
silently skip watermarking or use the wrong acceptance distribution.

They remain excluded. Supporting them requires an explicit host insertion point
or composable sampling stages, plus the library's indexed/batched metadata APIs.
Implementing a second GPU sampler merely to bypass this boundary would duplicate
mistral's inference stack and is outside this integration.

The library also prepares context-dependent operations from host `&[u32]` history.
A fully resident sampler would benefit from preparation that consumes device token
history, prompt lengths, and per-row positions. Reading committed tokens back just
to construct watermark seeds would add synchronization; this adapter does not add
that workaround. Batched calls should preserve independent per-sequence keys,
contexts, positions, payloads, and retry semantics.

## SemStamp boundary

SemStamp operates on sentence embeddings and returns signatures/acceptance masks.
Mistral's token sampler has no sentence-encoder and candidate-sentence retry hook.
This is a workflow mismatch, not a missing CUDA or Metal implementation in the
library. Its embedding operations are exposed through Rust and Python helpers;
ordinary text-generation requests reject the scheme explicitly.

## Verification

Tests cover every token scheme against the CPU reference, actual token IDs in
compact rows, filtering order, exclusion support, original reporting probabilities,
replay, key-stream positions, and SemStamp embedding evidence. Feature-gated CUDA
and Metal tests execute both the device tensor operations and the existing GPU
top-k sampling entry point. Run them on the corresponding hardware:

```bash
cargo test -p mistralrs-core --features metal watermark -- --include-ignored
cargo test -p mistralrs-core --features cuda watermark -- --include-ignored
```

GPU floating-point reductions can differ from the scalar f64 reference. Hashes
and partitions must agree, while probability comparisons use tolerances. Synthetic
integration tests do not establish model quality or performance; benchmark the
intended model, vocabulary, top-k, batch sizes, and scheme.
