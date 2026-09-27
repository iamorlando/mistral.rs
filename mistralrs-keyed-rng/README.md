# Keyed sampling and native Metal

This crate adapts KeyedRngs' CPU Threefry2x32-20 reference to Candle's Metal device.
The dependency is pinned to authenticated remote HEAD
`9269c77b9bd0095688f93fa6d9e97624211c883b` (default branch
`fix/benchmark-regressions`). The reference engine and PrngKey files are identical
to the previously reviewed `d196b6c2abc3bd4700140f55c521e9d6c69880a9` revision.
Building requires access to the private KeyedRngs repository. Burn and WGPU are
disabled; only the CPU reference is linked.

## Replay contract

`keyed-threefry2x32-v1` derives a sequence key from the admitted seed with
`PrngKey::new(seed).fold_in(1)`. It derives a purpose key with `fold_in(purpose)`:
generation 0, draft 1, acceptance 2, correction 3, diagnostics 4. The high seed
word is key0, the low word key1. Counter0 is the generated token position;
counter1 is the attempt number. Both counters are bounded u32 values; host
conversions reject overflow. Sequence identity never includes a batch row or
scheduler slot. Mistral's existing choice-seed admission assigns each response
choice a seed. Unseeded sequences get a seed from rand's system-seeded generator
at admission, once per sequence.

Replacing a sequence's prompt advances its key with
`fold_in(0x7265626173650001)` and clears device history. Prefix-cache views do not
advance it; sampling history uses all committed tokens, including cached prefixes.

Metal uses the same 20 rounds, wrapping u32 arithmetic, key injections and
high-24-bit f32 conversion as KeyedRngs. `threefry.metal` is a native translation
of its WGSL helper, tested against the pinned library rather than a second local
CPU implementation. This is an inference PRNG, not a secret-key generator.

Replaying an event does not advance a mutable RNG. Grammar rejection changes the
attempt from 0 to 1. A rejected sample never updates history. Speculative draft,
acceptance, correction, and bonus generation use distinct purposes in the generic
host verifier. Diagnostics use purpose 4 and do not consume generation draws.
Model floating-point differences and candidate ordering can still change complete
CPU/Metal generation even when RNG bits match exactly.

## Device interface

`metal::select` accepts candidate weights, vocabulary IDs, unmodified reporting
probabilities, and `[key0, key1, position, attempt]` event rows on one Candle device.
It applies top-k, top-p relative to retained top-k mass, then min-p and categorical
selection on device. Nucleus inputs must be sorted. `metal::candidates` provides a
stable descending merge sort with vocabulary-ID tie breaking at arbitrary widths.
Greedy mode selects the maximum score with no categorical draw, which also permits
position-keyed scoring consumers. Vocabulary IDs remain u32, including IDs above
the exact f32 integer range. Selected logprobs come from the reporting tensor.

`Selection::tokens` is a device `[rows, 1]` tensor. `readback` is an explicit compact
reporting boundary; `readback_batch` combines records into one readback. Invalid
weights produce an error record and never an arbitrary valid token. `probe` is a
separate diagnostics operation, absent from the production path.

`DeviceHistory` owns prompt/generated counts, committed history, generated
position, and active/EOS state. Penalties and accepted-token commits run on Candle's
own command encoder. A selection may be retried before commit. The caller must
commit only valid selections for active sequences, then stop scheduling inactive
rows. Selected tokens can directly become subsequent model inputs. State is
sequence-owned and survives reordering. All bindings check device identity,
contiguity, dtypes, dimensions and view offsets; read/write bindings participate
in Candle's automatic barriers and buffer lifetime management.

## Integration boundaries

Set `MISTRALRS_SAMPLING_RNG=keyed-threefry2x32-v1` before starting the process.
The default remains Isaac64. This intentionally changes seeded output when enabled.
Ordinary eligible Metal batches keep logits/candidates on device and read one
compact batch of selected records. Device history supplies penalties and the next
one-token text decode input. Full logprob requests retain device selection and read additional reporting
probabilities afterward. Custom processors and DRY use the host sampler with the
same logical RNG contract. Grammar checks still run on
the CPU and can request attempt 1. Speculative verification stays on the generic
host path; keyed mode bypasses CUDA sampling paths that still use Isaac64 uniforms.

The server scheduler still consumes selected tokens for text decoding, streaming,
stop strings, grammar, tools, and cache scheduling before the next model step.
This is not a fully asynchronous Metal model decode scheduler. The low-level API
can queue selection/history/feedback operations without intermediate readback;
that property is tested independently. Multimodal-specific input processors and
speculative model inputs retain their existing compatible host paths.

Watermark and teaching-trace code is absent from the `master` base of this branch.
The device selector accepts distinct transformed and reporting probabilities and
separate diagnostics streams as its integration seam. No code from the dirty
watermarking branch is imported.

## Validation

```sh
cargo test -p mistralrs-keyed-rng --features metal
MISTRALRS_SAMPLING_RNG=keyed-threefry2x32-v1 cargo test -p mistralrs-core --features metal keyed_metal
cargo check -p mistralrs-core --features metal
cargo check -p mistralrs-core
```

Tests require a real Metal device and fail if it is unavailable. Coverage includes
[Random123 known answers](https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors),
4,119 exact integer/f32-bit reference cases, nonzero buffer
offsets, same-hardware/different-Candle-device rejection, categorical frequencies,
conditional retry independence, replay and reordering, filtering, unmodified
reporting probabilities, large IDs, invalid mass, stable sorting through 131,071
candidates, and queued device history/penalty/EOS updates.

These checks use synthetic logits and device feedback. No complete model generation
or end-to-end throughput benchmark was run. CUDA was not built or run on this Mac.

The [sampling benchmark and runtime readback audit](benchmarks/README.md) compare
the actual CPU and Metal samplers, including completed GPU execution. Local
measurements are stored beside that harness documentation.
