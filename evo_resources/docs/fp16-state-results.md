# FP16 recurrent state: latest native WebGPU

The requested checkout at `E:/workspace/project/ws-ort/onnxruntime` was updated
to upstream `86d60730c14f546ba1a4ba06b4887b53285117a5` and built successfully on
2026-10-12. Evo now selects its packaged DLL, SHA256
`3082fe6fe9a425bcc49f821ee5b28daad8aace3aa622547f554aeac29af2eaf2`.
This is pristine upstream code, not the earlier subgroup experiment.
[Build and runtime selection](native-runtime.md).

## Precision and speed

GatedDeltaNet's schema and native kernel require FP32 state. Its `TS` constraint
also covers decay, beta, gate parameters and optional state updates. FP16 state
cannot be enabled by simply changing the ONNX input/output types.

The experiment therefore adds 24 input casts and 24 output casts around the
unchanged GDN computation. It stores FP16 caches, retaining the same weights,
MLP fusion, layout and FP32 gate parameters. This tests boundary rounding and
cast overhead, not native FP16-state GDN. Within a multi-token prefill, recurrent
accumulation remains FP32.

Across three 512-step English, Chinese and code trajectories, FP16 cache matches
**1533/1536 top-token predictions** of captured FP32 GDN. Mean KL is
**0.00008537**, maximum **0.0027702**. Both follow identical reference tokens
with persistent state and stop at EOS; all three prompts reach the step limit.
This is a limited precision screen against a quantized reference, not broad
task accuracy or long-context qualification.

The cache tensors halve from 48 MiB to 24 MiB per batch-one buffer set, or from
96 MiB to 48 MiB for separate past/present buffers. Cast tensors and scratch
allocations mean this is not a measured 48 MiB reduction in peak GPU memory.

**Graph capture was enabled for both models** in the following RTX 5080 test:

| Prompt tokens | FP32 GDN | FP16 cache + casts | Change |
| ---: | ---: | ---: | ---: |
| 128 | 142.52 tok/s | 138.44 tok/s | -2.87% |
| 512 | 141.08 tok/s | 136.27 tok/s | -3.41% |
| 1024 | 140.65 tok/s | 136.66 tok/s | -2.84% |

FP16 cache is slower in all 18 measured pairs. Both models use the same runtime,
batch one, 8192-token allocation, deterministic WebGFX token IDs, 128 output
tokens, two warmup pairs and six measured pairs per prompt. Models remain
resident and alternate AB/BA order. Known competing workloads are checked before
cases and after pairs. The collection completed and exited successfully.
Evidence: `artifacts/comparison-latest-fp16-cache-capture`.

The final native FP16 LinearAttention + MLP comparison completed on the same
upstream runtime with capture enabled and the same six-pair protocol:

| Prompt tokens | FP32 GDN decode | FP16 LinearAttention decode | FP32 TTFT | FP16 TTFT |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 122.63 tok/s | 119.70 tok/s | 70.49 ms | 62.55 ms |
| 512 | 122.54 tok/s | 121.18 tok/s | 221.31 ms | 178.30 ms |
| 1024 | 123.12 tok/s | 120.71 tok/s | 416.53 ms | 345.80 ms |

Native FP16 LinearAttention gives 11.3–19.4% lower TTFT and 1.1–2.4% slower
decode. It changes the attention operator and normalization/gate path as well
as state precision, so this does not isolate the cost of FP16 state.
Evidence: `artifacts/comparison-latest-linear-fp16-capture-final`. Absolute rates
vary across process collections; compare models within each paired collection.

## Capture replay and its remaining correctness issue

Traces prove replay for FP32 GDN, FP16-cache GDN and native FP16 LinearAttention
with MLP fusion. Each variant's trace contains 48 model calls: 9 ordinary executions and
**39 captured replays** across three prompts. Evidence:
`artifacts/comparison-latest-capture/*-replay.json`.

An identical-model control compares uncaptured and captured FP32 GDN. The first
three positions agree exactly; differences start at replay (step 3). Captured
execution shows 9 ordinary calls and 87 replays across 96 calls, whereas the
uncaptured session executes all 96 normally. Maximum KL reaches 1.399 across
three 32-step sequences. The discrepancy persists with this runtime/GenAI
combination; its root cause remains unresolved.

Captured checks against the original-framework fixtures, using 8192 allocation:

| Model | Matching reference top tokens | Mean KL |
| --- | ---: | ---: |
| FP32 GDN + MLP | 193/218 | 0.0884913 |
| FP16-cache GDN + MLP | 193/218 | 0.0879972 |
| Native FP16 LinearAttention + MLP | 194/218 | 0.0884346 |

The timings quantify captured execution cost; they do not qualify captured
generation as equivalent to uncaptured generation. Numerical/profile prechecks
overlapping other host work contain no performance measurements.
All measured samples generated the requested 128 tokens, but token IDs were not
deterministic across generator reuse for any of the measured model/prompt series.
This is additional capture/reuse correctness evidence, not a claim of stable
greedy generation. The raw reports retain every generated sequence.

One earlier smaller-context check printed its results but did not exit promptly
and was interrupted. All three checks using the performance test's 8192-token
allocation completed with exit 0. No causal explanation for the earlier
shutdown is claimed.

## Alternative and next runtime experiment

`artifacts/qwen35-4b-linear-fp16-mlp` preserves native FP16 LinearAttention and
adds the same 32 MLP fusions. Its captured numerical checks and replay trace
complete successfully. Its first timing collection was interrupted by a build;
partial records are excluded from Performance. Retry status:
`artifacts/comparison-latest-capture/status.json` now records completion; the
final complete results are tabulated above.

A useful next kernel experiment is **FP16 state storage with FP32 arithmetic
inside GatedDeltaNet**, eliminating the standalone casts. It requires a state
type separate from FP32 gate parameters, correct type/shape inference, typed
loads/stores, shader-cache coverage, and numerical/performance validation.
These results support investigating that design but do not prove its speedup.

Keep FP32 GDN selected for now. The cast-based candidate remains experimental;
resolving capture equivalence is separate from choosing state precision. No
ORT source patch was added to the updated upstream checkout.
