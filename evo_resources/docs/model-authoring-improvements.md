# Model authoring improvements

Evo can optimize the existing Qwen3.5-4B ONNX package without exporting or
quantizing its transformer weights again. The latest candidate is
`artifacts/qwen35-4b-gdn-mlp`. It combines 24 GatedDeltaNet layers with 32 fused
quantized MLPs and Mobius's existing WebGPU optimization pipeline.

## What Evo learned and implemented

| Source | Applied in Evo | Evidence / boundary |
| --- | --- | --- |
| [Mobius projection tracing](https://github.com/onnxruntime/mobius/blob/88f7d045388ccdcd98e8a23f4abcbcdfd39f65e6/src/mobius/rewrite_rules/_qmoe_fusion.py) | Match separate gate/up projections and `gate * sigmoid(gate) * up`, including commuted multiplication inputs. | `evo optimize --contrib` now emits 32 `com.microsoft::MatMulNBitsMlp` nodes in the previous Qwen graph. |
| [ORT fused MLP contract](https://github.com/microsoft/onnxruntime/blob/8fc9fd597453baae1ad73379c313f0586d812867/onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits_mlp.cc) | Preserve the original packed weights, scales, accuracy level, and activation contract. | Reject different inputs, quantization policies, exposed intermediates, explicit zero points, and unsupported layouts. No weight re-quantization. |
| [Mobius capability-selected passes](https://github.com/onnxruntime/mobius/blob/88f7d045388ccdcd98e8a23f4abcbcdfd39f65e6/src/mobius/_optimizations.py) | Continue to run cleanup, supported fusions/lowering, and post-weight folding after Evo-specific rewrites. | `optimization.json` records actual matches, operator counts, source fingerprints, and preserved I/O. |
| GenAI architecture-specific construction and ORT GatedDeltaNet | Fold Q/K normalization and Qwen gate arithmetic into 24 recurrent operators. | `python -m evo.qwen_gdn` verifies checkpoint gate parameters and changes recurrent state to FP32 V-major. Fresh caches and the custom runtime are required. |
| GenAI native provider configuration | Use canonical `WebGPU` in `set_provider_option`. | Shipped GenAI commit `83de55a1f3886cddbb32a5517e26adb3f9d65159` merges names before normalizing aliases. Lowercase calls can create duplicate entries and leave new options unused. |

`evo analyze` now reports the separate-projection fusion opportunities using the
same matcher in dry-run mode. A smaller graph or a fused operator is a candidate,
not automatic evidence of a speedup. The native MLP implementation can still
dispatch its unfused path for subgroup, DP4A, or wide-tile configurations.

## Native validation and performance

The candidate requires
`artifacts/runtimes/ort-webgpu-gdn-subgroup/onnxruntime_providers_webgpu.dll`,
SHA256 `37eada54d15c5ae36972c86f09e9e4185b313330fe3135fa5bd5770be6ac5f5d`.
The installed WebGPU plugin 0.4.0 does not contain GatedDeltaNet. The runtime
directory includes its source patch, build manifest, and matching DXC DLLs.

RTX 5080, Ryzen 9 9950X, Windows, ORT 1.31.0 and GenAI 0.17.1. The paired test
keeps both graphs resident on the **same custom runtime**, alternates AB/BA each
repetition, reuses generators, synchronizes each decode step through `is_done`,
and reads back the final sequence. D3D12, basic validation, capture off,
8,192-token allocation, batch one, two warmup pairs and eight measured pairs.
Input IDs follow the deterministic WebGFX 128/512/1024-token workloads; output
length is 128 tokens.

| Prompt tokens | Previous decode | Candidate decode | Median gain | Previous TTFT | Candidate TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 108.54 tok/s | 115.35 tok/s | 6.3% | 55.89 ms | 54.46 ms |
| 512 | 109.97 tok/s | 114.70 tok/s | 4.3% | 163.96 ms | 164.07 ms |
| 1024 | 111.85 tok/s | 114.74 tok/s | 2.6% | 323.01 ms | 316.41 ms |

The candidate wins 20 of 24 paired decode samples. The 512-token prompt's
paired bootstrap interval includes zero; its gain remains less certain.
These measurements isolate the combined graph changes, not the MLP rewrite
alone. They do not establish a universal speedup or compare absolute throughput
across different host-load conditions. Evidence:
`artifacts/comparison-gdn-interleaved/{previous,candidate,summary}.json`.

A confirmation using 512 generated tokens, two warmup pairs and four measured
pairs per prompt gives the following. All samples generated the requested 512
tokens and were deterministic within each model. The candidate wins 10/12 pairs;
the 512-token prompt's median improvement is within normal variation.

| Prompt tokens | Previous decode | Candidate decode | Median gain |
| ---: | ---: | ---: | ---: |
| 128 | 103.55 tok/s | 112.62 tok/s | 8.8% |
| 512 | 108.32 tok/s | 109.13 tok/s | 0.7% |
| 1024 | 111.68 tok/s | 114.60 tok/s | 2.6% |

Evidence: `artifacts/comparison-gdn-interleaved-512`. Both test series used the
same effective 8192-token setting; this is also the candidate package's final
default. The final config hash is recorded in the 512-output reports.

On the same runtime and reference trajectories, the previous graph matches
197/218 reference top tokens (mean KL 0.0403892); the candidate matches 200/218
(mean KL 0.04033497). Maximum KL is 0.72786 and 0.75398 respectively. The new MLP
rewrite leaves every saved per-position quality metric unchanged. These
are 15 English, Chinese, code and longer-text prompts with up to 16 cached
reference positions, stopping at EOS. This is a conversion regression screen,
not full task-accuracy or long-context qualification.

Free greedy generation on the same 15 prompts checked 1,048 finite-logit steps,
with a 128-token response limit. Examples include correct Paris and 391 answers,
Chinese explanation, translation, JSON content and Python/SQL generation.
Responses reaching the token limit are truncated. Evidence:
`artifacts/comparison-gdn-interleaved/candidate-generation.json`.

The automated suite passes 58 tests. The MLP tests include three native inputs
for both packed and separate layouts with CPU fallback disabled, plus rejection
of incompatible contracts. A full-model native trace additionally observes 512
`MatMulNBitsMlp` and 384 `GatedDeltaNet` executions on `WebGpuExecutionProvider`
across 16 steps (`artifacts/gdn-mlp-native-proof`). The runtime's 20 independent
GDN kernel/state checks also pass; these do not substitute for model-level checks.

## Capture remains experimental

After fixing the provider alias, a native profile verified real replay:
16 model runs, three ordinary executions followed by 13 runs without node
execution. However, both graphs showed worse cached-reference agreement with
capture: mean KL about 0.0884–0.0887, versus about 0.0404 without capture.
Faster capture timings are therefore not recommended results.

A mask-length rewrite modeled on GenAI's current capture preparation produced
identical logits and did **not** resolve this discrepancy. The experiment is
retained in `scripts/experiments/qwen_capture.py`; it is not a selected Evo pass.
The cause still needs isolation. In particular, the shipped GenAI source has
recurrent double-buffer graph variants, so their absence in an older local
checkout must not be reported as the cause.

Earlier reports made through lowercase provider-option calls cannot establish
that the requested runtime settings were applied. Their metadata now identifies
that limitation. Timing overlapping known workloads is rejected, including
compiler/test processes and Git pack operations. This guard cannot detect every
possible background workload.

## Use and reproduce

```powershell
.venv\Scripts\python -m evo.qwen_gdn E:\workspace\project\webgfx-china\ai-models\Qwen3.5-4B\onnx-webgpu --checkpoint models/qwen35-4b --output artifacts/my-gdn
.venv\Scripts\evo optimize artifacts/my-gdn --output artifacts/my-gdn-mlp --contrib
.venv\Scripts\evo benchmark artifacts/my-gdn-mlp --ep-library artifacts/runtimes/ort-webgpu-gdn-subgroup/onnxruntime_providers_webgpu.dll --cases benchmarks/qwen35-webgfx.json --max-length 8192 --new-tokens 128 --warmup 2 --repeats 8 --reuse-generator --synchronization is_done --require-idle --provider-options '{"dawnBackendType":"D3D12","validationMode":"basic","enableGraphCapture":"0"}' --output artifacts/my-gdn-mlp-benchmark.json
```

Use fresh output directories. The distributed candidate's default
`search.max_length` is also set to 8192; a fresh reproduction inherits the source
package's default until explicitly changed or overridden as above. Keep all
model assets together and all runtime DLLs together. Existing FP16 K-major
recurrent caches cannot be reused with this FP32 V-major graph.

## Suggested runtime work

These are source-grounded proposals, not implemented speedups:

1. Extend `MatMulNBitsMlp` fusion to the DP4A/subgroup paths used by relevant
   shapes. Its current fast-path guard bypasses fused decode when those unfused
   implementations qualify. Benchmark real 2560-by-9216 Qwen gate/up matrices
   and verify native shader selection, not just ONNX operator names.
2. Fuse convolution output preparation with the recurrent operator: avoid
   intermediate split/reshape/transpose traffic around 24 hybrid layers while
   preserving convolution and recurrent cache contracts.
3. Offer a quantized LM-head plus greedy-selection epilogue for callers that
   need only the chosen token. Retain full logits for sampling, penalties and
   validation; no compatible fused epilogue was found in the audited kernels.
4. Isolate the capture numerical discrepancy using recorded recurrent states,
   both graph variants and fixed reference tokens. Do not recommend capture
   from throughput alone.

The user-facing Wishlist remains a list of models to convert. Runtime proposals
belong in this engineering document, not in that list.
