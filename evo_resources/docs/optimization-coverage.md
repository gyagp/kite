# Upstream optimization coverage

Subsequent [comparison with the prior WebGFX package](previous-vs-evo.md) found
an important interaction: Evo's explicit gate/up projection fusion prevents the
native `MatMulNBitsMlp` pattern from matching in this graph, while the previous
package gets 32 such runtime nodes per prefill. Exposing or enabling an upstream
optimization does not guarantee the fastest combination. The previous package
is currently the faster native WebGPU baseline.

Audited on 2026-10-11 against:

- ONNX Runtime GenAI **0.17.1 installed wheel**, including its builder, Qwen
  specialization and quantization configuration. Source hashes and the 58
  discovered literal extra-option keys are in [the machine-readable inventory](upstream-capabilities.json).
- Mobius commit [`88f7d045`](https://github.com/onnxruntime/mobius/tree/88f7d045388ccdcd98e8a23f4abcbcdfd39f65e6),
  pinned in Evo's optional dependency. Its complete EP-aware pipeline is called
  directly; Evo does not maintain a copied subset of its rewrite rules.
- ORT runtime source commit [`8fc9fd59`](https://github.com/microsoft/onnxruntime/tree/8fc9fd597453baae1ad73379c313f0586d812867).
  Runtime tests used ORT 1.31.0 and WebGPU plugin 0.4.0. The audited source commit
  is not asserted to be the exact source of those released binaries.

“Exposed” means Evo can request a control through the upstream interface.
“Applied” means it ran in the selected pipeline. “Matched” means a graph pattern
actually changed. These are different claims. All controls are not mutually
compatible, nor beneficial on every model. The tests do not exhaust every
combination, quantization algorithm, architecture, or device.

## GenAI

`genai_options` forwards upstream extra options. `genai_config` exposes
`target_options`, `drafter_options`, `speculative_options`, `runtime_config`, and
`search`, retaining upstream's structured validation. Managed output, auth and
duplicate recipe settings cannot be overridden in the extra-options table.

| Optimization family | Evo coverage | Qwen3.5-4B outcome |
| --- | --- | --- |
| Architecture-specific graph construction and folding | Delegated to builder, automatically | Applied; includes hybrid convolution/recurrent state and 8 full-attention layers. |
| RMS/skip normalization, activation and bias fusions | Builder defaults and runtime graph transforms | Applied where compatible. Native trace includes QuickGelu and SkipSimplifiedLayerNormalization. |
| QKV packing, GQA, rotary/MRoPE and Q/K norm fusions | Builder defaults; extra-option controls | Native trace includes GroupQueryAttention and MatMulNBitsQkv. MRoPE/layout restrictions remain upstream-owned. |
| MLP gate/up projection fusion | `fuse_mlp_gate_up` | Enabled in selected recipe. |
| Last-position LM head | `prune_lm_head` | Enabled; generation-only output contract. Conflicts with MTP export. |
| Shared/tied embeddings and quantized Gather | `shared_embeddings`, quantization options; upstream defaults | Quantized embedding observed in export. Explicit sharing is available but was not separately tuned. |
| INT4/INT8/FP16/FP32 and activation dtype | Recipe precision and dtype | INT4 with FP16 activation request selected. Mixed state types remain exporter-owned. |
| Block size, accuracy level, symmetry, QDQ/QOperator | Recipe and `genai_options` | Compared block 32/128; selected block 32, accuracy level 4, MatMulNBits. |
| RTN/k-quant/other supported quantizers and mixed precision | `algo_config`, `matmul_mixed_precision`, structured target options | Exposed. Calibration-dependent alternatives not exhaustively evaluated. |
| Per-node exclusions, per-layer/type overrides, head/embedding policy | Structured target options and upstream extras | Exposed; no speculative blanket re-quantization of an existing ONNX file. |
| KV-cache quantization and calibration scales | Upstream KV-cache options | Exposed but not selected. Quality, shape, kernel and scale-file requirements need separate qualification. |
| Graph capture preparation | Recipe `graph_capture` | Built and measured; slower here, disabled in selected recipe. |
| MTP | Upstream default; `exclude_mtp`, MTP quant options | Baseline includes MTP; selected export disables it to permit head pruning. |
| Paged attention, compact state, block drafters/speculation | Structured configuration/upstream extras | Exposed, with upstream restrictions. No drafter checkpoint supplied; CUDA-only paths are not enabled for WebGPU. |
| CUDA/TensorRT prepacking or NVFP4/FP8 specializations | Upstream controls remain available | Not claimed as WebGPU optimizations. Unsupported provider combinations must fail upstream validation. |

## Mobius

Direct builds use `mobius.build()` followed by `ModelPackage.save()`. Public,
JSON-compatible build/save arguments are available through `mobius_options` and
`mobius_save`. Upstream `MOBIUS_*` environment flags are inherited by the worker;
the inventory records their defaults. Unknown public arguments fail signature
validation rather than being silently ignored.

Existing ONNX inputs use `evo optimize`, with role and floating dtype selected
explicitly or conservatively inferred. This runs:

1. Identity elimination, constant lifting, initializer deduplication, common
   subexpression elimination, dead-node/opset removal, symbolic inference,
   bounded constant folding and cleanup metadata passes.
2. The complete capability-selected fusion set: for FP16 WebGPU decoders,
   GQA, QKV packing, SkipNorm, SkipLayerNorm and GeluFusion.
3. Capability-selected custom-function inlining and lowering. For the audited
   WebGPU profile, StaticEmptyKV is selected. SeparateRoPE, UnpackQKV, rank-4
   normalization workarounds and other EP-specific lowerings remain gated by
   upstream capabilities instead of being forced onto WebGPU.
4. Final folding, CSE and dead-input cleanup, then post-weight Concat/Transpose
   folding and removal of unused nodes. Functional metadata is preserved.
5. Debug metadata stripping, external-weight serialization and structural checks.

The audited flags also cover opset compatibility and static-cache bias policy.
FP8 KV cache is an upstream CUDA-only capability at this revision; the inventory
records that restriction. Model-specific static-cache/Gemma/MoE optimizations
apply only when their graph and task contracts match.

On the chosen GenAI Qwen graph, all selected stages ran but found **zero
additional fusion matches**. Evo nevertheless retains this full pass pipeline
for other ONNX inputs. A test graph with a redundant Identity demonstrates an
actual standalone optimization with identical CPU outputs.

Two integration adaptations preserve GenAI's native contract: register real ORT
legacy schemas while checking/optimizing, and remove newly added MatMulNBits /
SkipSimplifiedLayerNormalization fallback functions when the source already
used those native kernels. Otherwise Mobius's opset-24 fallback bodies make a
valid opset-22 GenAI graph fail checking. Existing source functions are retained.

## Native runtime controls

`evo benchmark --provider-options` exposes native WebGPU options; `--search-options`
exposes GenAI runtime search settings. This includes graph capture, validation,
buffer pools, int64 and adapter hints where supported by the installed runtime.
They are recorded in benchmark results. GPU-resident generation/state handling
is delegated to native GenAI. The selected package defaults to a 4096-token
runtime limit to avoid allocating caches for the source's full 262144 context.

## Suggestions requiring development

`evo analyze` combines the graph inventory and optional ORT trace. It distinguishes
available transformations, existing runtime/export gaps, and proposed runtime work.
Trace durations are not treated as GPU shader timings, and a first run is omitted
when multiple `model_run` events exist to reduce cold-compilation distortion.

- **Existing capability, integration gap:** ORT's
  [WebGPU GatedDeltaNet](https://github.com/microsoft/onnxruntime/blob/8fc9fd597453baae1ad73379c313f0586d812867/onnxruntime/contrib_ops/webgpu/bert/gated_delta_net.cc)
  already implements Qwen gates and Q/K L2 normalization. GenAI 0.17.1's exporter
  restricts that path to CUDA. Evo now implements a checked Qwen3.5-4B rewrite,
  including FP32 V-major state and checkpoint-verified gate parameters. It requires
  the separate GDN-capable runtime. See [model authoring improvements](model-authoring-improvements.md).
- **Proposed kernel work:** fuse convolution-to-recurrent input preparation.
  The audited [convolution implementation](https://github.com/microsoft/onnxruntime/blob/8fc9fd597453baae1ad73379c313f0586d812867/onnxruntime/contrib_ops/webgpu/bert/causal_conv_with_state.cc)
  and recurrent implementation are separate. A prototype could reduce intermediate
  writes and dispatches, but must validate state updates and measure GPU costs.
- **Proposed kernel/runtime contract:** a quantized LM-head plus greedy selection
  epilogue. No argmax/top-k epilogue was identified in the audited
  [MatMulNBits implementation](https://github.com/microsoft/onnxruntime/blob/8fc9fd597453baae1ad73379c313f0586d812867/onnxruntime/contrib_ops/webgpu/quantization/matmul_nbits.cc)
  or its MLP/QKV helpers. Keep a full-logits path for sampling, penalties and
  consumers requiring logits. This is a bounded source observation, not a claim
  that no related implementation exists anywhere in ORT.

Unimplemented proposals have no claimed speedup. The current Qwen experiment
also retains a locally built GDN kernel patch and native validation; it has not
been published upstream. Evo does not file upstream issues automatically.
