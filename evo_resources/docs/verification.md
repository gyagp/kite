# Initial verification

This file records the initial setup. Current full-model conversion, native GPU
measurements and quality evidence are in [Qwen3.5 results](qwen35-results.md).

Verified locally on Windows x64 on 2026-10-11.

- 26 automated tests passed. They cover recipe/API/CLI validation, external data,
  nested graphs, preserving existing output directories, exporter failure states,
  artifact hashes, CPU numerical comparison, and ORT legacy schema validation.
- Ruff lint and formatting checks passed.
- The **real GenAI exporter** converted a small randomly initialized Qwen3.5
  checkpoint to INT4 ONNX targeting WebGPU. It contained three linear-attention
  layers and one full-attention layer. Export plus checking and hashing took
  approximately 4.5 seconds in this run; this is not an inference benchmark.
- The generated graph passed structural checks using ONNX plus the installed
  ORT legacy operator schema for `SimplifiedLayerNormalization`. No graph nodes
  were changed for the check. Tests also verify invalid attributes fail and
  temporary schema registration is removed afterward.
- Exported operators included `CausalConvWithState`, `LinearAttention`,
  `LinearAttentionGate`, `GroupQueryAttention`, `MRotaryEmbedding`, `GatedRMSNorm`,
  and `MatMulNBits`. Export success does not establish their GPU kernel coverage.

Local smoke evidence (ignored generated artifacts):

```text
artifacts/smoke-qwen35-verified/export/evo-manifest.json
artifacts/smoke-qwen35-verified/export/model.inspection.json
artifacts/smoke-qwen35-verified/export/export.log
```

Tested packages: Python 3.11.12, ONNX 1.23.1, ONNX Runtime 1.31.0,
ONNX Runtime GenAI 0.17.1, Torch 2.14.1, Transformers 5.18.0, ONNX IR 1.0.0,
ONNX Script 0.7.2, NumPy 2.4.6. The smoke manifest records exporter-environment
versions; the inspection report records its ORT schema source. These versions
are a tested local combination, not a cross-platform compatibility guarantee.

The Mobius 0.1.0 wheel's parser was inspected to verify its positional output
argument and WebGPU/dtype flags. No Mobius model export was executed.

Full pretrained Qwen3.5-4B weights have not been downloaded or exported. Native
WebGPU execution, multi-step reference accuracy, tokenizer parity, quality and
performance evaluation remain to be implemented and measured. The target recipe
is an initial candidate, not an optimality claim.
