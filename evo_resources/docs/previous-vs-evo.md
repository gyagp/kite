# Previous Qwen3.5-4B ONNX vs Evo

The [follow-up experiment](qwen35-optimization-v2.md) recovers most of this gap
using selective INT8 gates, without establishing a speedup over the previous model.
The figures below describe the earlier INT4 Evo export and its original workload.

**The previous WebGFX model is faster on the RTX 5080. Evo is 22.9% smaller, but decodes 30–35% slower.** The earlier reported Evo improvement was against its own initial export, not this tuned package.

## Packages and provenance

- Previous: `E:/workspace/project/webgfx-china/ai-models/Qwen3.5-4B/onnx-webgpu/`, downloaded from `webai-community/ai-models` at revision `6d5cd933ae1008d4cb78f88730248c464d3a50d1`. All eight requested ONNX-package/README files passed remote size and hash verification. GGUF was not downloaded.
- Evo: `artifacts/qwen35-4b-optimized/`.
- Both use the same original checkpoint revision, `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The previous README identifies the older GenAI base plus WebGFX-specific export/quantization changes.

## Performance

ORT 1.31.0, GenAI 0.17.1, WebGPU plugin 0.4.0, RTX 5080. Same source-tokenized workloads, batch 1, 32 forced generated tokens, max runtime length 3415. Runtime graph capture is off and validation is basic for both. Order: previous → Evo → Evo → previous. Each process used one warmup and five measured runs per prompt, yielding ten samples per model/prompt. Values below are medians.

| Prompt tokens | Previous TTFT | Evo TTFT | Previous decode | Evo decode | Evo decode change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 135 | 65.3 ms | 70.4 ms | 115.8 tok/s | 74.8 tok/s | -35.4% |
| 863 | 271.5 ms | 278.8 ms | 114.5 tok/s | 74.5 tok/s | -35.0% |
| 3383 | 1103.7 ms | 1125.8 ms | 105.6 tok/s | 73.5 tok/s | -30.4% |

The previous model had more run-to-run decode variation, but the result held in both orders. TTFT includes work through the first synchronized token observation; decode excludes the first observed token step. This is a native API workload comparison, not pure GPU shader timing.

## Size and graph differences

| Property | Previous | Evo |
| --- | --- | --- |
| Graph + external weights | 3,165,185,358 bytes (3.165 GB) | 2,441,906,186 bytes (2.442 GB) |
| Quantized MatMul nodes | 200 INT4 + 49 INT8 | 169 INT4 |
| Dense MatMul nodes | 0 | 48 |
| Block size | 32 | 32 |
| MTP | Disabled | Disabled |
| Head pruning | Enabled | Enabled |
| Gate/up graph layout | Separate projections | Fused projection with Split |
| Default max_length | 262144 | 4096 (benchmark overrides both) |

The model files are structurally valid. Tokenizer files differ in bytes, but all six raw-text checks and three chat-template checks produced identical tokens across previous, Evo and the source tokenizer. Benchmarking supplied the same source token IDs to both models.

## Runtime evidence

For two warmed 135-token prefill runs, the native profiles showed:

| Runtime node events, per run | Previous | Evo |
| --- | ---: | ---: |
| MatMulNBitsMlp | 32 | 0 |
| MatMulNBitsQkv | 8 | 8 |
| Dense MatMul | 0 | 48 |
| All node events | 774 | 900 |
| CPU node events | 1 | 6 |

The previous graph matches ORT WebGPU's fused `MatMulNBitsMlp` path; the Evo graph does not. Evo also retains 48 floating-point gate projections and has more shape/scalar CPU work. These are plausible contributors, not a measured causal breakdown of the decode gap: the profiles above are prefill profiles, and both precision and graph layout differ.

## Limited quality comparison

Both models matched the FP32 reference first token on all three prompts and produced finite logits. Lower KL is better:

| Prompt | Previous KL | Evo KL |
| --- | ---: | ---: |
| What is the capital of France? Answer with just the city name. | 0.000195 | 0.000423 |
| Write a Python function that returns the sum of the even numbers in a list. | 0.000236 | 0.000185 |
| Explain in two sentences why a smaller language model can run faster on a GPU. | 0.006983 | 0.049847 |

The previous model has lower KL on two prompts, Evo on one. These three first-token checks cannot establish overall quality. Generated 32-token continuations differ for all three benchmark prompts; the raw text is retained in `summary.json`.

## Recommendation

Use the previous package when native WebGPU throughput is the priority. Evo currently offers the smaller package, but it should not be presented as a performance upgrade over this prior model. The next controlled optimization work should preserve the graph pattern that permits native `MatMulNBitsMlp` fusion, compare the old mixed-precision selectors with the current floating gate policy, and test graph-capture preparation independently of runtime capture. Do not assume fewer ONNX nodes means faster native execution.

No model weights or runtime settings were modified during this comparison. The two packages remain intact.

## Raw evidence

All comparison files are under `artifacts/comparison-previous-onnx/`: `summary.json`, `artifacts.json`, `previous-a.json`, `evo-a.json`, `evo-b.json`, `previous-b.json`, `quality.json`, `tokenizers.json`, and `profiles.json`. The profile subdirectories contain the native traces and their input/model hashes.
