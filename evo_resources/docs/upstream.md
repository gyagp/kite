# Upstream choices

Reviewed on 2026-10-11. These are separate roles in the ecosystem, with some
overlap. Evo keeps its first layer small and delegates graph construction.

| Project | Role in Evo | Why / limit |
| --- | --- | --- |
| [ORT GenAI builder](https://github.com/microsoft/onnxruntime-genai/blob/5c19002edce50beaa3a5c7eeefd8fe5269ee8d19/src/python/py/models/README.md) | Initial INT4 adapter | Architecture-specific export, attention/KV-cache conventions, quantization and WebGPU flags; support varies by architecture/release. |
| [Mobius](https://github.com/onnxruntime/mobius/tree/88f7d045388ccdcd98e8a23f4abcbcdfd39f65e6) | Initial float adapter | ONNX-native graph construction and EP-aware lowering across several model families; graph construction does not itself prove target-runtime kernel support. |
| [Olive](https://github.com/microsoft/Olive/tree/a0e849c4376039f6b2b03945907bdb552a974806) | Future workflow integration | Already orchestrates conversion, quantization, evaluation and search. Use it when calibration or multi-pass optimization is needed, instead of implementing an equivalent search engine here. |
| [Optimum ONNX](https://github.com/huggingface/optimum-onnx) | Possible broader exporter | Useful to investigate for generic Transformers tasks beyond the initial adapters; not evaluated or integrated in this first version. |

The commits above identify the source reviewed, **not** the exporter builds used
by every Evo invocation. Initial dependency constraints select GenAI 0.17.x and
Mobius 0.1.0. Their actual installed versions are written to each manifest.
Model support and CLI options need rechecking when these versions change.

## GenAI

Evo passes `-e webgpu -p int4` and exposes `block_size`, `use_webgpu_fp32`, and
`enable_webgpu_graph`. The initial recipe uses INT4 block 32, FP16 I/O and graph
capture off as a comparison starting point. This is an engineering default,
not a measured recommendation for a particular GPU.

Keep exporter-generated assets. A GenAI package includes runtime conventions
that are not automatically a browser application's tokenizer/generation loop.
The full pretrained Qwen3.5-4B checkpoint has now run through GenAI and Evo's
Mobius post-processing path; see [measured results](qwen35-results.md).

## Mobius

Evo passes `--ep webgpu`, an explicit floating dtype, and ONNX external data.
Local checkpoints use `--config`; Hugging Face IDs use `--model`. Other Mobius
features, including quantized-source handling and GGUF conversion, are outside
this initial adapter. A dtype request does not guarantee every initializer is
that dtype, particularly for already-quantized source checkpoints.

The **0.1.0 PyPI wheel** uses a positional output directory while current `main`
documents `--output`. Evo now pins the reviewed source commit and invokes the
public build/save API, avoiding this CLI difference and exposing the full pass pipeline.

## Olive

[Olive workflows](https://microsoft.github.io/Olive/how-to/configure-workflows/build-workflow.html)
can incorporate model-specific passes and evaluation. The current upstream
includes GenAI and Mobius-related integration, so Evo should reuse it for
calibrated quantization and optimization experiments. Before adding an adapter,
select one concrete model and pin a working workflow, its output layout, and
the versions of the passes it uses. Adding an arbitrary CLI passthrough would
not establish those contracts.

## Runtime references

- [ORT WebGPU usage and I/O binding](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)
- [ORT Web external data and large-model loading](https://onnxruntime.ai/docs/tutorials/web/large-models.html)
- [WebGPU operator coverage in ORT](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md)

Inspect the operator/type/attribute combination against the exact deployed ORT
build. A static operator-name allowlist is insufficient: kernel support depends
on versions, dtypes, attributes and shapes, and runtime partitioning determines
where execution actually occurs. Evo intentionally reports this evidence as
`not_run` until runtime validation exists.
