# Evo

**Local UI:** the current instance is at **http://127.0.0.1:8873**.
Start it with `.venv\Scripts\evo ui --port 8873`. Performance combines models
and measurements by prompt and machine; Wishlist lists models to convert.
[Workbench guide](docs/workbench.md)

**Qwen-Image-2.1-Turbo** supports GGUF → ONNX conversion, native WebGPU denoising,
image editing, and matched speed/image-fidelity reports. The generation command,
patched-runtime requirements and measurement limits are in the
[Qwen-Image guide](docs/qwen-image21.md). Performance parity with the GGUF baseline
must be assessed from the measured results, not assumed from conversion success.

**EmbeddingGemma 2 multimodal embeddings** now support text, images, audio,
video, and mixed inputs on native WebGPU.
[Multimodal model and usage](docs/embeddinggemma2-multimodal.md);
[smaller text-only model](docs/embeddinggemma2.md).

**Latest Qwen model optimization:** Evo now recognizes separate quantized gate/up
projections and fuses all 32 MLPs in the previous model, retaining its weight
policy. The GDN + MLP candidate and source-backed runtime suggestions are in
[model authoring improvements](docs/model-authoring-improvements.md).

The current project runtime uses the locally built upstream WebGPU plugin.
[Runtime selection](docs/native-runtime.md) and
[captured FP16-state experiments](docs/fp16-state-results.md) record its exact
revision, measurements and the unresolved capture-equivalence issue.

Convert models to ONNX and **continue optimizing existing ONNX models** for native
ONNX Runtime WebGPU. Evo offers a Mobius-style API/CLI, upstream pass integration,
profiling, benchmarking, and suggestions for future runtime work.

**First target: Qwen/Qwen3.5-4B, text generation.** Start with
[`recipes/qwen35-4b-recommended.toml`](recipes/qwen35-4b-recommended.toml) for its
text decoder. [Model-specific notes](docs/models/qwen35-4b.md) describe the hybrid
state contract and the separate path needed for vision.

The initial export experiment measured approximately 20–23% lower time to first
token and up to 33% faster decode against Evo's first export on an RTX 5080.
That comparison does not establish a gain over the previously tuned WebGFX model.
See [results, limits and reproduction](docs/qwen35-results.md).

**Updated comparison:** the previously tuned WebGFX/Hugging Face package is
faster than Evo's earlier INT4 package. That export is 22.9% smaller but decodes 30–35% slower
on the same RTX 5080 workloads. The gains above were against Evo's initial export.
See [previous model versus Evo](docs/previous-vs-evo.md).

The [follow-up optimization](docs/qwen35-optimization-v2.md) identifies dense recurrent
gates as a major regression source. Selective INT8 gates recover most of the gap,
but the controlled comparison still favors the previous package. No new Qwen
replacement is recommended yet.

Evo provides:

- **Existing ONNX input**: Mobius cleanup/fusion/lowering/post-weight passes,
  preserved source artifacts and I/O contract checks.
- **GenAI model builder**: precision, quantization, fusion and structured builder controls.
- **Mobius**: FP16/FP32 exports with WebGPU-aware graph construction.
- **Inspection**: ONNX checking, operator/type inventory, nested graphs, and external
  weight file existence and declared range checks without loading those weights.
- **Provenance**: resolved recipe, exporter command, package versions, logs,
  elapsed time, and SHA-256 hashes of generated artifacts.
- **CPU comparison**: compare named NPZ inputs/expected outputs using ONNX Runtime.
- **Native WebGPU**: generation benchmarks and ONNX/NPZ profiling.
- **Analysis**: current graph opportunities, runtime/export gaps, and proposals
  for new kernel/runtime work.

Architecture support belongs to the selected upstream exporter. The supplied
generic recipes are starting candidates. The recommended Qwen3.5 recipe has the
limited qualification documented above. Automatic search, a browser runner and
an Olive adapter are not implemented. See [optimization coverage](docs/optimization-coverage.md).

## Optimize an existing ONNX model

Install the `mobius` and `validation` extras, then:

```powershell
.venv\Scripts\evo optimize path/to/model.onnx --output artifacts/optimized
.venv\Scripts\evo analyze artifacts/optimized/model.onnx --output analysis.json
```

A package directory is also accepted. Single-file input needs no recipe or
original checkpoint. External weights are serialized with the new graph and
the source is retained. Changes to the declared I/O contract cause a failed
report instead of silently changing caller bindings. `--role` and `--dtype`
select optimizer capabilities; they do not perform blanket dtype casts.

Use `--contrib` to try Evo's guarded `MatMulNBitsMlp` rewrite. It uses the actual
ORT contrib-op contract and slices compatible packed weights without requantization.
This is opt-in: the Qwen candidate did not show a meaningful decode gain, and
floating-point results changed. The UI records it for validation and comparison.
`evo contrib` lists installed operator schemas and reviewed WebGPU guidance.

Profile with exact named NPZ inputs (requires the `native` extra):

```powershell
.venv\Scripts\evo profile model.onnx --inputs inputs.npz --output artifacts/profile
.venv\Scripts\evo analyze model.onnx --profile artifacts/profile/ort_<timestamp>.json --output analysis.json
```

Suggestions distinguish available transformations from runtime development
proposals. Graph-only opportunities have no claimed performance gain.

## Start

Python 3.11+ is required. A separate environment per exporter is useful when their
Torch, Transformers, or ORT requirements differ.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\evo plan recipes/qwen35-4b-recommended.toml
```

Build directly, in the style of Mobius:

```powershell
# Add --dry-run to inspect the command without building.
.venv\Scripts\evo build --model Qwen/Qwen3.5-4B --output artifacts/qwen35-4b --ep webgpu --precision int4
```

Or use Python (after installing the GenAI extra below):

```python
import evo

manifest = evo.build(
    "Qwen/Qwen3.5-4B",
    output="artifacts/qwen35-4b",
    execution_provider="webgpu",
    precision="int4",
)
```

`build()` writes the package directly and returns the manifest. Use
`dry_run=True` to return a plan, or `source_kind="local"` for a checkpoint
directory. For Mobius float exports, set `backend="mobius", precision="fp16"`.

On Linux/macOS, use `.venv/bin/python` and `.venv/bin/evo`. `plan` requires no
exporter installation and downloads no model files. Paths in recipes are relative
to the recipe file, so invocation from another working directory is supported.

Install the exporter you need, then convert:

```powershell
.venv\Scripts\python -m pip install -e ".[genai]"
.venv\Scripts\evo convert recipes/qwen35-4b-recommended.toml

# A smaller, separate Mobius example (not the Qwen3.5 baseline):
python -m venv .venv-mobius
.venv-mobius\Scripts\python -m pip install -e ".[mobius]"
.venv\Scripts\evo convert recipes/qwen25-0.5b-mobius-fp16.toml --python .venv-mobius/Scripts/python.exe
```

Conversion downloads the selected model if necessary. Use the Hugging Face CLI
or environment for authentication; do not put credentials in recipes. The initial
adapters do not enable repository-defined Python code. Check the backend's model
support before selecting a checkpoint.

For a local checkpoint (including one downloaded at an immutable HF commit):

```toml
backend = "genai"
source = "../models/my-checkpoint"
source_kind = "local"
output = "../artifacts/my-checkpoint-int4"
precision = "int4"
activation_dtype = "fp16"
block_size = 32
graph_capture = false
```

Local sources must be checkpoint directories. GGUF and arbitrary PyTorch objects
are not supported by Evo's initial adapters. A Hugging Face model ID follows
upstream resolution and is **not revision-pinned**. For reproducibility, record
the source commit alongside a local snapshot recipe. Current manifests hash
outputs, not source checkpoints or exporter source trees.

Conversion requires an absent or empty output directory. Failed builds retain
their logs and manifest; use a new directory for a retry. Success produces:

```text
artifacts/<recipe>/
  *.onnx                    # One or several graphs, depending on exporter
  ...                       # External weights and exporter-owned assets
  *.inspection.json         # Structural evidence per graph
  export.log
  evo-manifest.json
```

Export progress goes to `export.log`. Follow it from another terminal with
`Get-Content <output>/export.log -Wait`. The manifest says `exported` only after
ONNX structural checks pass. Accuracy, WebGPU execution, and performance remain
`not_run`; a structurally valid graph can still lack kernels in a particular ORT
release or produce inaccurate outputs. Exporter activation settings are requests;
inspect actual tensor types because mixed-precision graph details vary by model.
Legacy ORT operators in the default ONNX domain are checked using schema
declarations from the installed `onnxruntime` package. Reports record those
schema imports and that runtime version; they still do not prove kernel support.
When using a separate exporter environment, install `[validation]` in Evo's
environment too if these schemas are required.

## Inspect or compare a model

```powershell
.venv\Scripts\evo inspect artifacts/my-model/model.onnx --report inspection.json
.venv\Scripts\python -m pip install -e ".[validation]"
.venv\Scripts\evo validate-cpu artifacts/my-model/model.onnx --inputs inputs.npz --expected expected.npz --report accuracy.json
```

Create NPZ files with `numpy.savez`, using exact ONNX input/output names as keys.
Expected outputs should come from the original framework on the same tokenized
inputs, masks, positions and cache state. All outputs are compared; NaN/Inf fail,
integer outputs use exact equality, and float outputs use explicit `--rtol` and
`--atol` tolerances (defaults: 0.001 and 0.0001).

This command runs **CPUExecutionProvider**. Some WebGPU-targeted graphs do not
have CPU kernels; CPU session failures are reported as errors, not compatibility
verdicts. A passing comparison validates only those supplied samples. Quantized
language models also need representative prompts and task-quality evaluation.

## What makes an export good for WebGPU?

A useful candidate balances quality, model size, GPU memory, prefill latency,
and decode speed on a specified device/runtime/workload. INT4, FP16 and graph
capture are candidates to measure, not universally optimal settings.

Native deployment needs matching ORT/GenAI WebGPU runtime packages and kernels,
GPU-resident caches and the Qwen3.5 hybrid-state bindings. The default export
extra includes CPU ORT for conversion and checking; the `native` extra adds the
WebGPU plugin. The local environment has both installed. Browser support needs a separate loading and
generation layer. See [the validation plan](docs/design.md) and
[the upstream comparison](docs/upstream.md).

## Development

```powershell
.venv\Scripts\python -m pip install -e ".[validation]" pytest ruff
.venv\Scripts\python -m pytest
.venv\Scripts\ruff check .
.venv\Scripts\ruff format --check .
```

Advanced upstream controls use `[genai_options]`, `[genai_config]`,
`[mobius_options]` and `[mobius_save]` tables in recipes, or matching dictionary
arguments to `evo.build()`. `evo capabilities` inventories installed controls,
pass selections and source fingerprints. See the
[coverage report](docs/optimization-coverage.md) for provider restrictions and
the distinction between exposed, applied and measured optimizations.

With `uv`, the equivalents are `uv sync --extra validation`, `uv run pytest`,
and `uv run ruff check .`. Heavy exporter extras are optional. Tests exercise real
ONNX graphs and CPU inference, plus simulated exporter success/failure paths;
they do not download pretrained models or establish exporter/WebGPU performance.

An optional, real exporter smoke test creates a small random Qwen3.5 checkpoint
with three linear-attention layers and one full-attention layer, then exports it:

```powershell
.venv\Scripts\python scripts/smoke_genai.py --output artifacts/my-smoke-run
```

It requires `[genai]` and a new output directory, and downloads no pretrained
weights. See [verified results](docs/verification.md) for the tested environment
and remaining qualification work.
