# Qwen-Image-2.1-Turbo: GGUF → ONNX

Evo can import Unsloth's Q4_K_M GGUF denoiser directly into ONNX. It translates
Q4_K, Q5_K, Q6_K and Q8_0 blocks to ORT `MatMulNBits` without a new calibration
or quantizer. Source integer codes are retained except for zero-scale constant
blocks, which need an equivalent encoding. FP16 scales/zero points and FP16
activations introduce rounding; this is not a bit-identical GGUF execution engine.

The ONNX target covers the denoiser, including cached text/reference-image prefix
attention. Text encoding and RGBA VAE decoding currently use the official
PyTorch components. Both text-to-image and image editing are supported. This is
a native ORT WebGPU pipeline; a browser-only package is not yet provided.

## Sources and dependencies

Tested source revisions:

- Qwen/Qwen-Image-2.1-Turbo: `fc13ba1986fcb905854dc8de2f52eee395f49b5b`.
- unsloth/Qwen-Image-2.1-Turbo-GGUF: `5d1f9f759ea877ae556cda88dbec958b62ad3809`.
- Diffusers: `b800f828f3951c4d206e154411bcbfa693cd322e`.
- Transformers: `536ecc007387a50e77603bb5d92100e9b07514cc`.

Use Python 3.11 and a compatible CUDA Torch/Torchvision installation for the
shared components and GGUF baseline. The tested image environment has Torch
2.14.1+cu130 and Torchvision 0.29.1. Install Evo and the pinned authoring libraries
from [requirements/qwen-image21.txt](../requirements/qwen-image21.txt) in an
isolated environment. Native inference additionally needs Evo's `native` extra.

Download the source and GGUF with Hugging Face Hub:

```python
import json
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

source = Path("models/qwen-image-2.1-turbo")
revision = "fc13ba1986fcb905854dc8de2f52eee395f49b5b"
snapshot_download("Qwen/Qwen-Image-2.1-Turbo", revision=revision, local_dir=source)
(source / "evo-source.json").write_text(json.dumps({
    "repository": "Qwen/Qwen-Image-2.1-Turbo", "revision": revision
}))
hf_hub_download(
    "unsloth/Qwen-Image-2.1-Turbo-GGUF",
    "qwen-image-2.1-turbo-Q4_K_M.gguf",
    revision="5d1f9f759ea877ae556cda88dbec958b62ad3809",
    local_dir="models/qwen-image-2.1-turbo-gguf",
)
```

The original source is needed for configuration, the text encoder and VAE;
GGUF supplies the denoiser weights. Observe the source model's Qwen Research
License when distributing derived artifacts.

## Generate the model

The following commands run from the Evo project directory. `--dependencies`
adds isolated library directories; omit those arguments in a dedicated environment
where the requirements are already installed. Each output directory must be new.

```powershell
$imageDeps = @('--dependencies', '.cache/embeddinggemma2/deps',
               '--dependencies', '.cache/qwen-image21/deps')
$source = 'models/qwen-image-2.1-turbo'
$gguf = 'models/qwen-image-2.1-turbo-gguf/qwen-image-2.1-turbo-Q4_K_M.gguf'
$package = 'artifacts/qwen-image-2.1-turbo-gguf-floatzp'

.venv/Scripts/evo image-export $source --output $package `
  --gguf $gguf --gguf-zero-points float @imageDeps
.venv/Scripts/evo image-optimize $package --short-blocks
```

The package contains prefix and denoising graphs plus external weights. The
`short-blocks` variant stores one Q6_K matrix as decoded FP16, avoiding its
block-size-16 kernel. It does not choose a new quantizer. Prefix and denoising
graphs share weights. The conversion manifest records hashes and per-tensor
rounding checks against independent GGUF dequantization.

`--gguf-zero-points float` requires the local ORT patch adding floating zero
points to the native WebGPU `MatMulNBits` kernels. Stock runtime support must
not be assumed from the ONNX schema alone. The compatible fallback is
`--gguf-zero-points correction`, which adds an affine-correction projection and
was slower on this machine.

Additional opt-in experiment:

```powershell
.venv/Scripts/evo image-optimize $package `
  --merge-projections artifacts/qwen-image-2.1-turbo-gguf-projections
```

This combines same-format attention and gate/up projections without requantizing
weights, and uses `--graph-variant projections`. Its one-prompt screen did not
improve total latency; do not assume fewer operators means faster inference.

## Prepare inputs, run and decode

Select a Vulkan-enabled self-built provider with the floating-zero-point patch.
The ORT Python host must expose session-scoped GPU OrtValue allocation and
`release_captured_graph`; our tested 1.31.0 host is build `402979a`. A stock
version number alone does not establish this API compatibility.

```powershell
$env:EVO_WEBGPU_LIBRARY = (Resolve-Path `
  'artifacts/runtimes/ort-webgpu-c64c9fd764-bk64/onnxruntime_providers_webgpu.dll').Path
$fixtures = 'artifacts/qwen-image-2.1-turbo-fixtures'
$prefix = 'artifacts/qwen-image-2.1-turbo-gguf-final-prefix'
$result = 'artifacts/comparison-qwen-image21/onnx-final'

.venv/Scripts/evo image-prepare $source `
  --cases benchmarks/qwen-image-2.1-turbo.json --output $fixtures @imageDeps
.venv/Scripts/evo image-generate prefix --source $source --package $package `
  --fixtures $fixtures --output $prefix --webgpu-backend Vulkan `
  --graph-variant short-blocks @imageDeps
.venv/Scripts/evo image-generate generate --source $source --package $package `
  --fixtures $fixtures --prefix-dir $prefix --output $result --backend onnx `
  --webgpu-backend Vulkan --graph-variant short-blocks `
  --warmup 1 --repeats 3 --require-idle @imageDeps
.venv/Scripts/evo image-generate decode --source $source `
  --fixtures $fixtures --output $result @imageDeps
```

Capture is enabled by default. Persistent GPU inputs and outputs are reused
while latent values and timestep features change each step. Nonlearned Sin/Cos
time features are prepared on the host using the source FP32 frequency constants;
the learned denoiser stays on WebGPU. `--profile` is for separate validation:
verify `model_run` count exceeds `SequentialExecutor::Execute`, and check provider
assignment. Compare the resulting eight-step trajectory with a separate
`--no-capture` run before treating capture speed as valid.

GGUF baseline, using the **same** prepared inputs:

```powershell
.venv/Scripts/evo image-generate generate --source $source --gguf $gguf `
  --fixtures $fixtures --output artifacts/comparison-qwen-image21/gguf-final `
  --backend gguf --warmup 1 --repeats 3 --require-idle @imageDeps
.venv/Scripts/evo image-generate decode --source $source --fixtures $fixtures `
  --output artifacts/comparison-qwen-image21/gguf-final @imageDeps
```

This baseline uses Diffusers' GGUF loader and CUDA/BF16 execution, not
stable-diffusion.cpp. The runtime, activation dtype, prefix handling and
quantization parameters differ; this is not a format-only speed comparison.
Compare denoising **plus separately measured ONNX prefix time**. These timings
exclude shared text encoding, VAE decode, model loading and shader compilation.

After decoding both results, generate a reproducible JSON summary and gallery
against saved images from `--backend reference`:

```powershell
.venv/Scripts/evo image-report --onnx $result `
  --gguf artifacts/comparison-qwen-image21/gguf-final `
  --reference artifacts/comparison-qwen-image21/reference `
  --prefix $prefix --output artifacts/comparison-qwen-image21/final-report @imageDeps
```

Reference generation uses the same command and fixtures with `--backend reference`;
the BF16 denoiser uses group offloading on this 16 GB GPU. It is a fidelity
reference and is not used as the speed baseline.

For image editing, use `benchmarks/qwen-image-2.1-turbo-edit.json` with distinct
fixture/prefix/result directories. It contains the official boat-sketch input
path; update that path when moving the project. Reference images are encoded
through the source VLM and VAE, with block-causal attention preserved in the
ONNX prefix cache.

**Use `--fp32-accumulation` for both `prefix` and `generate` when editing on
Vulkan.** The default FP16 path produced NaNs during capture replay on the boat
fixture. This is an unresolved runtime interaction: the first invalid values
appear in the first block's K/V projections; isolated projections pass 100
captured repetitions. Neither disabling ordinary memory reuse nor changing
submission batching resolved it. The FP32 option selects other MatMul kernels;
it is a validated workaround, not a root-cause runtime fix. The eight-step
captured and uncaptured editing trajectories match exactly with this option.

## Runtime work and evidence

The experimental worktree is
`E:/workspace/project/ws-ort/ort-prs/evo-qwen-image-gguf`, based on
`c64c9fd76404df85bbbac37485d5c0f2babcd194`. Runtime artifact directories contain
the exact patch, DLL hashes, CMake-cache hash and build metadata. Builds must
set `onnxruntime_ENABLE_DAWN_BACKEND_VULKAN=ON`; setting only
`DAWN_ENABLE_VULKAN=ON` is insufficient on this Windows configuration.

Changes under evaluation:

- Floating FP16/FP32 zero points in generic, wide and subgroup-matrix kernels;
  float zero points bypass integer DP4A, with input-type-sensitive shader keys.
- Larger subgroup-matrix K tiles, guarded by K divisibility and actual device
  workgroup storage limits. Performance findings are specific to this RTX 5080.
- Possible future improvements: share activation prepacking across parallel
  projections; support efficient native Q6_K blocks; specialize diffusion
  attention using GPU timestamp evidence. These are suggestions, not measured
  improvements or existing Evo features.

## Measured results, October 17–18, 2026

RTX 5080 16 GB, Ryzen 9950X, Windows 11, driver 616.64. Each main result uses one
warmup and three measured images per prompt, 1024×1024, eight steps, CFG 1.
ONNX uses the `short-blocks` graph and the BK64 native Vulkan provider. Denoising
plus prefix latency excludes shared text encoding/VAE and startup costs.

| Prompt | GGUF CUDA | ONNX WebGPU | ONNX latency change |
| --- | ---: | ---: | ---: |
| Photograph | 8.026 s | 8.787 s | +9.49% |
| Typography | 8.057 s | 8.831 s | +9.61% |
| Composition | 8.022 s | 8.843 s | +10.23% |
| Mean of prompt medians | 8.035 s | 8.820 s | +9.78% |

A reverse-order control retained the same ranking. The short-block model's
external files occupy about 4.85 GB, versus the 4.20 GB GGUF. Mean AlexNet LPIPS
distance from the BF16 reference is **0.1914 for ONNX**, **0.1803 for GGUF**;
lower means closer to those reference images. All three images preserve the
requested subjects and readable poster title, with changes to framing and
object geometry. This three-prompt screen does not establish broad image quality.

The capture trace shows **16 calls, one ordinary execution and 15 replays**.
All profiled operators use WebGPU. Eight captured latent states are bit-identical
to the uncaptured control for the photograph, and repeats are identical across
the three performance prompts. The 512×512 editing fixture separately passes
the same eight-step equivalence check with FP32 accumulation.

The ONNX candidate is usable for these validated workloads, but is **not a faster
or closer-to-reference replacement for GGUF** on these measurements. The target
of comparable-or-better latency and fidelity remains open. The GGUF baseline is
Diffusers/CUDA without CUDA graphs; stable-diffusion.cpp has not been benchmarked.

Evidence: `artifacts/comparison-qwen-image21/final-report/summary.json`,
`comparison.jpg`, and `capture-validation.json`. Performance in the local UI
contains six prompt-specific observations for the two models. Failed editing
attempts and preliminary screens remain on disk and are not performance records.
Evo's lightweight suite passed 73 tests (two optional suites skipped), and the
isolated image integration/translation suite passed 12 tests. The selected native
runtime passed 60 numerical cases across Vulkan and D3D12.
