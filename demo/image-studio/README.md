# The local imagination studio

These four PNG previews are actual outputs of the local Qwen-Image-2.1-Turbo
pipeline, using Kite's integrated Evo implementation. They are not mock results.
`presets.json` stores editable prompts, seed, canvas and reference assets.
The per-image JSON files preserve runtime identity and measured phase timings.
Selecting **Generate image** or **Transform image** performs a fresh local run.

The boat-sketch reference, `boat-sketch.png`, is from the official
[Qwen/Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)
`assets/turbo-boat-input.png`, source revision
`fc13ba1986fcb905854dc8de2f52eee395f49b5b`. See that model's license and model card.
The other three prompts were authored for Kite and use no reference image.
Text and fine details in previews retain the model's output, including its
imperfections; previews have not been retouched.

The local model derives its denoiser from
[unsloth/Qwen-Image-2.1-Turbo-GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF),
revision `5d1f9f759ea877ae556cda88dbec958b62ad3809`, translated into ONNX.
WebGPU denoising uses FP32 accumulation on the selected Vulkan runtime. Text
encoding and VAE decoding use local PyTorch/CUDA. Bundled previews require no
GPU to view; producing new images requires the local model assets and runtime.
