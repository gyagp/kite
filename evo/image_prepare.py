"""Prepare identical Qwen-Image 2.1 inputs for ONNX, GGUF and reference runs."""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

from evo.image_timing import ImageTimer
from evo.paths import asset_path
from evo.pipeline import hash_file, write_json


def prepare(source, cases, output, components=None, timings=None, progress=None):
    report = progress or (lambda message: None)
    report("Initializing the image engine…")
    timer = ImageTimer(timings)
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Rope
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3VLModel

    timer.mark("initialization_ms")
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new fixture directory")
    output.mkdir(parents=True)
    torch.set_num_threads(8)
    cached = components is not None and components.get("encoder") is not None
    report("Loading the text encoder onto the GPU…")
    encoder = (
        components["encoder"]
        if cached
        else (
            Qwen3VLModel if components is not None else Qwen3VLForConditionalGeneration
        ).from_pretrained(
            source / "text_encoder",
            dtype=torch.bfloat16,
            local_files_only=True,
            device_map="cpu" if components is not None else "auto",
            **({} if components is not None else {"max_memory": {0: "10GiB", "cpu": "40GiB"}}),
            attn_implementation="sdpa",
        )
    )
    if not cached and components is not None:
        from evo.image_features import ImageFeatureEncoder

        encoder = ImageFeatureEncoder(encoder)
    if components is not None:
        from evo.cpu_weights import CpuWeights

        if not cached or "encoder_cpu_weights" not in components:
            components["encoder_cpu_weights"] = CpuWeights(encoder)
        encoder = encoder.to("cuda")
    processor = (
        components["processor"]
        if cached
        else AutoProcessor.from_pretrained(source / "processor", local_files_only=True)
    )
    pipeline = QwenImage21Pipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(source / "scheduler"),
        vae=None,
        text_encoder=encoder,
        processor=processor,
        transformer=None,
    )
    config = json.loads((source / "transformer/config.json").read_text())
    rope = QwenImage21Rope(10000, config["axes_dims_rope"])
    timer.mark("model_load_ms", cuda=True)
    results = []
    for index, case in enumerate(cases):
        if case["width"] % 32 or case["height"] % 32:
            raise ValueError("Resolution must be divisible by 32")
        image_paths = case.get("images", [case["image"]] if case.get("image") else [])
        input_images, vae_images, image_shapes = [], [], []
        if image_paths:
            report("Preparing your reference images…")
            from diffusers import AutoencoderKLQwenImage21
            from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
            from PIL import Image

            if pipeline.vae is None:
                timer.mark("input_preparation_ms")
                pipeline.vae = (
                    components.get("vae") if components is not None else None
                ) or AutoencoderKLQwenImage21.from_pretrained(
                    source / "vae", dtype=torch.bfloat16, local_files_only=True
                )
                pipeline.vae = pipeline.vae.to("cuda")
                pipeline.vae.enable_tiling()
                timer.mark("model_load_ms", cuda=True)
            for filename in image_paths:
                with Image.open(asset_path(filename)) as image_file:
                    image = image_file.convert("RGBA")
                width, height, _ = calculate_dimensions(
                    case.get("condition_resolution", 1024) ** 2, image.width / image.height
                )
                input_images.append(
                    pipeline.image_processor.resize(image, width=width, height=height)
                )
                vae_images.append(
                    pipeline.image_processor.preprocess(
                        image, width=width, height=height
                    ).unsqueeze(2)
                )
                image_shapes.append((1, height // 16, width // 16))
        timer.mark("input_preparation_ms")
        report("Encoding your prompt and references…")
        with torch.no_grad():
            features, mask, image_mask = pipeline.encode_prompt(
                case["prompt"], image=input_images or None, device=torch.device("cuda")
            )
        features = features.cpu().float().numpy()
        vlm_image_mask = image_mask.cpu().numpy()
        timer.mark("inference_ms", cuda=True)
        expanded_mask = np.repeat(vlm_image_mask[0], np.where(vlm_image_mask[0], 4, 1))
        expanded_text = np.repeat(features, np.where(vlm_image_mask[0], 4, 1), axis=1)
        prefix = expanded_text.shape[1]
        h, w = case["height"] // 16, case["width"] // 16
        image_tokens = h * w
        position = rope(
            [*image_shapes, (1, h, w)],
            torch.from_numpy(np.concatenate([expanded_mask, np.ones(image_tokens, bool)])),
            torch.device("cpu"),
        )
        generator = torch.Generator("cpu").manual_seed(case["seed"])
        noise = torch.randn((1, 1, 64, h, w), generator=generator, dtype=torch.bfloat16)
        noise = pipeline._pack_latents(noise, 1, 64, h, w).float().numpy()
        condition_latents = np.zeros((1, 0, 64), np.float32)
        if vae_images:
            timer.mark("input_preparation_ms")
            with torch.no_grad():
                _, conditioning = pipeline.prepare_latents(
                    vae_images,
                    1,
                    64,
                    case["height"],
                    case["width"],
                    torch.bfloat16,
                    torch.device("cuda"),
                    generator,
                    torch.from_numpy(noise),
                )
            condition_latents = conditioning.float().cpu().numpy()
            timer.mark("inference_ms", cuda=True)
        prefix_images = np.zeros((1, prefix, 64), np.float32)
        prefix_images[:, expanded_mask] = condition_latents
        causal = np.tril(np.ones((prefix, prefix), bool))
        positions = np.flatnonzero(expanded_mask)
        offset = 0
        for _, height, width in image_shapes:
            block = positions[offset : offset + height * width]
            causal[np.ix_(block, block)] = True
            offset += height * width
        path = output / f"{index}.npz"
        timer.mark("input_preparation_ms")
        np.savez(
            path,
            text=features,
            latents=noise,
            cos=position.real.numpy()[None, :, None],
            sin=position.imag.numpy()[None, :, None],
            prefix_hidden=expanded_text,
            prefix_images=prefix_images,
            prefix_image_mask=expanded_mask[None, :, None],
            prefix_bias=np.where(causal, 0, -np.inf).astype(np.float16)[None, None],
            image_mask=vlm_image_mask,
            condition_latents=condition_latents,
            image_shapes=np.array([*image_shapes, (1, h, w)], np.int64),
        )
        results.append(
            {**case, "file": path.name, "sha256": hash_file(path), "prefix_tokens": prefix}
        )
        print(case["name"], features.shape, noise.shape, flush=True)
        timer.mark("output_save_ms")
    write_json(
        output / "cases.json",
        {
            "source": json.loads((source / "evo-source.json").read_text()),
            "sample_sigmas": json.loads((source / "model_index.json").read_text())["sample_sigmas"],
            "scope": "Shared BF16 text encoder, initial noise and source schedule; batch 1",
            "cases": results,
        },
    )
    timer.mark("output_save_ms")
    report("Freeing GPU memory for image generation…")
    if components is not None:
        components.update(encoder=components["encoder_cpu_weights"].restore(), processor=processor)
        if pipeline.vae is not None:
            components["vae"] = pipeline.vae.to("cpu")
    del pipeline, encoder
    gc.collect()
    torch.cuda.empty_cache()
    timer.mark("cleanup_ms", cuda=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    for path in args.dependencies:
        sys.path.insert(0, str(path.resolve()))
    prepare(args.source, json.loads(args.cases.read_text()), args.output)


if __name__ == "__main__":
    main()
