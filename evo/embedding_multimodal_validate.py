"""Independent multimodal fixtures and end-to-end ONNX parity checks."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from evo.pipeline import hash_file, write_json


def fixture_inputs(processor):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (144, 96), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 65, 80), fill="red")
    draw.ellipse((78, 20, 138, 80), fill="blue")
    portrait = image.rotate(90, expand=True)
    tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32)
    chirp = (
        0.15
        * np.sin(
            2 * np.pi * (120 * np.arange(32160) / 16000 + 600 * (np.arange(32160) / 16000) ** 2)
        )
    ).astype(np.float32)
    frames = np.stack(
        [np.asarray(image), np.asarray(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))]
    )
    cases = [
        ("text", {"text": "task: search result | query: a red rectangle and a blue circle"}),
        ("image-70", {"images": image, "images_kwargs": {"max_soft_tokens": 70}}),
        ("image-280", {"images": image}),
        ("image-portrait", {"images": portrait}),
        (
            "image-padded-batch",
            {"images": [[image], [portrait]], "text": ["<|image|>", "a picture: <|image|>"]},
        ),
        (
            "text-two-images",
            {
                "text": "Compare <|image|> with <|image|>",
                "images": [[image, portrait]],
                "images_kwargs": {"max_soft_tokens": 70},
            },
        ),
        ("audio-1s", {"audio": tone, "sampling_rate": 16000}),
        ("audio-2s", {"audio": chirp, "sampling_rate": 16000}),
        ("audio-padded-batch", {"audio": [tone, chirp], "sampling_rate": 16000}),
        ("video-two-frames", {"videos": frames, "videos_kwargs": {"do_sample_frames": False}}),
        (
            "text-image-audio",
            {
                "text": "A picture <|image|> and its sound <|audio|>",
                "images": image,
                "audio": tone,
                "sampling_rate": 16000,
                "images_kwargs": {"max_soft_tokens": 70},
            },
        ),
        (
            "all-modalities",
            {
                "text": "Picture <|image|> clip <|video|> sound <|audio|>",
                "images": image,
                "videos": frames,
                "audio": tone,
                "sampling_rate": 16000,
                "images_kwargs": {"max_soft_tokens": 70},
                "videos_kwargs": {"do_sample_frames": False, "max_soft_tokens": 70},
            },
        ),
    ]
    for name, args in cases:
        yield name, processor(**args, return_tensors="pt")


def prepare(source, output):
    import torch
    from transformers import AutoProcessor

    from evo.embedding_multimodal import (
        audio_block_mask,
        exportable_audio_context,
        load_model,
        vision_pool_indices,
        wrappers,
    )

    previous = {}
    if output.exists():
        saved = json.loads((output / "fixtures.json").read_text())
        if saved.get("complete") or saved["source"] != json.loads(
            (source / "evo-source.json").read_text()
        ):
            raise ValueError("Choose a new fixture directory")
        previous = {case["name"]: case for case in saved["cases"]}
    output.mkdir(parents=True, exist_ok=True)
    model = load_model(source)
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    backbone, vision, audio = wrappers(model)
    report = {
        "task": "multimodal-embedding",
        "reference": "Official FP32 EmbeddingGemma2Model + masked mean pooling + L2",
        "source": json.loads((source / "evo-source.json").read_text()),
        "cases": [],
    }
    with torch.inference_mode():
        for name, inputs in fixture_inputs(processor):
            if name in previous:
                case = previous[name]
                path = output / case["file"]
                if hash_file(path) != case["sha256"]:
                    raise ValueError("Existing reference fixture changed")
                with np.load(path) as saved:
                    if not all(
                        np.array_equal(value.numpy(), saved[key]) for key, value in inputs.items()
                    ):
                        raise ValueError("Existing reference inputs differ")
                report["cases"].append(case)
                print(f"Reusing verified reference {name}", flush=True)
                continue
            print(f"Reference {name}: {tuple(inputs['input_ids'].shape)}", flush=True)
            reference = model(**inputs).last_hidden_state.float()
            mask = inputs["attention_mask"][:, :, None].float()
            reference = torch.nn.functional.normalize(
                (reference * mask).sum(1) / mask.sum(1), dim=-1
            )
            # Independent end-to-end forward above, then verify authoring wrappers.
            ids = inputs["input_ids"]
            media = torch.zeros(*ids.shape, 512)
            for kind, pixel_key, pos_key in (
                ("image", "pixel_values", "image_position_ids"),
                ("video", "pixel_values_videos", "video_position_ids"),
            ):
                if pixel_key in inputs:
                    indices, valid = vision_pool_indices(inputs[pos_key].numpy())
                    values = vision(inputs[pixel_key], inputs[pos_key], torch.from_numpy(indices))
                    media[ids == getattr(model.config, f"{kind}_token_id")] = values[
                        torch.from_numpy(valid)
                    ]
            if "input_features" in inputs:
                blocked = torch.from_numpy(audio_block_mask(inputs["input_features_mask"].numpy()))
                features, valid = audio(
                    inputs["input_features"], inputs["input_features_mask"], blocked
                )
                media[ids == model.config.audio_token_id] = features[valid.bool()]
            actual = backbone(ids, inputs["attention_mask"], media)
            torch.testing.assert_close(actual, reference, rtol=5e-4, atol=2e-5)
            np.savez(
                output / f"{name}.npz",
                **{k: v.numpy() for k, v in inputs.items()},
                expected=reference.numpy(),
            )
            report["cases"].append(
                {
                    "name": name,
                    "file": f"{name}.npz",
                    "tokens": ids.shape[1],
                    "batch_size": ids.shape[0],
                    "sha256": hash_file(output / f"{name}.npz"),
                    "wrapper_max_absolute_error": float((actual - reference).abs().max()),
                }
            )
            write_json(output / "fixtures.json", report)
        # Check the export-specific unfold rewrite against the untouched implementation.
        original = audio(inputs["input_features"], inputs["input_features_mask"], blocked)[0]
        exportable_audio_context(model)
        rewritten = audio(inputs["input_features"], inputs["input_features_mask"], blocked)[0]
        torch.testing.assert_close(original, rewritten, rtol=0, atol=0)
    report["complete"] = True
    report["audio_gather_rewrite_exact"] = True
    write_json(output / "fixtures.json", report)


def validate(package, fixtures, output, cpu=False, min_cosine=0.999, max_abs=0.005):
    from evo.embedding_multimodal import MultimodalSession

    output.parent.mkdir(parents=True, exist_ok=True)
    fixture_report = json.loads((fixtures / "fixtures.json").read_text())
    if not fixture_report.get("complete"):
        raise ValueError("Reference fixture preparation is incomplete")
    runner = MultimodalSession(
        package, cpu=cpu, profile=output.with_suffix("") if not cpu else None
    )
    report = {
        "task": "multimodal-embedding",
        "package": str(package.resolve()),
        "reference": fixture_report["reference"],
        "provider": "CPU" if cpu else "Native WebGPU",
        "thresholds": {"min_cosine": min_cosine, "max_absolute_error": max_abs},
        "results": [],
        "artifacts": {
            p.name: {"sha256": hash_file(p), "bytes": p.stat().st_size}
            for p in package.glob("*.onnx*")
        },
    }
    for case in fixture_report["cases"]:
        path = fixtures / case["file"]
        if hash_file(path) != case["sha256"]:
            raise ValueError("Reference fixture changed")
        with np.load(path) as data:
            inputs = {k: data[k] for k in data.files if k != "expected"}
            expected = data["expected"].astype(np.float64)
        actual = runner.run(inputs).astype(np.float64)
        if actual.shape != expected.shape:
            raise ValueError("Output shape differs from reference")
        norms = np.linalg.norm(actual, axis=-1)
        cos = (actual * expected).sum(-1) / (norms * np.linalg.norm(expected, axis=-1))
        error = float(np.abs(actual - expected).max())
        row = {
            "name": case["name"],
            "shape": list(actual.shape),
            "tokens": case["tokens"],
            "minimum_cosine": float(cos.min()),
            "maximum_absolute_error": error,
            "norm_error": float(np.abs(norms - 1).max()),
            "fixture_sha256": case["sha256"],
            "passed": bool(
                cos.min() >= min_cosine and error <= max_abs and np.abs(norms - 1).max() < 0.001
            ),
        }
        report["results"].append(row)
        print(
            f"{case['name']}: cosine={cos.min():.9f}, max_error={error:.7f}, pass={row['passed']}",
            flush=True,
        )
        write_json(output, report)
    report["devices"], report["provider_options"] = runner.devices, runner.options
    if not cpu:
        report["profiles"], report["executed_nodes"] = {}, {}
        for name, session in runner.sessions.items():
            path = session.end_profiling()
            report["profiles"][name] = path
            events = json.loads(Path(path).read_text())
            counts = Counter(
                f"{e['args']['provider']}::{e['args']['op_name']}"
                for e in events
                if e.get("cat") == "Node" and e.get("args", {}).get("provider")
            )
            report["executed_nodes"][name] = dict(counts)
        report["native_gpu_observed"] = all(
            any(k.startswith("WebGpuExecutionProvider::") for k in counts)
            for counts in report["executed_nodes"].values()
        )
    report["passed"] = all(x["passed"] for x in report["results"]) and (
        cpu or report["native_gpu_observed"]
    )
    write_json(output, report)
    if not report["passed"]:
        raise ValueError(f"Multimodal validation failed: {output}")
    return report
