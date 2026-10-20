"""Summarize matched image benchmarks and their distance from a BF16 reference."""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

from evo.paths import asset_path
from evo.pipeline import write_json


def report(onnx_dir, gguf_dir, reference_dir, prefix_dir, output):
    import lpips
    import torch
    from PIL import Image, ImageDraw

    torch.set_num_threads(8)
    directories = {"reference": Path(reference_dir), "gguf": Path(gguf_dir), "onnx": Path(onnx_dir)}
    results = {
        name: json.loads((path / "results.json").read_text()) for name, path in directories.items()
    }
    if any(results[name].get("measurement_status") != "completed" for name in ("onnx", "gguf")):
        raise ValueError("Every input benchmark must be complete")
    # Older fidelity-only reference reports predate the benchmark lifecycle field.
    # Their complete case identities and image files are still checked below.
    if results["reference"].get("measurement_status", "completed") != "completed":
        raise ValueError("The fidelity reference is incomplete")
    control = results["gguf"]
    fields = ("name", "prompt", "seed", "width", "height", "sha256")
    expected = [[case[key] for key in fields] for case in control["cases"]]
    for name, result in results.items():
        if [[case[key] for key in fields] for case in result["cases"]] != expected:
            raise ValueError(f"Mismatched inputs: {name}")
        if any(result[key] != control[key] for key in ("source", "sample_sigmas")):
            raise ValueError(f"Mismatched source or schedule: {name}")
    if any(results["onnx"][key] != control[key] for key in ("machine", "gpu")):
        raise ValueError("Performance comparison requires the same machine and GPU")
    if not results["onnx"]["graph_capture"]:
        raise ValueError("ONNX benchmark must enable graph capture")
    prefix = json.loads((Path(prefix_dir) / "prefixes.json").read_text())
    if asset_path(prefix["package"]).resolve() != asset_path(results["onnx"]["model"]).resolve():
        raise ValueError("Prefix timings belong to a different model")
    prefix_times = {r["name"]: r["seconds"] for r in prefix["cases"]}
    loss = lpips.LPIPS(net="alex").eval()
    canvas = Image.new("RGB", (3 * 384, len(expected) * 420), "white")
    draw = ImageDraw.Draw(canvas)
    rows = []
    for row, case in enumerate(control["cases"]):
        images, alphas = {}, {}
        for col, (name, path) in enumerate(directories.items()):
            rgba = Image.open(path / f"{case['name']}.png").convert("RGBA")
            rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
            images[name] = (
                torch.from_numpy(np.array(rgb).transpose(2, 0, 1).copy()).float()[None] / 127.5 - 1
            )
            alphas[name] = np.array(rgba)[..., 3].astype(np.float32) / 255
            canvas.paste(rgb.resize((384, 384)), (col * 384, row * 420 + 30))
            draw.text((col * 384 + 8, row * 420 + 8), f"{name} / {case['name']}", fill="black")
        metrics = {}
        for name in ("gguf", "onnx"):
            entry = results[name]["cases"][row]
            samples = entry["denoise_seconds"]
            seconds = statistics.median(samples) + (
                prefix_times[case["name"]] if name == "onnx" else 0
            )
            with torch.no_grad():
                distance = float(loss(images[name], images["reference"]))
            metrics[name] = {
                "denoise_samples_seconds": samples,
                "prefix_seconds": prefix_times[case["name"]] if name == "onnx" else 0,
                "total_seconds": seconds,
                "lpips_alexnet_vs_bf16": distance,
                "alpha_mae_vs_bf16": float(np.abs(alphas[name] - alphas["reference"]).mean()),
            }
        rows.append(
            {
                "name": case["name"],
                "prompt": case["prompt"],
                **metrics,
                "onnx_latency_change_percent": (
                    metrics["onnx"]["total_seconds"] / metrics["gguf"]["total_seconds"] - 1
                )
                * 100,
            }
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    canvas.save(output / "comparison.jpg", quality=94)
    summary = {
        "cases": rows,
        "mean_seconds": {
            name: statistics.mean(row[name]["total_seconds"] for row in rows)
            for name in ("gguf", "onnx")
        },
        "mean_lpips_vs_bf16": {
            name: statistics.mean(row[name]["lpips_alexnet_vs_bf16"] for row in rows)
            for name in ("gguf", "onnx")
        },
        "evidence": {
            **{name: str(path.resolve()) for name, path in directories.items()},
            "prefix": str(Path(prefix_dir).resolve()),
        },
        "machine": control["machine"],
        "gpu": control["gpu"],
        "notes": [
            "Denoiser plus common Euler updates and ONNX prefix preparation; "
            "excludes text encoding, VAE, loading and compilation.",
            "ONNX FP16 native WebGPU versus Diffusers GGUF CUDA BF16; "
            "not a format-only comparison.",
            "LPIPS measures distance to the chosen BF16 images. "
            "A small prompt set does not establish broad image quality.",
            "Graph capture enabled; inspect separate replay/equivalence evidence "
            "before claiming correctness-qualified capture.",
        ],
    }
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("onnx", "gguf", "reference", "prefix", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    for path in args.dependencies:
        sys.path.insert(0, str(path.resolve()))
    os.environ.setdefault("TORCH_HOME", str(Path(".cache/torch").resolve()))
    print(
        json.dumps(report(args.onnx, args.gguf, args.reference, args.prefix, args.output), indent=2)
    )


if __name__ == "__main__":
    main()
