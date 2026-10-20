"""Profile/replay saved Image Studio fixtures without re-running the text encoder."""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "local/.cache/qwen-image21/deps")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from evo.image_generate import NativeDenoiser  # noqa: E402


def summarize_trace(path, warmup_steps=0):
    events = json.loads(Path(path).read_text())
    groups = defaultdict(lambda: {"count": 0, "total_us": 0})
    providers = Counter()
    examples = {}
    seen_gpu = Counter()
    gpu = defaultdict(lambda: {"count": 0, "total_us": 0})
    gpu_shapes = defaultdict(lambda: {"count": 0, "total_us": 0})
    for event in events:
        args = event.get("args", {})
        category = event.get("cat", "")
        if args.get("provider"):
            providers[args["provider"]] += 1
        if "dur" not in event:
            continue
        name = event["name"]
        if category == "Api" and "cache_key" in args:
            seen_gpu[name] += 1
            if seen_gpu[name] <= warmup_steps:
                continue
            shader = name.split("&")[-1]
            for mapping, key in (
                (gpu, shader),
                (gpu_shapes, shader + " " + args.get("shapes", "")),
            ):
                mapping[key]["count"] += 1
                mapping[key]["total_us"] += event["dur"]
            key = "GPU/" + shader
        else:
            key = f"{category}/{args.get('op_name', name)}"
        groups[key]["count"] += 1
        groups[key]["total_us"] += event["dur"]
        examples.setdefault(category, event)
    return {
        "providers": dict(providers),
        "categories": dict(Counter(e.get("cat") for e in events)),
        "groups": dict(sorted(groups.items(), key=lambda p: -p[1]["total_us"])),
        "example_events": examples,
        "gpu_shader_totals": dict(sorted(gpu.items(), key=lambda p: -p[1]["total_us"])),
        "gpu_shape_totals": dict(sorted(gpu_shapes.items(), key=lambda p: -p[1]["total_us"])[:20]),
        "gpu_timing_available": bool(gpu),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        type=Path,
        required=True,
        help="Run folder containing fixtures/ and prefix.npz",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--library",
        type=Path,
        help="Provider DLL; profiling defaults to Kite's instrumented build.",
    )
    parser.add_argument(
        "--package", type=Path, default=ROOT / "local/artifacts/qwen-image-2.1-turbo-gguf-floatzp"
    )
    parser.add_argument("--variant", default="short-blocks")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--provider-option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1:
        parser.error("warmup must be nonnegative and runs must be positive")
    if args.torch_threads < 1:
        parser.error("torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    try:
        provider_options = dict(option.split("=", 1) for option in args.provider_option)
    except ValueError:
        parser.error("provider-option must be KEY=VALUE")
    if args.library is None:
        variant = "ort-webgpu-qwen-image-d3d12-unrolled"
        args.library = (
            ROOT / "local/artifacts/runtimes" / variant / "onnxruntime_providers_webgpu.dll"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["EVO_WEBGPU_LIBRARY"] = str(args.library.resolve())
    os.environ["EVO_IMAGE_FP32_ACCUMULATION"] = "0"
    with np.load(args.fixtures / "fixtures/0.npz") as archive:
        data = dict(archive)
    with np.load(args.fixtures / "prefix.npz") as archive:
        prefix = dict(archive)
    manifest = json.loads((args.fixtures / "fixtures/cases.json").read_text())
    sigmas = manifest["sample_sigmas"] + [0.0]
    before = time.perf_counter()
    engine = NativeDenoiser(
        args.package,
        data,
        prefix,
        False,
        profile=args.output / "ort" if args.profile else None,
        webgpu_backend="D3D12",
        graph_variant=args.variant,
        provider_options=provider_options,
    )
    result = {
        "model_load_s": time.perf_counter() - before,
        "runtime": engine.runtime,
        "fixtures": str(args.fixtures.resolve()),
        "profile_enabled": args.profile,
        "warmup": args.warmup,
        "runs": [],
        "image_tokens": data["latents"].shape[1],
        "prefix_tokens": data["cos"].shape[1] - data["latents"].shape[1],
        "serialized_execution": os.environ.get("ORT_WEBGPU_EP_FORCE_LEGACY") == "1",
        "torch_threads": torch.get_num_threads(),
        "graph_sha256": hashlib.sha256(
            (args.package / f"denoise-{args.variant}.onnx").read_bytes()
        ).hexdigest(),
    }
    reference = None
    try:
        for repeat in range(args.warmup + args.runs):
            latents = data["latents"].copy()
            states, steps = [], []
            before = time.perf_counter()
            for index, sigma in enumerate(sigmas[:-1]):
                step_start = time.perf_counter()
                prediction = engine(latents, sigma, index)
                if not np.isfinite(prediction).all():
                    raise ValueError(f"Non-finite output: repeat {repeat}, step {index}")
                latents = (
                    torch.from_numpy(latents + (sigmas[index + 1] - sigma) * prediction)
                    .bfloat16()
                    .float()
                    .numpy()
                )
                steps.append(time.perf_counter() - step_start)
                states.append(latents.copy())
            elapsed = time.perf_counter() - before
            states = np.stack(states)
            if reference is None:
                reference = states
                np.save(args.output / "trajectory.npy", states)
                np.save(args.output / "image.npy", latents)
            item = {
                "warmup": repeat < args.warmup,
                "seconds": elapsed,
                "steps_s": steps,
                "trajectory_sha256": hashlib.sha256(states.tobytes()).hexdigest(),
                "exact_repeat": bool(np.array_equal(reference, states)),
                "max_abs_repeat": float(np.max(np.abs(reference - states))),
                "cosine_repeat": float(
                    np.dot(reference.ravel(), states.ravel())
                    / (np.linalg.norm(reference) * np.linalg.norm(states))
                ),
            }
            if not item["exact_repeat"]:
                np.save(args.output / f"trajectory-{repeat}.npy", states)
            result["runs"].append(item)
            print(json.dumps(item), flush=True)
        result["median_s"] = float(
            np.median([r["seconds"] for r in result["runs"] if not r["warmup"]])
        )
        result["exact_repeats"] = all(r["exact_repeat"] for r in result["runs"])
    finally:
        trace = engine.close()
        if trace:
            result["trace"] = trace
            result["profile_summary"] = summarize_trace(trace, args.warmup * (len(sigmas) - 1))
            if not result["profile_summary"]["gpu_timing_available"]:
                print(
                    "Warning: this DLL produced no GPU timestamps; CPU events are not GPU time.",
                    file=sys.stderr,
                )
        (args.output / "results.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps({k: result[k] for k in ("median_s", "exact_repeats", "model_load_s")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
