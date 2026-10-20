"""Reuse CPU feature/VAE weights while releasing GPU model sessions between phases."""

import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


def emit(kind, **data):
    print("KITE_EVENT " + json.dumps({"type": kind, **data}), flush=True)


def run_phase(request, stage, components=None):
    root = Path(__file__).resolve().parent.parent
    sys.path[:0] = [str(root), *request["dependencies"]]
    os.environ["EVO_WEBGPU_LIBRARY"] = request["library"]
    if request.get("dawn_cache_directory"):
        os.environ["ORT_WEBGPU_DAWN_CACHE_DIR"] = request["dawn_cache_directory"]
    # Uncaptured FP16 is the fast validated path. Keep FP32 for legacy capture.
    os.environ["EVO_IMAGE_FP32_ACCUMULATION"] = (
        "1" if stage == "prefix" or request["capture"] else "0"
    )
    import numpy as np

    from evo.pipeline import hash_file, write_json

    folder = Path(request["directory"])
    source, package = Path(request["source"]), Path(request["package"])
    fixtures = folder / "fixtures"
    started = time.perf_counter()
    metrics = {}

    def progress(message):
        emit("status", stage=stage, message=message)

    def record_profile(filename):
        from backend.image_profile import save_trace

        before = time.perf_counter()
        progress("Saving the GPU profiling trace…")
        metrics["profile"] = save_trace(filename, folder, stage)
        metrics["profiling_ms"] = (time.perf_counter() - before) * 1000

    initializer_cache = None
    if components is not None and stage in ("prefix", "denoise"):
        from evo.initializer_cache import InitializerFileCache

        if "initializer_files" not in components:
            components["initializer_files"] = InitializerFileCache()
        initializer_cache = components["initializer_files"]
    if stage == "prepare":
        from evo.image_prepare import prepare

        prepare(
            source,
            [
                {
                    "name": "image",
                    "prompt": request["prompt"],
                    "seed": request["seed"],
                    "width": request["width"],
                    "height": request["height"],
                    "images": request["images"],
                    "condition_resolution": 512,
                }
            ],
            fixtures,
            components=components,
            timings=metrics,
            progress=progress,
        )
    elif stage == "prefix":
        from evo.image_generate import NativeDenoiser, prefix_feeds, session

        case = json.loads((fixtures / "cases.json").read_text())["cases"][0]
        graph = package / "prefix-short-blocks.onnx"
        with np.load(fixtures / "0.npz") as data:
            progress("Loading the image model and preparing shaders…")
            before = time.perf_counter()
            runner, runtime = session(
                graph,
                case["prefix_tokens"],
                webgpu_backend="D3D12",
                initializer_cache=initializer_cache,
                profile=folder / "prefix-profile" if request.get("profile") else None,
            )
            metrics["model_load_ms"] = (time.perf_counter() - before) * 1000
            feeds = prefix_feeds(data)
            feeds.pop("timestep")
            feeds["time_projection"] = NativeDenoiser.time_projection(0)
            before = time.perf_counter()
            progress("Preparing attention for your image…")
            values = runner.run(None, feeds)
            metrics["inference_ms"] = (time.perf_counter() - before) * 1000
            if any(not np.isfinite(v).all() for v in values):
                raise ValueError("Non-finite prefix; generation stopped.")
            np.savez(
                folder / "prefix.npz",
                **{v.name: a for v, a in zip(runner.get_outputs(), values, strict=True)},
            )
            metrics.update(
                runtime=runtime, prefix_tokens=case["prefix_tokens"], graph_sha256=hash_file(graph)
            )
            if request.get("profile"):
                record_profile(runner.end_profiling())
    elif stage == "denoise":
        import onnxruntime as ort
        import torch

        metrics["host_version"] = ort.__version__

        from evo.image_generate import NativeDenoiser

        # Verify the actual package before running, including external weights.
        config = json.loads((package / "image_config.json").read_text())
        signature = []
        for name, expected in config["artifacts"].items():
            artifact = (package / name).resolve()
            if not artifact.is_relative_to(package.resolve()):
                raise ValueError(f"Unsafe model artifact: {name}")
            stat = artifact.stat()
            signature.append(
                (str(artifact), stat.st_size, stat.st_mtime_ns, stat.st_ino, expected["sha256"])
            )
        verified = components is not None and components.get("verified_artifacts") == signature
        progress("Checking model files…" if not verified else "Reusing verified model files…")
        before = time.perf_counter()
        for filename, _, _, _, expected_hash in [] if verified else signature:
            if hash_file(Path(filename)) != expected_hash:
                raise ValueError(f"Model artifact changed: {filename}")
        if components is not None:
            components["verified_artifacts"] = signature
        metrics.update(
            verification_cached=verified, verification_ms=(time.perf_counter() - before) * 1000
        )
        with np.load(fixtures / "0.npz") as archive:
            data = dict(archive)
        with np.load(folder / "prefix.npz") as archive:
            prefix = dict(archive)
        before = time.perf_counter()
        progress("Loading image-generation weights onto the GPU…")
        engine = NativeDenoiser(
            package,
            data,
            prefix,
            request["capture"],
            webgpu_backend="D3D12",
            graph_variant="short-blocks",
            initializer_cache=initializer_cache,
            profile=folder / "denoise-profile" if request.get("profile") else None,
        )
        metrics["model_load_ms"] = (time.perf_counter() - before) * 1000
        manifest = json.loads((fixtures / "cases.json").read_text())
        sigmas = manifest["sample_sigmas"] + [0.0]
        latents, steps = data["latents"].copy(), []
        emit("image_progress", completed=0, total=len(sigmas) - 1)
        for index, sigma in enumerate(sigmas[:-1]):
            before = time.perf_counter()
            prediction = engine(latents, sigma, index)
            if not np.isfinite(prediction).all():
                raise ValueError(f"Non-finite prediction at step {index + 1}; no image saved.")
            latents = (
                torch.from_numpy(latents + (sigmas[index + 1] - sigma) * prediction)
                .bfloat16()
                .float()
                .numpy()
            )
            steps.append((time.perf_counter() - before) * 1000)
            emit("image_progress", completed=index + 1, total=len(sigmas) - 1, step_ms=steps[-1])
        np.save(folder / "image.npy", latents)
        metrics.update(
            inference_ms=sum(steps),
            step_ms=steps,
            steps=len(steps),
            runtime=engine.runtime,
            capture=request["capture"],
        )
        trace = engine.close()
        if request.get("profile"):
            record_profile(trace)
    elif stage == "decode":
        from evo.image_generate import decode

        decode(source, fixtures, folder, components=components, timings=metrics, progress=progress)
    else:
        raise ValueError("Unknown image phase")
    if stage in ("prepare", "decode"):
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    metrics["total_ms"] = (time.perf_counter() - started) * 1000
    write_json(folder / f"{stage}.json", metrics)
    return metrics


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        components = {}
        for line in sys.stdin:
            try:
                message = json.loads(line)
                request, stage = message["request"], message["stage"]
                cached = (
                    "encoder" in components
                    if stage == "prepare"
                    else "vae" in components
                    if stage == "decode"
                    else False
                )
                log = Path(request["directory"]) / "process.log"
                with log.open("a", encoding="utf-8") as stream, contextlib.redirect_stderr(stream):
                    result = run_phase(request, stage, components)
                emit("result", data={**result, "components_reused": cached})
            except Exception as exc:
                traceback.print_exc()
                emit("error", message=str(exc))
            finally:
                emit("done")
    else:
        run_phase(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")), sys.argv[2])


if __name__ == "__main__":
    main()
