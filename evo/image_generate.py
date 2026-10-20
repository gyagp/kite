"""Run matched Qwen-Image 2.1 denoisers and decode their saved latents."""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from evo.paths import asset_path
from evo.pipeline import hash_file, write_json


def session(
    model,
    prefix,
    capture=False,
    profile=None,
    image_tokens=4096,
    webgpu_backend="D3D12",
    provider_options=None,
    initializer_cache=None,
):
    for backend in (
        webgpu_backend,
        (provider_options or {}).get("dawnBackendType", "D3D12"),
        (provider_options or {}).get("ep.webgpuexecutionprovider.dawnBackendType", "D3D12"),
    ):
        if backend != "D3D12":
            raise ValueError("Kite uses ORT WebGPU with D3D12.")
    import onnxruntime as ort

    from evo.runtime import register_webgpu

    devices, runtime = register_webgpu()
    runtime = {**runtime, "device": devices[0].device.metadata}
    options = ort.SessionOptions()
    options.log_severity_level = int(os.environ.get("EVO_IMAGE_LOG_LEVEL", "3"))
    options.add_free_dimension_override_by_name("batch", 1)
    options.add_free_dimension_override_by_name("prefix_tokens", prefix)
    options.add_free_dimension_override_by_name("image_tokens", image_tokens)
    provider = {
        "powerPreference": "high-performance",
        "dawnBackendType": webgpu_backend,
        "validationMode": "basic",
        "enableGraphCapture": "1" if capture else "0",
    }
    if os.environ.get("EVO_IMAGE_FP32_ACCUMULATION") == "1":
        provider["enableMatmulFp32Accumulation"] = "1"
    provider.update(provider_options or {})
    runtime["provider_options"] = dict(provider)
    options.add_provider_for_devices(devices[:1], provider)
    if profile:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile)
    mapping_owner = initializer_cache.bind(model, options) if initializer_cache else None
    result = ort.InferenceSession(str(model), options)
    if mapping_owner:
        result._kite_initializer_mappings = mapping_owner
        runtime["initializer_mappings"] = mapping_owner.metadata
    result.disable_fallback()
    return result, runtime


def prefix_feeds(data):
    if "prefix_hidden" in data:
        p = data["prefix_hidden"].shape[1]
        return {
            "hidden": data["prefix_hidden"].astype(np.float16),
            "timestep": np.zeros(1, np.float16),
            "cos": data["cos"][:, :p],
            "sin": data["sin"][:, :p],
            "image_latents": data["prefix_images"].astype(np.float16),
            "image_mask": data["prefix_image_mask"],
            "attention_bias": data["prefix_bias"],
        }
    p = data["text"].shape[1]
    return {
        "hidden": data["text"].astype(np.float16),
        "timestep": np.zeros(1, np.float16),
        "cos": data["cos"][:, :p],
        "sin": data["sin"][:, :p],
        "image_latents": np.zeros((1, p, 64), np.float16),
        "image_mask": np.zeros((1, p, 1), bool),
        "attention_bias": np.where(np.tril(np.ones((p, p), bool)), 0, -np.inf).astype(np.float16)[
            None, None
        ],
    }


def prepare_prefixes(package, fixtures, output, webgpu_backend="D3D12", graph_variant="capture"):
    package, fixtures, output = (
        Path(package).resolve(),
        Path(fixtures).resolve(),
        Path(output).resolve(),
    )
    output.mkdir(parents=True, exist_ok=True)
    graph = package / (
        f"prefix-{graph_variant}.onnx" if graph_variant != "capture" else "prefix.onnx"
    )
    records = []
    for case in json.loads((fixtures / "cases.json").read_text())["cases"]:
        with np.load(fixtures / case["file"]) as data:
            s, runtime = session(
                graph,
                case["prefix_tokens"],
                webgpu_backend=webgpu_backend,
            )
            from evo.measurement_guard import require_idle_host

            feeds_in = prefix_feeds(data)
            if graph_variant != "capture":
                feeds_in.pop("timestep")
                feeds_in["time_projection"] = NativeDenoiser.time_projection(0)
            s.run(None, feeds_in)
            samples = []
            for _ in range(3):
                require_idle_host()
                start = time.perf_counter()
                results = s.run(None, feeds_in)
                samples.append(time.perf_counter() - start)
                require_idle_host()
            elapsed = float(np.median(samples))
            if any(not np.isfinite(value).all() for value in results):
                raise ValueError(f"Non-finite prefix cache for {case['name']}")
            feeds = {v.name: a for v, a in zip(s.get_outputs(), results, strict=True)}
            dest = output / case["file"]
            np.savez(dest, **feeds)
            records.append(
                {
                    "name": case["name"],
                    "seconds": elapsed,
                    "samples": samples,
                    "file": dest.name,
                    "sha256": hash_file(dest),
                }
            )
            del s, results, feeds
            gc.collect()
            print("prefix", case["name"], elapsed, flush=True)
    write_json(
        output / "prefixes.json",
        {
            "package": str(package),
            "cases": records,
            "runtime": runtime,
            "graph_variant": graph_variant,
            "graph_sha256": hash_file(graph),
            "fixtures_sha256": hash_file(fixtures / "cases.json"),
        },
    )


class NativeDenoiser:
    def __init__(
        self,
        package,
        data,
        prefix,
        capture,
        profile=None,
        webgpu_backend="D3D12",
        graph_variant="capture",
        provider_options=None,
        initializer_cache=None,
    ):
        graph = Path(package) / f"denoise-{graph_variant}.onnx"
        if graph_variant != "capture" and not graph.is_file():
            raise ValueError(f"Requested graph does not exist: {graph}")
        self.host_time = graph.exists()
        prefix_tokens = data["cos"].shape[1] - data["latents"].shape[1]
        self.session, self.runtime = session(
            graph if self.host_time else Path(package) / "denoise.onnx",
            prefix_tokens,
            capture,
            profile,
            data["latents"].shape[1],
            webgpu_backend,
            provider_options,
            initializer_cache,
        )
        self.binding = self.session.io_binding()
        p = prefix_tokens
        feeds = {
            "hidden": data["latents"].astype(np.float16),
            "timestep": np.zeros(1, np.float16),
            "cos": data["cos"][:, p:],
            "sin": data["sin"][:, p:],
            **prefix,
        }
        if self.host_time:
            feeds.pop("timestep")
            feeds["time_projection"] = self.time_projection(0)
        self.inputs = {}
        for name, array in feeds.items():
            value = self.session.create_ortvalue_from_shape_and_type(
                array.shape, array.dtype, "webgpu"
            )
            value.update_inplace(np.ascontiguousarray(array))
            self.inputs[name] = value
            self.binding.bind_ortvalue_input(name, value)
        self.output = self.session.create_ortvalue_from_shape_and_type(
            data["latents"].shape, np.float16, "webgpu"
        )
        self.binding.bind_ortvalue_output("noise_pred", self.output)

    def __call__(self, latents, timestep, index):
        self.inputs["hidden"].update_inplace(np.ascontiguousarray(latents, dtype=np.float16))
        if self.host_time:
            self.inputs["time_projection"].update_inplace(self.time_projection(timestep))
        else:
            self.inputs["timestep"].update_inplace(np.array([timestep], dtype=np.float16))
        self.session.run_with_iobinding(self.binding)
        return self.output.numpy().astype(np.float32)

    @staticmethod
    def time_projection(timestep):
        import torch

        # Diffusers keeps this non-persistent buffer in FP32 even for BF16 weights.
        freqs = torch.exp(-np.log(10000) * torch.arange(128, dtype=torch.float32) / 128)
        argument = torch.tensor([timestep]).half().float()[:, None] * 1000 * freqs[None]
        return torch.cat([argument.cos(), argument.sin()], -1).half().numpy()

    def close(self):
        self.session.release_captured_graph(0)
        profile = self.session.end_profiling()
        self.binding.clear_binding_inputs()
        self.binding.clear_binding_outputs()
        self.inputs.clear()
        self.output = None
        self.binding = None
        self.session = None
        gc.collect()
        return profile


class TorchDenoiser:
    def __init__(self, source, data, gguf=None):
        import torch
        from diffusers import GGUFQuantizationConfig, QwenImage21Transformer2DModel

        self.torch = torch
        if gguf:
            quantization = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
            quantization.modules_to_not_convert = ["text_norm", "norm_q", "norm_k"]
            self.model = QwenImage21Transformer2DModel.from_single_file(
                str(gguf),
                config=str(source),
                subfolder="transformer",
                local_files_only=True,
                quantization_config=quantization,
                dtype=torch.bfloat16,
            ).to("cuda")
        else:
            self.model = QwenImage21Transformer2DModel.from_pretrained(
                Path(source) / "transformer", dtype=torch.bfloat16, local_files_only=True
            )
            self.model.enable_group_offload(
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                offload_type="block_level",
                num_blocks_per_group=1,
            )
        self.text = torch.from_numpy(data["text"].copy()).to("cuda", torch.bfloat16)
        n = data["latents"].shape[1]
        h, w = [int(x) for x in data["latent_grid"]]
        self.shapes = [[(1, h, w)]]
        self.mask = torch.tensor([[False] * self.text.shape[1] + [True] * (n // 4)], device="cuda")
        self.condition_latents = None
        if "image_shapes" in data:
            self.shapes = [data["image_shapes"].tolist()]
            self.mask = torch.from_numpy(
                np.concatenate([data["image_mask"], np.ones((1, n // 4), bool)], axis=1)
            ).to("cuda")
            if data["condition_latents"].shape[1]:
                self.condition_latents = torch.from_numpy(data["condition_latents"]).to(
                    "cuda", torch.bfloat16
                )
        self.cache = None

    def __call__(self, latents, timestep, index):
        torch = self.torch
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

        if index == 0:
            self.cache = QwenImage21KVCache(len(self.model.transformer_blocks))
        with torch.no_grad():
            hidden = torch.from_numpy(latents).to("cuda", torch.bfloat16)
            if self.condition_latents is not None:
                hidden = torch.cat([self.condition_latents, hidden], dim=1)
            result = self.model(
                hidden_states=hidden,
                encoder_hidden_states=self.text,
                timestep=torch.tensor([timestep], device="cuda", dtype=torch.bfloat16),
                img_shapes=self.shapes,
                img_mask=self.mask,
                kv_cache=self.cache,
                kv_cache_mode="extract" if index == 0 else "cached",
                return_dict=False,
            )[0]
        return result[:, -latents.shape[1] :].float().cpu().numpy()

    def close(self):
        self.cache = None
        self.model = None
        self.text = None
        self.condition_latents = None
        gc.collect()
        self.torch.cuda.empty_cache()


def generate(
    source,
    package,
    fixtures,
    output,
    backend,
    gguf=None,
    prefix_dir=None,
    repeats=3,
    warmup=1,
    capture=True,
    profile=False,
    require_idle=False,
    webgpu_backend="D3D12",
    graph_variant="capture",
):
    import torch

    from evo.measurement_guard import require_idle_host

    if require_idle:
        require_idle_host()
    if repeats < 1 or warmup < 0:
        raise ValueError("Use at least one measured repeat and a nonnegative warmup count")
    torch.set_num_threads(8)
    source, fixtures, output = (
        Path(source).resolve(),
        Path(fixtures).resolve(),
        Path(output).resolve(),
    )
    if output.exists():
        raise ValueError("Choose a new output directory")
    output.mkdir(parents=True)
    manifest = json.loads((fixtures / "cases.json").read_text())
    started = time.time()
    import platform

    import diffusers
    import onnxruntime as ort

    artifacts = {}
    if backend == "onnx":
        model_info = json.loads((Path(package) / "image_config.json").read_text())
        for name, expected in model_info["artifacts"].items():
            path = Path(package) / name
            if hash_file(path) != expected["sha256"]:
                raise ValueError(f"Model artifact changed: {name}")
        artifacts = model_info["artifacts"]
        prefix_info = json.loads((Path(prefix_dir) / "prefixes.json").read_text())
        if asset_path(prefix_info["package"]).resolve() != Path(package).resolve():
            raise ValueError("Prefix cache belongs to a different ONNX package")
        if "graph_variant" in prefix_info:
            prefix_graph = Path(package) / (
                f"prefix-{graph_variant}.onnx" if graph_variant != "capture" else "prefix.onnx"
            )
            if prefix_info["graph_variant"] != graph_variant or (
                hash_file(prefix_graph) != prefix_info["graph_sha256"]
            ):
                raise ValueError("Prefix cache belongs to a different graph variant")
            if hash_file(fixtures / "cases.json") != prefix_info["fixtures_sha256"]:
                raise ValueError("Prefix cache belongs to different prompt fixtures")
        for row in prefix_info["cases"]:
            if hash_file(Path(prefix_dir) / row["file"]) != row["sha256"]:
                raise ValueError("Prefix cache changed")
    elif gguf:
        artifacts[Path(gguf).name] = {"sha256": hash_file(gguf), "bytes": Path(gguf).stat().st_size}
    sigmas = manifest["sample_sigmas"] + [0.0]
    cases = []
    for case in manifest["cases"]:
        file = fixtures / case["file"]
        if hash_file(file) != case["sha256"]:
            raise ValueError("Fixture changed")
        with np.load(file) as archive:
            data = dict(archive)
        data["latent_grid"] = np.array([case["height"] // 16, case["width"] // 16])
        if backend == "onnx":
            with np.load(Path(prefix_dir) / case["file"]) as archive:
                prefix = dict(archive)
            engine = NativeDenoiser(
                package,
                data,
                prefix,
                capture,
                output / case["name"] if profile else None,
                webgpu_backend,
                graph_variant,
            )
        else:
            engine = TorchDenoiser(source, data, gguf if backend == "gguf" else None)
        samples, final, steps, repeat_errors = [], None, [], []
        for repeat in range(warmup + repeats):
            if require_idle:
                require_idle_host()
            latents = data["latents"].copy()
            trajectories = []
            start = time.perf_counter()
            step_times = []
            for index, sigma in enumerate(sigmas[:-1]):
                # Match the official pipeline's BF16 timestep conversion for every backend.
                timestep = (torch.tensor(sigma * 1000, dtype=torch.bfloat16) / 1000).float().item()
                step_start = time.perf_counter()
                prediction = engine(latents, timestep, index)
                if not np.isfinite(prediction).all():
                    hint = (
                        " Try --fp32-accumulation for native WebGPU." if backend == "onnx" else ""
                    )
                    raise ValueError(
                        f"Non-finite denoiser output for {case['name']} at step {index + 1}." + hint
                    )
                # Use one common Euler implementation and BF16 latent rounding for all backends.
                latents = latents + (sigmas[index + 1] - sigma) * prediction
                latents = torch.from_numpy(latents).bfloat16().float().numpy()
                step_times.append(time.perf_counter() - step_start)
                trajectories.append(latents.copy())
            elapsed = time.perf_counter() - start
            if require_idle:
                require_idle_host()
            if repeat >= warmup:
                samples.append(elapsed)
                steps.append(step_times)
            if final is not None:
                error = float(np.max(np.abs(final - latents)))
                repeat_errors.append(error)
                print("repeat_max_error", error, flush=True)
            final = latents.copy()
            print(backend, case["name"], repeat, elapsed, flush=True)
        np.save(output / f"{case['name']}.npy", final)
        np.save(output / f"{case['name']}-steps.npy", np.stack(trajectories))
        record = {
            **case,
            "denoise_seconds": samples,
            "step_seconds": steps,
            "median_denoise_seconds": float(np.median(samples)),
            "prefix_precomputed": backend == "onnx",
            "repeat_max_errors": repeat_errors,
        }
        if backend == "onnx":
            record["runtime"] = engine.runtime
        profile_path = engine.close()
        if profile_path:
            record["profile"] = profile_path
        del engine
        gc.collect()
        cases.append(record)
        write_json(
            output / "results.json",
            {
                "task": "image-generation",
                "created": started,
                "measurement_status": "completed"
                if len(cases) == len(manifest["cases"])
                else "running",
                "backend": backend,
                "graph_capture": capture if backend == "onnx" else False,
                "webgpu_backend": webgpu_backend if backend == "onnx" else None,
                "graph_variant": graph_variant if backend == "onnx" else None,
                "source": manifest["source"],
                "model": str(package or gguf or source),
                "artifacts": artifacts,
                "sample_sigmas": sigmas,
                "warmup": warmup,
                "repeats": repeats,
                "cases": cases,
                "timing_scope": (
                    "Denoiser plus common Euler updates; excludes shared text encoding/VAE "
                    "and ONNX one-time prefix preparation"
                ),
                "torch": torch.__version__,
                "diffusers": diffusers.__version__,
                "onnxruntime": ort.__version__,
                "machine": platform.node(),
                "gpu": record["runtime"]["device"].get("Description")
                if backend == "onnx"
                else torch.cuda.get_device_name(0),
                "require_idle_host": require_idle,
            },
        )


def decode(source, fixtures, directory, components=None, timings=None, progress=None):
    report = progress or (lambda message: None)
    report("Loading the image decoder…")
    from evo.image_timing import ImageTimer

    timer = ImageTimer(timings)
    import torch
    from diffusers import AutoencoderKLQwenImage21
    from diffusers.image_processor import VaeImageProcessor

    timer.mark("initialization_ms")
    torch.set_num_threads(8)
    model = (
        components.get("vae") if components is not None else None
    ) or AutoencoderKLQwenImage21.from_pretrained(
        Path(source) / "vae", dtype=torch.bfloat16, local_files_only=True
    )
    model = model.to("cuda")
    model.enable_tiling()
    processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64)
    timer.mark("model_load_ms", cuda=True)
    for case in json.loads((Path(fixtures) / "cases.json").read_text())["cases"]:
        latents = torch.from_numpy(np.load(Path(directory) / f"{case['name']}.npy")).to(
            "cuda", torch.bfloat16
        )
        h, w = case["height"] // 16, case["width"] // 16
        latents = latents.reshape(1, 1, h, w, 64).permute(0, 4, 1, 2, 3)
        mean = torch.tensor(model.config.latents_mean, device="cuda", dtype=torch.bfloat16).reshape(
            1, 64, 1, 1, 1
        )
        std = torch.tensor(model.config.latents_std, device="cuda", dtype=torch.bfloat16).reshape(
            1, 64, 1, 1, 1
        )
        timer.mark("input_preparation_ms", cuda=True)
        report("Decoding the final pixels…")
        with torch.no_grad():
            image = model.decode(latents * std + mean, return_dict=False)[0][:, :, 0]
        timer.mark("inference_ms", cuda=True)
        report("Saving your image…")
        processor.postprocess(image, output_type="pil")[0].save(
            Path(directory) / f"{case['name']}.png"
        )
        print("decoded", case["name"], flush=True)
        timer.mark("output_save_ms")
    if components is not None:
        components["vae"] = model.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    timer.mark("cleanup_ms", cuda=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prefix", "generate", "decode"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("onnx", "gguf", "reference"), default="onnx")
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--prefix-dir", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--no-capture", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--fp32-accumulation", action="store_true", help="Request FP32 MatMul accumulation"
    )
    parser.add_argument("--require-idle", action="store_true")
    parser.add_argument("--webgpu-backend", choices=("D3D12",), default="D3D12")
    parser.add_argument(
        "--graph-variant",
        choices=("capture", "rms", "short-blocks", "projections"),
        default="capture",
    )
    parser.add_argument("--dependencies", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    if args.fp32_accumulation:
        os.environ["EVO_IMAGE_FP32_ACCUMULATION"] = "1"
    for path in args.dependencies:
        sys.path.insert(0, str(path.resolve()))
    if args.action == "prefix":
        prepare_prefixes(
            args.package, args.fixtures, args.output, args.webgpu_backend, args.graph_variant
        )
    elif args.action == "decode":
        decode(args.source, args.fixtures, args.output)
    else:
        generate(
            args.source,
            args.package,
            args.fixtures,
            args.output,
            args.backend,
            args.gguf,
            args.prefix_dir,
            args.repeats,
            args.warmup,
            not args.no_capture,
            args.profile,
            args.require_idle,
            args.webgpu_backend,
            args.graph_variant,
        )


if __name__ == "__main__":
    main()
