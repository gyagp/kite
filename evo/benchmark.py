"""Native GenAI WebGPU measurements with synchronized token observation."""

import argparse
import gc
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

from evo.pipeline import hash_file, write_json


def artifact_hashes(package):
    from evo.inspect import inspect_model

    report = inspect_model(package / "model.onnx")
    if report["structure"] != "passed":
        raise ValueError(f"Model failed structural inspection: {report['errors']}")
    names = {"model.onnx", "genai_config.json"}
    names.update(t["location"] for t in report["external_data"])
    return {
        name: {"bytes": (package / name).stat().st_size, "sha256": hash_file(package / name)}
        for name in sorted(names)
    }


def gpu_observation():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def benchmark(
    package: Path,
    cases_path: Path,
    output: Path,
    *,
    repeats=3,
    warmup=1,
    new_tokens=32,
    provider_options=None,
    search_options=None,
    max_length=None,
    reuse_generator=False,
    synchronization="get_next_tokens",
    ep_library=None,
    require_idle=False,
) -> dict:
    import onnxruntime_genai as og

    from evo.measurement_guard import competing_workloads, require_idle_host
    from evo.runtime import register_webgpu

    if require_idle:
        require_idle_host()

    if repeats < 1 or warmup < 0 or new_tokens < 2:
        raise ValueError("Require repeats >= 1, warmup >= 0, new_tokens >= 2")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    if not cases or any(not c["input_ids"] for c in cases):
        raise ValueError("Need nonempty tokenized cases")
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = artifact_hashes(package)
    devices, runtime_library = register_webgpu(ep_library, genai=True)
    config = og.Config(str(package.resolve()))
    options = {"powerPreference": "high-performance", **(provider_options or {})}
    for key, value in options.items():
        # GenAI 0.17.1 merges against canonical names before normalizing new entries.
        # Lowercase "webgpu" can append duplicates whose options are never selected.
        config.set_provider_option("WebGPU", key, str(value))
    required_length = max(len(c["input_ids"]) for c in cases) + new_tokens
    requested_length = (search_options or {}).get("max_length")
    if requested_length is not None and max_length is not None and requested_length != max_length:
        raise ValueError("Conflicting maximum lengths in arguments and search options")
    max_length = requested_length or max_length or required_length
    if type(max_length) is not int or max_length < required_length:
        raise ValueError("max_length is too small for the requested inputs and outputs")
    if synchronization not in {"get_next_tokens", "is_done"}:
        raise ValueError("Invalid synchronization mode")
    config.overlay(json.dumps({"search": {"max_length": max_length, **(search_options or {})}}))
    started = time.perf_counter()
    model = og.Model(config)
    load_seconds = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "created": time.time(),
        "package": str(package.resolve()),
        "cases_sha256": hash_file(cases_path),
        "genai_config_sha256": artifacts["genai_config.json"]["sha256"],
        "model_sha256": artifacts["model.onnx"]["sha256"],
        "model_artifacts": artifacts,
        "machine": {"name": platform.node(), "platform": platform.platform()},
        "runtime_library": runtime_library,
        "require_idle_host": require_idle,
        "host_activity": [],
        "measurement_status": "running",
        "gpu_start": gpu_observation(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("onnxruntime", "onnxruntime-genai", "onnxruntime-ep-webgpu")
        },
        "devices": [
            {
                "vendor_id": d.device.vendor_id,
                "device_id": d.device.device_id,
                "metadata": d.device.metadata,
            }
            for d in devices
        ],
        "provider_options": options,
        "provider_option_api": "canonical-WebGPU",
        "search_options": search_options or {},
        "max_length": max_length,
        "new_tokens": new_tokens,
        "repeats": repeats,
        "warmup": warmup,
        "reuse_generator": reuse_generator,
        "synchronization": synchronization,
        "load_seconds": load_seconds,
        "timing": (
            f"wall clock; {synchronization} after each generated step; "
            "final sequence readback; no text decoding"
        ),
        "results": [],
    }
    for case in cases:
        if require_idle:
            require_idle_host()
        samples = []
        generator = params = None
        for index in range(warmup + repeats):
            length = len(case["input_ids"]) + new_tokens
            if generator is None:
                params = og.GeneratorParams(model)
                params.set_search_options(max_length=length, min_length=length, do_sample=False)
                generator = og.Generator(model, params)
            else:
                generator.rewind_to(0)
            before = time.perf_counter()
            generator.append_tokens(case["input_ids"])
            generator.generate_next_token()
            if synchronization == "get_next_tokens":
                generator.get_next_tokens()
            else:
                generator.is_done()
            first_count = len(generator.get_sequence(0)) - len(case["input_ids"])
            first = time.perf_counter()
            while not generator.is_done():
                generator.generate_next_token()
                if synchronization == "get_next_tokens":
                    generator.get_next_tokens()
            tokens = generator.get_sequence(0).tolist()[len(case["input_ids"]) :]
            end = time.perf_counter()
            sample = {
                "ttft_seconds": first - before,
                "decode_seconds": end - first,
                "first_step_tokens": first_count,
                "generated_tokens": len(tokens),
                "decode_tokens_per_second": (len(tokens) - first_count) / (end - first),
                "generated_ids": tokens,
            }
            if not reuse_generator:
                generator = params = None
            gc.collect()
            if index >= warmup:
                samples.append(sample)
            if require_idle:
                activity = competing_workloads()
                if activity:
                    report["host_activity"].append(activity)
                    report["measurement_status"] = "contaminated"
                    write_json(output, report)
                    raise RuntimeError(
                        "Competing work started during measurement; report marked contaminated"
                    )
            print(
                f"{case['name']} run {index + 1}: TTFT {sample['ttft_seconds']:.3f}s, "
                f"decode {sample['decode_tokens_per_second']:.1f} tokens/s",
                flush=True,
            )
        report["results"].append(
            {
                "name": case["name"],
                "prompt": case.get("text", case.get("prompt", case["name"])),
                "prompt_tokens": len(case["input_ids"]),
                "median_ttft_seconds": statistics.median(s["ttft_seconds"] for s in samples),
                "median_decode_tokens_per_second": statistics.median(
                    s["decode_tokens_per_second"] for s in samples
                ),
                "deterministic": all(
                    s["generated_ids"] == samples[0]["generated_ids"] for s in samples
                ),
                "samples": samples,
                "gpu_after": gpu_observation(),
            }
        )
        write_json(output, report)
        generator = params = None
    report["measurement_status"] = "completed"
    write_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--provider-options", default="{}", help="JSON provider option object")
    parser.add_argument("--search-options", default="{}", help="JSON runtime search option object")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--ep-library", type=Path, help="Use a specific native WebGPU library")
    parser.add_argument(
        "--require-idle",
        action="store_true",
        help="Reject timing with competing build/test processes",
    )
    parser.add_argument("--reuse-generator", action="store_true")
    parser.add_argument(
        "--synchronization", choices=["get_next_tokens", "is_done"], default="get_next_tokens"
    )
    args = parser.parse_args(argv)
    benchmark(
        args.package,
        args.cases,
        args.output,
        repeats=args.repeats,
        warmup=args.warmup,
        new_tokens=args.new_tokens,
        provider_options=json.loads(args.provider_options),
        search_options=json.loads(args.search_options),
        max_length=args.max_length,
        reuse_generator=args.reuse_generator,
        synchronization=args.synchronization,
        ep_library=args.ep_library,
        require_idle=args.require_idle,
    )


if __name__ == "__main__":
    main()
