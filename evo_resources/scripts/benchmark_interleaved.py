"""Interleave two resident GenAI models on one native WebGPU binary."""

import argparse
import gc
import importlib.metadata
import json
import platform
import statistics
import time
from pathlib import Path

import onnxruntime_genai as og

from evo.benchmark import artifact_hashes, gpu_observation
from evo.measurement_guard import require_idle_host
from evo.pipeline import hash_file, write_json
from evo.runtime import register_webgpu


def sample(generator, ids):
    generator.rewind_to(0)
    before = time.perf_counter()
    generator.append_tokens(ids)
    generator.generate_next_token()
    generator.is_done()
    first_count = len(generator.get_sequence(0)) - len(ids)
    first = time.perf_counter()
    while not generator.is_done():
        generator.generate_next_token()
    tokens = generator.get_sequence(0).tolist()[len(ids) :]
    end = time.perf_counter()
    return {
        "ttft_seconds": first - before,
        "decode_seconds": end - first,
        "first_step_tokens": first_count,
        "generated_tokens": len(tokens),
        "decode_tokens_per_second": (len(tokens) - first_count) / (end - first),
        "generated_ids": tokens,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--ep-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("benchmarks/qwen35-webgfx.json"))
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--graph-capture", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2 or args.new_tokens < 2:
        parser.error("At least two repetitions and two generated tokens are required")
    args.output.mkdir(parents=True, exist_ok=False)
    require_idle_host()
    devices, runtime = register_webgpu(args.ep_library, genai=True)
    options = {
        "powerPreference": "high-performance",
        "dawnBackendType": "D3D12",
        "validationMode": "basic",
        "enableGraphCapture": "1" if args.graph_capture else "0",
    }
    specs = json.loads(args.cases.read_text())["cases"]
    if any(len(c["input_ids"]) + args.new_tokens > 8192 for c in specs):
        parser.error("Workload exceeds the shared 8192-token context allocation")
    models, configs, reports = [], [], []
    for package in (args.previous, args.candidate):
        artifacts = artifact_hashes(package)
        config = og.Config(str(package.resolve()))
        for key, value in options.items():
            config.set_provider_option("WebGPU", key, value)
        config.overlay(json.dumps({"search": {"max_length": 8192}}))
        configs.append(config)
        models.append(og.Model(config))
        reports.append(
            {
                "schema_version": 1,
                "created": time.time(),
                "package": str(package.resolve()),
                "protocol": "Two resident models; alternating AB/BA per repetition; same runtime",
                "cases_sha256": hash_file(args.cases),
                "model_artifacts": artifacts,
                "model_sha256": artifacts["model.onnx"]["sha256"],
                "genai_config_sha256": artifacts["genai_config.json"]["sha256"],
                "machine": {"name": platform.node(), "platform": platform.platform()},
                "runtime_library": runtime,
                "provider_options": options,
                "provider_option_api": "canonical-WebGPU",
                "search_options": {},
                "versions": {
                    n: importlib.metadata.version(n)
                    for n in ("onnxruntime", "onnxruntime-genai", "onnxruntime-ep-webgpu")
                },
                "devices": [
                    {
                        "vendor_id": d.device.vendor_id,
                        "device_id": d.device.device_id,
                        "metadata": d.device.metadata,
                    }
                    for d in devices
                ],
                "max_length": 8192,
                "new_tokens": args.new_tokens,
                "repeats": args.repeats,
                "warmup": 2,
                "reuse_generator": True,
                "synchronization": "is_done",
                "require_idle_host": True,
                "host_activity": [],
                "measurement_status": "running",
                "results": [],
            }
        )
    names = ["previous", "candidate"]
    for case in specs:
        require_idle_host()
        generators, parameters = [], []
        samples = [[], []]
        for model in models:
            params = og.GeneratorParams(model)
            length = len(case["input_ids"]) + args.new_tokens
            params.set_search_options(max_length=length, min_length=length, do_sample=False)
            parameters.append(params)
            generators.append(og.Generator(model, params))
        for i in range(2 + args.repeats):
            pair = {}
            for index in [0, 1] if i % 2 == 0 else [1, 0]:
                pair[index] = sample(generators[index], case["input_ids"])
                if i >= 2:
                    samples[index].append(pair[index])
            try:
                require_idle_host()
            except RuntimeError:
                for name, report in zip(names, reports, strict=True):
                    report["measurement_status"] = "contaminated"
                    write_json(args.output / f"{name}.json", report)
                raise
            print(
                case["name"],
                i + 1,
                *(round(pair[k]["decode_tokens_per_second"], 2) for k in (0, 1)),
                flush=True,
            )
            gc.collect()
        for name, report, rows in zip(names, reports, samples, strict=True):
            report["results"].append(
                {
                    "name": case["name"],
                    "prompt": case.get("text", case.get("prompt", case["name"])),
                    "prompt_tokens": len(case["input_ids"]),
                    "samples": rows,
                    "median_ttft_seconds": statistics.median(r["ttft_seconds"] for r in rows),
                    "median_decode_tokens_per_second": statistics.median(
                        r["decode_tokens_per_second"] for r in rows
                    ),
                    "deterministic": all(
                        r["generated_ids"] == rows[0]["generated_ids"] for r in rows
                    ),
                }
            )
            write_json(args.output / f"{name}.json", report)
        generators.clear()
        parameters.clear()
        del params, model
        gc.collect()
    for name, report in zip(names, reports, strict=True):
        report["gpu_after"] = gpu_observation()
        report["measurement_status"] = "completed"
        write_json(args.output / f"{name}.json", report)
    models.clear()
    configs.clear()
    del config
    gc.collect()
    print("Both models released", flush=True)


if __name__ == "__main__":
    main()
