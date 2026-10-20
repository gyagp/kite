"""Measure synchronized native WebGPU embedding latency on fixed token fixtures."""

import argparse
import importlib.metadata
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from evo.embedding_validate import native_session
from evo.pipeline import hash_file, write_json


def benchmark(package, fixtures, output, repeats=5, warmup=2):
    if repeats < 1 or warmup < 1:
        raise ValueError("Require positive measured repetitions and warmup")
    data = json.loads((fixtures / "fixtures.json").read_text())
    cases = [r for r in data["cases"] if r["name"] in {"length-128", "length-512", "length-1024"}]
    if len(cases) != 3:
        raise ValueError("Require the 128, 512, and 1024 token fixtures")
    # Hash before loading/timing and record external weights, not just the protobuf.
    artifacts = {
        p.name: {"bytes": p.stat().st_size, "sha256": hash_file(p)}
        for p in package.iterdir()
        if p.name == "model.onnx"
        or p.name.startswith("model.onnx.")
        and not p.name.endswith("json")
    }
    started = time.perf_counter()
    session, device, options = native_session(package / "model.onnx")
    load_seconds = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "task": "embedding",
        "package": str(package.resolve()),
        "model_sha256": artifacts["model.onnx"]["sha256"],
        "model_artifacts": artifacts,
        "cases_sha256": hash_file(fixtures / "fixtures.json"),
        "machine": {"name": platform.node(), "platform": platform.platform()},
        "devices": [device],
        "provider_options": options,
        "runtime_library": device.get("runtime_library"),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("onnxruntime", "onnxruntime-ep-webgpu")
        },
        "warmup": warmup,
        "repeats": repeats,
        "load_seconds": load_seconds,
        "timing": (
            "session.run wall clock, includes CPU input upload and embedding readback; "
            "excludes tokenization, session creation and warmup; profiling disabled"
        ),
        "results": [],
    }
    for case in cases:
        path = fixtures / case["file"]
        if hash_file(path) != case["sha256"]:
            raise ValueError("Input fixture changed")
        with np.load(path) as data:
            feeds = {k: data[k] for k in ("input_ids", "attention_mask")}
        tokens = int(feeds["attention_mask"].sum())
        if tokens != case["tokens"] or feeds["input_ids"].shape != (1, tokens):
            raise ValueError("Token count does not match requested workload")
        samples = []
        for i in range(warmup + repeats):
            start = time.perf_counter()
            embedding = session.run(None, feeds)[0]
            seconds = time.perf_counter() - start
            if not np.isfinite(embedding).all() or seconds <= 0:
                raise ValueError("Invalid embedding measurement")
            if i >= warmup:
                samples.append({"latency_seconds": seconds})
        seconds = statistics.median(s["latency_seconds"] for s in samples)
        report["results"].append(
            {
                "name": case["name"],
                "prompt": case["prompt"],
                "prompt_tokens": tokens,
                "fixture_sha256": case["sha256"],
                "batch_size": 1,
                "median_latency_ms": seconds * 1000,
                "embeddings_per_second": 1 / seconds,
                "input_tokens_per_second": tokens / seconds,
                "samples": samples,
            }
        )
        print(
            f"{tokens} tokens: {seconds * 1000:.2f} ms, {1 / seconds:.1f} embeddings/s", flush=True
        )
        write_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args(argv)
    benchmark(args.package, args.fixtures, args.output, args.repeats, args.warmup)


if __name__ == "__main__":
    main()
