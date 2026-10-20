"""Validate embedding ONNX outputs against independent Transformers reference fixtures."""

import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from evo.pipeline import hash_file, write_json


def make_fixtures(source, output):
    import torch
    from transformers import AutoTokenizer

    from evo.embedding_export import load_text_model

    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    text_model = load_text_model(source)
    texts = [
        "task: search result | query: What causes the northern lights?",
        "title: none | text: Charged particles from the sun cause the northern lights.",
        "title: none | text: Mars is known as the Red Planet.",
        "task: search result | query: Which planet is known as the Red Planet?",
        "task: search result | query: 什么原因导致北极光？",
        "task: code retrieval | query: A Python function that adds even numbers.",
        "title: none | text: def sum_even(xs): return sum(x for x in xs if x % 2 == 0)",
        "task: sentence similarity | query: ",
    ]
    cases = [(f"text-{i}", [t], None) for i, t in enumerate(texts)]
    cases.append(("padded-batch", texts[:4], None))
    long_text = (
        "title: none | text: "
        + (
            "The northern lights arise when charged particles interact with gases "
            "in Earth's atmosphere. Scientists observe the colors, measure solar activity, "
            "and compare observations across seasons. "
        )
        * 180
    )
    cases.extend((f"length-{n}", [long_text], n) for n in (128, 512, 513, 1024))
    report = {"reference": "Official EmbeddingGemma2TextModel FP32, masked mean + L2", "cases": []}
    for name, items, length in cases:
        inputs = tokenizer(
            items, padding=True, truncation=True, max_length=length or 8192, return_tensors="pt"
        )
        if length and inputs["input_ids"].shape[1] != length:
            raise ValueError("Fixture did not reach its requested token length")
        with torch.inference_mode():
            hidden = text_model(**inputs).last_hidden_state.float()
            mask = inputs["attention_mask"][:, :, None].float()
            embeddings = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1)
        np.savez(
            output / f"{name}.npz",
            **{k: v.numpy() for k, v in inputs.items()},
            expected=embeddings.numpy(),
        )
        report["cases"].append(
            {
                "name": name,
                "file": f"{name}.npz",
                "batch_size": len(items),
                "tokens": inputs["input_ids"].shape[1],
                "prompt": tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False),
                "sha256": hash_file(output / f"{name}.npz"),
            }
        )
        print(f"Reference {name}: {tuple(embeddings.shape)}", flush=True)
    write_json(output / "fixtures.json", report)


def native_session(model, profile=None):
    import onnxruntime as ort

    from evo.runtime import register_webgpu

    devices, runtime = register_webgpu()
    options = ort.SessionOptions()
    options.log_severity_level = 3
    provider = {
        "powerPreference": "high-performance",
        "dawnBackendType": "D3D12",
        "validationMode": "basic",
        "enableGraphCapture": "0",
    }
    options.add_provider_for_devices(devices[:1], provider)
    if profile:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile.resolve())
    session = ort.InferenceSession(str(model.resolve()), sess_options=options)
    return (
        session,
        {
            "metadata": devices[0].device.metadata,
            "vendor_id": devices[0].device.vendor_id,
            "device_id": devices[0].device.device_id,
            "runtime_library": runtime,
        },
        provider,
    )


def validate(package, fixtures, output, cpu=False, min_cosine=0.999, max_abs=0.005):
    import onnxruntime as ort

    output.parent.mkdir(parents=True, exist_ok=True)
    model = package / "model.onnx"
    if cpu:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 8
        session = ort.InferenceSession(
            str(model.resolve()), sess_options=options, providers=["CPUExecutionProvider"]
        )
        device, provider = {}, {}
    else:
        session, device, provider = native_session(model, output.with_suffix(""))
    report = {
        "package": str(package.resolve()),
        "model_sha256": hash_file(model),
        "reference": json.loads((fixtures / "fixtures.json").read_text())["reference"],
        "provider": "CPU" if cpu else "Native WebGPU",
        "device": device,
        "provider_options": provider,
        "runtime_library": device.get("runtime_library"),
        "thresholds": {"min_cosine": min_cosine, "max_absolute_error": max_abs},
        "results": [],
    }
    for case in json.loads((fixtures / "fixtures.json").read_text())["cases"]:
        fixture = fixtures / case["file"]
        if hash_file(fixture) != case["sha256"]:
            raise ValueError("Reference fixture changed")
        with np.load(fixture) as data:
            feeds = {k: data[k] for k in ("input_ids", "attention_mask")}
            expected = data["expected"].astype(np.float64)
        actual = session.run(None, feeds)[0].astype(np.float64)
        finite = bool(np.isfinite(actual).all())
        if actual.shape != expected.shape or not finite:
            raise ValueError(
                f"Invalid embedding for {case['name']}: shape={actual.shape}, finite={finite}"
            )
        norms = np.linalg.norm(actual, axis=-1)
        cosine = np.sum(actual * expected, axis=-1) / (norms * np.linalg.norm(expected, axis=-1))
        error = float(np.max(np.abs(actual - expected)))
        row = {
            "name": case["name"],
            "tokens": case["tokens"],
            "shape": list(actual.shape),
            "minimum_cosine": float(cosine.min()),
            "maximum_absolute_error": error,
            "norm_error": float(np.max(np.abs(norms - 1))),
            "fixture_sha256": case["sha256"],
            "passed": bool(
                cosine.min() >= min_cosine
                and error <= max_abs
                and np.max(np.abs(norms - 1)) < 0.001
            ),
        }
        report["results"].append(row)
        print(
            case["name"],
            row["minimum_cosine"],
            row["maximum_absolute_error"],
            row["passed"],
            flush=True,
        )
    if not cpu:
        profile = session.end_profiling()
        events = json.loads(Path(profile).read_text())
        counts = Counter(
            f"{e['args']['provider']}::{e['args']['op_name']}"
            for e in events
            if e.get("cat") == "Node" and e.get("args", {}).get("provider")
        )
        report["profile"] = profile
        report["executed_nodes"] = dict(counts)
        report["native_gpu_observed"] = any(
            k.startswith("WebGpuExecutionProvider::") for k in counts
        )
    report["passed"] = all(r["passed"] for r in report["results"]) and (
        cpu or report["native_gpu_observed"]
    )
    write_json(output, report)
    del session
    gc.collect()
    if not report["passed"]:
        raise ValueError(f"Embedding validation failed: {output}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--transformers-path", type=Path)
    parser.add_argument("--multimodal", action="store_true")
    args = parser.parse_args(argv)
    if args.transformers_path:
        sys.path.insert(0, str(args.transformers_path.resolve()))
    if args.prepare:
        if args.multimodal:
            from evo.embedding_multimodal_validate import prepare

            prepare(args.package, args.fixtures)
        else:
            make_fixtures(args.package, args.fixtures)
    elif args.output:
        config = json.loads((args.package / "embedding_config.json").read_text())
        if config["task"] == "multimodal-embedding":
            from evo.embedding_multimodal_validate import validate as validate_multimodal

            validate_multimodal(args.package, args.fixtures, args.output, args.cpu)
        else:
            validate(args.package, args.fixtures, args.output, args.cpu)
    else:
        parser.error("--output is required for validation")


if __name__ == "__main__":
    main()
