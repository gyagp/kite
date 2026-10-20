"""Validate cached native Qwen decoding on fixed FP32-reference token trajectories."""

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from evo.pipeline import hash_file, write_json


def prepare(checkpoint, prompts_path, long_cases, output, steps):
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    cases = [
        {
            "name": f"chat-{i}",
            "text": text,
            "input_ids": tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            ),
        }
        for i, text in enumerate(prompts)
    ]
    if long_cases:
        cases += json.loads(long_cases.read_text())["cases"]
    torch.set_num_threads(8)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        checkpoint, dtype=torch.float32, local_files_only=True, attn_implementation="eager"
    ).eval()
    eos = {248044, 248046}
    report = {
        "reference": (
            "Original FP32 Transformers with persistent cache, greedy trajectory; stops at EOS"
        ),
        "checkpoint": str(checkpoint.resolve()),
        "steps_limit": steps,
        "transformers": importlib.metadata.version("transformers"),
        "cases": [],
    }
    for case in cases:
        cache, logits, chosen = None, [], []
        ids = list(case["input_ids"])
        with torch.inference_mode():
            for step in range(steps):
                inputs = torch.tensor([ids if step == 0 else chosen[-1:]], dtype=torch.int64)
                result = model(
                    input_ids=inputs,
                    attention_mask=torch.ones((1, len(ids) + step), dtype=torch.int64),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                row = result.logits[0, -1].float().numpy().copy()
                logits.append(row)
                chosen.append(int(row.argmax()))
                cache = result.past_key_values
                if chosen[-1] in eos:
                    break
        path = output / (case["name"] + ".npz")
        np.savez(
            path,
            input_ids=np.asarray(ids, np.int32),
            chosen=np.asarray(chosen, np.int32),
            logits=np.stack(logits),
        )
        report["cases"].append(
            {
                "name": case["name"],
                "prompt": case.get("text", ""),
                "prompt_tokens": len(ids),
                "steps": len(chosen),
                "file": path.name,
                "sha256": hash_file(path),
                "reference_continuation": tokenizer.decode(chosen),
            }
        )
        write_json(output / "fixtures.json", report)
        print(case["name"], len(ids), len(chosen), flush=True)
        del result, cache
        gc.collect()


def validate(
    package, fixtures, output, library, graph_capture=False, profile=None, max_length=8192
):
    import onnxruntime_genai as og

    from evo.runtime import register_webgpu

    _, runtime = register_webgpu(library, genai=True)
    specs = json.loads((fixtures / "fixtures.json").read_text())
    required_length = max(c["prompt_tokens"] + c["steps"] for c in specs["cases"]) + 8
    if max_length < required_length:
        raise ValueError("Context allocation is too small for the reference fixtures")
    config = og.Config(str(package.resolve()))
    provider = {
        "dawnBackendType": "D3D12",
        "validationMode": "basic",
        "enableGraphCapture": "1" if graph_capture else "0",
        "powerPreference": "high-performance",
    }
    for key, value in provider.items():
        config.set_provider_option("WebGPU", key, value)
    config.overlay(json.dumps({"search": {"max_length": max_length}}))
    if profile:
        profile.parent.mkdir(parents=True, exist_ok=True)
        config.overlay(
            json.dumps(
                {
                    "model": {
                        "decoder": {
                            "session_options": {
                                "enable_profiling": str(profile.resolve()),
                            }
                        }
                    }
                }
            )
        )
    model = og.Model(config)
    report = {
        "package": str(package.resolve()),
        "model_sha256": hash_file(package / "model.onnx"),
        "runtime_library": runtime,
        "fixtures_sha256": hash_file(fixtures / "fixtures.json"),
        "scope": (
            "Persistent native state; reference-chosen tokens forced through "
            "the sampler, no prefix recomputation"
        ),
        "provider_options": provider,
        "max_length": max_length,
        "results": [],
    }
    for case in specs["cases"]:
        path = fixtures / case["file"]
        if hash_file(path) != case["sha256"]:
            raise ValueError("Fixture changed")
        with np.load(path) as data:
            ids, chosen, expected = data["input_ids"], data["chosen"], data["logits"]
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=len(ids) + len(chosen) + 2, do_sample=False)
        generator = og.Generator(model, params)
        generator.append_tokens(ids)
        rows = []
        for step, (ref, token) in enumerate(zip(expected, chosen, strict=True)):
            shaped = np.array(generator.get_logits(), copy=True)
            actual = shaped.reshape(-1, shaped.shape[-1])[-1].astype(np.float64)
            ref = ref.astype(np.float64)
            if actual.shape != ref.shape or not np.isfinite(actual).all():
                raise ValueError(f"Invalid logits at {case['name']} step {step}")
            logp, logq = ref - ref.max(), actual - actual.max()
            logp -= np.log(np.exp(logp).sum())
            logq -= np.log(np.exp(logq).sum())
            rows.append(
                {
                    "step": step,
                    "top1_match": bool(actual.argmax() == ref.argmax()),
                    "kl": float(np.sum(np.exp(logp) * (logp - logq))),
                    "rmse": float(np.sqrt(np.mean((actual - ref) ** 2))),
                }
            )
            forced = np.full(shaped.shape, -1e10, dtype=np.float32)
            forced[..., int(token)] = 1e10
            generator.set_logits(forced)
            generator.generate_next_token()
            observed = int(generator.get_next_tokens()[0])
            if observed != token:
                raise ValueError("Reference token was not accepted by the sampler")
        report["results"].append({"name": case["name"], "prompt_tokens": len(ids), "rows": rows})
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
        print(
            case["name"],
            sum(r["top1_match"] for r in rows),
            len(rows),
            max(r["kl"] for r in rows),
            flush=True,
        )
        del generator, params
        gc.collect()
    # NumPy views of native logits must die before the model/session teardown.
    del shaped, actual, forced
    del model, config
    gc.collect()
    print("Native model released", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument(
        "--prompts", type=Path, default=Path("benchmarks/qwen35-quality-prompts.json")
    )
    parser.add_argument("--long-cases", type=Path)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ep-library", type=Path)
    parser.add_argument("--graph-capture", action="store_true")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--profile", type=Path, help="ORT profile prefix; excludes performance use")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.model, args.prompts, args.long_cases, args.fixtures, args.steps)
    elif args.output:
        validate(
            args.model,
            args.fixtures,
            args.output,
            args.ep_library,
            args.graph_capture,
            args.profile,
            args.max_length,
        )
    else:
        parser.error("--output is required for validation")
