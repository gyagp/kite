"""Compare two native cached trajectories, forcing reference tokens until EOS."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import onnxruntime_genai as og

from evo.benchmark import artifact_hashes
from evo.measurement_guard import competing_workloads, require_idle_host
from evo.pipeline import hash_file, write_json
from evo.runtime import register_webgpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--ep-library", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--reference-capture", action="store_true")
    parser.add_argument("--candidate-capture", action="store_true")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--state-output", action="append", default=[],
                        help="Compare a named managed output at each step for diagnosis")
    parser.add_argument(
        "--allow-busy-host",
        action="store_true",
        help="Numerical check only; record competing work without timing",
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    host_activity = competing_workloads()
    if not args.allow_busy_host:
        require_idle_host()
    _, runtime = register_webgpu(args.ep_library, genai=True)
    cases = json.loads(args.cases.read_text())["cases"]
    configs, models = [], []
    specs = []
    for index, (package, capture) in enumerate(
        [(args.reference, args.reference_capture), (args.candidate, args.candidate_capture)]
    ):
        config = og.Config(str(package.resolve()))
        provider = {
            "dawnBackendType": "D3D12",
            "validationMode": "basic",
            "enableGraphCapture": "1" if capture else "0",
            "powerPreference": "high-performance",
        }
        for key, value in provider.items():
            config.set_provider_option("WebGPU", key, value)
        config.overlay(json.dumps({"search": {"max_length": 8192}}))
        if args.profile:
            args.profile.mkdir(parents=True, exist_ok=True)
            config.overlay(
                json.dumps(
                    {
                        "model": {
                            "decoder": {
                                "session_options": {
                                    "enable_profiling": str(
                                        (args.profile / f"model-{index}").resolve()
                                    )
                                }
                            }
                        }
                    }
                )
            )
        configs.append(config)
        models.append(og.Model(config))
        specs.append(
            {
                "package": str(package.resolve()),
                "provider_options": provider,
                "artifacts": artifact_hashes(package),
            }
        )
    report = {
        "models": specs,
        "runtime_library": runtime,
        "cases_sha256": hash_file(args.cases),
        "scope": "Persistent state, reference greedy tokens forced, stops at EOS; "
        "relative to quantized reference, not independent FP32 task accuracy",
        "steps_limit": args.steps,
        "host_activity": host_activity,
        "performance_measurement": False,
        "results": [],
    }
    for case in cases:
        ids = case["input_ids"]
        if len(ids) + args.steps > 8192:
            raise ValueError("Case exceeds context allocation")
        generators, params = [], []
        for model in models:
            param = og.GeneratorParams(model)
            param.set_search_options(max_length=len(ids) + args.steps + 1, do_sample=False)
            params.append(param)
            generators.append(og.Generator(model, param))
            generators[-1].append_tokens(ids)
        rows, tokens = [], []
        g = shaped = forced = None
        for step in range(args.steps):
            logits = [np.array(g.get_logits(), copy=True) for g in generators]
            ref, actual = [x.reshape(-1, x.shape[-1])[-1].astype(np.float64) for x in logits]
            if not np.isfinite(ref).all() or not np.isfinite(actual).all():
                raise ValueError(f"Nonfinite logits: {case['name']} step {step}")
            logp, logq = ref - ref.max(), actual - actual.max()
            logp -= np.log(np.exp(logp).sum())
            logq -= np.log(np.exp(logq).sum())
            token = int(ref.argmax())
            rows.append(
                {
                    "step": step,
                    "top1_match": bool(token == actual.argmax()),
                    "kl": float(np.sum(np.exp(logp) * (logp - logq))),
                    "rmse": float(np.sqrt(np.mean((ref - actual) ** 2))),
                }
            )
            if args.state_output:
                state_errors = {}
                for name in args.state_output:
                    left = np.array(generators[0].get_output(name), copy=True).astype(np.float32)
                    right = np.array(generators[1].get_output(name), copy=True).astype(np.float32)
                    state_errors[name] = {
                        "max_abs": float(np.max(np.abs(left - right))),
                        "rmse": float(np.sqrt(np.mean((left - right) ** 2))),
                        "sums": [float(left.sum()), float(right.sum())],
                    }
                    if left.size <= 4:
                        state_errors[name]["values"] = [left.tolist(), right.tolist()]
                    if name.endswith(".key") and left.ndim == 4:
                        state_errors[name]["differing_positions"] = np.flatnonzero(
                            np.any(left != right, axis=(0, 1, 3))
                        ).tolist()
                        state_errors[name]["nonzero_tail"] = [
                            np.flatnonzero(np.any(value != 0, axis=(0, 1, 3)))[-8:].tolist()
                            for value in (left, right)
                        ]
                    del left, right
                rows[-1]["state_errors"] = state_errors
                print("step", step, "states", state_errors, flush=True)
            tokens.append(token)
            if token in {248044, 248046}:
                break
            for g, shaped in zip(generators, logits, strict=True):
                forced = np.full(shaped.shape, -1e10, dtype=np.float32)
                forced[..., token] = 1e10
                g.set_logits(forced)
                g.generate_next_token()
                if int(g.get_next_tokens()[0]) != token:
                    raise ValueError("Reference token not accepted")
        report["results"].append(
            {
                "name": case["name"],
                "prompt_tokens": len(ids),
                "rows": rows,
                "reference_tokens": tokens,
            }
        )
        write_json(args.output, report)
        print(
            case["name"],
            "steps",
            len(rows),
            "top1",
            sum(r["top1_match"] for r in rows),
            "max KL",
            max(r["kl"] for r in rows),
            flush=True,
        )
        generators.clear()
        params.clear()
        del g, param, model, logits, shaped, forced
        gc.collect()
    models.clear()
    configs.clear()
    del config
    gc.collect()
    print("Both models released", flush=True)


if __name__ == "__main__":
    main()
