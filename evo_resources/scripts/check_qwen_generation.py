"""Record real greedy responses and check finite logits through native cached decoding."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from evo.benchmark import artifact_hashes
from evo.measurement_guard import require_idle_host
from evo.pipeline import hash_file, write_json
from evo.runtime import register_webgpu


def main():
    import onnxruntime_genai as og

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--fixtures", type=Path, default=Path("artifacts/qwen35-cached-reference"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ep-library", type=Path)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--graph-capture", action="store_true")
    args = parser.parse_args()
    require_idle_host()
    _, runtime = register_webgpu(args.ep_library, genai=True)
    config = og.Config(str(args.package.resolve()))
    provider = {
        "dawnBackendType": "D3D12",
        "validationMode": "basic",
        "enableGraphCapture": "1" if args.graph_capture else "0",
        "powerPreference": "high-performance",
    }
    for key, value in provider.items():
        config.set_provider_option("WebGPU", key, value)
    config.overlay(json.dumps({"search": {"max_length": 8192}}))
    model = og.Model(config)
    tokenizer = og.Tokenizer(model)
    specs = json.loads((args.fixtures / "fixtures.json").read_text())
    report = {
        "package": str(args.package.resolve()),
        "runtime_library": runtime,
        "model_artifacts": artifact_hashes(args.package),
        "scope": "Free greedy generation smoke check, EOS respected; not task accuracy scoring",
        "new_tokens_limit": args.new_tokens,
        "max_length": 8192,
        "results": [],
        "provider_options": provider,
    }
    for case in specs["cases"]:
        path = args.fixtures / case["file"]
        if hash_file(path) != case["sha256"]:
            raise ValueError("Fixture changed")
        with np.load(path) as data:
            ids = data["input_ids"].copy()
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=len(ids) + args.new_tokens, do_sample=False)
        generator = og.Generator(model, params)
        generator.append_tokens(ids)
        checked = 0
        while not generator.is_done():
            logits = np.array(generator.get_logits(), copy=True)
            if not np.isfinite(logits).all():
                raise ValueError(f"Nonfinite logits in {case['name']} after {checked} steps")
            generator.generate_next_token()
            checked += 1
        tokens = generator.get_sequence(0).tolist()[len(ids) :]
        report["results"].append(
            {
                "name": case["name"],
                "prompt": case["prompt"],
                "prompt_tokens": len(ids),
                "generated_ids": tokens,
                "text": tokenizer.decode(tokens),
                "finite_logits_steps": checked,
                # GenAI omits the terminal EOS from get_sequence().
                "termination": "model_done" if len(tokens) < args.new_tokens else "token_limit",
            }
        )
        write_json(args.output, report)
        print(case["name"], "generated", len(tokens), "tokens; logits finite", flush=True)
        del generator, params, logits
        gc.collect()
    del tokenizer, model, config
    gc.collect()
    print("Native model released", flush=True)


if __name__ == "__main__":
    main()
