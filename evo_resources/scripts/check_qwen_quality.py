"""Small first-token logit comparison against the original FP32 Transformers model.

This is a regression screen, not a perplexity or full task-quality evaluation.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from evo.pipeline import hash_file, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint", type=Path)
parser.add_argument("--packages", type=Path, nargs="+", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--prompts", type=Path)
parser.add_argument("--continuation-steps", type=int, default=1)
parser.add_argument("--ep-library", type=Path)
args = parser.parse_args()
if args.continuation_steps < 1:
    parser.error("--continuation-steps must be positive")
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
prompts = [
    "What is the capital of France? Answer with just the city name.",
    "Write a Python function that returns the sum of the even numbers in a list.",
    "Explain in two sentences why a smaller language model can run faster on a GPU.",
]
if args.prompts:
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
tokens = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    for text in prompts
]
torch.set_num_threads(16)
print("Loading original FP32 checkpoint on CPU", flush=True)
reference = Qwen3_5ForConditionalGeneration.from_pretrained(
    args.checkpoint,
    dtype=torch.float32,
    local_files_only=True,
    attn_implementation="eager",
).eval()
expected = []
checked_tokens, checked_prompts, checked_steps = [], [], []
with torch.inference_mode():
    for index, ids in enumerate(tokens):
        prefix = list(ids)
        cache = None
        for step in range(args.continuation_steps):
            inputs = torch.tensor([prefix if cache is None else prefix[-1:]])
            result = reference(
                input_ids=inputs,
                attention_mask=torch.ones((1, len(prefix)), dtype=torch.int64),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            logits = result.logits[0, -1].float().numpy().copy()
            expected.append(logits)
            checked_tokens.append(prefix.copy())
            checked_prompts.append(prompts[index])
            checked_steps.append(step)
            prefix.append(int(logits.argmax()))
            cache = result.past_key_values
        print(f"Reference prompt {index + 1} done", flush=True)
        del cache
tokens, prompts = checked_tokens, checked_prompts
del reference, result
gc.collect()

import onnxruntime_genai as og  # noqa: E402

from evo.runtime import register_webgpu  # noqa: E402

_, runtime_library = register_webgpu(args.ep_library, genai=True)
report = {
    "reference": "Original Transformers checkpoint, FP32 CPU",
    "runtime_library": runtime_library,
    "scope": (
        f"{len(tokens)} logit checks over {args.continuation_steps} FP32 teacher-forced "
        "prefix lengths per prompt; exported model recomputes each prefix; "
        "not long-run recurrent-state or full task-quality validation"
    ),
    "tokenizer_sha256": hash_file(args.checkpoint / "tokenizer.json"),
    "packages": [],
}
for package in args.packages:
    config = og.Config(str(package.resolve()))
    config.set_provider_option("WebGPU", "powerPreference", "high-performance")
    config.set_provider_option("WebGPU", "dawnBackendType", "D3D12")
    config.overlay(json.dumps({"search": {"max_length": max(map(len, tokens)) + 2}}))
    model = og.Model(config)
    results = []
    for prompt, ids, ref, step in zip(prompts, tokens, expected, checked_steps, strict=True):
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=len(ids) + 2, do_sample=False)
        generator = og.Generator(model, params)
        generator.append_tokens(ids)
        # append_tokens computes prefill logits. generate_next_token advances
        # the model and get_logits would then describe the following token.
        actual = np.asarray(generator.get_logits()).reshape(-1, len(ref))[-1].astype(np.float64)
        ref = ref.astype(np.float64)
        finite = np.isfinite(actual) & np.isfinite(ref)
        x, y = actual[finite], ref[finite]
        p = np.exp(y - y.max())
        p /= p.sum()
        logq = x - x.max()
        logq -= np.log(np.exp(logq).sum())
        results.append(
            {
                "prompt": prompt,
                "input_tokens": len(ids),
                "teacher_forced_prefix_step": step,
                "finite_fraction": float(finite.mean()),
                "cosine": float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))),
                "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
                "kl_reference_to_export": float(np.sum(p * (np.log(p + 1e-300) - logq))),
                "top1_match": bool(actual.argmax() == ref.argmax()),
                "reference_top1": tokenizer.decode([int(ref.argmax())]),
                "export_top1": tokenizer.decode([int(actual.argmax())]),
            }
        )
        del generator, params
    report["packages"].append(
        {
            "path": str(package),
            "model_sha256": hash_file(package / "model.onnx"),
            "results": results,
        }
    )
    print(package, results, flush=True)
    del model, config
    gc.collect()
    write_json(args.output, report)
