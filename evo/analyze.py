"""Suggest export/graph work and distinguish it from proposed runtime development."""

import argparse
import json
from collections import Counter
from pathlib import Path

import onnx

from evo.inspect import inspect_model
from evo.pipeline import hash_file, write_json


def analyze(model: Path, profile: Path | None = None) -> dict:
    model = Path(model)
    report = inspect_model(model)
    graph = onnx.load(model, load_external_data=False).graph
    counts = Counter(node.op_type for node in graph.node)
    suggestions = []

    def add(identifier, category, title, evidence, action, status="unmeasured"):
        suggestions.append(
            {
                "id": identifier,
                "category": category,
                "title": title,
                "evidence": evidence,
                "action": action,
                "status": status,
            }
        )

    add(
        "mobius-pipeline",
        "available_now",
        "Run the full Mobius WebGPU pass pipeline",
        {"node_count": len(graph.node)},
        "Use evo optimize; inspect pass matches, preserve the I/O contract, and validate numerics.",
    )
    if counts["MatMulNBits"]:
        try:
            import onnx_ir as ir

            from evo.contrib_rules import fuse_separate_mlp, mlp_patterns

            graph_ir = ir.load(model)
            matches = mlp_patterns(graph_ir)
            separate = fuse_separate_mlp(graph_ir, dry_run=True)
            if separate:
                add(
                    "separate-mlp-contrib",
                    "available_now",
                    "Fuse separate quantized gate/up projections",
                    {
                        "matches": len(separate),
                        "source_nodes": [m["source_node"] for m in separate],
                    },
                    "Use evo optimize --contrib. Reuses original INT4/block-32 weights and "
                    "scales without re-quantization. Inspect native kernel selection and validate "
                    "quality and performance; a fused ONNX node alone does not prove "
                    "faster execution.",
                )
            if matches:
                add(
                    "packed-mlp-contrib",
                    "available_now",
                    "Try the MatMulNBitsMlp contrib operator",
                    {"matches": len(matches), "source_nodes": [m[0].name for m in matches]},
                    "evo optimize --contrib slices symmetric INT4/block-32 packed projections "
                    "without re-quantization. The kernel can change floating-point results; "
                    "validate the candidate. The Qwen trial did not show a decode gain.",
                )
        except ImportError:
            pass
        add(
            "quant-policy",
            "available_now",
            "Measure quantization and projection-fusion policies",
            {"MatMulNBits": counts["MatMulNBits"]},
            "Compare block sizes and accuracy_level on the same fixtures; use mixed-precision "
            "overrides for sensitive nodes. These require source weights for requantization.",
        )
    logits = next((v for v in report["outputs"] if v["name"] == "logits"), None)
    if logits and logits["shape"] and len(logits["shape"]) == 3 and logits["shape"][1] != 1:
        add(
            "last-token-head",
            "available_now",
            "Compute only required prompt logits",
            {"logits_shape": logits["shape"]},
            "GenAI prune_lm_head or Mobius prune_prefill_prefix; requires a generation-only output "
            "contract and is incompatible with losses/all-token logits and GenAI MTP export.",
        )
    if counts["LinearAttention"]:
        add(
            "gated-delta-integration",
            "available_now",
            "Evaluate the existing WebGPU GatedDeltaNet path",
            {
                "LinearAttention": counts["LinearAttention"],
                "LpNormalization": counts["LpNormalization"],
                "LinearAttentionGate": counts["LinearAttentionGate"],
            },
            "The audited ORT source already implements WebGPU GatedDeltaNet with fused Q/K L2 "
            "normalization and Qwen gates. GenAI 0.17.1 restricts its exporter to CUDA. "
            "The Qwen3.5-4B graph rewrite replaces the validated pattern with "
            "matching source gate parameters and FP32 V-major state. It requires the source "
            "checkpoint, fresh caches and a GatedDeltaNet-capable WebGPU plugin; "
            "installed plugin 0.4.0 is insufficient.",
        )
    if counts["CausalConvWithState"] and (counts["LinearAttention"] or counts["GatedDeltaNet"]):
        add(
            "conv-recurrent-fusion",
            "proposed_runtime_work",
            "Prototype fused convolution-to-recurrent input preparation",
            {
                "CausalConvWithState": counts["CausalConvWithState"],
                "Transpose": counts["Transpose"],
                "Reshape": counts["Reshape"],
            },
            "The audited implementations stage convolution and recurrent attention separately. "
            "Investigate combining convolution output layout/gating preparation with recurrent "
            "input preparation to remove intermediate writes and dispatches. Requires a kernel "
            "prototype, state-update tests and GPU timings; no speedup is claimed.",
        )
    if (
        logits
        and logits["shape"]
        and isinstance(logits["shape"][-1], int)
        and logits["shape"][-1] > 100000
    ):
        add(
            "lm-head-selection",
            "proposed_runtime_work",
            "Prototype a quantized LM-head plus greedy token-selection epilogue",
            {"vocabulary": logits["shape"][-1], "logits_shape": logits["shape"]},
            "Evaluate tiled MatMulNBits plus partial argmax reduction, avoiding full logits "
            "materialization for eligible greedy requests. No such epilogue was identified in "
            "the audited MatMulNBits WebGPU source. Preserve full-logit fallback for sampling, "
            "penalties and external consumers; measure whether bandwidth savings justify it.",
        )
    profile_summary = None
    if profile:
        profile = Path(profile)
        events = json.loads(profile.read_text(encoding="utf-8"))
        run_events = [event for event in events if event.get("name") == "model_run"]
        measured_start = run_events[1]["ts"] if len(run_events) > 1 else 0
        info_path = profile.parent / "profile-info.json"
        if info_path.is_file():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if info.get("model_sha256") != hash_file(model):
                raise ValueError("Profile model hash does not match this ONNX graph")
        totals, calls, providers = Counter(), Counter(), Counter()
        for event in events:
            if event.get("ts", 0) < measured_start:
                continue
            args = event.get("args", {})
            if event.get("cat") == "Node" and args.get("provider") and args.get("op_name"):
                key = f"{args['provider']}::{args['op_name']}"
                totals[key] += event.get("dur", 0)
                calls[key] += 1
                providers[args["provider"]] += 1
        profile_summary = {
            "excluded_first_run": bool(measured_start),
            "sha256": hash_file(profile),
            "provider_node_events": dict(providers),
            "largest_trace_durations": [
                {"operator": k, "event_us": v, "calls": calls[k]} for k, v in totals.most_common(15)
            ],
            "limitation": "Trace durations may measure host dispatch, not GPU kernel time.",
        }
        if providers.get("CPUExecutionProvider"):
            add(
                "cpu-placement",
                "profile_followup",
                "Inspect CPU partitions and transfer boundaries",
                dict(providers),
                "Distinguish inexpensive shape work from compute fallback. "
                "Target measured transfer/compute costs before proposing missing GPU kernels.",
                "trace_observed",
            )
    return {
        "model": str(model.resolve()),
        "model_sha256": hash_file(model),
        "structure": report["structure"],
        "errors": report["errors"],
        "operators": report["operators"],
        "profile": profile_summary,
        "suggestions": suggestions,
        "scope": "Candidates only; not measured gains or a complete runtime support matrix.",
        "source_audit": "docs/optimization-coverage.md",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = analyze(args.model, args.profile)
    if args.output:
        write_json(args.output, result)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
