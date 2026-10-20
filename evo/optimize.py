"""Apply Mobius's complete general and WebGPU-aware pipeline to an existing package."""

import argparse
import gc
import importlib.metadata
import logging
import shutil
import time
from pathlib import Path

from evo.inspect import inspect_model
from evo.pipeline import hash_file, write_json
from evo.schemas import ort_legacy_schemas


def optimize_package(source: Path, output: Path, dtype="auto", role="auto", contrib=False) -> dict:
    import onnx_ir as ir
    from mobius._optimizations import (
        fold_initializers_after_weights,
        optimize_model,
        strip_debug_metadata,
    )
    from onnx_ir.passes.common import RemoveUnusedFunctionsPass

    from evo.capabilities import source_provenance

    source, output = Path(source).resolve(), Path(output).resolve()
    single = source.is_file()
    source_root = source.parent if single else source
    if (not single and output.is_relative_to(source)) or source.is_relative_to(output):
        raise ValueError("Source and output must not overlap")
    if output.exists():
        raise ValueError("Choose a new output directory")
    graphs = [source] if single and source.suffix == ".onnx" else sorted(source.rglob("*.onnx"))
    if not graphs:
        raise ValueError("No ONNX graphs found")
    inspections = {path: inspect_model(path) for path in graphs}
    if any(r["structure"] != "passed" for r in inspections.values()):
        raise ValueError("Source graphs must pass structural inspection")
    excluded = set(graphs)
    for path, report in inspections.items():
        excluded.update(
            (path.parent / data["location"]).resolve() for data in report["external_data"]
        )
    output.mkdir(parents=True)
    log = logging.FileHandler(output / "optimization.log", encoding="utf-8")
    logger = logging.getLogger()
    old_level = logger.level
    logger.addHandler(log)
    logger.setLevel(logging.INFO)
    started = time.perf_counter()
    report = {
        "status": "running",
        "source": str(source),
        "output": str(output),
        "backend": "mobius",
        "version": importlib.metadata.version("mobius-onnx"),
        "target": "webgpu",
        "dtype": dtype,
        "role": role,
        "contrib_fusions": contrib,
        "source_provenance": source_provenance(
            "mobius", ["_optimizations.py", "_execution_providers.py"]
        ),
        "graphs": [],
        "accuracy": "not_run",
        "webgpu": "not_run",
        "performance": "not_run",
    }
    try:
        report["source_artifacts"] = [
            {
                "path": str(path.relative_to(source_root)),
                "bytes": path.stat().st_size,
                "sha256": hash_file(path),
            }
            for path in sorted(excluded)
        ]
        for path in [] if single else source.rglob("*"):
            if not path.is_file() or path in excluded:
                continue
            if path.name.endswith((".log", ".inspection.json")) or path.name == "evo-manifest.json":
                continue
            dest = output / path.relative_to(source_root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        for path in graphs:
            model = ir.load(path)
            resolved_dtype = dtype
            if dtype == "auto":
                types = inspections[path]["tensor_type_counts"]
                if "BFLOAT16" in types:
                    raise ValueError("BF16 model requires explicit dtype policy for WebGPU")
                resolved_dtype = (
                    "fp16" if types.get("FLOAT16", 0) > types.get("FLOAT", 0) else "fp32"
                )
            original_functions = set(model.functions)
            original_native_ops = {
                (node.domain, node.op_type, node.overload) for node in model.graph.all_nodes()
            } - original_functions
            before_inputs = [v.name for v in model.graph.inputs]
            before_outputs = [v.name for v in model.graph.outputs]
            resolved_role = role
            if role == "auto":
                resolved_role = (
                    "decoder"
                    if any(
                        name.startswith(("past", "present"))
                        for name in before_inputs + before_outputs
                    )
                    else "generic"
                )
            before_contract = [
                str(v.type) + str(v.shape) for v in (*model.graph.inputs, *model.graph.outputs)
            ]
            contrib_changes = []
            if contrib:
                from evo.contrib_rules import fuse_packed_mlp, fuse_separate_mlp

                contrib_changes = fuse_separate_mlp(model) + fuse_packed_mlp(model)
            names = {node.op_type for node in model.graph.all_nodes() if node.domain == ""}
            with ort_legacy_schemas(names, model.graph.opset_imports[""]):
                optimize_model(
                    model,
                    ep="webgpu",
                    dtype={"fp16": ir.DataType.FLOAT16, "fp32": ir.DataType.FLOAT}[resolved_dtype],
                    model_role=resolved_role,
                    trace=True,
                )
                fold_initializers_after_weights(model)
                strip_debug_metadata(model)
                # Mobius registers fallback function bodies at its own opset.
                # Unused bodies must not impose that opset on a GenAI graph.
                RemoveUnusedFunctionsPass()(model)
                # These source nodes already relied on native kernels, not local
                # functions. Restore that contract rather than introducing Mobius's
                # opset-24 fallback bodies into an older-opset GenAI graph.
                removed_fallbacks = []
                native_fallback_names = {
                    ("com.microsoft", "MatMulNBits"),
                    ("com.microsoft", "SkipSimplifiedLayerNormalization"),
                }
                for node in model.graph.all_nodes():
                    if (
                        (node.domain, node.op_type) in native_fallback_names
                        and (node.domain, node.op_type, "") in original_native_ops
                        and node.op_identifier() in model.functions
                        and node.op_identifier() not in original_functions
                    ):
                        node.overload = ""
                for identifier in list(model.functions):
                    if identifier not in original_functions and (
                        identifier in original_native_ops
                        or (
                            identifier[:2] in native_fallback_names
                            and (*identifier[:2], "") in original_native_ops
                        )
                    ):
                        removed_fallbacks.append("::".join(identifier))
                        del model.functions[identifier]
            after_contract = [
                str(v.type) + str(v.shape) for v in (*model.graph.inputs, *model.graph.outputs)
            ]
            if (
                before_inputs != [v.name for v in model.graph.inputs]
                or before_outputs != [v.name for v in model.graph.outputs]
                or before_contract != after_contract
            ):
                raise ValueError(
                    "Optimizer changed package I/O contract; runtime config needs adaptation: "
                    f"{before_contract} -> {after_contract}"
                )
            target = output / path.relative_to(source_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            ir.save(model, target, external_data=target.name + ".data")
            del model
            gc.collect()
            inspection = inspect_model(target)
            write_json(target.with_suffix(".inspection.json"), inspection)
            report["graphs"].append(
                {
                    "path": str(path.relative_to(source_root)),
                    "optimization_dtype": resolved_dtype,
                    "role": resolved_role,
                    "source_sha256": hash_file(path),
                    "output_sha256": hash_file(target),
                    "operators_before": inspections[path]["operators"],
                    "operators_after": inspection["operators"],
                    "structure": inspection["structure"],
                    "removed_new_native_fallbacks": removed_fallbacks,
                    "contrib_changes": contrib_changes,
                }
            )
            if inspection["structure"] != "passed":
                raise ValueError("Post-optimization structural check failed")
        report["status"] = "optimized"
        report["artifacts"] = [
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": hash_file(path),
            }
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.suffix != ".log"
        ]
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "optimization.json", report)
        logging.getLogger().removeHandler(log)
        logging.getLogger().setLevel(old_level)
        log.close()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["auto", "fp16", "fp32"], default="auto")
    parser.add_argument(
        "--role", choices=["auto", "decoder", "encoder", "vision", "generic"], default="auto"
    )
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument(
        "--contrib", action="store_true", help="Try validated-pattern contrib-op rewrites"
    )
    policy.add_argument("--no-contrib", action="store_false", dest="contrib")
    parser.set_defaults(contrib=False)
    args = parser.parse_args(argv)
    result = optimize_package(args.source, args.output, args.dtype, args.role, args.contrib)
    print(result["status"])


if __name__ == "__main__":
    main()
