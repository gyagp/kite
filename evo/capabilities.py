"""Inventory installed upstream controls and selected WebGPU optimization stages."""

import ast
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path


def source_provenance(module: str, files: list[str]) -> dict:
    root = Path(importlib.util.find_spec(module).origin).parent
    result = {"root": str(root), "files": {}}
    for name in files:
        path = root / name
        if path.is_file():
            result["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def inventory() -> dict:
    result = {
        "scope": "Installed upstream versions; exposed controls are not all mutually compatible"
    }
    spec = importlib.util.find_spec("onnxruntime_genai")
    if spec:
        root = Path(spec.origin).parent / "models"
        options = {}
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr not in {"get", "pop", "setdefault"} or not node.args:
                    continue
                owner = ast.unparse(node.func.value)
                if "extra_options" not in owner:
                    continue
                key = node.args[0]
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    options.setdefault(key.value, []).append(
                        f"{path.relative_to(root)}:{node.lineno}"
                    )
        result["genai"] = {
            "version": importlib.metadata.version("onnxruntime-genai"),
            "extra_options": dict(sorted(options.items())),
            "structured_sections": [
                "target_options",
                "drafter_options",
                "speculative_options",
                "runtime_config",
                "search",
            ],
            "source": source_provenance(
                "onnxruntime_genai",
                [
                    "models/builder.py",
                    "models/builders/base.py",
                    "models/builders/qwen.py",
                    "models/quantization/quant_config.py",
                ],
            ),
        }
    if importlib.util.find_spec("mobius"):
        import dataclasses
        import inspect

        import mobius
        import onnx_ir as ir
        from mobius._execution_providers import ep_registry
        from mobius._flags import list_flags
        from mobius._optimizations import _DEFAULT_PASSES, _get_optimization_passes

        caps = ep_registry.require("webgpu")
        fuse, lower = _get_optimization_passes(caps, ir.DataType.FLOAT16, "decoder")
        result["mobius"] = {
            "version": importlib.metadata.version("mobius-onnx"),
            "build_signature": str(inspect.signature(mobius.build)),
            "save_signature": str(inspect.signature(mobius.ModelPackage.save)),
            "cleanup": [type(p).__name__ for p in _DEFAULT_PASSES],
            "fp16_decoder_fusion": [name for name, _ in fuse],
            "fp16_decoder_lowering": [name for name, _ in lower],
            "post_weights": [
                "LiftConstantsToInitializersPass",
                "FoldConcatInitializersPass",
                "FoldTransposedInitializerPass",
                "RemoveUnusedNodesPass",
            ],
            "webgpu_capabilities": dataclasses.asdict(caps),
            "flags": list_flags(),
            "source": source_provenance(
                "mobius",
                ["_optimizations.py", "_execution_providers.py", "_builder.py", "_flags.py"],
            ),
        }
    # Capabilities contain DataType enums/frozensets; make a stable JSON inventory.
    return json.loads(
        json.dumps(
            result,
            default=lambda x: sorted(map(str, x)) if isinstance(x, (set, frozenset)) else str(x),
        )
    )


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    value = json.dumps(inventory(), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(value + "\n", encoding="utf-8")
    else:
        print(value)


if __name__ == "__main__":
    main()
