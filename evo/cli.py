import argparse
import json
import sys
from pathlib import Path

from evo.api import build
from evo.backends import make_plan
from evo.config import read_recipe
from evo.inspect import inspect_model
from evo.paths import remap_paths
from evo.pipeline import convert, write_json
from evo.validation import validate_cpu


def main(argv: list[str] | None = None, *, prog: str = "evo") -> int:
    routed = remap_paths(sys.argv[1:] if argv is None else argv)
    argv = routed
    if routed and routed[0] == "image-export":
        from evo.image_qwen21 import main as image_export

        image_export(routed[1:])
        return 0
    if routed and routed[0] in {
        "benchmark",
        "optimize",
        "capabilities",
        "analyze",
        "profile",
        "contrib",
        "ui",
        "embedding-export",
        "embedding-validate",
        "embedding-benchmark",
        "embed",
        "quantize",
        "image-prepare",
        "image-generate",
        "image-optimize",
        "image-report",
    }:
        from importlib import import_module

        try:
            import_module(f"evo.{routed[0].replace('-', '_')}").main(routed[1:])
            return 0
        except Exception as exc:
            print(f"{prog}: {exc}", file=sys.stderr)
            return 1
    parser = argparse.ArgumentParser(prog=prog, description="ONNX exports targeting ORT WebGPU")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("benchmark", help="Measure native GenAI WebGPU performance")
    commands.add_parser("optimize", help="Apply Mobius general and WebGPU passes to a package")
    commands.add_parser("capabilities", help="Inventory installed upstream optimization controls")
    commands.add_parser(
        "analyze", help="Suggest graph optimizations and future WebGPU runtime work"
    )
    commands.add_parser("profile", help="Profile an ONNX model using NPZ inputs on native WebGPU")
    commands.add_parser("contrib", help="Browse installed ONNX/ORT schemas and WebGPU guidance")
    commands.add_parser("ui", help="Start the local model workbench")
    commands.add_parser(
        "embedding-export", help="Export EmbeddingGemma 2 text or multimodal embeddings"
    )
    commands.add_parser("embedding-validate", help="Check embeddings against reference fixtures")
    commands.add_parser("embedding-benchmark", help="Measure native WebGPU embedding performance")
    commands.add_parser("embed", help="Generate text or multimodal embeddings on native ORT WebGPU")
    commands.add_parser("quantize", help="Selectively quantize constant MatMul weights")
    commands.add_parser("image-export", help="Export Qwen-Image 2.1 denoising graphs")
    commands.add_parser("image-prepare", help="Prepare shared diffusion reference inputs")
    commands.add_parser("image-generate", help="Generate and compare ONNX/GGUF images")
    commands.add_parser("image-optimize", help="Optimize an existing Qwen-Image ONNX package")
    commands.add_parser(
        "image-report", help="Summarize matched image speed and fidelity measurements"
    )
    builder = commands.add_parser("build", help="Build directly from a model ID or checkpoint")
    source = builder.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="Hugging Face model ID")
    source.add_argument("--config", help="Local checkpoint directory")
    builder.add_argument("--output", type=Path, required=True)
    builder.add_argument("--backend", choices=["genai", "mobius"], default="genai")
    builder.add_argument("--precision", choices=["int4", "int8", "fp16", "fp32"], default="int4")
    for name in (
        "genai_options",
        "genai_config",
        "mobius_options",
        "mobius_save",
        "runtime_provider_options",
    ):
        builder.add_argument("--" + name.replace("_", "-"), default="{}", help="JSON option object")
    builder.add_argument("--ep", choices=["webgpu"], default="webgpu")
    builder.add_argument("--activation-dtype", choices=["fp16", "fp32"])
    builder.add_argument("--block-size", type=int)
    builder.add_argument("--graph-capture", action="store_true")
    builder.add_argument("--python", default=sys.executable)
    builder.add_argument("--dry-run", action="store_true")
    for command, help_text in (
        ("plan", "Show the export command; no model downloads or output writes"),
        ("convert", "Export a model and record artifact evidence"),
    ):
        sub = commands.add_parser(command, help=help_text)
        sub.add_argument("recipe", type=Path)
        sub.add_argument("--python", default=sys.executable, help="Exporter Python executable")
    inspect = commands.add_parser(
        "inspect", help="Inspect an existing ONNX graph and external data"
    )
    inspect.add_argument("model", type=Path)
    inspect.add_argument("--report", type=Path)
    check = commands.add_parser(
        "validate-cpu", help="Compare ORT CPU outputs against reference NPZ"
    )
    check.add_argument("model", type=Path)
    check.add_argument("--inputs", type=Path, required=True)
    check.add_argument("--expected", type=Path, required=True)
    check.add_argument("--rtol", type=float, default=1e-3)
    check.add_argument("--atol", type=float, default=1e-4)
    check.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build(
                args.model or args.config,
                output=args.output,
                backend=args.backend,
                precision=args.precision,
                execution_provider=args.ep,
                source_kind="local" if args.config else "huggingface",
                activation_dtype=args.activation_dtype,
                block_size=args.block_size,
                graph_capture=args.graph_capture,
                python=args.python,
                dry_run=args.dry_run,
                genai_options=json.loads(args.genai_options),
                genai_config=json.loads(args.genai_config),
                mobius_options=json.loads(args.mobius_options),
                mobius_save=json.loads(args.mobius_save),
                runtime_provider_options=json.loads(args.runtime_provider_options),
            )
        elif args.command == "plan":
            result = make_plan(read_recipe(args.recipe), args.python)
        elif args.command == "convert":
            result = convert(read_recipe(args.recipe), args.python)
        elif args.command == "inspect":
            result = inspect_model(args.model)
        else:
            result = validate_cpu(args.model, args.inputs, args.expected, args.rtol, args.atol)
        if getattr(args, "report", None):
            write_json(args.report, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return int(result.get("structure") == "failed" or result.get("accuracy") == "failed")
    except KeyboardInterrupt:
        print(f"{prog}: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        return 1
