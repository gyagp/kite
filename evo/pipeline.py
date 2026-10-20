"""Run an exporter and retain logs, artifact hashes and validation evidence."""

import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from evo import __version__
from evo.backends import BACKEND_PACKAGES, make_plan
from evo.config import Recipe
from evo.inspect import inspect_model


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def hash_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def apply_runtime_provider_options(package: Path, options: dict):
    """Set runtime policy after graph export, without changing the graph contract."""
    if not options:
        return
    path = package / "genai_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    providers = config["model"]["decoder"]["session_options"]["provider_options"]
    provider = next((p["webgpu"] for p in providers if "webgpu" in p), None)
    if provider is None:
        raise ValueError("Exported decoder has no WebGPU provider")
    for key, value in options.items():
        provider[key] = str(int(value)) if isinstance(value, bool) else str(value)
    write_json(path, config)


def environment_info(python: str, backend: str) -> dict:
    # Query the selected interpreter, which can differ from Evo's environment.
    script = (
        "import importlib.metadata as m,json,platform; "
        "names="
        + repr([BACKEND_PACKAGES[backend], "onnx", "onnxruntime", "torch", "transformers"])
        + "; installed={d.metadata['Name'].lower().replace('_','-'):d.version "
        "for d in m.distributions()}; "
        "print(json.dumps({'python':platform.python_version(),"
        "'platform':platform.platform(),'packages':{n:installed.get(n) for n in names}}))"
    )
    result = subprocess.run([python, "-c", script], capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    if info["packages"][BACKEND_PACKAGES[backend]] is None:
        raise ValueError(
            f"{BACKEND_PACKAGES[backend]} is missing in {python}. "
            f"Install the [{backend}] dependencies in that environment; see README.md."
        )
    return info


def convert(recipe: Recipe, python: str = sys.executable) -> dict:
    if recipe.output.exists() and (not recipe.output.is_dir() or any(recipe.output.iterdir())):
        raise ValueError(f"Output must be absent or empty; choose a new directory: {recipe.output}")
    environment = environment_info(python, recipe.backend)
    plan = make_plan(recipe, python)
    recipe.output.mkdir(parents=True, exist_ok=True)
    manifest_path = recipe.output / "evo-manifest.json"
    manifest = {
        "schema_version": 1,
        "evo_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "running",
        **plan,
        "environment": environment,
        "accuracy": "not_run",
        "webgpu": "not_run",
        "performance": "not_run",
        "models": [],
        "artifacts": [],
    }
    if recipe.source_kind == "local":
        provenance = Path(recipe.source) / "evo-source.json"
        if provenance.is_file():
            manifest["declared_source_provenance"] = json.loads(
                provenance.read_text(encoding="utf-8")
            )
    write_json(manifest_path, manifest)
    started = time.monotonic()
    try:
        log_path = recipe.output / "export.log"
        print(f"Exporting with {recipe.backend}; log: {log_path}", file=sys.stderr, flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(plan["argv"], stdout=log, stderr=subprocess.STDOUT, check=False)
        manifest["export_exit_code"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"Exporter exited with code {result.returncode}; see {log_path}")
        apply_runtime_provider_options(recipe.output, recipe.runtime_provider_options)
        models = sorted(recipe.output.rglob("*.onnx"))
        if not models:
            raise RuntimeError("Exporter returned success but produced no .onnx files")
        for model in models:
            inspection = inspect_model(model)
            report_path = model.with_suffix(".inspection.json")
            write_json(report_path, inspection)
            manifest["models"].append(
                {
                    "path": model.relative_to(recipe.output).as_posix(),
                    "report": report_path.relative_to(recipe.output).as_posix(),
                    "structure": inspection["structure"],
                }
            )
        if any(model["structure"] != "passed" for model in manifest["models"]):
            raise RuntimeError("Exported graphs failed structural checks; see inspection reports")
        for artifact in sorted(recipe.output.rglob("*")):
            if artifact.is_file() and artifact != manifest_path:
                manifest["artifacts"].append(
                    {
                        "path": artifact.relative_to(recipe.output).as_posix(),
                        "bytes": artifact.stat().st_size,
                        "sha256": hash_file(artifact),
                    }
                )
        manifest["status"] = "exported"
    except (Exception, KeyboardInterrupt) as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc) or "Interrupted"
        raise
    finally:
        manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_json(manifest_path, manifest)
    return manifest
