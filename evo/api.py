"""Public build API, sharing the CLI's validation and export pipeline."""

import sys
from pathlib import Path

from evo.backends import make_plan
from evo.config import recipe_from_dict
from evo.pipeline import convert


def build(
    model: str,
    *,
    output: str | Path,
    backend: str = "genai",
    precision: str = "int4",
    execution_provider: str = "webgpu",
    source_kind: str = "huggingface",
    activation_dtype: str | None = None,
    block_size: int | None = None,
    graph_capture: bool = False,
    python: str = sys.executable,
    dry_run: bool = False,
    genai_options: dict | None = None,
    genai_config: dict | None = None,
    mobius_options: dict | None = None,
    mobius_save: dict | None = None,
    runtime_provider_options: dict | None = None,
) -> dict:
    """Export to ``output`` and return its manifest, or return a plan with dry_run.

    Native ORT WebGPU is the initial runtime target. For a local checkpoint,
    supply source_kind="local". Relative paths resolve against the current cwd.
    This API writes the package directly; no additional save() call is required.
    """
    if execution_provider != "webgpu":
        raise ValueError("The initial execution_provider is 'webgpu'")
    data = {
        "backend": backend,
        "source": model,
        "output": str(output),
        "precision": precision,
        "source_kind": source_kind,
        "graph_capture": graph_capture,
        "genai_options": genai_options or {},
        "genai_config": genai_config or {},
        "mobius_options": mobius_options or {},
        "mobius_save": mobius_save or {},
        "runtime_provider_options": runtime_provider_options or {},
    }
    if activation_dtype is not None:
        data["activation_dtype"] = activation_dtype
    if block_size is not None:
        data["block_size"] = block_size
    recipe = recipe_from_dict(data, Path.cwd())
    return make_plan(recipe, python) if dry_run else convert(recipe, python)
