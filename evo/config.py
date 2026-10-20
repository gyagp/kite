"""Small, strict recipe schema; paths are relative to the recipe file."""

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from evo.paths import asset_root


@dataclass(frozen=True)
class Recipe:
    backend: str
    source: str
    source_kind: str
    output: Path
    precision: str
    activation_dtype: str
    block_size: int = 32
    graph_capture: bool = False
    target: str = "ort-webgpu"
    runtime: str = "native"
    genai_options: dict = field(default_factory=dict)
    genai_config: dict = field(default_factory=dict)
    mobius_options: dict = field(default_factory=dict)
    mobius_save: dict = field(default_factory=dict)
    runtime_provider_options: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {**asdict(self), "output": str(self.output)}


def read_recipe(path: Path) -> Recipe:
    path = path.resolve()
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    base = path.parent
    if base == Path(__file__).resolve().parent.parent / "evo_resources" / "recipes":
        base = asset_root() / "recipes"
    return recipe_from_dict(data, base)


def recipe_from_dict(data: dict, base_dir: Path) -> Recipe:
    allowed = {
        "backend",
        "source",
        "source_kind",
        "output",
        "precision",
        "activation_dtype",
        "block_size",
        "graph_capture",
        "target",
        "runtime",
        "genai_options",
        "genai_config",
        "mobius_options",
        "mobius_save",
        "runtime_provider_options",
    }
    if unknown := data.keys() - allowed:
        raise ValueError(f"Unknown recipe fields: {', '.join(sorted(unknown))}")
    for key in ("backend", "source", "output", "precision"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"Recipe requires a non-empty string: {key}")
    backend, precision = data["backend"], data["precision"]
    supported = {"genai": {"int4", "int8", "fp16", "fp32"}, "mobius": {"fp16", "fp32"}}
    if backend not in supported:
        raise ValueError("backend must be 'genai' or 'mobius'")
    if precision not in supported[backend]:
        raise ValueError(f"{backend} supports these precisions: {sorted(supported[backend])}")
    if data.get("target", "ort-webgpu") != "ort-webgpu":
        raise ValueError("The initial target is 'ort-webgpu'")
    if data.get("runtime", "native") != "native":
        raise ValueError("The initial runtime target is 'native'")
    source_kind = data.get("source_kind", "huggingface")
    if source_kind not in ("huggingface", "local"):
        raise ValueError("source_kind must be 'huggingface' or 'local'")
    source = data["source"]
    if source_kind == "local":
        source_path = (base_dir / source).resolve()
        if not source_path.is_dir():
            raise ValueError(f"Local source must be a checkpoint directory: {source_path}")
        source = str(source_path)
    activation_dtype = data.get("activation_dtype", "fp32" if precision == "fp32" else "fp16")
    if activation_dtype not in ("fp16", "fp32"):
        raise ValueError("activation_dtype must be 'fp16' or 'fp32'")
    if backend == "mobius" and activation_dtype != precision:
        raise ValueError("Mobius float recipes require activation_dtype == precision")
    if backend == "genai" and precision in {"fp16", "fp32"} and activation_dtype != precision:
        raise ValueError("Float GenAI recipes require activation_dtype == precision")
    block_size = data.get("block_size", 32)
    if type(block_size) is not int or block_size not in (16, 32, 64, 128, 256):
        raise ValueError("block_size must be one of 16, 32, 64, 128, 256")
    graph_capture = data.get("graph_capture", False)
    if type(graph_capture) is not bool:
        raise ValueError("graph_capture must be a boolean")
    if backend == "mobius" and ("block_size" in data or graph_capture):
        raise ValueError("block_size and graph_capture are only implemented for GenAI recipes")
    for name in ("genai_options", "genai_config", "mobius_options", "mobius_save"):
        if not isinstance(data.get(name, {}), dict):
            raise ValueError(f"{name} must be a table")
        if name.startswith("genai") and backend != "genai" and data.get(name):
            raise ValueError(f"{name} requires backend='genai'")
        if name.startswith("mobius") and backend != "mobius" and data.get(name):
            raise ValueError(f"{name} requires backend='mobius'")
    managed = {
        "block_size",
        "use_webgpu_fp32",
        "enable_webgpu_graph",
        "hf_remote",
        "filename",
        "config_only",
        "hf_token",
    }
    if managed & data.get("genai_options", {}).keys():
        raise ValueError("genai_options cannot override managed settings or contain credentials")
    if set(data.get("genai_config", {})) - {
        "target_options",
        "drafter_options",
        "speculative_options",
        "runtime_config",
        "search",
    }:
        raise ValueError("Unknown structured GenAI configuration section")
    runtime_options = data.get("runtime_provider_options", {})
    if not isinstance(runtime_options, dict) or any(
        not isinstance(k, str) or not isinstance(v, (str, int, bool))
        for k, v in runtime_options.items()
    ):
        raise ValueError("runtime_provider_options must contain scalar provider options")
    if runtime_options and backend != "genai":
        raise ValueError("runtime_provider_options requires a GenAI runtime configuration")
    output = (base_dir / data["output"]).resolve()
    if source_kind == "local":
        source_path = Path(source)
        if output.is_relative_to(source_path) or source_path.is_relative_to(output):
            raise ValueError("Output and local source directories must not overlap")
    return Recipe(
        backend,
        source,
        source_kind,
        output,
        precision,
        activation_dtype,
        block_size,
        graph_capture,
        genai_options=data.get("genai_options", {}),
        genai_config=data.get("genai_config", {}),
        mobius_options=data.get("mobius_options", {}),
        mobius_save=data.get("mobius_save", {}),
        runtime_provider_options=runtime_options,
    )
