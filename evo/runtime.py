"""Select a native WebGPU library without replacing the installed package."""

import json
import os
from pathlib import Path

from evo.paths import asset_path, asset_root
from evo.pipeline import hash_file
from evo.runtime_artifacts import verify_runtime_artifacts

_dll_directories = {}
_registered = {}


def resolve_webgpu_library(library, installed):
    """Explicit argument, environment, project selection, then installed plugin."""
    if library:
        return asset_path(library).resolve(), "explicit"
    if os.environ.get("EVO_WEBGPU_LIBRARY"):
        return asset_path(os.environ["EVO_WEBGPU_LIBRARY"]).resolve(), "environment"
    settings = Path.cwd() / ".evo" / "runtime.json"
    if not settings.is_file():
        settings = asset_root() / ".evo" / "runtime.json"
    if settings.is_file():
        config = json.loads(settings.read_text(encoding="utf-8"))
        selected = Path(config["webgpu_library"])
        if not selected.is_absolute():
            selected = settings.parent / selected
        return selected.resolve(), "project"
    return Path(installed).resolve(), "installed"


def register_webgpu(library=None, *, genai=False):
    import onnxruntime as ort
    import onnxruntime_ep_webgpu as ep

    path, selection = resolve_webgpu_library(library, ep.get_library_path())
    if not path.is_file():
        raise ValueError(f"WebGPU library not found: {path}")
    info = {
        "path": str(path),
        "sha256": hash_file(path),
        "custom_library": selection != "installed",
        "selection": selection,
    }
    metadata = path.parent / "evo-runtime.json"
    if metadata.exists():
        info["build"] = json.loads(metadata.read_text(encoding="utf-8"))
        verify_runtime_artifacts(path.parent, info["build"])
    if os.name == "nt" and str(path.parent) not in _dll_directories:
        _dll_directories[str(path.parent)] = os.add_dll_directory(str(path.parent))
    modules = {"ort": ort}
    if genai:
        import onnxruntime_genai as og

        modules["genai"] = og
    for name, module in modules.items():
        if name in _registered and _registered[name] != path:
            raise ValueError("Use separate processes to compare different WebGPU libraries")
        if name not in _registered:
            module.register_execution_provider_library(ep.get_ep_name(), str(path))
            _registered[name] = path
    devices = [d for d in ort.get_ep_devices() if d.ep_name == ep.get_ep_name()]
    if not devices:
        raise RuntimeError("No native WebGPU device available")
    return devices, info
