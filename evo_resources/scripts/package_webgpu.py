"""Package a clean local WebGPU build with verified DXC and source provenance."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from evo.pipeline import hash_file, write_json


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="Revision recorded before the build")
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--build-log", type=Path, required=True)
    parser.add_argument("--build-exit", type=Path, required=True)
    parser.add_argument("--dxc-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, build = args.source.resolve(), args.build.resolve()
    if args.output.exists():
        raise ValueError("Use a new immutable runtime directory")
    if args.build_exit.read_text().strip() != "0":
        raise ValueError("Build did not complete successfully")
    if git(source, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Expected a clean tracked source checkout")
    revision = git(source, "rev-parse", "HEAD")
    if revision != args.revision:
        raise ValueError("Source HEAD changed since the build was started")
    cache = build / "CMakeCache.txt"
    entries = {}
    for line in cache.read_text().splitlines():
        if "=" in line and not line.startswith(("//", "#")):
            key, value = line.split("=", 1)
            entries[key.split(":")[0]] = value
    if Path(entries["CMAKE_HOME_DIRECTORY"]).resolve() != source / "cmake":
        raise ValueError("Build cache belongs to another checkout")
    dxc = json.loads((args.dxc_package / "evo-runtime.json").read_text())
    dependencies = {}
    for name in ("dxcompiler.dll", "dxil.dll"):
        path = args.dxc_package / name
        if hash_file(path) != dxc["dlls"][name]["sha256"]:
            raise ValueError(f"DXC package changed: {name}")
        dependencies[name] = path
    library = build / "onnxruntime_providers_webgpu.dll"
    if not library.exists():
        raise ValueError("Built provider DLL missing")
    args.output.mkdir(parents=True)
    for name, path in {**dependencies, library.name: library}.items():
        shutil.copyfile(path, args.output / name)
    manifest = {
        "source": git(source, "remote", "get-url", "origin"),
        "source_directory": str(source),
        "revision": revision,
        "branch": git(source, "branch", "--show-current"),
        "source_version": (source / "VERSION_NUMBER").read_text().strip(),
        "variant": "upstream-main-" + revision[:10],
        "source_patch": None,
        "config": entries["CMAKE_BUILD_TYPE"],
        "compiler": entries["CMAKE_CXX_COMPILER"],
        "generator": entries["CMAKE_GENERATOR"],
        "lto": entries.get("onnxruntime_ENABLE_LTO") == "ON",
        "rtti": entries.get("onnxruntime_DISABLE_RTTI") != "ON",
        "backends": {
            name: entries.get("onnxruntime_ENABLE_DAWN_BACKEND_" + name) == "ON"
            for name in ("D3D12", "VULKAN")
        },
        "cmake_source": entries["CMAKE_HOME_DIRECTORY"],
        "cmake_cache_sha256": hash_file(cache),
        "build_log": str(args.build_log.resolve()),
        "build_exit_code": 0,
        "dxc_archive_sha256": dxc["dxc_archive_sha256"],
        "dxc_source": "Official v1.8.2502, as selected by upstream plugin-win-webgpu-stage.yml",
        "dlls": {
            p.name: {"bytes": p.stat().st_size, "sha256": hash_file(p)}
            for p in args.output.glob("*.dll")
        },
    }
    write_json(args.output / "evo-runtime.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
