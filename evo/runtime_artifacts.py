"""Verify runtime binaries and module-relative support files before loading them."""

import hashlib
from pathlib import Path


def verify_runtime_artifacts(directory, manifest):
    directory = Path(directory).resolve()
    for group in ("dlls", "resources"):
        for name, expected in manifest.get(group, {}).items():
            relative = Path(name)
            dependency = (directory / relative).resolve()
            if (
                relative.anchor
                or ".." in relative.parts
                or not dependency.is_relative_to(directory)
            ):
                raise ValueError(f"Runtime manifest dependency escapes its directory: {name}")
            if not dependency.is_file():
                raise ValueError(f"Runtime binary differs from its build manifest: {name}")
            digest = hashlib.sha256()
            with dependency.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected["sha256"]:
                raise ValueError(f"Runtime binary differs from its build manifest: {name}")
