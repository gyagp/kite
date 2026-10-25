"""Guarded deletion of an explicitly selected model package directory."""

import os
import shutil
from pathlib import Path


def delete_package(model, confirmation, *, project, evo, data_directory, other_models):
    original = Path(model["path"]).absolute()
    root = original.resolve()
    if not isinstance(confirmation, str) or os.path.normcase(confirmation) != os.path.normcase(
        str(original)
    ):
        raise ValueError("Confirm the exact model directory before deleting its files.")
    if original.is_symlink() or os.path.normcase(str(root)) != os.path.normcase(str(original)):
        raise ValueError(
            "The model directory now points somewhere else. Re-import it before deletion."
        )
    protected = {
        Path(root.anchor),
        Path.home().resolve(),
        Path(project).resolve(),
        Path(project).resolve().parent,
        Path(evo).resolve(),
        (Path(evo) / "artifacts").resolve(),
        (Path(evo) / "models").resolve(),
        (Path(evo) / ".venv").resolve(),
        Path(data_directory).resolve(),
        (Path(data_directory) / "models").resolve(),
    }
    protected.update(
        (Path.home() / name).resolve()
        for name in ("Downloads", "Desktop", "Documents", "Pictures", "Music", "Videos", "AppData")
    )
    if root in protected:
        raise ValueError("Cannot delete a workspace, model collection, or runtime directory.")
    if not root.is_dir() or not any(
        (root / name).is_file()
        for name in (
            "genai_config.json",
            "embedding_config.json",
            "image_config.json",
            "segmentation_config.json",
        )
    ):
        raise ValueError(
            "This directory no longer looks like a model package. Remove the library reference instead."
        )
    for other in other_models:
        if other["id"] == model["id"]:
            continue
        neighbor = Path(other["path"]).resolve()
        if neighbor.is_relative_to(root) or root.is_relative_to(neighbor):
            raise ValueError(
                f"This folder overlaps another registered model: {other['name']}. Remove the overlapping reference first."
            )
    # Python 3.11+ does not traverse directory symlinks or Windows junctions during
    # rmtree. Only this confirmed package directory is passed to the deletion API.
    shutil.rmtree(root)
