"""Resolve relocated Evo evidence inside the owning Kite workspace."""

from pathlib import Path

from backend.paths import asset_root, remap_paths


def asset_path(value):
    return Path(remap_paths(str(value)))


__all__ = ["asset_path", "asset_root", "remap_paths"]
