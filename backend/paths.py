"""Kite-owned assets and relocation of historical paths without altering evidence."""

import json
import os
import sqlite3
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent


def asset_root():
    if os.environ.get("KITE_ASSET_ROOT"):
        return Path(os.environ["KITE_ASSET_ROOT"]).expanduser().resolve()
    for parent in (PROJECT, *PROJECT.parents):
        candidate = parent / "local"
        if (candidate / "kite-assets.json").is_file():
            return candidate.resolve()
    return PROJECT / "local"


def workspace_root():
    return asset_root().parent


def remap_paths(value, root=None):
    root = Path(root) if root else asset_root()
    marker = root / "kite-assets.json"
    if not marker.is_file():
        return value
    legacy = json.loads(marker.read_text(encoding="utf-8-sig")).get("legacy_root", "")
    if not legacy:
        return value
    legacy = legacy.replace("\\", "/").rstrip("/")

    def remap(item):
        if isinstance(item, str):
            normalized = item.replace("\\", "/")
            if normalized.lower() == legacy.lower():
                return str(root)
            if normalized.lower().startswith(legacy.lower() + "/"):
                return str(root / normalized[len(legacy) + 1 :])
        if isinstance(item, dict):
            return {k: remap(v) for k, v in item.items()}
        if isinstance(item, list):
            return [remap(v) for v in item]
        return item

    return remap(value)


def migrate_store(store):
    """Back up once, relocate references, and keep stable model/conversation IDs."""
    root = asset_root()
    if not (root / "kite-assets.json").is_file():
        return
    marker = "kite-owned-assets-v1"
    if store.get("private", marker):
        return
    backup = store.directory / "before-asset-relocation.db"
    if not backup.exists():
        target = sqlite3.connect(backup)
        try:
            store.db.backup(target)
        finally:
            target.close()
    rows = store.db.execute("SELECT kind,id,data FROM records").fetchall()
    for kind, identifier, raw in rows:
        value = json.loads(raw)
        updated = remap_paths(value, root)
        if kind == "settings":
            updated["asset_root"] = updated.pop("evo_root", updated.get("asset_root", str(root)))
        if updated != value:
            store.db.execute(
                "UPDATE records SET data=? WHERE kind=? AND id=?",
                (json.dumps(updated, ensure_ascii=False), kind, identifier),
            )
    store.db.commit()
    # Workbench snapshots use stable IDs too; leave original benchmark reports byte-for-byte intact.
    for path in (store.directory / "workbench").glob("*.json"):
        original = json.loads(path.read_text(encoding="utf-8"))
        updated = remap_paths(original, root)
        if updated != original:
            saved = path.with_suffix(".before-relocation.json")
            if not saved.exists():
                saved.write_bytes(path.read_bytes())
            path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
    store.put("private", {"id": marker, "asset_root": str(root)})
