"""Download metadata for models explicitly added to the user's library."""

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

REPOSITORY = "webai-community/ai-models"


@lru_cache(maxsize=1)
def snapshot():
    path = Path(__file__).with_name("supported-models.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"models": []}


def catalog_id(name):
    return hashlib.sha256(f"catalog:{REPOSITORY}/{name}".encode()).hexdigest()[:16]


def normalized(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def matches(model, remote):
    if model.get("kind") != remote.get("kind"):
        return False
    if model.get("kind") == "embedding" and model.get("modalities") != remote.get("modalities"):
        return False
    source = model.get("source") or {}
    download = model.get("download") or {}
    candidates = [model.get("name", ""), download.get("model", "")]
    if source.get("repository") == REPOSITORY:
        candidates.append(source.get("path", "").split("/")[0])
    else:
        candidates.append(source.get("repository", "").rsplit("/", 1)[-1])
    family = remote.get("download", {}).get("model", remote["name"])
    return normalized(family) in {normalized(name) for name in candidates if name}


def installed(model):
    path = model.get("path")
    return bool(
        path
        and Path(path).is_dir()
        and all((Path(path) / g).is_file() for g in model.get("graphs", []))
    )


def library_models(records, catalog=None):
    """Enrich registered models without adding Hugging Face discovery results."""
    remote_models = (catalog or snapshot()).get("models", [])
    result = []
    for model in records:
        remote = next((r for r in remote_models if matches(model, r)), None)
        result.append(
            {
                **model,
                **({"download": remote["download"]} if remote else {}),
                "available": installed(model),
            }
        )
    return result


def register_download(store, package, plan, restore_id=None):
    """Restore a retained record's identity so saved conversations stay linked."""
    existing = store.get("models", restore_id) if restore_id else None
    if not existing:
        existing = next(
            (
                m
                for m in store.list("models")
                if (m.get("source") or {}).get("repository") == REPOSITORY
                and (m.get("source") or {}).get("path") == plan["path"]
                and not installed(m)
            ),
            None,
        )
    if not existing and not restore_id:
        remote = {
            "name": plan["name"],
            "kind": package["kind"],
            "modalities": package["modalities"],
        }
        candidates = [m for m in store.list("models") if not installed(m) and matches(m, remote)]
        existing = candidates[0] if len(candidates) == 1 else None
    if existing:
        package["id"] = existing["id"]
        package["name"] = existing["name"]
    elif restore_id:
        package["id"] = restore_id
    package["download"] = {
        "repository": REPOSITORY,
        "model": plan["name"],
        "path": plan["path"],
        "revision": plan["revision"],
    }
    return store.put("models", package)
