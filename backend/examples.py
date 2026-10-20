"""Bundled, locally generated demonstration media; no remote downloads."""

import hashlib
import mimetypes
from pathlib import Path

from .catalog import read_json


def examples(root):
    return read_json(Path(root) / "demo" / "examples.json", [])


def list_examples(root):
    return [
        {k: v for k, v in item.items() if k not in ("samples", "query_files")}
        for item in examples(root)
    ]


def example_asset(root, name):
    """Expose only media explicitly used by the bundled collection."""
    allowed = set()
    for item in examples(root):
        allowed.update(
            item.get(key) for key in ("cover", "preview_video", "preview_audio") if item.get(key)
        )
        allowed.update(item.get("query_files", []))
        for sample in item.get("samples", []):
            allowed.update(sample.get("files", []))
    folder = (Path(root) / "demo").resolve()
    path = (folder / name).resolve()
    if name not in allowed or not path.is_relative_to(folder) or not path.is_file():
        raise ValueError("Example asset not found.")
    return path


def load_example(root, store, identifier):
    item = next((item for item in examples(root) if item["id"] == identifier), None)
    if item is None:
        raise ValueError("Example not found.")
    folder = (Path(root) / "demo").resolve()

    def attachments(names):
        result = []
        for name in names:
            path = (folder / name).resolve()
            if not path.is_relative_to(folder) or not path.is_file():
                raise ValueError(f"Bundled example file is missing: {name}")
            mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
            kind = (
                "image"
                if mime.startswith("image/")
                else "audio"
                if mime.startswith("audio/")
                else "video"
                if mime.startswith("video/")
                else None
            )
            if kind is None:
                raise ValueError("Invalid bundled media type.")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            record = store.put(
                "attachments",
                {
                    "id": "demo-" + digest[:24],
                    "name": name,
                    "kind": kind,
                    "mime": mime,
                    "size": path.stat().st_size,
                    "path": str(path),
                },
            )
            result.append({k: v for k, v in record.items() if k != "path"})
        return result

    return {
        **{k: v for k, v in item.items() if k not in ("samples", "query_files")},
        "query_attachments": attachments(item.get("query_files", [])),
        "samples": [
            {
                "text": sample["text"],
                "label": sample.get("label", ""),
                "attachments": attachments(sample.get("files", [])),
            }
            for sample in item["samples"]
        ],
    }
