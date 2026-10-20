"""Refresh the bundled ONNX catalog using metadata only from the approved repository."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.hub import BASE, REPOSITORY, fetch_json, resolve_url, tree  # noqa: E402
from backend.supported import catalog_id  # noqa: E402


def main():
    revision = fetch_json(f"{BASE}/api/models/{REPOSITORY}/revision/main")["sha"]
    families = [r["path"] for r in tree(revision) if r["type"] == "directory"]

    def inspect(name):
        rows = [r for r in tree(revision, name, recursive=True) if r["type"] == "file"]
        configs = [
            r
            for r in rows
            if PurePosixPath(r["path"]).name in ("genai_config.json", "embedding_config.json")
            and "onnx" in r["path"].lower()
        ]
        configs.sort(
            key=lambda r: (
                "webgpu" not in r["path"].lower(),
                "int4" not in r["path"].lower(),
                len(r["path"]),
                r["path"],
            )
        )
        for row in configs:
            folder = str(PurePosixPath(row["path"]).parent)
            files = [r for r in rows if r["path"].startswith(folder + "/")]
            if not any(r["path"] == folder + "/tokenizer.json" for r in files):
                continue
            config = fetch_json(resolve_url(revision, row["path"]))
            native = config.get("model", {})
            if native.get("type") == "whisper":
                continue
            embedding = PurePosixPath(row["path"]).name == "embedding_config.json"
            modalities = config.get("modalities", ["text"]) if embedding else ["text"]
            if not embedding:
                if "vision" in native:
                    modalities.append("image")
                if "speech" in native or "audio" in native:
                    modalities.append("audio")
            model = {
                "id": catalog_id(name),
                "name": name,
                "path": "",
                "kind": "embedding" if embedding else "chat",
                "architecture": config.get("architecture")
                if embedding
                else native.get("type", "GenAI"),
                "modalities": modalities,
                "size": sum(r["size"] for r in files if not r["path"].endswith(".py")),
                "precision": config.get(
                    "precision", "int4" if "int4" in folder.lower() else "mixed"
                ),
                "context": config.get("max_length", native.get("context_length", 0)),
                "dimensions": config.get("embedding_dimensions"),
                "tasks": [],
                "graphs": [],
                "variant": folder[len(name) + 1 :],
                "available": False,
                "source": {"repository": REPOSITORY, "path": folder, "revision": revision},
                "download": {
                    "repository": REPOSITORY,
                    "model": name,
                    "path": folder,
                    "revision": revision,
                },
            }
            print(name, folder, flush=True)
            return model
        print("No compatible package:", name, flush=True)
        return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        models = [m for m in pool.map(inspect, families) if m]
    result = {"repository": REPOSITORY, "revision": revision, "models": models}
    (ROOT / "backend/supported-models.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(models)} models at {revision}", flush=True)


if __name__ == "__main__":
    main()
