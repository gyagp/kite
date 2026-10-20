"""Revision-pinned, verified ONNX downloads from webai-community/ai-models."""

import hashlib
import json
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

from .catalog import inspect_package, read_json
from .supported import library_models, matches, register_download, snapshot

REPOSITORY = "webai-community/ai-models"
BASE = "https://huggingface.co"


def safe_path(root, name):
    if not isinstance(name, str) or "\\" in name or ":" in name:
        raise ValueError("Invalid package path.")
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(p in (".", "..") or p.endswith((" ", ".")) for p in relative.parts)
    ):
        raise ValueError("Invalid package path.")
    root = Path(root).resolve()
    path = (root / str(relative)).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Package files must remain inside their directory.")
    return path


def open_url(url, headers=None):
    return urllib.request.urlopen(
        urllib.request.Request(
            url, headers={"User-Agent": "Kite/0.1 local-model-client", **(headers or {})}
        ),
        timeout=20,
    )


def fetch_json(url):
    with open_url(url) as response:
        raw = response.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("Repository metadata is unexpectedly large.")
        return json.loads(raw)


def tree(revision, folder="", recursive=False):
    path = urllib.parse.quote(folder, safe="/")
    url = f"{BASE}/api/models/{REPOSITORY}/tree/{revision}/{path}?recursive={'true' if recursive else 'false'}&limit=1000"
    rows = []
    for _ in range(20):
        with open_url(url) as response:
            rows.extend(json.load(response))
            links = response.headers.get("Link", "")
        match = re.search(r'<([^>]+)>;\s*rel="next"', links)
        if not match:
            return rows
        url = match.group(1)
        if not url.startswith(f"{BASE}/api/models/{REPOSITORY}/tree/"):
            raise ValueError("Unexpected repository pagination link.")
    raise ValueError("Repository listing exceeded the supported page count.")


def revision_ok(revision):
    if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Choose a pinned repository revision from the catalog.")
    return revision


def resolve_url(revision, path):
    revision_ok(revision)
    return (
        f"{BASE}/{REPOSITORY}/resolve/{revision}/{urllib.parse.quote(path, safe='/')}?download=true"
    )


def verify_file(path, spec):
    if not path.is_file() or path.stat().st_size != spec["size"]:
        return False
    algorithm = hashlib.sha256() if spec.get("sha256") else hashlib.sha1()
    if not spec.get("sha256"):
        algorithm.update(f"blob {spec['size']}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            algorithm.update(block)
    return algorithm.hexdigest() == (spec.get("sha256") or spec["oid"])


class Cancelled(Exception):
    pass


class Hub:
    def __init__(self, store):
        self.store = store
        self.lock = threading.RLock()
        self.catalog_cache = None
        self.package_cache = {}
        self.stop_event = threading.Event()
        self.thread = None
        for item in store.list("downloads"):
            if item["status"] in ("downloading", "verifying"):
                store.put(
                    "downloads",
                    {
                        **item,
                        "status": "interrupted",
                        "error": "App closed. Download again to resume verified files.",
                    },
                )

    def catalog(self, refresh=False):
        with self.lock:
            if self.catalog_cache and not refresh:
                return self.catalog_cache
        try:
            info = fetch_json(f"{BASE}/api/models/{REPOSITORY}/revision/main")
            revision = revision_ok(info["sha"])
            rows = tree(revision)
        except (OSError, ValueError):
            cached = self.store.get("private", "hub-catalog") or snapshot()
            if not cached.get("revision"):
                raise
            return {
                **cached,
                "offline": True,
                "directory": (self.store.get("private", "hub-directory") or {}).get(
                    "value", str(self.store.directory / "models")
                ),
            }
        result = {
            "repository": REPOSITORY,
            "revision": revision,
            "models": [{"name": row["path"]} for row in rows if row["type"] == "directory"],
            "directory": (self.store.get("private", "hub-directory") or {}).get(
                "value", str(self.store.directory / "models")
            ),
        }
        with self.lock:
            self.catalog_cache = result
        self.store.put("private", {**result, "id": "hub-catalog"})
        return result

    def packages(self, model, revision):
        revision_ok(revision)
        if (
            not isinstance(model, str)
            or not re.fullmatch(r"[\w.-]+", model)
            or model in (".", "..")
        ):
            raise ValueError("Invalid model folder.")
        key = (model, revision)
        with self.lock:
            if key in self.package_cache:
                return self.package_cache[key]
        rows = [row for row in tree(revision, model, recursive=True) if row["type"] == "file"]
        config_rows = [
            row
            for row in rows
            if PurePosixPath(row["path"]).name in ("genai_config.json", "embedding_config.json")
        ]
        packages = []
        for config_row in config_rows:
            folder = str(PurePosixPath(config_row["path"]).parent)
            if "onnx" not in folder.lower():
                continue
            config = fetch_json(resolve_url(revision, config_row["path"]))
            native = config.get("model", {})
            # Whisper requires encoder audio features, outside Kite's chat input contract.
            if native.get("type") == "whisper":
                continue
            modalities = ["text"]
            if config.get("task") == "multimodal-embedding":
                modalities = config.get("modalities", modalities)
            else:
                if "vision" in native:
                    modalities.append("image")
                if "speech" in native or "audio" in native:
                    modalities.append("audio")
            files = []
            for row in rows:
                if not row["path"].startswith(folder + "/"):
                    continue
                name = row["path"][len(folder) + 1 :]
                # Data/configuration only; never download repository executable code.
                if PurePosixPath(name).name not in ("LICENSE", "NOTICE") and not re.search(
                    r"\.(?:onnx(?:[._][\w-]+)*|data(?:[._]\d+)?|json|txt|jinja|model|md|tiktoken|vocab|merges)$",
                    name,
                    re.I,
                ):
                    continue
                safe_path(Path.cwd(), name)
                files.append(
                    {
                        "name": name,
                        "remote": row["path"],
                        "size": row["size"],
                        "oid": row["oid"],
                        "sha256": row.get("lfs", {}).get("oid"),
                    }
                )
            if not any(f["name"] == "tokenizer.json" for f in files):
                continue
            packages.append(
                {
                    "id": hashlib.sha256(f"{folder}@{revision}".encode()).hexdigest()[:16],
                    "name": model,
                    "path": folder,
                    "variant": folder[len(model) + 1 :],
                    "revision": revision,
                    "size": sum(f["size"] for f in files),
                    "files": files,
                    "modalities": modalities,
                    "kind": "embedding" if "task" in config else "chat",
                    "architecture": native.get("type", config.get("architecture", "unknown")),
                    "source_url": f"{BASE}/{REPOSITORY}/tree/{revision}/{folder}",
                }
            )
        with self.lock:
            self.package_cache[key] = packages
        return packages

    def start(self, package, revision, directory, model_id=None):
        if not isinstance(package, str) or "/" not in package:
            raise ValueError("Select a model package.")
        plan = next(
            (p for p in self.packages(package.split("/")[0], revision) if p["path"] == package),
            None,
        )
        if not plan:
            raise ValueError("This is not an importable ONNX package.")
        if model_id:
            model = next(
                (m for m in library_models(self.store.list("models")) if m["id"] == model_id), None
            )
            remote = {
                "name": plan["name"],
                "kind": plan.get("kind", "chat"),
                "modalities": plan.get("modalities", ["text"]),
            }
            if not model or not matches(model, remote):
                raise ValueError("The download does not match the selected model.")
            if model.get("available"):
                raise ValueError("This model is already downloaded.")
        if not isinstance(directory, str) or not directory.strip():
            raise ValueError("Choose a download directory.")
        root = Path(directory).expanduser().resolve()
        target = safe_path(root, re.sub(r"[^\w.-]", "-", package) + "-" + revision[:10])
        staging = safe_path(root, ".kite-downloads/" + plan["id"])
        if target.exists():
            provenance = read_json(target / "kite-hub-source.json")
            if provenance.get("revision") != revision or provenance.get("path") != package:
                raise ValueError(
                    "The destination already exists and belongs to another package. Choose another directory."
                )
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError("Wait for the current download or cancel it first.")
            root.mkdir(parents=True, exist_ok=True)
            if (
                shutil.disk_usage(root).free < plan["size"] + 64 * 1024 * 1024
                and not target.exists()
            ):
                raise ValueError(
                    "Not enough free disk space for this package. Choose another drive."
                )
            self.stop_event = threading.Event()
            job = self.store.put(
                "downloads",
                {
                    "id": plan["id"],
                    "restore_model": model_id,
                    "name": plan["name"],
                    "variant": plan["variant"],
                    "path": plan["path"],
                    "revision": revision,
                    "directory": str(target),
                    "status": "downloading",
                    "total_bytes": plan["size"],
                    "downloaded_bytes": 0,
                    "file": "",
                    "speed": 0,
                    "error": None,
                },
            )
            self.store.put("private", {"id": "hub-directory", "value": str(root)})
            self.thread = threading.Thread(
                target=self._download,
                args=(job, plan, root, staging, target, self.stop_event),
                daemon=True,
            )
            self.thread.start()
            return job

    def cancel(self, identifier=None):
        if identifier:
            item = self.store.get("downloads", identifier)
            if not item or item["status"] not in ("downloading", "verifying"):
                return {"cancelled": False}
        self.stop_event.set()
        return {"cancelled": True}

    def _download(self, job, plan, root, staging, target, stop):
        started, written, last_report = time.monotonic(), 0, 0.0
        try:
            files_root = target if target.exists() else safe_path(staging, "files")
            files_root.mkdir(parents=True, exist_ok=True)
            for spec in plan["files"]:
                if stop.is_set():
                    raise Cancelled()
                final = safe_path(files_root, spec["name"])
                final.parent.mkdir(parents=True, exist_ok=True)
                job.update(file=spec["name"], status="verifying")
                self.store.put("downloads", job)
                if verify_file(final, spec):
                    job["downloaded_bytes"] += spec["size"]
                    continue
                if target.exists():
                    raise ValueError(
                        f"Existing file failed verification: {spec['name']}. Choose a fresh download directory."
                    )
                part = safe_path(files_root, spec["name"] + ".part")
                offset = part.stat().st_size if part.exists() else 0
                if offset == spec["size"] and verify_file(part, spec):
                    part.replace(final)
                    job["downloaded_bytes"] += spec["size"]
                    continue
                if offset > spec["size"]:
                    offset = 0
                job["status"] = "downloading"
                with open_url(
                    resolve_url(plan["revision"], spec["remote"]),
                    {"Range": f"bytes={offset}-"} if offset else {},
                ) as response:
                    if offset and response.status == 206:
                        if not response.headers.get("Content-Range", "").startswith(
                            f"bytes {offset}-"
                        ):
                            raise ValueError("The server returned an unexpected resume range.")
                    elif response.status == 200:
                        offset = 0
                    else:
                        raise ValueError(f"Unexpected download response: {response.status}")
                    completed_before = job["downloaded_bytes"]
                    job["downloaded_bytes"] += offset
                    with part.open("ab" if offset else "wb") as stream:
                        while True:
                            if stop.is_set():
                                raise Cancelled()
                            chunk = response.read1(256 * 1024)
                            if not chunk:
                                break
                            stream.write(chunk)
                            offset += len(chunk)
                            written += len(chunk)
                            if offset > spec["size"]:
                                raise ValueError(
                                    "Download exceeds the repository's declared file size."
                                )
                            job["downloaded_bytes"] = completed_before + offset
                            now = time.monotonic()
                            if now - last_report > 0.5:
                                job["speed"] = written / max(now - started, 0.01)
                                self.store.put("downloads", job)
                                last_report = now
                job["status"] = "verifying"
                self.store.put("downloads", job)
                if not verify_file(part, spec):
                    # A subsequent retry replaces an invalid complete partial.
                    with part.open("wb"):
                        pass
                    raise ValueError(
                        f"Checksum verification failed: {spec['name']}. Retry the download."
                    )
                part.replace(final)
            if stop.is_set():
                raise Cancelled()
            if not target.exists():
                (files_root / "kite-model.json").write_text(
                    json.dumps(
                        {
                            "name": plan["name"],
                            "source": {
                                "repository": REPOSITORY,
                                "revision": plan["revision"],
                                "path": plan["path"],
                            },
                        },
                        indent=2,
                    )
                )
                (files_root / "kite-hub-source.json").write_text(
                    json.dumps(
                        {
                            "repository": REPOSITORY,
                            "revision": plan["revision"],
                            "path": plan["path"],
                            "files": plan["files"],
                        },
                        indent=2,
                    )
                )
                inspect_package(files_root)
                if not files_root.resolve().is_relative_to(
                    root
                ) or not target.resolve().is_relative_to(root):
                    raise ValueError("Download destination escaped the selected directory.")
                files_root.rename(target)
            model = register_download(
                self.store, inspect_package(target), plan, job.get("restore_model")
            )
            job.update(
                status="completed", model=model["id"], downloaded_bytes=plan["size"], speed=0
            )
        except Cancelled:
            job.update(
                status="cancelled",
                error="Download stopped. Verified files and partial data are retained for resume.",
                speed=0,
            )
        except Exception as exc:
            job.update(status="failed", error=str(exc), speed=0)
        finally:
            self.store.put("downloads", job)

    def shutdown(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=25)
