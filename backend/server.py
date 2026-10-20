"""Loopback-only HTTP/SSE service and static frontend host. No web dependencies."""

import argparse
import base64
import binascii
import json
import mimetypes
import os
import re
import secrets
import signal
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .catalog import discover, inspect_package
from .examples import example_asset, list_examples, load_example
from .lab import COMMANDS, image_readiness, image_settings
from .service import PROJECT, BusyError, Service

MAX_BODY = 90 * 1024 * 1024


def upload(service, payload):
    name = Path(str(payload.get("name", "attachment"))).name[:160]
    try:
        raw = base64.b64decode(payload.get("data", ""), validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("Invalid attachment encoding.")
    if not raw or len(raw) > 64 * 1024 * 1024:
        raise ValueError("Attachments must be between 1 byte and 64 MB.")
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        kind, mime, suffix = "image", "image/png", ".png"
    elif raw.startswith(b"\xff\xd8\xff"):
        kind, mime, suffix = "image", "image/jpeg", ".jpg"
    elif raw.startswith(b"RIFF") and raw[8:12] == b"WAVE":
        kind, mime, suffix = "audio", "audio/wav", ".wav"
    elif raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        kind, mime, suffix = "image", "image/webp", ".webp"
    elif raw.startswith(b"fLaC"):
        kind, mime, suffix = "audio", "audio/flac", ".flac"
    elif raw.startswith(b"ID3") or (len(raw) > 2 and raw[0] == 0xFF and raw[1] & 0xE0 == 0xE0):
        kind, mime, suffix = "audio", "audio/mpeg", ".mp3"
    elif len(raw) >= 16 and raw[4:8] == b"ftyp":
        kind, mime, suffix = "video", "video/mp4", ".mp4"
    else:
        raise ValueError(
            "Supported media files: PNG, JPEG, WebP, WAV, FLAC, MP3, and MP4/MOV. For documents, use the text import in Embeddings."
        )
    identifier = uuid.uuid4().hex
    folder = service.store.directory / "attachments"
    folder.mkdir(exist_ok=True)
    path = folder / (identifier + suffix)
    path.write_bytes(raw)
    item = service.store.put(
        "attachments",
        {
            "id": identifier,
            "name": name,
            "kind": kind,
            "mime": mime,
            "size": len(raw),
            "path": str(path),
        },
    )
    return {k: v for k, v in item.items() if k != "path"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def service(self):
        return self.server.service

    def log_message(self, fmt, *args):
        if args and str(args[0]).startswith(("GET /api/status", "GET /api/jobs/")):
            return
        super().log_message(fmt, *args)

    def headers_common(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cache-Control", "no-store")

    def json(self, value, status=200):
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.headers_common()
        self.end_headers()
        self.wfile.write(raw)

    def body(self):
        if self.headers.get_content_type() != "application/json":
            raise ValueError("Requests must use application/json.")
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= MAX_BODY:
            raise ValueError("Invalid request size (maximum 90 MB).")
        self.connection.settimeout(30)
        try:
            data = json.loads(self.rfile.read(length))
        finally:
            self.connection.settimeout(None)
        if not isinstance(data, dict):
            raise ValueError("The request body must be an object.")
        return data

    def allowed(self):
        # Prevent DNS rebinding and cross-origin requests to the local runtime.
        host = self.headers.get("Host", "")
        allowed_hosts = {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }
        if host not in allowed_hosts:
            self.json({"error": "Invalid loopback host."}, 403)
            return False
        origin = self.headers.get("Origin")
        allowed_origins = {f"http://{h}" for h in allowed_hosts} | {
            "http://127.0.0.1:5175",
            "http://localhost:5175",
        }
        if origin and origin not in allowed_origins:
            self.json({"error": "Cross-origin access is disabled."}, 403)
            return False
        if self.path.startswith("/v1/"):
            token = self.headers.get("Authorization", "")
            if not secrets.compare_digest(token, "Bearer " + self.service.api_key["value"]):
                self.json(
                    {
                        "error": {
                            "message": "Use the local API key from Kite Settings.",
                            "type": "authentication_error",
                        }
                    },
                    401,
                )
                return False
        return True

    def do_GET(self):
        self.dispatch("GET")

    def do_POST(self):
        self.dispatch("POST")

    def do_DELETE(self):
        self.dispatch("DELETE")

    def dispatch(self, method):
        try:
            if not self.allowed():
                return
            self.route(method, unquote(urlparse(self.path).path))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except BusyError as exc:
            self.json({"error": str(exc)}, 409)
        except (ValueError, KeyError, TypeError) as exc:
            self.close_connection = True
            self.json({"error": str(exc)}, 400)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            self.json({"error": str(exc)}, 500)

    def route(self, method, path):
        service = self.service
        if path == "/api/images/config":
            if method == "POST":
                if service.active:
                    raise BusyError("Stop the current operation before changing image settings.")
                data = self.body()
                config = {
                    **image_settings(service),
                    **{
                        k: data[k]
                        for k in ("source", "package", "library", "python", "dependencies")
                        if k in data
                    },
                }
                if not isinstance(config["dependencies"], list) or any(
                    not isinstance(p, str) for p in config["dependencies"]
                ):
                    raise ValueError("Dependencies must be a list of local directories.")
                missing = image_readiness(config)
                if missing:
                    raise ValueError("Missing files: " + ", ".join(missing))
                service.store.put("preferences", {**config, "id": "image-runtime"})
            config = image_settings(service)
            return self.json(
                {
                    "settings": config,
                    "missing": image_readiness(config),
                    "output_directory": str(service.image_directory),
                }
            )
        if method == "GET" and path.startswith("/api/profiles/"):
            parts = path.split("/")
            record = service.store.get("profiles", parts[-2]) if len(parts) == 5 else None
            if not record or parts[-1] not in record["files"]:
                raise ValueError("Profile file not found.")
            return self.file(Path(record["directory"]) / parts[-1], download=True)
        if method == "GET" and path == "/api/images":
            return self.json(service.store.list("images"))
        if method == "GET" and path == "/api/images/presets":
            return self.json(
                json.loads((PROJECT / "demo/image-studio/presets.json").read_text(encoding="utf-8"))
            )
        if method == "GET" and path.startswith("/api/images/presets/"):
            folder = PROJECT / "demo/image-studio"
            asset = (folder / path[len("/api/images/presets/") :]).resolve()
            if not asset.is_relative_to(folder.resolve()) or asset.suffix not in (
                ".png",
                ".jpg",
                ".json",
            ):
                raise ValueError("Invalid preset asset.")
            return self.file(asset)
        if method == "GET" and path.startswith("/api/images/"):
            parts = path.split("/")
            if len(parts) != 5 or parts[-1] not in (
                "image.png",
                "result.json",
                "profile.json",
                "prefix-trace.json",
                "denoise-trace.json",
            ):
                raise ValueError("Unknown image asset.")
            record = service.store.get("images", parts[-2])
            if not record:
                raise ValueError("Image not found.")
            return self.file(
                Path(record["path"]).parent / parts[-1], download=parts[-1].endswith("-trace.json")
            )
        if method == "GET" and path == "/api/lab/catalog":
            return self.json(
                {
                    "commands": COMMANDS,
                    "schemas": json.loads(
                        (PROJECT / "evo_resources/tools.json").read_text(encoding="utf-8")
                    ),
                    "engine": str(PROJECT / "evo"),
                    "resources": str(PROJECT / "evo_resources"),
                    "recipes": [
                        str(p) for p in sorted((PROJECT / "evo_resources/recipes").glob("*.toml"))
                    ],
                    "evidence_root": service.settings["asset_root"],
                }
            )
        if method == "GET" and path.startswith("/api/lab/jobs/"):
            relative = path[len("/api/lab/jobs/") :]
            folder = (service.store.directory / "lab-jobs").resolve()
            file = (folder / relative).resolve()
            if not file.is_relative_to(folder):
                raise ValueError("Invalid job artifact path.")
            return self.file(file, download=True)
        if path.startswith("/api/evidence/"):
            wb = service.evidence.get()
            action = path[len("/api/evidence/") :]
            if method == "GET":
                if action == "models":
                    return self.json(wb.models)
                if action == "performance":
                    return self.json(wb.performance_records())
                if action == "wishlist":
                    return self.json(wb.wishlist)
                if action == "fixtures":
                    return self.json(wb.fixtures())
                if action == "catalog":
                    if hasattr(wb, "call"):
                        return self.json(wb.call("catalog"))
                    from evo.contrib import catalog

                    return self.json(catalog())
                if action.startswith("models/"):
                    return self.json(wb.details(action.split("/")[-1]))
            if method == "POST":
                data = self.body()
                if action == "models":
                    return self.json(wb.add(data["path"], data.get("name")))
                if action == "models/remove":
                    wb.remove(data["id"])
                    return self.json({"removed": True})
                if action == "performance":
                    return self.json(wb.save_performance(data))
                if action == "wishlist":
                    return self.json(wb.save_wish(data))
                if action == "compare":
                    return self.json(wb.compare(data["a"], data["b"], data.get("group")))
            raise ValueError("Unknown workbench operation.")
        if method == "GET" and path == "/api/health":
            return self.json({"app": "kite", "version": "0.2.0", "pid": os.getpid()})
        if method == "GET" and path == "/api/bootstrap":
            return self.json(
                {
                    "models": service.models(),
                    "settings": service.settings,
                    "chat_preferences": service.chat_preferences,
                    "status": service.status(),
                    "threads": service.store.list("threads"),
                    "api_key": service.api_key["value"],
                    "api_url": f"http://127.0.0.1:{self.server.server_port}/v1",
                    "data_directory": str(service.store.directory),
                }
            )
        if method == "GET" and path == "/api/status":
            return self.json(service.status())
        if method == "GET" and path == "/api/models":
            return self.json(service.models())
        if method == "GET" and path == "/api/examples":
            return self.json(list_examples(PROJECT))
        if method == "GET" and path.startswith("/api/examples/assets/"):
            return self.file(example_asset(PROJECT, path[len("/api/examples/assets/") :]))
        if method == "GET" and path == "/api/examples/credits":
            return self.json(
                json.loads((PROJECT / "demo/showcase/credits.json").read_text(encoding="utf-8"))
            )
        if method == "GET" and path == "/api/preferences/chat":
            return self.json(service.chat_preferences)
        if method == "POST" and path == "/api/preferences/chat":
            return self.json(service.save_chat_preferences(self.body()))
        if method == "POST" and path.startswith("/api/examples/"):
            self.body()
            return self.json(load_example(PROJECT, service.store, path.rsplit("/", 1)[1]))
        if method == "GET" and path == "/api/hub/catalog":
            return self.json(service.hub.catalog())
        if method == "POST" and path == "/api/hub/refresh":
            self.body()
            return self.json(service.hub.catalog(refresh=True))
        if method == "POST" and path == "/api/hub/packages":
            data = self.body()
            return self.json(service.hub.packages(data.get("model"), data.get("revision")))
        if method == "GET" and path == "/api/hub/downloads":
            return self.json(service.store.list("downloads"))
        if method == "POST" and path == "/api/hub/downloads":
            data = self.body()
            return self.json(
                service.hub.start(
                    data.get("path"),
                    data.get("revision"),
                    data.get("directory"),
                    data.get("model_id"),
                ),
                202,
            )
        if method == "POST" and path.startswith("/api/hub/cancel/"):
            self.body()
            return self.json(service.hub.cancel(path.rsplit("/", 1)[1]))
        if method == "POST" and path == "/api/models/import":
            package = inspect_package(self.body()["path"])
            existing = next(
                (
                    m
                    for m in service.models()
                    if Path(m["path"]).resolve() == Path(package["path"]).resolve()
                ),
                None,
            )
            if existing:
                package["id"] = existing["id"]
            return self.json(service.store.put("models", package))
        if method == "POST" and path == "/api/models/scan":
            found, errors = discover(self.body()["path"])
            return self.json({"models": found, "errors": errors})
        if method == "POST" and path.startswith("/api/models/") and path.endswith("/delete"):
            data = self.body()
            if type(data.get("delete_files")) is not bool:
                raise ValueError("Choose whether to delete the model files.")
            return self.json(
                service.delete_model(
                    path.split("/")[-2], data["delete_files"], data.get("confirm_path")
                )
            )
        if method == "DELETE" and path.startswith("/api/models/"):
            identifier = path.rsplit("/", 1)[1]
            return self.json(service.delete_model(identifier))
        if method == "POST" and path == "/api/settings":
            return self.json(service.save_settings(self.body()))
        if method == "GET" and path == "/api/runs":
            return self.json(service.store.list("runs"))
        if method == "GET" and path == "/api/diagnostics":
            return self.json(
                {
                    "settings": service.settings,
                    "status": service.status(),
                    "log": list(service.stderr),
                }
            )
        if method == "GET" and path == "/api/threads":
            return self.json(service.store.list("threads"))
        if path.startswith("/api/threads/"):
            identifier = path.rsplit("/", 1)[1]
            if method == "DELETE":
                if service.active:
                    raise BusyError(
                        "Wait for the current operation before deleting a conversation."
                    )
                service.store.delete("threads", identifier)
                return self.json({"removed": True})
            if method == "POST":
                item = service.store.get("threads", identifier)
                if not item:
                    return self.json({"error": "Conversation not found."}, 404)
                title = self.body().get("title", "").strip()
                if not title or len(title) > 100:
                    raise ValueError("Titles must contain 1–100 characters.")
                item["title"] = title
                return self.json(service.store.put("threads", item))
        if method == "POST" and path == "/api/attachments":
            return self.json(upload(service, self.body()))
        if method == "GET" and path.startswith("/api/attachments/"):
            item = service.store.get("attachments", path.rsplit("/", 1)[1])
            if not item:
                return self.json({"error": "Attachment not found."}, 404)
            return self.file(Path(item["path"]), item["mime"])
        if method == "POST" and path == "/api/jobs":
            job = service.submit(self.body())
            return self.json({"id": job.id}, 202)
        if method == "POST" and path == "/api/stop":
            self.body()
            return self.json(service.stop())
        if method == "POST" and path == "/api/shutdown":
            self.body()
            service.stop()
            service.hub.cancel()
            self.json({"stopping": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if method == "GET" and path.startswith("/api/jobs/"):
            parts = path.strip("/").split("/")
            job = service.jobs.get(parts[2])
            if not job:
                return self.json({"error": "Job not found. See Activity for completed runs."}, 404)
            if len(parts) == 4 and parts[3] == "events":
                return self.events(job)
            return self.json(job.snapshot())
        if method == "GET" and path == "/v1/models":
            return self.json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": m["id"],
                            "object": "model",
                            "created": 0,
                            "owned_by": "local",
                            "name": m["name"],
                            "capabilities": m["modalities"],
                            "kind": m["kind"],
                        }
                        for m in service.models()
                    ],
                }
            )
        if method == "POST" and path == "/v1/embeddings":
            data = self.body()
            if data.get("encoding_format", "float") != "float":
                raise ValueError("Only encoding_format=float is supported.")
            texts = (
                [data["input"]] if isinstance(data.get("input"), (str, dict)) else data.get("input")
            )
            job = service.submit(
                {
                    "kind": "embed",
                    "model": data.get("model"),
                    "samples": texts,
                    "dimensions": data.get("dimensions", 768),
                    "task": data.get("task", "SearchQuery"),
                }
            )
            self.wait_job(job)
            if job.status != "completed":
                return self.json({"error": {"message": job.error, "type": "inference_error"}}, 500)
            return self.json(
                {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": i, "embedding": v}
                        for i, v in enumerate(job.result["vectors"])
                    ],
                    "model": job.model,
                    "usage": {
                        "prompt_tokens": job.result["tokens"],
                        "total_tokens": job.result["tokens"],
                    },
                }
            )
        if method == "POST" and path == "/v1/chat/completions":
            data = self.body()
            for message in data.get("messages", []):
                if isinstance(message.get("content"), list):
                    texts, attachments = [], []
                    for part in message["content"]:
                        if part.get("type") == "text":
                            texts.append(part["text"])
                        elif part.get("type") == "image_url":
                            url = part.get("image_url", {}).get("url", "")
                            if not url.startswith(
                                ("data:image/png;base64,", "data:image/jpeg;base64,")
                            ):
                                raise ValueError(
                                    "Images must be local PNG/JPEG base64 data URLs; remote URLs are not fetched."
                                )
                            attachments.append(
                                upload(
                                    service,
                                    {"name": "API image", "data": url.split(",", 1)[1]},
                                )
                            )
                        elif (
                            part.get("type") == "input_audio"
                            and part.get("input_audio", {}).get("format") == "wav"
                        ):
                            attachments.append(
                                upload(
                                    service,
                                    {
                                        "name": "API audio.wav",
                                        "data": part["input_audio"]["data"],
                                    },
                                )
                            )
                        else:
                            raise ValueError(
                                "Unsupported content part. Use text, image_url, or input_audio (wav)."
                            )
                    message["content"], message["attachments"] = (
                        "\n".join(texts),
                        attachments,
                    )
            if any(data.get(k) for k in ("tools", "tool_choice", "response_format", "stop")):
                raise ValueError(
                    "Tools, structured output, and custom stop sequences are not supported by this API yet."
                )
            job = service.submit({**data, "kind": "chat", "save": False})
            if data.get("stream"):
                return self.events(job, openai=True)
            self.wait_job(job)
            if job.status != "completed":
                return self.json({"error": {"message": job.error, "type": "inference_error"}}, 500)
            r = job.result
            return self.json(
                {
                    "id": "chatcmpl-" + job.id,
                    "object": "chat.completion",
                    "created": int(job.created),
                    "model": job.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": r["text"]},
                            "finish_reason": r["finish_reason"],
                        }
                    ],
                    "usage": {
                        "prompt_tokens": r["prompt_tokens"],
                        "completion_tokens": r["completion_tokens"],
                        "total_tokens": r["prompt_tokens"] + r["completion_tokens"],
                    },
                }
            )
        if method == "GET" and not path.startswith(("/api/", "/v1/")):
            root = PROJECT / "dist"
            target = (root / path.lstrip("/")).resolve()
            if not target.is_relative_to(root.resolve()):
                return self.json({"error": "Invalid path."}, 403)
            if not target.is_file():
                target = root / "index.html"
            if not target.is_file():
                return self.json(
                    {
                        "error": "Frontend not built. Run npm run build, or use the Vite development server on port 5175."
                    },
                    503,
                )
            return self.file(target)
        return self.json({"error": "Endpoint not found."}, 404)

    def file(self, path, mime=None, download=False):
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
            try:
                if not match or not any(match.groups()):
                    raise ValueError()
                left, right = match.groups()
                if left:
                    start = int(left)
                    end = min(int(right), size - 1) if right else size - 1
                else:
                    start = max(0, size - int(right))
                if start > end or start >= size:
                    raise ValueError()
                status = 206
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        self.send_response(status)
        self.send_header(
            "Content-Type",
            mime or mimetypes.guess_type(path)[0] or "application/octet-stream",
        )
        self.send_header("Content-Length", str(end - start + 1))
        if download:
            self.send_header("Content-Disposition", "attachment")
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.headers_common()
        if path.suffix == ".html":
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; object-src 'none'",
            )
        self.end_headers()
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = stream.read(min(remaining, 1024 * 1024))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def wait_job(self, job):
        with job.condition:
            job.condition.wait_for(
                lambda: any(e["type"] == "done" for e in job.events), timeout=650
            )

    def events(self, job, openai=False):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Connection", "close")
        self.headers_common()
        self.end_headers()
        self.close_connection = True
        index = 0
        try:
            while True:
                with job.condition:
                    if index >= len(job.events):
                        job.condition.wait(timeout=10)
                    events = job.events[index:]
                    index += len(events)
                if not events:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                for event in events:
                    value = event
                    if openai:
                        if event["type"] in ("delta", "result"):
                            value = {
                                "id": "chatcmpl-" + job.id,
                                "object": "chat.completion.chunk",
                                "created": int(job.created),
                                "model": job.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": event["text"]}
                                        if event["type"] == "delta"
                                        else {},
                                        "finish_reason": event["data"]["finish_reason"]
                                        if event["type"] == "result"
                                        else None,
                                    }
                                ],
                            }
                        elif event["type"] in ("error", "cancelled"):
                            value = {
                                "error": {
                                    "message": event["message"],
                                    "type": "inference_error",
                                }
                            }
                        elif event["type"] != "done":
                            continue
                    payload = (
                        "[DONE]"
                        if openai and event["type"] == "done"
                        else json.dumps(value, ensure_ascii=False)
                    )
                    self.wfile.write(("data: " + payload + "\n\n").encode("utf-8"))
                    self.wfile.flush()
                    if event["type"] == "done":
                        return
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # OpenAI clients disconnect to cancel; UI reconnects can replay events.
            if openai and job.status == "running":
                self.service.stop()


def create_server(port=4317, directory=None):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.service = Service(directory)
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=4317)
    parser.add_argument("--data-dir")
    parser.add_argument("--parent-pid", type=int, help="Exit when the owning desktop process exits")
    args = parser.parse_args()
    server = create_server(args.port, args.data_dir)
    if args.parent_pid and os.name == "nt":
        # Hold a real process handle so PID reuse cannot keep an orphan service alive.
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        parent = kernel.OpenProcess(0x100000, False, args.parent_pid)
        if not parent:
            raise RuntimeError("The owning desktop process is no longer running.")

        def watch_parent():
            kernel.WaitForSingleObject(parent, 0xFFFFFFFF)
            kernel.CloseHandle(parent)
            server.service.stop()
            server.shutdown()

        threading.Thread(target=watch_parent, daemon=True).start()

    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop)
    print(f"Kite ready at http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.service.stop()
        server.service.hub.shutdown()
        server.service.store.close()
        server.server_close()


if __name__ == "__main__":
    main()
