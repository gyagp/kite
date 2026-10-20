"""Serve Evo's local workbench and a serialized queue of CLI operations."""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from evo.contrib import catalog
from evo.paths import asset_root
from evo.workbench import Workbench, read_json


class Jobs:
    def __init__(self, workbench):
        self.workbench = workbench
        self.root = workbench.directory / "jobs"
        self.root.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.pending = queue.Queue()
        self.active = {}
        self.records = {}
        for path in self.root.glob("*/job.json"):
            job = read_json(path)
            if job["status"] in {"running", "queued"}:
                job.update(status="interrupted", error="Workbench restarted; rerun explicitly")
            self.records[job["id"]] = job
            self.save(job)
        threading.Thread(target=self.worker, daemon=True).start()

    def save(self, job):
        path = self.root / job["id"] / "job.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(job, indent=2), encoding="utf-8")
        temp.replace(path)

    def create(self, data):
        operation = data.get("operation")
        if operation not in {
            "inspect",
            "analyze",
            "optimize",
            "build",
            "benchmark",
            "profile",
            "validate-cpu",
        }:
            raise ValueError("Unknown operation")
        identifier = uuid.uuid4().hex[:12]
        folder = self.root / identifier
        output = self.workbench.root / "artifacts/ui-outputs" / identifier
        report = folder / "result.json"
        command = [sys.executable, "-u", "-m", "evo", operation]
        model = None
        if operation != "build":
            model = self.workbench.record(data.get("model_id"))
            path = Path(model["path"])
        else:
            source = str(data.get("source", "")).strip()
            if not source:
                raise ValueError("A source model ID or checkpoint path is required")
            backend = data.get("backend", "genai")
            if backend not in {"genai", "mobius"}:
                raise ValueError("Invalid exporter")
            precision = data.get("precision", "int4")
            if precision not in {"int4", "int8", "fp16", "fp32"}:
                raise ValueError("Invalid precision")
            command += [
                "--config" if data.get("source_kind") == "local" else "--model",
                source,
                "--output",
                str(output),
                "--backend",
                backend,
                "--precision",
                precision,
            ]
            options = data.get("options", {})
            if not isinstance(options, dict):
                raise ValueError("Advanced exporter options must be a JSON object")
            command += [
                "--genai-options" if backend == "genai" else "--mobius-options",
                json.dumps(options),
            ]

        def file_arg(name):
            value = str(data.get(name, "")).strip()
            if not value:
                raise ValueError(f"{name} is required")
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = self.workbench.root / candidate
            if not candidate.is_file():
                raise ValueError(f"File not found: {candidate}")
            return str(candidate.resolve())

        if operation in {"inspect", "analyze"}:
            command += [
                str(path),
                "--report" if operation == "inspect" else "--output",
                str(report),
            ]
        elif operation == "optimize":
            # Only copy a package when the user explicitly chooses package mode.
            source = (
                path.parent
                if data.get("package", True) and (path.parent / "genai_config.json").exists()
                else path
            )
            command += [str(source), "--output", str(output)]
            if data.get("contrib", False):
                command.append("--contrib")
        elif operation == "benchmark":
            if not (path.parent / "genai_config.json").exists():
                raise ValueError(
                    "Token generation requires a GenAI package; use Profile for arbitrary ONNX"
                )
            repeats, tokens = int(data.get("repeats", 3)), int(data.get("tokens", 32))
            if not 1 <= repeats <= 20 or not 2 <= tokens <= 2048:
                raise ValueError("Use 1–20 repeats and 2–2048 generated tokens")
            command += [
                str(path.parent),
                "--cases",
                file_arg("cases"),
                "--output",
                str(report),
                "--repeats",
                str(repeats),
                "--new-tokens",
                str(tokens),
            ]
        elif operation == "profile":
            command += [str(path), "--inputs", file_arg("inputs"), "--output", str(output)]
        elif operation == "validate-cpu":
            command += [
                str(path),
                "--inputs",
                file_arg("inputs"),
                "--expected",
                file_arg("expected"),
                "--report",
                str(report),
            ]
        folder.mkdir()
        output.parent.mkdir(parents=True, exist_ok=True)
        job = {
            "id": identifier,
            "operation": operation,
            "model_name": model["name"] if model else source,
            "model_id": model["id"] if model else None,
            "status": "queued",
            "created": time.time(),
            "command": command,
            "output": str(output),
            "report": str(report),
            "progress": "Waiting",
        }
        with self.lock:
            self.records[identifier] = job
            self.save(job)
        self.pending.put(identifier)
        return job

    def cancel(self, identifier):
        with self.lock:
            job = self.records[identifier]
            if job["status"] not in {"queued", "running"}:
                return job
            job["status"] = "cancelled"
            process = self.active.get(identifier)
            if process and process.poll() is None:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                else:
                    process.terminate()
            self.save(job)
            return job

    def worker(self):
        while True:
            identifier = self.pending.get()
            job = self.records[identifier]
            try:
                with self.lock:
                    if job["status"] == "cancelled":
                        continue
                    job.update(status="running", started=time.time(), progress="Running")
                    self.save(job)
                log_path = self.root / identifier / "output.log"
                with log_path.open("wb") as log:
                    with self.lock:
                        if job["status"] == "cancelled":
                            continue
                        process = subprocess.Popen(
                            job["command"],
                            cwd=self.workbench.root,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                        )
                        self.active[identifier] = process
                    code = process.wait()
                with self.lock:
                    self.active.pop(identifier, None)
                    job.update(finished=time.time(), exit_code=code)
                    if job["status"] != "cancelled":
                        job["status"] = "completed" if code == 0 else "failed"
                    if (
                        code == 0
                        and job["status"] == "completed"
                        and job["operation"] in {"build", "optimize"}
                    ):
                        model = self.workbench.add(
                            job["output"],
                            f"{job['model_name']} · {job['operation']}",
                            origin={
                                "operation": job["operation"],
                                "parent_id": job["model_id"],
                                "job_id": identifier,
                                "created": time.time(),
                            },
                        )
                        job["result_model_id"] = model["id"]
                    job["progress"] = job["status"].capitalize()
                    self.save(job)
            except Exception as exc:
                with self.lock:
                    job.update(status="failed", error=str(exc), finished=time.time())
                    self.active.pop(identifier, None)
                    self.save(job)
            finally:
                self.pending.task_done()

    def list(self):
        with self.lock:
            return sorted([dict(j) for j in self.records.values()], key=lambda j: -j["created"])

    def detail(self, identifier):
        with self.lock:
            if identifier not in self.records:
                raise ValueError("Unknown run")
            job = dict(self.records[identifier])
        log = self.root / identifier / "output.log"
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 40000))
                job["log"] = stream.read().decode("utf-8", errors="replace").replace("\x00", "")
        else:
            job["log"] = "Waiting for output…"
        candidates = [
            Path(job["report"]),
            Path(job["output"]) / "optimization.json",
            Path(job["output"]) / "profile-info.json",
            Path(job["output"]) / "evo-manifest.json",
        ]
        job["result"] = None
        for path in candidates:
            if path.is_file():
                try:
                    job["result"] = read_json(path)
                    break
                except (ValueError, OSError):
                    pass
        return job


def make_server(root, port=8872):
    workbench = Workbench(root)
    jobs = Jobs(workbench)
    assets = Path(__file__).parent / "web"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, data, status=200, content_type="application/json"):
            if content_type == "application/json":
                data = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
            )
            self.end_headers()
            self.wfile.write(data)

        def allowed(self):
            allowed = {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }
            if self.headers.get("Host") not in allowed:
                self.send({"error": "Local connections only"}, 403)
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://{host}" for host in allowed}:
                self.send({"error": "Cross-origin requests are not allowed"}, 403)
                return False
            return True

        def do_GET(self):
            if not self.allowed():
                return
            url = urlparse(self.path)
            params = parse_qs(url.query)
            try:
                if url.path in {"/", "/app.js", "/style.css"}:
                    name = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}[
                        url.path
                    ]
                    mime = {
                        "index.html": "text/html",
                        "app.js": "text/javascript",
                        "style.css": "text/css",
                    }[name]
                    return self.send((assets / name).read_bytes(), content_type=mime)
                if url.path == "/api/models":
                    return self.send(workbench.summaries())
                if url.path.startswith("/api/models/"):
                    return self.send(workbench.details(url.path.split("/")[-1]))
                if url.path == "/api/catalog":
                    return self.send(catalog())
                if url.path == "/api/fixtures":
                    return self.send(workbench.fixtures())
                if url.path == "/api/wishlist":
                    return self.send(workbench.wishlist)
                if url.path == "/api/performance":
                    return self.send(workbench.performance_records())
                if url.path == "/api/compare":
                    return self.send(
                        workbench.compare(
                            params["a"][0], params["b"][0], params.get("group", [None])[0]
                        )
                    )
                if url.path == "/api/jobs":
                    return self.send(jobs.list())
                if url.path.startswith("/api/jobs/"):
                    return self.send(jobs.detail(url.path.split("/")[-1]))
                if url.path == "/api/status":
                    return self.send(
                        {
                            "root": str(workbench.root),
                            "runtime": "Native ORT WebGPU",
                            "queue": "One operation at a time",
                            "version": "0.2.0",
                        }
                    )
                self.send({"error": "Not found"}, 404)
            except (ValueError, KeyError, OSError) as exc:
                self.send({"error": str(exc)}, 400)
            except Exception as exc:
                self.send({"error": str(exc)}, 500)

        def do_POST(self):
            if not self.allowed():
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 262144:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")
                if self.path == "/api/models":
                    return self.send(workbench.add(data["path"], data.get("name")), 201)
                if self.path == "/api/performance":
                    return self.send(workbench.save_performance(data), 201)
                if self.path == "/api/models/remove":
                    workbench.remove(data["id"])
                    return self.send({"ok": True})
                if self.path == "/api/jobs":
                    return self.send(jobs.create(data), 201)
                if self.path == "/api/wishlist":
                    return self.send(workbench.save_wish(data), 201)
                if self.path == "/api/jobs/cancel":
                    return self.send(jobs.cancel(data["id"]))
                self.send({"error": "Not found"}, 404)
            except (ValueError, KeyError, OSError, TypeError) as exc:
                self.send({"error": str(exc)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.workbench = workbench
    server.jobs = jobs
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8872)
    parser.add_argument("--root", type=Path, default=asset_root())
    args = parser.parse_args(argv)
    server = make_server(args.root, args.port)
    print(f"Evo workbench: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
