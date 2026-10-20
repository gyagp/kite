"""Job lifecycle, GPU worker supervision, and application state."""

import collections
import json
import os
import queue
import secrets
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .catalog import inspect_package, read_json
from .hub import Hub
from .lab import Evidence, run_image, run_tool, tool_request, validate_image
from .model_files import delete_package
from .paths import asset_root, migrate_store, workspace_root
from .process import ProcessGroup
from .store import Store
from .supported import library_models
from .timing import embedding_performance, search_embedding_metrics

PROJECT = Path(__file__).resolve().parent.parent


class BusyError(ValueError):
    pass


class CancelledError(Exception):
    pass


class Job:
    def __init__(self, kind, model):
        self.id = uuid.uuid4().hex
        self.kind, self.model = kind, model
        self.status, self.result, self.error = "running", None, None
        self.created = time.time()
        self.model_load_ms = 0.0
        self.events = []
        self.condition = threading.Condition()
        self.cancelled = threading.Event()

    def emit(self, event):
        with self.condition:
            self.events.append(event)
            self.condition.notify_all()

    def snapshot(self):
        return {
            "id": self.id,
            "kind": self.kind,
            "model": self.model,
            "status": self.status,
            "created": self.created,
            "result": self.result,
            "error": self.error,
        }


class Service:
    def __init__(self, directory=None):
        self.store = Store(directory or os.environ.get("KITE_DATA_DIR", workspace_root() / ".kite"))
        migrate_store(self.store)
        self.image_directory = Path(
            os.environ.get(
                "KITE_IMAGE_DIR",
                self.store.directory / "images"
                if directory
                else workspace_root() / "gitignore/images",
            )
        ).resolve()
        self.lock = threading.RLock()
        self.worker, self.output, self.loaded = None, None, None
        self.process_group = None
        self.active, self.jobs = None, {}
        self.stderr = collections.deque(maxlen=60)
        self.api_key = self.store.get("private", "api-key") or self.store.put(
            "private", {"id": "api-key", "value": secrets.token_urlsafe(32)}
        )
        self.hub = Hub(self.store)
        self.evidence = Evidence(self)
        if self.store.get("settings", "default") is None:
            evo = asset_root()
            runtime = read_json(evo / ".evo" / "runtime.json")
            dll = Path(runtime.get("webgpu_library", ""))
            if runtime and not dll.is_absolute():
                dll = (evo / ".evo" / dll).resolve()
            python = evo / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            self.store.put(
                "settings",
                {
                    "id": "default",
                    "asset_root": str(evo),
                    "python": str(python if python.is_file() else Path(sys.executable)),
                    "library": str(dll) if runtime else "",
                    "context": 8192,
                },
            )
            for name in ("qwen35-4b-gdn-mlp",):
                package = evo / "artifacts" / name
                if package.is_dir():
                    self.store.put("models", inspect_package(package))
        # Upgrade existing workspaces once without replacing user selections or
        # repeatedly re-adding a model the user intentionally removed later.
        if not self.store.get("private", "embeddinggemma2-multimodal-v1"):
            package = (
                Path(self.settings["asset_root"]) / "artifacts" / "embeddinggemma-2-multimodal-fp16"
            )
            if (package / "embedding_config.json").is_file():
                try:
                    self.store.put("models", inspect_package(package))
                    self.store.put(
                        "private", {"id": "embeddinggemma2-multimodal-v1", "imported": True}
                    )
                except (ValueError, OSError) as exc:
                    self.stderr.append(f"Multimodal package is not ready for import: {exc}")

        if not self.store.get("private", "qwen-image-v1"):
            package = (
                Path(self.settings["asset_root"]) / "artifacts/qwen-image-2.1-turbo-gguf-floatzp"
            )
            if (package / "image_config.json").is_file():
                try:
                    self.store.put("models", inspect_package(package))
                    self.store.put("private", {"id": "qwen-image-v1", "imported": True})
                except (ValueError, OSError) as exc:
                    self.stderr.append(f"Image package is not ready: {exc}")

    @property
    def settings(self):
        return self.store.get("settings", "default")

    @property
    def chat_preferences(self):
        return {"max_tokens": 4096, **(self.store.get("preferences", "chat") or {})}

    def save_chat_preferences(self, data):
        value = data.get("max_tokens")
        if type(value) is not int or not 1 <= value <= 4096:
            raise ValueError("Max new tokens must be an integer between 1 and 4096.")
        return self.store.put("preferences", {"id": "chat", "max_tokens": value})

    def status(self):
        with self.lock:
            if self.worker and self.worker.poll() is not None:
                self.loaded = None
            return {
                "loaded": self.loaded,
                "active": self.active.snapshot() if self.active else None,
                "worker_running": bool(self.worker and self.worker.poll() is None),
            }

    def models(self):
        return library_models(self.store.list("models"))

    def delete_model(self, identifier, delete_files=False, confirmation=None):
        with self.lock:
            if self.active:
                raise BusyError("Stop inference before deleting a model.")
            model = self.store.get("models", identifier)
            if not model:
                raise ValueError("Model not found.")
            image_package = getattr(self, "image_worker_key", (None,))[-1]
            owns_image_files = (
                self.loaded
                and self.loaded.get("kind") == "image"
                and image_package
                and Path(image_package).resolve() == Path(model["path"]).resolve()
            )
            if self.loaded and (self.loaded["model"] == identifier or owns_image_files):
                self._terminate()
            if delete_files:
                delete_package(
                    model,
                    confirmation,
                    project=PROJECT,
                    evo=self.settings["asset_root"],
                    data_directory=self.store.directory,
                    other_models=self.store.list("models"),
                )
                self.store.put(
                    "models", {**model, "available": False, "files_deleted_at": time.time()}
                )
                for download in self.store.list("downloads"):
                    if download.get("model") == identifier:
                        self.store.put(
                            "downloads",
                            {
                                **download,
                                "status": "not_downloaded",
                                "downloaded_bytes": 0,
                                "speed": 0,
                            },
                        )
            else:
                self.store.delete("models", identifier)
            return {
                "removed": not delete_files,
                "files_deleted": delete_files,
                "path": model["path"],
            }

    def save_settings(self, data):
        with self.lock:
            if self.active:
                raise BusyError("Stop the current operation before changing the runtime.")
            settings = {
                **self.settings,
                **{k: data[k] for k in ("python", "library", "asset_root", "context") if k in data},
            }
            for key in ("python", "library"):
                if not isinstance(settings[key], str) or not Path(settings[key]).is_file():
                    raise ValueError(f"The {key} path must point to an existing file.")
            if type(settings["context"]) is not int or not 256 <= settings["context"] <= 32768:
                raise ValueError("Context must be an integer between 256 and 32768.")
            self._terminate()
            self.evidence.instance = None
            return self.store.put("settings", settings)

    def _terminate(self):
        worker = self.worker
        self.worker, self.output, self.loaded = None, None, None
        if self.process_group:
            self.process_group.close()
            self.process_group = None
        if worker:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=5)
            for stream in (worker.stdin, worker.stdout, worker.stderr):
                if stream:
                    stream.close()

    def stop(self):
        with self.lock:
            if self.active:
                self.active.cancelled.set()
            self._terminate()
        return {"stopped": True}

    def _spawn(self, image_request=None):
        settings = self.settings
        if not Path(settings["python"]).is_file():
            raise ValueError("Configure the inference Python executable in Settings.")
        if not Path(settings["library"]).is_file():
            raise ValueError("Configure the source-built WebGPU DLL in Settings.")
        self.stderr.clear()
        worker = subprocess.Popen(
            (
                [image_request["python"], "-u", "-m", "backend.image_worker", "--serve"]
                if image_request
                else [settings["python"], "-u", str(PROJECT / "backend" / "native.py")]
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "KITE_ASSET_ROOT": str(asset_root()),
                "PYTHONPATH": str(PROJECT),
            },
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            self.process_group = ProcessGroup(worker)
        except Exception:
            worker.kill()
            worker.wait(timeout=5)
            raise
        output = queue.Queue()
        self.worker, self.output = worker, output

        def read_stdout():
            try:
                for line in worker.stdout:
                    if line.startswith("KITE_EVENT "):
                        output.put(json.loads(line[len("KITE_EVENT ") :]))
                    elif line.strip():
                        self.stderr.append(line.strip())
            except (OSError, ValueError):
                pass
            finally:
                output.put({"type": "eof"})

        def read_stderr():
            try:
                for line in worker.stderr:
                    self.stderr.append(line.rstrip())
            except (OSError, ValueError):
                pass

        threading.Thread(target=read_stdout, daemon=True).start()
        threading.Thread(target=read_stderr, daemon=True).start()

    def _exchange(self, request, job, forward=True):
        worker, output = self.worker, self.output
        if job.cancelled.is_set():
            raise CancelledError()
        worker.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        worker.stdin.flush()
        result, error, deadline = None, None, time.monotonic() + 600
        while True:
            if job.cancelled.is_set():
                raise CancelledError()
            if time.monotonic() > deadline:
                self._terminate()
                raise RuntimeError("Inference timed out after 10 minutes. The worker was released.")
            try:
                event = output.get(timeout=0.25)
            except queue.Empty:
                continue
            kind = event["type"]
            if kind == "eof":
                if job.cancelled.is_set():
                    raise CancelledError()
                self.loaded = None
                raise RuntimeError(
                    "Native worker exited unexpectedly. " + "\n".join(list(self.stderr)[-8:])
                )
            if kind == "error":
                error = event["message"]
            elif kind == "loaded":
                self.loaded = event
                job.model_load_ms += event.get("load_ms", 0.0)
                job.emit(event)
            elif kind == "result":
                result = event["data"]
                if job.kind == "chat":
                    result["model_load_ms"] = job.model_load_ms
            elif kind == "done":
                if error:
                    raise RuntimeError(error)
                return result
            elif forward:
                if kind == "metrics":
                    event["data"]["model_load_ms"] = job.model_load_ms
                job.emit(event)

    def _load(self, model, job, profile_directory=None):
        if (
            self.loaded
            and self.loaded["model"] == model["id"]
            and self.worker
            and self.worker.poll() is None
            and not profile_directory
        ):
            return
        with self.lock:
            if job.cancelled.is_set():
                raise CancelledError()
            self._terminate()
            self._spawn()
        job.emit({"type": "status", "message": f"Loading {model['name']} on native WebGPU…"})
        self._exchange(
            {
                "command": "load",
                "model": model,
                "library": self.settings["library"],
                "context": self.settings["context"],
                "asset_root": self.settings["asset_root"],
                "profile_directory": str(profile_directory) if profile_directory else None,
            },
            job,
        )

    def validate(self, data):
        if type(data.get("profile", False)) is not bool:
            raise ValueError("Profile must be a boolean.")
        kind = data.get("kind")
        if kind == "image":
            data["image_request"] = validate_image(self, data)
            return {"id": "qwen-image-2.1-turbo"}
        if kind == "tool":
            tool_request(data)
            return {"id": "tool:" + data["operation"]}
        if kind not in ("load", "embed", "search", "chat"):
            raise ValueError("Unknown job kind.")
        model = self.store.get("models", data.get("model", ""))
        if not model:
            raise ValueError("Select an imported model.")
        # Re-inspect every load/request so stale registry data cannot hide removed graphs.
        model = {**inspect_package(model["path"]), "id": model["id"]}
        if kind == "load" and model["kind"] == "image":
            raise ValueError(
                "Open Image Studio to generate. Its CUDA and WebGPU stages load and release separately."
            )
        if kind in ("embed", "search"):
            if model["kind"] != "embedding":
                raise ValueError("Choose an embedding model.")
            if data.get("dimensions", 768) not in (128, 256, 512, 768):
                raise ValueError("Invalid embedding dimensions.")
            inputs = (
                data.get("samples", data.get("texts")) if kind == "embed" else data.get("documents")
            )
            data["samples" if kind == "embed" else "documents"] = self.embedding_samples(
                inputs, model
            )
            if kind == "embed" and data.get("task", "SearchQuery") not in model["tasks"]:
                raise ValueError("Unknown embedding task.")
            if kind == "search":
                data["query_sample"] = self.embedding_samples(
                    [
                        {
                            "text": data.get("query", ""),
                            "attachments": data.get("query_attachments", []),
                        }
                    ],
                    model,
                )[0]
        if kind == "chat":
            if model["kind"] != "chat":
                raise ValueError("Embedding models do not generate text. Select a chat model.")
            messages = data.get("messages")
            if not isinstance(messages, list) or not 1 <= len(messages) <= 100:
                raise ValueError("Provide 1–100 messages.")
            for message in messages:
                if (
                    not isinstance(message, dict)
                    or message.get("role") not in ("system", "user", "assistant")
                    or not isinstance(message.get("content"), str)
                    or len(message["content"]) > 200000
                ):
                    raise ValueError("Invalid chat message.")
                attachments = message.get("attachments", [])
                if not isinstance(attachments, list) or len(attachments) > 8:
                    raise ValueError("At most 8 attachments per message are supported.")
                for attachment in attachments:
                    item = (
                        self.store.get("attachments", attachment.get("id", ""))
                        if isinstance(attachment, dict)
                        else None
                    )
                    if not item:
                        raise ValueError("An attachment no longer exists. Attach the file again.")
                    if item["kind"] not in model["modalities"]:
                        raise ValueError(
                            f"{model['name']} has no exported {item['kind']} encoder. Import a compatible multimodal GenAI package."
                        )
            for key, minimum, maximum, default in (
                ("max_tokens", 1, 4096, 4096),
                ("temperature", 0, 2, 0.7),
                ("top_p", 0.01, 1, 0.9),
                ("repetition_penalty", 0.5, 2, 1),
            ):
                value = data.get(key, default)
                if (
                    type(value) not in (int, float)
                    or not minimum <= value <= maximum
                    or (key == "max_tokens" and type(value) is not int)
                ):
                    raise ValueError(f"Invalid {key}.")
        return model

    def embedding_samples(self, inputs, model):
        if not isinstance(inputs, list) or not 1 <= len(inputs) <= 128:
            raise ValueError("Provide 1–128 nonempty embedding inputs.")
        samples = []
        for value in inputs:
            sample = {"text": value} if isinstance(value, str) else value
            if not isinstance(sample, dict) or not isinstance(sample.get("text", ""), str):
                raise ValueError(
                    "Each embedding input must be text or an object with text and attachments."
                )
            text, attachments = sample.get("text", "").strip(), sample.get("attachments", [])
            if len(text) > 100000 or not isinstance(attachments, list) or len(attachments) > 8:
                raise ValueError(
                    "Each input allows 100,000 text characters and at most 8 media files."
                )
            if not text and not attachments:
                raise ValueError("Provide nonempty text or attach an image, audio, or video file.")
            verified = []
            for attachment in attachments:
                identifier = (
                    attachment.get("id", "") if isinstance(attachment, dict) else attachment
                )
                item = (
                    self.store.get("attachments", identifier)
                    if isinstance(identifier, str)
                    else None
                )
                if not item or not Path(item["path"]).is_file():
                    raise ValueError("An attachment no longer exists. Attach the file again.")
                if item["kind"] not in model["modalities"]:
                    raise ValueError(
                        f"{model['name']} does not have an exported {item['kind']} encoder. Choose EmbeddingGemma 2 · Multimodal."
                    )
                verified.append({k: item[k] for k in ("id", "name", "kind", "mime", "size")})
            markers = {kind: text.count(f"<|{kind}|>") for kind in ("image", "audio", "video")}
            if any(markers.values()):
                if any(
                    count != sum(a["kind"] == kind for a in verified)
                    for kind, count in markers.items()
                ):
                    raise ValueError(
                        "Media placeholders must match the attached files for each modality. Remove placeholders to place attachments automatically."
                    )
            label = sample.get("label", "")
            if not isinstance(label, str) or len(label) > 160:
                raise ValueError("Input labels must contain at most 160 characters.")
            samples.append({"text": text, "attachments": verified, "label": label})
        return samples

    def worker_samples(self, samples):
        return [
            {
                "text": sample["text"],
                "attachments": [
                    self.store.get("attachments", a["id"]) for a in sample["attachments"]
                ],
            }
            for sample in samples
        ]

    def submit(self, data):
        model = self.validate(data)
        with self.lock:
            if self.active:
                raise BusyError("Another operation is running. Stop it or wait for completion.")
            job = Job(data["kind"], model["id"])
            self.active = job
            # Retain a bounded replay window; completed runs remain in SQLite.
            if len(self.jobs) >= 32:
                oldest = next(iter(self.jobs))
                del self.jobs[oldest]
            self.jobs[job.id] = job
            threading.Thread(target=self._run, args=(job, data, model), daemon=True).start()
            return job

    def _run(self, job, data, model):
        thread, response = None, ""
        started = time.perf_counter()
        try:
            if job.kind in ("image", "tool"):
                if job.kind == "tool":
                    with self.lock:
                        self._terminate()
                job.result = (
                    run_image(self, job, data["image_request"])
                    if job.kind == "image"
                    else run_tool(self, job, data)
                )
                if job.cancelled.is_set():
                    raise CancelledError()
                job.status = "completed"
                job.emit({"type": "result", "data": job.result})
                return
            profile_directory = None
            if data.get("profile") and job.kind != "load":
                profile_directory = self.image_directory.parent / "profiles" / job.id
                profile_directory.mkdir(parents=True)
            self._load(model, job, profile_directory)
            setup_ms = (time.perf_counter() - started) * 1000 if job.model_load_ms > 0 else 0.0
            if job.kind == "load":
                job.result = self.loaded
            elif job.kind == "search":
                job.emit({"type": "status", "message": "Embedding the query and documents…"})
                query = self._exchange(
                    {
                        "command": "embed",
                        "samples": self.worker_samples([data["query_sample"]]),
                        "dimensions": data.get("dimensions", 768),
                        "task": "SearchQuery",
                    },
                    job,
                )
                docs = self._exchange(
                    {
                        "command": "embed",
                        "samples": self.worker_samples(data["documents"]),
                        "dimensions": data.get("dimensions", 768),
                        "task": "Document",
                    },
                    job,
                )
                ranking_started = time.perf_counter()
                scores = [
                    sum(a * b for a, b in zip(query["vectors"][0], v)) for v in docs["vectors"]
                ]
                ranking = sorted(
                    [
                        {"index": i, **t, "score": scores[i]}
                        for i, t in enumerate(data["documents"])
                    ],
                    key=lambda d: d["score"],
                    reverse=True,
                )
                job.result = {
                    **search_embedding_metrics(query, docs),
                    "ranking": ranking,
                    "query": data["query_sample"]["text"],
                    "query_attachments": data["query_sample"]["attachments"],
                    "dimensions": docs["dimensions"],
                    "ranking_ms": (time.perf_counter() - ranking_started) * 1000,
                    "model": model["id"],
                }
            elif job.kind == "embed":
                job.result = self._exchange(
                    {**data, "samples": self.worker_samples(data["samples"]), "command": "embed"},
                    job,
                )
            else:
                messages = data["messages"]
                if data.get("save", True):
                    thread_id = data.get("thread_id") or uuid.uuid4().hex
                    old = self.store.get("threads", thread_id)
                    title = (
                        old["title"]
                        if old
                        else next(
                            (m["content"][:65] for m in messages if m["role"] == "user"),
                            "New conversation",
                        )
                    )
                    thread = self.store.put(
                        "threads",
                        {
                            "id": thread_id,
                            "title": title,
                            "model": model["id"],
                            "messages": messages,
                        },
                    )
                    job.emit({"type": "thread", "id": thread["id"]})
                worker_messages = [
                    {
                        **m,
                        "attachments": [
                            self.store.get("attachments", a["id"]) for a in m.get("attachments", [])
                        ],
                    }
                    for m in messages
                ]
                job.result = self._exchange(
                    {**data, "messages": worker_messages, "command": "chat"}, job
                )
                response = job.result["text"]
                if thread:
                    job.result["thread_id"] = thread["id"]
            if job.cancelled.is_set():
                raise CancelledError()
            if profile_directory:
                from .profiling import finish_profile

                job.emit({"type": "status", "message": "Saving the performance profile…"})
                self._exchange({"command": "finish_profile"}, job)
                job.result["profile"] = finish_profile(profile_directory, job.id, job.kind)
                self.store.put(
                    "profiles",
                    {
                        "id": job.id,
                        "directory": str(profile_directory),
                        "files": [p.name for p in profile_directory.glob("*.json")],
                    },
                )
                with self.lock:
                    self._terminate()
            if job.kind in ("embed", "search"):
                job.result.update(
                    embedding_performance(
                        job.result,
                        request_ms=(time.perf_counter() - started) * 1000,
                        setup_ms=setup_ms,
                        model_load_ms=job.model_load_ms,
                    )
                )
            job.status = "completed"
            job.emit({"type": "result", "data": job.result})
        except CancelledError:
            job.status = "cancelled"
            job.error = "Stopped. GPU memory released."
            job.emit({"type": "cancelled", "message": job.error})
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.emit({"type": "error", "message": job.error})
            with self.lock:
                self._terminate()
        finally:
            if job.kind == "chat" and job.result is None:
                latest = next(
                    (e["data"] for e in reversed(job.events) if e["type"] == "metrics"), None
                )
                if latest:
                    job.result = {**latest, "partial": True}
            if thread:
                response = response or "".join(
                    e.get("text", "") for e in job.events if e["type"] == "delta"
                )
                thread["messages"] = [
                    *thread["messages"],
                    {
                        "role": "assistant",
                        "content": response,
                        "metrics": job.result,
                        "status": job.status,
                        "error": job.error,
                    },
                ]
                self.store.put("threads", thread)
            summary = {
                **job.snapshot(),
                "result": {
                    k: v
                    for k, v in (job.result or {}).items()
                    if k not in ("vectors", "text", "ranking", "runtime")
                },
                "elapsed_ms": (time.time() - job.created) * 1000,
            }
            self.store.put("runs", summary)
            for old in self.store.list("runs")[100:]:
                self.store.delete("runs", old["id"])
            with self.lock:
                self.active = None
            job.emit({"type": "done", "status": job.status})
