"""Kite's integrated Evo tools, evidence registry and image generation jobs."""

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .paths import asset_root
from .process import ProcessGroup

PROJECT = Path(__file__).resolve().parent.parent
COMMANDS = {
    "build": "Export a checkpoint",
    "plan": "Preview a recipe",
    "convert": "Run an export recipe",
    "optimize": "Optimize ONNX graphs",
    "quantize": "Selective weight quantization",
    "inspect": "Inspect a graph",
    "analyze": "Optimization opportunities",
    "contrib": "Operator catalog",
    "capabilities": "Installed exporter controls",
    "validate-cpu": "Compare reference tensors",
    "profile": "WebGPU operator profiling",
    "benchmark": "Chat performance benchmark",
    "embedding-export": "Export EmbeddingGemma 2",
    "embedding-validate": "Validate embeddings",
    "embedding-benchmark": "Benchmark embeddings",
    "embed": "Multimodal embedding CLI",
    "image-export": "GGUF → ONNX image export",
    "image-prepare": "Prepare matched image inputs",
    "image-generate": "Image generation / reference runs",
    "image-optimize": "Optimize image graphs",
    "image-report": "Image fidelity and performance report",
}


def image_settings(service):
    evo = Path(service.settings["asset_root"])
    defaults = {
        "source": str(evo / "models/qwen-image-2.1-turbo"),
        "package": str(evo / "artifacts/qwen-image-2.1-turbo-gguf-floatzp"),
        "library": str(
            evo
            / "artifacts/runtimes/ort-webgpu-qwen-image-d3d12-unrolled/onnxruntime_providers_webgpu.dll"
        ),
        "dependencies": [
            str(evo / ".cache/qwen-image21/deps"),
            str(evo / ".cache/embeddinggemma2/deps"),
        ],
        "python": service.settings["python"],
        "dawn_cache_directory": str(service.image_directory.parent / "dawn-cache"),
    }
    return {
        **defaults,
        **(service.store.get("preferences", "image-runtime") or {}),
        "webgpu_backend": "D3D12",
    }


def image_readiness(settings):
    required = [
        Path(settings["python"]),
        Path(settings["library"]),
        Path(settings["source"]) / "model_index.json",
        Path(settings["package"]) / "image_config.json",
        Path(settings["package"]) / "denoise-short-blocks.onnx",
        Path(settings["package"]) / "prefix-short-blocks.onnx",
    ]
    return [str(p) for p in required if not p.is_file()]


def validate_image(service, data):
    settings = image_settings(service)
    missing = image_readiness(settings)
    if missing:
        raise ValueError("Configure Image Studio runtime. Missing: " + ", ".join(missing))
    if not isinstance(data.get("prompt"), str) or not 1 <= len(data["prompt"].strip()) <= 8000:
        raise ValueError("Enter a prompt with 1–8000 characters.")
    for key in ("width", "height"):
        if type(data.get(key)) is not int or data[key] not in (512, 768, 1024):
            raise ValueError("Image dimensions must be 512, 768 or 1024.")
    if type(data.get("seed")) is not int or not 0 <= data["seed"] < 2**32:
        raise ValueError("Seed must be an integer from 0 to 4294967295.")
    attachments = data.get("attachments", [])
    if not isinstance(attachments, list) or len(attachments) > 3:
        raise ValueError("Use up to three reference images.")
    images = []
    for identifier in attachments:
        item = service.store.get("attachments", identifier) if isinstance(identifier, str) else None
        if not item or item["kind"] != "image" or not Path(item["path"]).is_file():
            raise ValueError("Reference image is missing; attach it again.")
        images.append(item["path"])
    if type(data.get("capture", False)) is not bool:
        raise ValueError("Capture must be a boolean.")
    if type(data.get("profile", False)) is not bool:
        raise ValueError("Profile must be a boolean.")
    return {
        **settings,
        **{k: data[k] for k in ("prompt", "width", "height", "seed")},
        "capture": data.get("capture", False),
        "profile": data.get("profile", False),
        "images": images,
        "attachments": attachments,
    }


def run_process(service, job, command, directory, env=None):
    from .service import CancelledError

    log = Path(directory) / "process.log"
    with service.lock:
        if job.cancelled.is_set():
            raise CancelledError()
        process = subprocess.Popen(
            command,
            cwd=PROJECT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "PYTHONPATH": str(PROJECT),
                "KITE_ASSET_ROOT": str(asset_root()),
                **(env or {}),
            },
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            service.process_group = ProcessGroup(process)
        except Exception:
            process.kill()
            process.wait()
            raise
        service.worker = process
    tail = []
    try:
        with log.open("a", encoding="utf-8") as stream:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                if line.startswith("KITE_EVENT "):
                    job.emit(json.loads(line[11:]))
                else:
                    tail.append(line.rstrip())
                    tail = tail[-20:]
                    if line.strip():
                        # Bound event memory; full logs remain on disk.
                        if len(job.events) < 1200:
                            job.emit({"type": "log", "text": line[-3000:]})
        code = process.wait()
        if job.cancelled.is_set():
            raise CancelledError()
        if code:
            raise RuntimeError(f"Process exited with code {code}.\n" + "\n".join(tail))
    except (OSError, ValueError):
        if job.cancelled.is_set():
            raise CancelledError()
        raise
    finally:
        with service.lock:
            service._terminate()


def run_image(service, job, request):
    started = time.perf_counter()
    folder = service.image_directory / job.id
    folder.mkdir(parents=True)
    request = {**request, "directory": str(folder)}
    (folder / "request.json").write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    phases = {}
    for stage, label in [
        ("prepare", "Encoding your prompt and reference images"),
        ("prefix", "Building the WebGPU attention cache"),
        ("denoise", "Bringing your image to life"),
        ("decode", "Developing the final image"),
    ]:
        job.emit({"type": "status", "message": label + "…", "stage": stage})
        before = time.perf_counter()
        key = tuple(
            str(request[k]) for k in ("python", "library", "dependencies", "source", "package")
        )
        with service.lock:
            if (
                not service.worker
                or service.worker.poll() is not None
                or getattr(service, "image_worker_key", None) != key
                or not service.loaded
                or service.loaded.get("kind") != "image"
            ):
                service._terminate()
                service._spawn(image_request=request)
                service.image_worker_key = key
                service.loaded = {
                    "model": job.model,
                    "kind": "image",
                    "load_ms": 0,
                    "runtime": {"path": request["library"], "host_version": "", "devices": []},
                }
        phase_result = service._exchange({"request": request, "stage": stage}, job)

        phases[stage] = {
            **phase_result,
            "wall_ms": (time.perf_counter() - before) * 1000,
        }
    image = folder / "image.png"
    if not image.is_file():
        raise RuntimeError("The image decoder did not produce an image.")
    runtime = phases["denoise"]["runtime"]
    profile = None
    if request.get("profile"):
        from .image_profile import build_profile

        profile = build_profile(phases, job.id)
        (folder / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    service.loaded = {
        "model": job.model,
        "kind": "image",
        "load_ms": 0,
        "runtime": {
            **runtime,
            "host_version": phases["denoise"].get("host_version", ""),
            "devices": [{"name": "WebGPU", "metadata": runtime.get("device", {})}],
        },
    }
    record = {
        "id": job.id,
        "prompt": request["prompt"],
        "width": request["width"],
        "height": request["height"],
        "seed": request["seed"],
        "attachments": request["attachments"],
        "image_url": f"/api/images/{job.id}/image.png",
        "created": time.time(),
        "phases": phases,
        "profile": profile,
        "request_ms": (time.perf_counter() - started) * 1000,
        "timing_version": 2,
        "steps": phases["denoise"]["steps"],
        "capture": request["capture"],
        "path": str(image),
        "package": request["package"],
        "source": request["source"],
        "timing_method": "Interactive wall clock, including shader compilation and model loading when needed. CPU text/VAE components are reused between images. Denoiser: ORT WebGPU D3D12. Text/reference encoding and VAE: PyTorch CUDA. Not a warmed benchmark.",
    }
    (folder / "result.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    service.store.put("images", record)
    # These are disposable files created by this job, never model/source data.
    workspace = service.image_directory.resolve()
    if not folder.resolve().is_relative_to(workspace):
        raise ValueError("Image scratch directory escaped the workspace.")
    for name in ("prefix.npz", "image.npy", "fixtures/0.npz", "fixtures/cases.json"):
        (folder / name).unlink(missing_ok=True)
    (folder / "fixtures").rmdir()
    return record


def tool_request(data):
    command, args = data.get("operation"), data.get("arguments", [])
    if command not in COMMANDS or not isinstance(args, list) or len(args) > 150:
        raise ValueError("Choose a supported Model Lab operation and argument list.")
    if any(not isinstance(a, str) or len(a) > 20000 or "\x00" in a for a in args):
        raise ValueError("Invalid operation arguments.")
    return [command, *args]


def run_tool(service, job, data):
    folder = service.store.directory / "lab-jobs" / job.id
    folder.mkdir(parents=True)
    # Relative outputs resolve to the workspace; {output} gives every job a fresh artifact path.
    args = [a.replace("{output}", str(folder / "output")) for a in tool_request(data)]
    for original_arg, resolved_arg in zip(tool_request(data), args, strict=True):
        if "{output}" in original_arg:
            target = Path(resolved_arg).resolve()
            if not target.is_relative_to(folder.resolve()):
                raise ValueError("The {output} placeholder must stay within its job directory.")
            target.parent.mkdir(parents=True, exist_ok=True)
    if args[0] in ("plan", "convert") and len(args) > 1:
        recipe = Path(args[1]).resolve()
        resources = (PROJECT / "evo_resources/recipes").resolve()
        if recipe.is_relative_to(resources) and recipe.is_file():
            # Packaged recipes keep source checkpoints in place and write new outputs to Kite.
            text = recipe.read_text(encoding="utf-8")
            original = Path(service.settings["asset_root"]) / "recipes" / recipe.name
            import tomllib

            spec = tomllib.loads(text)
            if spec.get("source_kind") in ("local", "onnx"):
                source = (original.parent / spec["source"]).resolve()
                text = re.sub(
                    r"^source\s*=.*$",
                    lambda _: "source = " + json.dumps(str(source)),
                    text,
                    count=1,
                    flags=re.M,
                )
            text = re.sub(
                r"^output\s*=.*$",
                lambda _: "output = " + json.dumps(str(folder / "output")),
                text,
                count=1,
                flags=re.M,
            )
            (folder / "recipe.toml").write_text(text, encoding="utf-8")
            args[1] = str(folder / "recipe.toml")
    library = service.settings["library"]
    python = service.settings["python"]
    if args[0].startswith("image-"):
        settings = image_settings(service)
        library, python = settings["library"], settings["python"]
        if args[0] != "image-optimize" and "--dependencies" not in args:
            for dependency in settings["dependencies"]:
                args += ["--dependencies", dependency]
        if args[0] == "image-generate":
            if "--webgpu-backend" not in args:
                args += ["--webgpu-backend", "D3D12"]
            if "--graph-variant" not in args:
                args += ["--graph-variant", "short-blocks"]
            if "--fp32-accumulation" not in args:
                args.append("--fp32-accumulation")
    elif (
        args[0] in ("embedding-export", "embedding-validate", "embed")
        and "--transformers-path" not in args
    ):
        deps = Path(service.settings["asset_root"]) / ".cache/embeddinggemma2/deps"
        if deps.is_dir():
            args += ["--transformers-path", str(deps)]
    command = [python, "-u", "-m", "backend.tool_worker", *args]
    (folder / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    job.emit({"type": "status", "message": COMMANDS[data["operation"]] + "…"})
    run_process(service, job, command, folder, {"EVO_WEBGPU_LIBRARY": library})
    from .catalog import inspect_package

    output = folder / "output"
    imported = None
    if output.is_dir() and any(
        (output / name).is_file()
        for name in ("genai_config.json", "embedding_config.json", "image_config.json")
    ):
        try:
            imported = service.store.put("models", inspect_package(output))
        except ValueError:
            pass  # Intermediate graphs remain available in job artifacts.
    return {
        "operation": data["operation"],
        "imported_model": imported,
        "command": command,
        "directory": str(folder),
        "log_url": f"/api/lab/jobs/{job.id}/process.log",
        "files": [str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()][:200],
    }


class Evidence:
    def __init__(self, service):
        self.service = service
        self.lock = threading.RLock()
        self.instance = None

    def get(self):
        with self.lock:
            if self.instance:
                return self.instance
            root = Path(self.service.settings["asset_root"])
            target = self.service.store.directory / "workbench"
            target.mkdir(exist_ok=True)
            # First use snapshots the old editable data; Evo originals stay intact.
            for name in ("models.json", "performance.json", "wishlist.json"):
                source = root / "artifacts/workbench" / name
                if not (target / name).exists():
                    if source.is_file():
                        shutil.copy2(source, target / name)
                    else:
                        (target / name).write_text("[]")
            try:
                from evo.workbench import Workbench
            except ImportError:
                self.instance = EvidenceProxy(self.service.settings["python"], root, target)
            else:
                self.instance = Workbench(root, directory=target)
            return self.instance


class EvidenceProxy:
    """Keep the loopback HTTP host usable with standard-library-only Python."""

    def __init__(self, python, root, directory):
        self.python, self.root, self.directory = python, root, directory
        self.lock = threading.RLock()

    def call(self, method, *arguments):
        with self.lock:
            process = subprocess.Popen(
                [self.python, "-m", "backend.evidence_worker"],
                cwd=PROJECT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "PYTHONPATH": str(PROJECT),
                    "KITE_ASSET_ROOT": str(asset_root()),
                    "PYTHONIOENCODING": "utf-8",
                },
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            group = None
            try:
                group = ProcessGroup(process)
                out, err = process.communicate(
                    json.dumps(
                        {
                            "root": str(self.root),
                            "directory": str(self.directory),
                            "method": method,
                            "arguments": arguments,
                        }
                    ),
                    timeout=180,
                )
                if process.returncode:
                    raise ValueError(err[-3000:])
                return json.loads(
                    next(
                        line[14:] for line in out.splitlines() if line.startswith("KITE_EVIDENCE ")
                    )
                )
            finally:
                if group:
                    group.close()
                if process.poll() is None:
                    process.kill()
                    process.wait()

    @property
    def models(self):
        return self.call("models")

    @property
    def wishlist(self):
        return self.call("wishlist")

    def __getattr__(self, method):
        return lambda *args: self.call(method, *args)
