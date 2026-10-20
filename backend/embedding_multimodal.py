"""Adapt Evo's validated EmbeddingGemma 2 preprocessing and graph orchestration.

All learned computation stays in native ONNX Runtime. Evo's isolated Transformers
and media decoder dependencies handle preprocessing; no PyTorch model is loaded.
"""

import importlib
import json
import sys
import time
from pathlib import Path


def configure_evo(root):
    root = Path(root).resolve()
    dependencies = root / ".cache" / "embeddinggemma2" / "deps"
    if not (dependencies / "transformers").is_dir():
        raise ValueError(
            f"Kite's multimodal processing dependencies are missing at {dependencies}. Install evo_resources/requirements/embeddinggemma2-processing.txt into that directory."
        )
    sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(dependencies)]
    importlib.invalidate_caches()
    return dependencies


class EvoEmbedding:
    def __init__(self, package, devices, provider_options, context, profile_directory=None):
        import onnxruntime as ort

        from evo.embedding_multimodal import MultimodalSession, prepare_files

        self.prepare_files = prepare_files
        self.package = Path(package)
        self.context = context
        self.config = json.loads((self.package / "embedding_config.json").read_text())
        declarations = self.config["graphs"]

        class SelectedRuntimeSession(MultimodalSession):
            def session(runner, name):
                if name not in runner.sessions:
                    path = (runner.package / declarations[name]["file"]).resolve()
                    if not path.is_relative_to(runner.package.resolve()):
                        raise ValueError("A model graph must stay inside its package.")
                    options = ort.SessionOptions()
                    options.log_severity_level = 3
                    if profile_directory:
                        options.enable_profiling = True
                        options.profile_file_prefix = str(Path(profile_directory) / name)
                    options.add_provider_for_devices(devices[:1], provider_options)
                    session = ort.InferenceSession(str(path), sess_options=options)
                    if not any(
                        "webgpu" in provider.lower() for provider in session.get_providers()
                    ):
                        raise RuntimeError(f"{name} did not activate native WebGPU.")
                    runner.sessions[name] = session
                return runner.sessions[name]

        self.runner = SelectedRuntimeSession(self.package)
        for name in ("model", "vision", "audio"):
            self.runner.session(name)

    def encode(self, samples, task, dimensions, emit):
        import numpy as np

        started = time.perf_counter()
        vectors, details = [], []
        inference_ms, preparation_ms, postprocessing_ms = 0.0, 0.0, 0.0
        tokens, padded_tokens = 0, 0
        for index, sample in enumerate(samples):
            emit("status", message=f"Preparing input {index + 1} of {len(samples)}…")
            before = time.perf_counter()
            attachments = sample.get("attachments", [])
            text = sample.get("text", "").strip()
            # Without explicit placeholders, preserve the actual attachment order,
            # including interleaved modalities. Evo then resolves each modality's
            # file list in its own placeholder order.
            if (
                text
                and attachments
                and not any(marker in text for marker in ("<|image|>", "<|audio|>", "<|video|>"))
            ):
                text += " " + " ".join(f"<|{a['kind']}|>" for a in attachments)
            processed = self.prepare_files(
                self.package,
                texts=[text] if text else None,
                images=[Path(a["path"]) for a in attachments if a["kind"] == "image"],
                audios=[Path(a["path"]) for a in attachments if a["kind"] == "audio"],
                videos=[Path(a["path"]) for a in attachments if a["kind"] == "video"],
                task=task,
            )
            if processed["input_ids"].shape[1] > self.context:
                raise ValueError(
                    f"Input {index + 1} exceeds the configured {self.context}-token context after media expansion. Use fewer/shorter media files or raise the context limit."
                )
            preparation_ms += (time.perf_counter() - before) * 1000
            before = time.perf_counter()
            vector = self.runner.run(processed)
            inference_ms += (time.perf_counter() - before) * 1000
            before = time.perf_counter()
            vector = vector[:, :dimensions].astype(np.float32)
            vector /= np.linalg.norm(vector, axis=-1, keepdims=True).clip(min=1e-12)
            if not np.isfinite(vector).all() or vector.shape != (1, dimensions):
                raise ValueError("The multimodal runtime returned an invalid embedding.")
            count = int(np.asarray(processed["attention_mask"]).sum())
            tokens += count
            padded_tokens += int(processed["input_ids"].size)
            vectors.extend(vector.tolist())
            details.append(
                {
                    "tokens": count,
                    "modalities": (["text"] if sample.get("text", "").strip() else [])
                    + list(dict.fromkeys(a["kind"] for a in attachments)),
                    "video_frames": np.asarray(processed.get("num_frames_per_video", []))
                    .astype(int)
                    .tolist(),
                }
            )
            postprocessing_ms += (time.perf_counter() - before) * 1000
            emit("progress", completed=index + 1, total=len(samples))
        return {
            "vectors": vectors,
            "dimensions": dimensions,
            "tokens": tokens,
            "inference_ms": inference_ms,
            "preparation_ms": preparation_ms,
            "postprocessing_ms": postprocessing_ms,
            "input_count": len(samples),
            "batch_count": len(samples),
            "padded_tokens": padded_tokens,
            "total_ms": (time.perf_counter() - started) * 1000,
            "task": task,
            "inputs": details,
        }
