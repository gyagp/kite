"""Persistent, isolated native inference worker. JSON protocol lives on stdout.

Runs with the configured inference Python (Evo's venv by default). Never downloads
weights, edits packages, or falls back to another GPU provider. Multimodal
preprocessing can import Torch via the official processor; model execution is ORT.
"""

import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

if __package__:
    from .embedding_multimodal import EvoEmbedding, configure_evo
    from .timing import generation_metrics
else:
    from embedding_multimodal import EvoEmbedding, configure_evo
    from timing import generation_metrics

PREFIX = "KITE_EVENT "


def emit(event, **data):
    print(
        PREFIX + json.dumps({"type": event, **data}, ensure_ascii=False, allow_nan=False),
        flush=True,
    )


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def register_runtime(library, genai=False):
    import onnxruntime as ort

    path = Path(library).resolve()
    if not path.is_file():
        raise ValueError(
            f"WebGPU DLL not found: {path}. Select your source-built runtime in Settings."
        )
    info = {"path": str(path), "sha256": digest(path), "host_version": ort.__version__}
    sidecar = path.parent / "evo-runtime.json"
    if sidecar.is_file():
        from evo.runtime_artifacts import verify_runtime_artifacts

        info["build"] = json.loads(sidecar.read_text())
        verify_runtime_artifacts(path.parent, info["build"])
    if os.name == "nt":
        # Keep the DLL search handle alive throughout inference.
        register_runtime.dll_handle = os.add_dll_directory(str(path.parent))
    ort.register_execution_provider_library("WebGpuExecutionProvider", str(path))
    if genai:
        import onnxruntime_genai as og

        og.register_execution_provider_library("WebGpuExecutionProvider", str(path))
        info["genai_version"] = og.__version__
    devices = [d for d in ort.get_ep_devices() if "webgpu" in d.ep_name.lower()]
    if not devices:
        raise RuntimeError("The selected native WebGPU runtime found no compatible GPU.")
    info["devices"] = [
        {
            "name": d.ep_name,
            "vendor_id": d.device.vendor_id,
            "device_id": d.device.device_id,
            "metadata": dict(d.device.metadata),
        }
        for d in devices
    ]
    return devices, info


class Engine:
    def __init__(self, request):
        multimodal = request["model"].get("embedding_task") == "multimodal-embedding"
        if multimodal:
            configure_evo(request.get("asset_root", ""))
        import numpy as np
        from transformers import AutoTokenizer

        self.np = np
        self.package = Path(request["model"]["path"])
        self.spec = request["model"]
        self.profile_directory = request.get("profile_directory")
        self.options = {
            "powerPreference": "high-performance",
            "dawnBackendType": "D3D12",
            "validationMode": "basic",
            "enableGraphCapture": "0",
        }
        start = time.perf_counter()
        devices, self.runtime = register_runtime(request["library"], self.spec["kind"] == "chat")
        self.tokenizer = None
        self.multimodal = None
        self.context = min(request.get("context", 8192), self.spec.get("context", 8192))
        if multimodal:
            self.multimodal = EvoEmbedding(
                self.package, devices, self.options, self.context, self.profile_directory
            )
            self.runtime["embedding_graphs"] = ["model", "vision", "audio"]
            self.runtime["preprocessing"] = "Official EmbeddingGemma 2 processor"
        elif self.spec["kind"] == "embedding":
            import onnxruntime as ort

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.package, local_files_only=True, trust_remote_code=False
            )
            options = ort.SessionOptions()
            options.log_severity_level = 3
            if self.profile_directory:
                options.enable_profiling = True
                options.profile_file_prefix = str(Path(self.profile_directory) / "embedding")
            options.add_provider_for_devices(devices[:1], self.options)
            self.model = ort.InferenceSession(
                str(self.package / "model.onnx"), sess_options=options
            )
            if not any("webgpu" in p.lower() for p in self.model.get_providers()):
                raise RuntimeError(
                    "WebGPU provider was not activated; refusing a CPU-only session."
                )
            self.prompts = json.loads(
                (self.package / "config_sentence_transformers.json").read_text()
            )["prompts"]
        else:
            import onnxruntime_genai as og

            self.og = og
            config = og.Config(str(self.package))
            config.clear_providers()
            config.append_provider("WebGPU")
            for key, value in self.options.items():
                config.set_provider_option("WebGPU", key, value)
            config.overlay(json.dumps({"search": {"max_length": request.get("context", 8192)}}))
            if self.profile_directory:
                config.overlay(
                    json.dumps(
                        {
                            "model": {
                                "decoder": {
                                    "session_options": {
                                        "enable_profiling": str(
                                            Path(self.profile_directory) / "chat"
                                        )
                                    }
                                }
                            }
                        }
                    )
                )
            self.model = og.Model(config)
            self.decoder = og.Tokenizer(self.model)
            self.processor = (
                self.model.create_multimodal_processor()
                if len(self.spec["modalities"]) > 1
                else None
            )
            if self.processor is None:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.package, local_files_only=True, trust_remote_code=False
                    )
                except (ValueError, OSError):
                    # GenAI packages can supply a native tokenizer without a Transformers class.
                    self.tokenizer = None
        self.context = min(request.get("context", 8192), self.spec.get("context", 8192))
        emit(
            "loaded",
            model=self.spec["id"],
            runtime=self.runtime,
            load_ms=(time.perf_counter() - start) * 1000,
            provider_options=self.options,
        )

    def finish_profile(self):
        if self.multimodal:
            for session in self.multimodal.runner.sessions.values():
                session.end_profiling()
        elif self.spec["kind"] == "embedding":
            self.model.end_profiling()
        else:
            # GenAI owns its ORT sessions and flushes session profiling on release.
            self.processor = None
            self.decoder = None
            self.model = None
            gc.collect()
        emit("result", data={"profile_directory": self.profile_directory})

    def embed(self, request):
        np = self.np
        if self.spec["kind"] != "embedding":
            raise ValueError("Select an embedding model for this request.")
        dimensions = request.get("dimensions", 768)
        if dimensions not in (128, 256, 512, 768):
            raise ValueError("Dimensions must be 128, 256, 512, or 768.")
        task = request.get("task", "SearchQuery")
        if self.multimodal:
            samples = request.get("samples") or [
                {"text": t, "attachments": []} for t in request.get("texts", [])
            ]
            result = self.multimodal.encode(samples, task, dimensions, emit)
            emit("result", data={**result, "model": self.spec["id"]})
            return
        if task not in self.prompts:
            raise ValueError(f"Unknown task: {task}")
        samples = request.get("samples")
        if samples and any(sample.get("attachments") for sample in samples):
            raise ValueError("This embedding export supports text only.")
        texts = [sample["text"] for sample in samples] if samples else request["texts"]
        if (
            not texts
            or len(texts) > 128
            or any(not isinstance(t, str) or not t.strip() for t in texts)
        ):
            raise ValueError("Provide between 1 and 128 nonempty texts.")
        started = time.perf_counter()
        vectors, details, tokens, padded_tokens, batches = [], [], 0, 0, 0
        inference_ms, preparation_ms, postprocessing_ms = 0.0, 0.0, 0.0
        # Small sequential batches bound dynamic attention memory for long inputs.
        for offset in range(0, len(texts), 4):
            before = time.perf_counter()
            inputs = self.tokenizer(
                [self.prompts[task] + t for t in texts[offset : offset + 4]],
                padding=True,
                return_tensors="np",
                return_token_type_ids=False,
            )
            if inputs["input_ids"].shape[1] > min(self.context, self.spec["context"]):
                raise ValueError(
                    f"Input exceeds the configured {self.context}-token context; shorten it or raise the limit."
                )
            preparation_ms += (time.perf_counter() - before) * 1000
            before = time.perf_counter()
            output = self.model.run(["sentence_embedding"], dict(inputs))[0]
            inference_ms += (time.perf_counter() - before) * 1000
            before = time.perf_counter()
            output = output[:, :dimensions].astype(np.float32)
            output /= np.linalg.norm(output, axis=-1, keepdims=True).clip(min=1e-12)
            if not np.isfinite(output).all():
                raise ValueError("The runtime returned non-finite embeddings.")
            vectors.extend(output.tolist())
            counts = inputs["attention_mask"].sum(axis=1).astype(int).tolist()
            tokens += sum(counts)
            padded_tokens += int(inputs["input_ids"].size)
            batches += 1
            details.extend(
                {"tokens": count, "modalities": ["text"], "video_frames": []} for count in counts
            )
            postprocessing_ms += (time.perf_counter() - before) * 1000
            emit("progress", completed=min(offset + 4, len(texts)), total=len(texts))
        emit(
            "result",
            data={
                "vectors": vectors,
                "dimensions": dimensions,
                "tokens": tokens,
                "inference_ms": inference_ms,
                "preparation_ms": preparation_ms,
                "postprocessing_ms": postprocessing_ms,
                "input_count": len(texts),
                "batch_count": batches,
                "padded_tokens": padded_tokens,
                "inputs": details,
                "total_ms": (time.perf_counter() - started) * 1000,
                "task": task,
                "model": self.spec["id"],
            },
        )

    def chat(self, request):
        preparation_started = time.perf_counter()
        if self.spec["kind"] != "chat":
            raise ValueError(
                "Embedding models do not generate chat responses. Select a chat model."
            )
        messages = request["messages"]
        if not messages:
            raise ValueError("Provide at least one message.")
        og = self.og
        attachments = [a for m in messages for a in m.get("attachments", [])]
        for attachment in attachments:
            if attachment["kind"] not in self.spec["modalities"]:
                raise ValueError(f"This export does not include a {attachment['kind']} encoder.")
        # Preserve the package's actual chat template, including Qwen thinking control.
        template_messages = []
        image_index, audio_index = 0, 0
        for message in messages:
            content = message["content"]
            if message.get("attachments"):
                content, image_index, audio_index = media_content(
                    self.spec["architecture"],
                    content,
                    message["attachments"],
                    image_index,
                    audio_index,
                )
            template_messages.append({"role": message["role"], "content": content})
        if self.tokenizer is not None:
            prompt = self.tokenizer.apply_chat_template(
                template_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=request.get("thinking", False),
            )
        else:
            template = self.package / "chat_template.jinja"
            prompt = self.decoder.apply_chat_template(
                json.dumps(template_messages, ensure_ascii=False),
                template_str=template.read_text(encoding="utf-8") if template.is_file() else None,
                add_generation_prompt=True,
            )
        params = og.GeneratorParams(self.model)
        input_ids = (
            self.tokenizer.encode(prompt, add_special_tokens=False)
            if self.tokenizer is not None
            else self.decoder.encode(prompt)
        )
        max_tokens = request.get("max_tokens", 4096)
        prompt_tokens = len(input_ids)
        named_inputs = None
        media_handles = []
        if attachments:
            kwargs = {}
            for kind, argument, klass in (
                ("image", "images", og.Images),
                ("audio", "audios", og.Audios),
            ):
                paths = [a["path"] for a in attachments if a["kind"] == kind]
                if paths:
                    handle = klass.open(*paths)
                    media_handles.append(handle)
                    kwargs[argument] = handle
            named_inputs = self.processor(prompt, **kwargs)
            # Use expanded media token length; never count just placeholder tokens.
            prompt_tokens = int(named_inputs["input_ids"].shape()[1])
        if prompt_tokens + max_tokens > self.context:
            raise ValueError(
                f"Prompt ({prompt_tokens} tokens) + output limit ({max_tokens}) exceeds context ({self.context}). Start a new chat or increase context in Settings."
            )
        temperature = request.get("temperature", 0.7)
        search = {
            "max_length": prompt_tokens + max_tokens,
            "do_sample": temperature > 0,
            "repetition_penalty": request.get("repetition_penalty", 1.0),
        }
        if temperature > 0:
            search.update(temperature=temperature, top_p=request.get("top_p", 0.9), top_k=50)
        params.set_search_options(**search)
        preparation_finished = time.perf_counter()
        generator = og.Generator(self.model, params)
        stream = self.decoder.create_stream()
        started, first, count = time.perf_counter(), None, 0
        last, first_count, last_metrics = None, 0, started
        prefill_finished = started

        def metrics(observed):
            return generation_metrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=count,
                first_step_tokens=first_count,
                started=started,
                prefill_finished=prefill_finished,
                first_token=first,
                last_token=last,
                observed=observed,
                input_preparation_ms=(preparation_finished - preparation_started) * 1000,
                generator_setup_ms=(started - preparation_finished) * 1000,
            )

        text = ""
        try:
            if named_inputs is not None:
                generator.set_inputs(named_inputs)
            else:
                generator.append_tokens(input_ids)
            # AppendTokens/SetInputs computes prompt logits. Their GPU work may be
            # asynchronous: the public GetLogits copies to CPU and establishes a
            # completed-prefill boundary before first-token sampling. It does not
            # rerun the prompt when logits are already available.
            prompt_logits = generator.get_logits()
            prefill_finished = time.perf_counter()
            del prompt_logits
            emit("metrics", data=metrics(prefill_finished))
            while not generator.is_done() and count < max_tokens:
                generator.generate_next_token()
                next_tokens = generator.get_next_tokens()
                now = time.perf_counter()
                if first is None:
                    first = now
                    first_count = len(next_tokens)
                last = now
                for token in next_tokens:
                    delta = stream.decode(int(token))
                    text += delta
                    count += 1
                    emit("delta", text=delta)
                if count == first_count or now - last_metrics >= 0.25:
                    emit("metrics", data=metrics(time.perf_counter()))
                    last_metrics = now
            completed = metrics(time.perf_counter())
            emit(
                "result",
                data={
                    **completed,
                    "text": text,
                    "finish_reason": "length" if count >= max_tokens else "stop",
                    "model": self.spec["id"],
                },
            )
        finally:
            del generator, params, stream
            gc.collect()


def media_content(architecture, text, attachments, image_index=0, audio_index=0):
    """Use the GenAI upstream model-mm media marker contracts; preserve history order."""
    tags = []
    for item in attachments:
        kind = item["kind"]
        if kind == "image":
            image_index += 1
        else:
            audio_index += 1
        if architecture in ("phi3v", "phi4mm"):
            index = image_index if kind == "image" else audio_index
            tags.append(f"<|{kind}_{index}|>\n")
        elif architecture in ("qwen2_5_vl", "qwen3_vl", "qwen3_5", "fara"):
            if kind != "image":
                raise ValueError("This Qwen GenAI adapter supports image inputs only.")
            tags.append("<|vision_start|><|image_pad|><|vision_end|>")
        elif architecture == "mistral3":
            if kind != "image":
                raise ValueError("The Mistral GenAI adapter supports image inputs only.")
            tags.append("[IMG]")
        else:
            tags.append({"type": kind})
    content = (
        "".join(tags) + text
        if tags and isinstance(tags[0], str)
        else tags + [{"type": "text", "text": text}]
    )
    return content, image_index, audio_index


def main():
    engine = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request["command"] == "load":
                if request["model"]["kind"] == "segmentation":
                    from backend.sam3 import Sam3Engine

                    engine = Sam3Engine(request)
                else:
                    engine = Engine(request)
            elif engine is None:
                raise ValueError("Load a model before inference.")
            elif request["command"] == "embed":
                engine.embed(request)
            elif request["command"] == "chat":
                engine.chat(request)
            elif request["command"] == "track":
                engine.track(request)
            elif request["command"] == "finish_profile":
                engine.finish_profile()
            else:
                raise ValueError("Unknown inference command.")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit("error", message=str(exc))
        finally:
            emit("done")


if __name__ == "__main__":
    main()
