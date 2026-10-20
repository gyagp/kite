"""EmbeddingGemma 2 multimodal ONNX authoring and native inference.

The official processor owns media preprocessing and placeholder ordering. ONNX
owns all learned computation. Modality graphs return padded soft tokens; the
host strips padding and places them in the shared backbone's input sequence.
"""

import json
import shutil
from pathlib import Path
from types import MethodType

import numpy as np

from evo.pipeline import hash_file, write_json


def load_model(source, precision="fp32"):
    import torch
    from transformers import EmbeddingGemma2Model

    torch.set_num_threads(8)
    model = EmbeddingGemma2Model.from_pretrained(
        source, local_files_only=True, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    if precision == "fp16":
        model.half()
    return model


def vision_pool_indices(positions, kernel=3):
    """Group patches by their 2D coordinates, preserving official pool order."""
    positions = np.asarray(positions)
    batch, patches, _ = positions.shape
    length = patches // (kernel * kernel)
    # The sentinel selects an appended zero patch, including when no padding exists.
    indices = np.full((batch, length, kernel * kernel), patches, dtype=np.int64)
    mask = np.zeros((batch, length), dtype=bool)
    for b, pos in enumerate(positions):
        valid = np.all(pos >= 0, axis=-1)
        width = (int(pos[valid, 0].max()) + 1) // kernel
        groups = {}
        for i in np.flatnonzero(valid):
            x, y = pos[i]
            group = int(x // kernel + width * (y // kernel))
            groups.setdefault(group, []).append(int(i))
        for group, items in groups.items():
            if len(items) != kernel * kernel or not 0 <= group < length:
                raise ValueError("Image patches must form complete pooling cells")
            indices[b, group] = items
            mask[b, group] = True
    return indices, mask


def audio_block_mask(feature_mask, chunk=12, left=12, right=0):
    """Exact Gemma4 audio window (strict bound), in blocked attention layout."""
    valid = np.asarray(feature_mask, dtype=bool)[:, ::2][:, ::2]
    batch, length = valid.shape
    blocks = (length + chunk - 1) // chunk
    q = np.arange(blocks)[:, None, None] * chunk + np.arange(chunk)[None, :, None]
    k = np.arange(blocks)[:, None, None] * chunk - left
    k = k + np.arange(chunk + left + right)[None, None, :]
    distance = q - k
    nearby = ((distance >= 0) & (distance < left)) | ((distance < 0) & (-distance < right))
    allowed = nearby & (q < length) & (k >= 0) & (k < length)
    allowed = allowed[None] & valid[:, np.clip(k, 0, length - 1)]
    return np.logical_not(allowed).reshape(batch, 1, blocks, chunk, chunk + left + right)


def wrappers(model):
    import torch

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text = model.language_model

        def forward(self, input_ids, attention_mask, media_embeddings):
            media = (
                (input_ids == model.config.image_token_id)
                | (input_ids == model.config.video_token_id)
                | (input_ids == model.config.audio_token_id)
            )
            safe_ids = torch.where(media, self.text.config.pad_token_id, input_ids)
            embeddings = torch.where(
                media[:, :, None], media_embeddings, self.text.embed_tokens(safe_ids)
            )
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            valid = attention_mask[:, None, None, :].bool()
            nearby = (
                positions[:, None] - positions[None, :]
            ).abs() <= self.text.config.sliding_window
            zero = torch.zeros((), dtype=embeddings.dtype)
            blocked = torch.full((), torch.finfo(embeddings.dtype).min, dtype=embeddings.dtype)
            masks = {
                "full_attention": torch.where(valid, zero, blocked),
                "sliding_attention": torch.where(valid & nearby[None, None], zero, blocked),
            }
            hidden = self.text(
                inputs_embeds=embeddings, attention_mask=masks, position_ids=positions[None]
            ).last_hidden_state.float()
            weight = attention_mask[:, :, None].float()
            pooled = (hidden * weight).sum(1) / weight.sum(1).clamp(min=1)
            return torch.nn.functional.normalize(pooled, dim=-1, eps=1e-12)

    class Vision(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tower = model.vision_tower
            self.projector = model.embed_vision

        def forward(self, pixel_values, pixel_position_ids, pool_indices):
            padding = (pixel_position_ids == -1).all(-1)
            hidden = self.tower.patch_embedder(pixel_values, pixel_position_ids, padding)
            rotary = self.tower.encoder.rotary_emb(hidden, pixel_position_ids)
            mask = torch.where(
                padding[:, None, None],
                torch.full((), torch.finfo(hidden.dtype).min, dtype=hidden.dtype),
                torch.zeros((), dtype=hidden.dtype),
            )
            for layer in self.tower.encoder.layers:
                hidden = layer(
                    hidden,
                    attention_mask=mask,
                    position_embeddings=rotary,
                    position_ids=pixel_position_ids,
                )
            hidden = hidden.masked_fill(padding[:, :, None], 0)
            hidden = torch.cat([hidden, torch.zeros_like(hidden[:, :1])], dim=1)
            flat = pool_indices.reshape(pool_indices.shape[0], -1)
            grouped = torch.gather(hidden, 1, flat[:, :, None].expand(-1, -1, hidden.shape[-1]))
            grouped = grouped.reshape(pool_indices.shape[0], pool_indices.shape[1], 9, -1)
            # Match the reference pooler's rounding before sqrt(hidden_size) scaling.
            pooled = (grouped.float().sum(2) / 9).to(hidden.dtype)
            pooled = pooled.float() * self.tower.pooler.root_hidden_size
            if self.tower.config.standardize:
                pooled = (pooled - self.tower.std_bias.float()) * self.tower.std_scale.float()
            return self.projector(pooled.to(hidden.dtype))

    class Audio(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tower = model.audio_tower
            self.projector = model.embed_audio

        def forward(self, input_features, input_features_mask, blocked_mask):
            hidden, output_mask = self.tower.subsample_conv_projection(
                input_features, input_features_mask
            )
            rotary = self.tower.rel_pos_enc(hidden)
            for layer in self.tower.layers:
                hidden = layer(hidden, attention_mask=blocked_mask, position_embeddings=rotary)
            return self.projector(self.tower.output_proj(hidden)), output_mask

    return Backbone().eval(), Vision().eval(), Audio().eval()


def exportable_audio_context(model):
    """Express dynamic unfold as indexed gather; no change to window semantics."""
    import torch

    def context(self, hidden):
        padded = torch.nn.functional.pad(
            hidden,
            (0, 0, 0, 0, self.max_past_horizon, self.max_future_horizon + self.chunk_size - 1),
        )
        blocks = (hidden.shape[1] + self.chunk_size - 1) // self.chunk_size
        indices = torch.arange(blocks)[:, None] * self.chunk_size + torch.arange(self.context_size)
        return padded[:, indices]

    for layer in model.audio_tower.layers:
        layer.self_attn._extract_block_context = MethodType(context, layer.self_attn)


def export(source, output, precision="fp16"):
    import onnx
    import torch
    import transformers
    from transformers import AutoProcessor

    from evo.embedding_export import fuse_native_norms

    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a new output directory")
    model = load_model(source, precision)
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    exportable_audio_context(model)
    norm_count = fuse_native_norms(model.language_model)
    backbone, vision, audio = wrappers(model)
    dtype = next(model.parameters()).dtype
    text = processor(
        text=["task: search result | query: a bird", "title: none | text: hello"],
        return_tensors="pt",
    )
    from PIL import Image

    image = processor(
        images=Image.new("RGB", (96, 64), "blue"),
        images_kwargs={"max_soft_tokens": 70},
        return_tensors="pt",
    )
    pool, _ = vision_pool_indices(image["image_position_ids"].numpy())
    sound = processor(
        audio=np.sin(np.arange(16000) * 0.1).astype(np.float32),
        sampling_rate=16000,
        return_tensors="pt",
    )
    audio_mask = audio_block_mask(sound["input_features_mask"].numpy())
    parts = [
        (
            "model",
            backbone,
            (
                text["input_ids"],
                text["attention_mask"],
                torch.zeros(*text["input_ids"].shape, 512, dtype=dtype),
            ),
            ["input_ids", "attention_mask", "media_embeddings"],
            ["sentence_embedding"],
            {
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "media_embeddings": {0: "batch", 1: "sequence"},
                "sentence_embedding": {0: "batch"},
            },
        ),
        (
            "vision",
            vision,
            (image["pixel_values"].to(dtype), image["image_position_ids"], torch.from_numpy(pool)),
            ["pixel_values", "pixel_position_ids", "pool_indices"],
            ["soft_tokens"],
            {
                "pixel_values": {0: "images", 1: "patches"},
                "pixel_position_ids": {0: "images", 1: "patches"},
                "pool_indices": {0: "images", 1: "soft_length"},
                "soft_tokens": {0: "images", 1: "soft_length"},
            },
        ),
        (
            "audio",
            audio,
            (
                sound["input_features"].to(dtype),
                sound["input_features_mask"].bool(),
                torch.from_numpy(audio_mask),
            ),
            ["input_features", "input_features_mask", "blocked_mask"],
            ["soft_tokens", "soft_token_mask"],
            {
                "input_features": {0: "audios", 1: "frames"},
                "input_features_mask": {0: "audios", 1: "frames"},
                "blocked_mask": {0: "audios", 2: "blocks"},
                "soft_tokens": {0: "audios", 1: "soft_length"},
                "soft_token_mask": {0: "audios", 1: "soft_length"},
            },
        ),
    ]
    output.mkdir(parents=True)
    for name, module, inputs, input_names, output_names, dynamic_axes in parts:
        print(f"Exporting {name} ({precision})", flush=True)
        path = output / f"{name}.onnx"
        torch.onnx.export(
            module,
            inputs,
            str(path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=22,
            dynamo=False,
        )
        graph = onnx.load(path)
        graph.graph.output[0].type.tensor_type.shape.dim[-1].dim_value = (
            768 if name == "model" else 512
        )
        onnx.save_model(
            graph,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{name}.onnx.data",
            size_threshold=1024,
        )
        del graph
    processor.save_pretrained(output)
    for name in ("config.json", "config_sentence_transformers.json", "evo-source.json"):
        shutil.copy2(source / name, output / name)
    write_json(
        output / "embedding_config.json",
        {
            "task": "multimodal-embedding",
            "architecture": "EmbeddingGemma2Model",
            "modalities": ["text", "image", "audio", "video"],
            "precision": precision,
            "embedding_dimensions": 768,
            "max_length": 8192,
            "native_rms_norms": norm_count,
            "source": json.loads((source / "evo-source.json").read_text()),
            "source_weights_sha256": hash_file(source / "model.safetensors"),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            "graphs": {
                name: {"file": f"{name}.onnx", "sha256": hash_file(output / f"{name}.onnx")}
                for name in ("model", "vision", "audio")
            },
            "validation": {"passed": False, "status": "pending"},
        },
    )


class MultimodalSession:
    def __init__(self, package, cpu=False, profile=None):
        self.package = Path(package)
        self.config = json.loads((self.package / "embedding_config.json").read_text())
        self.model_config = json.loads((self.package / "config.json").read_text())
        self.dtype = np.float16 if self.config["precision"] == "fp16" else np.float32
        self.cpu, self.profile = cpu, profile
        self.sessions, self.devices, self.options = {}, {}, {}

    def session(self, name):
        if name not in self.sessions:
            path = self.package / f"{name}.onnx"
            if self.cpu:
                import onnxruntime as ort

                options = ort.SessionOptions()
                options.intra_op_num_threads = 8
                self.sessions[name] = ort.InferenceSession(
                    str(path), options, providers=["CPUExecutionProvider"]
                )
            else:
                from evo.embedding_validate import native_session

                profile = Path(f"{self.profile}-{name}") if self.profile else None
                session, device, options = native_session(path, profile)
                self.sessions[name], self.devices[name], self.options[name] = (
                    session,
                    device,
                    options,
                )
        return self.sessions[name]

    def run(self, inputs):
        ids = np.asarray(inputs["input_ids"], dtype=np.int64)
        mask = np.asarray(inputs["attention_mask"], dtype=np.int64)
        if ids.ndim != 2 or ids.shape != mask.shape or ids.shape[1] > self.config["max_length"]:
            raise ValueError("Expected matching [batch, sequence] inputs within 8192 tokens")
        media = np.zeros((*ids.shape, 512), dtype=self.dtype)
        for kind, pixels, positions in (
            ("image", "pixel_values", "image_position_ids"),
            ("video", "pixel_values_videos", "video_position_ids"),
        ):
            slots = ids == self.model_config[f"{kind}_token_id"]
            if pixels in inputs:
                pool, valid = vision_pool_indices(inputs[positions])
                features = self.session("vision").run(
                    None,
                    {
                        "pixel_values": np.asarray(inputs[pixels], dtype=self.dtype),
                        "pixel_position_ids": np.asarray(inputs[positions], dtype=np.int64),
                        "pool_indices": pool,
                    },
                )[0][valid]
                if int(slots.sum()) != len(features):
                    raise ValueError(f"{kind} token and feature counts differ")
                media[slots] = features
            elif slots.any():
                raise ValueError(f"Missing {kind} inputs")
        slots = ids == self.model_config["audio_token_id"]
        if "input_features" in inputs:
            feature_mask = np.asarray(inputs["input_features_mask"], dtype=bool)
            features, valid = self.session("audio").run(
                None,
                {
                    "input_features": np.asarray(inputs["input_features"], dtype=self.dtype),
                    "input_features_mask": feature_mask,
                    "blocked_mask": audio_block_mask(feature_mask),
                },
            )
            features = features[valid.astype(bool)]
            if int(slots.sum()) != len(features):
                raise ValueError("Audio token and feature counts differ")
            media[slots] = features
        elif slots.any():
            raise ValueError("Missing audio inputs")
        vectors = self.session("model").run(
            None, {"input_ids": ids, "attention_mask": mask, "media_embeddings": media}
        )[0]
        if not np.isfinite(vectors).all():
            raise ValueError("Model returned non-finite embeddings")
        return vectors


def prepare_files(package, texts=None, images=None, audios=None, videos=None, task="SearchQuery"):
    """Prepare one interleaved sample, or a text-only batch, with official processing."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(package, local_files_only=True)
    images, audios, videos = images or [], audios or [], videos or []
    has_media = bool(images or audios or videos)
    if not texts and not has_media:
        raise ValueError("Provide text, image, audio, or video inputs")
    if has_media and texts and len(texts) != 1:
        raise ValueError(
            "Use one --text for a mixed sample; the Python API supports batched inputs"
        )
    prompts = json.loads((Path(package) / "config_sentence_transformers.json").read_text())[
        "prompts"
    ]
    if task not in prompts:
        raise ValueError(f"Unknown task prefix: {task}")
    kwargs = {}
    if texts:
        if any(not text.strip() for text in texts):
            raise ValueError("Provide nonempty text")
        texts = [prompts[task] + text for text in texts]
        if has_media and not any(
            token in texts[0] for token in ("<|image|>", "<|audio|>", "<|video|>")
        ):
            texts[0] += " " + " ".join(
                ["<|image|>"] * len(images)
                + ["<|video|>"] * len(videos)
                + ["<|audio|>"] * len(audios)
            )
        kwargs["text"] = texts
    if images:
        from PIL import Image

        loaded = []
        for path in images:
            with Image.open(path) as image:
                loaded.append(image.convert("RGB"))
        kwargs["images"] = [loaded]
    if audios:
        import soundfile as sf
        import soxr

        waveforms = []
        for path in audios:
            samples, rate = sf.read(path, dtype="float32", always_2d=True)
            samples = samples.mean(axis=1)
            if rate != 16000:
                samples = soxr.resample(samples, rate, 16000)
            if not len(samples) or not np.isfinite(samples).all():
                raise ValueError(f"Invalid audio: {path}")
            waveforms.append(samples)
        kwargs["audio"] = waveforms
        kwargs["sampling_rate"] = 16000
        # Do not silently discard audio or inherit an undefined tokenizer length.
        # Enforce the combined model-token budget after placeholder expansion.
        kwargs["audio_kwargs"] = {"truncation": False}
    if videos:
        import av
        from transformers.video_utils import VideoMetadata

        clips = []
        for path in videos:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                rate = float(stream.average_rate)
                count = stream.frames
                if not count:
                    count = sum(1 for _ in container.decode(stream))
                    container.seek(0)
                metadata = VideoMetadata(total_num_frames=count, fps=rate, duration=count / rate)
                indices = processor.video_processor.sample_frames(
                    metadata,
                    fps=processor.video_processor.fps,
                    max_frames=processor.video_processor.max_frames,
                    overflow_strategy=processor.video_processor.overflow_strategy,
                )
                wanted = set(int(i) for i in indices)
                frames = {
                    i: frame.to_ndarray(format="rgb24")
                    for i, frame in enumerate(container.decode(stream))
                    if i in wanted
                }
                if not frames or any(int(i) not in frames for i in indices):
                    raise ValueError(f"Could not decode sampled video frames: {path}")
                clips.append(np.stack([frames[int(i)] for i in indices]))
        kwargs["videos"] = clips
        kwargs["videos_kwargs"] = {"do_sample_frames": False}
    # Group multiple pure-media files into one sample, maintaining placeholder order.
    if has_media and not texts:
        kwargs["text"] = [
            " ".join(
                ["<|image|>"] * len(images)
                + ["<|video|>"] * len(videos)
                + ["<|audio|>"] * len(audios)
            )
        ]
    processed = processor(**kwargs, return_tensors="np")
    if processed["input_ids"].shape[1] > 8192:
        raise ValueError("Combined text and media exceed the 8192-token context")
    return dict(processed)
