"""Discover local ONNX packages without loading weights or modifying exports."""

import hashlib
import json
from pathlib import Path


def read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default if default is not None else {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def inspect_package(path):
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Choose an existing model package directory.")
    segmentation = read_json(root / "segmentation_config.json")
    if segmentation:
        from .segmentation import inspect_segmentation

        return inspect_segmentation(root, segmentation)
    image = read_json(root / "image_config.json")
    if image:
        if image.get("task") != "image-generation":
            raise ValueError("Unsupported image model contract.")
        for name, artifact in image.get("artifacts", {}).items():
            file = (root / name).resolve()
            if (
                not file.is_relative_to(root)
                or not file.is_file()
                or file.stat().st_size != artifact.get("bytes")
            ):
                raise ValueError(f"Missing or incomplete image artifact: {name}")
        for name in ("prefix-short-blocks.onnx", "denoise-short-blocks.onnx"):
            if not (root / name).is_file():
                raise ValueError(f"Image Studio requires {name}.")
        return {
            "id": hashlib.sha256(str(root).lower().encode()).hexdigest()[:16],
            "name": "Qwen-Image · 2.1 Turbo",
            "path": str(root),
            "kind": "image",
            "architecture": "QwenImage21",
            "modalities": ["text", "image"],
            "size": sum(p.stat().st_size for p in root.iterdir() if p.is_file()),
            "precision": "GGUF mixed",
            "context": 0,
            "tasks": ["text-to-image", "image-edit"],
            "graphs": ["prefix-short-blocks.onnx", "denoise-short-blocks.onnx"],
            "source": image.get("source", {}),
            "variant": root.name,
        }
    embedding = read_json(root / "embedding_config.json")
    genai = read_json(root / "genai_config.json")
    if not embedding and not genai:
        raise ValueError(
            "Choose a package with embedding_config.json, genai_config.json, image_config.json, or segmentation_config.json and its model files."
        )
    if not (root / "tokenizer.json").is_file():
        raise ValueError("The package is missing tokenizer.json.")
    model = genai.get("model", {})
    multimodal = embedding.get("task") == "multimodal-embedding"
    graphs = ["model.onnx"] if embedding else []
    if multimodal:
        if embedding.get("architecture") != "EmbeddingGemma2Model":
            raise ValueError("Unsupported multimodal embedding architecture.")
        declarations = embedding.get("graphs", {})
        if not all(
            isinstance(declarations.get(role), dict)
            and isinstance(declarations[role].get("file"), str)
            for role in ("model", "vision", "audio")
        ):
            raise ValueError(
                "EmbeddingGemma 2 multimodal needs model, vision, and audio graph declarations."
            )
        graphs = [declarations[role]["file"] for role in ("model", "vision", "audio")]
        for required in (
            "processor_config.json",
            "config.json",
            "config_sentence_transformers.json",
        ):
            if not (root / required).is_file():
                raise ValueError(f"The multimodal package is missing {required}.")

    # Include each graph declared by GenAI, including multimodal submodels.
    def collect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "filename" and isinstance(item, str) and item.endswith(".onnx"):
                    graphs.append(item)
                else:
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    if genai:
        collect(model)
    if not graphs:
        raise ValueError("No ONNX graphs are declared by this model package.")
    for name in graphs:
        resolved = (root / name).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError(f"Missing or unsafe graph path: {name}")
    modalities = ["text"]
    if multimodal:
        if embedding.get("modalities") != ["text", "image", "audio", "video"]:
            raise ValueError("Expected the EmbeddingGemma 2 text/image/audio/video contract.")
        modalities = list(embedding["modalities"])
        for name, artifact in embedding.get("artifacts", {}).items():
            dependency = (root / name).resolve()
            if not dependency.is_relative_to(root) or not dependency.is_file():
                raise ValueError(f"Missing or unsafe model artifact: {name}")
            if dependency.stat().st_size != artifact.get("bytes"):
                raise ValueError(f"Incomplete model artifact: {name}")
    elif not embedding:
        if "vision" in model:
            modalities.append("image")
        if "speech" in model or "audio" in model:
            modalities.append("audio")
    prompts = read_json(root / "config_sentence_transformers.json").get("prompts", {})
    manifest = read_json(root / "kite-model.json")
    size = sum(
        p.stat().st_size
        for p in root.rglob("*")
        if p.is_file() and (p.suffix == ".onnx" or ".data" in p.name)
    )
    kind = "embedding" if embedding else "chat"
    name = manifest.get("name") or (
        "EmbeddingGemma 2 · Multimodal"
        if multimodal
        else "EmbeddingGemma 2"
        if embedding.get("architecture", "").startswith("EmbeddingGemma2")
        else "Qwen3.5 · 4B"
        if "qwen3_5" in model.get("type", "")
        else root.name
    )
    return {
        "id": hashlib.sha256(str(root).lower().encode()).hexdigest()[:16],
        "name": name,
        "path": str(root),
        "kind": kind,
        "embedding_task": embedding.get("task", "text-embedding") if embedding else None,
        "architecture": embedding.get("architecture") or model.get("type", "GenAI"),
        "modalities": modalities,
        "size": size,
        "precision": embedding.get("precision") or manifest.get("precision", "mixed"),
        "context": embedding.get("max_length", model.get("context_length", 8192)),
        "dimensions": embedding.get("embedding_dimensions"),
        "tasks": list(prompts),
        "graphs": graphs,
        "validation": embedding.get("validation"),
        "source": manifest.get("source") or embedding.get("source", {}),
        "variant": root.name,
    }


def discover(root):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("The scan directory does not exist.")
    found, errors = [], []
    # Deliberately shallow: model artifacts can occupy hundreds of GB.
    for folder in [root, *sorted(p for p in root.iterdir() if p.is_dir())]:
        if not any(
            (folder / name).exists()
            for name in (
                "embedding_config.json",
                "genai_config.json",
                "image_config.json",
                "segmentation_config.json",
            )
        ):
            continue
        try:
            found.append(inspect_package(folder))
        except (ValueError, OSError) as exc:
            errors.append({"path": str(folder), "error": str(exc)})
    return found, errors
