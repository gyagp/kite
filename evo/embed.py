"""Generate normalized text embeddings using native ORT WebGPU."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from evo.embedding_validate import native_session


def encode(package, texts, task="SearchQuery", dimensions=768):
    from transformers import AutoTokenizer

    if dimensions not in {128, 256, 512, 768}:
        raise ValueError("Embedding dimensions must be 128, 256, 512, or 768")
    if not texts or any(not t.strip() for t in texts):
        raise ValueError("Provide nonempty text")
    prompts = json.loads((package / "config_sentence_transformers.json").read_text())["prompts"]
    if task not in prompts:
        raise ValueError(f"Unknown task prefix: {task}")
    tokenizer = AutoTokenizer.from_pretrained(package, local_files_only=True)
    inputs = dict(tokenizer([prompts[task] + t for t in texts], padding=True, return_tensors="np"))
    if inputs["input_ids"].shape[1] > 8192:
        raise ValueError("Inputs exceed the documented 8192-token context")
    session, _, _ = native_session(package / "model.onnx")
    vectors = session.run(None, inputs)[0][:, :dimensions]
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True).clip(min=1e-12)
    if not np.isfinite(vectors).all():
        raise ValueError("Runtime returned non-finite embeddings")
    return vectors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--text", action="append")
    parser.add_argument("--image", type=Path, action="append")
    parser.add_argument("--audio", type=Path, action="append")
    parser.add_argument("--video", type=Path, action="append")
    parser.add_argument("--transformers-path", type=Path)
    parser.add_argument("--task", default="SearchQuery")
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.transformers_path:
        sys.path.insert(0, str(args.transformers_path.resolve()))
    config = json.loads((args.package / "embedding_config.json").read_text())
    if config["task"] == "multimodal-embedding":
        from evo.embedding_multimodal import MultimodalSession, prepare_files

        if args.dimensions not in {128, 256, 512, 768}:
            raise ValueError("Embedding dimensions must be 128, 256, 512, or 768")
        inputs = prepare_files(
            args.package, args.text, args.image, args.audio, args.video, args.task
        )
        vectors = MultimodalSession(args.package).run(inputs)[:, : args.dimensions]
        vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True).clip(min=1e-12)
    else:
        if args.image or args.audio or args.video:
            raise ValueError("Media inputs require a multimodal embedding package")
        vectors = encode(args.package, args.text, args.task, args.dimensions)
    if args.output:
        np.save(args.output, vectors)
        print(f"Saved {vectors.shape} embeddings to {args.output}")
    else:
        print(json.dumps(vectors.tolist()))


if __name__ == "__main__":
    main()
