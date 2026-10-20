"""Export the EmbeddingGemma 2 text encoder with masked mean pooling and L2 norm."""

import argparse
import json
import shutil
import sys
from pathlib import Path

from evo.pipeline import hash_file, write_json


def load_text_model(source, dtype="fp32"):
    import torch
    from safetensors import safe_open
    from transformers import EmbeddingGemma2TextConfig, EmbeddingGemma2TextModel

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "embedding_gemma2":
        raise ValueError("Expected an EmbeddingGemma 2 checkpoint")
    text_config = EmbeddingGemma2TextConfig(**config["text_config"])
    text_config._attn_implementation = "eager"
    model = EmbeddingGemma2TextModel(text_config)
    weights = {}
    with safe_open(source / "model.safetensors", framework="pt") as checkpoint:
        for key in checkpoint.keys():
            if key.startswith("language_model."):
                weights[key.removeprefix("language_model.")] = checkpoint.get_tensor(key).to(
                    torch.float32 if dtype == "fp32" else torch.float16
                )
    model.load_state_dict(weights, strict=True, assign=True)
    # RoPE buffers are non-persistent and must be initialized off the meta device.
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import (
        EmbeddingGemma2RotaryEmbedding,
    )

    model.rotary_emb = EmbeddingGemma2RotaryEmbedding(text_config)
    return model.eval()


def wrap_text_model(model):
    import torch

    class TextEmbedding(torch.nn.Module):
        def __init__(self, text):
            super().__init__()
            self.text = text

        def forward(self, input_ids, attention_mask):
            # Explicit dynamic bidirectional masks avoid tracing HF's mask dispatch.
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            valid = attention_mask[:, None, None, :].bool()
            nearby = (
                positions[:, None] - positions[None, :]
            ).abs() <= self.text.config.sliding_window
            dtype = self.text.embed_tokens.weight.dtype
            zero = torch.tensor(0.0, dtype=dtype, device=input_ids.device)
            blocked = torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=input_ids.device)
            masks = {
                "full_attention": torch.where(valid, zero, blocked),
                "sliding_attention": torch.where(valid & nearby[None, None, :, :], zero, blocked),
            }
            hidden = self.text(
                input_ids=input_ids,
                attention_mask=masks,
                position_ids=positions[None, :],
            ).last_hidden_state.float()
            weights = attention_mask[:, :, None].float()
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1)
            return torch.nn.functional.normalize(pooled, p=2, dim=-1, eps=1e-12)

    return TextEmbedding(model).eval()


def fuse_native_norms(model):
    """Emit ORT's RMSNorm, retaining the source FP32 accumulation contract."""
    import torch
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import (
        EmbeddingGemma2RMSNorm,
    )

    class NativeRMS(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, scale, epsilon):
            x = value.float()
            return (x * (x.square().mean(-1, keepdim=True) + epsilon).pow(-0.5) * scale.float()).to(
                value.dtype
            )

        @staticmethod
        def symbolic(g, value, scale, epsilon):
            return g.op(
                "SimplifiedLayerNormalization",
                value,
                scale,
                axis_i=-1,
                epsilon_f=epsilon,
                stash_type_i=1,
            ).setType(value.type())

    class Norm(torch.nn.Module):
        def __init__(self, original, dim):
            super().__init__()
            self.epsilon = original.eps
            self.register_buffer(
                "scale",
                original.weight.detach()
                if original.with_scale
                else torch.ones(dim, dtype=next(model.parameters()).dtype),
            )

        def forward(self, value):
            return NativeRMS.apply(value, self.scale, self.epsilon)

    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, EmbeddingGemma2RMSNorm):
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name)
            dim = parent.head_dim if child_name == "v_norm" else module.weight.numel()
            setattr(parent, child_name, Norm(module, dim))
            count += 1
    return count


def export(source: Path, output: Path, precision="fp32", native_norms=False):
    import onnx
    import torch
    import transformers
    from transformers import AutoTokenizer

    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Choose a new output directory")
    torch.set_num_threads(8)
    text = load_text_model(source, precision)
    model = wrap_text_model(text)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    example = tokenizer(
        [
            "task: search result | query: What causes the northern lights?",
            "title: none | text: A cat.",
        ],
        padding=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        reference_hidden = text(**example).last_hidden_state.float()
        mask = example["attention_mask"][:, :, None].float()
        reference = torch.nn.functional.normalize(
            (reference_hidden * mask).sum(1) / mask.sum(1), dim=-1
        )
        actual = model(example["input_ids"], example["attention_mask"])
    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
    norm_count = fuse_native_norms(text) if native_norms else 0
    output.mkdir(parents=True)
    print(f"Exporting {precision} text encoder", flush=True)
    torch.onnx.export(
        model,
        (example["input_ids"], example["attention_mask"]),
        str(output / "model.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["sentence_embedding"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "sentence_embedding": {0: "batch_size"},
        },
        opset_version=22,
        dynamo=False,
    )
    graph = onnx.load(output / "model.onnx")
    graph.graph.output[0].type.tensor_type.shape.dim[1].dim_value = text.config.embedding_dim
    onnx.save_model(
        graph,
        output / "model.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        size_threshold=1024,
    )
    del graph
    tokenizer.save_pretrained(output)
    for filename in ("config_sentence_transformers.json", "evo-source.json"):
        if (source / filename).exists():
            shutil.copy2(source / filename, output / filename)
    source_data = json.loads((source / "evo-source.json").read_text())
    manifest = {
        "task": "text-embedding",
        "architecture": "EmbeddingGemma2TextModel",
        "source": source_data,
        "source_weights_sha256": hash_file(source / "model.safetensors"),
        "precision": precision,
        "embedding_dimensions": 768,
        "max_length": 8192,
        "native_rms_norms": norm_count,
        "inputs": {"input_ids": "int64[B,S]", "attention_mask": "int64[B,S]"},
        "outputs": {"sentence_embedding": "float32[B,768], L2 normalized"},
        "pooling": "mean over non-padding tokens, includes task prefix",
        "attention": "bidirectional; sliding window inclusive radius 512; no KV cache",
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "validation": (
            "wrapper matched original text forward on padded two-item batch; "
            "ONNX validation pending"
        ),
        "model_sha256": hash_file(output / "model.onnx"),
    }
    write_json(output / "embedding_config.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--transformers-path", type=Path)
    parser.add_argument("--native-norms", action="store_true")
    parser.add_argument(
        "--multimodal", action="store_true", help="Export text, vision and audio graphs"
    )
    args = parser.parse_args(argv)
    if args.transformers_path:
        sys.path.insert(0, str(args.transformers_path.resolve()))
    if args.multimodal:
        from evo.embedding_multimodal import export as export_multimodal

        export_multimodal(args.source, args.output, args.precision)
    else:
        export(args.source, args.output, args.precision, args.native_norms)


if __name__ == "__main__":
    main()
