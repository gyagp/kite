"""Exercise the real Qwen3.5 exporter on a tiny random checkpoint, without downloads.

This verifies export plumbing and hybrid graph construction, not pretrained model
quality or WebGPU inference. Run with an environment containing Evo's genai extra.
"""

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3_5Config, Qwen3_5ForConditionalGeneration

from evo import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory for smoke artifacts"
    )
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        raise ValueError(f"Choose a new directory: {root}")
    checkpoint = root / "checkpoint"
    checkpoint.mkdir(parents=True)
    torch.manual_seed(0)
    config = Qwen3_5Config(
        text_config={
            "vocab_size": 128,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 64,
            "max_position_embeddings": 128,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "mtp_num_hidden_layers": 0,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [2, 3, 3],
            },
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 128,
            "num_position_embeddings": 16,
        },
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
    )
    model = Qwen3_5ForConditionalGeneration(config).eval()
    model.save_pretrained(checkpoint)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f"tok{i}": i for i in range(128)}, unk_token="tok0")),
        unk_token="tok0",
        bos_token="tok1",
        eos_token="tok2",
        pad_token="tok0",
    )
    tokenizer.save_pretrained(checkpoint)
    manifest = build(str(checkpoint), source_kind="local", output=root / "export")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "models": manifest["models"],
                "elapsed_seconds": manifest["elapsed_seconds"],
                "checkpoint": "tiny randomly initialized Qwen3.5; not pretrained Qwen3.5-4B",
                "webgpu": manifest["webgpu"],
                "accuracy": manifest["accuracy"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
