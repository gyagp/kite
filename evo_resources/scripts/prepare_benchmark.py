"""Prepare deterministic text workloads using the source tokenizer and template."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from evo.pipeline import hash_file

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
cases = []
for count in (4, 30, 120):
    text = (
        "A team is building an efficient inference engine. They measure latency, memory use, "
        "and output accuracy on the same workloads before choosing optimizations. "
    ) * count
    text += "\nSummarize the passage in three concise sentences."
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    cases.append({"name": f"text-{len(ids)}", "input_ids": ids, "text": text})
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(
    json.dumps(
        {
            "tokenizer_sha256": hash_file(args.checkpoint / "tokenizer.json"),
            "enable_thinking": False,
            "cases": cases,
        },
        indent=2,
    ),
    encoding="utf-8",
)
print([(c["name"], len(c["input_ids"])) for c in cases])
