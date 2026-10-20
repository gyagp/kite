"""Experimental mask rewrite; did not resolve capture quality drift. Not recommended."""

import argparse
import json
import shutil
from pathlib import Path

import onnx
from onnx import numpy_helper

from evo.inspect import inspect_model
from evo.pipeline import hash_file, write_json


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Use a fresh output directory")
    config = json.loads((source / "genai_config.json").read_text())
    if config["model"]["type"] != "qwen3_5_text":
        raise ValueError("This preparation is scoped to Qwen3.5 text generation")
    model = onnx.load(source / "model.onnx", load_external_data=False)
    nodes = {v: n for n in model.graph.node for v in n.output}
    constants = {t.name: t for t in model.graph.initializer}
    constants.update(
        {
            v: onnx.helper.get_attribute_value(n.attribute[0])
            for n in model.graph.node
            if n.op_type == "Constant" and n.attribute and n.attribute[0].name == "value"
            for v in n.output
        }
    )
    attentions = [
        n
        for n in model.graph.node
        if n.domain == "com.microsoft" and n.op_type == "GroupQueryAttention"
    ]
    total_names = {n.input[6] for n in attentions}
    seq_names = {n.input[5] for n in attentions}
    if len(attentions) != 8 or len(total_names) != 1 or len(seq_names) != 1:
        raise ValueError("Expected Qwen3.5-4B's eight attention layers with shared lengths")
    total = nodes[next(iter(total_names))]
    seq = nodes[next(iter(seq_names))]
    if total.op_type != "ReduceMax" or seq.op_type != "Sub":
        raise ValueError("Unrecognized attention length preparation")
    valid = nodes[seq.input[0]]
    padded = nodes[total.input[0]]
    if valid.op_type != "ReduceSum" or padded.op_type != "ReduceSum":
        raise ValueError("Expected reduced valid and padded lengths")
    ones = nodes[padded.input[0]]
    if ones.op_type != "Add" or ones.input[1] not in constants:
        raise ValueError("Expected mask-shaped ones")
    one = numpy_helper.to_array(constants[ones.input[1]], base_dir=str(source))
    if one.tolist() != [1]:
        raise ValueError("Expected mask-shaped ones")
    zeros = nodes[ones.input[0]]
    if zeros.op_type != "Sub" or list(zeros.input) != [valid.input[0], valid.input[0]]:
        raise ValueError("Expected mask-shaped zeros")
    cast = nodes[valid.input[0]]
    if cast.op_type != "Cast" or list(cast.input) != ["attention_mask"]:
        raise ValueError("Expected attention mask cast")
    # GenAI capture keeps a fixed-size mask with zero padding. Shape-derived length
    # includes this padding; ReduceMax(ReduceSum(mask)) preserves the live length.
    total.input[0] = valid.output[0]
    removed = {n.name for n in (padded, ones, zeros)}
    dead = {v for n in (padded, ones, zeros) for v in n.output}
    if any(v in dead for n in model.graph.node if n.name not in removed for v in n.input):
        raise ValueError("Padded-length intermediates have another consumer")
    kept = [n for n in model.graph.node if n.name not in removed]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    values = [v for v in model.graph.value_info if v.name not in dead]
    del model.graph.value_info[:]
    model.graph.value_info.extend(values)
    shutil.copytree(source, output)
    onnx.save(model, output / "model.onnx")
    for options in config["model"]["decoder"]["session_options"]["provider_options"]:
        for name, values in options.items():
            if name.lower() == "webgpu":
                values["enableGraphCapture"] = "1"
    config["search"]["max_length"] = 8192
    write_json(output / "genai_config.json", config)
    inspection = inspect_model(output / "model.onnx")
    report = {
        "source": str(source),
        "source_sha256": hash_file(source / "model.onnx"),
        "model_sha256": hash_file(output / "model.onnx"),
        "transformation": "Padded mask length -> maximum valid attention length",
        "scope": "Batch-one unpadded text prompts; static zero-padded capture mask",
        "reference": "GenAI WebGPU.make_attention_mask_graph_capture_reformatting_for_gqa",
        "structure": inspection["structure"],
        "quality": "pending",
        "performance": "pending",
    }
    write_json(output / "capture-export.json", report)
    if inspection["structure"] != "passed":
        raise ValueError(inspection["errors"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
