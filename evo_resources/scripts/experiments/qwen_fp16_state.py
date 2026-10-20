"""Test FP16 GDN cache storage with explicit casts; not a native FP16 GDN kernel."""

import argparse
import json
import shutil
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from evo.inspect import inspect_model
from evo.pipeline import hash_file, write_json


def rewrite(model):
    inputs = {v.name: v for v in model.graph.input}
    outputs = {v.name: v for v in model.graph.output}
    original_names = {v for n in model.graph.node for v in (*n.input, *n.output)}
    states = set()
    replacement = []
    layers = 0
    for node in model.graph.node:
        if (node.domain, node.op_type) != ("com.microsoft", "GatedDeltaNet"):
            replacement.append(node)
            continue
        if len(node.input) < 7 or len(node.output) != 2:
            raise ValueError("Expected GDN with explicit initial/final recurrent state")
        past, present = node.input[6], node.output[1]
        if past not in inputs or present not in outputs:
            raise ValueError("Require graph input/output recurrent states")
        for value in (inputs[past], outputs[present]):
            tensor = value.type.tensor_type
            dims = [d.dim_value for d in tensor.shape.dim]
            if tensor.elem_type != TensorProto.FLOAT or dims[1:] != [32, 128, 128]:
                raise ValueError("Expected original FP32 Qwen3.5-4B V-major state")
        if sum(past in n.input for n in model.graph.node) != 1:
            raise ValueError("Initial state has another consumer")
        if any(present in n.input for n in model.graph.node):
            raise ValueError("Final state has an internal consumer")
        if past in states or present in states:
            raise ValueError("State is shared by multiple recurrent nodes")
        states.update((past, present))
        internal_in, internal_out = past + ".evo_fp32", present + ".evo_fp32"
        if internal_in in original_names or internal_out in original_names:
            raise ValueError("Tensor name collision")
        updated = onnx.NodeProto()
        updated.CopyFrom(node)
        updated.input[6], updated.output[1] = internal_in, internal_out
        replacement.extend(
            [
                helper.make_node(
                    "Cast",
                    [past],
                    [internal_in],
                    to=TensorProto.FLOAT,
                    name=node.name + "/LoadFP16State",
                ),
                updated,
                helper.make_node(
                    "Cast",
                    [internal_out],
                    [present],
                    to=TensorProto.FLOAT16,
                    name=node.name + "/StoreFP16State",
                ),
            ]
        )
        layers += 1
    if layers != 24:
        raise ValueError(f"Expected 24 Qwen recurrent layers, found {layers}")
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        if value.name in states:
            value.type.tensor_type.elem_type = TensorProto.FLOAT16
    del model.graph.node[:]
    model.graph.node.extend(replacement)
    return layers


def prepare(source, output):
    source, output = source.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Use a fresh output directory outside the source")
    if inspect_model(source / "model.onnx")["structure"] != "passed":
        raise ValueError("Source failed inspection")
    model = onnx.load(source / "model.onnx", load_external_data=False)
    layers = rewrite(model)
    # Copy runtime assets, not source validation manifests that describe another graph.
    output.mkdir(parents=True)
    for name in (
        "model.onnx.data",
        "genai_config.json",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        if (source / name).exists():
            shutil.copyfile(source / name, output / name)
    onnx.save(model, output / "model.onnx")
    report = {
        "status": "experimental",
        "source": str(source),
        "source_sha256": hash_file(source / "model.onnx"),
        "model_sha256": hash_file(output / "model.onnx"),
        "layers": layers,
        "state_contract": "FP16 V-major cache, FP32 GDN computation via 48 explicit casts",
        "limitations": (
            "Tests boundary rounding and cast overhead, not native FP16 GDN performance; "
            "within-prefill recurrent accumulation remains FP32"
        ),
        "cache_bytes_saved_per_buffer_set_batch1": layers * 32 * 128 * 128 * 2,
        "quality": "pending",
        "performance": "pending",
        "structure": inspect_model(output / "model.onnx")["structure"],
    }
    write_json(output / "fp16-state-experiment.json", report)
    if report["structure"] != "passed":
        raise ValueError("Candidate failed structural inspection")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
