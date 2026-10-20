"""Experimental Qwen3.5 LinearAttention-to-GatedDeltaNet export with explicit state ABI."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from evo.inspect import inspect_model
from evo.pipeline import hash_file, write_json


def convert(source, checkpoint, output, external_norm=False):
    import torch
    from safetensors import safe_open

    source, checkpoint, output = [Path(p).resolve() for p in (source, checkpoint, output)]
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Choose a new output directory outside the source")
    if inspect_model(source / "model.onnx")["structure"] != "passed":
        raise ValueError("Source must pass structural inspection")
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("model_type") != "qwen3_5":
        raise ValueError("Require the matching original Qwen3.5 checkpoint")
    parameters = {}
    for path in checkpoint.glob("*.safetensors"):
        with safe_open(path, framework="pt") as tensors:
            for key in tensors.keys():
                if ".linear_attn." in key and key.endswith((".A_log", ".dt_bias")):
                    parameters[key.split("language_model.")[-1]] = (
                        tensors.get_tensor(key).float().numpy()
                    )
    model = onnx.load(source / "model.onnx", load_external_data=False)
    by_name = {n.name: n for n in model.graph.node}
    initializers = {t.name: t for t in model.graph.initializer}
    constants = {
        o: helper.get_attribute_value(n.attribute[0])
        for n in model.graph.node
        if n.op_type == "Constant" and n.attribute[0].name == "value"
        for o in n.output
    }
    replacements, removed, additions, layers = {}, set(), [], []
    for node in model.graph.node:
        if (node.domain, node.op_type) != ("com.microsoft", "LinearAttention"):
            continue
        root = node.name.rsplit("/", 1)[0]
        layer = int(root.split("layers.")[1].split("/")[0])
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if attrs != {
            "q_num_heads": 16,
            "kv_num_heads": 32,
            "update_rule": b"gated_delta",
            "scale": 1.0,
            "state_window": 0,
        }:
            raise ValueError(f"Unsupported LinearAttention contract: {node.name}")
        gate = by_name[root + "/LinearAttentionGate"]
        q_flat = by_name[root + "/q_l2norm/flat/Reshape"]
        k_flat = by_name[root + "/k_l2norm/flat/Reshape"]
        split = by_name[root + "/split_qkv/Split"]
        q_scaled = by_name[root + "/q_scaled/Mul"]
        if list(node.input[4:]) != list(gate.output) or node.input[0] != q_scaled.output[0]:
            raise ValueError("Unexpected gate or query-scale consumers")
        for name in ("q", "k"):
            norm = by_name[root + f"/{name}_l2norm/LpNormalization"]
            na = {a.name: helper.get_attribute_value(a) for a in norm.attribute}
            if na != {"axis": -1, "p": 2}:
                raise ValueError("Expected per-head L2 normalization")

        def tensor(name):
            return numpy_helper.to_array(
                initializers[name] if name in initializers else constants[name],
                base_dir=str(source),
            )

        if not np.array_equal(tensor(q_flat.input[1]), [0, 0, 16, 128]) or not np.array_equal(
            tensor(split.input[1]), [2048, 2048, 4096]
        ):
            raise ValueError("Unsupported head dimensions")
        suffix = f"layers.{layer}.linear_attn."
        a_log = next(v for k, v in parameters.items() if k.endswith(suffix + "A_log"))
        dt_bias = next(v for k, v in parameters.items() if k.endswith(suffix + "dt_bias"))
        # Verify this checkpoint really supplied the source gate parameters.
        saved_bias, saved_decay = tensor(gate.input[1]), tensor(gate.input[2])
        exp_a = -torch.from_numpy(a_log).exp()
        decay_forms = [
            exp_a.to(dtype).float().numpy().astype(saved_decay.dtype)
            for dtype in (torch.float32, torch.float16, torch.bfloat16)
        ]
        if not np.array_equal(saved_bias, dt_bias.astype(saved_bias.dtype)) or not any(
            np.array_equal(saved_decay, form) for form in decay_forms
        ):
            raise ValueError("Checkpoint gate parameters do not match the ONNX source")
        if not np.isclose(float(tensor(q_scaled.input[1])), 128**-0.5, rtol=0.001):
            raise ValueError("Unexpected query scaling")
        prefix = f"evo_gdn_{layer}"
        for name, values in [
            ("a_log", a_log),
            ("dt_bias", dt_bias),
            ("value_shape", np.array([0, 0, 32, 128], np.int64)),
            ("output_shape", np.array([0, 0, 4096], np.int64)),
        ]:
            if prefix + "_" + name in initializers:
                raise ValueError("GDN initializer collision")
            additions.append(numpy_helper.from_array(values, prefix + "_" + name))
        replacements[node.name] = [
            helper.make_node(
                "Reshape", [split.output[2], prefix + "_value_shape"], [prefix + "_value"]
            ),
            helper.make_node("Cast", [gate.input[0]], [prefix + "_decay"], to=TensorProto.FLOAT),
            helper.make_node("Cast", [gate.input[3]], [prefix + "_beta"], to=TensorProto.FLOAT),
            helper.make_node(
                "GatedDeltaNet",
                [
                    by_name[root + "/q_l2norm/LpNormalization"].output[0]
                    if external_norm
                    else q_flat.output[0],
                    by_name[root + "/k_l2norm/LpNormalization"].output[0]
                    if external_norm
                    else k_flat.output[0],
                    prefix + "_value",
                    "",
                    prefix + "_decay",
                    prefix + "_beta",
                    node.input[3],
                    prefix + "_a_log",
                    prefix + "_dt_bias",
                ],
                [prefix + "_output", node.output[1]],
                name=root + "/GatedDeltaNet",
                domain="com.microsoft",
                qk_l2_norm=0 if external_norm else 1,
                gate_activation="qwen",
                beta_activation="sigmoid",
                update_rule="gated_delta",
                scale=0.0,
                chunk_size=64,
            ),
            helper.make_node(
                "Reshape", [prefix + "_output", prefix + "_output_shape"], [node.output[0]]
            ),
        ]
        removed.update(
            [
                gate.name,
                q_scaled.name,
                root + "/q_l2norm/LpNormalization",
                root + "/k_l2norm/LpNormalization",
                root + "/q_l2norm/unflat/Reshape",
                root + "/k_l2norm/unflat/Reshape",
            ]
        )
        if external_norm:
            removed.difference_update(
                [root + "/q_l2norm/LpNormalization", root + "/k_l2norm/LpNormalization"]
            )
        layers.append(layer)
    if len(layers) != 24:
        raise ValueError(f"Expected 24 Qwen3.5-4B recurrent layers, found {len(layers)}")
    dead_outputs = {o for n in model.graph.node if n.name in removed for o in n.output}
    live_consumers = {
        i
        for n in model.graph.node
        if n.name not in removed and n.name not in replacements
        for i in n.input
    }
    live_consumers.update(v.name for v in model.graph.output)
    if dead_outputs & live_consumers:
        raise ValueError("A removed intermediate is exposed outside the attention pattern")
    nodes = []
    for node in model.graph.node:
        if node.name not in removed:
            nodes.extend(replacements.get(node.name, [node]))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(additions)
    for value in (*model.graph.input, *model.graph.output):
        if value.name.endswith(".recurrent"):
            shape = value.type.tensor_type.shape.dim
            if len(shape) != 4 or [d.dim_value for d in shape[1:]] != [32, 128, 128]:
                raise ValueError("Unexpected recurrent state dimensions")
            value.type.tensor_type.elem_type = TensorProto.FLOAT
    values = [
        v
        for v in model.graph.value_info
        if v.name not in dead_outputs and not v.name.endswith(".recurrent")
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(values)
    shutil.copytree(source, output)
    onnx.save(model, output / "model.onnx")
    inspection = inspect_model(output / "model.onnx")
    report = {
        "source": str(source),
        "source_sha256": hash_file(source / "model.onnx"),
        "model_sha256": hash_file(output / "model.onnx"),
        "checkpoint": str(checkpoint),
        "layers": layers,
        "state_contract": (
            "FP32 V-major [B,Hv,Dv,Dk]; requires fresh caches; "
            "incompatible with original FP16 K-major states"
        ),
        "required_kernel": (
            "Native WebGPU com.microsoft::GatedDeltaNet with Qwen gate and Q/K normalization"
        ),
        "structure": inspection["structure"],
        "quality": "pending",
        "performance": "pending",
        "external_norm": external_norm,
    }
    write_json(output / "gdn-export.json", report)
    if inspection["structure"] != "passed":
        raise ValueError(inspection["errors"])
    print(json.dumps(report, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external-norm", action="store_true")
    args = parser.parse_args(argv)
    convert(args.source, args.checkpoint, args.output, args.external_norm)


if __name__ == "__main__":
    main()
