"""Selectively quantize constant MatMul weights in an existing ONNX package."""

import argparse
import re
import shutil
from pathlib import Path

import onnx

from evo.inspect import inspect_model
from evo.pipeline import hash_file, write_json


def quantize(source, output, node_regex, bits=8, block_size=32):
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig,
        DefaultWeightOnlyQuantizer,
    )

    source, output = source.resolve(), output.resolve()
    if bits not in {4, 8} or block_size not in {16, 32, 64, 128, 256}:
        raise ValueError("Unsupported bit width or block size")
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Choose a new output directory outside the source")
    pattern = re.compile(node_regex)
    model_path = source / "model.onnx"
    if inspect_model(model_path)["structure"] != "passed":
        raise ValueError("Source must pass structural inspection")
    model = onnx.load(model_path, load_external_data=False)
    tensors = {t.name: t for t in model.graph.initializer}
    selected = [
        n
        for n in model.graph.node
        if n.op_type == "MatMul" and not n.domain and pattern.search(n.name)
    ]
    if not selected:
        raise ValueError("No constant MatMul nodes matched")
    quantizer = DefaultWeightOnlyQuantizer(
        DefaultWeightOnlyQuantConfig(
            bits=bits,
            block_size=block_size,
            is_symmetric=True,
            accuracy_level=4,
        )
    )
    changes = []
    old_weights = set()
    quantized_weights = {}
    for node in selected:
        tensor = tensors.get(node.input[1])
        if (
            tensor is None
            or len(tensor.dims) != 2
            or tensor.data_type not in {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16}
        ):
            raise ValueError(f"{node.name}: require a constant FP16/FP32 matrix")
        weight = onnx.numpy_helper.to_array(tensor, base_dir=str(source))
        weight_name, scale_name = tensor.name + f".evo_q{bits}", tensor.name + ".evo_scales"
        if weight_name in tensors or scale_name in tensors:
            raise ValueError("Quantized initializer name collision")
        if tensor.name not in quantized_weights:
            packed, scales, _ = quantizer.qbits_block_quant(weight)
            model.graph.initializer.extend(
                [
                    onnx.numpy_helper.from_array(packed, weight_name),
                    onnx.numpy_helper.from_array(scales, scale_name),
                ]
            )
            quantized_weights[tensor.name] = True
        replacement = onnx.helper.make_node(
            "MatMulNBits",
            [node.input[0], weight_name, scale_name],
            list(node.output),
            name=node.name + f"_EvoQ{bits}",
            domain="com.microsoft",
            K=weight.shape[0],
            N=weight.shape[1],
            bits=bits,
            block_size=block_size,
            accuracy_level=4,
        )
        changes.append({"node": node.name, "shape": list(weight.shape), "bits": bits})
        old_weights.add(tensor.name)
        node.CopyFrom(replacement)
    from evo.inspect import _graphs

    used = {name for graph in _graphs(model.graph) for node in graph.node for name in node.input}
    used.update(v.name for v in (*model.graph.input, *model.graph.output))
    keep = [t for t in model.graph.initializer if t.name not in old_weights or t.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    if not any(o.domain == "com.microsoft" for o in model.opset_import):
        model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    shutil.copytree(source, output)
    if (output / "evo-manifest.json").exists():
        (output / "evo-manifest.json").replace(output / "source-export-manifest.json")
    onnx.save(model, output / "model.onnx")
    report = {
        "source": str(source),
        "source_sha256": hash_file(model_path),
        "changes": changes,
        "model_sha256": hash_file(output / "model.onnx"),
        "block_size": block_size,
        "quality": "requires validation",
        "performance": "requires measurement",
    }
    write_json(output / "selective-quantization.json", report)
    write_json(output / "evo-manifest.json", {**report, "status": "quantized"})
    if inspect_model(output / "model.onnx")["structure"] != "passed":
        raise ValueError("Quantized graph failed structural inspection")
    print(f"Quantized {len(changes)} matrices to INT{bits}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node-regex", required=True)
    parser.add_argument("--bits", type=int, choices=[4, 8], default=8)
    parser.add_argument("--block-size", type=int, choices=[16, 32, 64, 128, 256], default=32)
    args = parser.parse_args(argv)
    quantize(args.source, args.output, args.node_regex, args.bits, args.block_size)


if __name__ == "__main__":
    main()
