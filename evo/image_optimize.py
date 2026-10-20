"""Qwen-Image graph rewrites preserving weight values and RMS rounding points."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from evo.pipeline import hash_file, write_json


def fuse_qk_rms(model):
    producers = {out: n for n in model.graph.node for out in n.output}
    initializers = {t.name: t for t in model.graph.initializer}
    replacements = {}
    added = {}
    for node in model.graph.node:
        if node.op_type != "Mul" or len(node.input) != 2:
            continue
        gamma = next(
            (x for x in node.input if x.endswith(("attn.norm_q.weight", "attn.norm_k.weight"))),
            None,
        )
        if gamma is None or gamma not in initializers:
            continue
        cast = producers.get(next(x for x in node.input if x != gamma))
        if (
            cast is None
            or cast.op_type != "Cast"
            or helper.get_attribute_value(cast.attribute[0]) != onnx.TensorProto.FLOAT16
        ):
            continue
        multiply = producers.get(cast.input[0])
        if multiply is None or multiply.op_type != "Mul":
            continue
        left, right = [producers.get(x) for x in multiply.input]
        if left is None or right is None or left.op_type != "Cast" or right.op_type != "Div":
            continue
        sqrt = producers.get(right.input[1])
        if sqrt is None or sqrt.op_type != "Sqrt":
            continue
        add = producers.get(sqrt.input[0])
        if add is None or add.op_type != "Add":
            continue
        mean, epsilon_node = [producers.get(x) for x in add.input]
        if (
            mean is None
            or epsilon_node is None
            or mean.op_type != "ReduceMean"
            or epsilon_node.op_type != "Constant"
        ):
            continue
        power = producers.get(mean.input[0])
        if power is None or power.op_type != "Pow":
            continue
        power_input = producers.get(power.input[0])
        if (
            power_input is None
            or power_input.op_type != "Cast"
            or power_input.input[0] != left.input[0]
        ):
            continue
        exponent = numpy_helper.to_array(
            helper.get_attribute_value(producers[power.input[1]].attribute[0])
        )
        axes = numpy_helper.to_array(
            helper.get_attribute_value(producers[mean.input[1]].attribute[0])
        )
        if float(exponent) != 2 or axes.tolist() != [-1]:
            continue
        epsilon = float(
            numpy_helper.to_array(helper.get_attribute_value(epsilon_node.attribute[0]))
        )
        size = initializers[gamma].dims[0]
        name = f"evo.rms.ones.{size}"
        added[name] = numpy_helper.from_array(np.ones(size, np.float32), name)
        # FP16 input, FP32 output: keep inverse RMS and multiplication in FP32.
        # Retain the source Cast-to-FP16 followed by learned-scale multiplication.
        replacements[multiply.name] = helper.make_node(
            "SimplifiedLayerNormalization",
            [left.input[0], name],
            list(multiply.output),
            name=multiply.name + ".fused",
            axis=-1,
            epsilon=epsilon,
            stash_type=1,
        )
    nodes = [replacements.get(n.name, n) for n in model.graph.node]
    live = {v.name for v in model.graph.output}
    kept = []
    for node in reversed(nodes):
        if live.intersection(node.output):
            kept.append(node)
            live.update(x for x in node.input if x)
    initializers.update(added)
    tensors = [t for name, t in initializers.items() if name in live]
    del model.graph.node[:]
    model.graph.node.extend(reversed(kept))
    del model.graph.initializer[:]
    model.graph.initializer.extend(tensors)
    return len(replacements)


def optimize(package):
    package = Path(package)
    manifest = json.loads((package / "image_config.json").read_text())
    counts = {}
    for part in ("prefix", "denoise"):
        model = onnx.load(package / f"{part}-capture.onnx", load_external_data=False)
        counts[part] = fuse_qk_rms(model)
        expected = manifest["transformer_config"]["num_layers"] * 2 - (part == "prefix")
        if counts[part] != expected:
            raise ValueError("Unexpected Q/K normalization pattern count")
        path = package / f"{part}-rms.onnx"
        onnx.save_model(model, path)
        manifest["artifacts"][path.name] = {"sha256": hash_file(path), "bytes": path.stat().st_size}
    manifest["rms_fusion"] = {
        "counts": counts,
        "rounding": "FP32 normalization, FP16 cast, FP16 learned scale",
    }
    write_json(package / "image_config.json", manifest)
    return counts


def promote_short_blocks(package):
    """Materialize the lone Q6_K/16-block matrix for hardware matrix kernels.

    Its quantized weight values are retained; this changes their storage and
    dispatch, not the learned parameters or their quantization decisions.
    """
    package = Path(package)
    saved = {}
    with (package / "short-blocks.data").open("wb") as stream:
        for part in ("prefix", "denoise"):
            model = onnx.load(package / f"{part}-capture.onnx", load_external_data=False)
            tensors = {t.name: t for t in model.graph.initializer}
            nodes = []
            for node in model.graph.node:
                attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
                if (
                    node.op_type == "MatMulNBits"
                    and attrs.get("bits") == 8
                    and attrs.get("block_size") == 16
                    and len(node.input) == 3
                ):
                    name = node.input[1] + ".dense"
                    if name not in saved:
                        q = (
                            numpy_helper.to_array(
                                tensors[node.input[1]], base_dir=str(package)
                            ).astype(np.float32)
                            - 128
                        )
                        scales = numpy_helper.to_array(
                            tensors[node.input[2]], base_dir=str(package)
                        ).astype(np.float32)
                        n, k = attrs["N"], attrs["K"]
                        weight = (
                            (q * scales.reshape(n, -1, 1)).reshape(n, k).T.astype(np.float16).copy()
                        )
                        tensor = onnx.TensorProto(
                            name=name, data_type=onnx.TensorProto.FLOAT16, dims=weight.shape
                        )
                        tensor.data_location = onnx.TensorProto.EXTERNAL
                        for key, value in {
                            "location": "short-blocks.data",
                            "offset": str(stream.tell()),
                            "length": str(weight.nbytes),
                        }.items():
                            entry = tensor.external_data.add()
                            entry.key = key
                            entry.value = value
                        stream.write(weight.tobytes())
                        saved[name] = tensor
                    nodes.append(
                        helper.make_node(
                            "MatMul",
                            [node.input[0], name],
                            list(node.output),
                            name=node.name + ".dense",
                        )
                    )
                else:
                    nodes.append(node)
            used = {name for node in nodes for name in node.input}
            tensors.update(saved)
            del model.graph.node[:]
            model.graph.node.extend(nodes)
            del model.graph.initializer[:]
            model.graph.initializer.extend(t for name, t in tensors.items() if name in used)
            onnx.save_model(model, package / f"{part}-short-blocks.onnx")
    if not saved:
        raise ValueError("No eligible short-block matrices")
    manifest = json.loads((package / "image_config.json").read_text())
    manifest["short_block_promotions"] = list(saved)
    for name in ("prefix-short-blocks.onnx", "denoise-short-blocks.onnx", "short-blocks.data"):
        manifest["artifacts"][name] = {
            "sha256": hash_file(package / name),
            "bytes": (package / name).stat().st_size,
        }
    write_json(package / "image_config.json", manifest)
    return list(saved)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument(
        "--short-blocks", action="store_true", help="Promote Q6_K short blocks to FP16"
    )
    parser.add_argument(
        "--merge-projections",
        type=Path,
        metavar="OUTPUT",
        help="Merge parallel projections into a compact new package",
    )
    parser.add_argument(
        "--base-variant", default="short-blocks", choices=("capture", "short-blocks")
    )
    args = parser.parse_args(argv)
    if args.merge_projections:
        from evo.image_projections import merge_projections

        print(
            json.dumps(merge_projections(args.package, args.merge_projections, args.base_variant))
        )
        return
    print(
        json.dumps(
            promote_short_blocks(args.package) if args.short_blocks else optimize(args.package)
        )
    )


if __name__ == "__main__":
    main()
