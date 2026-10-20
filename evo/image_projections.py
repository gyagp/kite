"""Merge parallel projections without requantizing their weights."""

import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from evo.pipeline import hash_file, write_json


def merge_projections(package, output, variant="short-blocks"):
    """Restore combined gate/up and combine compatible Q/K/V projections.

    Matching MatMulNBits formats and a common activation are required. Split
    preserves the original consumers; all packed codes/scales/zeros are retained.
    """
    package, output = Path(package).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a new output directory")
    models = {
        part: onnx.load(package / f"{part}-{variant}.onnx", load_external_data=False)
        for part in ("prefix", "denoise")
    }
    output.mkdir(parents=True)
    saved, counts = {}, {}
    with (output / "weights.data").open("wb") as stream:

        def save_tensor(name, array):
            tensor = numpy_helper.from_array(array, name)
            raw = tensor.raw_data
            tensor.ClearField("raw_data")
            tensor.data_location = onnx.TensorProto.EXTERNAL
            for key, value in {
                "location": "weights.data",
                "offset": str(stream.tell()),
                "length": str(len(raw)),
            }.items():
                entry = tensor.external_data.add()
                entry.key, entry.value = key, value
            stream.write(raw)
            saved[name] = tensor
            return tensor

        for part, model in models.items():
            tensors = {t.name: t for t in model.graph.initializer}
            groups = defaultdict(list)
            for index, node in enumerate(model.graph.node):
                if (
                    node.domain != "com.microsoft"
                    or node.op_type != "MatMulNBits"
                    or len(node.input) not in (3, 4)
                ):
                    continue
                if any(name not in tensors for name in node.input[1:]):
                    continue
                attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
                signature = (
                    node.input[0],
                    tuple(sorted((k, v) for k, v in attrs.items() if k != "N")),
                    tuple(
                        (tensors[name].data_type, tuple(tensors[name].dims[1:]))
                        for name in node.input[1:]
                    ),
                )
                groups[signature].append((index, node, attrs))
            replacements, removed = {}, set()
            merged_count = 0
            for group in groups.values():
                if len(group) < 2:
                    continue
                first_index, first, attrs = group[0]
                merged = copy.deepcopy(first)
                merged.name += ".merged"
                merged.output[0] += ".merged"
                for attr in merged.attribute:
                    if attr.name == "N":
                        attr.i = sum(a["N"] for _, _, a in group)
                for column in range(1, len(first.input)):
                    names = [node.input[column] for _, node, _ in group]
                    name = "evo.merged." + "__".join(names)
                    if name not in saved:
                        arrays = [
                            numpy_helper.to_array(tensors[n], base_dir=str(package)) for n in names
                        ]
                        save_tensor(name, np.concatenate(arrays, axis=0))
                    tensors[name] = saved[name]
                    merged.input[column] = name
                splits = merged.name + ".sizes"
                tensors[splits] = numpy_helper.from_array(
                    np.array([a["N"] for _, _, a in group], np.int64), splits
                )
                split = helper.make_node(
                    "Split",
                    [merged.output[0], splits],
                    [node.output[0] for _, node, _ in group],
                    name=merged.name + ".split",
                    axis=-1,
                )
                replacements[first_index] = [merged, split]
                removed.update(index for index, _, _ in group[1:])
                merged_count += len(group) - 1
            nodes = []
            for index, node in enumerate(model.graph.node):
                if index not in removed:
                    nodes.extend(replacements.get(index, [node]))
            used = {name for node in nodes for name in node.input}
            kept = []
            for name, tensor in tensors.items():
                if name not in used:
                    continue
                if tensor.data_location == onnx.TensorProto.EXTERNAL:
                    if name not in saved:
                        save_tensor(name, numpy_helper.to_array(tensor, base_dir=str(package)))
                    kept.append(saved[name])
                else:
                    kept.append(tensor)
            del model.graph.node[:]
            model.graph.node.extend(nodes)
            del model.graph.initializer[:]
            model.graph.initializer.extend(kept)
            onnx.save_model(model, output / f"{part}-projections.onnx")
            counts[part] = merged_count
    manifest = json.loads((package / "image_config.json").read_text())
    manifest["projection_merge"] = {
        "source": str(package),
        "variant": variant,
        "eliminated_projections": counts,
    }
    manifest["validation"] = "pending"
    manifest["artifacts"] = {
        name: {"sha256": hash_file(output / name), "bytes": (output / name).stat().st_size}
        for name in ("prefix-projections.onnx", "denoise-projections.onnx", "weights.data")
    }
    write_json(output / "image_config.json", manifest)
    return counts
