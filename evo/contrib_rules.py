"""Conservative, byte-preserving projection slicing for ORT's fused WebGPU MLP."""

import copy

import numpy as np


def _int(node, key, default=None):
    return node.attributes[key].as_int() if key in node.attributes else default


def _single(value, op):
    nodes = list(value.consumers())
    return (
        nodes[0] if len(nodes) == 1 and nodes[0].domain == "" and nodes[0].op_type == op else None
    )


def mlp_patterns(model):
    """Match only symmetric INT4/block-32 packed SiLU MLPs with no exposed intermediates."""
    matches = []
    outputs = set(model.graph.outputs)
    for mm in model.graph:
        if mm.domain != "com.microsoft" or mm.op_type != "MatMulNBits" or len(mm.outputs) != 1:
            continue
        if (_int(mm, "bits", 4), _int(mm, "block_size"), _int(mm, "weight_prepacked", 0)) != (
            4,
            32,
            0,
        ):
            continue
        if len(mm.inputs) != 3 or any(v is None for v in mm.inputs):
            continue  # Explicit zero points, g_idx and bias need separate handling.
        if getattr(mm.inputs[0].dtype, "name", "") not in {"FLOAT16", "FLOAT"}:
            continue
        split = _single(mm.outputs[0], "Split")
        n, k = _int(mm, "N", 0), _int(mm, "K", 0)
        if split is None or len(split.outputs) != 2 or n <= 0 or n % 2 or k <= 0:
            continue
        if k % 4 or (n // 2) % 4:
            continue
        rank = len(mm.outputs[0].shape) if mm.outputs[0].shape is not None else None
        axis = _int(split, "axis", 0)
        if axis != -1 and (rank is None or axis != rank - 1):
            continue
        if len(split.inputs) > 1 and split.inputs[1] is not None:
            value = split.inputs[1].const_value
            if value is None or value.numpy().tolist() != [n // 2, n // 2]:
                continue
        gate, up = split.outputs
        sigmoid = next(
            (x for x in gate.consumers() if x.domain == "" and x.op_type == "Sigmoid"), None
        )
        if sigmoid is None or len(sigmoid.outputs) != 1:
            continue
        silu = _single(sigmoid.outputs[0], "Mul")
        if silu is None or set(silu.inputs) != {gate, sigmoid.outputs[0]}:
            continue
        product = _single(silu.outputs[0], "Mul")
        if product is None or set(product.inputs) != {silu.outputs[0], up}:
            continue
        if set(gate.consumers()) != {sigmoid, silu} or set(up.consumers()) != {product}:
            continue
        if any(
            v in outputs for v in (mm.outputs[0], gate, up, sigmoid.outputs[0], silu.outputs[0])
        ):
            continue
        b, scales = (v.const_value for v in mm.inputs[1:])
        blocks = (k + 31) // 32
        if b is None or scales is None or list(b.shape) != [n, blocks, 16]:
            continue
        if b.dtype.name != "UINT8" or scales.dtype != mm.inputs[0].dtype:
            continue
        if list(scales.shape) not in ([n, blocks], [n * blocks]):
            continue
        matches.append((mm, split, sigmoid, silu, product))
    return matches


def fuse_packed_mlp(model):
    """Emit com.microsoft::MatMulNBitsMlp without re-quantizing or changing I/O."""
    import onnx_ir as ir

    from evo.contrib import catalog

    patterns = mlp_patterns(model)
    if not patterns:
        return []
    known = {op["id"] for op in catalog()["operators"]}
    if "com.microsoft::MatMulNBitsMlp" not in known:
        raise ValueError("Installed ORT does not expose a MatMulNBitsMlp schema")
    changes = []
    for mm, split, sigmoid, silu, product in patterns:
        width = _int(mm, "N") // 2
        weights = []
        for part in range(2):
            for source in mm.inputs[1:3]:
                array = source.const_value.numpy()
                rows = array.reshape(_int(mm, "N"), -1)
                sliced = rows[part * width : (part + 1) * width]
                shape = [width, *array.shape[1:]] if array.ndim > 1 else [sliced.size]
                sliced = sliced.reshape(shape)
                name = source.name + f".evo_mlp_{part}"
                if name in model.graph.initializers:
                    raise ValueError(f"Initializer name collision: {name}")
                value = ir.Value(
                    name=name,
                    type=source.type,
                    shape=ir.Shape(shape),
                    const_value=ir.Tensor(np.ascontiguousarray(sliced)),
                )
                model.graph.register_initializer(value)
                weights.append(value)
        output = product.outputs[0]
        replacement = ir.Value(
            name=output.name, type=output.type, shape=copy.deepcopy(output.shape)
        )
        attrs = {
            key: copy.deepcopy(value)
            for key, value in mm.attributes.items()
            if key in {"K", "accuracy_level", "bits", "block_size"}
        }
        attrs["N"] = ir.AttrInt64("N", width)
        attrs["activation"] = ir.AttrString("activation", "silu")
        node = ir.Node(
            "com.microsoft",
            "MatMulNBitsMlp",
            [mm.inputs[0], None, None, weights[0], weights[1], None, weights[2], weights[3], None],
            attrs,
            outputs=[replacement],
            name=mm.name + "/EvoFusedMlp",
        )
        model.graph.insert_before(mm, node)
        output.replace_all_uses_with(replacement, replace_graph_outputs=True)
        model.graph.remove([product, silu, sigmoid, split, mm], safe=True)
        changes.append(
            {
                "source_node": mm.name,
                "operator": "com.microsoft::MatMulNBitsMlp",
                "K": _int(mm, "K"),
                "N": width,
                "policy": "symmetric INT4/block 32; packed weights sliced without requantization",
            }
        )
    model.graph.sort()
    return changes


def fuse_separate_mlp(model, *, dry_run=False):
    """Fuse separate symmetric gate/up projections, retaining their original tensors."""
    import onnx_ir as ir

    from evo.contrib import catalog

    known = {op["id"] for op in catalog()["operators"]}
    if "com.microsoft::MatMulNBitsMlp" not in known:
        raise ValueError("Installed ORT does not expose a MatMulNBitsMlp schema")
    changes = []
    for sigmoid in list(model.graph):
        if sigmoid.domain or sigmoid.op_type != "Sigmoid" or len(sigmoid.inputs) != 1:
            continue
        gate_value = sigmoid.inputs[0]
        gate = gate_value.producer() if gate_value is not None else None
        if gate is None or (gate.domain, gate.op_type) != ("com.microsoft", "MatMulNBits"):
            continue
        silu = _single(sigmoid.outputs[0], "Mul")
        if silu is None or set(silu.inputs) != {gate_value, sigmoid.outputs[0]}:
            continue
        product = _single(silu.outputs[0], "Mul")
        if product is None or len(product.inputs) != 2:
            continue
        up_value = next((v for v in product.inputs if v is not silu.outputs[0]), None)
        up = up_value.producer() if up_value is not None else None
        if up is None or (up.domain, up.op_type) != ("com.microsoft", "MatMulNBits"):
            continue
        if gate is up or len(gate.inputs) != 3 or len(up.inputs) != 3:
            continue
        if gate.inputs[0] is not up.inputs[0] or gate.inputs[0] is None:
            continue
        if set(gate_value.consumers()) != {sigmoid, silu} or set(up_value.consumers()) != {product}:
            continue
        if any(
            v in model.graph.outputs
            for v in (gate_value, up_value, sigmoid.outputs[0], silu.outputs[0])
        ):
            continue
        keys = {"K", "N", "bits", "block_size", "accuracy_level"}
        if set(gate.attributes) - keys or set(up.attributes) - keys:
            continue
        if any(_int(gate, k) != _int(up, k) for k in keys):
            continue
        k, n = _int(gate, "K", 0), _int(gate, "N", 0)
        if (_int(gate, "bits", 4), _int(gate, "block_size")) != (4, 32):
            continue
        if k <= 0 or n <= 0 or k % 4 or n % 4:
            continue
        dtype = gate.inputs[0].dtype
        if getattr(dtype, "name", "") not in {"FLOAT16", "FLOAT"}:
            continue
        blocks = (k + 31) // 32
        valid = True
        for projection in (gate, up):
            weight, scale = projection.inputs[1:]
            if (
                weight is None
                or scale is None
                or weight.const_value is None
                or scale.const_value is None
            ):
                valid = False
                break
            if (
                weight.dtype != ir.DataType.UINT8
                or scale.dtype != dtype
                or list(weight.shape) != [n, blocks, 16]
                or list(scale.shape) not in ([n, blocks], [n * blocks])
            ):
                valid = False
                break
        if not valid:
            continue
        change = {
            "source_node": gate.name,
            "operator": "com.microsoft::MatMulNBitsMlp",
            "K": k,
            "N": n,
            "policy": "Separate symmetric INT4/block-32 projections; original weights reused",
        }
        if dry_run:
            changes.append(change)
            continue
        old = product.outputs[0]
        replacement = ir.Value(name=old.name, type=old.type, shape=copy.deepcopy(old.shape))
        attrs = {key: copy.deepcopy(value) for key, value in gate.attributes.items()}
        attrs["activation"] = ir.AttrString("activation", "silu")
        fused = ir.Node(
            "com.microsoft",
            "MatMulNBitsMlp",
            [
                gate.inputs[0],
                None,
                None,
                gate.inputs[1],
                gate.inputs[2],
                None,
                up.inputs[1],
                up.inputs[2],
                None,
            ],
            attrs,
            outputs=[replacement],
            name=gate.name + "/EvoFusedMlp",
        )
        # Both independent projections can appear in either topological order.
        model.graph.insert_before(product, fused)
        old.replace_all_uses_with(replacement, replace_graph_outputs=True)
        model.graph.remove([product, silu, sigmoid, gate, up], safe=True)
        changes.append(change)
    if changes and not dry_run:
        model.graph.sort()
    return changes
