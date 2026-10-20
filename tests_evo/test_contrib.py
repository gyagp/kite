import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from evo.contrib import catalog
from evo.contrib_rules import fuse_packed_mlp, fuse_separate_mlp, mlp_patterns


def packed_mlp(path):
    rng = np.random.default_rng(42)
    weights = rng.integers(0, 256, (128, 2, 16), dtype=np.uint8)
    scales = rng.uniform(0.01, 0.04, (128, 2)).astype(np.float32)
    nodes = [
        helper.make_node(
            "MatMulNBits",
            ["x", "w", "s"],
            ["packed"],
            domain="com.microsoft",
            name="packed_gate_up",
            K=64,
            N=128,
            bits=4,
            block_size=32,
            accuracy_level=1,
        ),
        helper.make_node("Split", ["packed"], ["gate", "up"], axis=-1, num_outputs=2),
        helper.make_node("Sigmoid", ["gate"], ["sigmoid"]),
        helper.make_node("Mul", ["gate", "sigmoid"], ["silu"]),
        helper.make_node("Mul", ["silu", "up"], ["y"]),
    ]
    values = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2, width])
        for name, width in (
            ("packed", 128),
            ("gate", 64),
            ("up", 64),
            ("sigmoid", 64),
            ("silu", 64),
        )
    ]
    graph = helper.make_graph(
        nodes,
        "packed_mlp",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 64])],
        [numpy_helper.from_array(weights, "w"), numpy_helper.from_array(scales, "s")],
        value_info=values,
    )
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )
    onnx.save(model, path)
    return model, weights, scales


def test_contrib_catalog_separates_schema_and_kernel_support():
    data = catalog()
    ops = {op["id"]: op for op in data["operators"]}
    assert "ai.onnx::MatMul" in ops
    assert "com.microsoft::MatMulNBitsMlp" in ops
    assert ops["com.microsoft::MatMulNBitsMlp"]["guidance"]["constraints"]
    assert "does not prove kernel support" in ops["ai.onnx::MatMul"]["guidance"]["webgpu"]


def test_mlp_rewrite_slices_quantized_weights_exactly(tmp_path):
    ir = pytest.importorskip("onnx_ir")
    path = tmp_path / "model.onnx"
    _, weights, scales = packed_mlp(path)
    model = ir.load(path)
    changes = fuse_packed_mlp(model)
    assert len(changes) == 1
    nodes = list(model.graph)
    assert len(nodes) == 1 and nodes[0].op_type == "MatMulNBitsMlp"
    for index, tensor in ((3, weights[:64]), (4, scales[:64]), (6, weights[64:]), (7, scales[64:])):
        np.testing.assert_array_equal(nodes[0].inputs[index].const_value.numpy(), tensor)
    assert model.graph.outputs[0].name == "y"
    assert list(model.graph.outputs[0].shape) == [1, 2, 64]


@pytest.mark.parametrize("case", ["zero_points", "extra_consumer", "other_domain", "block_size"])
def test_mlp_rewrite_rejects_unproven_contracts(tmp_path, case):
    ir = pytest.importorskip("onnx_ir")
    path = tmp_path / "model.onnx"
    model, _, _ = packed_mlp(path)
    if case == "zero_points":
        model.graph.node[0].input.append("zero")
        model.graph.initializer.append(
            numpy_helper.from_array(np.zeros((128, 1), dtype=np.uint8), "zero")
        )
    elif case == "extra_consumer":
        model.graph.output.append(
            helper.make_tensor_value_info("gate", TensorProto.FLOAT, [1, 2, 64])
        )
    elif case == "other_domain":
        model.graph.node[0].domain = "test.custom"
    else:
        next(a for a in model.graph.node[0].attribute if a.name == "block_size").i = 128
    onnx.save(model, path)
    assert not mlp_patterns(ir.load(path))


def separate_mlp(path):
    model, weights, scales = packed_mlp(path)
    del model.graph.initializer[:]
    for name, array in (
        ("gw", weights[:64]),
        ("gs", scales[:64]),
        ("uw", weights[64:]),
        ("us", scales[64:]),
    ):
        model.graph.initializer.append(numpy_helper.from_array(array.copy(), name))
    projections = [
        helper.make_node(
            "MatMulNBits",
            ["x", w, s],
            [out],
            domain="com.microsoft",
            name=out + "_projection",
            K=64,
            N=64,
            bits=4,
            block_size=32,
            accuracy_level=1,
        )
        for w, s, out in (("gw", "gs", "gate"), ("uw", "us", "up"))
    ]
    rest = list(model.graph.node)[2:]
    del model.graph.node[:]
    model.graph.node.extend(projections + rest)
    onnx.save(model, path)
    return model, weights, scales


def test_separate_mlp_preserves_weights_and_commuted_inputs(tmp_path):
    ir = pytest.importorskip("onnx_ir")
    path = tmp_path / "model.onnx"
    proto, _, _ = separate_mlp(path)
    for n in proto.graph.node:
        if n.op_type == "Mul":
            n.input.reverse()
    onnx.save(proto, path)
    model = ir.load(path)
    before = dict(model.graph.initializers)
    before_nodes = list(model.graph)
    assert len(fuse_separate_mlp(model, dry_run=True)) == 1
    assert list(model.graph) == before_nodes
    assert len(fuse_separate_mlp(model)) == 1
    (node,) = model.graph
    assert node.op_type == "MatMulNBitsMlp"
    for index, name in ((3, "gw"), (4, "gs"), (6, "uw"), (7, "us")):
        assert node.inputs[index] is before[name]
    assert len(model.graph.initializers) == 4
    assert model.graph.outputs[0].name == "y"


@pytest.mark.parametrize("case", ["output", "accuracy", "input", "scale", "zero_points"])
def test_separate_mlp_rejects_incompatible_or_observable_intermediates(tmp_path, case):
    ir = pytest.importorskip("onnx_ir")
    path = tmp_path / "model.onnx"
    model, _, _ = separate_mlp(path)
    if case == "output":
        model.graph.output.append(
            helper.make_tensor_value_info("gate", TensorProto.FLOAT, [1, 2, 64])
        )
    elif case == "accuracy":
        next(a for a in model.graph.node[1].attribute if a.name == "accuracy_level").i = 4
    elif case == "input":
        model.graph.input.append(helper.make_tensor_value_info("x2", TensorProto.FLOAT, [1, 2, 64]))
        model.graph.node[1].input[0] = "x2"
    elif case == "scale":
        model.graph.initializer[-1].CopyFrom(numpy_helper.from_array(np.ones(8, np.float32), "us"))
    else:
        model.graph.node[1].input.append("zero")
        model.graph.initializer.append(numpy_helper.from_array(np.zeros((64, 1), np.uint8), "zero"))
    onnx.save(model, path)
    graph = ir.load(path)
    assert fuse_separate_mlp(graph) == []
    assert len(graph.graph) == len(model.graph.node)


@pytest.mark.parametrize("layout", ["packed", "separate"])
def test_native_mlp_matches_reference_cpu(tmp_path, layout):
    ir = pytest.importorskip("onnx_ir")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxruntime_ep_webgpu")
    from evo.runtime import register_webgpu

    path = tmp_path / "source.onnx"
    (packed_mlp if layout == "packed" else separate_mlp)(path)
    model = ir.load(path)
    (fuse_packed_mlp if layout == "packed" else fuse_separate_mlp)(model)
    target = tmp_path / "fused.onnx"
    ir.save(model, target)
    reference = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    devices, _ = register_webgpu()
    if not devices:
        pytest.skip("No WebGPU adapter")
    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(devices[:1], {"powerPreference": "high-performance"})
    session = ort.InferenceSession(str(target), sess_options=options)
    for seed in range(3):
        x = np.random.default_rng(seed).normal(size=(1, 2, 64)).astype(np.float32)
        np.testing.assert_allclose(
            session.run(None, {"x": x})[0], reference.run(None, {"x": x})[0], rtol=2e-4, atol=2e-5
        )
