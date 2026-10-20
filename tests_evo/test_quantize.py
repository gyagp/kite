import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from evo.quantize import quantize


def test_selective_quantization_handles_shared_weights_and_preserves_source(tmp_path):
    source, output = tmp_path / "source", tmp_path / "quantized"
    source.mkdir()
    rng = np.random.default_rng(13)
    weight = rng.normal(0, 0.1, (32, 16)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "weight"], ["a"], name="gate/a"),
            helper.make_node("MatMul", ["x", "weight"], ["b"], name="gate/b"),
            helper.make_node("Add", ["a", "b"], ["out"]),
        ],
        "shared",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 32])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 16])],
        [numpy_helper.from_array(weight, "weight")],
    )
    path = source / "model.onnx"
    onnx.save(
        helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 21)]), path
    )
    before = path.read_bytes()
    quantize(source, output, "gate/", bits=8)
    assert path.read_bytes() == before
    result = onnx.load(output / "model.onnx")
    names = [i.name for i in result.graph.initializer]
    assert len(names) == len(set(names))
    assert len([n for n in result.graph.node if n.op_type == "MatMulNBits"]) == 2
    x = rng.normal(0, 0.2, (2, 32)).astype(np.float32)
    session = ort.InferenceSession(str(output / "model.onnx"), providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(session.run(None, {"x": x})[0], 2 * (x @ weight), atol=0.006)
