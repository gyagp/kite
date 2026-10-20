import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from evo.benchmark import artifact_hashes


def test_weight_changes_are_detected_even_when_graph_is_unchanged(tmp_path):
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("MatMul", ["x", "weight"], ["y"])],
            "external-weight",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
            [numpy_helper.from_array(np.eye(2, dtype=np.float32), "weight")],
        ),
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 21)],
    )
    onnx.save_model(
        model,
        tmp_path / "model.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )
    (tmp_path / "genai_config.json").write_text("{}")
    before = artifact_hashes(tmp_path)
    (tmp_path / "weights.bin").write_bytes(np.zeros((2, 2), dtype=np.float32).tobytes())
    after = artifact_hashes(tmp_path)
    assert before["model.onnx"] == after["model.onnx"]
    assert before["weights.bin"]["sha256"] != after["weights.bin"]["sha256"]
