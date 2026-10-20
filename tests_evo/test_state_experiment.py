import runpy
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper


def example():
    inputs, outputs, nodes = [], [], []
    for i in range(24):
        past, present = f"past.{i}.recurrent", f"present.{i}.recurrent"
        shape = ["batch", 32, 128, 128]
        inputs.append(helper.make_tensor_value_info(past, TensorProto.FLOAT, shape))
        outputs.append(helper.make_tensor_value_info(present, TensorProto.FLOAT, shape))
        nodes.append(
            helper.make_node(
                "GatedDeltaNet",
                ["q", "k", "v", "", "a", "b", past],
                [f"y.{i}", present],
                domain="com.microsoft",
                name=f"gdn.{i}",
            )
        )
    return helper.make_model(helper.make_graph(nodes, "state-test", inputs, outputs))


def transform():
    source = (
        Path(__file__).resolve().parents[1] / "evo_resources/scripts/experiments/qwen_fp16_state.py"
    )
    return runpy.run_path(str(source))["rewrite"]


def test_fp16_storage_keeps_gdn_state_computation_in_fp32():
    model = example()
    assert transform()(model) == 24
    assert len(model.graph.node) == 72
    assert all(
        v.type.tensor_type.elem_type == TensorProto.FLOAT16
        for v in (*model.graph.input, *model.graph.output)
    )
    for i in range(24):
        load, gdn, store = model.graph.node[i * 3 : i * 3 + 3]
        assert load.op_type == store.op_type == "Cast"
        assert onnx.helper.get_attribute_value(load.attribute[0]) == TensorProto.FLOAT
        assert onnx.helper.get_attribute_value(store.attribute[0]) == TensorProto.FLOAT16
        assert gdn.input[6] == load.output[0]
        assert gdn.output[1] == store.input[0]
        assert store.output[0] == f"present.{i}.recurrent"


@pytest.mark.parametrize("reason", ["already_fp16", "other_consumer", "wrong_shape"])
def test_fp16_storage_rejects_other_contracts(reason):
    model = example()
    if reason == "already_fp16":
        model.graph.input[0].type.tensor_type.elem_type = TensorProto.FLOAT16
    elif reason == "wrong_shape":
        model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 16
    else:
        model.graph.node.append(helper.make_node("Identity", ["past.0.recurrent"], ["extra"]))
    with pytest.raises(ValueError):
        transform()(model)
