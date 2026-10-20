"""Check merged projections against the independent CPU MatMulNBits kernel."""

import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh

from evo.image_projections import merge_projections


def test_merging_preserves_distinct_consumers_and_external_weights(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    source = tmp_path / "source"
    source.mkdir()
    rng = np.random.default_rng(13)
    nodes, tensors, outputs = [], [], []
    for i, n in enumerate((17, 31, 8)):
        # The last projection has a different activation and must stay separate.
        q = rng.integers(0, 256, (n, 2, 16), dtype=np.uint8)
        scale = rng.uniform(0.001, 0.02, (n, 2)).astype(np.float32)
        zero = rng.uniform(6, 9, (n, 2)).astype(np.float32)
        names = [f"b{i}", f"s{i}", f"z{i}"]
        tensors.extend(
            nh.from_array(a, name) for a, name in zip((q, scale, zero), names, strict=True)
        )
        nodes.append(
            h.make_node(
                "MatMulNBits",
                ["x" if i < 2 else "other", *names],
                [f"y{i}"],
                name=f"projection{i}",
                domain="com.microsoft",
                N=n,
                K=64,
                bits=4,
                block_size=32,
                accuracy_level=4,
            )
        )
        outputs.append(h.make_tensor_value_info(f"y{i}", T.FLOAT, [1, 3, n]))
    graph = h.make_graph(
        nodes,
        "merge",
        [h.make_tensor_value_info(n, T.FLOAT, [1, 3, 64]) for n in ("x", "other")],
        outputs,
        tensors,
    )
    model = h.make_model(
        graph,
        ir_version=10,
        opset_imports=[h.make_opsetid("", 21), h.make_opsetid("com.microsoft", 1)],
    )
    onnx.save_model(
        model,
        source / "prefix-capture.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=0,
    )
    onnx.save_model(model, source / "denoise-capture.onnx")
    (source / "image_config.json").write_text(json.dumps({"artifacts": {}}))
    output = tmp_path / "result"
    assert merge_projections(source, output, "capture") == {"prefix": 1, "denoise": 1}
    feeds = {n: rng.normal(size=(1, 3, 64)).astype(np.float32) for n in ("x", "other")}
    for part in ("prefix", "denoise"):
        before = ort.InferenceSession(
            str(source / f"{part}-capture.onnx"), providers=["CPUExecutionProvider"]
        )
        after = ort.InferenceSession(
            str(output / f"{part}-projections.onnx"), providers=["CPUExecutionProvider"]
        )
        for actual, expected in zip(after.run(None, feeds), before.run(None, feeds), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        onnx.checker.check_model(str(output / f"{part}-projections.onnx"))
