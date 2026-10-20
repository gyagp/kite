import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from evo import build
from evo.backends import command_for, make_plan
from evo.cli import main
from evo.config import Recipe, read_recipe
from evo.inspect import inspect_model
from evo.pipeline import convert
from evo.validation import compare_outputs, validate_cpu


@pytest.fixture
def recipe(tmp_path):
    return Recipe("genai", "org/model", "huggingface", tmp_path / "out", "int4", "fp16")


def write_model(path, external=False):
    weights = numpy_helper.from_array(np.eye(2, dtype=np.float32), name="weight")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"])],
        "test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
        [weights],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10)
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=external,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=0,
    )
    return model


def test_plan_does_not_write_or_require_backend(recipe):
    plan = make_plan(recipe, "path with spaces/python.exe")
    assert plan["argv"][0] == "path with spaces/python.exe"
    assert "use_webgpu_fp32=false" in plan["argv"]
    assert "enable_webgpu_graph=false" in plan["argv"]
    assert not recipe.output.exists()
    assert "-i" in command_for(replace(recipe, source_kind="local"))


def test_runtime_capture_can_be_disabled_after_capture_prepared_export(tmp_path, recipe):
    from evo.pipeline import apply_runtime_provider_options

    prepared = replace(
        recipe, graph_capture=True, runtime_provider_options={"enableGraphCapture": "0"}
    )
    assert "enable_webgpu_graph=true" in command_for(prepared)
    path = tmp_path / "genai_config.json"
    config = {
        "model": {
            "decoder": {
                "filename": "model.onnx",
                "session_options": {
                    "provider_options": [
                        {"webgpu": {"enableGraphCapture": "1", "validationMode": "basic"}}
                    ]
                },
            }
        },
        "search": {"past_present_share_buffer": True},
    }
    path.write_text(json.dumps(config))
    apply_runtime_provider_options(tmp_path, prepared.runtime_provider_options)
    actual = json.loads(path.read_text())
    provider = actual["model"]["decoder"]["session_options"]["provider_options"][0]["webgpu"]
    assert provider == {"enableGraphCapture": "0", "validationMode": "basic"}
    assert actual["search"] == config["search"]
    assert actual["model"]["decoder"]["filename"] == "model.onnx"


def test_recipe_paths_are_relative_to_file(tmp_path, monkeypatch):
    folder = tmp_path / "recipes"
    folder.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    file = folder / "local.toml"
    file.write_text(
        'backend="mobius"\nsource="../checkpoint"\nsource_kind="local"\n'
        'output="../out"\nprecision="fp32"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(folder)
    recipe = read_recipe(file)
    assert recipe.source == str(checkpoint)
    assert recipe.output == tmp_path / "out"
    payload = json.loads(command_for(recipe)[-1])
    assert payload["source_kind"] == "local"
    assert payload["precision"] == "fp32"


@pytest.mark.parametrize(
    "extra",
    [
        "typo=true",
        'graph_capture="false"',
        "block_size=true",
        'activation_dtype="bf16"',
        'target="cuda"',
    ],
)
def test_invalid_config_fails_before_export(tmp_path, extra):
    file = tmp_path / "bad.toml"
    file.write_text(
        'backend="genai"\nsource="org/model"\noutput="out"\nprecision="int4"\n' + extra,
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        read_recipe(file)


def test_external_data_inspection_does_not_load_weights(tmp_path, monkeypatch):
    path = tmp_path / "model.onnx"
    write_model(path, external=True)
    monkeypatch.setattr(
        onnx.external_data_helper,
        "load_external_data_for_model",
        lambda *args: pytest.fail("should not load external data"),
    )
    report = inspect_model(path)
    assert report["structure"] == "passed"
    assert report["webgpu"] == "not_run"
    assert report["external_data"] == [{"location": "weights.bin", "exists": True, "bytes": 16}]
    (tmp_path / "weights.bin").write_bytes(b"x")
    assert inspect_model(path)["structure"] == "failed"


def test_missing_external_file_fails_cli(tmp_path, capsys):
    path = tmp_path / "model.onnx"
    write_model(path, external=True)
    (tmp_path / "weights.bin").unlink()
    assert main(["inspect", str(path)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert "Missing external data: weights.bin" in result["errors"]


@pytest.mark.parametrize("location", ["../outside.bin", "C:\\outside.bin", "/outside.bin"])
def test_external_path_escape_is_rejected_before_checker(tmp_path, location, monkeypatch):
    path = tmp_path / "model.onnx"
    write_model(path, external=True)
    model = onnx.load(path, load_external_data=False)
    model.graph.initializer[0].external_data[0].value = location
    path.write_bytes(model.SerializeToString())
    monkeypatch.setattr(onnx.checker, "check_model", lambda *args: pytest.fail("unsafe check"))
    assert inspect_model(path)["structure"] == "failed"


def test_subgraph_operator_inventory(tmp_path):
    path = tmp_path / "model.onnx"
    model = write_model(path)
    branch = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["branch_out"])],
        "branch",
        [],
        [helper.make_tensor_value_info("branch_out", TensorProto.FLOAT, [1, 2])],
    )
    model.graph.initializer.append(numpy_helper.from_array(np.array(True), name="condition"))
    del model.graph.node[:]
    model.graph.node.append(
        helper.make_node(
            "If",
            ["condition"],
            ["output"],
            then_branch=branch,
            else_branch=branch,
        )
    )
    onnx.save(model, path)
    report = inspect_model(path)
    assert report["operators"]["ai.onnx::Identity"] == 2
    assert report["structure"] == "passed"


def test_output_collision_preserves_data(recipe):
    recipe.output.mkdir()
    marker = recipe.output / "keep.txt"
    marker.write_text("existing", encoding="utf-8")
    with pytest.raises(ValueError, match="absent or empty"):
        convert(recipe)
    assert marker.read_text() == "existing"


@pytest.mark.parametrize("outcome", ["success", "exit", "empty", "invalid"])
def test_pipeline_records_export_and_failure_evidence(recipe, monkeypatch, outcome):
    monkeypatch.setattr("evo.pipeline.environment_info", lambda *args: {"test": True})

    def fake_export(*args, **kwargs):
        if outcome in ("success", "invalid"):
            write_model(recipe.output / "model.onnx", external=True)
            if outcome == "invalid":
                (recipe.output / "weights.bin").unlink()
        return subprocess.CompletedProcess(args[0], 3 if outcome == "exit" else 0)

    monkeypatch.setattr("evo.pipeline.subprocess.run", fake_export)
    if outcome == "success":
        result = convert(recipe)
        assert result["status"] == "exported"
        assert any(
            a["path"] == "weights.bin" and len(a["sha256"]) == 64 for a in result["artifacts"]
        )
        assert result["accuracy"] == "not_run"
    else:
        with pytest.raises(RuntimeError):
            convert(recipe)
        manifest = json.loads((recipe.output / "evo-manifest.json").read_text())
        assert manifest["status"] == "failed"
        assert manifest["error"]


def test_cpu_reference_integration(tmp_path):
    pytest.importorskip("onnxruntime")
    path = tmp_path / "model.onnx"
    write_model(path, external=True)
    values = np.array([[1.5, -2.0]], dtype=np.float32)
    np.savez(tmp_path / "inputs.npz", input=values)
    np.savez(tmp_path / "expected.npz", output=values)
    report = validate_cpu(path, tmp_path / "inputs.npz", tmp_path / "expected.npz", 1e-4, 1e-5)
    assert report["accuracy"] == "passed"
    assert report["webgpu"] == "not_run"
    np.savez(tmp_path / "expected.npz", output=values + 1)
    assert (
        main(
            [
                "validate-cpu",
                str(path),
                "--inputs",
                str(tmp_path / "inputs.npz"),
                "--expected",
                str(tmp_path / "expected.npz"),
            ]
        )
        == 1
    )


@pytest.mark.parametrize(
    "actual,expected",
    [
        (np.array([np.nan]), np.array([np.nan])),
        (np.array([np.inf]), np.array([np.inf])),
        (np.array([1]), np.array([[1]])),
        (np.array([2**60], dtype=np.int64), np.array([2**60 + 1], dtype=np.int64)),
    ],
)
def test_bad_numerical_results_fail(actual, expected):
    assert compare_outputs({"x": actual}, {"x": expected}, 1e-3, 1e-4)["accuracy"] == "failed"


def test_module_entrypoint():
    result = subprocess.run([sys.executable, "-m", "evo", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "validate-cpu" in result.stdout


def test_build_api_and_cli_share_recipe_validation(tmp_path, capsys):
    plan = build("Qwen/Qwen3.5-4B", output=tmp_path / "out", dry_run=True)
    assert plan["recipe"]["runtime"] == "native"
    assert not (tmp_path / "out").exists()
    assert (
        main(
            [
                "build",
                "--model",
                "Qwen/Qwen3.5-4B",
                "--output",
                str(tmp_path / "out"),
                "--dry-run",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == plan
    with pytest.raises(ValueError, match="supports these Evo precisions"):
        build("org/model", output=tmp_path / "out", backend="mobius", dry_run=True)


def test_ort_legacy_schema_checks_attributes_and_restores_registry(tmp_path):
    pytest.importorskip("onnxruntime")
    path = tmp_path / "legacy.onnx"
    model = write_model(path)
    del model.graph.node[:]
    model.graph.node.append(
        helper.make_node(
            "SimplifiedLayerNormalization",
            ["input", "weight"],
            ["output"],
            epsilon=1e-5,
        )
    )
    onnx.save(model, path)
    report = inspect_model(path)
    assert report["structure"] == "passed"
    assert report["ort_legacy_schemas"]["registered"] == ["SimplifiedLayerNormalization"]
    assert not onnx.defs.has("SimplifiedLayerNormalization", 21, "")
    model.graph.node[0].attribute[0].CopyFrom(helper.make_attribute("epsilon", "invalid"))
    onnx.save(model, path)
    assert inspect_model(path)["structure"] == "failed"
    assert not onnx.defs.has("SimplifiedLayerNormalization", 21, "")


def test_advanced_export_options_preserved(tmp_path):
    result = build(
        "Qwen/Qwen3.5-4B",
        output=tmp_path / "out",
        dry_run=True,
        genai_options={"exclude_mtp": True, "prune_lm_head": True},
        genai_config={"search": {"max_length": 2048}},
    )
    assert "exclude_mtp=true" in result["argv"]
    assert "prune_lm_head=true" in result["argv"]
    assert json.loads(result["argv"][-1]) == {"max_length": 2048}
    with pytest.raises(ValueError, match="managed"):
        build(
            "org/model",
            output=tmp_path / "out",
            dry_run=True,
            genai_options={"filename": "../escaped.onnx"},
        )


def test_standalone_onnx_optimization_preserves_source_and_numerics(tmp_path):
    pytest.importorskip("mobius")
    ort = pytest.importorskip("onnxruntime")
    from evo.optimize import optimize_package
    from evo.pipeline import hash_file

    source = tmp_path / "original.onnx"
    model = write_model(source, external=True)
    model.graph.node[0].output[0] = "temporary"
    model.graph.node.append(helper.make_node("Identity", ["temporary"], ["output"]))
    source.write_bytes(model.SerializeToString())
    digest = hash_file(source)
    output = tmp_path / "optimized"
    result = optimize_package(source, output)
    assert result["status"] == "optimized"
    assert hash_file(source) == digest
    feeds = {"input": np.array([[4.0, -3.0]], dtype=np.float32)}
    before = ort.InferenceSession(str(source)).run(None, feeds)[0]
    after = ort.InferenceSession(str(output / source.name)).run(None, feeds)[0]
    np.testing.assert_array_equal(before, after)
    assert not list(output.glob("*.toml"))


def test_analysis_separates_proposals_and_available_passes(tmp_path):
    from evo.analyze import analyze

    path = tmp_path / "model.onnx"
    write_model(path)
    report = analyze(path)
    assert report["suggestions"][0]["category"] == "available_now"
    assert report["suggestions"][0]["status"] == "unmeasured"
