import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from evo.ui import make_server
from evo.workbench import Workbench


def fixture_model(path):
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "w"], ["y"])],
        "add",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(np.ones((1, 2), dtype=np.float32), "w")],
    )
    onnx.save(
        helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 21)]), path
    )


def test_registry_and_wishlist_survive_restart_without_touching_models(tmp_path):
    path = tmp_path / "input.onnx"
    fixture_model(path)
    before = path.read_bytes()
    workbench = Workbench(tmp_path)
    model = workbench.add(path, "My model")
    wish = workbench.save_wish(
        {
            "title": "Profile normalization",
            "model_id": model["id"],
            "operator": "ai.onnx::Add",
            "priority": "high",
        }
    )
    workbench.save_wish({"id": wish["id"], "status": "investigating"})
    restored = Workbench(tmp_path)
    assert restored.record(model["id"])["name"] == "My model"
    assert restored.wishlist[0]["status"] == "investigating"
    restored.remove(model["id"])
    assert path.read_bytes() == before


@pytest.fixture
def service(tmp_path):
    server = make_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def request(server, path, data=None, headers=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}{path}",
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req) as response:
        return json.load(response)


def test_local_server_blocks_cross_origin_mutations(service):
    with pytest.raises(urllib.error.HTTPError) as error:
        request(
            service, "/api/wishlist", {"title": "unexpected"}, {"Origin": "https://example.org"}
        )
    assert error.value.code == 403
    assert request(service, "/api/wishlist") == []
    with pytest.raises(urllib.error.HTTPError):
        request(service, "/api/jobs", {"operation": "arbitrary-shell"})


def test_performance_records_do_not_merge_different_gpus(tmp_path):
    workbench = Workbench(tmp_path)
    folder = tmp_path / "artifacts/benchmarks"
    folder.mkdir()
    for gpu in ("A", "B"):
        report = {
            "model_sha256": "model",
            "cases_sha256": "fixture",
            "new_tokens": 32,
            "max_length": 100,
            "versions": {"onnxruntime": "1.31"},
            "provider_options": {},
            "search_options": {},
            "devices": [{"metadata": {"Description": gpu}}],
            "results": [
                {
                    "prompt_tokens": 68,
                    "samples": [
                        {
                            "ttft_seconds": 0.1,
                            "decode_tokens_per_second": 50,
                            "generated_ids": [1] * 32,
                        }
                    ],
                }
            ],
        }
        (folder / f"{gpu}.json").write_text(json.dumps(report), encoding="utf-8")
    records = workbench.benchmarks({"sha256": "model"})
    assert len(records) == 2
    assert records[0]["protocol_id"] != records[1]["protocol_id"]


def test_same_version_different_runtime_binaries_are_not_merged(tmp_path):
    workbench = Workbench(tmp_path)
    folder = tmp_path / "artifacts/benchmarks"
    folder.mkdir()
    base = {
        "model_sha256": "model",
        "cases_sha256": "fixture",
        "new_tokens": 32,
        "max_length": 100,
        "versions": {"onnxruntime": "1.31", "onnxruntime-ep-webgpu": "0.4"},
        "provider_options": {},
        "search_options": {},
        "devices": [{"metadata": {"Description": "GPU"}}],
        "results": [
            {
                "prompt_tokens": 68,
                "samples": [
                    {"ttft_seconds": 0.1, "decode_tokens_per_second": 50, "generated_ids": [1] * 32}
                ],
            }
        ],
    }
    for binary in ("first", "second"):
        (folder / f"{binary}.json").write_text(
            json.dumps({**base, "runtime_library": {"sha256": binary}})
        )
    records = workbench.benchmarks({"sha256": "model"})
    assert len(records) == 2
    assert records[0]["protocol_id"] != records[1]["protocol_id"]


def test_queue_runs_real_inspection_and_retains_report(service, tmp_path):
    path = tmp_path / "input.onnx"
    fixture_model(path)
    model = request(service, "/api/models", {"path": str(path), "name": "Input"})
    job = request(service, "/api/jobs", {"operation": "inspect", "model_id": model["id"]})
    for _ in range(100):
        current = request(service, "/api/jobs/" + job["id"])
        if current["status"] not in {"queued", "running"}:
            break
        time.sleep(0.1)
    assert current["status"] == "completed", current
    assert current["result"]["structure"] == "passed"
    assert (service.jobs.root / job["id"] / "job.json").is_file()


def test_manual_performance_keeps_models_prompts_machines_and_units(tmp_path):
    path = tmp_path / "input.onnx"
    fixture_model(path)
    workbench = Workbench(tmp_path)
    model = workbench.add(path, "Embedding model")
    base = {
        "model_id": model["id"],
        "task": "embedding",
        "latency_ms": 12.5,
        "embeddings_per_second": 80,
    }
    for prompt, machine in [
        ("Question A", "Desktop"),
        ("Question B", "Desktop"),
        ("Question A", "Laptop"),
    ]:
        workbench.save_performance({**base, "prompt": prompt, "machine": machine})
    restored = Workbench(tmp_path)
    rows = restored.performance_records()
    assert len(rows) == 3
    assert {(r["prompt"], r["machine"]) for r in rows} == {
        ("Question A", "Desktop"),
        ("Question B", "Desktop"),
        ("Question A", "Laptop"),
    }
    assert all(r["task"] == "embedding" and "decode_tps" not in r for r in rows)
    assert all(r["latency_ms"] == 12.5 for r in rows)


def test_invalid_performance_does_not_overwrite_saved_observation(tmp_path):
    path = tmp_path / "input.onnx"
    fixture_model(path)
    workbench = Workbench(tmp_path)
    model = workbench.add(path)
    data = {"model_id": model["id"], "prompt": "Hello", "machine": "Desktop", "decode_tps": 75}
    saved = workbench.save_performance(data)
    for invalid in [float("nan"), float("inf"), -1, 0]:
        with pytest.raises(ValueError):
            workbench.save_performance({**data, "id": saved["id"], "decode_tps": invalid})
    assert workbench.performance_records()[0]["decode_tps"] == 75
    assert len(Workbench(tmp_path).performance) == 1


def test_conversion_wishlist_preserves_model_request_and_status(tmp_path):
    workbench = Workbench(tmp_path)
    item = workbench.save_wish({"title": "EmbeddingGemma 2", "kind": "conversion_model"})
    workbench.save_wish({"id": item["id"], "status": "investigating"})
    restored = Workbench(tmp_path).wishlist[0]
    assert restored["title"] == "EmbeddingGemma 2"
    assert restored["kind"] == "conversion_model"
    assert restored["status"] == "investigating"


def test_image_perf_and_gguf_registry_survive_reload(tmp_path):
    path = tmp_path / "baseline.gguf"
    path.write_bytes(b"GGUF" + b"\0" * 32)
    workbench = Workbench(tmp_path)
    model = workbench.add(path, "Image GGUF baseline")
    workbench.save_performance(
        {
            "model_id": model["id"],
            "prompt": "A red boat",
            "machine": "RTX test",
            "task": "image",
            "latency_ms": 8000,
            "images_per_second": 0.125,
            "notes": "1024x1024, 8 steps; denoiser only",
        }
    )
    restored = Workbench(tmp_path)
    assert restored.details(model["id"])["inspection"]["structure"] == "GGUF"
    row = restored.performance_records()[0]
    assert row["task"] == "image" and row["images_per_second"] == 0.125
    assert "decode_tps" not in row


def test_performance_index_accepts_reference_prompt_lists(tmp_path):
    folder = tmp_path / "benchmarks"
    folder.mkdir()
    (folder / "quality-prompts.json").write_text(json.dumps(["Hello", "Explain gravity"]))
    (folder / "tokens.json").write_text(json.dumps({"cases": [{"name": "tiny", "input_ids": [1]}]}))
    assert Workbench(tmp_path).performance_records() == []


def test_performance_index_ignores_operator_traces(tmp_path):
    folder = tmp_path / "artifacts/comparison-image"
    folder.mkdir(parents=True)
    (folder / "trace.json").write_text(json.dumps([{"name": "model_run", "dur": 100}]))
    workbench = Workbench(tmp_path)
    assert workbench.performance_records() == []
    assert workbench.benchmarks({"sha256": "not-a-model"}) == []


@pytest.mark.parametrize("status", ["running", "incomplete", "contaminated"])
def test_performance_omits_unfinished_measurements(tmp_path, status):
    from evo.pipeline import hash_file

    package = tmp_path / "model"
    package.mkdir()
    model = package / "model.onnx"
    fixture_model(model)
    workbench = Workbench(tmp_path)
    workbench.add(model)
    folder = tmp_path / "artifacts/benchmarks"
    folder.mkdir()
    report = {
        "package": str(package),
        "model_sha256": hash_file(model),
        "measurement_status": status,
        "new_tokens": 128,
        "versions": {"onnxruntime": "1.31"},
        "results": [
            {
                "prompt_tokens": 128,
                "samples": [{"ttft_seconds": 0.1, "decode_tokens_per_second": 100}],
            }
        ],
    }
    (folder / "partial.json").write_text(json.dumps(report))
    assert workbench.performance_records() == []
    assert workbench.benchmarks({"sha256": hash_file(model)}) == []
