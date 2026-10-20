import base64
import http.client
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.catalog import discover, inspect_package
from backend.native import media_content
from backend.process import ProcessGroup
from backend.server import create_server, upload
from backend.service import BusyError, Service
from backend.store import Store
from backend.timing import embedding_performance, generation_metrics, search_embedding_metrics


class TimingTests(unittest.TestCase):
    def test_embedding_throughput_uses_summed_nonpadding_tokens_and_inference_only(self):
        query = {
            "tokens": 100,
            "padded_tokens": 100,
            "input_count": 1,
            "batch_count": 1,
            "preparation_ms": 20,
            "inference_ms": 50,
            "postprocessing_ms": 5,
            "total_ms": 80,
            "inputs": [{"tokens": 100, "video_frames": [2]}],
        }
        docs = {
            "tokens": 300,
            "padded_tokens": 600,
            "input_count": 5,
            "batch_count": 2,
            "preparation_ms": 70,
            "inference_ms": 450,
            "postprocessing_ms": 15,
            "total_ms": 550,
            "inputs": [{"tokens": 300, "video_frames": []}],
        }
        combined = search_embedding_metrics(query, docs)
        self.assertEqual(combined["tokens"], 400)
        self.assertEqual(combined["padded_tokens"], 700)
        self.assertEqual(combined["input_count"], 6)
        self.assertEqual(combined["batch_count"], 3)
        self.assertEqual(combined["query_inputs"], query["inputs"])
        self.assertEqual(combined["inputs"], docs["inputs"])
        self.assertEqual(combined["search_phases"]["query"]["preparation_ms"], 20)
        measured = embedding_performance(combined, request_ms=1200, setup_ms=500, model_load_ms=450)
        self.assertEqual(measured["inference_tokens_per_second"], 800)
        self.assertEqual(measured["inference_inputs_per_second"], 12)
        self.assertEqual(combined["total_ms"], 630)
        self.assertEqual(measured["request_ms"], 1200)

    def test_embedding_zero_duration_has_no_invented_rate(self):
        measured = embedding_performance(
            {"tokens": 10, "input_count": 1, "inference_ms": 0},
            request_ms=0,
            setup_ms=0,
            model_load_ms=0,
        )
        self.assertIsNone(measured["inference_tokens_per_second"])
        self.assertIsNone(measured["inference_inputs_per_second"])

    def measure(self, **values):
        return generation_metrics(
            **{
                "prompt_tokens": 100,
                "completion_tokens": 5,
                "first_step_tokens": 1,
                "started": 10.0,
                "prefill_finished": 10.5,
                "first_token": 10.6,
                "last_token": 11.0,
                "observed": 11.3,
                **values,
            }
        )

    def test_prefill_excludes_sampling_and_decode_excludes_first_step_and_tail(self):
        m = self.measure()
        self.assertAlmostEqual(m["prefill_ms"], 500)
        self.assertAlmostEqual(m["prefill_tokens_per_second"], 200)
        self.assertAlmostEqual(m["ttft_ms"], 600)
        self.assertEqual(m["decode_tokens"], 4)
        self.assertAlmostEqual(m["decode_ms"], 400)
        self.assertAlmostEqual(m["decode_tokens_per_second"], 10)
        self.assertAlmostEqual(m["decode_ms_per_token"], 100)
        self.assertAlmostEqual(m["total_ms"], 1300)

    def test_zero_and_one_step_have_no_invented_decode_rate(self):
        for count, first in ((0, None), (1, 10.6)):
            m = self.measure(
                completion_tokens=count,
                first_step_tokens=count,
                first_token=first,
                last_token=first,
            )
            self.assertIsNone(m["decode_tokens_per_second"])
            self.assertIsNone(m["decode_ms_per_token"])
            self.assertEqual(m["decode_ms"], 0)
            if first is None:
                self.assertIsNone(m["ttft_ms"])

    def test_excludes_all_tokens_from_first_generated_step(self):
        m = self.measure(first_step_tokens=3)
        self.assertEqual(m["decode_tokens"], 2)
        self.assertAlmostEqual(m["decode_tokens_per_second"], 5)


def package(root, embedding=True, vision=False):
    root.mkdir(parents=True, exist_ok=True)
    (root / "tokenizer.json").write_text("{}")
    (root / "model.onnx").write_bytes(b"test graph placeholder")
    if embedding:
        (root / "embedding_config.json").write_text(
            json.dumps(
                {
                    "architecture": "EmbeddingGemma2TextModel",
                    "max_length": 8192,
                    "embedding_dimensions": 768,
                }
            )
        )
        (root / "config_sentence_transformers.json").write_text(
            json.dumps({"prompts": {"SearchQuery": "query: ", "Document": "doc: "}})
        )
    else:
        model = {"type": "qwen3_5_text", "decoder": {"filename": "model.onnx"}}
        if vision:
            model["vision"] = {"filename": "vision.onnx"}
            (root / "vision.onnx").write_bytes(b"test vision placeholder")
        (root / "genai_config.json").write_text(json.dumps({"model": model}))
    return inspect_package(root)


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_text_export_does_not_inherit_checkpoint_vision(self):
        item = package(self.root / "text", embedding=False)
        (self.root / "text" / "config.json").write_text('{"vision_config": {}}')
        self.assertEqual(inspect_package(item["path"])["modalities"], ["text"])

    def test_multimodal_requires_declared_vision_graph(self):
        item = package(self.root / "vision", embedding=False, vision=True)
        self.assertEqual(item["modalities"], ["text", "image"])
        (self.root / "vision" / "vision.onnx").unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            inspect_package(item["path"])

    def test_external_graph_path_is_rejected(self):
        item = package(self.root / "escape", embedding=False)
        (self.root / "escape" / "genai_config.json").write_text(
            '{"model":{"decoder":{"filename":"../model.onnx"}}}'
        )
        (self.root / "model.onnx").write_bytes(b"graph")
        with self.assertRaisesRegex(ValueError, "unsafe"):
            inspect_package(item["path"])

    def test_discovery_reports_broken_packages(self):
        package(self.root / "valid")
        package(self.root / "broken")
        (self.root / "broken" / "tokenizer.json").unlink()
        found, errors = discover(self.root)
        self.assertEqual((len(found), len(errors)), (1, 1))

    def test_missing_config_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "embedding_config.json"):
            inspect_package(self.root)

    def test_media_markers_preserve_order_across_turns(self):
        content, images, audios = media_content(
            "phi4mm", "Describe", [{"kind": "image"}, {"kind": "audio"}], 2, 3
        )
        self.assertEqual(content, "<|image_3|>\n<|audio_4|>\nDescribe")
        self.assertEqual((images, audios), (3, 4))
        qwen, _, _ = media_content("qwen2_5_vl", "Look", [{"kind": "image"}])
        self.assertEqual(qwen, "<|vision_start|><|image_pad|><|vision_end|>Look")
        gemma, _, _ = media_content("gemma3", "Look", [{"kind": "image"}])
        self.assertEqual(gemma, [{"type": "image"}, {"type": "text", "text": "Look"}])


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.service = Service(Path(self.temp.name) / "state")
        self.model = package(Path(self.temp.name) / "model")
        self.service.store.put("models", self.model)

    def tearDown(self):
        self.service.stop()
        if self.service.active:
            with self.service.active.condition:
                self.service.active.condition.wait_for(
                    lambda: self.service.active is None, timeout=5
                )
        self.service.store.close()
        self.temp.cleanup()

    def test_empty_and_wrong_kind_requests_rejected_before_loading(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.service.submit({"kind": "embed", "model": self.model["id"], "texts": [""]})
        with self.assertRaisesRegex(ValueError, "do not generate"):
            self.service.submit({"kind": "chat", "model": self.model["id"], "messages": []})
        self.assertIsNone(self.service.worker)

    def test_uploaded_media_cannot_bypass_text_capability(self):
        model = package(Path(self.temp.name) / "chat", embedding=False)
        self.service.store.put("models", model)
        attachment = upload(
            self.service,
            {
                "name": "a.png",
                "data": base64.b64encode(b"\x89PNG\r\n\x1a\nfixture").decode(),
            },
        )
        with self.assertRaisesRegex(ValueError, "image encoder"):
            self.service.submit(
                {
                    "kind": "chat",
                    "model": model["id"],
                    "messages": [{"role": "user", "content": "look", "attachments": [attachment]}],
                }
            )

    def test_competing_jobs_rejected_and_errors_release_slot(self):
        entered, release = threading.Event(), threading.Event()

        def fail_load(*_):
            entered.set()
            release.wait(3)
            raise RuntimeError("simulated device loss")

        with patch.object(self.service, "_load", side_effect=fail_load):
            job = self.service.submit({"kind": "load", "model": self.model["id"]})
            self.assertTrue(entered.wait(2))
            with self.assertRaises(BusyError):
                self.service.submit({"kind": "load", "model": self.model["id"]})
            release.set()
            with job.condition:
                job.condition.wait_for(
                    lambda: any(e["type"] == "done" for e in job.events), timeout=3
                )
            self.assertEqual(job.status, "failed")
            self.assertIn("device loss", job.error)
            self.assertIsNone(self.service.active)
            self.assertEqual(self.service.store.get("runs", job.id)["status"], "failed")

    def test_settings_reject_missing_runtime(self):
        with self.assertRaisesRegex(ValueError, "existing file"):
            self.service.save_settings({"library": "Z:/does-not-exist/provider.dll"})

    def test_store_survives_reopen(self):
        folder = Path(self.temp.name) / "persistence"
        store = Store(folder)
        store.put(
            "threads",
            {
                "id": "test",
                "title": "你好",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        store.close()
        reopened = Store(folder)
        self.assertEqual(reopened.get("threads", "test")["title"], "你好")
        reopened.close()


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server = create_server(0, self.temp.name)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.service.stop()
        self.server.service.store.close()
        self.server.server_close()
        self.temp.cleanup()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        conn.request(
            method,
            path,
            body=json.dumps(body) if body is not None else None,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        response = conn.getresponse()
        result = response.status, json.loads(response.read())
        conn.close()
        return result

    def test_health_and_local_api_authentication(self):
        self.assertEqual(self.request("GET", "/api/health")[0], 200)
        self.assertEqual(self.request("GET", "/v1/models")[0], 401)
        status, body = self.request(
            "GET",
            "/v1/models",
            headers={"Authorization": "Bearer " + self.server.service.api_key["value"]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "list")

    def test_cross_origin_and_rebinding_rejected(self):
        self.assertEqual(
            self.request("GET", "/api/bootstrap", headers={"Origin": "https://outside.example"})[0],
            403,
        )
        self.assertEqual(
            self.request("GET", "/api/bootstrap", headers={"Host": "outside.example"})[0],
            403,
        )

    def test_non_json_mutation_rejected(self):
        self.assertEqual(
            self.request("POST", "/api/stop", {}, {"Content-Type": "text/plain"})[0],
            400,
        )

    def test_unknown_job_and_unsafe_upload_fail(self):
        self.assertEqual(self.request("GET", "/api/jobs/missing")[0], 404)
        status, _ = self.request(
            "POST",
            "/api/attachments",
            {"name": "bad.exe", "data": base64.b64encode(b"MZ executable").decode()},
        )
        self.assertEqual(status, 400)

    def test_unknown_model_does_not_start_worker(self):
        status, _ = self.request("POST", "/api/jobs", {"kind": "load", "model": "missing"})
        self.assertEqual(status, 400)
        self.assertIsNone(self.server.service.worker)


@unittest.skipUnless(os.name == "nt", "Windows process lifecycle")
class WindowsLifecycleTests(unittest.TestCase):
    def test_job_close_terminates_descendant(self):
        # Child is created after assignment, just like the venv native worker launcher.
        code = "import subprocess,sys,time; sys.stdin.readline(); p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(p.pid,flush=True); time.sleep(60)"
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        group = ProcessGroup(process)
        try:
            process.stdin.write("go\n")
            process.stdin.flush()
            child = int(process.stdout.readline())
            group.close()
            process.wait(timeout=5)
            import ctypes
            from ctypes import wintypes

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.OpenProcess(0x100000, False, child)
            if handle:
                self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0)
                kernel.CloseHandle(handle)
        finally:
            group.close()
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdin.close()
            process.stdout.close()


if __name__ == "__main__":
    unittest.main()
