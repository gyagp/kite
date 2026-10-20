import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.catalog import inspect_package
from backend.hub import REPOSITORY, Hub
from backend.service import Service
from backend.store import Store
from backend.supported import catalog_id, library_models


class SupportedModelTests(unittest.TestCase):
    def test_catalog_only_enriches_explicitly_added_models_including_missing_files(self):
        remote = {
            "id": catalog_id("Qwen3.5-4B"),
            "name": "Qwen3.5-4B",
            "kind": "chat",
            "path": "",
            "modalities": ["text"],
            "download": {"model": "Qwen3.5-4B", "repository": REPOSITORY},
        }
        only_remote = library_models([], {"models": [remote]})
        self.assertEqual(only_remote, [])
        retained = {
            "id": "saved-model-id",
            "name": "Qwen3.5 · 4B",
            "kind": "chat",
            "path": "",
            "modalities": ["text"],
        }
        merged = library_models([retained], {"models": [remote]})
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["id"], "saved-model-id")
        self.assertEqual(merged[0]["download"]["model"], "Qwen3.5-4B")
        self.assertFalse(merged[0]["available"])

    def test_text_only_embedding_does_not_inherit_multimodal_download(self):
        remote = {
            "id": "multimodal",
            "name": "EmbeddingGemma-2",
            "kind": "embedding",
            "path": "",
            "modalities": ["text", "image", "audio", "video"],
            "download": {"model": "EmbeddingGemma-2"},
        }
        legacy = {
            "id": "text",
            "name": "EmbeddingGemma 2",
            "kind": "embedding",
            "path": "",
            "modalities": ["text"],
        }
        models = library_models([legacy], {"models": [remote]})
        self.assertEqual({m["id"] for m in models}, {"text"})
        self.assertNotIn("download", models[0])

    def test_catalog_is_available_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            try:
                catalog = {
                    "repository": REPOSITORY,
                    "revision": "a" * 40,
                    "models": [{"name": "Test"}],
                }
                with (
                    patch("backend.hub.fetch_json", side_effect=OSError("offline")),
                    patch("backend.hub.snapshot", return_value=catalog),
                ):
                    result = Hub(store).catalog()
                self.assertTrue(result["offline"])
                self.assertEqual(result["models"], catalog["models"])
            finally:
                store.close()

    def test_delete_and_download_retains_identity_and_conversation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = Service(root / "state")
            try:
                package = root / "installed"
                package.mkdir()
                payloads = {
                    "model.onnx": b"fixture",
                    "tokenizer.json": b"{}",
                    "genai_config.json": json.dumps(
                        {"model": {"type": "test", "decoder": {"filename": "model.onnx"}}}
                    ).encode(),
                }
                for name, value in payloads.items():
                    (package / name).write_bytes(value)
                model = inspect_package(package)
                model["name"] = "Test"
                model["source"] = {"repository": REPOSITORY, "path": "Test/onnx"}
                service.store.put("models", model)
                service.store.put(
                    "threads", {"id": "conversation", "model": model["id"], "messages": ["keep"]}
                )
                service.delete_model(model["id"], True, str(package))
                self.assertFalse(package.exists())
                self.assertIsNotNone(service.store.get("models", model["id"]))
                plan = {
                    "id": "download",
                    "name": "Test",
                    "kind": "chat",
                    "modalities": ["text"],
                    "variant": "onnx",
                    "path": "Test/onnx",
                    "revision": "a" * 40,
                    "size": sum(map(len, payloads.values())),
                    "files": [
                        {
                            "name": n,
                            "remote": "Test/onnx/" + n,
                            "size": len(v),
                            "sha256": hashlib.sha256(v).hexdigest(),
                        }
                        for n, v in payloads.items()
                    ],
                }

                def response(url, headers=None):
                    stream = io.BytesIO(payloads[url.split("?")[0].rsplit("/", 1)[1]])
                    stream.status, stream.headers = 200, {}
                    return stream

                with (
                    patch.object(service.hub, "packages", return_value=[plan]),
                    patch("backend.hub.open_url", side_effect=response),
                ):
                    job = service.hub.start(
                        plan["path"], plan["revision"], str(root / "downloads"), model["id"]
                    )
                    service.hub.thread.join(5)
                result = service.store.get("downloads", job["id"])
                self.assertEqual(result["status"], "completed", result.get("error"))
                self.assertEqual(result["model"], model["id"])
                restored = service.store.get("models", model["id"])
                self.assertTrue(Path(restored["path"]).is_dir())
                self.assertEqual(
                    service.store.get("threads", "conversation")["model"], restored["id"]
                )
            finally:
                service.stop()
                service.hub.shutdown()
                service.store.close()
