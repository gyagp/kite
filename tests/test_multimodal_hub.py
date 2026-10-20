import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.catalog import inspect_package
from backend.hub import Hub, safe_path, verify_file
from backend.service import Service
from backend.store import Store


class HubTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "state")
        self.hub = Hub(self.store)

    def tearDown(self):
        self.hub.shutdown()
        self.store.close()
        self.temp.cleanup()

    def test_paths_cannot_escape_download_root(self):
        for bad in (
            "../outside",
            "/absolute",
            "C:/Windows",
            "a\\b",
            "a:stream",
            "a/../b",
            "folder.",
        ):
            with self.assertRaises(ValueError):
                safe_path(self.root, bad)

    def test_verifies_both_git_blobs_and_lfs_files(self):
        path = self.root / "weight.data"
        path.write_bytes(b"model bytes")
        content = path.read_bytes()
        self.assertTrue(
            verify_file(path, {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        )
        self.assertTrue(
            verify_file(
                path,
                {
                    "size": len(content),
                    "oid": hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest(),
                },
            )
        )
        path.write_bytes(b"wrong bytes")
        self.assertFalse(
            verify_file(path, {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        )

    def test_verified_download_import_and_resume_complete_partial(self):
        payloads = {
            "model.onnx": b"test graph placeholder",
            "genai_config.json": json.dumps(
                {"model": {"type": "test", "decoder": {"filename": "model.onnx"}}}
            ).encode(),
            "tokenizer.json": b"{}",
        }
        plan = {
            "id": "test-package",
            "name": "Test",
            "variant": "onnx-webgpu",
            "path": "Test/onnx-webgpu",
            "revision": "a" * 40,
            "size": sum(map(len, payloads.values())),
            "files": [
                {
                    "name": n,
                    "remote": "Test/onnx-webgpu/" + n,
                    "size": len(v),
                    "sha256": hashlib.sha256(v).hexdigest(),
                }
                for n, v in payloads.items()
            ],
        }
        # Simulate a previous download stopped after receiving the full graph.
        partial = self.root / "models/.kite-downloads/test-package/files/model.onnx.part"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(payloads["model.onnx"])

        def response(url, headers=None):
            name = url.split("?")[0].rsplit("/", 1)[1]
            self.assertNotEqual(name, "model.onnx")
            stream = io.BytesIO(payloads[name])
            stream.status = 200
            stream.headers = {}
            return stream

        with (
            patch.object(self.hub, "packages", return_value=[plan]),
            patch("backend.hub.open_url", side_effect=response),
        ):
            job = self.hub.start(plan["path"], plan["revision"], str(self.root / "models"))
            self.hub.thread.join(5)
        finished = self.store.get("downloads", job["id"])
        self.assertEqual(finished["status"], "completed", finished.get("error"))
        self.assertEqual(len(self.store.list("models")), 1)
        self.assertEqual(self.store.list("models")[0]["source"]["revision"], "a" * 40)

    def test_unrelated_destination_is_never_overwritten(self):
        plan = {
            "id": "x",
            "name": "Test",
            "variant": "onnx",
            "path": "Test/onnx",
            "revision": "a" * 40,
            "size": 1,
            "files": [],
        }
        destination = self.root / ("Test-onnx-" + "a" * 10)
        destination.mkdir()
        (destination / "mine.txt").write_text("keep")
        with (
            patch.object(self.hub, "packages", return_value=[plan]),
            self.assertRaisesRegex(ValueError, "already exists"),
        ):
            self.hub.start(plan["path"], plan["revision"], str(self.root))
        self.assertEqual((destination / "mine.txt").read_text(), "keep")

    def test_cancel_preserves_partial_and_resume_uses_verified_range(self):
        payloads = {
            "model.onnx": b"a model graph fixture",
            "genai_config.json": json.dumps(
                {"model": {"type": "test", "decoder": {"filename": "model.onnx"}}}
            ).encode(),
            "tokenizer.json": b"{}",
        }
        plan = {
            "id": "cancel-test",
            "name": "Test",
            "variant": "onnx",
            "path": "Test/onnx",
            "revision": "b" * 40,
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
        hub = self.hub

        class InterruptedStream(io.BytesIO):
            status = 200
            headers = {}

            def read1(self, size=-1):
                data = super().read(4)
                hub.stop_event.set()
                return data

        with (
            patch.object(hub, "packages", return_value=[plan]),
            patch("backend.hub.open_url", return_value=InterruptedStream(payloads["model.onnx"])),
        ):
            job = hub.start(plan["path"], plan["revision"], str(self.root / "models"))
            hub.thread.join(5)
        self.assertEqual(self.store.get("downloads", job["id"])["status"], "cancelled")
        ranges = []

        def resume(url, headers=None):
            name = url.split("?")[0].rsplit("/", 1)[1]
            offset = 4 if name == "model.onnx" else 0
            if offset:
                ranges.append(headers["Range"])
            stream = io.BytesIO(payloads[name][offset:])
            stream.status = 206 if offset else 200
            stream.headers = {
                "Content-Range": f"bytes {offset}-{len(payloads[name]) - 1}/{len(payloads[name])}"
            }
            return stream

        with (
            patch.object(hub, "packages", return_value=[plan]),
            patch("backend.hub.open_url", side_effect=resume),
        ):
            hub.start(plan["path"], plan["revision"], str(self.root / "models"))
            hub.thread.join(5)
        self.assertEqual(ranges, ["bytes=4-"])
        self.assertEqual(self.store.get("downloads", job["id"])["status"], "completed")


class MultimodalContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_full_export_requires_all_graphs_and_processor_assets(self):
        for name in (
            "model.onnx",
            "vision.onnx",
            "audio.onnx",
            "tokenizer.json",
            "processor_config.json",
            "config.json",
            "config_sentence_transformers.json",
        ):
            (self.root / name).write_text("{}")
        (self.root / "embedding_config.json").write_text(
            json.dumps(
                {
                    "task": "multimodal-embedding",
                    "architecture": "EmbeddingGemma2Model",
                    "modalities": ["text", "image", "audio", "video"],
                    "graphs": {n: {"file": n + ".onnx"} for n in ("model", "vision", "audio")},
                }
            )
        )
        model = inspect_package(self.root)
        self.assertEqual(model["modalities"], ["text", "image", "audio", "video"])
        (self.root / "audio.onnx").unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            inspect_package(self.root)

    def test_attachment_metadata_is_resolved_from_store_not_client(self):
        service = Service(self.root / "state")
        try:
            file = self.root / "image.png"
            file.write_bytes(b"image")
            service.store.put(
                "attachments",
                {
                    "id": "image",
                    "name": "image.png",
                    "kind": "image",
                    "mime": "image/png",
                    "size": 5,
                    "path": str(file),
                },
            )
            model = {"name": "multimodal", "modalities": ["text", "image", "audio", "video"]}
            samples = service.embedding_samples(
                [
                    {
                        "text": "",
                        "attachments": [
                            {"id": "image", "kind": "audio", "path": "C:/Windows/secret"}
                        ],
                    }
                ],
                model,
            )
            self.assertEqual(samples[0]["attachments"][0]["kind"], "image")
            self.assertNotIn("path", samples[0]["attachments"][0])
            self.assertEqual(
                service.worker_samples(samples)[0]["attachments"][0]["path"], str(file)
            )
            with self.assertRaisesRegex(ValueError, "placeholders"):
                service.embedding_samples(
                    [{"text": "<|image|> <|audio|>", "attachments": [{"id": "image"}]}], model
                )
        finally:
            service.hub.shutdown()
            service.store.close()
