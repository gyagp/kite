import json
import tempfile
import unittest
from pathlib import Path

from backend.catalog import inspect_package
from backend.examples import example_asset
from backend.model_files import delete_package
from backend.service import Service


class ModelManagementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.service = Service(self.root / "state")
        self.package = self.root / "disposable-test-model"
        self.package.mkdir()
        (self.package / "tokenizer.json").write_text("{}")
        (self.package / "model.onnx").write_bytes(b"test fixture")
        (self.package / "genai_config.json").write_text(
            json.dumps({"model": {"type": "test", "decoder": {"filename": "model.onnx"}}})
        )
        self.model = self.service.store.put("models", inspect_package(self.package))

    def tearDown(self):
        self.service.stop()
        self.service.hub.shutdown()
        self.service.store.close()
        self.tmp.cleanup()

    def test_remove_reference_keeps_files(self):
        result = self.service.delete_model(self.model["id"])
        self.assertFalse(result["files_deleted"])
        self.assertTrue((self.package / "model.onnx").is_file())
        self.assertIsNone(self.service.store.get("models", self.model["id"]))

    def test_delete_from_disk_requires_matching_confirmation_then_removes_package(self):
        with self.assertRaisesRegex(ValueError, "exact model directory"):
            self.service.delete_model(self.model["id"], True, "C:/different-model")
        self.assertTrue(self.package.exists())
        result = self.service.delete_model(self.model["id"], True, str(self.package))
        self.assertTrue(result["files_deleted"])
        self.assertFalse(self.package.exists())
        self.assertFalse(result["removed"])
        self.assertEqual(
            self.service.store.get("models", self.model["id"])["name"], self.model["name"]
        )
        retained = next(m for m in self.service.models() if m["id"] == self.model["id"])
        self.assertFalse(retained["available"])

    def test_shared_or_overlapping_model_is_not_deleted(self):
        self.service.store.put(
            "models", {"id": "other", "name": "Nested model", "path": str(self.package / "nested")}
        )
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.service.delete_model(self.model["id"], True, str(self.package))
        self.assertTrue(self.package.exists())

    def test_workspace_root_is_never_deleted(self):
        with self.assertRaisesRegex(ValueError, "workspace"):
            delete_package(
                self.model,
                str(self.package),
                project=self.package,
                evo=self.root / "evo",
                data_directory=self.root / "state",
                other_models=[],
            )
        self.assertTrue(self.package.exists())

    def test_chat_defaults_to_4096_and_preferences_survive_restart(self):
        self.assertEqual(self.service.chat_preferences["max_tokens"], 4096)
        self.service.save_chat_preferences({"max_tokens": 1024})
        second = Service(self.root / "state")
        try:
            self.assertEqual(second.chat_preferences["max_tokens"], 1024)
        finally:
            second.store.close()
        for value in (0, 4097, "512", True):
            with self.assertRaises(ValueError):
                self.service.save_chat_preferences({"max_tokens": value})

    def test_only_declared_example_assets_are_served(self):
        demo = self.root / "demo"
        demo.mkdir()
        (demo / "public.jpg").write_bytes(b"image")
        (demo / "private.txt").write_text("not exposed")
        (demo / "examples.json").write_text(
            json.dumps([{"id": "example", "cover": "public.jpg", "samples": []}])
        )
        self.assertEqual(example_asset(self.root, "public.jpg"), demo / "public.jpg")
        for value in ("private.txt", "../private.txt", str(self.package / "genai_config.json")):
            with self.assertRaises(ValueError):
                example_asset(self.root, value)
