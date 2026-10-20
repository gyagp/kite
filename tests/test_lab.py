import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.catalog import inspect_package
from backend.lab import COMMANDS, Evidence, image_settings, tool_request, validate_image
from backend.service import Service


class IntegratedLabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.service = Service(self.root / "state")

    def tearDown(self):
        self.service.stop()
        self.service.hub.shutdown()
        self.service.store.close()
        self.temp.cleanup()

    def test_complete_tool_catalog_and_local_engine(self):
        import evo

        root = Path(__file__).resolve().parents[1]
        self.assertTrue(Path(evo.__file__).resolve().is_relative_to(root))
        schemas = json.loads((root / "evo_resources/tools.json").read_text())
        self.assertEqual(set(schemas), set(COMMANDS))
        self.assertEqual(len(COMMANDS), 21)

    def test_tools_use_argument_arrays_and_reject_unknown_commands(self):
        self.assertEqual(
            tool_request({"operation": "inspect", "arguments": ["a; echo secret"]}),
            ["inspect", "a; echo secret"],
        )
        for payload in (
            {"operation": "python"},
            {"operation": "inspect", "arguments": "file.onnx"},
            {"operation": "inspect", "arguments": ["a\0"]},
        ):
            with self.assertRaises(ValueError):
                tool_request(payload)

    def test_image_input_contract_rejects_invalid_sizes_seeds_and_media(self):
        valid = {"prompt": "A kite", "width": 768, "height": 768, "seed": 42}
        with patch("backend.lab.image_readiness", return_value=[]):
            self.assertEqual(validate_image(self.service, valid)["images"], [])
            for overrides in (
                {"width": 800},
                {"height": True},
                {"seed": -1},
                {"seed": 2**32},
                {"prompt": ""},
                {"attachments": ["missing"]},
                {"capture": "false"},
                {"profile": "false"},
            ):
                with self.assertRaises(ValueError):
                    validate_image(self.service, {**valid, **overrides})

    def test_image_runtime_is_separate_from_chat_runtime(self):
        chat_library = self.service.settings["library"]
        self.service.store.put("preferences", {"id": "image-runtime", "library": "image.dll"})
        self.assertEqual(image_settings(self.service)["library"], "image.dll")
        self.assertEqual(self.service.settings["library"], chat_library)

    def test_evidence_migration_copies_state_without_modifying_evo(self):
        try:
            import onnx  # noqa: F401
        except ImportError:
            self.skipTest("Evidence inspection needs the authoring environment")
        source = self.root / "legacy"
        old = source / "artifacts/workbench"
        old.mkdir(parents=True)
        original = '[{"id":"w1","title":"Model A","status":"open","priority":"medium"}]'
        (old / "wishlist.json").write_text(original)
        self.service.store.put("settings", {**self.service.settings, "asset_root": str(source)})
        wb = Evidence(self.service).get()
        wb.save_wish({"id": "w1", "status": "done"})
        self.assertEqual((old / "wishlist.json").read_text(), original)
        self.assertEqual(wb.wishlist[0]["status"], "done")
        self.assertTrue(wb.directory.is_relative_to(self.service.store.directory))

    def test_image_packages_do_not_require_a_genai_tokenizer(self):
        package = self.root / "image"
        package.mkdir()
        for name in ("prefix-short-blocks.onnx", "denoise-short-blocks.onnx"):
            (package / name).write_bytes(b"fixture")
        (package / "image_config.json").write_text(
            json.dumps(
                {
                    "task": "image-generation",
                    "artifacts": {
                        "prefix-short-blocks.onnx": {"bytes": 7},
                        "denoise-short-blocks.onnx": {"bytes": 7},
                    },
                }
            )
        )
        self.assertEqual(inspect_package(package)["kind"], "image")
        (package / "denoise-short-blocks.onnx").write_bytes(b"short")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            inspect_package(package)


if __name__ == "__main__":
    unittest.main()
