import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from backend.catalog import discover, inspect_package
from backend.sam3 import MemoryBank
from backend.segmentation import GRAPHS, validate_tracking
from backend.store import Store


class SegmentationTests(unittest.TestCase):
    def test_video_package_requires_all_memory_artifacts_and_is_discoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = [*(g + ".onnx" for g in GRAPHS), "constants.json", "memory_constants.npz"]
            for name in names:
                (root / name).write_bytes(b"test")
            config = {
                "version": 1,
                "architecture": "Sam3TrackerVideo",
                "graphs": GRAPHS,
                "artifacts": {name: {"bytes": 4} for name in names},
            }
            (root / "segmentation_config.json").write_text(json.dumps(config))
            self.assertEqual(inspect_package(root)["kind"], "segmentation")
            self.assertEqual(len(discover(root)[0]), 1)
            (root / "memory_constants.npz").unlink()
            with self.assertRaisesRegex(ValueError, "memory_constants"):
                inspect_package(root)

    def test_request_rejects_unusable_prompts_and_nonfinite_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            path = Path(directory) / "clip.mp4"
            path.write_bytes(b"video")
            store.put("attachments", {"id": "clip", "kind": "video", "path": str(path)})
            service = SimpleNamespace(store=store)
            valid = {
                "attachment": "clip",
                "objects": [{"points": [{"x": 0.4, "y": 0.6, "label": 1}]}],
            }
            clean = validate_tracking(service, dict(valid))
            self.assertEqual(clean["objects"][0]["id"], 1)
            self.assertEqual(clean["stride"], 1)
            cases = [
                {"start_time": float("nan")},
                {"stride": True},
                {"max_frames": 601},
                {"objects": [{"points": [{"x": 0.4, "y": 0.6, "label": 0}]}]},
                {"objects": [{"box": [0.7, 0.7, 0.1, 0.2]}]},
                {"objects": [{"points": [{"x": float("inf"), "y": 0.4, "label": 1}]}]},
            ]
            for invalid in cases:
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    validate_tracking(service, {**valid, **invalid})
            store.close()

    def test_long_video_retains_conditioning_and_bounded_recent_memory(self):
        constants = {
            "memory_temporal_positional_encoding": np.arange(7, dtype=np.float32).reshape(
                7, 1, 1, 1
            ),
            "temporal_positional_encoding_projection_layer.weight": np.zeros((64, 256), np.float32),
            "temporal_positional_encoding_projection_layer.bias": np.zeros(64, np.float32),
        }
        bank = MemoryBank(constants, 100)
        for i in range(40):
            bank.add(
                i,
                np.full((2, 1, 64), i, np.float32),
                np.zeros((2, 1, 64), np.float32),
                np.full((1, 256), i, np.float32),
            )
        self.assertEqual(len(bank.frames), 17)
        feeds = bank.inputs(40)
        self.assertEqual(feeds["spatial_memory"].shape, (14, 1, 64))
        self.assertEqual(feeds["spatial_memory"][::2, 0, 0].tolist(), [0, 34, 35, 36, 37, 38, 39])
        self.assertEqual(
            feeds["spatial_memory_position_embeddings"][::2, 0, 0].tolist(), [6, 5, 4, 3, 2, 1, 0]
        )
        self.assertEqual(feeds["pointer_memory"].shape, (64, 1, 64))
        self.assertEqual(feeds["pointer_memory"][::4, 0, 0].tolist(), [0, *range(39, 24, -1)])


if __name__ == "__main__":
    unittest.main()
