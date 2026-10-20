import gc
import mmap
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from backend.service import Service
from evo.initializer_cache import InitializerFileCache


class InitializerCacheTests(unittest.TestCase):
    def test_borrowed_mappings_reuse_invalidate_and_outlive_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = helper.make_graph(
                [helper.make_node("Add", ["x", "weight"], ["y"])],
                "mapped",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4096])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4096])],
                [numpy_helper.from_array(np.arange(4096, dtype=np.float32), "weight")],
            )
            model = helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10
            )
            path = root / "model.onnx"
            onnx.save_model(
                model,
                path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="weights.bin",
                size_threshold=0,
            )
            cache = InitializerFileCache()
            options = ort.SessionOptions()
            owner = cache.bind(path, options)
            self.assertEqual(owner.metadata["reused_files"], 0)
            session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            session._kite_initializer_mappings = owner
            reused = cache.bind(path, ort.SessionOptions())
            self.assertIs(reused.mappings[0], owner.mappings[0])
            self.assertEqual(reused.metadata["reused_files"], 1)
            with self.assertRaises(TypeError):
                owner.mappings[0][0] = 1
            stamp = (root / "weights.bin").stat()
            os.utime(
                root / "weights.bin", ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 2_000_000_000)
            )
            refreshed = cache.bind(path, ort.SessionOptions())
            self.assertIsNot(refreshed.mappings[0], owner.mappings[0])
            cache.clear()
            del owner, reused, refreshed, cache
            gc.collect()
            expected = np.arange(4096, dtype=np.float32) + 1
            np.testing.assert_array_equal(
                session.run(None, {"x": np.ones(4096, np.float32)})[0], expected
            )
            del session, options
            gc.collect()

    def test_image_deletion_releases_mapped_package_even_with_legacy_job_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "image-package"
            package.mkdir()
            (package / "image_config.json").write_text("{}")
            weights = package / "weights.bin"
            weights.write_bytes(b"test weights")
            with weights.open("rb") as stream:
                mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            service = Service(root / "state")
            try:
                service.store.put(
                    "models",
                    {"id": "image", "name": "Image", "kind": "image", "path": str(package)},
                )
                service.loaded = {"model": "legacy-image-job-id", "kind": "image"}
                service.image_worker_key = ("python", "library", "deps", "source", str(package))
                with patch.object(service, "_terminate", side_effect=mapping.close) as stop:
                    service.delete_model("image", True, str(package))
                stop.assert_called_once()
                self.assertFalse(package.exists())
                self.assertIsNotNone(service.store.get("models", "image"))
            finally:
                mapping.close()
                service.stop()
                service.hub.shutdown()
                service.store.close()
