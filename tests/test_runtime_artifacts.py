import hashlib
import tempfile
import unittest
from pathlib import Path

from evo.runtime_artifacts import verify_runtime_artifacts


class RuntimeArtifactsTests(unittest.TestCase):
    def test_module_relative_sdk_is_checked_for_corruption_and_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "D3D12").mkdir()
            path = root / "D3D12/D3D12Core.dll"
            path.write_bytes(b"sdk fixture")
            expected = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            manifest = {"resources": {"D3D12/D3D12Core.dll": expected}}
            verify_runtime_artifacts(root, manifest)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "differs from its build manifest"):
                verify_runtime_artifacts(root, manifest)
            for unsafe in ("../outside.dll", str(path.resolve())):
                with self.subTest(path=unsafe), self.assertRaisesRegex(ValueError, "escapes"):
                    verify_runtime_artifacts(root, {"resources": {unsafe: expected}})
