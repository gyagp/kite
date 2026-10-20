import json
import tempfile
import unittest
from pathlib import Path

from backend.image_profile import build_profile, save_trace, summarize_trace
from backend.profiling import finish_profile


class ImageProfileTests(unittest.TestCase):
    def test_cpu_only_profile_is_usable_without_mislabeling_gpu_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat_runtime.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "cat": "Node",
                            "name": "matrix_kernel_time",
                            "dur": 1200,
                            "args": {"op_name": "MatMul", "provider": "WebGpuExecutionProvider"},
                        }
                    ]
                )
            )
            result = finish_profile(directory, "job", "chat")
            self.assertEqual(result["gpu_total_ms"], 0)
            self.assertEqual(result["cpu_total_ms"], 1.2)
            self.assertEqual(result["top_cpu_ops"][0]["name"], "MatMul")
            self.assertTrue(result["warnings"])
            self.assertTrue((Path(directory) / "profile.json").is_file())

    def test_only_gpu_timestamps_contribute_to_shader_totals(self):
        events = [
            {
                "cat": "Node",
                "name": "CPU dispatch",
                "dur": 900000,
                "args": {"provider": "WebGpuExecutionProvider"},
            },
            {"cat": "Session", "name": "session_initialization", "dur": 1000000, "args": {}},
            {"cat": "Api", "name": "key&MatMul", "dur": 1200, "args": {"cache_key": "a"}},
            {"cat": "Api", "name": "other&MatMul", "dur": 800, "args": {"cache_key": "b"}},
            {"cat": "Api", "name": "key&Softmax", "dur": 500, "args": {"cache_key": "c"}},
            {"cat": "Api", "name": "Invalid", "dur": -100, "args": {"cache_key": "d"}},
            {"cat": "Api", "name": "Host API", "dur": 300000, "args": {}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps(events))
            summary = summarize_trace(path)
        self.assertEqual(summary["gpu_total_ms"], 2.5)
        self.assertEqual(summary["gpu_events"], 3)
        self.assertEqual(summary["shaders"][0], {"name": "MatMul", "calls": 2, "total_ms": 2.0})
        self.assertEqual(summary["cpu_nodes_by_provider"], {"WebGpuExecutionProvider": 1})
        combined = build_profile(
            {"prefix": {"profile": summary}, "denoise": {"profile": summary}}, "image"
        )
        self.assertEqual(combined["gpu_total_ms"], 5)
        self.assertEqual(sum(row["percent"] for row in combined["top_shaders"]), 100)

    def test_missing_gpu_events_and_invalid_traces_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.json"
            path.write_text('{"traceEvents": []}')
            summary = save_trace(path, directory, "prefix")
            self.assertTrue((Path(directory) / "prefix-trace.json").is_file())
            self.assertEqual(summary["gpu_events"], 0)
            combined = build_profile({"prefix": {"profile": summary}}, "id")
            self.assertEqual(len(combined["warnings"]), 2)
            self.assertEqual(combined["traces"][0]["url"], "/api/images/id/prefix-trace.json")
            path.write_text("invalid json")
            self.assertIn("error", save_trace(path, directory, "denoise"))
            self.assertIn("error", save_trace("", directory, "denoise"))
