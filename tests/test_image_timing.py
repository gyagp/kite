import types
import unittest
from unittest.mock import Mock, patch

from evo.image_timing import ImageTimer


class ImageTimingTests(unittest.TestCase):
    def test_cuda_completion_time_belongs_to_execution(self):
        clock = [10.0]

        def synchronize():
            clock[0] += 0.25

        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=synchronize))
        with patch.dict("sys.modules", {"torch": torch}), patch(
            "evo.image_timing.time.perf_counter", side_effect=lambda: clock[0]
        ):
            metrics = {}
            timer = ImageTimer(metrics)
            clock[0] += 0.1
            timer.mark("inference_ms", cuda=True)
            clock[0] += 0.05
            timer.mark("output_save_ms")
        self.assertAlmostEqual(metrics["inference_ms"], 350)
        self.assertAlmostEqual(metrics["output_save_ms"], 50)
        self.assertAlmostEqual(sum(metrics.values()), 400)

    def test_optional_metrics_do_not_synchronize_cli_workloads(self):
        synchronize = Mock()
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=synchronize))
        with patch.dict("sys.modules", {"torch": torch}):
            ImageTimer().mark("inference_ms", cuda=True)
        synchronize.assert_not_called()
