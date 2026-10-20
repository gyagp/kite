import unittest

import torch

from evo.cpu_weights import CpuWeights


class CpuWeightsTests(unittest.TestCase):
    def test_restores_original_storage_ties_and_nonpersistent_buffers(self):
        model = torch.nn.Module()
        model.first = torch.nn.Linear(4, 4, bias=False)
        model.second = torch.nn.Linear(4, 4, bias=False)
        model.second.weight = model.first.weight
        model.register_buffer("position", torch.arange(4, dtype=torch.float32), persistent=False)
        weight = model.first.weight
        pointer = weight.data_ptr()
        expected = weight.detach().clone()
        cpu = CpuWeights(model)
        for _ in range(3):
            model.to(dtype=torch.float64)
            self.assertNotEqual(model.first.weight.data_ptr(), pointer)
            cpu.restore()
            self.assertIs(model.first.weight, weight)
            self.assertIs(model.first.weight, model.second.weight)
            self.assertEqual(model.first.weight.data_ptr(), pointer)
            self.assertEqual(model.position.dtype, torch.float32)
            torch.testing.assert_close(model.first.weight, expected)
