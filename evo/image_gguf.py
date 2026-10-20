"""Translate GGUF blocks to ORT packed weights without choosing new quantized integers.

Q4_K/Q5_K have fractional block offsets. Preserve those as a small correction
projection instead of pretending they are representable by integer zero points.
"""

import numpy as np


class GGUFWeightStore:
    def __init__(self, path, floating_zero_points=False):
        import gguf

        self.gguf = gguf
        self.floating_zero_points = floating_zero_points
        self.zero_points = {}
        self.reader = gguf.GGUFReader(str(path))
        self.mapping = {
            t.name.removeprefix("model.diffusion_model."): (t, None) for t in self.reader.tensors
        }
        for name, (tensor, _) in list(self.mapping.items()):
            if name.endswith("img_mlp.gate_up.weight"):
                rows = int(tensor.shape[-1]) // 2
                self.mapping[name.replace("gate_up", "gate_layer")] = (tensor, slice(0, rows))
                self.mapping[name.replace("gate_up", "proj")] = (tensor, slice(rows, rows * 2))
        self.validation = {}

    def tensor(self, name):
        tensor, rows = self.mapping[name.removeprefix("model.")]
        data = tensor.data if rows is None else tensor.data[rows]
        shape = list(reversed(tensor.shape.tolist()))
        if rows is not None:
            shape[0] = rows.stop - rows.start
        return tensor, data, shape

    def is_quantized(self, name):
        tensor, _, _ = self.tensor(name)
        if tensor.tensor_type.name not in {"Q4_K", "Q5_K", "Q6_K", "Q8_0", "F32", "F16", "BF16"}:
            raise ValueError(f"Unsupported GGUF tensor format: {tensor.tensor_type.name}")
        return tensor.tensor_type.name in {"Q4_K", "Q5_K", "Q6_K", "Q8_0"}

    def get(self, name):
        if name.removeprefix("model.") == "time_text_embed.time_proj.freqs":
            import torch

            return torch.exp(-np.log(10000) * torch.arange(128, dtype=torch.float32) / 128).numpy()
        tensor, data, shape = self.tensor(name)
        return self.gguf.quants.dequantize(data, tensor.tensor_type).reshape(shape)

    def packed(self, name):
        from gguf.quants import Q4_K

        tensor, data, (n, k) = self.tensor(name)
        kind = tensor.tensor_type.name
        _, byte_size = self.gguf.GGML_QUANT_SIZES[tensor.tensor_type]
        blocks = data.reshape(-1, byte_size)
        count = len(blocks)
        correction = None
        if kind in {"Q4_K", "Q5_K"}:
            d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
            dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)
            scale_codes, min_codes = Q4_K.get_scale_min(blocks[:, 4:16].copy())
            scale = (d * scale_codes).reshape(count, 8, 1)
            minimum = (dmin * min_codes).reshape(count, 8, 1)
            low = blocks[:, 16:] if kind == "Q4_K" else blocks[:, 48:]
            q = (
                (low.reshape(count, 4, 1, 32) >> np.array([0, 4], np.uint8)[None, None, :, None])
                & 15
            ).reshape(count, 8, 32)
            if kind == "Q5_K":
                high = (
                    (
                        blocks[:, 16:48].reshape(count, 1, 1, 32)
                        >> np.arange(8, dtype=np.uint8)[None, None, :, None]
                    )
                    & 1
                ).reshape(count, 8, 32)
                q = q | (high << 4)
            midpoint = 8 if kind == "Q4_K" else 16
            correction = (scale * midpoint - minimum).reshape(n, k // 32).T.copy()
            original_scale = scale.reshape(n, k // 32)
            q = q.reshape(n, k // 32, 32)
            if kind == "Q4_K":
                packed = q[..., ::2] | (q[..., 1::2] << 4)
                bits = 4
            else:
                packed = q + np.uint8(112)
                bits = 8
            block_size = 32
        elif kind == "Q8_0":
            original_scale = (
                blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(n, k // 32)
            )
            packed = (
                (blocks[:, 2:].copy().view(np.int8).astype(np.int16) + 128)
                .astype(np.uint8)
                .reshape(n, k // 32, 32)
            )
            bits, block_size = 8, 32
        elif kind == "Q6_K":
            low = (
                (
                    blocks[:, :128].reshape(count, 2, 1, 64)
                    >> np.array([0, 4], np.uint8)[None, None, :, None]
                )
                & 15
            ).reshape(count, 8, 32)
            high = (
                (
                    blocks[:, 128:192].reshape(count, 2, 1, 32)
                    >> np.array([0, 2, 4, 6], np.uint8)[None, None, :, None]
                )
                & 3
            ).reshape(count, 8, 32)
            q = (low | (high << 4)).astype(np.int16) - 32
            scale_codes = blocks[:, 192:208].copy().view(np.int8).astype(np.float32)
            d = blocks[:, 208:].copy().view(np.float16).astype(np.float32)
            original_scale = (d * scale_codes).reshape(n, k // 16)
            packed = (q.reshape(n, k // 16, 16) + 128).astype(np.uint8)
            bits, block_size = 8, 16
        else:
            raise ValueError(f"Unsupported GGUF quantization: {kind}")

        float_zero = None
        constants = 0
        if self.floating_zero_points and correction is not None:
            center = 8 if bits == 4 else 128
            if kind == "Q5_K":
                # Floating zero points need no offset-binary shift; keep 5-bit codes small.
                packed = packed - np.uint8(112)
                center = 16
            nonzero = original_scale != 0
            float_zero = center - correction.T / np.where(nonzero, original_scale, 1)
            constant = ~nonzero & (correction.T != 0)
            constants = int(np.count_nonzero(constant))
            if constants:
                # An affine constant block has no meaningful original integer code.
                packed[constant] = 0
                original_scale[constant] = -correction.T[constant]
                float_zero[constant] = 1
            float_zero[~nonzero & ~constant] = 0
            self.zero_points[name] = float_zero.astype(np.float16)
            correction = None
        scales = original_scale.astype(np.float16)
        offset = correction.astype(np.float16) if correction is not None else None
        # Validate decoded samples from independent GGUF code before serializing.
        rows = np.unique(np.linspace(0, n - 1, min(8, n), dtype=int))
        expected = self.gguf.quants.dequantize(data[rows].copy(), tensor.tensor_type).reshape(
            len(rows), k
        )
        if bits == 4:
            actual_q = (
                np.stack([packed[rows] & 15, packed[rows] >> 4], -1)
                .reshape(len(rows), k // block_size, block_size)
                .astype(np.float32)
            )
        else:
            actual_q = packed[rows].astype(np.float32)
        if float_zero is not None:
            exact = (actual_q - float_zero[rows, :, None]) * original_scale[rows, :, None]
            rounded = (
                actual_q - self.zero_points[name][rows, :, None].astype(np.float32)
            ) * scales[rows, :, None].astype(np.float32)
        else:
            actual_q -= 8 if bits == 4 else 128
            exact = actual_q * original_scale[rows, :, None]
            rounded = actual_q * scales[rows, :, None].astype(np.float32)
        if correction is not None:
            exact += correction.T[rows, :, None]
            rounded += offset.T[rows, :, None].astype(np.float32)
        np.testing.assert_allclose(exact.reshape(len(rows), k), expected, rtol=2e-5, atol=2e-7)
        self.validation[name] = {
            "source_type": kind,
            "ort_bits": bits,
            "block_size": block_size,
            "sample_rows": rows.tolist(),
            "integer_codes_preserved": constants == 0,
            "constant_blocks_reencoded": constants,
            "floating_zero_points": float_zero is not None,
            "max_fp16_parameter_rounding_error": float(
                np.max(np.abs(rounded.reshape(len(rows), k) - expected))
            ),
        }
        return packed, scales.reshape(-1), offset, bits, block_size
