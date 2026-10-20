"""Exercise packed GGUF translation against GGUF's independent dequantizer."""

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")


@pytest.mark.parametrize("kind", ["Q4_K", "Q5_K", "Q6_K", "Q8_0"])
@pytest.mark.parametrize("floating", [False, True])
def test_gguf_blocks_preserve_codes_and_bound_parameter_rounding(tmp_path, kind, floating):
    from evo.image_gguf import GGUFWeightStore

    qtype = getattr(gguf.GGMLQuantizationType, kind)
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    n, k = 16, 256
    random = np.random.default_rng(20261017)
    blocks = random.integers(0, 256, (n * k // block, size), dtype=np.uint8)
    d = np.array([1 / 256], np.float16).view(np.uint8)
    if kind in {"Q4_K", "Q5_K"}:
        blocks[:, :2] = d
        blocks[:, 2:4] = np.array([1 / 4096], np.float16).view(np.uint8)
        blocks[0, :2] = 0  # A constant affine block: scale=0, nonzero minimum.
    elif kind == "Q6_K":
        blocks[:, -2:] = d
    else:
        blocks[:, :2] = d
    data = blocks.reshape(n, -1)
    path = tmp_path / "test.gguf"
    writer = gguf.GGUFWriter(str(path), "qwen_image")
    writer.add_tensor("model.diffusion_model.test.weight", data, raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    store = GGUFWeightStore(path, floating_zero_points=floating)
    packed, scales, offset, bits, bs = store.packed("model.test.weight")
    if bits == 4:
        codes = (
            np.stack([packed & 15, packed >> 4], -1).reshape(n, k // bs, bs).astype(np.float32) - 8
        )
    else:
        codes = packed.astype(np.float32) - 128
    if "model.test.weight" in store.zero_points:
        codes += 8 if bits == 4 else 128
        codes -= store.zero_points["model.test.weight"][:, :, None].astype(np.float32)
    values = codes * scales.reshape(n, k // bs, 1).astype(np.float32)
    if offset is not None:
        values += offset.T[:, :, None].astype(np.float32)
    expected = gguf.quants.dequantize(data, qtype).reshape(n, k)
    np.testing.assert_allclose(values.reshape(n, k), expected, rtol=0.001, atol=0.002)
    validation = store.validation["model.test.weight"]
    assert validation["integer_codes_preserved"] or validation["constant_blocks_reencoded"] > 0
    if kind in {"Q4_K", "Q5_K"}:
        assert (offset is not None or floating) and np.any(expected[0])
