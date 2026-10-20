"""Independent full-prefix versus cached ONNX diffusion checks on small models.

Run with Evo's pinned diffusion dependencies on sys.path and native WebGPU.
The normal lightweight environment skips this optional integration suite.
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")


@pytest.mark.parametrize("bits", [0, 4, 8])
def test_cached_denoiser_matches_reference_and_replays(tmp_path, bits):
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
    from safetensors.torch import save_file

    from evo.image_generate import NativeDenoiser, prefix_feeds, session
    from evo.image_qwen21 import WeightStore, bind_weights, prepare_capture_graphs, trace_graphs

    torch.set_num_threads(4)
    torch.manual_seed(19)
    model = (
        diffusers.QwenImage21Transformer2DModel(
            num_layers=2,
            num_attention_heads=1,
            attention_head_dim=128,
            context_in_dim=128,
        )
        .half()
        .eval()
    )
    # from_pretrained preserves the source's unlearned FP32 frequency buffer.
    model.time_text_embed.time_proj.freqs = torch.exp(
        -np.log(10000) * torch.arange(128, dtype=torch.float32) / 128
    )
    save_file(model.state_dict(), tmp_path / "tiny.safetensors")
    trace_graphs(dict(model.config), tmp_path)
    bind_weights(tmp_path, WeightStore(tmp_path), bits)
    (tmp_path / "image_config.json").write_text(json.dumps({"artifacts": {}}))
    prepare_capture_graphs(tmp_path)

    # A reference image between two text segments exercises block-causal prefix semantics.
    features = torch.randn(1, 9, 128).half()
    mask = torch.tensor([[False] * 3 + [True] * 4 + [False] * 2 + [True] * 4])
    expanded = np.array([False] * 3 + [True] * 16 + [False] * 2)
    prefix = len(expanded)
    condition = torch.randn(1, 16, 64).half()
    latent = torch.randn(1, 16, 64).half()
    shape = [[(1, 4, 4), (1, 4, 4)]]
    rot = model.pos_embed(shape[0], torch.tensor([*expanded, *([True] * 16)]), torch.device("cpu"))
    image_latents = np.zeros((1, prefix, 64), np.float16)
    image_latents[:, expanded] = condition.numpy()
    bias = np.tril(np.ones((prefix, prefix), bool))
    bias[3:19, 3:19] = True
    data = {
        "text": features.numpy(),
        "latents": latent.numpy(),
        "prefix_hidden": np.repeat(features.numpy(), [1] * 3 + [4] * 4 + [1] * 2, axis=1),
        "prefix_images": image_latents,
        "prefix_image_mask": expanded[None, :, None],
        "prefix_bias": np.where(bias, 0, -np.inf).astype(np.float16)[None, None],
        "cos": rot.real.numpy()[None, :, None],
        "sin": rot.imag.numpy()[None, :, None],
    }
    s, _ = session(tmp_path / "prefix.onnx", prefix, image_tokens=16)
    values = s.run(None, prefix_feeds(data))
    cache = dict(zip([v.name for v in s.get_outputs()], values, strict=True))
    del s
    native = NativeDenoiser(tmp_path, data, cache, True, tmp_path / "replay")
    errors = []
    for i, timestep in enumerate([0.8, 0.6, 0.4, 0.8]):
        with torch.no_grad():
            reference = model(
                torch.cat([condition, latent], dim=1),
                features,
                timestep=torch.tensor([timestep]).half(),
                img_shapes=shape,
                img_mask=mask,
                kv_cache=QwenImage21KVCache(2),
                kv_cache_mode="extract",
                return_dict=False,
            )[0][:, -16:]
        actual = native(latent.numpy(), timestep, i)
        expected = reference.float().numpy()
        cosine = float(
            np.dot(actual.ravel(), expected.ravel())
            / (np.linalg.norm(actual) * np.linalg.norm(expected))
        )
        assert cosine > (0.99 if bits == 4 else 0.999)
        errors.append(actual)
    np.testing.assert_array_equal(errors[0], errors[3])
    profile = json.loads(open(native.close(), encoding="utf-8").read())
    executions = sum(e.get("name") == "SequentialExecutor::Execute" for e in profile)
    runs = sum(e.get("name") == "model_run" for e in profile)
    assert runs == 4 and executions < runs
    assert not any(e.get("args", {}).get("provider") == "CPUExecutionProvider" for e in profile)
