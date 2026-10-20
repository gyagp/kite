"""The image feature adapter must preserve pre-norm text and vision features."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.mark.parametrize("media", [False, True])
def test_feature_encoder_matches_original_hidden_states(media):
    from evo.image_features import ImageFeatureEncoder

    torch.manual_seed(31)
    config = transformers.Qwen3VLConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3]},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 64,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=125,
        video_token_id=126,
        vision_start_token_id=120,
        vision_end_token_id=121,
    )
    model = transformers.Qwen3VLForConditionalGeneration(config).eval()
    model.set_attn_implementation("eager")
    feature = ImageFeatureEncoder(model.model)
    hook = model.model.language_model.norm.register_forward_hook(lambda m, args, out: args[0])
    ids = torch.tensor([[3, 4, 120, 125, 121, 5, 6]] if media else [[3, 4, 5, 6, 7, 8, 9]])
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    if media:
        inputs.update(
            pixel_values=torch.randn(4, 1536),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            mm_token_type_ids=torch.tensor([[0, 0, 0, 1, 0, 0, 0]]),
        )
    try:
        with torch.no_grad():
            expected = model(**inputs, output_hidden_states=True).hidden_states[-1]
            actual = feature(**inputs).hidden_states[-1]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not hasattr(feature, "lm_head")
    finally:
        hook.remove()
