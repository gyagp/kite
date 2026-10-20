import numpy as np
import pytest

from evo.embedding_multimodal import audio_block_mask, vision_pool_indices


def test_spatial_pooling_uses_coordinates_not_patch_storage_order():
    # Two 3x3 cells, deliberately shuffled, followed by padding.
    positions = np.array([(x, y) for y in range(3) for x in range(6)], dtype=np.int64)
    positions = positions[np.random.default_rng(4).permutation(18)]
    positions = np.concatenate([positions, np.full((9, 2), -1)])
    indices, valid = vision_pool_indices(positions[None])
    assert valid.tolist() == [[True, True, False]]
    assert set(map(tuple, positions[indices[0, 0]])) == {(x, y) for y in range(3) for x in range(3)}
    assert set(map(tuple, positions[indices[0, 1]])) == {
        (x, y) for y in range(3) for x in range(3, 6)
    }
    assert np.all(indices[0, 2] == 27)


def test_spatial_pooling_rejects_incomplete_cells():
    positions = np.full((1, 9, 2), -1, dtype=np.int64)
    positions[0, 0] = [0, 0]
    with pytest.raises(ValueError, match="complete pooling cells"):
        vision_pool_indices(positions)


def test_audio_context_strict_window_and_chunk_boundary():
    mask = audio_block_mask(np.ones((1, 99), dtype=bool))
    assert mask.shape == (1, 1, 3, 12, 24)
    # Block 1 starts at query 12 and key 0. Distance 12 is excluded;
    # distance 11 and the current key are included; future keys are excluded.
    assert mask[0, 0, 1, 0, 0]
    assert not mask[0, 0, 1, 0, 1]
    assert not mask[0, 0, 1, 0, 12]
    assert mask[0, 0, 1, 0, 13]
    assert mask[0, 0, 0, 0, :12].all()  # negative context positions
    assert mask[0, 0, 2, 1:].all()  # queries beyond the 25 valid positions


def test_audio_padding_keeps_per_sample_validity():
    features = np.ones((2, 99), dtype=bool)
    features[1, 48:] = False
    mask = audio_block_mask(features)
    assert not mask[0, 0, 1, 0, 12]
    assert mask[1, 0, 1, 0, 12]
    assert not mask[1, 0, 1, 0, 11]
