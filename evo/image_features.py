"""The exact hidden-feature path used by Qwen-Image; no vocabulary projection."""

from types import SimpleNamespace

import torch


class ImageFeatureEncoder(torch.nn.Module):
    """Keep the official VLM backbone and expose its final pre-normalization features.

    Diffusers installs its existing final-norm hook on ``model.language_model``.
    Qwen-Image never consumes vocabulary logits, intermediate layers, or a KV cache.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype

    def forward(self, **kwargs):
        kwargs.update(output_hidden_states=False, use_cache=False, return_dict=True)
        output = self.model(**kwargs)
        return SimpleNamespace(hidden_states=(output.last_hidden_state,))
