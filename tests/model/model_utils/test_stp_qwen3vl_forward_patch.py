# Copyright 2026 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import types

import pytest
import torch

import llamafactory.model.model_utils.stp as stp
from llamafactory.extras.packages import is_transformers_version_greater_than
from llamafactory.model.model_utils.stp import patch_qwen3vl_forward_with_token_removal


@pytest.mark.skipif(not is_transformers_version_greater_than("4.57.0"), reason="Requires transformers>=4.57.0")
def test_stp_qwen3vl_forward_uses_visual_merge_size(monkeypatch: pytest.MonkeyPatch):
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast  # noqa: F401
    except Exception:
        pytest.skip("transformers does not provide Qwen3-VL modeling module")

    recorded: dict[str, int] = {}

    def _fake_compute_keep_mask(pixel_values, grid_thw, *args, merge_size=None, **kwargs):
        recorded["merge_size"] = int(merge_size)
        merged_tokens = 0
        for t, h, w in grid_thw.tolist():
            merged_tokens += int(t) * (int(h) // int(merge_size)) * (int(w) // int(merge_size))
        return torch.ones(merged_tokens, dtype=torch.bool, device=pixel_values.device)

    monkeypatch.setattr(stp, "compute_token_keep_mask_from_pixels", _fake_compute_keep_mask)

    class _DummyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(use_cache=True)

        def forward(self, **kwargs):
            embeds = kwargs["inputs_embeds"]
            return types.SimpleNamespace(
                last_hidden_state=embeds,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            )

    class _DummyVisual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # The vision tower is the source-of-truth.
            self.spatial_merge_size = 1
            self.dtype = torch.float32
            self.config = types.SimpleNamespace(spatial_merge_size=1, patch_size=14, temporal_patch_size=2)

        def forward(self, pixel_values, grid_thw=None, **kwargs):
            # Not used in this test (we keep all tokens), but must exist for patching.
            n = int((grid_thw.prod(dim=-1).sum().item()) // (self.spatial_merge_size**2))
            return torch.zeros((n, 3), device=pixel_values.device, dtype=torch.float32), None

    class _DummyInner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Intentionally stale config to ensure we read from `visual.*`.
            self.config = types.SimpleNamespace(
                output_attentions=False,
                output_hidden_states=False,
                use_return_dict=True,
                vision_config=types.SimpleNamespace(spatial_merge_size=2, patch_size=14, temporal_patch_size=2),
            )
            self.visual = _DummyVisual()
            self.language_model = _DummyLM()
            self.rope_deltas = None

        def get_image_features(self, pixel_values, grid_thw):
            # Produce embeddings sized to the vision tower merge size (1 -> 4 merged tokens for 2x2).
            n = int((grid_thw.prod(dim=-1).sum().item()) // (self.visual.spatial_merge_size**2))
            return (torch.zeros((n, 3), device=pixel_values.device, dtype=torch.float32),), None

        def get_placeholder_mask(self, input_ids, inputs_embeds=None, image_features=None, video_features=None):
            bsz, seqlen, hidden = inputs_embeds.shape
            mask = torch.zeros((bsz, seqlen, hidden), dtype=torch.bool, device=inputs_embeds.device)
            if image_features is not None:
                pos = (input_ids == 1).nonzero(as_tuple=False)
                for b, s in pos.tolist():
                    mask[b, s, :] = True
                return mask, None
            raise AssertionError("video path not used")

    class _DummyOuter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _DummyInner()
            self.config = types.SimpleNamespace(model_type="qwen3_vl", image_token_id=1)

        def forward(self, **kwargs):
            return self.model.forward(**kwargs)

    model = _DummyOuter()
    model_args = types.SimpleNamespace(
        use_stp=True,
        stp_mode="forward_removal",
        stp_threshold=0.5,
        stp_skip_ratio=0.0,
        stp_large_comp_threshold=0,
        stp_patch_level=False,
        stp_patch_to_token_strategy="any",
        stp_temporal_aggregation="first",
    )

    patch_qwen3vl_forward_with_token_removal(model, model_args)

    # 1 sample, 4 image placeholders.
    input_ids = torch.tensor([[0, 1, 1, 1, 1, 2]], dtype=torch.long)
    inputs_embeds = torch.zeros((1, 6, 3), dtype=torch.float32)
    position_ids = torch.zeros((3, 1, 6), dtype=torch.long)
    grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)
    pixel_values = torch.zeros((4, 3), dtype=torch.float32)

    model.model.forward(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        attention_mask=torch.ones((1, 6), dtype=torch.long),
        pixel_values=pixel_values,
        image_grid_thw=grid_thw,
    )

    assert recorded["merge_size"] == 1


@pytest.mark.skipif(not is_transformers_version_greater_than("4.57.0"), reason="Requires transformers>=4.57.0")
def test_stp_qwen3vl_training_disables_use_cache_by_default(monkeypatch: pytest.MonkeyPatch):
    """STP forward_removal patch should not accidentally enable KV cache during training.

    If `use_cache=None` is propagated down to the language model, it may fall back to
    config.use_cache=True and explode memory usage for forward+backward.
    """
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast  # noqa: F401
    except Exception:
        pytest.skip("transformers does not provide Qwen3-VL modeling module")

    def _fake_compute_keep_mask(pixel_values, grid_thw, *args, merge_size=None, **kwargs):
        merged_tokens = 0
        for t, h, w in grid_thw.tolist():
            merged_tokens += int(t) * (int(h) // int(merge_size)) * (int(w) // int(merge_size))
        return torch.ones(merged_tokens, dtype=torch.bool, device=pixel_values.device)

    monkeypatch.setattr(stp, "compute_token_keep_mask_from_pixels", _fake_compute_keep_mask)

    class _DummyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(use_cache=True)
            self.last_use_cache = None

        def forward(self, **kwargs):
            self.last_use_cache = kwargs.get("use_cache", "MISSING")
            embeds = kwargs["inputs_embeds"]
            return types.SimpleNamespace(
                last_hidden_state=embeds,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            )

    class _DummyVisual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.spatial_merge_size = 1
            self.dtype = torch.float32
            self.config = types.SimpleNamespace(spatial_merge_size=1, patch_size=14, temporal_patch_size=2)

    class _DummyInner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(
                output_attentions=False,
                output_hidden_states=False,
                use_return_dict=True,
                vision_config=types.SimpleNamespace(spatial_merge_size=1, patch_size=14, temporal_patch_size=2),
            )
            self.visual = _DummyVisual()
            self.language_model = _DummyLM()
            self.rope_deltas = None

        def get_image_features(self, pixel_values, grid_thw):
            n = int((grid_thw.prod(dim=-1).sum().item()) // (self.visual.spatial_merge_size**2))
            return (torch.zeros((n, 3), device=pixel_values.device, dtype=torch.float32),), None

        def get_placeholder_mask(self, input_ids, inputs_embeds=None, image_features=None, video_features=None):
            bsz, seqlen, hidden = inputs_embeds.shape
            mask = torch.zeros((bsz, seqlen, hidden), dtype=torch.bool, device=inputs_embeds.device)
            pos = (input_ids == 1).nonzero(as_tuple=False)
            for b, s in pos.tolist():
                mask[b, s, :] = True
            return mask, None

    class _DummyOuter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _DummyInner()
            self.config = types.SimpleNamespace(model_type="qwen3_vl", image_token_id=1)

        def forward(self, **kwargs):
            return self.model.forward(**kwargs)

    model = _DummyOuter()
    model_args = types.SimpleNamespace(
        use_stp=True,
        stp_mode="forward_removal",
        stp_threshold=0.5,
        stp_skip_ratio=0.0,
        stp_large_comp_threshold=0,
        stp_patch_level=False,
        stp_patch_to_token_strategy="any",
        stp_temporal_aggregation="first",
    )

    patch_qwen3vl_forward_with_token_removal(model, model_args)
    model.train()

    input_ids = torch.tensor([[0, 1, 1, 1, 1, 2]], dtype=torch.long)
    inputs_embeds = torch.zeros((1, 6, 3), dtype=torch.float32)
    position_ids = torch.zeros((3, 1, 6), dtype=torch.long)
    grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)
    pixel_values = torch.zeros((4, 3), dtype=torch.float32)

    model.model.forward(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        attention_mask=torch.ones((1, 6), dtype=torch.long),
        pixel_values=pixel_values,
        image_grid_thw=grid_thw,
    )

    assert model.model.language_model.last_use_cache is False

