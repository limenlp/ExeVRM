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

import torch

from llamafactory.model.model_utils.stp import (
    _stp_ensure_keep_mask_at_least_one_per_frame,
    _stp_qwen3vl_visual_forward_pruned,
)


class _DummyVisionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_seq_lens = []

    def forward(self, hidden_states, **kwargs):
        # Record the number of patch tokens this block processed.
        self.seen_seq_lens.append(int(hidden_states.shape[0]))
        return hidden_states + 1.0


class _DummyMerger(torch.nn.Module):
    """Mimics Qwen3-VL's patch merger interface.

    It accepts patch-level hidden states of shape (num_patches, hidden_size)
    and produces merged tokens by grouping consecutive patches in chunks of
    `merge_unit`.
    """

    def __init__(self, merge_unit: int, hidden_size: int, out_hidden_size: int):
        super().__init__()
        self.merge_unit = merge_unit
        self.proj = torch.nn.Linear(merge_unit * hidden_size, out_hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.dim() == 2
        n, d = hidden_states.shape
        assert n % self.merge_unit == 0
        x = hidden_states.view(-1, self.merge_unit * d)
        return self.proj(x)


class _DummyVision(torch.nn.Module):
    def __init__(self, spatial_merge_size: int = 2, in_dim: int = 6, hidden_size: int = 8, out_hidden_size: int = 4):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.config = types.SimpleNamespace(hidden_size=hidden_size, out_hidden_size=out_hidden_size)
        self.patch_embed = torch.nn.Linear(in_dim, hidden_size, bias=False)
        self.blocks = torch.nn.ModuleList([_DummyVisionBlock(), _DummyVisionBlock()])
        merge_unit = spatial_merge_size**2
        self.merger = _DummyMerger(merge_unit, hidden_size, out_hidden_size)
        self.deepstack_visual_indexes = [1]
        self.deepstack_merger_list = torch.nn.ModuleList(
            [_DummyMerger(merge_unit, hidden_size, out_hidden_size)]
        )

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        n = int(grid_thw.prod(dim=-1).sum().item())
        return torch.zeros((n, self.config.hidden_size), device=grid_thw.device, dtype=torch.float32)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        n = int(grid_thw.prod(dim=-1).sum().item())
        # Any feature dim works for this unit test because blocks ignore it.
        return torch.zeros((n, 2), device=grid_thw.device, dtype=torch.float32)


def test_stp_keep_mask_nonempty_per_frame():
    # 1 video with t=2 frames, each frame has (h=w=4) -> merged tokens per frame = (4/2)*(4/2)=4
    grid_thw = torch.tensor([[2, 4, 4]], dtype=torch.int64)
    merge_size = 2
    keep_mask = torch.zeros(8, dtype=torch.bool)
    fixed = _stp_ensure_keep_mask_at_least_one_per_frame(keep_mask, grid_thw, merge_size)
    assert fixed.shape == keep_mask.shape
    # At least 1 token per frame.
    assert fixed[:4].any().item() is True
    assert fixed[4:].any().item() is True


def test_stp_qwen3vl_visual_forward_pruned_scatter_and_compute_reduction():
    vision = _DummyVision(spatial_merge_size=2, in_dim=6, hidden_size=8, out_hidden_size=4)
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int64)

    # Total patch tokens = t*h*w = 16.
    pixel_values = torch.randn(16, 6)

    # Total merged tokens = 16 / (2*2) = 4.
    merged_keep_mask = torch.tensor([True, False, True, False], dtype=torch.bool)

    merged_full, deepstack_full = _stp_qwen3vl_visual_forward_pruned(
        vision, pixel_values, grid_thw, merged_keep_mask
    )

    assert merged_full.shape == (4, 4)
    assert isinstance(deepstack_full, list) and len(deepstack_full) == 1
    assert deepstack_full[0].shape == (4, 4)

    # Dropped tokens are scattered as zeros.
    assert torch.allclose(merged_full[1], torch.zeros_like(merged_full[1]))
    assert torch.allclose(merged_full[3], torch.zeros_like(merged_full[3]))

    # Kept tokens should be non-zero (very likely).
    assert not torch.allclose(merged_full[0], torch.zeros_like(merged_full[0]))
    assert not torch.allclose(merged_full[2], torch.zeros_like(merged_full[2]))

    # Compute reduction signal: each block should see only kept patches.
    # Keep mask keeps 2 merged tokens -> kept patches = 2 * 4 = 8.
    for blk in vision.blocks:
        assert blk.seen_seq_lens == [8]
