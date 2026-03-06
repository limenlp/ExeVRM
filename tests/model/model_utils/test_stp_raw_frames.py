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

import torch

from llamafactory.model.model_utils.stp import compute_token_keep_mask_from_pixels


def test_stp_use_raw_frames_in_stp_ors_across_temporal_frames_and_cache_key():
    # 1 "temporal patch group" (t=1) with a 2x2 patch grid.
    # temporal_patch_size=2 means each patch contains two raw temporal frames.
    grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int64)

    # Minimal patch representation:
    # channel=1, patch_size=1, temporal_patch_size=2 -> patch_dim=2.
    # We interpret each patch vector as [frame0_value, frame1_value].
    # Frame0 is uniform (no edges). Frame1 varies per patch (edges everywhere).
    pixel_values = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
        ],
        dtype=torch.float32,
    )

    # With "first" aggregation, STP sees a uniform image -> single component -> keeps 1 token.
    keep_first = compute_token_keep_mask_from_pixels(
        pixel_values=pixel_values,
        grid_thw=grid_thw,
        threshold=0.1,
        skip_ratio=0.75,
        large_comp_threshold=0,
        patch_size=1,
        temporal_patch_size=2,
        merge_size=1,
        channel=1,
        patch_level=False,
        temporal_aggregation="first",
        use_raw_frames_in_stp=False,
    )
    assert keep_first.shape == (4,)
    assert int(keep_first.sum().item()) == 1

    # With raw-frame OR enabled, any UI element present in either raw frame should be kept.
    # Frame1 produces 4 singleton components -> keeps all 4.
    keep_or = compute_token_keep_mask_from_pixels(
        pixel_values=pixel_values,
        grid_thw=grid_thw,
        threshold=0.1,
        skip_ratio=0.75,
        large_comp_threshold=0,
        patch_size=1,
        temporal_patch_size=2,
        merge_size=1,
        channel=1,
        patch_level=False,
        temporal_aggregation="first",
        use_raw_frames_in_stp=True,
    )
    assert keep_or.shape == (4,)
    assert int(keep_or.sum().item()) == 4
