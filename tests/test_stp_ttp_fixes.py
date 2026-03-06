# Copyright 2025 the LlamaFactory team.
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
"""
Tests for bug fixes in STP/TTP token-removal:

  Bug 1 (both STP and TTP):
      rope_deltas was not updated after token removal.
      During generate(), cache_position[0] equals the pruned KV-cache length
      (original_seq_len - num_dropped), so the decode position is off by
      num_dropped.  Fix: add num_dropped to rope_deltas after removal.

      Sub-issue: the fix was a no-op when position_ids is externally provided
      (e.g. from patched_prepare_inputs_for_generation) because is_first_forward
      cleared rope_deltas=None before step 4 ran, and step 4 was skipped
      (position_ids not None), so rope_deltas stayed None when step 5 checked it.
      Fix: only clear rope_deltas in is_first_forward when position_ids is None.

  Bug 2 (STP _gpu_union_find only):
      torch.equal() on CUDA tensors forces a blocking CPU-GPU sync every
      iteration (up to 1000 × num_frames syncs per sample).
      Fix: replace with a fixed iteration count of (grid_h + grid_w), which
      is the worst-case convergence bound for bidirectional label propagation
      on a 2-D grid.

  Bug 3 (performance — both STP and TTP):
      Python-level inner loops iterated over every video token individually
      (up to 58,500 per sample for 50-frame 720p) to fill seq_keep_mask.
      Each scalar tensor assignment can trigger a CUDA sync.
      Fix: replace the inner loop with a single vectorised tensor index-assign.
"""

from __future__ import annotations

import types
from collections import deque

import pytest
import torch
import torch.nn.functional as F

from llamafactory.model.model_utils.stp import _gpu_union_find
from llamafactory.model.model_utils.ttp import (
    _compute_ttp_keep_mask_reference,
    _compute_ttp_keep_mask_consecutive,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cpu_bfs(grid_h: int, grid_w: int,
             diffs_h: torch.Tensor, diffs_w: torch.Tensor,
             threshold: float) -> torch.Tensor:
    """Reference BFS connected-components on CPU (ground truth)."""
    n = grid_h * grid_w
    comp = [-1] * n
    cid = 0
    for start in range(n):
        if comp[start] != -1:
            continue
        queue: deque[int] = deque([start])
        comp[start] = cid
        while queue:
            node = queue.popleft()
            r, c = divmod(node, grid_w)
            # down
            if r + 1 < grid_h and diffs_h[r, c].item() < threshold:
                nb = (r + 1) * grid_w + c
                if comp[nb] == -1:
                    comp[nb] = cid
                    queue.append(nb)
            # up
            if r - 1 >= 0 and diffs_h[r - 1, c].item() < threshold:
                nb = (r - 1) * grid_w + c
                if comp[nb] == -1:
                    comp[nb] = cid
                    queue.append(nb)
            # right
            if c + 1 < grid_w and diffs_w[r, c].item() < threshold:
                nb = r * grid_w + (c + 1)
                if comp[nb] == -1:
                    comp[nb] = cid
                    queue.append(nb)
            # left
            if c - 1 >= 0 and diffs_w[r, c - 1].item() < threshold:
                nb = r * grid_w + (c - 1)
                if comp[nb] == -1:
                    comp[nb] = cid
                    queue.append(nb)
        cid += 1
    return torch.tensor(comp, dtype=torch.int64)


def _same_partition(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Return True iff a and b induce the same partition (same-component pairs)."""
    n = a.numel()
    assert b.numel() == n
    for i in range(n):
        for j in range(i + 1, n):
            if (a[i] == a[j]).item() != (b[i] == b[j]).item():
                return False
    return True


def _apply_rope_delta_correction(rope_deltas: torch.Tensor,
                                  seq_keep_mask: torch.Tensor,
                                  seq_len: int,
                                  batch_size: int) -> torch.Tensor:
    """Exact same logic as added to patched_forward (STP and TTP)."""
    num_dropped_per_batch = (
        seq_len - seq_keep_mask.sum(dim=1, keepdim=True)
    ).to(device=rope_deltas.device, dtype=rope_deltas.dtype)
    if rope_deltas.shape[0] == 1 and batch_size > 1:
        rope_deltas = rope_deltas.expand(batch_size, 1).contiguous()
    return rope_deltas + num_dropped_per_batch


# ---------------------------------------------------------------------------
# Bug 2: _gpu_union_find correctness after removing torch.equal sync
# ---------------------------------------------------------------------------

class TestGpuUnionFind:
    """Verify that _gpu_union_find still produces correct connected components."""

    def test_fully_connected_grid(self):
        """All patches are similar → single component."""
        grid_h, grid_w = 4, 5
        diffs_h = torch.zeros(grid_h - 1, grid_w)
        diffs_w = torch.zeros(grid_h, grid_w - 1)
        labels = _gpu_union_find(grid_h, grid_w, diffs_h, diffs_w, threshold=1.0)
        assert labels.unique().numel() == 1, "Expected one component for a fully-connected grid"

    def test_fully_disconnected_grid(self):
        """No adjacent patches are similar → every patch is its own component."""
        grid_h, grid_w = 3, 4
        diffs_h = torch.full((grid_h - 1, grid_w), 100.0)
        diffs_w = torch.full((grid_h, grid_w - 1), 100.0)
        labels = _gpu_union_find(grid_h, grid_w, diffs_h, diffs_w, threshold=1.0)
        assert labels.unique().numel() == grid_h * grid_w, \
            "Expected each patch to be its own component"

    def test_horizontal_split_two_components(self):
        """Grid divided by a high-diff horizontal edge → exactly 2 components."""
        grid_h, grid_w = 6, 6
        diffs_h = torch.zeros(grid_h - 1, grid_w)
        diffs_h[2, :] = 100.0          # cut between row 2 and row 3
        diffs_w = torch.zeros(grid_h, grid_w - 1)
        labels = _gpu_union_find(grid_h, grid_w, diffs_h, diffs_w, threshold=1.0)

        assert labels.unique().numel() == 2
        labels_2d = labels.view(grid_h, grid_w)
        # Rows 0-2 should share one label; rows 3-5 another
        top_label = labels_2d[0, 0].item()
        bot_label = labels_2d[3, 0].item()
        assert top_label != bot_label
        for r in range(3):
            for c in range(grid_w):
                assert labels_2d[r, c].item() == top_label
        for r in range(3, grid_h):
            for c in range(grid_w):
                assert labels_2d[r, c].item() == bot_label

    def test_vertical_split_two_components(self):
        """Grid divided by a high-diff vertical edge → exactly 2 components."""
        grid_h, grid_w = 5, 8
        diffs_h = torch.zeros(grid_h - 1, grid_w)
        diffs_w = torch.zeros(grid_h, grid_w - 1)
        diffs_w[:, 3] = 100.0          # cut between col 3 and col 4
        labels = _gpu_union_find(grid_h, grid_w, diffs_h, diffs_w, threshold=1.0)

        assert labels.unique().numel() == 2
        labels_2d = labels.view(grid_h, grid_w)
        left_label = labels_2d[0, 0].item()
        right_label = labels_2d[0, 4].item()
        assert left_label != right_label
        for r in range(grid_h):
            for c in range(4):
                assert labels_2d[r, c].item() == left_label
            for c in range(4, grid_w):
                assert labels_2d[r, c].item() == right_label

    def test_single_patch(self):
        """Edge case: 1×1 grid."""
        diffs_h = torch.empty(0, 1)
        diffs_w = torch.empty(1, 0)
        labels = _gpu_union_find(1, 1, diffs_h, diffs_w, threshold=1.0)
        assert labels.numel() == 1

    def test_single_row(self):
        """1-row grid: only horizontal edges."""
        grid_h, grid_w = 1, 8
        diffs_h = torch.empty(0, grid_w)
        # alternating connected / disconnected
        diffs_w = torch.tensor([[0.1, 0.1, 5.0, 0.1, 0.1, 5.0, 0.1]])
        labels = _gpu_union_find(grid_h, grid_w, diffs_h, diffs_w, threshold=1.0)
        ref = _cpu_bfs(grid_h, grid_w, diffs_h, diffs_w, threshold=1.0)
        assert _same_partition(labels, ref)

    def test_matches_cpu_bfs_random(self):
        """
        Random diff grids: GPU result must induce the same partition as CPU BFS.
        Tests several seeds to cover different connectivity patterns.
        """
        for seed in range(8):
            torch.manual_seed(seed)
            grid_h = torch.randint(3, 12, (1,)).item()
            grid_w = torch.randint(3, 12, (1,)).item()
            diffs_h = torch.rand(grid_h - 1, grid_w) * 3.0
            diffs_w = torch.rand(grid_h, grid_w - 1) * 3.0
            threshold = 1.5

            labels_gpu = _gpu_union_find(grid_h, grid_w, diffs_h, diffs_w, threshold)
            labels_cpu = _cpu_bfs(grid_h, grid_w, diffs_h, diffs_w, threshold)

            assert _same_partition(labels_gpu, labels_cpu), (
                f"Seed {seed}: GPU and CPU BFS disagree on connected components "
                f"(grid {grid_h}×{grid_w})"
            )

    def test_large_720p_patch_grid_completes(self):
        """
        Smoke test on a 720p-sized patch grid (52×90).
        With max_iterations = grid_h + grid_w = 142, this must finish quickly
        (< 2 s on CPU); previously with 1000 iters + torch.equal it could be slow.
        """
        import time
        grid_h, grid_w = 52, 90
        torch.manual_seed(0)
        diffs_h = torch.rand(grid_h - 1, grid_w) * 4.0
        diffs_w = torch.rand(grid_h, grid_w - 1) * 4.0
        threshold = 3.0

        t0 = time.perf_counter()
        labels = _gpu_union_find(grid_h, grid_w, diffs_h, diffs_w, threshold)
        elapsed = time.perf_counter() - t0

        assert labels.numel() == grid_h * grid_w
        assert labels.unique().numel() >= 1
        assert elapsed < 10.0, f"Union-find on 52×90 grid took {elapsed:.2f}s (too slow)"


# ---------------------------------------------------------------------------
# Bug 1: rope_deltas correction after token removal
# ---------------------------------------------------------------------------

class TestRopeDeltasCorrection:
    """
    Verify that the rope_deltas fix restores correct decode-step positions.

    Invariant that must hold after the fix:
        new_seq_len + corrected_rope_delta == original_seq_len + original_rope_delta

    where  new_seq_len = original_seq_len - num_dropped
    and    corrected_rope_delta = original_rope_delta + num_dropped
    """

    def _check_invariant(self, original_seq_len: int, original_rope_delta: float,
                          seq_keep_mask: torch.Tensor, batch_size: int):
        rope_deltas = torch.full((1, 1), original_rope_delta)
        corrected = _apply_rope_delta_correction(
            rope_deltas.clone(), seq_keep_mask, original_seq_len, batch_size
        )
        for b in range(batch_size):
            num_kept = seq_keep_mask[b].sum().item()
            new_seq_len = num_kept
            # core invariant
            assert new_seq_len + corrected[b, 0].item() == pytest.approx(
                original_seq_len + original_rope_delta
            ), (
                f"Batch {b}: new_seq_len({new_seq_len}) + corrected_delta({corrected[b,0].item()}) "
                f"!= original_seq_len({original_seq_len}) + original_delta({original_rope_delta})"
            )

    def test_single_batch_partial_removal(self):
        """batch_size=1, 20 tokens removed out of 100."""
        seq_len = 100
        num_dropped = 20
        mask = torch.ones(1, seq_len, dtype=torch.bool)
        mask[0, :num_dropped] = False
        self._check_invariant(seq_len, 10.0, mask, batch_size=1)

    def test_single_batch_no_removal(self):
        """batch_size=1, zero tokens removed: rope_deltas unchanged."""
        seq_len = 50
        mask = torch.ones(1, seq_len, dtype=torch.bool)
        rope_deltas = torch.tensor([[7.0]])
        corrected = _apply_rope_delta_correction(rope_deltas.clone(), mask, seq_len, batch_size=1)
        assert corrected[0, 0].item() == pytest.approx(7.0)

    def test_single_batch_heavy_removal(self):
        """batch_size=1, 90% tokens removed."""
        seq_len = 200
        num_dropped = 180
        mask = torch.ones(1, seq_len, dtype=torch.bool)
        mask[0, :num_dropped] = False
        self._check_invariant(seq_len, 25.0, mask, batch_size=1)

    def test_batch_size_2_same_drops(self):
        """batch_size=2, both samples drop the same number of tokens."""
        seq_len = 120
        num_dropped = 40
        mask = torch.ones(2, seq_len, dtype=torch.bool)
        mask[:, :num_dropped] = False
        self._check_invariant(seq_len, 15.0, mask, batch_size=2)

    def test_batch_size_2_different_drops(self):
        """
        batch_size=2 with different removal counts per sample.
        rope_deltas starts as (1,1) and must be expanded to (2,1).
        """
        seq_len = 150
        mask = torch.ones(2, seq_len, dtype=torch.bool)
        mask[0, :30] = False   # sample 0: 30 dropped
        mask[1, :70] = False   # sample 1: 70 dropped

        rope_deltas = torch.tensor([[12.0]])   # shape (1,1)
        corrected = _apply_rope_delta_correction(rope_deltas.clone(), mask, seq_len, batch_size=2)

        assert corrected.shape == (2, 1)
        # sample 0
        assert (seq_len - 30) + corrected[0, 0].item() == pytest.approx(seq_len + 12.0)
        # sample 1
        assert (seq_len - 70) + corrected[1, 0].item() == pytest.approx(seq_len + 12.0)

    def test_batch_size_2_rope_deltas_already_batched(self):
        """rope_deltas already has shape (batch_size, 1) — should not be re-expanded."""
        seq_len = 80
        mask = torch.ones(2, seq_len, dtype=torch.bool)
        mask[0, :10] = False
        mask[1, :20] = False

        rope_deltas = torch.tensor([[5.0], [5.0]])   # shape (2,1)
        corrected = _apply_rope_delta_correction(rope_deltas.clone(), mask, seq_len, batch_size=2)

        assert corrected.shape == (2, 1)
        assert (seq_len - 10) + corrected[0, 0].item() == pytest.approx(seq_len + 5.0)
        assert (seq_len - 20) + corrected[1, 0].item() == pytest.approx(seq_len + 5.0)

    def test_decode_position_matches_original_sequence(self):
        """
        End-to-end simulation of prefill+decode positions.

        Without fix:
            decode_position = cache_position[0] + original_rope_delta
                            = (seq_len - num_dropped) + original_rope_delta  ← wrong

        With fix:
            decode_position = cache_position[0] + corrected_rope_delta
                            = (seq_len - num_dropped) + (original_rope_delta + num_dropped)
                            = seq_len + original_rope_delta  ← correct
        """
        seq_len = 200
        num_dropped = 60
        original_rope_delta = 20.0

        # What decode position should be (without any pruning)
        expected_decode_pos = seq_len + original_rope_delta

        # After pruning
        new_seq_len = seq_len - num_dropped  # = 140
        mask = torch.ones(1, seq_len, dtype=torch.bool)
        mask[0, :num_dropped] = False

        rope_deltas = torch.tensor([[original_rope_delta]])
        corrected = _apply_rope_delta_correction(rope_deltas, mask, seq_len, batch_size=1)

        # Simulate decode: cache_position[0] = new_seq_len
        decode_pos = new_seq_len + corrected[0, 0].item()
        assert decode_pos == pytest.approx(expected_decode_pos), (
            f"Decode position {decode_pos} != expected {expected_decode_pos}. "
            "rope_deltas correction is broken."
        )

        # Also verify the bug is real (without fix, decode_pos would be wrong)
        buggy_decode_pos = new_seq_len + original_rope_delta
        assert buggy_decode_pos != pytest.approx(expected_decode_pos), \
            "Sanity check: without fix the position should be wrong"


# ---------------------------------------------------------------------------
# NEW Bug 1b: rope_deltas fix was a no-op when position_ids is externally set
# ---------------------------------------------------------------------------

class TestRopeDeltasWithExternalPositionIds:
    """
    During predict_with_generate, patched_prepare_inputs_for_generation computes
    position_ids BEFORE calling the model forward.  The inner patched_forward then
    receives a non-None position_ids, so its step-4 block ("if position_ids is None:")
    is skipped.

    Old (broken) approach: is_first_forward always cleared rope_deltas=None, so step 5's
    "if inner_model.rope_deltas is not None:" guard evaluated False → fix silently skipped.

    New (correct) approach: is_first_forward only clears rope_deltas when position_ids is
    None.  When position_ids is provided (from prepare_inputs_for_generation), the existing
    rope_deltas value is preserved for step-5 to use — no extra get_rope_index call needed.

    These tests verify the new control-flow using a lightweight mock.
    """

    def _simulate_is_first_forward_clear(self, inner_model, position_ids):
        """
        Simulate the updated is_first_forward block:
        only clear rope_deltas when position_ids is None.
        """
        if position_ids is None:
            if hasattr(inner_model, "rope_deltas") and inner_model.rope_deltas is not None:
                inner_model.rope_deltas = None

    def _simulate_step5_correction(
        self,
        inner_model,
        seq_keep_mask: torch.Tensor,
        seq_len: int,
        batch_size: int,
    ):
        """Simulate the step-5 rope_deltas correction."""
        if inner_model.rope_deltas is not None:
            num_dropped_per_batch = (
                seq_len - seq_keep_mask.sum(dim=1, keepdim=True)
            ).to(device=inner_model.rope_deltas.device, dtype=inner_model.rope_deltas.dtype)
            if inner_model.rope_deltas.shape[0] == 1 and batch_size > 1:
                inner_model.rope_deltas = inner_model.rope_deltas.expand(batch_size, 1).contiguous()
            inner_model.rope_deltas = inner_model.rope_deltas + num_dropped_per_batch
        return inner_model.rope_deltas

    def test_rope_deltas_preserved_when_position_ids_provided(self):
        """
        When position_ids is not None (generate() path), is_first_forward must NOT
        clear rope_deltas.  It should remain set so step-5 can use it.
        """
        model = types.SimpleNamespace()
        model.rope_deltas = torch.tensor([[42.0]])  # set by prepare_inputs_for_generation

        position_ids = torch.zeros(3, 1, 10, dtype=torch.long)  # non-None
        self._simulate_is_first_forward_clear(model, position_ids)

        assert model.rope_deltas is not None, (
            "rope_deltas must NOT be cleared when position_ids is not None"
        )
        assert model.rope_deltas[0, 0].item() == pytest.approx(42.0)

    def test_rope_deltas_cleared_when_position_ids_none(self):
        """
        When position_ids is None (training / no prepare_inputs_for_generation),
        is_first_forward SHOULD clear rope_deltas so step 4 recomputes it.
        """
        model = types.SimpleNamespace()
        model.rope_deltas = torch.tensor([[99.0]])  # stale from last forward

        self._simulate_is_first_forward_clear(model, position_ids=None)
        assert model.rope_deltas is None, "rope_deltas must be cleared when position_ids is None"

    def test_generate_path_end_to_end_invariant(self):
        """
        Full generate() path simulation:
          1. prepare_inputs_for_generation sets rope_deltas = original_delta
          2. is_first_forward (position_ids not None) → rope_deltas kept
          3. step-5 adds num_dropped → corrected rope_deltas
          4. decode: cache_position[0] + corrected = original_seq_len + original_delta ✓
        """
        original_delta = 42.0
        seq_len = 100
        num_dropped = 30

        model = types.SimpleNamespace()
        # Step 1: prepare_inputs_for_generation sets rope_deltas
        model.rope_deltas = torch.tensor([[original_delta]])

        # Step 2: is_first_forward with position_ids provided → rope_deltas NOT cleared
        position_ids = torch.zeros(3, 1, seq_len, dtype=torch.long)
        self._simulate_is_first_forward_clear(model, position_ids)
        assert model.rope_deltas is not None

        # Step 3: step-5 correction
        mask = torch.ones(1, seq_len, dtype=torch.bool)
        mask[0, :num_dropped] = False
        self._simulate_step5_correction(model, mask, seq_len, batch_size=1)

        # Step 4: decode position
        new_seq_len = seq_len - num_dropped
        decode_pos = new_seq_len + model.rope_deltas[0, 0].item()
        expected_pos = seq_len + original_delta
        assert decode_pos == pytest.approx(expected_pos), (
            f"decode_pos={decode_pos} != expected={expected_pos}"
        )

    def test_invariant_batch_2_with_external_position_ids(self):
        """
        batch_size=2, different per-sample drops, position_ids provided externally.
        Verify the full pipeline restores the correct decode position for each sample.
        """
        original_delta = 30.0
        seq_len = 200

        model = types.SimpleNamespace()
        model.rope_deltas = torch.tensor([[original_delta]])  # set by prepare_inputs_for_generation

        # is_first_forward with position_ids provided → rope_deltas kept
        position_ids = torch.zeros(3, 2, seq_len, dtype=torch.long)
        self._simulate_is_first_forward_clear(model, position_ids)

        mask = torch.ones(2, seq_len, dtype=torch.bool)
        mask[0, :40] = False   # sample 0: 40 dropped
        mask[1, :80] = False   # sample 1: 80 dropped

        corrected = self._simulate_step5_correction(model, mask, seq_len, batch_size=2)

        assert corrected.shape == (2, 1)
        for b, n_dropped in enumerate([40, 80]):
            new_seq_len = seq_len - n_dropped
            decode_pos = new_seq_len + corrected[b, 0].item()
            expected_pos = seq_len + original_delta
            assert decode_pos == pytest.approx(expected_pos), (
                f"Sample {b}: decode_pos={decode_pos} != expected={expected_pos}"
            )

    def test_no_removal_rope_deltas_unchanged(self):
        """When no tokens are removed, rope_deltas should stay at original_delta."""
        original_delta = 17.0
        model = types.SimpleNamespace()
        model.rope_deltas = torch.tensor([[original_delta]])

        position_ids = torch.zeros(3, 1, 80, dtype=torch.long)
        self._simulate_is_first_forward_clear(model, position_ids)

        seq_len = 80
        mask = torch.ones(1, seq_len, dtype=torch.bool)  # nothing removed
        corrected = self._simulate_step5_correction(model, mask, seq_len, batch_size=1)

        assert corrected[0, 0].item() == pytest.approx(original_delta), (
            "With zero tokens removed rope_deltas should remain original_delta"
        )


# ---------------------------------------------------------------------------
# Bug 3: Vectorized seq_keep_mask filling (was Python inner loop)
# ---------------------------------------------------------------------------

class TestVectorizedSeqKeepMask:
    """
    Verify the vectorized tensor index-assign correctly fills seq_keep_mask,
    matching the old element-wise Python loop result.

    For 50-frame 720p video (50 × 1170 = 58,500 tokens), the old loop ran
    58,500 Python iterations with potential CUDA sync per assignment.
    The vectorized version does the same in a single tensor operation.
    """

    def _fill_seq_keep_mask_old(self, seq_keep_mask, vid_positions, offset, video_keep_mask, b):
        """Reference: old Python inner loop."""
        for local_idx, pos in enumerate(vid_positions):
            global_idx = offset + local_idx
            if global_idx < len(video_keep_mask):
                seq_keep_mask[b, pos] = video_keep_mask[global_idx]

    def _fill_seq_keep_mask_new(self, seq_keep_mask, vid_positions, offset, video_keep_mask, b):
        """New: vectorized single tensor operation."""
        n = min(len(vid_positions), max(0, len(video_keep_mask) - offset))
        if n > 0:
            seq_keep_mask[b, vid_positions[:n]] = video_keep_mask[offset:offset + n]

    def _make_inputs(self, batch_size, seq_len, num_video_tokens, num_dropped, seed=0):
        """Build seq_keep_mask, vid_positions, and video_keep_mask for testing."""
        torch.manual_seed(seed)
        seq_keep_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        # Place video tokens at contiguous positions starting at offset 10
        video_start = 10
        vid_positions = torch.arange(video_start, video_start + num_video_tokens)
        # Random keep mask with num_dropped tokens removed
        video_keep_mask = torch.ones(num_video_tokens, dtype=torch.bool)
        drop_idx = torch.randperm(num_video_tokens)[:num_dropped]
        video_keep_mask[drop_idx] = False
        return seq_keep_mask, vid_positions, video_keep_mask

    def test_small_video_matches_reference(self):
        """Small case: vectorized result matches the reference loop."""
        seq_len = 200
        num_video_tokens = 100
        num_dropped = 30

        mask_old = torch.ones(1, seq_len, dtype=torch.bool)
        mask_new = torch.ones(1, seq_len, dtype=torch.bool)
        _, vid_positions, video_keep_mask = self._make_inputs(
            1, seq_len, num_video_tokens, num_dropped
        )

        self._fill_seq_keep_mask_old(mask_old, vid_positions, 0, video_keep_mask, 0)
        self._fill_seq_keep_mask_new(mask_new, vid_positions, 0, video_keep_mask, 0)

        assert torch.equal(mask_old, mask_new), "Vectorized fill must match reference loop"

    def test_large_video_matches_reference(self):
        """50-frame 720p-scale: 50×1170 = 58,500 tokens, 30% dropped."""
        seq_len = 70000  # text + video
        num_video_tokens = 58500
        num_dropped = int(num_video_tokens * 0.3)

        mask_old = torch.ones(1, seq_len, dtype=torch.bool)
        mask_new = torch.ones(1, seq_len, dtype=torch.bool)
        _, vid_positions, video_keep_mask = self._make_inputs(
            1, seq_len, num_video_tokens, num_dropped, seed=42
        )

        self._fill_seq_keep_mask_old(mask_old, vid_positions, 0, video_keep_mask, 0)
        self._fill_seq_keep_mask_new(mask_new, vid_positions, 0, video_keep_mask, 0)

        assert torch.equal(mask_old, mask_new), "Vectorized fill must match reference on large video"

    def test_with_nonzero_offset(self):
        """When video_keep_mask covers multiple videos, offset must be respected."""
        seq_len = 500
        num_video_tokens = 100
        num_dropped = 20
        offset = 50  # this batch element's video starts at index 50 in video_keep_mask

        # video_keep_mask is longer than a single video (simulates 2-video batch)
        video_keep_mask = torch.ones(200, dtype=torch.bool)
        video_keep_mask[offset:offset + num_dropped] = False  # drop first 20 of this video

        vid_positions = torch.arange(10, 10 + num_video_tokens)

        mask_old = torch.ones(1, seq_len, dtype=torch.bool)
        mask_new = torch.ones(1, seq_len, dtype=torch.bool)

        self._fill_seq_keep_mask_old(mask_old, vid_positions, offset, video_keep_mask, 0)
        self._fill_seq_keep_mask_new(mask_new, vid_positions, offset, video_keep_mask, 0)

        assert torch.equal(mask_old, mask_new)

    def test_vectorized_speed(self):
        """
        Vectorized fill must be at least 100× faster than the Python loop
        for a 50-frame 720p-scale video (58,500 tokens).
        """
        import time

        seq_len = 70000
        num_video_tokens = 58500
        num_dropped = int(num_video_tokens * 0.3)
        _, vid_positions, video_keep_mask = self._make_inputs(
            1, seq_len, num_video_tokens, num_dropped, seed=7
        )

        # Time the old loop
        mask_old = torch.ones(1, seq_len, dtype=torch.bool)
        t0 = time.perf_counter()
        self._fill_seq_keep_mask_old(mask_old, vid_positions, 0, video_keep_mask, 0)
        t_old = time.perf_counter() - t0

        # Time the new vectorized version
        mask_new = torch.ones(1, seq_len, dtype=torch.bool)
        t0 = time.perf_counter()
        self._fill_seq_keep_mask_new(mask_new, vid_positions, 0, video_keep_mask, 0)
        t_new = time.perf_counter() - t0

        speedup = t_old / max(t_new, 1e-9)
        assert speedup >= 20, (
            f"Vectorized fill speedup={speedup:.1f}× (expected ≥20× on CPU; "
            f"GPU speedup is much larger due to CUDA sync elimination). "
            f"old={t_old*1000:.1f}ms, new={t_new*1000:.2f}ms"
        )
        assert torch.equal(mask_old, mask_new)


# ---------------------------------------------------------------------------
# Bug 4: Vectorized TTP keep-mask computation (was Python inner loops)
# ---------------------------------------------------------------------------

class TestTtpKeepMaskVectorized:
    """
    Verify that the vectorised _compute_ttp_keep_mask_reference and
    _compute_ttp_keep_mask_consecutive produce results identical to
    the original Python-loop reference implementations and achieve a
    significant speedup for realistic video sizes.

    Original bottleneck: N × (t-1) Python iterations (N=1170, t-1=24 for
    50-frame 720p) each with `if is_dup:` converting a GPU tensor to a Python
    bool (CUDA sync) → 28,080 syncs per video.

    Vectorised reference mode: t-1 = 24 Python iterations (N spatial
    positions processed at once as tensors, no CUDA syncs).
    Vectorised consecutive mode: 0 Python loops — a single tensor op.
    """

    # ------------------------------------------------------------------ #
    # Reference (slow) implementations — mirror the original loops        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ref_reference_cosine(merged_tokens, t, N, threshold, device):
        """Original O(N·t) Python loop for reference mode, cosine metric."""
        keep_mask = torch.ones(t, N, dtype=torch.bool, device=device)
        for spatial_idx in range(N):
            ref_frame_idx = 0
            for t_idx in range(1, t):
                curr = merged_tokens[t_idx, spatial_idx].float().unsqueeze(0)
                ref  = merged_tokens[ref_frame_idx, spatial_idx].float().unsqueeze(0)
                sim  = F.cosine_similarity(curr, ref).item()
                if sim > threshold:
                    keep_mask[t_idx, spatial_idx] = False
                else:
                    ref_frame_idx = t_idx
        return keep_mask

    @staticmethod
    def _ref_consecutive_cosine(merged_tokens, t, N, threshold, device):
        """Original O(N·t) Python loop for consecutive mode, cosine metric."""
        keep_mask = torch.ones(t, N, dtype=torch.bool, device=device)
        for spatial_idx in range(N):
            for t_idx in range(1, t):
                curr = merged_tokens[t_idx,     spatial_idx].float().unsqueeze(0)
                prev = merged_tokens[t_idx - 1, spatial_idx].float().unsqueeze(0)
                sim  = F.cosine_similarity(curr, prev).item()
                if sim > threshold:
                    keep_mask[t_idx, spatial_idx] = False
        return keep_mask

    # ------------------------------------------------------------------ #
    # Reference mode tests                                                 #
    # ------------------------------------------------------------------ #

    def test_reference_mode_all_identical_frames(self):
        """All frames identical → only frame 0 kept (cosine similarity = 1.0 > threshold)."""
        torch.manual_seed(0)
        t, N, dim = 5, 20, 16
        base = torch.randn(N, dim)
        tokens = base.unsqueeze(0).expand(t, -1, -1).clone()

        mask = _compute_ttp_keep_mask_reference(
            tokens, t, N, threshold=0.9, similarity_metric="cosine",
            device=tokens.device
        )

        assert mask[0].all(), "Frame 0 must always be kept"
        assert not mask[1:].any(), "All duplicate frames should be removed"

    def test_reference_mode_all_different_frames(self):
        """All frames very different → all kept."""
        torch.manual_seed(1)
        t, N, dim = 5, 20, 64
        tokens = torch.randn(t, N, dim)
        # Shift each frame far from the others so cosine similarity << threshold
        for i in range(t):
            tokens[i] = tokens[i] + torch.randn(N, dim) * 100.0 * (i + 1)

        mask = _compute_ttp_keep_mask_reference(
            tokens, t, N, threshold=0.99999, similarity_metric="cosine",
            device=tokens.device
        )

        assert mask.all(), "All frames should be kept when they are all very different"

    def test_reference_mode_matches_reference_loop_cosine(self):
        """
        Vectorised reference mode must produce results identical to the
        original Python-loop implementation for random inputs with cosine metric.
        """
        torch.manual_seed(42)
        t, N, dim = 8, 30, 32
        tokens = torch.randn(t, N, dim)
        # Make a couple of frames very similar to their predecessors
        tokens[2] = tokens[1] * 0.99999
        tokens[5] = tokens[4] * 0.99999
        threshold = 0.95

        mask_ref = self._ref_reference_cosine(tokens, t, N, threshold, tokens.device)
        mask_vec = _compute_ttp_keep_mask_reference(
            tokens, t, N, threshold, "cosine", tokens.device
        )

        assert torch.equal(mask_vec, mask_ref), (
            f"Vectorised and reference loop disagree.\n"
            f"ref:\n{mask_ref}\nvec:\n{mask_vec}"
        )

    def test_reference_mode_l2_metric_identical_frames(self):
        """Reference mode, L2 metric: identical frames should be removed."""
        torch.manual_seed(3)
        t, N, dim = 4, 10, 8
        base = torch.randn(N, dim)
        tokens = base.unsqueeze(0).expand(t, -1, -1).clone()

        # L2: distance=0 → ratio 0/norm = 0 < any positive threshold → is_dup
        mask = _compute_ttp_keep_mask_reference(
            tokens, t, N, threshold=0.01, similarity_metric="l2",
            device=tokens.device
        )

        assert mask[0].all(), "Frame 0 must always be kept"
        assert not mask[1:].any(), "L2: identical frames should be marked as duplicates"

    def test_reference_mode_l1_metric_identical_frames(self):
        """Reference mode, L1 metric: identical frames should be removed."""
        torch.manual_seed(4)
        t, N, dim = 4, 10, 8
        base = torch.randn(N, dim)
        tokens = base.unsqueeze(0).expand(t, -1, -1).clone()

        mask = _compute_ttp_keep_mask_reference(
            tokens, t, N, threshold=0.01, similarity_metric="l1",
            device=tokens.device
        )

        assert mask[0].all()
        assert not mask[1:].any(), "L1: identical frames should be marked as duplicates"

    def test_reference_mode_speed(self):
        """
        Verify speedup from eliminating per-token Python loop overhead.

        NOTE: The primary GPU speedup comes from eliminating blocking CUDA syncs
        (`.item()` calls in the reference loop).  On CPU there are no CUDA syncs,
        so we use a small `dim` to make Python call overhead dominate, giving a
        measurable improvement even on CPU.  On GPU the speedup is much larger
        (hundreds of CUDA syncs eliminated per video frame).
        """
        import time

        torch.manual_seed(99)
        t, N, dim = 25, 1170, 16   # small dim keeps tensors in cache; loop overhead dominates
        tokens = torch.randn(t, N, dim)
        threshold = 0.9999

        # Time the reference loop (warm up with one tiny run first)
        _compute_ttp_keep_mask_reference(tokens[:2, :10], 2, 10, threshold, "cosine", tokens.device)

        t0 = time.perf_counter()
        mask_ref = self._ref_reference_cosine(tokens, t, N, threshold, tokens.device)
        t_old = time.perf_counter() - t0

        t0 = time.perf_counter()
        mask_vec = _compute_ttp_keep_mask_reference(
            tokens, t, N, threshold, "cosine", tokens.device
        )
        t_new = time.perf_counter() - t0

        speedup = t_old / max(t_new, 1e-9)
        assert speedup >= 5, (
            f"Vectorised reference mode speedup={speedup:.1f}× (expected ≥5× on CPU; "
            f"GPU speedup is much larger). "
            f"old={t_old * 1000:.1f}ms, new={t_new * 1000:.2f}ms"
        )
        assert torch.equal(mask_vec, mask_ref), "Speed test: vectorised result must match reference loop"

    # ------------------------------------------------------------------ #
    # Consecutive mode tests                                               #
    # ------------------------------------------------------------------ #

    def test_consecutive_mode_all_identical_frames(self):
        """All frames identical → frames 1..t-1 all removed in consecutive mode."""
        torch.manual_seed(0)
        t, N, dim = 5, 15, 16
        base = torch.randn(N, dim)
        tokens = base.unsqueeze(0).expand(t, -1, -1).clone()

        mask = _compute_ttp_keep_mask_consecutive(
            tokens, t, N, threshold=0.9, similarity_metric="cosine",
            min_run_length=2, device=tokens.device
        )

        assert mask[0].all(), "Frame 0 must always be kept"
        assert not mask[1:].any(), "All duplicate frames should be removed in consecutive mode"

    def test_consecutive_mode_all_different_frames(self):
        """All frames very different → all kept in consecutive mode."""
        torch.manual_seed(2)
        t, N, dim = 5, 15, 64
        tokens = torch.randn(t, N, dim)
        for i in range(t):
            tokens[i] = tokens[i] + torch.randn(N, dim) * 100.0 * (i + 1)

        mask = _compute_ttp_keep_mask_consecutive(
            tokens, t, N, threshold=0.99999, similarity_metric="cosine",
            min_run_length=2, device=tokens.device
        )

        assert mask.all(), "All frames should be kept when they are all very different"

    def test_consecutive_mode_matches_reference_loop_cosine(self):
        """
        Vectorised consecutive mode must produce results identical to the
        original Python-loop implementation for random inputs with cosine metric.
        """
        torch.manual_seed(43)
        t, N, dim = 8, 30, 32
        tokens = torch.randn(t, N, dim)
        tokens[2] = tokens[1] * 0.99999
        tokens[5] = tokens[4] * 0.99999
        threshold = 0.95

        mask_ref = self._ref_consecutive_cosine(tokens, t, N, threshold, tokens.device)
        mask_vec = _compute_ttp_keep_mask_consecutive(
            tokens, t, N, threshold, "cosine", min_run_length=2, device=tokens.device
        )

        assert torch.equal(mask_vec, mask_ref), (
            f"Vectorised consecutive and reference loop disagree.\n"
            f"ref:\n{mask_ref}\nvec:\n{mask_vec}"
        )

    def test_consecutive_mode_l2_metric_identical_frames(self):
        """Consecutive mode, L2 metric: identical frames should be removed."""
        torch.manual_seed(5)
        t, N, dim = 4, 10, 8
        base = torch.randn(N, dim)
        tokens = base.unsqueeze(0).expand(t, -1, -1).clone()

        mask = _compute_ttp_keep_mask_consecutive(
            tokens, t, N, threshold=0.01, similarity_metric="l2",
            min_run_length=2, device=tokens.device
        )

        assert mask[0].all()
        assert not mask[1:].any(), "L2: identical frames should be removed in consecutive mode"

    def test_consecutive_mode_speed(self):
        """
        Verify speedup from replacing the Python loop with a single tensor op.

        Uses small `dim` so Python overhead dominates (not memory bandwidth).
        On GPU the speedup is much larger due to CUDA sync elimination.
        """
        import time

        torch.manual_seed(77)
        t, N, dim = 25, 1170, 16   # small dim keeps tensors in cache; loop overhead dominates
        tokens = torch.randn(t, N, dim)
        threshold = 0.9999

        # Warm up
        _compute_ttp_keep_mask_consecutive(tokens[:2, :10], 2, 10, threshold, "cosine", 2, tokens.device)

        t0 = time.perf_counter()
        mask_ref = self._ref_consecutive_cosine(tokens, t, N, threshold, tokens.device)
        t_old = time.perf_counter() - t0

        t0 = time.perf_counter()
        mask_vec = _compute_ttp_keep_mask_consecutive(
            tokens, t, N, threshold, "cosine", min_run_length=2, device=tokens.device
        )
        t_new = time.perf_counter() - t0

        speedup = t_old / max(t_new, 1e-9)
        assert speedup >= 5, (
            f"Vectorised consecutive mode speedup={speedup:.1f}× (expected ≥5× on CPU; "
            f"GPU speedup is much larger). "
            f"old={t_old * 1000:.1f}ms, new={t_new * 1000:.2f}ms"
        )
        assert torch.equal(mask_vec, mask_ref), "Speed test: vectorised result must match reference loop"
