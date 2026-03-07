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
#
# Temporal Token Pruning (TTP) for Video Language Models
"""
Temporal Token Pruning (TTP) — removes temporally redundant visual tokens from videos.

Core idea: consecutive video frames share many identical patches. TTP compares
adjacent (or reference-based) frames, detects "runs" of similar patches, and
keeps only the first occurrence per run. This reduces the visual token count by
50-90% for typical screen-recording videos with minimal information loss.

Usage modes:
  1. Standalone  — TTP-only temporal reduction (patch_qwen3vl_forward_with_ttp)
  2. STP + TTP   — spatial reduction first, then temporal reduction (combined masks)

Supported architectures: Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE
"""

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import ModelArguments

logger = logging.get_logger(__name__)


def compute_run_length_keep_mask(
    pixel_values: "torch.Tensor",
    grid_thw: "torch.Tensor",
    threshold: float = 0.1,
    min_run_length: int = 2,
    similarity_metric: str = "cosine",
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    channel: int = 3,
    use_raw_frames_in_ttp: bool = False,
    comparison_mode: str = "reference",
) -> "torch.Tensor":
    """
    Compute a per-token keep mask via Temporal Token Pruning.

    Two comparison modes are available:
    - ``reference``:   compare each frame against the last *kept* frame (more aggressive).
    - ``consecutive``: compare only adjacent frames — true run-length encoding.

    Args:
        pixel_values:      Flattened patch tensor, shape ``(num_patches, patch_dim)``.
        grid_thw:          Per-image/video grid dims, shape ``(N, 3)`` with ``(t, h, w)``.
        threshold:         Similarity threshold (cosine: ``>`` = duplicate; L2/L1: ``<`` = duplicate).
        min_run_length:    Minimum consecutive duplicates to trigger removal (``consecutive`` mode only).
        similarity_metric: One of ``cosine``, ``l2``, ``l1``.
        patch_size:        ViT patch size (e.g. 14).
        temporal_patch_size: Frames per temporal patch (e.g. 2).
        merge_size:        Spatial merge factor (e.g. 2 → 2×2 patches merge into 1 token).
        channel:           Image channels.
        use_raw_frames_in_ttp: Compare at raw-frame level and OR the keep decisions.
        comparison_mode:   ``reference`` (default) or ``consecutive``.

    Returns:
        Boolean mask ``(num_merged_tokens,)`` — ``True`` = keep.
    """
    device = pixel_values.device
    dtype = pixel_values.dtype
    grid_thw_np = grid_thw.cpu().numpy()

    keep_list = []
    patch_offset = 0

    for img_idx in range(len(grid_thw_np)):
        t, h, w = grid_thw_np[img_idx]

        # For single-frame images, keep all tokens
        if t <= 1:
            num_merged_tokens = (h // merge_size) * (w // merge_size)
            keep_list.append(torch.ones(num_merged_tokens, dtype=torch.bool, device=device))
            patch_offset += t * h * w
            continue

        num_patches = t * h * w

        # Get patches for this image/video
        img_patches = pixel_values[patch_offset : patch_offset + num_patches]

        # Output grid dimensions (after merge)
        out_h = h // merge_size
        out_w = w // merge_size
        num_merged_per_frame = out_h * out_w

        # Reshape to (t, out_h, out_w, merge_size, merge_size, patch_dim)
        # The processor outputs patches in merge-block order
        patches_by_merge = img_patches.reshape(
            t, out_h, out_w, merge_size, merge_size, -1
        )

        if use_raw_frames_in_ttp and temporal_patch_size > 1:
            # Raw frame level comparison: unpack temporal_patch_size dimension
            # patch_dim = channel * temporal_patch_size * patch_size * patch_size
            # Reshape to (t, out_h, out_w, merge_size, merge_size, channel, temporal_patch_size, patch_size, patch_size)
            patches_with_temporal = patches_by_merge.view(
                t, out_h, out_w, merge_size, merge_size,
                channel, temporal_patch_size, patch_size, patch_size
            )

            # For each raw frame within temporal_patch_size, compute TTP mask separately
            per_tp_keep_masks = []
            for tp_idx in range(temporal_patch_size):
                # Extract this raw frame: (t, out_h, out_w, merge_size, merge_size, channel, patch_size, patch_size)
                frame_patches = patches_with_temporal[:, :, :, :, :, :, tp_idx, :, :]
                # Flatten spatial patch dimensions: (t, out_h, out_w, merge_size, merge_size, raw_patch_dim)
                # where raw_patch_dim = channel * patch_size * patch_size
                frame_patches = frame_patches.reshape(t, out_h, out_w, merge_size, merge_size, -1)

                # Compute merged token representations by averaging sub-patches
                # Shape: (t, num_merged_per_frame, raw_patch_dim)
                merged_tokens = frame_patches.reshape(t, out_h * out_w, merge_size * merge_size, -1).mean(dim=2)

                # Compute keep mask for this raw frame
                tp_keep_mask = _compute_ttp_keep_mask_for_tokens(
                    merged_tokens, t, num_merged_per_frame, threshold, similarity_metric, device,
                    comparison_mode=comparison_mode, min_run_length=min_run_length
                )
                per_tp_keep_masks.append(tp_keep_mask)

            # OR the masks: keep if ANY raw frame differs (conservative approach)
            # Stack: (temporal_patch_size, t, num_merged_per_frame)
            stacked_masks = torch.stack(per_tp_keep_masks, dim=0)
            # If any raw frame says "keep", we keep the token
            keep_mask = stacked_masks.any(dim=0)  # (t, num_merged_per_frame)
        else:
            # Original behavior: compare at merged temporal step level
            # Compute merged token representations by averaging sub-patches
            # Shape: (t, out_h * out_w, patch_dim)
            merged_tokens = patches_by_merge.reshape(t, out_h * out_w, merge_size * merge_size, -1).mean(dim=2)

            # Compute keep mask
            keep_mask = _compute_ttp_keep_mask_for_tokens(
                merged_tokens, t, num_merged_per_frame, threshold, similarity_metric, device,
                comparison_mode=comparison_mode, min_run_length=min_run_length
            )

        # Flatten to (t * num_merged_per_frame,)
        keep_list.append(keep_mask.flatten())
        patch_offset += num_patches

    return torch.cat(keep_list, dim=0)


def _compute_ttp_keep_mask_reference(
    merged_tokens: "torch.Tensor",
    t: int,
    num_merged_per_frame: int,
    threshold: float,
    similarity_metric: str,
    device: "torch.device",
) -> "torch.Tensor":
    """
    Reference-based TTP keep mask (vectorised over spatial positions).

    Each frame is compared against the last *kept* frame (reference).
    More aggressive than consecutive mode: can skip frames similar to
    a distant reference.  Loops only over ``t-1`` temporal steps with
    zero CUDA syncs (all N spatial positions processed in one tensor op).

    Returns:
        Boolean mask ``(t, num_merged_per_frame)`` — frame 0 is always kept.
    """
    # keep_mask[t, i] = True if token i at frame t should be kept
    # Frame 0 is always kept.
    keep_mask = torch.ones(t, num_merged_per_frame, dtype=torch.bool, device=device)

    if similarity_metric == "cosine":
        # Pre-normalise all frames once: (t, N, dim)
        tokens_norm = F.normalize(merged_tokens.float(), p=2, dim=-1)
        # Reference starts at frame 0 for every spatial position: (N, dim)
        ref_norm = tokens_norm[0].clone()

        for t_idx in range(1, t):
            curr_norm = tokens_norm[t_idx]               # (N, dim)
            sims = (ref_norm * curr_norm).sum(dim=-1)    # (N,) — all positions at once
            is_dup = sims > threshold                    # (N,) — GPU tensor, no CUDA sync
            keep_mask[t_idx] = ~is_dup
            # Advance reference only for positions whose frame was *kept*
            ref_norm = torch.where(is_dup.unsqueeze(-1), ref_norm, curr_norm)

    elif similarity_metric == "l2":
        ref_token = merged_tokens[0].float().clone()     # (N, dim)

        for t_idx in range(1, t):
            curr_token = merged_tokens[t_idx].float()
            diff_norm = (ref_token - curr_token).norm(p=2, dim=-1)   # (N,)
            ref_norms = ref_token.norm(p=2, dim=-1)                  # (N,)
            is_dup = diff_norm / (ref_norms + 1e-8) < threshold      # (N,)
            keep_mask[t_idx] = ~is_dup
            ref_token = torch.where(is_dup.unsqueeze(-1), ref_token, curr_token)

    elif similarity_metric == "l1":
        ref_token = merged_tokens[0].float().clone()     # (N, dim)

        for t_idx in range(1, t):
            curr_token = merged_tokens[t_idx].float()
            diff = (ref_token - curr_token).abs().sum(dim=-1)        # (N,)
            ref_norms = ref_token.abs().sum(dim=-1)                  # (N,)
            is_dup = diff / (ref_norms + 1e-8) < threshold           # (N,)
            keep_mask[t_idx] = ~is_dup
            ref_token = torch.where(is_dup.unsqueeze(-1), ref_token, curr_token)

    else:
        raise ValueError(f"Unknown similarity metric: {similarity_metric}")

    return keep_mask


def _compute_ttp_keep_mask_consecutive(
    merged_tokens: "torch.Tensor",
    t: int,
    num_merged_per_frame: int,
    threshold: float,
    similarity_metric: str,
    min_run_length: int,
    device: "torch.device",
) -> "torch.Tensor":
    """
    Consecutive-comparison TTP keep mask (true run-length encoding).

    Only adjacent frames are compared — less aggressive but more faithful
    to the RLE concept.  Fully vectorised: all ``(t-1) × N`` similarities
    are computed in a single tensor operation.

    Returns:
        Boolean mask ``(t, num_merged_per_frame)``.
    """
    keep_mask = torch.ones(t, num_merged_per_frame, dtype=torch.bool, device=device)

    if similarity_metric == "cosine":
        # All consecutive similarities in one shot: (t-1, N)
        tokens_norm = F.normalize(merged_tokens.float(), p=2, dim=-1)
        sims = (tokens_norm[1:] * tokens_norm[:-1]).sum(dim=-1)
        keep_mask[1:] = sims <= threshold

    elif similarity_metric == "l2":
        tokens_f = merged_tokens.float()
        diff = (tokens_f[1:] - tokens_f[:-1]).norm(p=2, dim=-1)          # (t-1, N)
        ref_norms = tokens_f[:-1].norm(p=2, dim=-1)                       # (t-1, N)
        keep_mask[1:] = diff / (ref_norms + 1e-8) >= threshold

    elif similarity_metric == "l1":
        tokens_f = merged_tokens.float()
        diff = (tokens_f[1:] - tokens_f[:-1]).abs().sum(dim=-1)           # (t-1, N)
        ref_norms = tokens_f[:-1].abs().sum(dim=-1)                        # (t-1, N)
        keep_mask[1:] = diff / (ref_norms + 1e-8) >= threshold

    else:
        raise ValueError(f"Unknown similarity metric: {similarity_metric}")

    # Apply min_run_length constraint if needed
    if min_run_length > 2:
        keep_mask = _apply_min_run_length(keep_mask, min_run_length)

    return keep_mask


def _compute_ttp_keep_mask_for_tokens(
    merged_tokens: "torch.Tensor",
    t: int,
    num_merged_per_frame: int,
    threshold: float,
    similarity_metric: str,
    device: "torch.device",
    comparison_mode: str = "reference",
    min_run_length: int = 2,
) -> "torch.Tensor":
    """Dispatch to reference or consecutive TTP comparison."""
    if comparison_mode == "consecutive":
        return _compute_ttp_keep_mask_consecutive(
            merged_tokens, t, num_merged_per_frame, threshold,
            similarity_metric, min_run_length, device
        )
    else:  # reference mode (default)
        return _compute_ttp_keep_mask_reference(
            merged_tokens, t, num_merged_per_frame, threshold,
            similarity_metric, device
        )


def _apply_min_run_length(
    keep_mask: "torch.Tensor",
    min_run_length: int,
) -> "torch.Tensor":
    """Restore tokens in runs shorter than *min_run_length* (consecutive mode only)."""
    t, num_tokens = keep_mask.shape
    result = keep_mask.clone()

    for token_idx in range(num_tokens):
        run_start = 0
        run_length = 1

        for t_idx in range(1, t):
            if not keep_mask[t_idx, token_idx]:  # This was marked as duplicate
                run_length += 1
            else:
                # End of run - check if we should restore some tokens
                if run_length < min_run_length:
                    # Run too short, restore all tokens in this run
                    for restore_t in range(run_start + 1, t_idx):
                        result[restore_t, token_idx] = True
                run_start = t_idx
                run_length = 1

        # Handle last run
        if run_length < min_run_length:
            for restore_t in range(run_start + 1, t):
                result[restore_t, token_idx] = True

    return result


def compute_run_length_keep_mask_from_embeddings(
    embeddings: "torch.Tensor",
    grid_thw: "torch.Tensor",
    threshold: float = 0.9,
    min_run_length: int = 2,
    similarity_metric: str = "cosine",
    merge_size: int = 2,
) -> "torch.Tensor":
    """
    Compute TTP keep mask from post-encoder vision embeddings.

    Alternative to pixel-based ``compute_run_length_keep_mask`` when
    embeddings are already materialised (e.g. embedding-level TTP).
    """
    device = embeddings.device
    grid_thw_np = grid_thw.cpu().numpy()

    keep_list = []
    token_offset = 0

    for img_idx in range(len(grid_thw_np)):
        t, h, w = grid_thw_np[img_idx]
        out_h = h // merge_size
        out_w = w // merge_size
        num_tokens_per_frame = out_h * out_w
        total_tokens = t * num_tokens_per_frame

        if t <= 1:
            keep_list.append(torch.ones(total_tokens, dtype=torch.bool, device=device))
            token_offset += total_tokens
            continue

        # Get embeddings for this image/video
        img_embeds = embeddings[token_offset : token_offset + total_tokens]
        img_embeds = img_embeds.reshape(t, num_tokens_per_frame, -1)

        # Compare consecutive frames
        keep_mask = torch.ones(t, num_tokens_per_frame, dtype=torch.bool, device=device)

        for t_idx in range(1, t):
            prev_embeds = img_embeds[t_idx - 1]
            curr_embeds = img_embeds[t_idx]

            if similarity_metric == "cosine":
                prev_norm = F.normalize(prev_embeds.float(), p=2, dim=-1)
                curr_norm = F.normalize(curr_embeds.float(), p=2, dim=-1)
                similarity = (prev_norm * curr_norm).sum(dim=-1)
                is_duplicate = similarity > threshold
            elif similarity_metric == "l2":
                diff = (prev_embeds.float() - curr_embeds.float()).norm(p=2, dim=-1)
                diff_normalized = diff / (prev_embeds.float().norm(p=2, dim=-1) + 1e-8)
                is_duplicate = diff_normalized < threshold
            else:  # l1
                diff = (prev_embeds.float() - curr_embeds.float()).abs().sum(dim=-1)
                diff_normalized = diff / (prev_embeds.float().abs().sum(dim=-1) + 1e-8)
                is_duplicate = diff_normalized < threshold

            keep_mask[t_idx] = ~is_duplicate

        if min_run_length > 2:
            keep_mask = _apply_min_run_length(keep_mask, min_run_length)

        keep_list.append(keep_mask.flatten())
        token_offset += total_tokens

    return torch.cat(keep_list, dim=0)


def combine_stp_and_ttp_masks(
    stp_mask: Optional["torch.Tensor"],
    ttp_mask: Optional["torch.Tensor"],
) -> Optional["torch.Tensor"]:
    """AND-combine STP (spatial) and TTP (temporal) keep masks.

    A token survives only if both masks say keep.  Returns ``None`` if
    both inputs are ``None``.
    """
    if stp_mask is None and ttp_mask is None:
        return None
    if stp_mask is None:
        return ttp_mask
    if ttp_mask is None:
        return stp_mask

    # Both masks exist - combine with AND
    # Ensure same device and length
    if stp_mask.shape != ttp_mask.shape:
        logger.warning_rank0(
            f"STP mask shape {stp_mask.shape} != TTP mask shape {ttp_mask.shape}. "
            "Using STP mask only."
        )
        return stp_mask

    return stp_mask & ttp_mask


def compute_ttp_keep_mask_after_stp(
    pixel_values: "torch.Tensor",
    grid_thw: "torch.Tensor",
    stp_keep_mask: "torch.Tensor",
    threshold: float = 0.9,
    min_run_length: int = 2,
    similarity_metric: str = "cosine",
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    channel: int = 3,
) -> "torch.Tensor":
    """
    Compute TTP keep mask restricted to STP-surviving tokens.

    For each spatial position, only frames kept by STP are compared
    temporally. Returns the AND-combined mask (STP ∩ TTP).
    """
    device = pixel_values.device
    grid_thw_np = grid_thw.cpu().numpy()

    # Start with STP mask
    combined_mask = stp_keep_mask.clone()

    patch_offset = 0
    token_offset = 0

    for img_idx in range(len(grid_thw_np)):
        t, h, w = grid_thw_np[img_idx]
        num_patches = t * h * w
        out_h = h // merge_size
        out_w = w // merge_size
        num_merged_per_frame = out_h * out_w
        total_merged = t * num_merged_per_frame

        if t <= 1:
            patch_offset += num_patches
            token_offset += total_merged
            continue

        # Get patches for this image/video
        img_patches = pixel_values[patch_offset : patch_offset + num_patches]

        # Reshape to get merged token representations
        patches_by_merge = img_patches.reshape(t, out_h, out_w, merge_size, merge_size, -1)
        merged_tokens = patches_by_merge.reshape(t, num_merged_per_frame, merge_size * merge_size, -1).mean(dim=2)

        # Get STP mask for this image
        img_stp_mask = stp_keep_mask[token_offset : token_offset + total_merged]
        img_stp_mask = img_stp_mask.reshape(t, num_merged_per_frame)

        # For each spatial position, compare across time only for STP-kept tokens
        for spatial_idx in range(num_merged_per_frame):
            # Find which frames have this token kept by STP
            kept_frames = img_stp_mask[:, spatial_idx].nonzero(as_tuple=True)[0]

            if len(kept_frames) <= 1:
                continue  # Nothing to compare

            # Compare consecutive kept frames
            for i in range(1, len(kept_frames)):
                prev_frame = kept_frames[i - 1].item()
                curr_frame = kept_frames[i].item()

                prev_token = merged_tokens[prev_frame, spatial_idx]
                curr_token = merged_tokens[curr_frame, spatial_idx]

                if similarity_metric == "cosine":
                    prev_norm = F.normalize(prev_token.float().unsqueeze(0), p=2, dim=-1)
                    curr_norm = F.normalize(curr_token.float().unsqueeze(0), p=2, dim=-1)
                    similarity = (prev_norm * curr_norm).sum()
                    is_duplicate = similarity > threshold
                elif similarity_metric == "l2":
                    diff = (prev_token.float() - curr_token.float()).norm(p=2)
                    diff_normalized = diff / (prev_token.float().norm(p=2) + 1e-8)
                    is_duplicate = diff_normalized < threshold
                else:  # l1
                    diff = (prev_token.float() - curr_token.float()).abs().sum()
                    diff_normalized = diff / (prev_token.float().abs().sum() + 1e-8)
                    is_duplicate = diff_normalized < threshold

                if is_duplicate:
                    # Mark current frame's token for removal
                    global_idx = token_offset + curr_frame * num_merged_per_frame + spatial_idx
                    combined_mask[global_idx] = False

        patch_offset += num_patches
        token_offset += total_merged

    return combined_mask


def patch_qwen2vl_forward_with_ttp(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
) -> None:
    """Monkey-patch Qwen2VL/Qwen2.5VL inner forward to apply TTP (+ optional STP)."""
    if not getattr(model_args, "use_ttp", False):
        return

    threshold = getattr(model_args, "ttp_threshold", 0.9)
    min_run_length = getattr(model_args, "ttp_min_run_length", 2)
    similarity_metric = getattr(model_args, "ttp_similarity_metric", "cosine")
    use_raw_frames_in_ttp = getattr(model_args, "use_raw_frames_in_ttp", False)
    comparison_mode = getattr(model_args, "ttp_comparison_mode", "reference")

    model_type = getattr(model.config, "model_type", None)
    if model_type not in ["qwen2_vl", "qwen2_5_vl"]:
        logger.warning_rank0(
            f"TTP patch is only supported for Qwen2VL/Qwen2.5VL, got: {model_type}"
        )
        return

    # Check if STP is also enabled
    use_stp = getattr(model_args, "use_stp", False)
    stp_mode = getattr(model_args, "stp_mode", "preprocess")

    # Get the model's internal model
    if hasattr(model, "model"):
        inner_model = model.model
    else:
        inner_model = model

    # Get vision config
    vision_config = getattr(inner_model.config, "vision_config", None)
    spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)
    patch_size = getattr(vision_config, "patch_size", 14)
    temporal_patch_size = getattr(vision_config, "temporal_patch_size", 2)

    # Store original forward
    original_forward = inner_model.forward

    # Import STP functions if needed
    if use_stp and stp_mode == "forward_removal":
        from .stp import compute_token_keep_mask_from_pixels

    def patched_forward(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        **kwargs,
    ):
        # For video inputs, compute TTP mask
        video_ttp_mask = None
        if pixel_values_videos is not None and video_grid_thw is not None:
            video_ttp_mask = compute_run_length_keep_mask(
                pixel_values_videos,
                video_grid_thw,
                threshold=threshold,
                min_run_length=min_run_length,
                similarity_metric=similarity_metric,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                merge_size=spatial_merge_size,
                use_raw_frames_in_ttp=use_raw_frames_in_ttp,
                comparison_mode=comparison_mode,
            )

            # If STP is also enabled, combine masks
            if use_stp and stp_mode == "forward_removal":
                stp_threshold = getattr(model_args, "stp_threshold", 0.0)
                if stp_threshold > 0:
                    stp_mask = compute_token_keep_mask_from_pixels(
                        pixel_values_videos,
                        video_grid_thw,
                        threshold=stp_threshold,
                        skip_ratio=getattr(model_args, "stp_skip_ratio", 0.5),
                        large_comp_threshold=getattr(model_args, "stp_large_comp_threshold", 0),
                        patch_size=patch_size,
                        temporal_patch_size=temporal_patch_size,
                        merge_size=spatial_merge_size,
                        patch_level=getattr(model_args, "stp_patch_level", False),
                        patch_to_token_strategy=getattr(model_args, "stp_patch_to_token_strategy", "any"),
                        temporal_aggregation=getattr(model_args, "stp_temporal_aggregation", "first"),
                        use_raw_frames_in_stp=getattr(model_args, "use_raw_frames_in_stp", False),
                    )
                    video_ttp_mask = combine_stp_and_ttp_masks(stp_mask, video_ttp_mask)

        # Store TTP mask for use in forward
        kwargs["_ttp_video_keep_mask"] = video_ttp_mask

        return original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            **kwargs,
        )

    inner_model.forward = patched_forward

    logger.info_rank0(
        f"TTP patch applied (Qwen2VL): threshold={threshold}, "
        f"min_run_length={min_run_length}, metric={similarity_metric}, "
        f"use_raw_frames={use_raw_frames_in_ttp}, mode={comparison_mode}"
    )


def patch_qwen3vl_forward_with_ttp(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
) -> None:
    """
    Monkey-patch Qwen3VL model forward to apply TTP (and optionally STP).

    Replaces both ``inner_model.forward`` (patched_forward — handles ViT
    call, mask computation, token removal, and LLM forward) and
    ``model.forward`` (patched_outer_forward — handles label shortening
    and loss computation after token removal).

    When ``debug_token_removal: true`` in config YAML, detailed logs are
    written to ``/tmp/ttp_forward_debug.log``.
    """
    if not getattr(model_args, "use_ttp", False):
        return

    threshold = getattr(model_args, "ttp_threshold", 0.9)
    min_run_length = getattr(model_args, "ttp_min_run_length", 2)
    similarity_metric = getattr(model_args, "ttp_similarity_metric", "cosine")
    use_raw_frames_in_ttp = getattr(model_args, "use_raw_frames_in_ttp", False)
    comparison_mode = getattr(model_args, "ttp_comparison_mode", "reference")

    model_type = getattr(model.config, "model_type", None)
    if model_type not in ["qwen3_vl", "qwen3_vl_moe"]:
        logger.warning_rank0(
            f"TTP patch is only supported for Qwen3VL/Qwen3VL-MoE, got: {model_type}"
        )
        return

    # Check if STP is also enabled
    use_stp = getattr(model_args, "use_stp", False)
    stp_mode = getattr(model_args, "stp_mode", "preprocess")

    # Get the model's internal model
    if hasattr(model, "model"):
        inner_model = model.model
    else:
        inner_model = model

    # Get vision config - prefer values from vision module over config
    vision_config = getattr(inner_model.config, "vision_config", None)
    vision_module = getattr(inner_model, "visual", None)
    vision_module_config = getattr(vision_module, "config", None)

    spatial_merge_size = getattr(
        vision_module,
        "spatial_merge_size",
        getattr(vision_module_config, "spatial_merge_size", getattr(vision_config, "spatial_merge_size", 2)),
    )
    patch_size = getattr(vision_module_config, "patch_size", getattr(vision_config, "patch_size", 14))
    temporal_patch_size = getattr(
        vision_module_config, "temporal_patch_size", getattr(vision_config, "temporal_patch_size", 2)
    )

    # Store original forward
    original_forward = inner_model.forward

    # When STP+TTP are co-enabled, patcher.py skips the STP forward patch
    # (to avoid conflicting forward replacements).  We still need the STP
    # vision-pruning helper for keep-mask-based ViT speedup.
    if use_stp and stp_mode == "forward_removal":
        from .stp import compute_token_keep_mask_from_pixels
        from .stp import patch_stp_qwen3vl_vision_encoder_with_pruning
        stp_threshold = getattr(model_args, "stp_threshold", 0.0)
        if stp_threshold > 0:
            patch_stp_qwen3vl_vision_encoder_with_pruning(model, model_args)

    # Debug flag captured as closure variable
    _debug_token_removal = getattr(model_args, "debug_token_removal", False)

    def patched_forward(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        **kwargs,
    ):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModelOutputWithPast

        import os
        import time as _time_mod
        _local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # Debug logging (only when debug_token_removal: true in config)
        if _debug_token_removal and _local_rank == 0:
            with open("/tmp/ttp_forward_debug.log", "a") as f:
                f.write(f"[TTP Forward] Start, video_grid_thw={video_grid_thw}\n")

        # Handle defaults
        output_attentions = kwargs.get("output_attentions", inner_model.config.output_attentions)
        output_hidden_states = kwargs.get("output_hidden_states", inner_model.config.output_hidden_states)
        return_dict = kwargs.get("return_dict", inner_model.config.use_return_dict)
        use_cache = kwargs.get("use_cache", None)
        if use_cache is None:
            use_cache = False if inner_model.training else getattr(inner_model.language_model.config, "use_cache", True)

        # Determine effective STP/TTP settings for this batch
        # Default to global settings from closure
        global_use_ttp = True  # TTP patch is applied, so default is True
        global_use_stp = use_stp and stp_mode == "forward_removal"

        # Per-video STP/TTP overrides (from collator or stashed on model
        # by trainer.prediction_step before generate()).
        _outer = getattr(inner_model, "_ttp_outer_model_ref", None)
        per_video_use_stp = kwargs.pop(
            "_per_video_use_stp",
            getattr(inner_model, "_ttp_per_video_use_stp",
                    getattr(_outer, "_ttp_per_video_use_stp", None)),
        )
        per_video_use_ttp = kwargs.pop(
            "_per_video_use_ttp",
            getattr(inner_model, "_ttp_per_video_use_ttp",
                    getattr(_outer, "_ttp_per_video_use_ttp", None)),
        )

        # Store settings on model for debugging
        inner_model._ttp_global_use_ttp = global_use_ttp
        inner_model._ttp_global_use_stp = global_use_stp
        inner_model._ttp_per_video_use_stp = per_video_use_stp
        inner_model._ttp_per_video_use_ttp = per_video_use_ttp

        # Step 1: Get text embeddings
        if inputs_embeds is None:
            inputs_embeds = inner_model.get_input_embeddings()(input_ids)


        # Clear last-run masks (for debugging / scripts) - use explicit deletion to release memory
        # Only clear at the start of a new sequence (not during generation steps)
        is_first_forward = cache_position is None or cache_position[0] == 0
        if is_first_forward:
            if hasattr(inner_model, "_ttp_video_keep_mask") and inner_model._ttp_video_keep_mask is not None:
                del inner_model._ttp_video_keep_mask
            if hasattr(inner_model, "_ttp_video_keep_mask_raw") and inner_model._ttp_video_keep_mask_raw is not None:
                del inner_model._ttp_video_keep_mask_raw
            if hasattr(inner_model, "_stp_video_keep_mask") and inner_model._stp_video_keep_mask is not None:
                del inner_model._stp_video_keep_mask
            inner_model._ttp_video_keep_mask = None
            inner_model._ttp_video_keep_mask_raw = None
            inner_model._stp_video_keep_mask = None
            inner_model._ttp_num_removed = 0

            # Only clear rope_deltas when position_ids is None (Step 4 will
            # recompute).  When position_ids is provided (e.g. from
            # prepare_inputs_for_generation), keep existing rope_deltas.
            if position_ids is None:
                if hasattr(inner_model, "rope_deltas") and inner_model.rope_deltas is not None:
                    inner_model.rope_deltas = None

        video_token_mask = None
        video_keep_mask = None
        video_keep_mask_raw = None

        # Precomputed TTP+STP mask from data-loader (avoids on-the-fly computation).
        _precomputed_video_mask = kwargs.pop("video_ttp_keep_mask", None)

        # Step 2: Process videos — compute STP/TTP keep mask and call ViT.
        # All GPUs always compute a mask (all-True when disabled) for ZeRO-3 compat.
        deepstack_video_embeds = None
        if pixel_values_videos is not None:
            num_videos = video_grid_thw.shape[0]
            merge_unit = spatial_merge_size * spatial_merge_size

            # Debug log
            if _debug_token_removal and _local_rank == 0:
                with open("/tmp/ttp_forward_debug.log", "a") as f:
                    f.write(f"[TTP Forward] Step 2: Processing {num_videos} videos\n")
                    f.write(f"[TTP Forward] per_video_use_stp={per_video_use_stp}\n")
                    f.write(f"[TTP Forward] per_video_use_ttp={per_video_use_ttp}\n")
                    f.write(f"[TTP Forward] precomputed_mask={'yes' if _precomputed_video_mask is not None else 'no'}\n")

            # Per-video overrides invalidate the precomputed mask.
            _has_per_video_override = (
                (per_video_use_ttp is not None and any(v is not None for v in per_video_use_ttp))
                or (per_video_use_stp is not None and any(v is not None for v in per_video_use_stp))
            )

            if _precomputed_video_mask is not None and not _has_per_video_override:
                # Fast path: reuse precomputed mask (just move to device).
                video_keep_mask = _precomputed_video_mask.to(pixel_values_videos.device)
                if _debug_token_removal:
                    video_keep_mask_raw = video_keep_mask.clone().detach()
                    inner_model._ttp_video_keep_mask_raw = video_keep_mask_raw
                    inner_model._ttp_video_keep_mask = video_keep_mask.detach()
            else:
                # Slow path: compute mask on the fly (no precomputed or has overrides).
                # Per-video token counts for mask construction
                tokens_per_video = (torch.prod(video_grid_thw, dim=1) // merge_unit).tolist()

                # Full TTP mask (disabled videos overridden with all-True below)
                full_ttp_mask = compute_run_length_keep_mask(
                    pixel_values_videos,
                    video_grid_thw,
                    threshold=threshold,
                    min_run_length=min_run_length,
                    similarity_metric=similarity_metric,
                    patch_size=patch_size,
                    temporal_patch_size=temporal_patch_size,
                    merge_size=spatial_merge_size,
                    use_raw_frames_in_ttp=use_raw_frames_in_ttp,
                    comparison_mode=comparison_mode,
                )
                # Only clone for debug; otherwise reuse directly
                video_keep_mask_raw = full_ttp_mask.clone().detach() if _debug_token_removal else full_ttp_mask

                # Full STP mask (if globally enabled)
                full_stp_mask = None
                stp_threshold = getattr(model_args, "stp_threshold", 0.0)
                if global_use_stp and stp_threshold > 0:
                    full_stp_mask = compute_token_keep_mask_from_pixels(
                        pixel_values_videos,
                        video_grid_thw,
                        threshold=stp_threshold,
                        skip_ratio=getattr(model_args, "stp_skip_ratio", 0.5),
                        large_comp_threshold=getattr(model_args, "stp_large_comp_threshold", 0),
                        patch_size=patch_size,
                        temporal_patch_size=temporal_patch_size,
                        merge_size=spatial_merge_size,
                        patch_level=getattr(model_args, "stp_patch_level", False),
                        patch_to_token_strategy=getattr(model_args, "stp_patch_to_token_strategy", "any"),
                        temporal_aggregation=getattr(model_args, "stp_temporal_aggregation", "first"),
                        use_raw_frames_in_stp=getattr(model_args, "use_raw_frames_in_stp", False),
                    )
                    if _debug_token_removal:
                        inner_model._stp_video_keep_mask = full_stp_mask.detach()

                # Build per-video final mask: actual mask or all-True for disabled videos
                video_keep_mask_parts = []
                token_offset = 0
                for vid_idx in range(num_videos):
                    num_tokens = int(tokens_per_video[vid_idx])

                    # None = use global setting, True/False overrides
                    vid_use_ttp = per_video_use_ttp[vid_idx] if per_video_use_ttp else None
                    vid_use_stp = per_video_use_stp[vid_idx] if per_video_use_stp else None

                    effective_vid_use_ttp = vid_use_ttp if vid_use_ttp is not None else global_use_ttp
                    effective_vid_use_stp = vid_use_stp if vid_use_stp is not None else global_use_stp

                    # Mask segments for this video
                    ttp_segment = full_ttp_mask[token_offset:token_offset + num_tokens]
                    stp_segment = full_stp_mask[token_offset:token_offset + num_tokens] if full_stp_mask is not None else None

                    if effective_vid_use_ttp and effective_vid_use_stp and stp_segment is not None:
                        vid_mask = combine_stp_and_ttp_masks(stp_segment, ttp_segment)
                    elif effective_vid_use_ttp:
                        vid_mask = ttp_segment
                    elif effective_vid_use_stp and stp_segment is not None:
                        vid_mask = stp_segment
                    else:
                        vid_mask = torch.ones(num_tokens, dtype=torch.bool, device=pixel_values_videos.device)

                    video_keep_mask_parts.append(vid_mask)
                    token_offset += num_tokens

                    if _debug_token_removal and _local_rank == 0:
                        with open("/tmp/ttp_forward_debug.log", "a") as f:
                            f.write(f"[TTP Forward] Video {vid_idx}: use_ttp={effective_vid_use_ttp}, use_stp={effective_vid_use_stp}, kept={vid_mask.sum()}/{num_tokens}\n")

                video_keep_mask = torch.cat(video_keep_mask_parts, dim=0)

                # Store masks for debug scripts only
                if _debug_token_removal:
                    inner_model._ttp_video_keep_mask_raw = video_keep_mask_raw.detach() if video_keep_mask_raw is not None else None
                    inner_model._ttp_video_keep_mask = video_keep_mask.detach() if video_keep_mask is not None else None

                del video_keep_mask_parts

            # Always route through pruned ViT path — required for DeepSpeed ZeRO-3
            # which needs identical collective call graphs on every GPU.  When the
            # mask is all-True the overhead is negligible (one nonzero + scatter).
            # --- Diagnostic: mask stats + ViT timing (rank 0) ---
            if _local_rank == 0:
                _mask_total = video_keep_mask.numel()
                _mask_kept = int(video_keep_mask.sum())
                _mask_ratio = _mask_kept / _mask_total if _mask_total > 0 else 1.0
                print(f"[TTP] ViT call: kept={_mask_kept}/{_mask_total} ({_mask_ratio:.2%}), "
                      f"pixel_values_videos.shape={pixel_values_videos.shape}, "
                      f"video_grid_thw={video_grid_thw.tolist()}")

            _vit_t0 = _time_mod.time()

            from .stp import _stp_qwen3vl_visual_forward_pruned
            # Match training dtype: patched_get_video_features converts to visual.dtype
            _pv = pixel_values_videos.type(inner_model.visual.dtype)
            _vit_out, _deepstack_out = _stp_qwen3vl_visual_forward_pruned(
                inner_model.visual,
                _pv,
                video_grid_thw,
                video_keep_mask,
                _debug_token_removal,
            )
            # Split output into per-video tensors (same contract as get_video_features).
            _split_sizes = (video_grid_thw.prod(-1) // merge_unit).tolist()
            video_result = (torch.split(_vit_out, _split_sizes), _deepstack_out)

            if _local_rank == 0:
                torch.cuda.synchronize()
                _vit_elapsed = _time_mod.time() - _vit_t0
                print(f"[TTP] ViT done in {_vit_elapsed:.3f}s")
            if isinstance(video_result, tuple) and len(video_result) == 2:
                video_embeds_tuple, deepstack_video_embeds = video_result
                if isinstance(video_embeds_tuple, (tuple, list)):
                    video_embeds_cat = torch.cat(video_embeds_tuple, dim=0)
                else:
                    video_embeds_cat = video_embeds_tuple
            elif hasattr(video_result, "pooler_output"):
                # BaseModelOutputWithDeepstackFeatures (newer transformers)
                deepstack_video_embeds = getattr(video_result, "deepstack_features", None)
                video_embeds_tuple = video_result.pooler_output
                if isinstance(video_embeds_tuple, (tuple, list)):
                    video_embeds_cat = torch.cat(video_embeds_tuple, dim=0)
                else:
                    video_embeds_cat = video_embeds_tuple
            else:
                video_embeds_cat = video_result

            video_embeds_cat = video_embeds_cat.to(inputs_embeds.device, inputs_embeds.dtype)

            _, video_mask = inner_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds_cat
            )
            video_token_mask = video_mask[..., 0] if video_mask.dim() > 2 else video_mask.squeeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds_cat)

        # Step 3: Process images (no TTP applied to single-frame images)
        image_token_mask = None
        deepstack_image_embeds = None
        if pixel_values is not None:
            image_result = inner_model.get_image_features(pixel_values, image_grid_thw)
            if isinstance(image_result, tuple) and len(image_result) == 2:
                image_embeds_tuple, deepstack_image_embeds = image_result
                if isinstance(image_embeds_tuple, (tuple, list)):
                    image_embeds_cat = torch.cat(image_embeds_tuple, dim=0)
                else:
                    image_embeds_cat = image_embeds_tuple
            elif hasattr(image_result, "pooler_output"):
                # BaseModelOutputWithDeepstackFeatures (newer transformers)
                deepstack_image_embeds = getattr(image_result, "deepstack_features", None)
                image_embeds_tuple = image_result.pooler_output
                if isinstance(image_embeds_tuple, (tuple, list)):
                    image_embeds_cat = torch.cat(image_embeds_tuple, dim=0)
                else:
                    image_embeds_cat = image_embeds_tuple
            else:
                image_embeds_cat = image_result

            image_embeds_cat = image_embeds_cat.to(inputs_embeds.device, inputs_embeds.dtype)

            image_mask, _ = inner_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds_cat
            )
            image_token_mask = image_mask[..., 0]
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_cat)

        # Build visual_pos_masks
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_token_mask is not None and video_token_mask is not None:
            visual_pos_masks = image_token_mask | video_token_mask
        elif image_token_mask is not None:
            visual_pos_masks = image_token_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_token_mask is not None:
            visual_pos_masks = video_token_mask
            deepstack_visual_embeds = deepstack_video_embeds

        # Step 4: Compute position IDs
        if position_ids is None:
            if inner_model.rope_deltas is None or cache_position is None or cache_position[0] == 0:
                position_ids, rope_deltas = inner_model.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                inner_model.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + inner_model.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids + delta.to(position_ids.device)

        # Step 5: Remove pruned video tokens from the sequence.
        # GenerationMixin may have expanded attention_mask to 4D; reset to
        # None so the LLM rebuilds a causal mask from the (now shorter) 2D mask.
        if attention_mask is not None and attention_mask.ndim != 2:
            if _local_rank == 0:
                print(f"[TTP] WARNING: attention_mask is {attention_mask.ndim}D (expected 2D), setting to None")
            attention_mask = None

        _is_prefill = pixel_values_videos is not None  # only time during prefill
        if _local_rank == 0 and _is_prefill:
            _step5_t0 = _time_mod.time()
        should_remove_video = video_token_mask is not None and video_keep_mask is not None and not video_keep_mask.all()

        if should_remove_video:
            batch_size, seq_len = inputs_embeds.shape[:2]
            seq_keep_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=inputs_embeds.device)

            merge_size = spatial_merge_size
            tokens_per_sample = (video_grid_thw.prod(dim=-1) // (merge_size * merge_size)).tolist()
            video_offsets = [0]
            for t in tokens_per_sample[:-1]:
                video_offsets.append(video_offsets[-1] + t)

            for b in range(batch_size):
                vid_positions = video_token_mask[b].nonzero(as_tuple=True)[0]
                offset = video_offsets[b] if b < len(video_offsets) else 0
                n = min(len(vid_positions), max(0, len(video_keep_mask) - offset))
                if n > 0:
                    seq_keep_mask[b, vid_positions[:n]] = video_keep_mask[offset:offset + n]

            # Diagnostic: verify text tokens are NOT removed
            if _local_rank == 0:
                for b in range(batch_size):
                    _vid_mask_b = video_token_mask[b]  # True=video position
                    _text_mask_b = ~_vid_mask_b  # True=text position
                    _text_removed = int((_text_mask_b & ~seq_keep_mask[b]).sum())
                    _vid_total = int(_vid_mask_b.sum())
                    _vid_removed = int((_vid_mask_b & ~seq_keep_mask[b]).sum())
                    _vid_offset = video_offsets[b] if b < len(video_offsets) else 0
                    _vid_n = min(int(_vid_mask_b.sum()), max(0, len(video_keep_mask) - _vid_offset))
                    print(f"[TTP Step5] sample {b}: text_removed={_text_removed}, "
                          f"vid_total={_vid_total}, vid_removed={_vid_removed}, "
                          f"vid_positions={len(video_token_mask[b].nonzero(as_tuple=True)[0])}, "
                          f"mask_offset={_vid_offset}, mask_n={_vid_n}, "
                          f"tokens_per_sample={tokens_per_sample}")

            # Remove tokens
            new_inputs_embeds_list = []
            new_position_ids_list = []
            new_attention_mask_list = []

            for b in range(batch_size):
                keep_indices = seq_keep_mask[b].nonzero(as_tuple=True)[0]
                # Ensure keep_indices is on the same device as the tensors being indexed
                keep_indices_embeds = keep_indices.to(inputs_embeds.device)
                keep_indices_pos = keep_indices.to(position_ids.device)
                new_inputs_embeds_list.append(inputs_embeds[b, keep_indices_embeds])
                new_position_ids_list.append(position_ids[:, b, keep_indices_pos])
                if attention_mask is not None:
                    keep_indices_mask = keep_indices.to(attention_mask.device)
                    new_attention_mask_list.append(attention_mask[b, keep_indices_mask])

            max_len = max(e.shape[0] for e in new_inputs_embeds_list)
            padded_embeds = []
            padded_positions = []
            padded_masks = []

            for b in range(batch_size):
                emb = new_inputs_embeds_list[b]
                pos = new_position_ids_list[b]
                cur_len = emb.shape[0]
                pad_len = max_len - cur_len

                if pad_len > 0:
                    # LEFT-pad so valid tokens are contiguous at the end,
                    # which is required by Flash Attention with batch_size>1.
                    emb = torch.nn.functional.pad(emb, (0, 0, pad_len, 0))
                    pos = torch.nn.functional.pad(pos, (pad_len, 0))
                    if attention_mask is not None:
                        mask = new_attention_mask_list[b]
                        mask = torch.nn.functional.pad(mask, (pad_len, 0), value=0)
                        padded_masks.append(mask)
                else:
                    if attention_mask is not None:
                        padded_masks.append(new_attention_mask_list[b])

                padded_embeds.append(emb)
                padded_positions.append(pos)

            inputs_embeds = torch.stack(padded_embeds, dim=0)
            position_ids = torch.stack(padded_positions, dim=1)
            if attention_mask is not None:
                attention_mask = torch.stack(padded_masks, dim=0)

            del new_inputs_embeds_list, new_position_ids_list, new_attention_mask_list
            del padded_embeds, padded_positions, padded_masks

            if cache_position is not None:
                cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)

            # Do NOT correct rope_deltas: cache_position keeps original length L
            # (see detailed explanation in the second rope_deltas note below).

            if _local_rank == 0:
                _n_removed = seq_len - inputs_embeds.shape[1]
                print(f"[TTP prefill] Step5: seq_len {seq_len} → {inputs_embeds.shape[1]} "
                      f"(removed {_n_removed}), "
                      f"rope_deltas={inner_model.rope_deltas.flatten().tolist() if inner_model.rope_deltas is not None else None}, "
                      f"pos_ids range=[{position_ids.min().item()}, {position_ids.max().item()}], "
                      f"attn_mask sum={attention_mask.sum(dim=-1).tolist()}")

            # Update visual_pos_masks
            if visual_pos_masks is not None:
                new_visual_pos_masks_list = []
                for b in range(batch_size):
                    keep_indices = seq_keep_mask[b].nonzero(as_tuple=True)[0]
                    keep_indices = keep_indices.to(visual_pos_masks.device)
                    new_visual_pos_masks_list.append(visual_pos_masks[b, keep_indices])

                padded_visual_masks = []
                for b in range(batch_size):
                    vm = new_visual_pos_masks_list[b]
                    cur_len = vm.shape[0]
                    pad_len = max_len - cur_len
                    if pad_len > 0:
                        vm = torch.nn.functional.pad(vm, (0, pad_len), value=False)
                    padded_visual_masks.append(vm)
                visual_pos_masks = torch.stack(padded_visual_masks, dim=0)

                del new_visual_pos_masks_list, padded_visual_masks

            # Filter deepstack_visual_embeds to match kept tokens
            if deepstack_visual_embeds is not None and video_keep_mask is not None:
                if isinstance(deepstack_visual_embeds, (list, tuple)):
                    deepstack_visual_embeds = [
                        emb[video_keep_mask.to(emb.device)] for emb in deepstack_visual_embeds
                    ]
                else:
                    deepstack_visual_embeds = deepstack_visual_embeds[
                        video_keep_mask.to(deepstack_visual_embeds.device)
                    ]

            # Store for label shortening in patched_outer_forward
            inner_model._ttp_seq_keep_mask = seq_keep_mask.detach()
            inner_model._ttp_max_len = max_len

            inner_model._ttp_step5_attn_mask = attention_mask.detach()
            inner_model._ttp_num_removed = seq_len - inputs_embeds.shape[1]

            # IMPORTANT: Do NOT correct rope_deltas after token removal.
            #
            # model_kwargs["cache_position"] retains the ORIGINAL sequence
            # length L (not the pruned length), because
            # _update_model_kwargs_for_generation operates on the un-shortened
            # model_kwargs.  During decode step k, position is:
            #
            #   pos = cache_position[0] + rope_deltas
            #       = (L + k) + (mrope_max + 1 - L)
            #       = mrope_max + 1 + k              ← already correct
            #
            # Adding num_removed would double-count the removal.

        # Fix 4D→3D position_ids: prepare_inputs_for_generation may build
        # [text, t, h, w] but M-RoPE expects [t, h, w]. Drop dim 0.
        if position_ids is not None and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]  # [4, B, L] → [3, B, L]

        # Step 6: Call language model
        if _local_rank == 0 and _is_prefill:
            torch.cuda.synchronize()
            _step5_elapsed = _time_mod.time() - _step5_t0
            _emb_shape = inputs_embeds.shape
            print(f"[TTP] Step5 (token removal): {_step5_elapsed:.3f}s, "
                  f"should_remove={should_remove_video}, inputs_embeds.shape={_emb_shape}")
            _llm_t0 = _time_mod.time()

        outputs = inner_model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if _local_rank == 0 and _is_prefill:
            torch.cuda.synchronize()
            _llm_elapsed = _time_mod.time() - _llm_t0
            print(f"[TTP] Step6 (LLM forward): {_llm_elapsed:.3f}s")

        output = Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=inner_model.rope_deltas,
        )
        return output if return_dict else output.to_tuple()

    inner_model.forward = patched_forward
    # Use __dict__ to avoid nn.Module.__setattr__ registering `model` as a
    # child of inner_model (would create model↔inner_model circular ref
    # causing RecursionError in model.train()/eval()).
    inner_model.__dict__["_ttp_outer_model_ref"] = model

    # Patch outer model forward for label shortening after TTP token removal.
    original_outer_forward = model.forward

    def patched_outer_forward(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        **kwargs,
    ):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast

        # Clear previous masks (explicit deletion to release GPU memory)
        for attr_name in ["_ttp_seq_keep_mask", "_ttp_max_len", "_ttp_video_keep_mask",
                          "_ttp_video_keep_mask_raw", "_stp_video_keep_mask"]:
            if hasattr(inner_model, attr_name):
                old_val = getattr(inner_model, attr_name, None)
                if old_val is not None:
                    delattr(inner_model, attr_name)
        inner_model._ttp_seq_keep_mask = None
        inner_model._ttp_max_len = None

        # Clear model-level stale labels
        if hasattr(model, "_ttp_updated_labels"):
            old_labels = getattr(model, "_ttp_updated_labels", None)
            if old_labels is not None:
                del model._ttp_updated_labels

        outputs = model.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Shorten labels to match reduced sequence length after TTP removal
        if labels is not None and getattr(inner_model, "_ttp_seq_keep_mask", None) is not None:
            seq_keep_mask = inner_model._ttp_seq_keep_mask
            max_len = inner_model._ttp_max_len
            batch_size = labels.shape[0]
            labels_seq_len = labels.shape[1]
            mask_seq_len = seq_keep_mask.shape[1]

            if labels_seq_len != mask_seq_len:
                if labels_seq_len < mask_seq_len:
                    seq_keep_mask = seq_keep_mask[:, :labels_seq_len]
                else:
                    pad_len = labels_seq_len - mask_seq_len
                    seq_keep_mask = torch.nn.functional.pad(seq_keep_mask, (0, pad_len), value=True)

            new_labels_list = []
            for b in range(batch_size):
                keep_indices = seq_keep_mask[b].nonzero(as_tuple=True)[0]
                new_labels_list.append(labels[b, keep_indices])

            padded_labels = []
            for b in range(batch_size):
                lab = new_labels_list[b]
                cur_len = lab.shape[0]
                pad_len = max_len - cur_len
                if pad_len > 0:
                    lab = torch.nn.functional.pad(lab, (0, pad_len), value=-100)
                padded_labels.append(lab)
            labels = torch.stack(padded_labels, dim=0)

            del new_labels_list, padded_labels

            # Store shortened labels for trainer access (detached)
            model._ttp_updated_labels = labels.detach().clone()

            # Clear stale mask references
            inner_model._ttp_seq_keep_mask = None
            inner_model._ttp_max_len = None

        logits_to_keep = kwargs.pop("logits_to_keep", 0)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = model.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = model.loss_function(logits=logits, labels=labels, vocab_size=model.config.text_config.vocab_size)

        return_dict = kwargs.get("return_dict", getattr(model.config, "use_return_dict", True))
        output = Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )
        return output if return_dict else output.to_tuple()

    if not getattr(model, "_ttp_qwen3vl_outer_forward_patched", False):
        model._ttp_qwen3vl_outer_forward_patched = True
        model.forward = patched_outer_forward

    logger.info_rank0(
        f"TTP patch applied (Qwen3VL): threshold={threshold}, "
        f"min_run_length={min_run_length}, metric={similarity_metric}, "
        f"use_raw_frames={use_raw_frames_in_ttp}, mode={comparison_mode}"
    )


def apply_ttp_forward_patch(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
) -> None:
    """Main entry point: detect model type and apply the appropriate TTP patch."""
    if not getattr(model_args, "use_ttp", False):
        return

    model_type = getattr(model.config, "model_type", None)

    if model_type in ["qwen2_vl", "qwen2_5_vl"]:
        patch_qwen2vl_forward_with_ttp(model, model_args)
    elif model_type in ["qwen3_vl", "qwen3_vl_moe"]:
        patch_qwen3vl_forward_with_ttp(model, model_args)
    else:
        logger.warning_rank0(
            f"TTP forward patch is not supported for model type: {model_type}. "
            f"Supported types: qwen2_vl, qwen2_5_vl, qwen3_vl, qwen3_vl_moe"
        )
