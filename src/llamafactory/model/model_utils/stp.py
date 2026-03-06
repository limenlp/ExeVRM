from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import ModelArguments

logger = logging.get_logger(__name__)

# ============================================================================
# PERFORMANCE OPTIMIZATION: Caching and GPU-accelerated computation
# ============================================================================

# Global cache for keep masks (keyed by pixel_values hash)
_keep_mask_cache: dict[int, torch.Tensor] = {}
_CACHE_MAX_SIZE = 32  # Maximum number of cached masks


def _get_pixel_hash(pixel_values: torch.Tensor) -> int:
    """Compute a hash for pixel values for caching purposes."""
    # Use a fast hash based on shape, first/last values, and sum
    # This is not collision-proof but is fast for caching
    shape_hash = hash(pixel_values.shape)
    if pixel_values.numel() > 0:
        # Sample a few values for hash
        flat = pixel_values.view(-1)
        sample_indices = torch.tensor([0, len(flat)//4, len(flat)//2, 3*len(flat)//4, -1], device=flat.device)
        sample_indices = sample_indices.clamp(0, len(flat)-1)
        samples = flat[sample_indices].float().cpu().numpy()
        data_hash = hash(tuple(samples.tolist()) + (float(flat.sum().item()),))
    else:
        data_hash = 0
    return hash((shape_hash, data_hash))


def _get_keep_mask_cache_key(
    pixel_values: torch.Tensor,
    *,
    threshold: float,
    skip_ratio: float,
    large_comp_threshold: int,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
    channel: int,
    patch_level: bool,
    patch_to_token_strategy: str,
    temporal_aggregation: str,
    use_raw_frames_in_stp: bool,
) -> int:
    """Compute a cache key for keep mask computation.

    IMPORTANT: the keep mask depends not only on pixel_values but also on
    configuration knobs (threshold/skip_ratio/etc.).
    """
    # Pixel hash dominates the key; include config to prevent incorrect reuse.
    pv_hash = _get_pixel_hash(pixel_values)
    cfg = (
        float(threshold),
        float(round(skip_ratio, 8)),
        int(large_comp_threshold),
        int(patch_size),
        int(temporal_patch_size),
        int(merge_size),
        int(channel),
        bool(patch_level),
        str(patch_to_token_strategy),
        str(temporal_aggregation),
        bool(use_raw_frames_in_stp),
    )
    return hash((pv_hash, cfg))


def _cache_keep_mask(key: int, mask: torch.Tensor) -> None:
    """Cache a keep mask with LRU eviction."""
    global _keep_mask_cache
    if len(_keep_mask_cache) >= _CACHE_MAX_SIZE:
        # Remove oldest entry (simple FIFO, not true LRU)
        oldest_key = next(iter(_keep_mask_cache))
        del _keep_mask_cache[oldest_key]
    _keep_mask_cache[key] = mask.detach()


def _get_cached_mask(key: int) -> Optional[torch.Tensor]:
    """Get cached keep mask if exists."""
    return _keep_mask_cache.get(key)


def clear_stp_cache() -> None:
    """Clear the STP keep mask cache."""
    global _keep_mask_cache
    _keep_mask_cache.clear()


# ============================================================================
# GPU-ACCELERATED Union-Find using PyTorch
# ============================================================================

def _gpu_union_find(
    grid_h: int,
    grid_w: int,
    diffs_h: torch.Tensor,  # (grid_h-1, grid_w) vertical diffs
    diffs_w: torch.Tensor,  # (grid_h, grid_w-1) horizontal diffs
    threshold: float,
) -> torch.Tensor:
    """
    GPU-accelerated Union-Find using iterative label propagation.

    Uses a simple but robust approach: repeatedly propagate minimum labels
    to connected neighbors until convergence.

    The number of iterations is fixed to (grid_h + grid_w), which is the
    theoretical worst-case diameter for bidirectional propagation on a 2-D grid.
    This avoids torch.equal() which forces a CPU-GPU sync on every iteration.

    Returns:
        Component labels with shape (grid_h * grid_w,)
    """
    device = diffs_h.device if diffs_h.numel() > 0 else diffs_w.device
    num_patches = grid_h * grid_w

    # Initialize each patch with its own label
    labels = torch.arange(num_patches, device=device, dtype=torch.int64).view(grid_h, grid_w)

    # Create adjacency masks based on threshold
    connect_h = diffs_h < threshold if diffs_h.numel() > 0 else torch.zeros((0, grid_w), dtype=torch.bool, device=device)
    connect_w = diffs_w < threshold if diffs_w.numel() > 0 else torch.zeros((grid_h, 0), dtype=torch.bool, device=device)

    # Fixed iteration count: grid_h + grid_w covers the worst-case propagation
    # distance on a 2-D grid with bidirectional sweeps, with no CUDA sync needed.
    num_iterations = grid_h + grid_w
    for _ in range(num_iterations):
        # Forward pass: propagate from top-left to bottom-right
        # Propagate down
        if grid_h > 1 and connect_h.numel() > 0:
            propagated = torch.minimum(labels[:-1], labels[1:])
            labels[:-1] = torch.where(connect_h, propagated, labels[:-1])
            labels[1:] = torch.where(connect_h, propagated, labels[1:])

        # Propagate right
        if grid_w > 1 and connect_w.numel() > 0:
            propagated = torch.minimum(labels[:, :-1], labels[:, 1:])
            labels[:, :-1] = torch.where(connect_w, propagated, labels[:, :-1])
            labels[:, 1:] = torch.where(connect_w, propagated, labels[:, 1:])

        # Backward pass: propagate from bottom-right to top-left
        # Propagate up
        if grid_h > 1 and connect_h.numel() > 0:
            propagated = torch.minimum(labels[:-1], labels[1:])
            labels[1:] = torch.where(connect_h, propagated, labels[1:])
            labels[:-1] = torch.where(connect_h, propagated, labels[:-1])

        # Propagate left
        if grid_w > 1 and connect_w.numel() > 0:
            propagated = torch.minimum(labels[:, :-1], labels[:, 1:])
            labels[:, 1:] = torch.where(connect_w, propagated, labels[:, 1:])
            labels[:, :-1] = torch.where(connect_w, propagated, labels[:, :-1])

    return labels.view(-1)


def _get_select_mask_gpu(
    labels: torch.Tensor,  # (num_patches,) component labels
    skip_ratio: float,
    large_comp_threshold: int,
) -> torch.Tensor:
    """
    GPU-accelerated selection mask computation.

    This implementation matches the CPU version (_get_select_mask_stp) exactly:
    - Uses uniform sampling (linspace) to select tokens within each component
    - Single-patch components are always kept
    - Large components (> large_comp_threshold) are entirely skipped

    Args:
        labels: Component labels for each patch
        skip_ratio: Ratio of patches to skip within each component
        large_comp_threshold: Components larger than this are entirely skipped

    Returns:
        Boolean mask where True = keep
    """
    device = labels.device
    num_patches = labels.shape[0]

    # Get unique labels and counts
    unique_labels, inverse, counts = torch.unique(labels, return_inverse=True, return_counts=True)

    # Map each patch to its component size
    comp_sizes = counts[inverse]  # (num_patches,)

    # Initialize keep mask to False (same as CPU version)
    keep_mask = torch.zeros(num_patches, dtype=torch.bool, device=device)

    # Sort by label to group patches by component (stable sort to match CPU behavior)
    sorted_indices = torch.argsort(labels, stable=True)
    sorted_labels = labels[sorted_indices]
    sorted_comp_sizes = comp_sizes[sorted_indices]

    # Compute position within each component (0-indexed)
    # When labels change, reset position to 0
    label_changes = torch.cat([
        torch.ones(1, dtype=torch.bool, device=device),
        sorted_labels[1:] != sorted_labels[:-1]
    ])

    # Cumulative count within each component
    ones = torch.ones(num_patches, dtype=torch.long, device=device)
    cumsum = torch.cumsum(ones, dim=0)

    # Get the cumsum value at each label change point
    change_cumsum = cumsum * label_changes.long()
    # Forward fill the change points
    change_cumsum_filled = torch.cummax(change_cumsum, dim=0)[0]
    # Position within component = cumsum - change_cumsum_filled
    positions_in_comp = cumsum - change_cumsum_filled

    # Compute num_to_keep for each patch based on its component size
    # num_to_skip = round(comp_size * skip_ratio)
    # num_to_keep = max(1, comp_size - num_to_skip)
    num_to_skip = torch.round(sorted_comp_sizes.float() * skip_ratio).long()
    num_to_keep = torch.clamp(sorted_comp_sizes - num_to_skip, min=1)

    # Handle edge cases
    single_patch = sorted_comp_sizes == 1
    keep_all = num_to_keep >= sorted_comp_sizes
    keep_first_only = num_to_keep == 1

    # For uniform sampling with linspace, we keep positions at:
    # indices = linspace(0, comp_size-1, num_to_keep).astype(int)
    # Position i is kept if there exists integer k in [0, num_to_keep) such that:
    #   floor(k * (comp_size-1) / (num_to_keep-1)) == i
    #
    # This is equivalent to checking if there's an integer k in [low, high) where:
    #   low = i * (num_to_keep-1) / (comp_size-1)
    #   high = (i+1) * (num_to_keep-1) / (comp_size-1)
    #
    # Which happens when: ceil(low) < high AND ceil(low) < num_to_keep

    # Avoid division by zero
    comp_size_m1 = torch.where(sorted_comp_sizes > 1, sorted_comp_sizes - 1, torch.ones_like(sorted_comp_sizes))
    num_to_keep_m1 = torch.where(num_to_keep > 1, num_to_keep - 1, torch.ones_like(num_to_keep))

    step_inv = num_to_keep_m1.float() / comp_size_m1.float()

    low = positions_in_comp.float() * step_inv
    high = (positions_in_comp.float() + 1) * step_inv

    k_low = torch.ceil(low).long()

    # Position is kept if k_low < num_to_keep AND k_low < high
    is_linspace_position = (k_low < num_to_keep) & (k_low.float() < high)

    # Combine conditions
    sorted_keep = is_linspace_position | single_patch | keep_all | (keep_first_only & (positions_in_comp == 0))

    # Handle large component skip
    if large_comp_threshold > 0:
        sorted_keep = sorted_keep & (sorted_comp_sizes <= large_comp_threshold)

    # Unsort to get back to original order
    unsort_indices = torch.argsort(sorted_indices)
    keep_mask = sorted_keep[unsort_indices]

    return keep_mask


def _compute_keep_mask_gpu_single_frame(
    patches: torch.Tensor,  # (grid_h, grid_w, patch_dim)
    grid_h: int,
    grid_w: int,
    threshold: float,
    skip_ratio: float,
    large_comp_threshold: int,
) -> torch.Tensor:
    """
    Compute keep mask for a single frame using GPU-accelerated operations.

    Args:
        patches: Flattened patch tensor with shape (grid_h, grid_w, patch_dim)
        grid_h: Height of the grid
        grid_w: Width of the grid
        threshold: Threshold for merging similar patches
        skip_ratio: Ratio of patches to skip
        large_comp_threshold: Skip components larger than this

    Returns:
        Boolean mask with shape (grid_h * grid_w,) where True = keep
    """
    device = patches.device

    # Compute pairwise differences on GPU
    # Horizontal differences: (i,j) vs (i,j+1)
    if grid_w > 1:
        diffs_w = torch.norm(patches[:, :-1] - patches[:, 1:], dim=-1)  # (grid_h, grid_w-1)
    else:
        diffs_w = torch.empty((grid_h, 0), device=device)

    # Vertical differences: (i,j) vs (i+1,j)
    if grid_h > 1:
        diffs_h = torch.norm(patches[:-1, :] - patches[1:, :], dim=-1)  # (grid_h-1, grid_w)
    else:
        diffs_h = torch.empty((0, grid_w), device=device)

    # Run GPU Union-Find
    labels = _gpu_union_find(grid_h, grid_w, diffs_h, diffs_w, threshold)

    # Compute selection mask
    keep_mask = _get_select_mask_gpu(labels, skip_ratio, large_comp_threshold)

    return keep_mask


def compute_token_keep_mask_from_pixels_gpu(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    threshold: float = 1.0,
    skip_ratio: float = 0.5,
    large_comp_threshold: int = 0,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    channel: int = 3,
    temporal_aggregation: str = "first",
    use_raw_frames_in_stp: bool = False,
) -> torch.Tensor:
    """
    GPU-accelerated version of compute_token_keep_mask_from_pixels.

    This function performs all computations on GPU without CPU transfers,
    providing significant speedup for inference.

    Args:
        pixel_values: Tensor with shape (total_patches, patch_dim)
        grid_thw: Tensor with shape (num_images, 3) containing (t, h, w)
        threshold: Patch similarity threshold
        skip_ratio: Ratio of patches to skip
        large_comp_threshold: Skip components larger than this
        patch_size: Size of each patch
        temporal_patch_size: Temporal patch size
        merge_size: Spatial merge size
        channel: Number of image channels
        temporal_aggregation: How to handle temporal patches for edge detection:
            - "first": Use only the first temporal frame (recommended for video)
            - "mean": Average across temporal frames
            - "all": Use all temporal frames concatenated (original behavior)
        use_raw_frames_in_stp: If True and temporal_patch_size>1, build a UI graph per raw
            temporal frame (within each temporal patch group) and OR the resulting keep masks.
            This tends to preserve small/boundary components that appear in any raw frame.

    Returns:
        Boolean mask where True = keep, False = remove
    """
    device = pixel_values.device

    if threshold <= 0.0:
        total_tokens = (grid_thw.prod(dim=-1) // (merge_size * merge_size)).sum().item()
        return torch.ones(int(total_tokens), dtype=torch.bool, device=device)

    keep_list = []
    patch_offset = 0

    for img_idx in range(grid_thw.shape[0]):
        t, h, w = grid_thw[img_idx].tolist()
        t, h, w = int(t), int(h), int(w)
        num_patches = t * h * w

        # Get patches for this image (stay on GPU)
        img_patches = pixel_values[patch_offset : patch_offset + num_patches].float()

        # Output grid dimensions (after merge)
        out_h = h // merge_size
        out_w = w // merge_size

        # Reshape patches to merge-block order
        # (t * out_h * out_w * ms * ms, patch_dim) -> (t, out_h, out_w, ms, ms, patch_dim)
        patches_by_merge = img_patches.view(
            t, out_h, out_w, merge_size, merge_size, -1
        )

        # For each temporal slice, compute keep mask
        for t_idx in range(t):
            # Get frame patches: (out_h, out_w, ms, ms, patch_dim)
            frame_patches = patches_by_merge[t_idx]

            # Optionally do STP analysis on each raw temporal frame BEFORE temporal merge.
            if use_raw_frames_in_stp and temporal_patch_size > 1:
                frame_with_temporal = frame_patches.view(
                    out_h, out_w, merge_size, merge_size,
                    channel, temporal_patch_size, patch_size, patch_size
                )
                per_tp_masks = []
                for tp_idx in range(temporal_patch_size):
                    frame_t = frame_with_temporal[:, :, :, :, :, tp_idx, :, :]
                    frame_flat = frame_t.reshape(out_h, out_w, -1)
                    per_tp_masks.append(
                        _compute_keep_mask_gpu_single_frame(
                            frame_flat, out_h, out_w, threshold, skip_ratio, large_comp_threshold
                        )
                    )
                frame_keep_mask = torch.stack(per_tp_masks, dim=0).any(dim=0)
            else:
                # Handle temporal aggregation
                if temporal_aggregation == "all" or temporal_patch_size <= 1:
                    # Original behavior: use all temporal frames concatenated
                    frame_flat = frame_patches.view(out_h, out_w, -1)
                else:
                    # Need to reshape to access temporal dimension
                    frame_with_temporal = frame_patches.view(
                        out_h, out_w, merge_size, merge_size,
                        channel, temporal_patch_size, patch_size, patch_size
                    )
                    if temporal_aggregation == "first":
                        # Use only first temporal frame
                        frame_t0 = frame_with_temporal[:, :, :, :, :, 0, :, :]
                        frame_flat = frame_t0.reshape(out_h, out_w, -1)
                    elif temporal_aggregation == "mean":
                        # Average across temporal frames
                        frame_mean = frame_with_temporal.mean(dim=5)  # Average over temporal dim
                        frame_flat = frame_mean.reshape(out_h, out_w, -1)
                    else:
                        raise ValueError(f"Unknown temporal_aggregation: {temporal_aggregation}")

                # Compute keep mask using GPU operations
                frame_keep_mask = _compute_keep_mask_gpu_single_frame(
                    frame_flat, out_h, out_w, threshold, skip_ratio, large_comp_threshold
                )
            keep_list.append(frame_keep_mask)

        patch_offset += num_patches

    # Concatenate all masks
    return torch.cat(keep_list, dim=0)


def _map_patch_mask_to_token_mask_gpu(
    patch_mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
    merge_size: int = 2,
    strategy: str = "any",
) -> torch.Tensor:
    """
    GPU version of _map_patch_mask_to_token_mask.

    Map patch-level keep mask to merged token-level mask.

    Args:
        patch_mask: Boolean mask at patch level, shape (grid_h * grid_w,)
        grid_h: Full height of patch grid (e.g., 48)
        grid_w: Full width of patch grid (e.g., 48)
        merge_size: Merge factor (e.g., 2 means 2x2 patches -> 1 token)
        strategy: How to combine patch decisions:
            - "any": Keep token if ANY of its patches are kept
            - "all": Keep token only if ALL of its patches are kept
            - "majority": Keep token if more than half of its patches are kept

    Returns:
        Boolean mask at token level, shape (out_h * out_w,)
    """
    out_h = grid_h // merge_size
    out_w = grid_w // merge_size

    # Reshape to (out_h, merge_size, out_w, merge_size)
    patch_mask_2d = patch_mask.view(grid_h, grid_w)

    # Group by merge blocks: (out_h, out_w, merge_size, merge_size)
    patch_mask_grouped = patch_mask_2d.view(
        out_h, merge_size, out_w, merge_size
    ).permute(0, 2, 1, 3)

    # Apply strategy
    if strategy == "any":
        token_mask = patch_mask_grouped.any(dim=(2, 3))
    elif strategy == "all":
        token_mask = patch_mask_grouped.all(dim=(2, 3))
    elif strategy == "majority":
        keep_count = patch_mask_grouped.sum(dim=(2, 3))
        token_mask = keep_count > (merge_size * merge_size) // 2
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return token_mask.flatten()


def compute_token_keep_mask_from_pixels_gpu_patch_level(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    threshold: float = 1.0,
    skip_ratio: float = 0.5,
    large_comp_threshold: int = 30,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    channel: int = 3,
    patch_to_token_strategy: str = "any",
    temporal_aggregation: str = "first",
    use_raw_frames_in_stp: bool = False,
) -> torch.Tensor:
    """
    GPU-accelerated version for patch-level STP analysis.

    This function performs STP analysis at individual patch level (e.g., 48x48)
    instead of merged token level (e.g., 24x24), then maps the results to token level.

    Args:
        pixel_values: Tensor with shape (total_patches, patch_dim)
        grid_thw: Tensor with shape (num_images, 3) containing (t, h, w)
        threshold: Patch similarity threshold
        skip_ratio: Ratio of patches to skip
        large_comp_threshold: Skip components larger than this (at token level)
        patch_size: Size of each patch
        temporal_patch_size: Temporal patch size
        merge_size: Spatial merge size
        channel: Number of image channels
        patch_to_token_strategy: How to map patch decisions to tokens ("any", "all", "majority")
        temporal_aggregation: How to handle temporal patches ("first", "mean", "all")
        use_raw_frames_in_stp: If True and temporal_patch_size>1, run patch-level STP on
            each raw temporal frame (within each temporal patch group) and OR the keep masks.

    Returns:
        Boolean mask where True = keep, False = remove (at token level)
    """
    device = pixel_values.device

    if threshold <= 0.0:
        total_tokens = (grid_thw.prod(dim=-1) // (merge_size * merge_size)).sum().item()
        return torch.ones(int(total_tokens), dtype=torch.bool, device=device)

    keep_list = []
    patch_offset = 0

    # Scale large_comp_threshold for patch level (each token has merge_size^2 patches)
    patch_large_comp_threshold = large_comp_threshold * (merge_size * merge_size) if large_comp_threshold > 0 else 0

    for img_idx in range(grid_thw.shape[0]):
        t, h, w = grid_thw[img_idx].tolist()
        t, h, w = int(t), int(h), int(w)
        num_patches = t * h * w

        # Get patches for this image (stay on GPU)
        img_patches = pixel_values[patch_offset : patch_offset + num_patches].float()

        # Output grid dimensions (after merge)
        out_h = h // merge_size
        out_w = w // merge_size

        # Reshape patches to merge-block order
        # (t * out_h * out_w * ms * ms, patch_dim) -> (t, out_h, out_w, ms, ms, patch_dim)
        patches_by_merge = img_patches.view(
            t, out_h, out_w, merge_size, merge_size, -1
        )

        # For each temporal slice, compute keep mask at patch level
        for t_idx in range(t):
            # Get frame patches: (out_h, out_w, ms, ms, patch_dim)
            frame_patches = patches_by_merge[t_idx]

            # Optionally do patch-level STP on each raw temporal frame BEFORE temporal merge.
            if use_raw_frames_in_stp and temporal_patch_size > 1:
                frame_with_temporal = frame_patches.view(
                    out_h, out_w, merge_size, merge_size,
                    channel, temporal_patch_size, patch_size, patch_size
                )
                frame_with_temporal = frame_with_temporal.permute(0, 2, 1, 3, 4, 5, 6, 7).reshape(
                    h, w, channel, temporal_patch_size, patch_size, patch_size
                )
                per_tp_token_masks = []
                for tp_idx in range(temporal_patch_size):
                    frame_t = frame_with_temporal[:, :, :, tp_idx, :, :]
                    frame_individual = frame_t.reshape(h, w, -1)

                    diffs_w = torch.norm(frame_individual[:, :-1] - frame_individual[:, 1:], dim=-1)
                    diffs_h = torch.norm(frame_individual[:-1, :] - frame_individual[1:, :], dim=-1)

                    patch_labels = _gpu_union_find(h, w, diffs_h, diffs_w, threshold)
                    patch_select_mask = _get_select_mask_gpu(patch_labels, skip_ratio, patch_large_comp_threshold)
                    token_mask = _map_patch_mask_to_token_mask_gpu(
                        patch_select_mask, h, w, merge_size, strategy=patch_to_token_strategy
                    )
                    per_tp_token_masks.append(token_mask)

                token_mask = torch.stack(per_tp_token_masks, dim=0).any(dim=0)
            else:
                # Handle temporal aggregation for patch-level analysis
                if temporal_aggregation == "all" or temporal_patch_size <= 1:
                    # Original behavior: use all temporal frames
                    frame_individual = frame_patches.permute(0, 2, 1, 3, 4).reshape(h, w, -1)
                else:
                    # Reshape to access temporal dimension
                    # patch_dim = c * tp * ps * ps
                    frame_with_temporal = frame_patches.view(
                        out_h, out_w, merge_size, merge_size,
                        channel, temporal_patch_size, patch_size, patch_size
                    )
                    # Rearrange to (h, w, c, tp, ps, ps)
                    frame_with_temporal = frame_with_temporal.permute(0, 2, 1, 3, 4, 5, 6, 7).reshape(
                        h, w, channel, temporal_patch_size, patch_size, patch_size
                    )
                    if temporal_aggregation == "first":
                        # Use only first temporal frame
                        frame_t0 = frame_with_temporal[:, :, :, 0, :, :]
                        frame_individual = frame_t0.reshape(h, w, -1)
                    elif temporal_aggregation == "mean":
                        # Average across temporal frames
                        frame_mean = frame_with_temporal.mean(dim=3)
                        frame_individual = frame_mean.reshape(h, w, -1)
                    else:
                        raise ValueError(f"Unknown temporal_aggregation: {temporal_aggregation}")

                # Compute differences at patch level
                diffs_w = torch.norm(frame_individual[:, :-1] - frame_individual[:, 1:], dim=-1)
                diffs_h = torch.norm(frame_individual[:-1, :] - frame_individual[1:, :], dim=-1)

                # GPU union-find at patch level
                patch_labels = _gpu_union_find(h, w, diffs_h, diffs_w, threshold)

                # Get selection mask at patch level
                patch_select_mask = _get_select_mask_gpu(patch_labels, skip_ratio, patch_large_comp_threshold)

                # Map patch-level mask to token-level mask
                token_mask = _map_patch_mask_to_token_mask_gpu(
                    patch_select_mask, h, w, merge_size, strategy=patch_to_token_strategy
                )
            keep_list.append(token_mask)

        patch_offset += num_patches

    # Concatenate all masks
    return torch.cat(keep_list, dim=0)


class UnionFind:
    """Union-Find data structure for constructing UI patches."""

    def __init__(self, size: int):
        self.parent = np.arange(size)

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        px = self.find(x)
        py = self.find(y)
        if px != py:
            self.parent[py] = px


def build_ui_graph(
    patches: np.ndarray,
    grid_h: int,
    grid_w: int,
    threshold: float,
    adaptive: bool = True,
    max_component_ratio: float = 0.5,
) -> np.ndarray:
    """
    Build UI graph by merging similar adjacent patches using Union-Find.

    Args:
        patches: Patch array with shape (num_patches, patch_dim)
        grid_h: Height grid count
        grid_w: Width grid count
        threshold: Patch-wise difference threshold for merging.
            If threshold > 0, uses fixed threshold.
            If threshold < 0, uses adaptive threshold at the |threshold| percentile.
        adaptive: If True, automatically reduce threshold if largest component is too large
        max_component_ratio: Maximum allowed ratio for largest component (only used if adaptive=True)

    Returns:
        UI graph assignment array with shape (grid_h * grid_w,)
    """
    num_patches = grid_h * grid_w

    def idx(i: int, j: int) -> int:
        return i * grid_w + j

    # Pre-compute all pairwise differences
    diffs = []
    diff_pairs = []
    for i in range(grid_h):
        for j in range(grid_w):
            current_idx = idx(i, j)
            current_patch = patches[current_idx]

            if j + 1 < grid_w:
                right_patch = patches[idx(i, j + 1)]
                diff = np.linalg.norm(current_patch - right_patch)
                diffs.append(diff)
                diff_pairs.append((current_idx, idx(i, j + 1), diff))

            if i + 1 < grid_h:
                bottom_patch = patches[idx(i + 1, j)]
                diff = np.linalg.norm(current_patch - bottom_patch)
                diffs.append(diff)
                diff_pairs.append((current_idx, idx(i + 1, j), diff))

    diffs = np.array(diffs)

    # Determine effective threshold
    if threshold < 0:
        # Negative threshold means use percentile
        effective_threshold = np.percentile(diffs, -threshold)
    else:
        effective_threshold = threshold

    def build_with_threshold(thresh):
        uf = UnionFind(num_patches)
        for idx1, idx2, diff in diff_pairs:
            if diff < thresh:
                uf.union(idx1, idx2)
        return np.array([uf.find(x) for x in range(num_patches)])

    # Build graph with initial threshold
    uigraph_assign = build_with_threshold(effective_threshold)

    # Check if adaptive adjustment is needed
    if adaptive and max_component_ratio < 1.0:
        unique_vals, counts = np.unique(uigraph_assign, return_counts=True)
        max_component_size = counts.max()

        # If largest component is too large, reduce threshold iteratively
        attempts = 0
        while max_component_size > num_patches * max_component_ratio and attempts < 10:
            # Reduce threshold by using a lower percentile of differences
            percentile = max(5, 50 - attempts * 10)  # Start from 50th, go down
            effective_threshold = np.percentile(diffs, percentile)
            uigraph_assign = build_with_threshold(effective_threshold)
            unique_vals, counts = np.unique(uigraph_assign, return_counts=True)
            max_component_size = counts.max()
            attempts += 1

    # Rerank to consecutive integers
    unique_vals = np.unique(uigraph_assign)
    mapping = {v: i for i, v in enumerate(unique_vals)}
    uigraph_assign = np.array([mapping[v] for v in uigraph_assign])

    return uigraph_assign


def get_select_mask(
    patch_assign: np.ndarray,
    skip_ratio: float = 0.5,
    large_comp_threshold: int = 0,
    min_keep_ratio: float = 0.1,
    small_comp_skip_ratio: float = 0.0,
    large_comp_skip_ratio: float | None = None,
    grid_h: int | None = None,
    grid_w: int | None = None,
) -> np.ndarray:
    """
    Get selection mask based on patch assignment.

    Extended STP logic with large component threshold:
    - Small components (size <= large_comp_threshold): Apply small_comp_skip_ratio (default 0 = keep all)
    - Large components (size > large_comp_threshold): Apply large_comp_skip_ratio (default = skip_ratio)
    - Single-patch components are always kept
    - If large_comp_threshold=0, apply skip_ratio to all components (original STP behavior)

    For large components, we prioritize keeping boundary tokens (tokens at component edges)
    rather than uniform sampling, as boundaries contain more semantic information.

    Args:
        patch_assign: 1D array of component assignments for each patch
        skip_ratio: Default ratio of patches to skip (0.0 to 1.0), used when threshold=0
        large_comp_threshold: Components with size <= this are "small" (0 = disabled)
        min_keep_ratio: Minimum ratio of tokens to keep (safety mechanism)
        small_comp_skip_ratio: Skip ratio for small components (default 0 = keep all)
        large_comp_skip_ratio: Skip ratio for large components (default None = use skip_ratio)
        grid_h: Grid height (optional, for boundary-aware sampling)
        grid_w: Grid width (optional, for boundary-aware sampling)

    Returns:
        Boolean mask indicating which patches to keep
    """
    retain_mask = np.zeros(len(patch_assign), dtype=bool)
    unique_components = np.unique(patch_assign)
    total_patches = len(patch_assign)

    # Infer grid dimensions if not provided
    if grid_h is None or grid_w is None:
        # Try to infer square-ish grid
        grid_size = int(np.sqrt(total_patches))
        if grid_size * grid_size == total_patches:
            grid_h = grid_w = grid_size
        else:
            # Can't infer, will use uniform sampling
            grid_h = grid_w = None

    # Determine skip ratios
    if large_comp_skip_ratio is None:
        large_comp_skip_ratio = skip_ratio

    for comp in unique_components:
        positions = np.where(patch_assign == comp)[0]
        num_positions = len(positions)

        if num_positions == 1:
            # Single patch component - always keep
            retain_mask[positions] = True
        elif large_comp_threshold > 0 and num_positions <= large_comp_threshold:
            # Small component - apply small_comp_skip_ratio
            effective_skip = small_comp_skip_ratio
            num_to_skip = int(round(num_positions * effective_skip))
            num_to_retain = max(1, num_positions - num_to_skip)
            indices = np.linspace(0, num_positions - 1, num_to_retain).astype(int)
            positions_to_retain = positions[indices]
            retain_mask[positions_to_retain] = True
        else:
            # Large component - use boundary-aware sampling
            effective_skip = large_comp_skip_ratio if large_comp_threshold > 0 else skip_ratio
            num_to_skip = int(round(num_positions * effective_skip))
            num_to_retain = max(1, num_positions - num_to_skip)

            if grid_h is not None and grid_w is not None:
                # Use boundary-aware sampling for large components
                positions_to_retain = _boundary_aware_sampling(
                    positions, patch_assign, grid_h, grid_w, num_to_retain
                )
            else:
                # Fallback to uniform sampling
                indices = np.linspace(0, num_positions - 1, num_to_retain).astype(int)
                positions_to_retain = positions[indices]

            retain_mask[positions_to_retain] = True

    # Safety: ensure at least min_keep_ratio of tokens are kept
    if retain_mask.sum() < max(1, int(total_patches * min_keep_ratio)):
        num_to_keep = max(1, int(total_patches * (1.0 - skip_ratio)))
        indices = np.linspace(0, total_patches - 1, num_to_keep).astype(int)
        retain_mask = np.zeros(total_patches, dtype=bool)
        retain_mask[indices] = True

    return retain_mask


def _boundary_aware_sampling(
    positions: np.ndarray,
    patch_assign: np.ndarray,
    grid_h: int,
    grid_w: int,
    num_to_retain: int,
) -> np.ndarray:
    """
    Sample tokens from a component, prioritizing boundary tokens.

    Boundary tokens are those adjacent to tokens from different components,
    as they contain more semantic information (edges, transitions).

    Args:
        positions: Array of position indices belonging to this component
        patch_assign: Full assignment array
        grid_h: Grid height
        grid_w: Grid width
        num_to_retain: Number of tokens to retain

    Returns:
        Array of position indices to retain
    """
    if len(positions) <= num_to_retain:
        return positions

    # Create a set for fast lookup
    position_set = set(positions)
    comp_id = patch_assign[positions[0]]

    # Identify boundary tokens (adjacent to different component)
    boundary_positions = []
    interior_positions = []

    for pos in positions:
        row = pos // grid_w
        col = pos % grid_w
        is_boundary = False

        # Check 4-connected neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < grid_h and 0 <= nc < grid_w:
                neighbor_pos = nr * grid_w + nc
                if neighbor_pos not in position_set:
                    # Adjacent to a different component
                    is_boundary = True
                    break
            else:
                # Edge of the grid is also a boundary
                is_boundary = True
                break

        if is_boundary:
            boundary_positions.append(pos)
        else:
            interior_positions.append(pos)

    boundary_positions = np.array(boundary_positions)
    interior_positions = np.array(interior_positions)

    # Strategy: Keep all boundary tokens, then sample from interior if needed
    if len(boundary_positions) >= num_to_retain:
        # Too many boundary tokens, sample uniformly from boundaries
        indices = np.linspace(0, len(boundary_positions) - 1, num_to_retain).astype(int)
        return boundary_positions[indices]
    else:
        # Keep all boundaries, sample from interior for the rest
        result = list(boundary_positions)
        remaining = num_to_retain - len(boundary_positions)

        if remaining > 0 and len(interior_positions) > 0:
            if len(interior_positions) <= remaining:
                result.extend(interior_positions)
            else:
                # Sample uniformly from interior
                indices = np.linspace(0, len(interior_positions) - 1, remaining).astype(int)
                result.extend(interior_positions[indices])

        return np.array(result)


def apply_stp_token_selection(
    pixel_values: "torch.Tensor",
    image_grid_thw: "torch.Tensor",
    threshold: float = 50.0,
    skip_ratio: float = 0.5,
    merge_size: int = 2,
    large_comp_threshold: int = 0,
) -> tuple["torch.Tensor", "torch.Tensor", int, list[np.ndarray]]:
    """
    Apply STP-style UI-guided visual token selection.

    This implements the core STP idea: use UI graph to identify similar regions,
    then selectively keep representative tokens from each component.

    Args:
        pixel_values: Tensor with shape (total_patches, patch_dim)
        image_grid_thw: Tensor with shape (num_images, 3) containing (t, h, w) for each image
        threshold: Patch similarity threshold for UI graph construction
        skip_ratio: Ratio of patches to skip within each component (0.0 to 1.0)
        merge_size: Merge size used by the vision encoder (default 2)
        large_comp_threshold: Components larger than this are fully skipped (0 = disabled)

    Returns:
        Tuple of:
        - selected_pixel_values: Tensor with selected patches
        - original_grid_thw: Original grid dimensions (unchanged for reference)
        - num_reduced: Number of tokens reduced
        - select_masks: List of selection masks for each image (for visualization)
    """
    if threshold <= 0.0:
        return pixel_values, image_grid_thw, 0, []

    device = pixel_values.device
    dtype = pixel_values.dtype
    pixel_values_np = pixel_values.cpu().numpy()
    grid_thw_np = image_grid_thw.cpu().numpy()

    selected_patches_list = []
    select_masks = []
    total_original = 0
    total_selected = 0

    patch_offset = 0
    for img_idx in range(len(grid_thw_np)):
        t, h, w = grid_thw_np[img_idx]
        num_patches = t * h * w
        total_original += num_patches
        img_patches = pixel_values_np[patch_offset : patch_offset + num_patches]

        # Reshape patches to grid format: (t, h, w, patch_dim)
        patches_reshaped = img_patches.reshape(t, h, w, -1)

        # Process each temporal slice
        frame_selected_patches = []
        frame_masks = []

        for t_idx in range(t):
            frame_patches = patches_reshaped[t_idx]  # (h, w, patch_dim)
            frame_flat = frame_patches.reshape(h * w, -1)

            # Build UI graph on the full resolution patches
            patch_assign = build_ui_graph(
                patches=frame_flat,
                grid_h=h,
                grid_w=w,
                threshold=threshold,
            )

            # Get selection mask using UI-guided selection with boundary-aware sampling
            select_mask = get_select_mask(
                patch_assign, skip_ratio=skip_ratio, large_comp_threshold=large_comp_threshold,
                grid_h=h, grid_w=w
            )
            frame_masks.append(select_mask)

            # Select only the retained patches
            selected = frame_flat[select_mask]
            frame_selected_patches.append(selected)
            total_selected += len(selected)

        # Concatenate selected patches for this image
        selected_patches_list.append(np.concatenate(frame_selected_patches, axis=0))
        select_masks.append(np.stack(frame_masks, axis=0))  # (t, h*w)

        patch_offset += num_patches

    # Concatenate all selected patches
    selected_pixel_values = torch.from_numpy(np.concatenate(selected_patches_list, axis=0)).to(
        device=device, dtype=dtype
    )

    num_reduced = total_original - total_selected

    return selected_pixel_values, image_grid_thw, num_reduced, select_masks


def apply_stp_token_selection_with_positions(
    pixel_values: "torch.Tensor",
    image_grid_thw: "torch.Tensor",
    threshold: float = 50.0,
    skip_ratio: float = 0.5,
    large_comp_threshold: int = 0,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", int]:
    """
    Apply STP-style token selection and return position indices for selected tokens.

    This function performs true UI-guided token selection and returns the (t, h, w)
    position of each selected token, enabling custom position encoding.

    Args:
        pixel_values: Tensor with shape (total_patches, patch_dim)
        image_grid_thw: Tensor with shape (num_images, 3) containing (t, h, w) for each image
        threshold: Patch similarity threshold for UI graph construction
        skip_ratio: Ratio of patches to skip within each component (0.0 to 1.0)
        large_comp_threshold: Components larger than this are fully skipped (0 = disabled)

    Returns:
        Tuple of:
        - selected_pixel_values: Tensor with shape (num_selected, patch_dim)
        - selected_positions: Tensor with shape (num_selected, 3) containing (t, h, w) for each token
        - num_selected_per_image: Tensor with shape (num_images,) containing count of selected tokens
        - num_reduced: Total number of tokens reduced
    """
    if threshold <= 0.0:
        # No reduction - return all tokens with their positions
        device = pixel_values.device
        dtype = pixel_values.dtype
        positions = []
        offset = 0
        for img_idx in range(len(image_grid_thw)):
            t, h, w = image_grid_thw[img_idx].tolist()
            for ti in range(t):
                for hi in range(h):
                    for wi in range(w):
                        positions.append([ti, hi, wi])
            offset += t * h * w
        positions = torch.tensor(positions, device=device, dtype=torch.long)
        num_per_image = torch.tensor([t * h * w for t, h, w in image_grid_thw.tolist()], device=device)
        return pixel_values, positions, num_per_image, 0

    device = pixel_values.device
    dtype = pixel_values.dtype
    pixel_values_np = pixel_values.cpu().numpy()
    grid_thw_np = image_grid_thw.cpu().numpy()

    selected_patches_list = []
    selected_positions_list = []
    num_selected_per_image = []
    total_original = 0
    total_selected = 0

    patch_offset = 0
    for img_idx in range(len(grid_thw_np)):
        t, h, w = grid_thw_np[img_idx]
        num_patches = t * h * w
        total_original += num_patches
        img_patches = pixel_values_np[patch_offset : patch_offset + num_patches]

        # Reshape patches to grid format: (t, h, w, patch_dim)
        patches_reshaped = img_patches.reshape(t, h, w, -1)

        image_selected_patches = []
        image_selected_positions = []

        for t_idx in range(t):
            frame_patches = patches_reshaped[t_idx]  # (h, w, patch_dim)
            frame_flat = frame_patches.reshape(h * w, -1)

            # Build UI graph
            patch_assign = build_ui_graph(
                patches=frame_flat,
                grid_h=h,
                grid_w=w,
                threshold=threshold,
            )

            # Get selection mask with boundary-aware sampling
            select_mask = get_select_mask(
                patch_assign, skip_ratio=skip_ratio, large_comp_threshold=large_comp_threshold,
                grid_h=h, grid_w=w
            )

            # Get selected patches and their positions
            selected_indices = np.where(select_mask)[0]
            for idx in selected_indices:
                h_pos = idx // w
                w_pos = idx % w
                image_selected_positions.append([t_idx, h_pos, w_pos])
                image_selected_patches.append(frame_flat[idx])

        if image_selected_patches:
            selected_patches_list.append(np.stack(image_selected_patches, axis=0))
            selected_positions_list.extend(image_selected_positions)
            num_selected = len(image_selected_patches)
        else:
            # Edge case: no patches selected, keep at least one
            selected_patches_list.append(img_patches[:1])
            selected_positions_list.append([0, 0, 0])
            num_selected = 1

        num_selected_per_image.append(num_selected)
        total_selected += num_selected
        patch_offset += num_patches

    # Convert to tensors
    selected_pixel_values = torch.from_numpy(np.concatenate(selected_patches_list, axis=0)).to(
        device=device, dtype=dtype
    )
    selected_positions = torch.tensor(selected_positions_list, device=device, dtype=torch.long)
    num_selected_tensor = torch.tensor(num_selected_per_image, device=device, dtype=torch.long)

    num_reduced = total_original - total_selected

    return selected_pixel_values, selected_positions, num_selected_tensor, num_reduced


def apply_stp_token_reduction(
    pixel_values: "torch.Tensor",
    image_grid_thw: "torch.Tensor",
    threshold: float = 50.0,
    skip_ratio: float = 0.5,
    merge_size: int = 2,
) -> tuple["torch.Tensor", "torch.Tensor", int]:
    """
    Apply STP-style UI-guided token reduction while maintaining grid structure.

    This implements true UI-guided selection with grid compatibility:
    1. Build UI graph to identify similar patches (components)
    2. For large uniform components (backgrounds), apply aggressive pooling
    3. For small/detailed components (UI elements), preserve more detail
    4. Maintain grid structure for Qwen2VL's mrope position encoding

    The key insight: instead of random selection, we use adaptive block-wise pooling
    where the pooling factor depends on local UI complexity.

    Args:
        pixel_values: Tensor with shape (total_patches, patch_dim)
        image_grid_thw: Tensor with shape (num_images, 3) containing (t, h, w) for each image
        threshold: Patch similarity threshold for UI graph construction
        skip_ratio: Target skip ratio for uniform regions (0.0 to 1.0)
        merge_size: Merge size used by the vision encoder (default 2)

    Returns:
        Tuple of:
        - reduced_pixel_values: Tensor with reduced patches
        - reduced_grid_thw: Updated grid dimensions
        - num_reduced: Number of tokens reduced
    """
    if threshold <= 0.0:
        return pixel_values, image_grid_thw, 0

    device = pixel_values.device
    dtype = pixel_values.dtype
    pixel_values_np = pixel_values.cpu().numpy()
    grid_thw_np = image_grid_thw.cpu().numpy()

    reduced_patches_list = []
    reduced_grid_list = []
    total_original = 0

    patch_offset = 0
    for img_idx in range(len(grid_thw_np)):
        t, h, w = grid_thw_np[img_idx]
        num_patches = t * h * w
        total_original += num_patches
        img_patches = pixel_values_np[patch_offset : patch_offset + num_patches]

        # We use 2x2 pooling as the reduction unit
        pool_size = 2
        if h >= pool_size and w >= pool_size and h % pool_size == 0 and w % pool_size == 0:
            new_h = h // pool_size
            new_w = w // pool_size

            # Reshape patches to grid format: (t, h, w, patch_dim)
            patches_reshaped = img_patches.reshape(t, h, w, -1)

            # Process each temporal slice
            reduced_frames = []

            for t_idx in range(t):
                frame_patches = patches_reshaped[t_idx]  # (h, w, patch_dim)
                frame_flat = frame_patches.reshape(h * w, -1)

                # Build UI graph to identify components
                patch_assign = build_ui_graph(
                    patches=frame_flat,
                    grid_h=h,
                    grid_w=w,
                    threshold=threshold,
                )
                assign_grid = patch_assign.reshape(h, w)

                # Calculate component sizes
                unique_comps, comp_counts = np.unique(patch_assign, return_counts=True)
                comp_size_map = dict(zip(unique_comps, comp_counts))

                # For each 2x2 block, decide pooling strategy based on UI complexity
                reduced_frame = np.zeros((new_h, new_w, frame_patches.shape[-1]))

                for bi in range(new_h):
                    for bj in range(new_w):
                        # Get the 2x2 block
                        block_assign = assign_grid[
                            bi * pool_size : (bi + 1) * pool_size,
                            bj * pool_size : (bj + 1) * pool_size,
                        ]
                        block_patches = frame_patches[
                            bi * pool_size : (bi + 1) * pool_size,
                            bj * pool_size : (bj + 1) * pool_size,
                        ]

                        unique_in_block = np.unique(block_assign)

                        if len(unique_in_block) == 1:
                            # Uniform block - all patches belong to same component
                            comp_id = unique_in_block[0]
                            comp_size = comp_size_map[comp_id]

                            # Large components (backgrounds) use simple average
                            # Small components (UI details) use weighted average preserving edges
                            if comp_size > 100:
                                # Large uniform region: simple average
                                reduced_frame[bi, bj] = block_patches.mean(axis=(0, 1))
                            else:
                                # Small component: preserve center/important features
                                # Use weighted average favoring high-variance patches
                                variances = block_patches.var(axis=-1)
                                weights = variances / (variances.sum() + 1e-8)
                                weights = weights.reshape(-1)
                                block_flat = block_patches.reshape(pool_size * pool_size, -1)
                                reduced_frame[bi, bj] = (block_flat * weights[:, None]).sum(axis=0)
                        else:
                            # Non-uniform block - boundary between components
                            # Use weighted average based on component importance (inverse of size)
                            block_flat = block_patches.reshape(pool_size * pool_size, -1)
                            block_assign_flat = block_assign.flatten()

                            # Weight smaller components more (they're more important UI elements)
                            weights = np.array(
                                [1.0 / comp_size_map[c] for c in block_assign_flat]
                            )
                            weights = weights / weights.sum()
                            reduced_frame[bi, bj] = (block_flat * weights[:, None]).sum(axis=0)

                reduced_frames.append(reduced_frame.reshape(-1, frame_patches.shape[-1]))

            # Stack all reduced frames
            reduced_patches_list.append(np.concatenate(reduced_frames, axis=0))
            reduced_grid_list.append([t, new_h, new_w])
        else:
            # Cannot pool, keep original but flatten to consistent format
            img_patches_flat = img_patches.reshape(num_patches, -1)
            reduced_patches_list.append(img_patches_flat)
            reduced_grid_list.append([t, h, w])

        patch_offset += num_patches

    # Concatenate all reduced patches
    # Note: reduced patches are flattened to (num_patches, patch_dim)
    # We need to reshape back to original format if needed
    if len(reduced_patches_list) == 0:
        return pixel_values, image_grid_thw, 0

    # Check if we need to reshape back to original format
    original_shape = pixel_values.shape[1:]  # e.g., (3, 14, 14)
    patch_dim = np.prod(original_shape)

    reduced_patches_concat = np.concatenate(reduced_patches_list, axis=0)

    # Reshape back to original format
    if len(original_shape) > 1:
        reduced_patches_concat = reduced_patches_concat.reshape(-1, *original_shape)

    reduced_pixel_values = torch.from_numpy(reduced_patches_concat).to(device=device, dtype=dtype)
    reduced_grid_thw = torch.tensor(reduced_grid_list, device=device, dtype=image_grid_thw.dtype)

    num_reduced = total_original - reduced_pixel_values.shape[0]

    return reduced_pixel_values, reduced_grid_thw, num_reduced


def apply_stp_embedding_selection(
    embeddings: "torch.Tensor",
    grid_thw: "torch.Tensor",
    threshold: float = 50.0,
    skip_ratio: float = 0.5,
    large_comp_threshold: int = 0,
) -> tuple["torch.Tensor", "torch.Tensor", int]:
    """
    Apply STP-style token selection on visual embeddings (after visual encoder).

    This function is designed to be used AFTER the visual encoder has processed
    all patches and generated embeddings. It selects representative tokens based
    on embedding similarity.

    Args:
        embeddings: Tensor with shape (total_tokens, hidden_dim) - visual embeddings
        grid_thw: Tensor with shape (num_images, 3) containing (t, h, w) for each image
        threshold: Embedding similarity threshold for component construction
        skip_ratio: Ratio of tokens to skip within each component (0.0 to 1.0)
        large_comp_threshold: Components larger than this are fully skipped (0 = disabled)

    Returns:
        Tuple of:
        - selected_embeddings: Tensor with selected embeddings
        - selection_indices: Tensor with indices of selected tokens (for position tracking)
        - num_reduced: Number of tokens reduced
    """
    if threshold <= 0.0:
        indices = torch.arange(embeddings.shape[0], device=embeddings.device)
        return embeddings, indices, 0

    device = embeddings.device
    dtype = embeddings.dtype

    # Work in float32 for stability
    embeddings_np = embeddings.float().cpu().numpy()
    grid_thw_np = grid_thw.cpu().numpy()

    selected_embeddings_list = []
    selected_indices_list = []
    total_original = 0
    total_selected = 0

    token_offset = 0
    for img_idx in range(len(grid_thw_np)):
        t, h, w = grid_thw_np[img_idx]
        num_tokens = t * h * w
        total_original += num_tokens
        img_embeddings = embeddings_np[token_offset : token_offset + num_tokens]

        # Reshape to grid format: (t, h, w, hidden_dim)
        emb_reshaped = img_embeddings.reshape(t, h, w, -1)

        # Process each temporal slice
        for t_idx in range(t):
            frame_emb = emb_reshaped[t_idx]  # (h, w, hidden_dim)
            frame_flat = frame_emb.reshape(h * w, -1)

            # Build UI graph based on embedding similarity
            patch_assign = build_ui_graph(
                patches=frame_flat,
                grid_h=h,
                grid_w=w,
                threshold=threshold,
            )

            # Get selection mask with boundary-aware sampling
            select_mask = get_select_mask(
                patch_assign, skip_ratio=skip_ratio, large_comp_threshold=large_comp_threshold,
                grid_h=h, grid_w=w
            )

            # Get selected embeddings and their original indices
            frame_indices = np.arange(h * w)
            selected_local_indices = frame_indices[select_mask]

            # Convert to global indices
            frame_global_offset = token_offset + t_idx * h * w
            selected_global_indices = selected_local_indices + frame_global_offset

            selected_embeddings_list.append(frame_flat[select_mask])
            selected_indices_list.extend(selected_global_indices.tolist())
            total_selected += len(selected_local_indices)

        token_offset += num_tokens

    # Handle case where all tokens are skipped
    if total_selected == 0:
        # Keep at least one token per image
        selected_embeddings_list = []
        selected_indices_list = []
        token_offset = 0
        for img_idx in range(len(grid_thw_np)):
            t, h, w = grid_thw_np[img_idx]
            num_tokens = t * h * w
            # Keep the first token of each image
            selected_embeddings_list.append(embeddings_np[token_offset : token_offset + 1])
            selected_indices_list.append(token_offset)
            token_offset += num_tokens
        total_selected = len(grid_thw_np)

    # Concatenate results
    selected_embeddings = torch.from_numpy(np.concatenate(selected_embeddings_list, axis=0)).to(
        device=device, dtype=dtype
    )
    selection_indices = torch.tensor(selected_indices_list, device=device, dtype=torch.long)

    num_reduced = total_original - total_selected

    return selected_embeddings, selection_indices, num_reduced


# Global storage for STP config (used by hooks)
_stp_config: dict = {}


def get_stp_mask_for_embeddings(
    embeddings: "torch.Tensor",
    grid_thw: "torch.Tensor",
    threshold: float = 50.0,
    skip_ratio: float = 0.5,
    large_comp_threshold: int = 0,
) -> "torch.Tensor":
    """
    Get a boolean mask indicating which embeddings should be masked (set to zero/placeholder).

    This function computes which visual tokens should be "skipped" based on UI similarity,
    returning a mask that can be used to zero out or replace those embeddings while
    preserving the grid structure and position encoding.

    Args:
        embeddings: Tensor with shape (total_tokens, hidden_dim) - visual embeddings
        grid_thw: Tensor with shape (num_images, 3) containing (t, h, w) for each image
        threshold: Embedding similarity threshold for component construction
        skip_ratio: Ratio of tokens to skip within each component (0.0 to 1.0)
        large_comp_threshold: Components larger than this are fully skipped (0 = disabled)

    Returns:
        Boolean mask tensor with shape (total_tokens,) - True means the token should be masked
    """
    if threshold <= 0.0:
        return torch.zeros(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)

    device = embeddings.device

    # Work in float32 for stability
    embeddings_np = embeddings.float().cpu().numpy()
    grid_thw_np = grid_thw.cpu().numpy()

    mask_list = []

    token_offset = 0
    for img_idx in range(len(grid_thw_np)):
        t, h, w = grid_thw_np[img_idx]
        num_tokens = t * h * w
        img_embeddings = embeddings_np[token_offset : token_offset + num_tokens]

        # Reshape to grid format: (t, h, w, hidden_dim)
        emb_reshaped = img_embeddings.reshape(t, h, w, -1)

        # Process each temporal slice
        for t_idx in range(t):
            frame_emb = emb_reshaped[t_idx]  # (h, w, hidden_dim)
            frame_flat = frame_emb.reshape(h * w, -1)

            # Build UI graph based on embedding similarity
            patch_assign = build_ui_graph(
                patches=frame_flat,
                grid_h=h,
                grid_w=w,
                threshold=threshold,
            )

            # Get selection mask (True = keep, False = skip) with boundary-aware sampling
            select_mask = get_select_mask(
                patch_assign, skip_ratio=skip_ratio, large_comp_threshold=large_comp_threshold,
                grid_h=h, grid_w=w
            )

            # Invert to get mask (True = should be masked/skipped)
            skip_mask = ~select_mask
            mask_list.append(skip_mask)

        token_offset += num_tokens

    # Concatenate all masks
    full_mask = np.concatenate(mask_list, axis=0)
    return torch.from_numpy(full_mask).to(device=device)


def patch_stp_visual_encoder_with_masking(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
) -> None:
    """
    Patch the visual encoder to apply STP token masking after embedding generation.

    This approach preserves grid structure and position encoding by:
    1. Computing which tokens should be "skipped" based on UI similarity
    2. Masking (zeroing out) the skipped tokens' embeddings
    3. Keeping all tokens in place so position IDs remain correct

    This is different from token removal - the masked tokens still exist but
    carry no information, effectively making them "invisible" to the model.

    Args:
        model: The pretrained VLM model
        model_args: Model arguments containing STP configuration
    """
    if not getattr(model_args, "use_stp", False):
        return

    threshold = getattr(model_args, "stp_threshold", 0.0)
    if threshold <= 0.0:
        return

    large_comp_threshold = getattr(model_args, "stp_large_comp_threshold", 0)
    if large_comp_threshold == 0:
        # Use token reduction mode in data preprocessing instead
        return

    model_type = getattr(model.config, "model_type", None)
    if model_type not in ["qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe"]:
        logger.warning_rank0(f"STP visual encoder masking is not supported for model type: {model_type}")
        return

    # Store config globally for use in hooks
    _stp_config["threshold"] = threshold
    _stp_config["skip_ratio"] = getattr(model_args, "stp_skip_ratio", 0.5)
    _stp_config["large_comp_threshold"] = large_comp_threshold
    _stp_config["patch_level"] = getattr(model_args, "stp_patch_level", False)
    _stp_config["patch_to_token_strategy"] = getattr(model_args, "stp_patch_to_token_strategy", "any")
    _stp_config["temporal_aggregation"] = getattr(model_args, "stp_temporal_aggregation", "first")
    _stp_config["use_raw_frames_in_stp"] = getattr(model_args, "use_raw_frames_in_stp", False)

    # Get the model's internal model (handles different transformers versions)
    if hasattr(model, "model"):
        inner_model = model.model
    else:
        inner_model = model

    # Get vision encoder parameters from model config
    vision_config = getattr(model.config, "vision_config", None)
    if vision_config is not None:
        _stp_config["patch_size"] = getattr(vision_config, "patch_size", 14)
        _stp_config["temporal_patch_size"] = getattr(vision_config, "temporal_patch_size", 2)
        _stp_config["in_channels"] = getattr(vision_config, "in_channels", 3)
    else:
        # Default values for Qwen2-VL
        _stp_config["patch_size"] = 14
        _stp_config["temporal_patch_size"] = 2
        _stp_config["in_channels"] = 3

    # Get merge_size from image processor if available
    if hasattr(inner_model, "visual") and hasattr(inner_model.visual, "merge_size"):
        _stp_config["merge_size"] = inner_model.visual.merge_size
    else:
        _stp_config["merge_size"] = 2

    _stp_config["channel"] = _stp_config.get("in_channels", 3)

    # Patch get_image_features method to apply masking
    original_get_image_features = inner_model.get_image_features

    def patched_get_image_features(pixel_values, grid_thw):
        # Call original method
        result = original_get_image_features(pixel_values, grid_thw)

        # Handle different return types (Qwen2VL vs Qwen3VL)
        if isinstance(result, tuple):
            image_embeds = result[0]
            extra = result[1:]
        else:
            image_embeds = result
            extra = None

        # Apply STP masking on embeddings (zero out skipped tokens)
        if _stp_config.get("large_comp_threshold", 0) > 0:
            # Use pixel-based UI graph construction (STP original approach)
            # This is more accurate than embedding-based construction
            keep_mask = compute_token_keep_mask_from_pixels(
                pixel_values=pixel_values,
                grid_thw=grid_thw,
                threshold=_stp_config["threshold"],
                skip_ratio=_stp_config["skip_ratio"],
                patch_size=_stp_config.get("patch_size", 14),
                temporal_patch_size=_stp_config.get("temporal_patch_size", 2),
                merge_size=_stp_config.get("merge_size", 2),
                channel=_stp_config.get("channel", 3),
                patch_level=_stp_config.get("patch_level", False),
                patch_to_token_strategy=_stp_config.get("patch_to_token_strategy", "any"),
                temporal_aggregation=_stp_config.get("temporal_aggregation", "first"),
                use_raw_frames_in_stp=_stp_config.get("use_raw_frames_in_stp", False),
            )

            # Invert to get skip mask (True = should be masked/skipped)
            skip_mask = ~keep_mask

            # Zero out the masked embeddings
            # This preserves grid structure while "hiding" redundant tokens
            if skip_mask.any():
                image_embeds = image_embeds.clone()
                image_embeds[skip_mask] = 0.0
                num_masked = skip_mask.sum().item()
                _stp_config["last_num_masked"] = num_masked

        if extra is not None:
            # For Qwen3VL which returns (image_embeds, deepstack_embeds)
            # We also need to mask deepstack_embeds if present
            if len(extra) > 0 and extra[0] is not None:
                deepstack = extra[0]
                if "last_num_masked" in _stp_config and _stp_config["last_num_masked"] > 0:
                    # Apply same mask to deepstack
                    if deepstack.shape[0] == image_embeds.shape[0]:
                        deepstack = deepstack.clone()
                        deepstack[skip_mask] = 0.0
                return (image_embeds, deepstack) + extra[1:]
            return (image_embeds,) + extra
        return image_embeds

    inner_model.get_image_features = patched_get_image_features

    logger.info_rank0(
        f"STP visual encoder masking applied: threshold={threshold}, "
        f"skip_ratio={_stp_config['skip_ratio']}, "
        f"large_comp_threshold={large_comp_threshold}"
    )


def get_image_token_indices(
    input_ids: "torch.Tensor",
    image_token_id: int,
) -> "torch.Tensor":
    """
    Get the indices of image tokens in the input sequence.

    Args:
        input_ids: Tensor with shape (batch, seq_len) or (seq_len,)
        image_token_id: The token ID used as image placeholder

    Returns:
        Boolean mask with shape matching input_ids, True for image tokens
    """
    return input_ids == image_token_id


def get_rope_index_with_stp(
    input_ids: "torch.LongTensor",
    attention_mask: "torch.Tensor",
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
    image_grid_thw: Optional["torch.LongTensor"] = None,
    video_grid_thw: Optional["torch.LongTensor"] = None,
    stp_token_positions: Optional["torch.LongTensor"] = None,
    stp_num_selected: Optional["torch.LongTensor"] = None,
    stp_video_token_positions: Optional["torch.LongTensor"] = None,
    stp_video_num_selected: Optional["torch.LongTensor"] = None,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Compute RoPE indices for Qwen-VL mRoPE when STP removed visual tokens in preprocessing.

    This is used by the data collator when mm preprocessing returns:
    - `stp_token_positions` + `stp_num_selected` (for images)
    - `stp_video_token_positions` + `stp_video_num_selected` (for videos)

    The key idea: visual tokens are fewer (after removal), but each kept token still
    carries its original (t,h,w) position, so we must build 3D position_ids from
    these preserved positions.
    """
    # Check if we have STP position info
    use_stp_positions = stp_token_positions is not None and stp_num_selected is not None
    use_stp_video_positions = (
        stp_video_token_positions is not None and stp_video_num_selected is not None
    )

    # Handle video_grid_thw splitting for Qwen3VL (t>1 gets expanded into per-frame grids)
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)

        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        image_index, video_index = 0, 0
        stp_image_offset, stp_video_offset = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)

        for i, input_ids_i in enumerate(total_input_ids):
            input_ids_i = input_ids_i[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(input_ids_i == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids_i[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids_i.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                if ed_image < ed_video:
                    # Processing image
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    ed = ed_image
                    is_image = True
                    current_image_index = image_index
                    image_index += 1
                    remain_images -= 1
                else:
                    # Processing video
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    ed = ed_video
                    is_image = False
                    current_video_index = video_index
                    video_index += 1
                    remain_videos -= 1

                text_len = ed - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # Compute vision token positions
                if is_image and use_stp_positions:
                    # Use pre-computed STP positions for this image
                    num_tokens = stp_num_selected[current_image_index].item()
                    positions = stp_token_positions[stp_image_offset : stp_image_offset + num_tokens]
                    stp_image_offset += num_tokens

                    # positions is (num_tokens, 3) with (t, h, w) for each token
                    t_index = positions[:, 0]
                    h_index = positions[:, 1]
                    w_index = positions[:, 2]
                    vision_pos_ids = torch.stack([t_index, h_index, w_index]).to(input_ids.device)
                    llm_pos_ids_list.append(vision_pos_ids + text_len + st_idx)
                    st = ed + num_tokens

                elif not is_image and use_stp_video_positions:
                    # Use pre-computed STP positions for this video frame
                    num_tokens = stp_video_num_selected[current_video_index].item()
                    positions = stp_video_token_positions[stp_video_offset : stp_video_offset + num_tokens]
                    stp_video_offset += num_tokens

                    t_index = positions[:, 0]
                    h_index = positions[:, 1]
                    w_index = positions[:, 2]
                    vision_pos_ids = torch.stack([t_index, h_index, w_index]).to(input_ids.device)
                    llm_pos_ids_list.append(vision_pos_ids + text_len + st_idx)
                    st = ed + num_tokens

                else:
                    # Standard position computation (no STP or fallback)
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]).to(input_ids.device) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    # Fallback for pure text
    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    else:
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            .view(1, 1, -1)
            .expand(3, input_ids.shape[0], -1)
        )
        mrope_position_deltas = torch.zeros(
            [input_ids.shape[0], 1],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )

    return position_ids, mrope_position_deltas


def compute_token_keep_mask_from_pixels(
    pixel_values: "torch.Tensor",
    grid_thw: "torch.Tensor",
    threshold: float = 1.0,
    skip_ratio: float = 0.5,
    large_comp_threshold: int = 0,
    patch_size: int = 14,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    channel: int = 3,
    patch_level: bool = False,
    patch_to_token_strategy: str = "any",
    temporal_aggregation: str = "first",
    use_raw_frames_in_stp: bool = False,
) -> "torch.Tensor":
    """
    Compute which image tokens to keep based on STP algorithm using PIXEL VALUES.

    This follows the exact STP implementation: reshape patches to the STP format
    and build UI graph by comparing adjacent patches.

    IMPORTANT: The Qwen2VL/Qwen3VL processor outputs pixel_values in a specific order:
    - The patches are arranged by merge blocks: (out_h, out_w, merge_size, merge_size)
    - NOT in simple row-major order (h, w)
    - This is because the processor uses permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
      which groups patches by their spatial merge block

    Args:
        pixel_values: Tensor with shape (total_patches, patch_dim) where
            patch_dim = channel * temporal_patch_size * patch_size * patch_size
            Patches are ordered as (out_h, out_w, merge_size, merge_size) NOT (h, w)
        grid_thw: Tensor with shape (num_images, 3) containing (t, h, w) for each image
            where h and w are in terms of patch grid (before merge)
        threshold: Patch similarity threshold for UI graph construction.
            For Qwen2-VL (CLIP normalization): use threshold=1 as default
            For Qwen3-VL (simple normalization): use threshold=1.7 as equivalent
        skip_ratio: Ratio of patches to skip within each component
        large_comp_threshold: Components with size > this use different skip ratio
        patch_size: Size of each patch (14 for Qwen2VL, 16 for Qwen3VL)
        temporal_patch_size: Temporal patch size (default 2)
        merge_size: Spatial merge size (default 2, output tokens = h/2 * w/2)
        channel: Number of image channels (default 3)
        patch_level: If True, perform STP analysis at individual patch level (e.g., 48x48)
            instead of merged token level (e.g., 24x24). This preserves more fine-grained
            information. The patch-level decisions are then mapped to token-level using
            patch_to_token_strategy.
        patch_to_token_strategy: How to map patch-level decisions to token-level when
            patch_level=True. Options:
            - "any": Keep token if ANY of its patches are kept (conservative, keeps more)
            - "all": Keep token only if ALL of its patches are kept (aggressive)
            - "majority": Keep token if more than half of its patches are kept
        temporal_aggregation: How to handle temporal patches when temporal_patch_size > 1.
            For video where consecutive frames may differ significantly, mixing all temporal
            frames can confuse edge detection. Options:
            - "first": Use only the first temporal frame for edge detection (recommended for video)
            - "mean": Average across temporal frames
            - "all": Use all temporal frames concatenated (original behavior, good for static images)
        use_raw_frames_in_stp: If True and temporal_patch_size>1, run STP edge detection
            on each raw temporal frame (within each temporal patch group) and OR the keep masks.
            This can preserve tokens corresponding to UI elements that are present in any frame.

    Returns:
        Boolean mask where True = keep, False = remove
        Shape: (total_output_tokens,) where output tokens account for merge_size
    """
    if threshold <= 0.0:
        # Calculate total output tokens using Qwen3VL formula
        # See: transformers/models/qwen3_vl/modeling_qwen3_vl.py get_image_features
        # split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        total_tokens = (grid_thw.prod(dim=-1) // (merge_size * merge_size)).sum().item()
        return torch.ones(int(total_tokens), dtype=torch.bool, device=pixel_values.device)

    # Check cache first for faster inference
    cache_key = _get_keep_mask_cache_key(
        pixel_values,
        threshold=threshold,
        skip_ratio=skip_ratio,
        large_comp_threshold=large_comp_threshold,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        merge_size=merge_size,
        channel=channel,
        patch_level=patch_level,
        patch_to_token_strategy=patch_to_token_strategy,
        temporal_aggregation=temporal_aggregation,
        use_raw_frames_in_stp=use_raw_frames_in_stp,
    )
    cached_mask = _get_cached_mask(cache_key)
    if cached_mask is not None:
        # Ensure mask is on correct device
        if cached_mask.device != pixel_values.device:
            cached_mask = cached_mask.to(pixel_values.device)
        return cached_mask

    # Try GPU-accelerated version first (much faster, no CPU transfer)
    if pixel_values.is_cuda:
        try:
            if patch_level:
                # GPU version for patch-level analysis
                result = compute_token_keep_mask_from_pixels_gpu_patch_level(
                    pixel_values=pixel_values,
                    grid_thw=grid_thw,
                    threshold=threshold,
                    skip_ratio=skip_ratio,
                    large_comp_threshold=large_comp_threshold,
                    patch_size=patch_size,
                    temporal_patch_size=temporal_patch_size,
                    merge_size=merge_size,
                    channel=channel,
                    patch_to_token_strategy=patch_to_token_strategy,
                    temporal_aggregation=temporal_aggregation,
                    use_raw_frames_in_stp=use_raw_frames_in_stp,
                )
            else:
                # GPU version for token-level analysis
                result = compute_token_keep_mask_from_pixels_gpu(
                    pixel_values=pixel_values,
                    grid_thw=grid_thw,
                    threshold=threshold,
                    skip_ratio=skip_ratio,
                    large_comp_threshold=large_comp_threshold,
                    patch_size=patch_size,
                    temporal_patch_size=temporal_patch_size,
                    merge_size=merge_size,
                    channel=channel,
                    temporal_aggregation=temporal_aggregation,
                    use_raw_frames_in_stp=use_raw_frames_in_stp,
                )
            # Cache and return
            _cache_keep_mask(cache_key, result)
            return result
        except Exception as e:
            # Fall back to CPU version if GPU version fails
            logger.warning_rank0(f"GPU STP failed, falling back to CPU: {e}")

    # CPU version (original implementation)
    device = pixel_values.device
    pixel_np = pixel_values.float().cpu().numpy()
    grid_thw_np = grid_thw.cpu().numpy()

    keep_list = []
    patch_offset = 0

    for img_idx in range(len(grid_thw_np)):
        t, h, w = grid_thw_np[img_idx]
        num_patches = t * h * w

        # Get patches for this image
        img_patches = pixel_np[patch_offset : patch_offset + num_patches]

        # Output grid dimensions (after merge)
        out_h = h // merge_size
        out_w = w // merge_size

        # IMPORTANT: Processor outputs patches in merge-block order, NOT row-major
        # The flatten order is: (t, out_h, out_w, merge_h, merge_w) -> linear index
        # So we reshape directly to this structure:
        # Step 1: (t * out_h * out_w * ms * ms, patch_dim) -> (t, out_h, out_w, ms, ms, patch_dim)
        patches_by_merge = img_patches.reshape(
            t, out_h, out_w, merge_size, merge_size, -1
        )

        # Step 2: Further expand patch_dim to (c, tp, ps, ps)
        # patch_dim = c * tp * ps * ps
        patches_stp = patches_by_merge.reshape(
            t, out_h, out_w, merge_size, merge_size,
            channel, temporal_patch_size, patch_size, patch_size
        )

        # Now patches_stp has shape (t, out_h, out_w, ms, ms, c, tp, ps, ps)

        # Build UI graph for each temporal slice
        for t_idx in range(t):
            if patch_level:
                # Work at individual patch level (finer granularity)
                # Reshape to (h, w, c, tp, ps, ps) - individual patches
                # From (out_h, out_w, ms, ms, c, tp, ps, ps) -> (h, w, c, tp, ps, ps)
                frame_patches = patches_stp[t_idx]  # (out_h, out_w, ms, ms, c, tp, ps, ps)

                # Rearrange to get individual patches in row-major order
                # (out_h, out_w, ms, ms, c, tp, ps, ps) -> (out_h, ms, out_w, ms, c, tp, ps, ps)
                # -> (h, w, c, tp, ps, ps)
                patches_individual = frame_patches.transpose(0, 2, 1, 3, 4, 5, 6, 7)
                patches_individual = patches_individual.reshape(h, w, channel, temporal_patch_size, patch_size, patch_size)

                # Scale large_comp_threshold for patch level
                # Each token has merge_size^2 patches, so multiply threshold accordingly
                patch_large_comp_threshold = large_comp_threshold * (merge_size * merge_size) if large_comp_threshold > 0 else 0

                # Optionally compute a keep mask for each raw temporal frame and OR them.
                if use_raw_frames_in_stp and temporal_patch_size > 1:
                    per_tp_token_masks = []
                    for tp_idx in range(temporal_patch_size):
                        patches_for_graph = patches_individual[:, :, :, tp_idx:tp_idx + 1, :, :]
                        patch_assign = _build_uigraph_at_patch_level(
                            patches_for_graph, h, w, threshold
                        )
                        patch_select_mask = _get_select_mask_stp(
                            patch_assign, skip_ratio, patch_large_comp_threshold
                        )
                        per_tp_token_masks.append(
                            _map_patch_mask_to_token_mask(
                                patch_select_mask, h, w, merge_size, strategy=patch_to_token_strategy
                            )
                        )
                    select_mask = np.logical_or.reduce(per_tp_token_masks)
                else:
                    # Apply temporal aggregation for edge detection
                    if temporal_aggregation == "all" or temporal_patch_size <= 1:
                        # Use all temporal frames (original behavior)
                        patches_for_graph = patches_individual
                    elif temporal_aggregation == "first":
                        # Use only first temporal frame
                        patches_for_graph = patches_individual[:, :, :, 0:1, :, :]
                    elif temporal_aggregation == "mean":
                        # Average across temporal frames
                        patches_for_graph = patches_individual.mean(axis=3, keepdims=True)
                    else:
                        raise ValueError(f"Unknown temporal_aggregation: {temporal_aggregation}")

                    # Build UI graph at patch level
                    patch_assign = _build_uigraph_at_patch_level(
                        patches_for_graph, h, w, threshold
                    )

                    # Get selection mask at patch level
                    patch_select_mask = _get_select_mask_stp(
                        patch_assign, skip_ratio, patch_large_comp_threshold
                    )

                    # Map patch-level mask to token-level mask
                    select_mask = _map_patch_mask_to_token_mask(
                        patch_select_mask, h, w, merge_size, strategy=patch_to_token_strategy
                    )
            else:
                # Work at merged token level (original behavior)
                # patches_stp[t_idx] has shape (out_h, out_w, ms, ms, c, tp, ps, ps)
                frame_patches = patches_stp[t_idx]

                # Optionally compute a keep mask for each raw temporal frame and OR them.
                if use_raw_frames_in_stp and temporal_patch_size > 1:
                    per_tp_masks = []
                    for tp_idx in range(temporal_patch_size):
                        patches_for_graph = frame_patches[:, :, :, :, :, tp_idx:tp_idx + 1, :, :]
                        patch_assign = _build_uigraph_stp(
                            patches_for_graph, out_h, out_w, threshold
                        )
                        per_tp_masks.append(
                            _get_select_mask_stp(patch_assign, skip_ratio, large_comp_threshold)
                        )
                    select_mask = np.logical_or.reduce(per_tp_masks)
                else:
                    # Apply temporal aggregation for edge detection
                    if temporal_aggregation == "all" or temporal_patch_size <= 1:
                        # Use all temporal frames (original behavior)
                        patches_for_graph = frame_patches
                    elif temporal_aggregation == "first":
                        # Use only first temporal frame
                        patches_for_graph = frame_patches[:, :, :, :, :, 0:1, :, :]
                    elif temporal_aggregation == "mean":
                        # Average across temporal frames
                        patches_for_graph = frame_patches.mean(axis=5, keepdims=True)
                    else:
                        raise ValueError(f"Unknown temporal_aggregation: {temporal_aggregation}")

                    patch_assign = _build_uigraph_stp(
                        patches_for_graph, out_h, out_w, threshold
                    )

                    # Get selection mask using STP's uniform sampling with large component skip
                    select_mask = _get_select_mask_stp(patch_assign, skip_ratio, large_comp_threshold)

            keep_list.append(select_mask)

        patch_offset += num_patches

    full_keep_mask = np.concatenate(keep_list, axis=0)
    result = torch.from_numpy(full_keep_mask).to(device=device)

    # Cache the result for faster subsequent calls
    _cache_keep_mask(cache_key, result)

    return result


def preprocess_visual_tokens_with_stp(
    pixel_values: "torch.Tensor",
    grid_thw: "torch.Tensor",
    threshold: float = 1.7,
    skip_ratio: float = 0.5,
    large_comp_threshold: int = 0,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    channel: int = 3,
    patch_level: bool = False,
    patch_to_token_strategy: str = "any",
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", int]:
    """
    Preprocess visual tokens using STP algorithm: compute positions BEFORE token removal.

    This function is designed for use in the data preprocessing stage (mm_plugin).
    It computes the position (t, h, w) for each token BEFORE removal, then removes
    tokens while preserving their position information.

    IMPORTANT: The input pixel_values from processor are in merge-block order:
      (t, out_h, out_w, merge_size, merge_size) flattened
    Each group of merge_size^2 consecutive patches belongs to one merged token.

    This function outputs:
    - The unmerged patches for the kept merged tokens (for model input)
    - The position of each kept MERGED token (for position embedding)
    - A keep mask that maps from output merged token index to original merged token index

    Args:
        pixel_values: Tensor with shape (total_patches, patch_dim)
            Patches are in processor output order: (t, out_h, out_w, ms, ms) flattened
            Total patches = sum(t * h * w) for all images
        grid_thw: Tensor with shape (num_images, 3) containing (t, h, w) for each image
            where h and w are in terms of patch grid (BEFORE merge)
        threshold: Patch similarity threshold for UI graph construction
        skip_ratio: Ratio of patches to skip within each component
        large_comp_threshold: Skip entire components with size > this (0 = disabled)
        patch_size: Patch size (16 for Qwen3VL)
        temporal_patch_size: Temporal patch size (default 2)
        merge_size: Spatial merge size (default 2)
        channel: Number of channels (default 3)
        patch_level: If True, perform STP analysis at individual patch level (e.g., 48x48)
            instead of merged token level (e.g., 24x24). This preserves more fine-grained
            information. The patch-level decisions are then mapped to token-level using
            patch_to_token_strategy.
        patch_to_token_strategy: How to map patch-level decisions to token-level when
            patch_level=True. Options:
            - "any": Keep token if ANY of its patches are kept (conservative, keeps more)
            - "all": Keep token only if ALL patches are kept (aggressive)
            - "majority": Keep token if more than half of patches are kept

    Returns:
        Tuple of:
        - selected_pixel_values: Tensor (num_selected_patches, patch_dim)
            The unmerged patches for kept merged tokens.
            num_selected_patches = num_selected_merged_tokens * merge_size^2
        - selected_positions: Tensor (num_selected_merged_tokens, 3)
            (t, h, w) position for each kept merged token in the ORIGINAL grid
        - keep_mask: Boolean tensor (total_merged_tokens,)
            True for merged tokens that are kept
        - num_selected_per_image: Tensor (num_images,)
            Number of kept merged tokens per image
        - num_removed: Total number of merged tokens removed
    """
    ms2 = merge_size * merge_size  # Number of sub-patches per merged token

    if threshold <= 0.0:
        # No removal - return all tokens with their positions
        device = pixel_values.device
        positions_list = []
        num_per_image = []
        total_merged = 0

        for img_idx in range(len(grid_thw)):
            t, h, w = grid_thw[img_idx].tolist()
            out_h = h // merge_size
            out_w = w // merge_size
            num_merged_tokens = t * out_h * out_w

            # Generate positions for all merged tokens
            for t_idx in range(t):
                for i in range(out_h):
                    for j in range(out_w):
                        # Position of merged token in the original grid (top-left corner)
                        positions_list.append([t_idx, i * merge_size, j * merge_size])

            num_per_image.append(num_merged_tokens)
            total_merged += num_merged_tokens

        positions = torch.tensor(positions_list, device=device, dtype=torch.long)
        num_selected = torch.tensor(num_per_image, device=device, dtype=torch.long)
        keep_mask = torch.ones(total_merged, dtype=torch.bool, device=device)
        return pixel_values, positions, keep_mask, num_selected, 0

    device = pixel_values.device
    dtype = pixel_values.dtype
    pixel_np = pixel_values.float().cpu().numpy()
    grid_thw_np = grid_thw.cpu().numpy()

    selected_patches_list = []  # List of numpy arrays, each (ms^2, patch_dim)
    selected_positions_list = []  # List of [t, h, w]
    keep_mask_list = []  # List of boolean arrays
    num_selected_per_image = []
    total_original = 0
    total_selected = 0

    patch_offset = 0
    for img_idx in range(len(grid_thw_np)):
        t, h, w = grid_thw_np[img_idx]
        num_patches = t * h * w
        out_h = h // merge_size
        out_w = w // merge_size
        num_merged_tokens = t * out_h * out_w
        total_original += num_merged_tokens

        img_patches = pixel_np[patch_offset : patch_offset + num_patches]

        # Reshape to separate merged tokens and sub-patches
        # processor order: (t, out_h, out_w, ms, ms) flattened
        # Reshape to: (t, out_h, out_w, ms, ms, patch_dim)
        patches_by_merge = img_patches.reshape(
            t, out_h, out_w, merge_size, merge_size, -1
        )

        # Also reshape for STP graph building
        patches_stp = patches_by_merge.reshape(
            t, out_h, out_w, merge_size, merge_size,
            channel, temporal_patch_size, patch_size, patch_size
        )

        image_keep_mask = []

        for t_idx in range(t):
            if patch_level:
                # Work at individual patch level (finer granularity)
                # Reshape to (h, w, c, tp, ps, ps) - individual patches
                frame_patches = patches_stp[t_idx]  # (out_h, out_w, ms, ms, c, tp, ps, ps)

                # Rearrange to get individual patches in row-major order
                patches_individual = frame_patches.transpose(0, 2, 1, 3, 4, 5, 6, 7)
                patches_individual = patches_individual.reshape(h, w, channel, temporal_patch_size, patch_size, patch_size)

                # Build UI graph at patch level
                patch_assign = _build_uigraph_at_patch_level(
                    patches_individual, h, w, threshold
                )

                # Scale large_comp_threshold for patch level
                # Each token has merge_size^2 patches, so multiply threshold accordingly
                patch_large_comp_threshold = large_comp_threshold * (merge_size * merge_size) if large_comp_threshold > 0 else 0

                # Get selection mask at patch level
                patch_select_mask = _get_select_mask_stp(patch_assign, skip_ratio, patch_large_comp_threshold)

                # Map patch-level mask to token-level mask
                select_mask = _map_patch_mask_to_token_mask(
                    patch_select_mask, h, w, merge_size, strategy=patch_to_token_strategy
                )
            else:
                # Work at merged token level (original behavior)
                patch_assign = _build_uigraph_stp(
                    patches_stp[t_idx], out_h, out_w, threshold
                )

                # Get selection mask with large component skip
                select_mask = _get_select_mask_stp(
                    patch_assign, skip_ratio, large_comp_threshold
                )

            image_keep_mask.append(select_mask)

            # Extract selected patches and their positions
            selected_indices = np.where(select_mask)[0]
            for idx in selected_indices:
                i = idx // out_w
                j = idx % out_w
                # Position in original grid (top-left of merge block)
                selected_positions_list.append([t_idx, i * merge_size, j * merge_size])
                # Get all sub-patches for this merged token
                # Shape: (ms, ms, patch_dim) -> (ms^2, patch_dim)
                sub_patches = patches_by_merge[t_idx, i, j].reshape(ms2, -1)
                selected_patches_list.append(sub_patches)

            total_selected += len(selected_indices)

        keep_mask_list.append(np.concatenate(image_keep_mask))
        num_selected_per_image.append(total_selected - sum(num_selected_per_image))
        patch_offset += num_patches

    # Handle edge case: no tokens selected
    if total_selected == 0:
        # Keep at least one token per image
        selected_patches_list = []
        selected_positions_list = []
        num_selected_per_image = []
        keep_mask_list = []
        patch_offset = 0
        for img_idx in range(len(grid_thw_np)):
            t, h, w = grid_thw_np[img_idx]
            out_h = h // merge_size
            out_w = w // merge_size
            num_merged = t * out_h * out_w

            # Keep first merged token
            first_merged_patches = pixel_np[patch_offset : patch_offset + ms2]
            selected_patches_list.append(first_merged_patches)
            selected_positions_list.append([0, 0, 0])
            num_selected_per_image.append(1)

            # Create keep mask with only first token kept
            mask = np.zeros(num_merged, dtype=bool)
            mask[0] = True
            keep_mask_list.append(mask)

            patch_offset += t * h * w
        total_selected = len(grid_thw_np)

    # Concatenate all patches: (num_selected, ms^2, patch_dim) -> (num_selected * ms^2, patch_dim)
    selected_pixel_values = torch.from_numpy(
        np.concatenate(selected_patches_list, axis=0)
    ).to(device=device, dtype=dtype)

    selected_positions = torch.tensor(selected_positions_list, device=device, dtype=torch.long)
    keep_mask = torch.from_numpy(np.concatenate(keep_mask_list)).to(device=device)
    num_selected_tensor = torch.tensor(num_selected_per_image, device=device, dtype=torch.long)

    num_removed = total_original - total_selected

    return selected_pixel_values, selected_positions, keep_mask, num_selected_tensor, num_removed


def _build_uigraph_stp(
    patches: np.ndarray,
    grid_h_half: int,
    grid_w_half: int,
    threshold: float,
) -> np.ndarray:
    """
    Build UI graph following STP's exact implementation.

    Args:
        patches: Array with shape (h_half, w_half, ms, ms, c, tp, ps, ps)
        grid_h_half: Height of merged grid
        grid_w_half: Width of merged grid
        threshold: Difference threshold for merging

    Returns:
        Array of component assignments with shape (h_half * w_half,)
    """
    num_patches = grid_h_half * grid_w_half
    uf = UnionFind(num_patches)

    def idx(i: int, j: int) -> int:
        return i * grid_w_half + j

    # Compare adjacent patches
    for i in range(grid_h_half):
        for j in range(grid_w_half):
            current_idx = idx(i, j)
            current_patch = patches[i, j]

            # Compare with right neighbor
            if j + 1 < grid_w_half:
                right_patch = patches[i, j + 1]
                diff = np.linalg.norm(current_patch - right_patch)
                if diff < threshold:
                    uf.union(current_idx, idx(i, j + 1))

            # Compare with bottom neighbor
            if i + 1 < grid_h_half:
                bottom_patch = patches[i + 1, j]
                diff = np.linalg.norm(current_patch - bottom_patch)
                if diff < threshold:
                    uf.union(current_idx, idx(i + 1, j))

    # Get component assignments
    uigraph_assign = np.array([uf.find(x) for x in range(num_patches)])

    # Rerank values to be contiguous
    mapping = {}
    new_arr = np.empty_like(uigraph_assign)
    next_value = 0
    for idx_val, x in enumerate(uigraph_assign):
        if x not in mapping:
            mapping[x] = next_value
            next_value += 1
        new_arr[idx_val] = mapping[x]

    return new_arr


def _build_uigraph_at_patch_level(
    patches: np.ndarray,
    grid_h: int,
    grid_w: int,
    threshold: float,
) -> np.ndarray:
    """
    Build UI graph at the PATCH level (before merge), not at merged token level.

    This provides finer granularity for STP analysis:
    - Original: works on 24x24 merged tokens
    - This function: works on 48x48 individual patches

    Args:
        patches: Array with shape (h, w, c, tp, ps, ps) - individual patches, NOT merged
        grid_h: Full height of patch grid (e.g., 48)
        grid_w: Full width of patch grid (e.g., 48)
        threshold: Difference threshold for merging similar patches

    Returns:
        Array of component assignments with shape (grid_h * grid_w,)
    """
    num_patches = grid_h * grid_w
    uf = UnionFind(num_patches)

    def idx(i: int, j: int) -> int:
        return i * grid_w + j

    # Compare adjacent patches at full resolution
    for i in range(grid_h):
        for j in range(grid_w):
            current_idx = idx(i, j)
            current_patch = patches[i, j]

            # Compare with right neighbor
            if j + 1 < grid_w:
                right_patch = patches[i, j + 1]
                diff = np.linalg.norm(current_patch - right_patch)
                if diff < threshold:
                    uf.union(current_idx, idx(i, j + 1))

            # Compare with bottom neighbor
            if i + 1 < grid_h:
                bottom_patch = patches[i + 1, j]
                diff = np.linalg.norm(current_patch - bottom_patch)
                if diff < threshold:
                    uf.union(current_idx, idx(i + 1, j))

    # Get component assignments
    uigraph_assign = np.array([uf.find(x) for x in range(num_patches)])

    # Rerank values to be contiguous
    mapping = {}
    new_arr = np.empty_like(uigraph_assign)
    next_value = 0
    for idx_val, x in enumerate(uigraph_assign):
        if x not in mapping:
            mapping[x] = next_value
            next_value += 1
        new_arr[idx_val] = mapping[x]

    return new_arr


def _map_patch_mask_to_token_mask(
    patch_mask: np.ndarray,
    grid_h: int,
    grid_w: int,
    merge_size: int = 2,
    strategy: str = "any",
) -> np.ndarray:
    """
    Map patch-level keep mask to merged token-level mask.

    Args:
        patch_mask: Boolean mask at patch level, shape (grid_h * grid_w,)
        grid_h: Full height of patch grid (e.g., 48)
        grid_w: Full width of patch grid (e.g., 48)
        merge_size: Merge factor (e.g., 2 means 2x2 patches -> 1 token)
        strategy: How to combine patch decisions:
            - "any": Keep token if ANY of its patches are kept (most conservative, keeps more)
            - "all": Keep token only if ALL of its patches are kept (aggressive, removes more)
            - "majority": Keep token if more than half of its patches are kept

    Returns:
        Boolean mask at token level, shape (out_h * out_w,)
    """
    out_h = grid_h // merge_size
    out_w = grid_w // merge_size

    # Reshape to (out_h, merge_size, out_w, merge_size)
    patch_mask_2d = patch_mask.reshape(grid_h, grid_w)

    # Group by merge blocks
    patch_mask_grouped = patch_mask_2d.reshape(
        out_h, merge_size, out_w, merge_size
    ).transpose(0, 2, 1, 3)  # (out_h, out_w, merge_size, merge_size)

    # Apply strategy
    if strategy == "any":
        # Keep token if ANY patch in the 2x2 block is kept
        token_mask = patch_mask_grouped.any(axis=(2, 3))
    elif strategy == "all":
        # Keep token only if ALL patches in the 2x2 block are kept
        token_mask = patch_mask_grouped.all(axis=(2, 3))
    elif strategy == "majority":
        # Keep token if more than half of patches are kept
        keep_count = patch_mask_grouped.sum(axis=(2, 3))
        token_mask = keep_count > (merge_size * merge_size) // 2
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return token_mask.flatten()


def _get_select_mask_stp(
    patch_assign: np.ndarray,
    skip_ratio: float = 0.5,
    large_comp_threshold: int = 0,
) -> np.ndarray:
    """
    Get selection mask following STP's exact implementation with large component skip.

    Uses uniform sampling (linspace) to select tokens within each component.
    When large_comp_threshold > 0, components larger than the threshold are entirely skipped.

    Args:
        patch_assign: Component assignment array
        skip_ratio: Ratio of tokens to skip within each component
        large_comp_threshold: If > 0, skip entire components with size > this threshold.
            This is useful for removing large uniform background regions.

    Returns:
        Boolean mask where True = keep
    """
    retain_mask = np.zeros(len(patch_assign), dtype=bool)

    for comp in np.unique(patch_assign):
        positions = np.where(patch_assign == comp)[0]
        num_positions = len(positions)

        # Large component skip: if component is larger than threshold, skip entirely
        if large_comp_threshold > 0 and num_positions > large_comp_threshold:
            # Skip entire component - don't set any positions to True
            continue

        if num_positions == 1:
            # Single-patch components are always kept (UI elements)
            retain_mask[positions] = True
        else:
            # Use uniform sampling for multi-patch components
            num_to_skip = int(round(num_positions * skip_ratio))
            num_to_retain = max(1, num_positions - num_to_skip)
            indices = np.linspace(0, num_positions - 1, num_to_retain).astype(int)
            retain_mask[positions[indices]] = True

    return retain_mask


def compute_token_keep_mask(
    image_embeds: "torch.Tensor",
    grid_thw: "torch.Tensor",
    threshold: float = 50.0,
    skip_ratio: float = 0.5,
    large_comp_threshold: int = 0,
) -> "torch.Tensor":
    """
    Compute which image tokens to keep based on STP algorithm.

    NOTE: This function uses embeddings, which is NOT the original STP approach.
    For the correct implementation, use compute_token_keep_mask_from_pixels() instead.

    Returns a boolean mask where True = keep, False = remove.
    """
    if threshold <= 0.0:
        return torch.ones(image_embeds.shape[0], dtype=torch.bool, device=image_embeds.device)

    device = image_embeds.device
    embeddings_np = image_embeds.float().cpu().numpy()
    grid_thw_np = grid_thw.cpu().numpy()

    keep_list = []
    token_offset = 0

    for img_idx in range(len(grid_thw_np)):
        t, h, w = grid_thw_np[img_idx]
        num_tokens = t * h * w
        img_embeddings = embeddings_np[token_offset : token_offset + num_tokens]

        emb_reshaped = img_embeddings.reshape(t, h, w, -1)

        for t_idx in range(t):
            frame_emb = emb_reshaped[t_idx]
            frame_flat = frame_emb.reshape(h * w, -1)

            patch_assign = build_ui_graph(frame_flat, h, w, threshold)
            select_mask = get_select_mask(
                patch_assign, skip_ratio, large_comp_threshold, grid_h=h, grid_w=w
            )
            keep_list.append(select_mask)

        token_offset += num_tokens

    full_keep_mask = np.concatenate(keep_list, axis=0)
    return torch.from_numpy(full_keep_mask).to(device=device)
def _stp_ensure_keep_mask_at_least_one_per_frame(
    keep_mask: Optional["torch.Tensor"],
    grid_thw: Optional["torch.Tensor"],
    merge_size: int,
) -> Optional["torch.Tensor"]:
    """Ensure we never produce an empty frame (all-false) keep mask.

    Qwen3-VL vision attention uses per-frame variable length segments via `cu_seqlens`.
    If any segment length becomes 0 after pruning, FlashAttention/SDPA paths may error.
    """
    if keep_mask is None or grid_thw is None:
        return keep_mask

    if keep_mask.numel() == 0:
        return keep_mask

    # Clone to avoid in-place side effects for callers.
    keep_mask = keep_mask.clone()
    offset = 0
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        out_h, out_w = h // merge_size, w // merge_size
        tokens_per_frame = out_h * out_w
        # Reshape to [t, tokens_per_frame] for efficient per-frame operations
        if t * tokens_per_frame > 0:
            frame_mask = keep_mask[offset : offset + t * tokens_per_frame].view(t, tokens_per_frame)
            # Check which frames have no kept tokens (sum == 0)
            frame_sums = frame_mask.sum(dim=1)  # [t]
            empty_frames = (frame_sums == 0)  # [t]
            # For empty frames, set the first token to True
            if empty_frames.any():
                # Get indices of empty frames
                empty_indices = empty_frames.nonzero(as_tuple=False).squeeze(1)
                for fi in empty_indices.tolist():
                    keep_mask[offset + fi * tokens_per_frame] = True
        offset += t * tokens_per_frame

    return keep_mask


def _stp_qwen3vl_visual_forward_pruned(
    vision_module: "torch.nn.Module",
    pixel_values: "torch.Tensor",
    grid_thw: "torch.Tensor",
    merged_keep_mask: "torch.Tensor",
    debug_enabled: bool = False,
    **kwargs,
):
    """Run Qwen3VLVisionModel forward on a subset of tokens, then scatter back.

    - `merged_keep_mask` is at the *merged token* level (length = sum(t*h*w/merge^2)).
    - We expand it to patch-level (repeat_interleave(merge^2)) to filter `pixel_values`.
    - We compute attention on the kept patches only (recomputing `cu_seqlens`).
    - We then apply the merger on kept patches, and finally scatter merged outputs
      (and deepstack outputs) back to the full merged-token length.

    Args:
        debug_enabled: If True, write detailed logs to /tmp/ttp_forward_debug_rank{rank}.log
    """
    import os
    _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    _debug_rank0 = debug_enabled and _local_rank == 0

    # Log from ALL ranks to see if they're all taking the same path
    if debug_enabled:
        with open(f"/tmp/ttp_forward_debug_rank{_local_rank}.log", "a") as f:
            f.write(f"[_pruned] Enter (rank {_local_rank})\n")
            f.flush()

    merge_size = getattr(vision_module, "spatial_merge_size", getattr(vision_module.config, "spatial_merge_size", 2))
    merge_unit = merge_size * merge_size

    # Ensure at least 1 merged token is kept per temporal group.
    # Without this, cu_seqlens would have 0-length segments (causing flash-attn
    # crashes) or need a clamp that desynchronises cu_seqlens from hidden_states.
    merged_keep_mask = merged_keep_mask.to(device=pixel_values.device, dtype=torch.bool).clone()
    _offset = 0
    for _t, _h, _w in grid_thw.tolist():
        _t, _h, _w = int(_t), int(_h), int(_w)
        _tpf = (_h // merge_size) * (_w // merge_size)  # tokens per frame
        # Vectorised: check all frames of this video at once (1 GPU op, no per-frame sync)
        _vid_mask = merged_keep_mask[_offset : _offset + _t * _tpf].view(_t, _tpf)
        _frame_has_any = _vid_mask.any(dim=1)  # [t] bool tensor on GPU
        # Indices of the first token of each empty frame
        _first_tok_idx = _offset + torch.arange(_t, device=_vid_mask.device) * _tpf
        merged_keep_mask[_first_tok_idx[~_frame_has_any]] = True
        _offset += _t * _tpf

    # Expand merged keep mask to patch-level keep mask.
    patch_keep_mask = merged_keep_mask.repeat_interleave(merge_unit)
    kept_pixel_values = pixel_values[patch_keep_mask]

    if _debug_rank0:
        with open("/tmp/ttp_forward_debug.log", "a") as f:
            f.write(f"[_pruned] patch_embed...\n")

    # Patch embedding + positional embedding for kept patches.
    hidden_states = vision_module.patch_embed(kept_pixel_values)

    if _debug_rank0:
        with open("/tmp/ttp_forward_debug.log", "a") as f:
            f.write(f"[_pruned] pos_embed...\n")

    pos_embeds_full = vision_module.fast_pos_embed_interpolate(grid_thw)
    patch_keep_mask_pos = patch_keep_mask
    if patch_keep_mask_pos.device != pos_embeds_full.device:
        patch_keep_mask_pos = patch_keep_mask_pos.to(pos_embeds_full.device)
    hidden_states = hidden_states + pos_embeds_full[patch_keep_mask_pos]

    if _debug_rank0:
        with open("/tmp/ttp_forward_debug.log", "a") as f:
            f.write(f"[_pruned] rotary_embed...\n")

    # Rotary embeddings + per-frame cu_seqlens for variable-length vision attention.
    rotary_pos_emb_full = vision_module.rot_pos_emb(grid_thw)
    if patch_keep_mask_pos.device != rotary_pos_emb_full.device:
        patch_keep_mask_pos = patch_keep_mask_pos.to(rotary_pos_emb_full.device)
    rotary_pos_emb = rotary_pos_emb_full[patch_keep_mask_pos]

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    if _debug_rank0:
        with open("/tmp/ttp_forward_debug.log", "a") as f:
            f.write(f"[_pruned] cu_seqlens...\n")

    # Build `cu_seqlens` after pruning: one segment per frame.
    # Pre-compute per-frame kept counts using tensor operations to avoid .item() calls
    cu_seqlens_dtype = grid_thw.dtype if torch.jit.is_tracing() else torch.int32

    # Build cu_lengths using tensor operations
    cu_lengths_list: list[torch.Tensor] = []
    offset = 0
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        out_h, out_w = h // merge_size, w // merge_size
        tokens_per_frame = out_h * out_w
        # Get the mask slice for this video
        video_mask = merged_keep_mask[offset : offset + t * tokens_per_frame]
        # Reshape to [t, tokens_per_frame] and sum per frame
        video_mask_reshaped = video_mask.view(t, tokens_per_frame)
        kept_per_frame = video_mask_reshaped.sum(dim=1)  # [t]
        # No clamp needed: the mask guarantees ≥1 token per frame (see above).
        # Multiply by merge_unit to get patch counts
        cu_lengths_list.append(kept_per_frame * merge_unit)
        offset += t * tokens_per_frame

    # Concatenate all frame lengths and compute cumsum
    cu_lengths_tensor = torch.cat(cu_lengths_list).to(device=hidden_states.device, dtype=cu_seqlens_dtype)
    cu_seqlens = cu_lengths_tensor.cumsum(dim=0)
    cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

    if _debug_rank0:
        with open("/tmp/ttp_forward_debug.log", "a") as f:
            f.write(f"[_pruned] vision blocks (total={len(vision_module.blocks)})...\n")
            f.write(f"[_pruned] hidden_states.shape={hidden_states.shape}, cu_seqlens.shape={cu_seqlens.shape}\n")
            f.write(f"[_pruned] cu_seqlens[:10]={cu_seqlens[:10].tolist()}\n")
            f.write(f"[_pruned] rotary_pos_emb.shape={rotary_pos_emb.shape}\n")

    deepstack_feature_lists = []
    for i, blk in enumerate(vision_module.blocks):
        if _debug_rank0:
            with open("/tmp/ttp_forward_debug.log", "a") as f:
                f.write(f"[_pruned] Block {i} start...\n")
                f.flush()
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if _debug_rank0:
            with open("/tmp/ttp_forward_debug.log", "a") as f:
                f.write(f"[_pruned] Block {i} done\n")
                f.flush()
        if hasattr(vision_module, "deepstack_visual_indexes") and (i + 1) in vision_module.deepstack_visual_indexes:
            deepstack_feature_lists.append(hidden_states)

    if _debug_rank0:
        with open("/tmp/ttp_forward_debug.log", "a") as f:
            f.write(f"[_pruned] merger...\n")

    # Merge patches -> merged tokens for kept blocks only.
    merged_kept = vision_module.merger(hidden_states)

    # Deepstack merges
    deepstack_outputs_kept = []
    if deepstack_feature_lists:
        for ds_feat, ds_merger in zip(deepstack_feature_lists, vision_module.deepstack_merger_list):
            deepstack_outputs_kept.append(ds_merger(ds_feat))

    if _debug_rank0:
        with open("/tmp/ttp_forward_debug.log", "a") as f:
            f.write(f"[_pruned] scatter...\n")

    # Scatter kept merged tokens back to the full merged-token length.
    full_len = int(merged_keep_mask.numel())
    keep_indices = merged_keep_mask.nonzero(as_tuple=False).squeeze(1)

    merged_full = merged_kept.new_zeros((full_len, merged_kept.shape[-1]))
    merged_full[keep_indices] = merged_kept

    deepstack_full = []
    for ds in deepstack_outputs_kept:
        ds_full = ds.new_zeros((full_len, ds.shape[-1]))
        ds_full[keep_indices] = ds
        deepstack_full.append(ds_full)

    if _debug_rank0:
        with open("/tmp/ttp_forward_debug.log", "a") as f:
            f.write(f"[_pruned] Done\n")

    return merged_full, (deepstack_full if deepstack_full else None)



def patch_qwen2vl_forward_with_token_removal(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
) -> None:
    """
    Patch Qwen2VL forward to remove visual tokens AFTER position IDs are computed.

    This approach:
    1. Lets the model compute position_ids based on original grid
    2. After position_ids computation, removes selected image tokens from:
       - inputs_embeds
       - position_ids
       - attention_mask
    3. Then calls language_model.forward with reduced sequence

    Args:
        model: The pretrained VLM model
        model_args: Model arguments containing STP configuration
    """
    if not getattr(model_args, "use_stp", False):
        return

    threshold = getattr(model_args, "stp_threshold", 0.0)
    if threshold <= 0.0:
        return

    model_type = getattr(model.config, "model_type", None)
    if model_type not in ["qwen2_vl", "qwen2_5_vl"]:
        logger.warning_rank0(
            f"STP forward patch with token removal is only supported for Qwen2VL/Qwen2.5VL, "
            f"got: {model_type}"
        )
        return

    skip_ratio = getattr(model_args, "stp_skip_ratio", 0.5)
    large_comp_threshold = getattr(model_args, "stp_large_comp_threshold", 0)
    patch_level = getattr(model_args, "stp_patch_level", False)
    patch_to_token_strategy = getattr(model_args, "stp_patch_to_token_strategy", "any")
    temporal_aggregation = getattr(model_args, "stp_temporal_aggregation", "first")

    use_raw_frames_in_stp = getattr(model_args, "use_raw_frames_in_stp", False)

    # Get the model's internal model
    if hasattr(model, "model"):
        inner_model = model.model
    else:
        inner_model = model

    # Get image token ID from config
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        logger.warning_rank0("Cannot find image_token_id in config, STP token removal disabled")
        return

    # Store original forward method
    original_forward = inner_model.forward

    def patched_forward(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        cache_position=None,
        **kwargs,
    ):
        # Check for preprocessed STP info - this indicates a configuration mismatch
        stp_token_positions = kwargs.pop("stp_token_positions", None)
        stp_video_token_positions = kwargs.pop("stp_video_token_positions", None)
        if stp_token_positions is not None or stp_video_token_positions is not None:
            logger.warning_rank0(
                "STP forward_removal mode detected preprocessed STP data. "
                "Falling back to original forward without STP token removal."
            )
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                rope_deltas=rope_deltas,
                cache_position=cache_position,
                **kwargs,
            )
        # Handle defaults
        output_attentions = output_attentions if output_attentions is not None else inner_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else inner_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else inner_model.config.use_return_dict

        # IMPORTANT: disable KV-cache during training by default.
        # Some HF trainers call forward with `use_cache=None`. If we propagate None down
        # to the language model, it may fall back to config.use_cache=True which can
        # massively inflate memory usage for forward+backward.
        if use_cache is None:
            use_cache = False if inner_model.training else getattr(inner_model.config, "use_cache", True)

        # Step 1: Get text embeddings
        if inputs_embeds is None:
            inputs_embeds = inner_model.get_input_embeddings()(input_ids)

        # Track image token positions for later removal
        image_token_mask = None
        image_keep_mask = None

        # Step 2: Process images
        if pixel_values is not None:
            image_embeds = inner_model.get_image_features(pixel_values, image_grid_thw)
            image_embeds_cat = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

            # Get vision config parameters
            vision_config = getattr(inner_model.config, "vision_config", None)
            spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)
            patch_size = getattr(vision_config, "patch_size", 14)
            temporal_patch_size = getattr(vision_config, "temporal_patch_size", 2)

            # Compute keep mask using pixel values
            image_keep_mask = compute_token_keep_mask_from_pixels(
                pixel_values,
                image_grid_thw,
                threshold,
                skip_ratio,
                large_comp_threshold,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                merge_size=spatial_merge_size,
                patch_level=patch_level,
                patch_to_token_strategy=patch_to_token_strategy,
                temporal_aggregation=temporal_aggregation,
                use_raw_frames_in_stp=use_raw_frames_in_stp,
            )

            # Get placeholder mask
            image_mask, _ = inner_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds_cat
            )
            image_token_mask = image_mask.squeeze(-1)

            # Insert image embeddings
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_cat)

        # Step 3: Process videos
        video_token_mask = None
        video_keep_mask = None
        if pixel_values_videos is not None:
            video_embeds = inner_model.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds_cat = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)

            vision_config = getattr(inner_model.config, "vision_config", None)
            spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)
            patch_size = getattr(vision_config, "patch_size", 14)
            temporal_patch_size = getattr(vision_config, "temporal_patch_size", 2)

            video_keep_mask = compute_token_keep_mask_from_pixels(
                pixel_values_videos,
                video_grid_thw,
                threshold,
                skip_ratio,
                large_comp_threshold,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                merge_size=spatial_merge_size,
                patch_level=patch_level,
                patch_to_token_strategy=patch_to_token_strategy,
                temporal_aggregation=temporal_aggregation,
                use_raw_frames_in_stp=use_raw_frames_in_stp,
            )

            _, video_mask = inner_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds_cat
            )
            video_token_mask = video_mask[..., 0] if video_mask.dim() > 2 else video_mask.squeeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds_cat)

        # Step 4: Compute position IDs (based on original grid)
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

        # Step 5: Remove selected visual tokens AFTER position IDs computed
        should_remove_image = image_token_mask is not None and image_keep_mask is not None and not image_keep_mask.all()
        should_remove_video = video_token_mask is not None and video_keep_mask is not None and not video_keep_mask.all()

        if should_remove_image or should_remove_video:
            batch_size, seq_len = inputs_embeds.shape[:2]
            seq_keep_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=inputs_embeds.device)

            if should_remove_image:
                for b in range(batch_size):
                    img_positions = image_token_mask[b].nonzero(as_tuple=True)[0]
                    keep_idx = 0
                    for pos in img_positions:
                        if keep_idx < len(image_keep_mask):
                            seq_keep_mask[b, pos] = image_keep_mask[keep_idx]
                            keep_idx += 1

            if should_remove_video:
                for b in range(batch_size):
                    vid_positions = video_token_mask[b].nonzero(as_tuple=True)[0]
                    keep_idx = 0
                    for pos in vid_positions:
                        if keep_idx < len(video_keep_mask):
                            seq_keep_mask[b, pos] = video_keep_mask[keep_idx]
                            keep_idx += 1

            # Remove tokens from inputs_embeds, position_ids, attention_mask
            new_inputs_embeds_list = []
            new_position_ids_list = []
            new_attention_mask_list = []

            for b in range(batch_size):
                keep_indices = seq_keep_mask[b].nonzero(as_tuple=True)[0]
                new_inputs_embeds_list.append(inputs_embeds[b, keep_indices])
                new_position_ids_list.append(position_ids[:, b, keep_indices])
                if attention_mask is not None:
                    new_attention_mask_list.append(attention_mask[b, keep_indices])

            # Pad to same length and stack
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
                    emb = torch.nn.functional.pad(emb, (0, 0, 0, pad_len))
                    pos = torch.nn.functional.pad(pos, (0, pad_len))
                    if attention_mask is not None:
                        mask = new_attention_mask_list[b]
                        mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
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

            if cache_position is not None:
                cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)

        # Step 6: Call language model
        outputs = inner_model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModelOutputWithPast

        output = Qwen2VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=inner_model.rope_deltas,
        )
        return output if return_dict else output.to_tuple()

    inner_model.forward = patched_forward

    logger.info_rank0(
        f"STP forward patch with token removal applied (Qwen2VL): threshold={threshold}, "
        f"skip_ratio={skip_ratio}, large_comp_threshold={large_comp_threshold}"
    )



def patch_qwen3vl_forward_with_token_removal(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
) -> None:
    """
    Patch Qwen3VL forward to remove visual tokens AFTER position IDs are computed.

    This approach:
    1. Lets the model compute position_ids based on original grid
    2. After position_ids computation, removes selected image tokens from:
       - inputs_embeds
       - position_ids
       - attention_mask
    3. Then calls language_model.forward with reduced sequence

    Args:
        model: The pretrained VLM model
        model_args: Model arguments containing STP configuration
    """
    if not getattr(model_args, "use_stp", False):
        return

    threshold = getattr(model_args, "stp_threshold", 0.0)
    if threshold <= 0.0:
        return

    model_type = getattr(model.config, "model_type", None)
    if model_type not in ["qwen3_vl", "qwen3_vl_moe"]:
        logger.warning_rank0(
            f"STP Qwen3VL forward patch is only supported for Qwen3VL/Qwen3VL-MoE, got: {model_type}"
        )
        return

    skip_ratio = getattr(model_args, "stp_skip_ratio", 0.5)
    large_comp_threshold = getattr(model_args, "stp_large_comp_threshold", 0)
    patch_level = getattr(model_args, "stp_patch_level", False)
    patch_to_token_strategy = getattr(model_args, "stp_patch_to_token_strategy", "any")
    temporal_aggregation = getattr(model_args, "stp_temporal_aggregation", "first")

    use_raw_frames_in_stp = getattr(model_args, "use_raw_frames_in_stp", False)

    # Get the model's internal model
    if hasattr(model, "model"):
        inner_model = model.model
    else:
        inner_model = model

    # Get image token ID from config
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        logger.warning_rank0("Cannot find image_token_id in config, STP token removal disabled")
        return

    # Store original forward method
    original_forward = inner_model.forward

    # Patch vision encoder (Qwen3-VL) to *actually* skip computation for dropped tokens.
    # This is only used in forward_removal mode (token removal in LM) where keep masks
    # are computed from pixel_values.
    if not getattr(inner_model, "_stp_qwen3vl_vision_pruning_patched", False) and hasattr(inner_model, "visual"):
        inner_model._stp_qwen3vl_vision_pruning_patched = True

        original_visual_forward = inner_model.visual.forward
        original_get_image_features = inner_model.get_image_features
        original_get_video_features = getattr(inner_model, "get_video_features", None)

        def patched_visual_forward(pixel_values, grid_thw=None, **kwargs):
            stp_keep_mask = kwargs.pop("stp_keep_mask", None)
            if stp_keep_mask is None:
                return original_visual_forward(pixel_values, grid_thw=grid_thw, **kwargs)

            merged_keep_mask = stp_keep_mask.to(device=pixel_values.device, dtype=torch.bool)
            merge_size = getattr(
                inner_model.visual, "spatial_merge_size", getattr(inner_model.visual.config, "spatial_merge_size", 2)
            )
            merged_keep_mask = _stp_ensure_keep_mask_at_least_one_per_frame(merged_keep_mask, grid_thw, merge_size)

            # If nothing is pruned, fall back to original (avoids overhead / potential edge cases).
            # Note: Avoid .all().item() as it causes CUDA sync which can hang in distributed training
            if merged_keep_mask is None or merged_keep_mask.numel() == 0:
                return original_visual_forward(pixel_values, grid_thw=grid_thw, **kwargs)
            if merged_keep_mask.sum() == merged_keep_mask.numel():
                return original_visual_forward(pixel_values, grid_thw=grid_thw, **kwargs)

            # Sanity checks: mask lengths must match what the vision module expects.
            merge_unit = merge_size * merge_size
            expected_patches = int(torch.prod(grid_thw, dim=1).sum().item())
            expected_merged = int((torch.prod(grid_thw, dim=1) // merge_unit).sum().item())
            if pixel_values.shape[0] != expected_patches or merged_keep_mask.numel() != expected_merged:
                logger.warning_rank0(
                    "STP: keep mask length mismatch for Qwen3-VL vision pruning; "
                    f"pixel_values.shape[0]={pixel_values.shape[0]} (expected {expected_patches}), "
                    f"merged_keep_mask.numel()={merged_keep_mask.numel()} (expected {expected_merged}). "
                    "Falling back to original vision forward."
                )
                return original_visual_forward(pixel_values, grid_thw=grid_thw, **kwargs)

            return _stp_qwen3vl_visual_forward_pruned(
                inner_model.visual,
                pixel_values,
                grid_thw,
                merged_keep_mask,
                **kwargs,
            )

        def patched_get_image_features(pixel_values, grid_thw, stp_keep_mask=None):
            # Default path (no pruning)
            if stp_keep_mask is None:
                return original_get_image_features(pixel_values, grid_thw)

            # If the mask keeps everything, keep original path.
            # Note: Avoid .all().item() as it causes CUDA sync which can hang in distributed training
            if stp_keep_mask.numel() == 0 or stp_keep_mask.sum() == stp_keep_mask.numel():
                return original_get_image_features(pixel_values, grid_thw)

            pixel_values = pixel_values.type(inner_model.visual.dtype)
            image_embeds, deepstack_image_embeds = inner_model.visual(
                pixel_values, grid_thw=grid_thw, stp_keep_mask=stp_keep_mask
            )
            split_sizes = (grid_thw.prod(-1) // inner_model.visual.spatial_merge_size**2).tolist()
            image_embeds = torch.split(image_embeds, split_sizes)
            return image_embeds, deepstack_image_embeds

        def patched_get_video_features(pixel_values_videos, video_grid_thw, stp_keep_mask=None):
            if original_get_video_features is None:
                # Should not happen for Qwen3-VL, but keep a safe fallback.
                return original_get_image_features(pixel_values_videos, video_grid_thw)
            if stp_keep_mask is None:
                return original_get_video_features(pixel_values_videos, video_grid_thw)

            # Note: Avoid .all().item() as it causes CUDA sync which can hang in distributed training
            if stp_keep_mask.numel() == 0 or stp_keep_mask.sum() == stp_keep_mask.numel():
                return original_get_video_features(pixel_values_videos, video_grid_thw)

            pixel_values_videos = pixel_values_videos.type(inner_model.visual.dtype)
            video_embeds, deepstack_video_embeds = inner_model.visual(
                pixel_values_videos, grid_thw=video_grid_thw, stp_keep_mask=stp_keep_mask
            )
            split_sizes = (video_grid_thw.prod(-1) // inner_model.visual.spatial_merge_size**2).tolist()
            video_embeds = torch.split(video_embeds, split_sizes)
            return video_embeds, deepstack_video_embeds

        inner_model.visual.forward = patched_visual_forward
        inner_model.get_image_features = patched_get_image_features
        if original_get_video_features is not None:
            inner_model.get_video_features = patched_get_video_features

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

        # Check for preprocessed STP info - this indicates a configuration mismatch
        stp_token_positions = kwargs.pop("stp_token_positions", None)
        stp_video_token_positions = kwargs.pop("stp_video_token_positions", None)
        if stp_token_positions is not None or stp_video_token_positions is not None:
            logger.warning_rank0(
                "STP forward_removal mode detected preprocessed STP data. "
                "Falling back to original forward without STP token removal."
            )
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
        # Handle defaults - use language_model.config for use_cache as it's not in Qwen3VLConfig
        output_attentions = kwargs.get("output_attentions", inner_model.config.output_attentions)
        output_hidden_states = kwargs.get("output_hidden_states", inner_model.config.output_hidden_states)
        return_dict = kwargs.get("return_dict", inner_model.config.use_return_dict)
        use_cache = kwargs.get("use_cache", None)
        if use_cache is None:
            # IMPORTANT: disable KV-cache during training by default.
            # Our STP patch replaces Qwen3VL outer forward; if we accidentally keep
            # `use_cache=True` in training, memory can blow up (KV for every layer).
            use_cache = False if inner_model.training else getattr(inner_model.language_model.config, "use_cache", True)

        # Step 1: Get text embeddings
        if inputs_embeds is None:
            inputs_embeds = inner_model.get_input_embeddings()(input_ids)

        # Clear rope_deltas only at the start of a new sequence (not during generation steps)
        # During generation, rope_deltas must persist across forward calls.
        # We only clear when position_ids is None (step 4 will recompute it).
        # When position_ids is provided externally (e.g., from prepare_inputs_for_generation),
        # we preserve the existing rope_deltas so the step-5 correction can use it directly,
        # avoiding a redundant second call to get_rope_index.
        is_first_forward = cache_position is None or cache_position[0] == 0
        if is_first_forward and position_ids is None:
            if hasattr(inner_model, "rope_deltas") and inner_model.rope_deltas is not None:
                inner_model.rope_deltas = None

        image_token_mask = None
        image_keep_mask = None

        # Vision config parameters
        # NOTE: Qwen3-VL's effective merge size is defined by the vision tower.
        # Some checkpoints/configs may have stale values in `config.vision_config`.
        # Using the wrong merge size here breaks alignment between:
        #   - how many <image>/<video> placeholder tokens exist
        #   - how many merged vision tokens are produced
        #   - the STP keep-mask length (merged-token level)
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

        # Step 2: Process images (Qwen3VL get_image_features returns (image_embeds_tuple, deepstack_embeds))
        deepstack_image_embeds = None
        if pixel_values is not None:
            image_keep_mask = compute_token_keep_mask_from_pixels(
                pixel_values,
                image_grid_thw,
                threshold,
                skip_ratio,
                large_comp_threshold,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                merge_size=spatial_merge_size,
                patch_level=patch_level,
                patch_to_token_strategy=patch_to_token_strategy,
                temporal_aggregation=temporal_aggregation,
                use_raw_frames_in_stp=use_raw_frames_in_stp,
            )
            image_keep_mask = _stp_ensure_keep_mask_at_least_one_per_frame(
                image_keep_mask, image_grid_thw, spatial_merge_size
            )

            # Pass keep mask through to the vision encoder so it can skip computation.
            image_result = inner_model.get_image_features(
                pixel_values, image_grid_thw, stp_keep_mask=image_keep_mask
            )
            if isinstance(image_result, tuple) and len(image_result) == 2:
                image_embeds_tuple, deepstack_image_embeds = image_result
                if isinstance(image_embeds_tuple, (tuple, list)):
                    image_embeds_cat = torch.cat(image_embeds_tuple, dim=0)
                else:
                    image_embeds_cat = image_embeds_tuple
            elif hasattr(image_result, "pooler_output"):
                # BaseModelOutputWithDeepstackFeatures (new transformers dataclass return type)
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
            # Qwen3VL image_mask is (batch, seq_len, hidden_dim)
            image_token_mask = image_mask[..., 0]
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_cat)

        # Step 3: Process videos
        deepstack_video_embeds = None
        video_token_mask = None
        video_keep_mask = None
        if pixel_values_videos is not None:
            # Detect low-resolution videos (e.g., 360p) based on video_grid_thw
            # video_grid_thw contains [T, H, W] where H and W are PATCH counts (not token counts)
            # Token count per frame = (H * W) / (merge_size^2) = (H * W) / 4
            # 720p (1280x720): h=52, w=90 → tokens = 52*90/4 = 1170 tokens/frame
            # 360p (640x360): h=26, w=46 → tokens = 26*46/4 = 299 tokens/frame
            # We use a threshold of 500 tokens/frame to distinguish them
            skip_stp_for_video = False
            if video_grid_thw is not None:
                for i in range(video_grid_thw.shape[0]):
                    t, h, w = video_grid_thw[i].tolist()
                    # Token count = (h * w) / (spatial_merge_size^2)
                    tokens_per_frame = (h * w) // (spatial_merge_size * spatial_merge_size)
                    # If any video has low resolution (< 500 tokens/frame), skip STP
                    if tokens_per_frame < 500:
                        skip_stp_for_video = True
                        break

            # Only compute STP mask if not skipping
            if not skip_stp_for_video:
                video_keep_mask = compute_token_keep_mask_from_pixels(
                    pixel_values_videos,
                    video_grid_thw,
                    threshold,
                    skip_ratio,
                    large_comp_threshold,
                    patch_size=patch_size,
                    temporal_patch_size=temporal_patch_size,
                    merge_size=spatial_merge_size,
                    patch_level=patch_level,
                    patch_to_token_strategy=patch_to_token_strategy,
                    temporal_aggregation=temporal_aggregation,
                    use_raw_frames_in_stp=use_raw_frames_in_stp,
                )
                video_keep_mask = _stp_ensure_keep_mask_at_least_one_per_frame(
                    video_keep_mask, video_grid_thw, spatial_merge_size
                )

            video_result = inner_model.get_video_features(
                pixel_values_videos, video_grid_thw, stp_keep_mask=video_keep_mask
            )
            if isinstance(video_result, tuple) and len(video_result) == 2:
                video_embeds_tuple, deepstack_video_embeds = video_result
                if isinstance(video_embeds_tuple, (tuple, list)):
                    video_embeds_cat = torch.cat(video_embeds_tuple, dim=0)
                else:
                    video_embeds_cat = video_embeds_tuple
            elif hasattr(video_result, "pooler_output"):
                # BaseModelOutputWithDeepstackFeatures (new transformers dataclass return type)
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
            # video_mask is (batch, seq_len, hidden_dim) or (batch, seq_len, 1)
            video_token_mask = video_mask[..., 0] if video_mask.dim() > 2 else video_mask.squeeze(-1)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds_cat)

        # Build visual_pos_masks and deepstack_visual_embeds for language model
        visual_pos_masks = None
        deepstack_visual_embeds = None
        image_mask_in_visual = None
        video_mask_in_visual = None
        if image_token_mask is not None and video_token_mask is not None:
            visual_pos_masks = image_token_mask | video_token_mask
            if deepstack_image_embeds is not None and deepstack_video_embeds is not None:
                deepstack_visual_embeds = []
                image_mask_in_visual = image_token_mask[visual_pos_masks]
                video_mask_in_visual = video_token_mask[visual_pos_masks]
                for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                    embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                    embed_joint[image_mask_in_visual, :] = img_embed
                    embed_joint[video_mask_in_visual, :] = vid_embed
                    deepstack_visual_embeds.append(embed_joint)
        elif image_token_mask is not None:
            visual_pos_masks = image_token_mask
            deepstack_visual_embeds = deepstack_image_embeds
            if deepstack_image_embeds is not None:
                # The visual sequence length is determined by `visual_pos_masks` (placeholders),
                # not by deepstack length (which may be stale/misaligned if configs differ).
                image_mask_in_visual = torch.ones(
                    int(visual_pos_masks.sum().item()), dtype=torch.bool, device=inputs_embeds.device
                )
        elif video_token_mask is not None:
            visual_pos_masks = video_token_mask
            deepstack_visual_embeds = deepstack_video_embeds
            if deepstack_video_embeds is not None:
                video_mask_in_visual = torch.ones(
                    int(visual_pos_masks.sum().item()), dtype=torch.bool, device=inputs_embeds.device
                )

        # Step 4: Compute position IDs (based on original grid)
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

        # Step 5: Remove selected visual tokens AFTER position IDs computed
        should_remove_image = image_token_mask is not None and image_keep_mask is not None and not image_keep_mask.all()
        should_remove_video = video_token_mask is not None and video_keep_mask is not None and not video_keep_mask.all()

        if should_remove_image or should_remove_video:
            batch_size, seq_len = inputs_embeds.shape[:2]
            seq_keep_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=inputs_embeds.device)

            if should_remove_image:
                merge_size = spatial_merge_size
                tokens_per_image = (image_grid_thw.prod(dim=-1) // (merge_size * merge_size)).tolist()
                image_offsets = [0]
                for t in tokens_per_image[:-1]:
                    image_offsets.append(image_offsets[-1] + t)

                for b in range(batch_size):
                    img_positions = image_token_mask[b].nonzero(as_tuple=True)[0]
                    offset = image_offsets[b] if b < len(image_offsets) else 0
                    n = min(len(img_positions), max(0, len(image_keep_mask) - offset))
                    if n > 0:
                        seq_keep_mask[b, img_positions[:n]] = image_keep_mask[offset:offset + n]

            if should_remove_video:
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

            new_inputs_embeds_list = []
            new_position_ids_list = []
            new_attention_mask_list = []

            for b in range(batch_size):
                keep_indices = seq_keep_mask[b].nonzero(as_tuple=True)[0]
                new_inputs_embeds_list.append(inputs_embeds[b, keep_indices])
                new_position_ids_list.append(position_ids[:, b, keep_indices])
                if attention_mask is not None:
                    new_attention_mask_list.append(attention_mask[b, keep_indices])

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
                    emb = torch.nn.functional.pad(emb, (0, 0, 0, pad_len))
                    pos = torch.nn.functional.pad(pos, (0, pad_len))
                    if attention_mask is not None:
                        mask = new_attention_mask_list[b]
                        mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
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

            # Clean up intermediate lists to release memory
            del new_inputs_embeds_list, new_position_ids_list, new_attention_mask_list
            del padded_embeds, padded_positions, padded_masks

            if cache_position is not None:
                cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)

            # Fix rope_deltas to account for removed tokens.
            # During generation (predict_with_generate), the decode step computes the
            # next-token RoPE position as:  cache_position[0] + rope_deltas
            # After STP removes N tokens, the KV cache length is (seq_len - N), so
            # cache_position[0] = seq_len - N  instead of  seq_len.
            # Without correction, the generated token has a position N steps too early,
            # breaking RoPE attention and causing severe quality degradation.
            if inner_model.rope_deltas is not None:
                num_dropped_per_batch = (
                    seq_len - seq_keep_mask.sum(dim=1, keepdim=True)
                ).to(device=inner_model.rope_deltas.device, dtype=inner_model.rope_deltas.dtype)
                if inner_model.rope_deltas.shape[0] == 1 and batch_size > 1:
                    inner_model.rope_deltas = inner_model.rope_deltas.expand(batch_size, 1).contiguous()
                inner_model.rope_deltas = inner_model.rope_deltas + num_dropped_per_batch

            # store for outer forward label update (detach to prevent gradient accumulation)
            inner_model._stp_seq_keep_mask = seq_keep_mask.detach()
            inner_model._stp_max_len = max_len

            # Update visual_pos_masks to match new sequence length
            if visual_pos_masks is not None:
                new_visual_pos_masks_list = []
                for b in range(batch_size):
                    keep_indices = seq_keep_mask[b].nonzero(as_tuple=True)[0]
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

                # Update deepstack_visual_embeds to only include kept visual tokens
                if deepstack_visual_embeds is not None:
                    num_visual = deepstack_visual_embeds[0].shape[0]
                    visual_keep_mask = torch.ones(num_visual, dtype=torch.bool, device=inputs_embeds.device)

                    if image_mask_in_visual is not None and image_keep_mask is not None:
                        image_positions = image_mask_in_visual.nonzero(as_tuple=True)[0]
                        for i, pos in enumerate(image_positions):
                            if i < len(image_keep_mask):
                                visual_keep_mask[pos] = image_keep_mask[i]

                    if video_mask_in_visual is not None and video_keep_mask is not None:
                        video_positions = video_mask_in_visual.nonzero(as_tuple=True)[0]
                        for i, pos in enumerate(video_positions):
                            if i < len(video_keep_mask):
                                visual_keep_mask[pos] = video_keep_mask[i]

                    new_deepstack = []
                    for ds_embed in deepstack_visual_embeds:
                        new_deepstack.append(ds_embed[visual_keep_mask])
                    deepstack_visual_embeds = new_deepstack


        # Safety: deepstack_visual_embeds must align to `visual_pos_masks` token count.
        # If not, drop it to avoid shape mismatches inside Qwen3-VL attention.
        if deepstack_visual_embeds is not None and visual_pos_masks is not None:
            expected_visual = int(visual_pos_masks.sum().item())
            actual_visual = int(deepstack_visual_embeds[0].shape[0])
            if actual_visual != expected_visual:
                logger.warning_rank0(
                    "STP: deepstack_visual_embeds length mismatch (actual=%d, expected=%d). "
                    "Disabling deepstack for this forward to avoid shape errors.",
                    actual_visual,
                    expected_visual,
                )
                deepstack_visual_embeds = None

        # Step 6: Call language model
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

        output = Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=inner_model.rope_deltas,
        )
        return output if return_dict else output.to_tuple()

    inner_model.forward = patched_forward

    # Patch the outer model forward to update labels to match token-removed sequence.
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

        # Clear previous keep mask with explicit deletion to release memory
        for attr_name in ["_stp_seq_keep_mask", "_stp_max_len"]:
            if hasattr(inner_model, attr_name):
                old_val = getattr(inner_model, attr_name, None)
                if old_val is not None:
                    delattr(inner_model, attr_name)
        inner_model._stp_seq_keep_mask = None
        inner_model._stp_max_len = None

        # Also clear model-level attributes
        if hasattr(model, "_stp_modified_labels"):
            old_labels = getattr(model, "_stp_modified_labels", None)
            if old_labels is not None:
                del model._stp_modified_labels

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

        # Update labels if tokens were removed
        if labels is not None and getattr(inner_model, "_stp_seq_keep_mask", None) is not None:
            seq_keep_mask = inner_model._stp_seq_keep_mask
            max_len = inner_model._stp_max_len
            batch_size = labels.shape[0]
            labels_seq_len = labels.shape[1]
            mask_seq_len = seq_keep_mask.shape[1]

            if labels_seq_len != mask_seq_len:
                # Truncate or pad seq_keep_mask to match labels length
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

            # Clean up intermediate lists to release memory
            del new_labels_list
            del padded_labels

            # Clear seq_keep_mask reference on inner_model
            inner_model._stp_seq_keep_mask = None
            inner_model._stp_max_len = None

        logits_to_keep = kwargs.pop("logits_to_keep", 0)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = model.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = model.loss_function(logits=logits, labels=labels, vocab_size=model.config.text_config.vocab_size)

        # Store modified labels (detach to prevent gradient accumulation)
        if labels is not None:
            model._stp_modified_labels = labels.detach().clone()
        else:
            model._stp_modified_labels = None

        return_dict = kwargs.get("return_dict", getattr(model.config, "use_return_dict", True))
        output = Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )
        return output if return_dict else output.to_tuple()

    # Only patch once.
    if not getattr(model, "_stp_qwen3vl_outer_forward_patched", False):
        model._stp_qwen3vl_outer_forward_patched = True
        model.forward = patched_outer_forward

    logger.info_rank0(
        f"STP forward patch with token removal applied (Qwen3VL): threshold={threshold}, "
        f"skip_ratio={skip_ratio}, large_comp_threshold={large_comp_threshold}"
    )



def apply_stp_forward_patch(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
) -> None:
    """
    Apply STP forward patch based on model type.

    This is the main entry point for applying STP token removal.
    It detects the model type and applies the appropriate patch.

    Args:
        model: The pretrained VLM model
        model_args: Model arguments containing STP configuration
    """
    if not getattr(model_args, "use_stp", False):
        return

    threshold = getattr(model_args, "stp_threshold", 0.0)
    if threshold <= 0.0:
        return

    model_type = getattr(model.config, "model_type", None)

    if model_type in ["qwen2_vl", "qwen2_5_vl"]:
        patch_qwen2vl_forward_with_token_removal(model, model_args)
    elif model_type in ["qwen3_vl", "qwen3_vl_moe"]:
        patch_qwen3vl_forward_with_token_removal(model, model_args)
    else:
        logger.warning_rank0(
            f"STP forward patch is not supported for model type: {model_type}. "
            f"Supported types: qwen2_vl, qwen2_5_vl, qwen3_vl, qwen3_vl_moe"
        )


def patch_stp_qwen3vl_vision_encoder_with_pruning(
    model: "PreTrainedModel",
    model_args: Optional["ModelArguments"] = None,
) -> None:
    """
    Patch Qwen3VL vision encoder to accept stp_keep_mask parameter.

    This allows the vision encoder to skip computation for pruned tokens,
    which is useful when combining STP with TTP (Temporal Token Pruning).

    This function is designed to be called from ttp.py when both STP and TTP
    are enabled, to avoid applying the full STP forward patch which would
    conflict with the TTP forward patch.

    Debug Logging:
        When `debug_token_removal: true` is set in the training config YAML,
        detailed logs are written to /tmp/ttp_forward_debug_rank{rank}.log
        for each GPU rank. This is useful for debugging distributed training hangs.

        Example YAML config:
            debug_token_removal: true

        To view logs after training hangs:
            cat /tmp/ttp_forward_debug_rank*.log
    """
    inner_model = getattr(model, "model", model)

    # Skip if already patched
    if getattr(inner_model, "_stp_qwen3vl_vision_pruning_patched", False):
        return

    if not hasattr(inner_model, "visual"):
        logger.warning_rank0(
            "patch_stp_qwen3vl_vision_encoder_with_pruning: model has no 'visual' attribute, skipping."
        )
        return

    inner_model._stp_qwen3vl_vision_pruning_patched = True

    # Store debug flag on model for access in forward functions
    debug_token_removal = getattr(model_args, "debug_token_removal", False) if model_args else False
    inner_model._debug_token_removal = debug_token_removal

    original_visual_forward = inner_model.visual.forward
    original_get_image_features = inner_model.get_image_features
    original_get_video_features = getattr(inner_model, "get_video_features", None)

    def patched_visual_forward(pixel_values, grid_thw=None, **kwargs):
        import os
        _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        _debug_enabled = getattr(inner_model, "_debug_token_removal", False)

        stp_keep_mask = kwargs.pop("stp_keep_mask", None)

        merge_size = getattr(
            inner_model.visual, "spatial_merge_size", getattr(inner_model.visual.config, "spatial_merge_size", 2)
        )
        merge_unit = merge_size * merge_size

        # CRITICAL: In DeepSpeed ZeRO-3, all GPUs must execute the same code path.
        # If some GPUs have mask=None and others have mask, they will deadlock on all-gather.
        # Solution: When mask is None, create an all-True mask so all GPUs go through pruned forward.
        if stp_keep_mask is None:
            # Create an all-True mask (no pruning, but same code path)
            if grid_thw is not None:
                expected_merged = int((torch.prod(grid_thw, dim=1) // merge_unit).sum().item())
                stp_keep_mask = torch.ones(expected_merged, dtype=torch.bool, device=pixel_values.device)
                if _debug_enabled:
                    with open(f"/tmp/ttp_forward_debug_rank{_local_rank}.log", "a") as f:
                        f.write(f"[visual_forward] No mask provided, created all-True mask (rank {_local_rank})\n")
                        f.flush()
            else:
                # No grid_thw means no video, just call original
                if _debug_enabled:
                    with open(f"/tmp/ttp_forward_debug_rank{_local_rank}.log", "a") as f:
                        f.write(f"[visual_forward] No mask and no grid_thw, calling original (rank {_local_rank})\n")
                        f.flush()
                return original_visual_forward(pixel_values, grid_thw=grid_thw, **kwargs)

        # Log from ALL ranks when debug is enabled
        if _debug_enabled:
            with open(f"/tmp/ttp_forward_debug_rank{_local_rank}.log", "a") as f:
                f.write(f"[visual_forward] Has mask, calling pruned (rank {_local_rank})\n")
                f.flush()

        merged_keep_mask = stp_keep_mask.to(device=pixel_values.device, dtype=torch.bool)
        merged_keep_mask = _stp_ensure_keep_mask_at_least_one_per_frame(merged_keep_mask, grid_thw, merge_size)

        # Note: We no longer fall back to original_visual_forward even if all tokens are kept.
        # This ensures all GPUs execute the same code path for DeepSpeed ZeRO-3 compatibility.

        # Sanity checks - but don't fall back to original, just log warning
        expected_patches = int(torch.prod(grid_thw, dim=1).sum().item())
        expected_merged = int((torch.prod(grid_thw, dim=1) // merge_unit).sum().item())
        if pixel_values.shape[0] != expected_patches or merged_keep_mask.numel() != expected_merged:
            if _debug_enabled and _local_rank == 0:
                with open("/tmp/ttp_forward_debug.log", "a") as f:
                    f.write(f"[visual_forward] Mismatch! pixel_values={pixel_values.shape[0]}, expected={expected_patches}, mask={merged_keep_mask.numel()}, expected_merged={expected_merged}\n")
            logger.warning_rank0(
                "STP: keep mask length mismatch for Qwen3-VL vision pruning; "
                f"pixel_values.shape[0]={pixel_values.shape[0]} (expected {expected_patches}), "
                f"merged_keep_mask.numel()={merged_keep_mask.numel()} (expected {expected_merged}). "
                "Creating corrected mask."
            )
            # Create a corrected all-True mask instead of falling back
            merged_keep_mask = torch.ones(expected_merged, dtype=torch.bool, device=pixel_values.device)

        if _debug_enabled and _local_rank == 0:
            with open("/tmp/ttp_forward_debug.log", "a") as f:
                f.write(f"[visual_forward] Calling _stp_qwen3vl_visual_forward_pruned...\n")

        result = _stp_qwen3vl_visual_forward_pruned(
            inner_model.visual,
            pixel_values,
            grid_thw,
            merged_keep_mask,
            _debug_enabled,
            **kwargs,
        )

        if _debug_enabled and _local_rank == 0:
            with open("/tmp/ttp_forward_debug.log", "a") as f:
                f.write(f"[visual_forward] Done\n")

        return result

    def patched_get_image_features(pixel_values, grid_thw, stp_keep_mask=None):
        if stp_keep_mask is None:
            return original_get_image_features(pixel_values, grid_thw)

        # Note: Avoid .all().item() as it causes CUDA sync which can hang in distributed training
        # Use sum comparison instead
        if stp_keep_mask.numel() == 0 or stp_keep_mask.sum() == stp_keep_mask.numel():
            return original_get_image_features(pixel_values, grid_thw)

        pixel_values = pixel_values.type(inner_model.visual.dtype)
        image_embeds, deepstack_image_embeds = inner_model.visual(
            pixel_values, grid_thw=grid_thw, stp_keep_mask=stp_keep_mask
        )
        split_sizes = (grid_thw.prod(-1) // inner_model.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def patched_get_video_features(pixel_values_videos, video_grid_thw, stp_keep_mask=None):
        import os
        _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        _debug_rank0 = getattr(inner_model, "_debug_token_removal", False) and _local_rank == 0

        if _debug_rank0:
            with open("/tmp/ttp_forward_debug.log", "a") as f:
                f.write(f"[get_video_features] Enter, stp_keep_mask={stp_keep_mask is not None}\n")

        if original_get_video_features is None:
            if _debug_rank0:
                with open("/tmp/ttp_forward_debug.log", "a") as f:
                    f.write(f"[get_video_features] No original, using image features\n")
            return original_get_image_features(pixel_values_videos, video_grid_thw)
        if stp_keep_mask is None:
            if _debug_rank0:
                with open("/tmp/ttp_forward_debug.log", "a") as f:
                    f.write(f"[get_video_features] No mask, calling original\n")
            return original_get_video_features(pixel_values_videos, video_grid_thw)

        # Note: Avoid .all().item() as it causes CUDA sync which can hang in distributed training
        # Use sum comparison instead
        mask_sum = stp_keep_mask.sum()
        mask_numel = stp_keep_mask.numel()
        if _debug_rank0:
            with open("/tmp/ttp_forward_debug.log", "a") as f:
                f.write(f"[get_video_features] mask_sum={mask_sum}, mask_numel={mask_numel}\n")

        if mask_numel == 0 or mask_sum == mask_numel:
            if _debug_rank0:
                with open("/tmp/ttp_forward_debug.log", "a") as f:
                    f.write(f"[get_video_features] All kept, calling original\n")
            return original_get_video_features(pixel_values_videos, video_grid_thw)

        if _debug_rank0:
            with open("/tmp/ttp_forward_debug.log", "a") as f:
                f.write(f"[get_video_features] Calling inner_model.visual with pruning...\n")

        pixel_values_videos = pixel_values_videos.type(inner_model.visual.dtype)
        video_embeds, deepstack_video_embeds = inner_model.visual(
            pixel_values_videos, grid_thw=video_grid_thw, stp_keep_mask=stp_keep_mask
        )

        if _debug_rank0:
            with open("/tmp/ttp_forward_debug.log", "a") as f:
                f.write(f"[get_video_features] inner_model.visual done\n")

        split_sizes = (video_grid_thw.prod(-1) // inner_model.visual.spatial_merge_size**2).tolist()
        video_embeds = torch.split(video_embeds, split_sizes)

        if _debug_rank0:
            with open("/tmp/ttp_forward_debug.log", "a") as f:
                f.write(f"[get_video_features] Done\n")

        return video_embeds, deepstack_video_embeds

    inner_model.visual.forward = patched_visual_forward
    inner_model.get_image_features = patched_get_image_features
    if original_get_video_features is not None:
        inner_model.get_video_features = patched_get_video_features

    logger.info_rank0("STP vision encoder pruning patch applied (Qwen3VL, for TTP integration)")
