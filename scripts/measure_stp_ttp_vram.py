"""Measure peak GPU VRAM savings from STP / TTP for Qwen3-VL.

It loads defaults from a LLaMA-Factory YAML training config and runs 1 step
(forward+backward) on a single video, comparing 4 settings:

  - Baseline:      STP=off, TTP=off
  - STP only:   STP=on,  TTP=off
  - TTP only:      STP=off, TTP=on
  - STP + TTP:  STP=on,  TTP=on

Example:
  python scripts/measure_stp_ttp_vram.py \
    --config qwen3vl_8B_rm.yaml \
    --video /export/home/AgentNet/ubuntu_videos_720p/0022f50b-7983-4c7a-9005-01a6d924c152_success.mp4
"""

import argparse
import gc
import time
from types import SimpleNamespace

import torch


def _reset_cuda_stats():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _mb(x: int) -> float:
    return x / 1024 / 1024


def _mem_now_mb():
    torch.cuda.synchronize()
    return _mb(torch.cuda.memory_allocated()), _mb(torch.cuda.memory_reserved())


def _mem_peak_mb():
    torch.cuda.synchronize()
    return _mb(torch.cuda.max_memory_allocated()), _mb(torch.cuda.max_memory_reserved())


def _safe_div(n: float, d: float) -> float:
    if d == 0 or d != d:
        return float("nan")
    return n / d


def _pct_saved(base: float, cur: float) -> float:
    if base == 0 or base != base:
        return float("nan")
    return (base - cur) / base * 100.0


def _print_vs(label: str, base: dict, cur: dict):
    if not (base.get("ok") and cur.get("ok")):
        print(f"- {label}: skipped ratio (ok={cur.get('ok')})")
        return

    r_ratio = _safe_div(cur["peak_reserved"], base["peak_reserved"])
    a_ratio = _safe_div(cur["peak_alloc"], base["peak_alloc"])
    dt_ratio = _safe_div(cur["dt"], base["dt"])  # <1 means faster
    speedup = _safe_div(base["dt"], cur["dt"])   # >1 means faster

    print(
        f"- {label}: "
        f"reserved {cur['peak_reserved']:.1f}MB ({r_ratio:.3f}x, saved {base['peak_reserved'] - cur['peak_reserved']:.1f}MB, {_pct_saved(base['peak_reserved'], cur['peak_reserved']):.1f}%) | "
        f"allocated {cur['peak_alloc']:.1f}MB ({a_ratio:.3f}x, saved {base['peak_alloc'] - cur['peak_alloc']:.1f}MB, {_pct_saved(base['peak_alloc'], cur['peak_alloc']):.1f}%) | "
        f"runtime {cur['dt'] * 1000:.2f}ms (dt_ratio {dt_ratio:.3f}x, speedup {speedup:.2f}x)"
    )


def _load_yaml(path: str) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load --config.") from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _maybe_freeze_qwen3vl(model, freeze_vision_tower: bool, freeze_mm_projector: bool, freeze_language_model: bool):
    if not (freeze_vision_tower or freeze_mm_projector or freeze_language_model):
        return
    vision_keys = ("visual.patch_embed", "visual.blocks")
    projector_key = "visual.merger"
    language_keys = ("language_model", "lm_head")
    for name, p in model.named_parameters():
        if freeze_vision_tower and any(k in name for k in vision_keys):
            p.requires_grad_(False)
        if freeze_mm_projector and projector_key in name:
            p.requires_grad_(False)
        if freeze_language_model and any(k in name for k in language_keys):
            p.requires_grad_(False)


def _build_inputs(processor, video_path: str, fps: float, max_frames: int, max_pixels: int):
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": fps, "max_frames": max_frames, "max_pixels": max_pixels},
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    return processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt", **video_kwargs)


def _apply_patches(model, cfg: dict, enable_stp: bool, enable_ttp: bool):
    from llamafactory.model.model_utils.ttp import apply_ttp_forward_patch
    from llamafactory.model.model_utils.stp import apply_stp_forward_patch

    model_args = SimpleNamespace(
        use_stp=enable_stp,
        stp_mode=cfg.get("stp_mode", "forward_removal"),
        stp_threshold=cfg.get("stp_threshold", 0.0),
        stp_skip_ratio=cfg.get("stp_skip_ratio", 0.5),
        stp_large_comp_threshold=cfg.get("stp_large_comp_threshold", 0),
        stp_patch_level=cfg.get("stp_patch_level", False),
        stp_patch_to_token_strategy=cfg.get("stp_patch_to_token_strategy", "any"),
        stp_temporal_aggregation=cfg.get("stp_temporal_aggregation", "first"),
        use_raw_frames_in_stp=cfg.get("use_raw_frames_in_stp", False),
        use_ttp=enable_ttp,
        ttp_threshold=cfg.get("ttp_threshold", 0.9),
        ttp_min_run_length=cfg.get("ttp_min_run_length", 2),
        ttp_similarity_metric=cfg.get("ttp_similarity_metric", "cosine"),
    )

    if enable_ttp:
        apply_ttp_forward_patch(model, model_args)  # supports combined STP+TTP
    elif enable_stp:
        apply_stp_forward_patch(model, model_args)
    return model_args


def _print_keep_stats(model):
    inner = model.model if hasattr(model, "model") else model
    # Sequence-level masks (length = original seq_len); True = kept
    for tag, attr in [("STP", "_stp_seq_keep_mask"), ("TTP", "_ttp_seq_keep_mask")]:
        m = getattr(inner, attr, None)
        if m is None:
            continue
        kept = m.sum(dim=1).tolist()
        total = [int(m.shape[1])] * int(m.shape[0])
        removed = [t - k for t, k in zip(total, kept)]
        print(f"{tag} seq keep/total per sample: {list(zip(kept, total))} (removed={removed})")

    # Video-token-level masks (length = total video tokens across batch); True = kept
    def _print_1d_mask(tag: str, m):
        if m is None:
            return
        m = m.to(dtype=torch.bool)
        kept = int(m.sum().item())
        total = int(m.numel())
        removed = total - kept
        ratio = kept / total if total > 0 else 1.0
        print(f"{tag}: keep={kept}/{total} (removed={removed}, keep_ratio={ratio:.4f})")

    _print_1d_mask("STP video mask", getattr(inner, "_stp_video_keep_mask", None))
    _print_1d_mask("TTP   video mask (raw)", getattr(inner, "_ttp_video_keep_mask_raw", None))
    _print_1d_mask("TTP   video mask (final)", getattr(inner, "_ttp_video_keep_mask", None))


def run_one(cfg: dict, video_path: str, enable_stp: bool, enable_ttp: bool):
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float16
    _reset_cuda_stats()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=dtype,
        device_map="cuda",
        trust_remote_code=cfg.get("trust_remote_code", True),
    )
    processor = Qwen3VLProcessor.from_pretrained(cfg["model_name_or_path"], trust_remote_code=cfg.get("trust_remote_code", True))

    model.train()
    _apply_patches(model, cfg, enable_stp, enable_ttp)
    _maybe_freeze_qwen3vl(
        model,
        bool(cfg.get("freeze_vision_tower", False)),
        bool(cfg.get("freeze_multi_modal_projector", False)),
        bool(cfg.get("freeze_language_model", False)),
    )

    inputs = _build_inputs(
        processor,
        video_path,
        float(cfg.get("video_fps", 1.0)),
        int(cfg.get("video_maxlen", 32)),
        int(cfg.get("video_max_pixels", cfg.get("image_max_pixels", 921600))),
    ).to("cuda")

    labels = inputs["input_ids"].clone()
    pad_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
    if pad_id is not None:
        labels[labels == pad_id] = -100
    inputs["labels"] = labels

    _reset_cuda_stats()
    base_alloc, base_reserved = _mem_now_mb()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        out = model(**inputs)
        out.loss.backward()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.synchronize()
        peak_alloc, peak_reserved = _mem_peak_mb()
        print("[OOM] CUDA out of memory during forward/backward.")
        _print_keep_stats(model)
        del model, processor, inputs, labels
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "ok": False,
            "base_alloc": base_alloc,
            "base_reserved": base_reserved,
            "peak_alloc": peak_alloc,
            "peak_reserved": peak_reserved,
            "dt": float("nan"),
        }
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak_alloc, peak_reserved = _mem_peak_mb()

    _print_keep_stats(model)
    del model, processor, inputs, out, labels
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "ok": True,
        "base_alloc": base_alloc,
        "base_reserved": base_reserved,
        "peak_alloc": peak_alloc,
        "peak_reserved": peak_reserved,
        "dt": dt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="qwen3vl_8B_rm.yaml")
    ap.add_argument("--video", type=str, required=True)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)

    print(f"Config: {args.config}")
    print(f"Model:  {cfg.get('model_name_or_path')}")
    print(f"Video:  {args.video}")
    print(
        "Video sampling: "
        f"fps={cfg.get('video_fps')}, max_frames={cfg.get('video_maxlen')}, max_pixels={cfg.get('video_max_pixels')}"
    )
    print(f"Config flags: use_stp={cfg.get('use_stp')}, use_ttp={cfg.get('use_ttp')}")

    runs = [
        ("Baseline", False, False),
        ("STP only", True, False),
        ("TTP only", False, True),
        ("STP+TTP", True, True),
    ]

    results = {}
    for name, en_stp, en_ttp in runs:
        print(f"\n=== {name} (STP={'on' if en_stp else 'off'}, TTP={'on' if en_ttp else 'off'}) ===")
        results[name] = run_one(cfg, args.video, enable_stp=en_stp, enable_ttp=en_ttp)

    base = results["Baseline"]
    stp_only = results["STP only"]
    ttp_only = results["TTP only"]
    both = results["STP+TTP"]

    print("\n=== SUMMARY (1-step train: forward+backward) ===")
    print("\nRaw peaks:")
    for name in ["Baseline", "STP only", "TTP only", "STP+TTP"]:
        r = results[name]
        print(
            f"- {name:10s}: ok={r['ok']}, "
            f"peak_reserved={r['peak_reserved']:.1f}MB, peak_alloc={r['peak_alloc']:.1f}MB, dt={r['dt'] * 1000:.2f}ms"
        )

    print("\nRatios vs Baseline:")
    _print_vs("STP only", base, stp_only)
    _print_vs("TTP only", base, ttp_only)
    _print_vs("STP+TTP", base, both)

    print("\nKey pairwise comparisons:")
    _print_vs("STP+TTP vs TTP only", ttp_only, both)
    _print_vs("STP+TTP vs STP only", stp_only, both)


if __name__ == "__main__":
    main()

