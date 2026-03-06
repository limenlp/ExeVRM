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

from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
from peft import PeftModel
from transformers import GenerationMixin, PreTrainedModel, PreTrainedTokenizerBase
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from ..extras import logging
from ..extras.misc import infer_optim_dtype
from ..extras.packages import is_transformers_version_greater_than
from .model_utils.attention import configure_attn_implementation, print_attn_implementation
from .model_utils.checkpointing import prepare_model_for_training
from .model_utils.embedding import resize_embedding_layer
from .model_utils.kv_cache import configure_kv_cache
from .model_utils.longlora import configure_longlora
from .model_utils.moe import add_z3_leaf_module, configure_moe
from .model_utils.packing import configure_packing
from .model_utils.quantization import configure_quantization
from .model_utils.rope import configure_rope
from .model_utils.stp import patch_stp_visual_encoder_with_masking
from .model_utils.stp import apply_stp_forward_patch
from .model_utils.valuehead import prepare_valuehead_model
from .model_utils.visual import autocast_projector_dtype, configure_visual_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer, ProcessorMixin
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import ModelArguments

if is_transformers_version_greater_than("4.57.0"):
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe


logger = logging.get_logger(__name__)


def patch_qwen3_omni_moe_thinker_text_sparse_moe_block():
    if is_transformers_version_greater_than("4.57.0") and not is_transformers_version_greater_than("4.58.0"):
        from .model_utils.moe import Qwen3OmniMoeThinkerTextSparseMoeBlock

        logger.warning_rank0(
            "You are using transformers with 4.x version, the Qwen3OmniMoeThinkerTextSparseMoeBlock will have some issues about deepspeed zero2 and fsdp2 training, so that we patched this model to avoid it. Transformers v5.0.0rc0 has fixed the issue, you can also try to update the transformers to using qwen3_omni. See more information on https://github.com/hiyouga/LLaMA-Factory/issues/9628."
        )

        modeling_qwen3_omni_moe.Qwen3OmniMoeThinkerTextSparseMoeBlock = Qwen3OmniMoeThinkerTextSparseMoeBlock


def _fix_attention_mask_shape(attention_mask, target_seq_len):
    """
    Fix attention_mask shape to match target sequence length.
    Handles both 2D (batch_size, seq_len) and 4D (batch_size, 1, seq_len, seq_len) masks.
    """
    if attention_mask is None:
        return None

    ndim = attention_mask.ndim
    if ndim == 2:
        # 2D mask: (batch_size, seq_len)
        mask_len = attention_mask.shape[-1]
        if mask_len == target_seq_len:
            return attention_mask
        if mask_len < target_seq_len:
            padding = torch.ones(
                attention_mask.shape[0],
                target_seq_len - mask_len,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            return torch.cat([attention_mask, padding], dim=-1)
        else:
            return attention_mask[:, :target_seq_len]
    elif ndim == 4:
        # 4D mask: (batch_size, 1, seq_len, seq_len) or (batch_size, num_heads, seq_len, seq_len)
        # The last two dimensions should match target_seq_len
        _, num_heads, h, w = attention_mask.shape
        if h == target_seq_len and w == target_seq_len:
            return attention_mask
        # Need to recreate the mask - just return None to let the model create a new one
        return None
    else:
        return attention_mask


def patch_qwen2_5_vl_generation():
    """
    Patch Qwen2.5-VL's prepare_inputs_for_generation to fix shape mismatch during generation.

    The bug occurs when attention_mask shape doesn't match input_ids shape during
    the generation (decoding) phase. This is a known issue in transformers library
    (GitHub issue #41093) that affects all versions up to 5.1.0.

    The fix completely rewrites prepare_inputs_for_generation to ensure all tensors
    have consistent shapes throughout the generation process.
    """
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        from transformers.generation.utils import GenerationMixin
    except ImportError:
        return  # Qwen2.5-VL not available

    # Skip if already patched
    if getattr(Qwen2_5_VLForConditionalGeneration, "_generation_patched", False):
        return

    def patched_prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        video_ttp_keep_mask=None,
        **kwargs,
    ):
        # Get target sequence length from input_ids
        target_seq_len = input_ids.shape[-1] if input_ids is not None else None

        # Fix: Ensure attention_mask matches input_ids shape BEFORE any processing
        if target_seq_len is not None:
            attention_mask = _fix_attention_mask_shape(attention_mask, target_seq_len)

        # Call grandparent's prepare_inputs_for_generation (GenerationMixin)
        model_inputs = GenerationMixin.prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        # Fix model_inputs attention_mask after GenerationMixin processing
        # Get the actual sequence length from model_inputs
        if "input_ids" in model_inputs and model_inputs["input_ids"] is not None:
            actual_seq_len = model_inputs["input_ids"].shape[-1]
        elif "inputs_embeds" in model_inputs and model_inputs["inputs_embeds"] is not None:
            actual_seq_len = model_inputs["inputs_embeds"].shape[1]
        else:
            actual_seq_len = target_seq_len

        if "attention_mask" in model_inputs and actual_seq_len is not None:
            model_inputs["attention_mask"] = _fix_attention_mask_shape(
                model_inputs["attention_mask"], actual_seq_len
            )

        # Qwen2-5-VL position_ids are prepared with rope_deltas
        if position_ids is None:
            if cache_position[0] == 0 or self.model.rope_deltas is None:
                # Use the fixed attention_mask for get_rope_index
                vision_positions, rope_deltas = self.model.get_rope_index(
                    model_inputs.get("input_ids", None),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,  # Use fixed attention_mask
                )
                self.model.rope_deltas = rope_deltas
            elif "position_ids" in model_inputs:
                batch_size, seq_length = model_inputs["position_ids"].shape
                device = model_inputs["position_ids"].device
                pos_ids = torch.arange(seq_length, device=device)
                pos_ids = pos_ids.view(1, 1, -1).expand(3, batch_size, -1)
                delta = cache_position[0] + self.model.rope_deltas
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                vision_positions = pos_ids + delta.expand_as(pos_ids)

            # Concatenate "text + vision" positions into [4, bs, seq-len]
            text_positions = model_inputs["position_ids"][None, ...]
            model_inputs["position_ids"] = torch.cat([text_positions, vision_positions], dim=0)

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["video_ttp_keep_mask"] = None
        elif video_ttp_keep_mask is not None:
            model_inputs["video_ttp_keep_mask"] = video_ttp_keep_mask

        return model_inputs

    Qwen2_5_VLForConditionalGeneration.prepare_inputs_for_generation = patched_prepare_inputs_for_generation
    Qwen2_5_VLForConditionalGeneration._generation_patched = True
    logger.info_rank0("Patched Qwen2.5-VL prepare_inputs_for_generation to fix attention_mask shape mismatch.")


def patch_qwen3_vl_generation():
    """
    Patch Qwen3-VL's prepare_inputs_for_generation to fix shape mismatch during generation.

    Same issue as Qwen2.5-VL - attention_mask shape doesn't match input_ids shape
    during the generation (decoding) phase. The bug occurs in get_rope_index where
    input_ids (trimmed to 1 token) is indexed with the full-length attention_mask.
    """
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        from transformers.generation.utils import GenerationMixin
    except ImportError:
        return  # Qwen3-VL not available

    # Skip if already patched
    if getattr(Qwen3VLForConditionalGeneration, "_generation_patched", False):
        return

    def patched_prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        video_ttp_keep_mask=None,
        **kwargs,
    ):
        # Get target sequence length from input_ids
        target_seq_len = input_ids.shape[-1] if input_ids is not None else None

        # Fix: Ensure attention_mask matches input_ids shape BEFORE any processing
        if target_seq_len is not None:
            attention_mask = _fix_attention_mask_shape(attention_mask, target_seq_len)

        # Save the 2D attention_mask BEFORE GenerationMixin potentially converts
        # it to a 4D causal mask (step 6).  patched_forward's Step 5 needs 2D.
        _original_2d_attn_mask = (
            attention_mask if attention_mask is not None and attention_mask.ndim == 2 else None
        )

        # ---- TTP Step-5 fix: realign attention_mask & cache_position ----
        # After TTP removes video tokens in patched_forward (Step 5), the KV cache
        # has fewer entries than the original sequence length.  But model_kwargs
        # still carries the original-length attention_mask and cache_position,
        # causing the model to attend to non-existent KV positions during decode.
        # This corrupts the output and prevents EOS generation.
        # Fix: rebuild attention_mask from the Step-5 shortened mask, and set
        # cache_position to the actual KV-cache length each decode step.
        if is_first_iteration:
            # Clear stale mask from previous batch
            if hasattr(self.model, '_ttp_step5_attn_mask'):
                delattr(self.model, '_ttp_step5_attn_mask')
        # Build the rebuilt mask for attention (but do NOT replace attention_mask yet —
        # GenerationMixin needs the ORIGINAL mask for correct position_ids via cumsum).
        _ttp_rebuilt_attn_mask = None
        if not is_first_iteration:
            _step5_mask = getattr(self.model, '_ttp_step5_attn_mask', None)
            if _step5_mask is not None and past_key_values is not None:
                _actual_cache_len = past_key_values.get_seq_length()
                _num_gen = _actual_cache_len + 1 - _step5_mask.shape[-1]
                if _num_gen > 0:
                    _gen_ones = torch.ones(
                        _step5_mask.shape[0], _num_gen,
                        dtype=_step5_mask.dtype, device=_step5_mask.device,
                    )
                    _ttp_rebuilt_attn_mask = torch.cat([_step5_mask, _gen_ones], dim=-1)
                else:
                    _ttp_rebuilt_attn_mask = _step5_mask

        # Call grandparent's prepare_inputs_for_generation (GenerationMixin)
        model_inputs = GenerationMixin.prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        # NOTE: Do NOT trim model_inputs["attention_mask"] to match the trimmed input_ids.
        # During decode with KV cache, GenerationMixin trims input_ids to 1 token but
        # attention_mask must remain at full sequence length so the model can attend to
        # all cached key-value entries. Trimming it to 1 would break KV cache attention
        # and force full recomputation at every decode step.

        # Qwen3-VL position_ids are prepared with rope_deltas
        if position_ids is None:
            is_prefill = cache_position[0] == 0

            # Step 1: Ensure rope_deltas is computed (using full input_ids)
            if is_prefill or self.model.rope_deltas is None:
                vision_positions, rope_deltas = self.model.get_rope_index(
                    input_ids,  # Use original untrimmed input_ids
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,  # Use fixed attention_mask (matches input_ids)
                )
                self.model.rope_deltas = rope_deltas

            # Step 2: Build vision_positions that match model_inputs["position_ids"] shape
            if not is_prefill and "position_ids" in model_inputs:
                # Decode phase: model_inputs has trimmed input (1 token),
                # so compute vision_positions from delta to match trimmed shape.
                batch_size, seq_length = model_inputs["position_ids"].shape
                device = model_inputs["position_ids"].device
                pos_ids = torch.arange(seq_length, device=device)
                pos_ids = pos_ids.view(1, 1, -1).expand(3, batch_size, -1)
                delta = cache_position[0] + self.model.rope_deltas
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                vision_positions = pos_ids + delta.expand_as(pos_ids)

            # Concatenate "text + vision" positions into [4, bs, seq-len]
            text_positions = model_inputs["position_ids"][None, ...]
            model_inputs["position_ids"] = torch.cat([text_positions, vision_positions], dim=0)

        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["video_ttp_keep_mask"] = None
        elif video_ttp_keep_mask is not None:
            model_inputs["video_ttp_keep_mask"] = video_ttp_keep_mask

        # Override attention_mask with the rebuilt TTP mask AFTER position_ids
        # are computed.  Position_ids were derived from the original (full-length)
        # attention_mask and are correct.  The rebuilt mask is shorter, matching
        # the actual KV-cache length, so the model only attends to real entries.
        if _ttp_rebuilt_attn_mask is not None:
            model_inputs["attention_mask"] = _ttp_rebuilt_attn_mask
        elif is_first_iteration and _original_2d_attn_mask is not None:
            # During prefill, GenerationMixin step 6 may have converted the 2D
            # mask to 4D.  patched_forward's Step 5 requires 2D for token
            # filtering, so restore the original 2D mask.
            if "attention_mask" in model_inputs and model_inputs["attention_mask"] is not None:
                if model_inputs["attention_mask"].ndim != 2:
                    model_inputs["attention_mask"] = _original_2d_attn_mask

        return model_inputs

    Qwen3VLForConditionalGeneration.prepare_inputs_for_generation = patched_prepare_inputs_for_generation
    Qwen3VLForConditionalGeneration._generation_patched = True
    logger.info_rank0("Patched Qwen3-VL prepare_inputs_for_generation to fix attention_mask shape mismatch.")


def patch_qwen2_vl_generation():
    """
    Patch Qwen2-VL's prepare_inputs_for_generation to fix shape mismatch during generation.

    Same issue as Qwen2.5-VL - attention_mask shape doesn't match input_ids shape
    during the generation (decoding) phase.
    """
    try:
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
        from transformers.generation.utils import GenerationMixin
        from torch._dynamo import is_compiling as is_torchdynamo_compiling
    except ImportError:
        return  # Qwen2-VL not available

    # Skip if already patched
    if getattr(Qwen2VLForConditionalGeneration, "_generation_patched", False):
        return

    def patched_prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        video_ttp_keep_mask=None,
        **kwargs,
    ):
        # Get target sequence length from input_ids
        target_seq_len = input_ids.shape[-1] if input_ids is not None else None

        # Fix: Ensure attention_mask matches input_ids shape BEFORE any processing
        if target_seq_len is not None:
            attention_mask = _fix_attention_mask_shape(attention_mask, target_seq_len)

        # Call grandparent's prepare_inputs_for_generation (GenerationMixin)
        model_inputs = GenerationMixin.prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        # Fix model_inputs attention_mask after GenerationMixin processing
        if "input_ids" in model_inputs and model_inputs["input_ids"] is not None:
            actual_seq_len = model_inputs["input_ids"].shape[-1]
        elif "inputs_embeds" in model_inputs and model_inputs["inputs_embeds"] is not None:
            actual_seq_len = model_inputs["inputs_embeds"].shape[1]
        else:
            actual_seq_len = target_seq_len

        if "attention_mask" in model_inputs and actual_seq_len is not None:
            model_inputs["attention_mask"] = _fix_attention_mask_shape(
                model_inputs["attention_mask"], actual_seq_len
            )

        # Qwen2-VL position_ids are prepared with rope_deltas
        if position_ids is None:
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
                vision_positions, rope_deltas = self.model.get_rope_index(
                    model_inputs.get("input_ids", None),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                )
                self.model.rope_deltas = rope_deltas
            elif "position_ids" in model_inputs:
                batch_size, seq_length = model_inputs["position_ids"].shape
                device = model_inputs["position_ids"].device
                pos_ids = torch.arange(seq_length, device=device)
                pos_ids = pos_ids.view(1, 1, -1).expand(3, batch_size, -1)
                delta = cache_position[0] + self.model.rope_deltas
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                vision_positions = pos_ids + delta.expand_as(pos_ids)

            # Concatenate "text + vision" positions into [4, bs, seq-len]
            text_positions = model_inputs["position_ids"][None, ...]
            model_inputs["position_ids"] = torch.cat([text_positions, vision_positions], dim=0)

        if model_inputs.get("cache_position") is not None and model_inputs["cache_position"][0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["video_ttp_keep_mask"] = None
        elif video_ttp_keep_mask is not None:
            model_inputs["video_ttp_keep_mask"] = video_ttp_keep_mask

        return model_inputs

    Qwen2VLForConditionalGeneration.prepare_inputs_for_generation = patched_prepare_inputs_for_generation
    Qwen2VLForConditionalGeneration._generation_patched = True
    logger.info_rank0("Patched Qwen2-VL prepare_inputs_for_generation to fix attention_mask shape mismatch.")


def patch_tokenizer(tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments") -> None:
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    if model_args.model_max_length is not None and int(tokenizer.model_max_length) < int(model_args.model_max_length):
        tokenizer.model_max_length = int(model_args.model_max_length)  # enlarge the tokenizer max length

    if model_args.add_tokens is not None:
        num_added_tokens = tokenizer.add_tokens(new_tokens=model_args.add_tokens, special_tokens=False)
        logger.info_rank0("Add tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New tokens have been added, changed `resize_vocab` to True.")

    if model_args.add_special_tokens is not None:
        num_added_special_tokens = tokenizer.add_tokens(new_tokens=model_args.add_special_tokens, special_tokens=True)
        logger.info_rank0(
            "Add special tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_special_tokens))
        )
        if num_added_special_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New special tokens have been added, changed `resize_vocab` to True.")


def patch_processor(
    processor: "ProcessorMixin",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
) -> None:
    setattr(processor, "tokenizer", tokenizer)
    setattr(processor, "image_max_pixels", model_args.image_max_pixels)
    setattr(processor, "image_min_pixels", model_args.image_min_pixels)
    setattr(processor, "image_do_pan_and_scan", model_args.image_do_pan_and_scan)
    setattr(processor, "crop_to_patches", model_args.crop_to_patches)
    setattr(processor, "video_max_pixels", model_args.video_max_pixels)
    setattr(processor, "video_min_pixels", model_args.video_min_pixels)
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is not None and hasattr(video_processor, "size"):
        # smart_resize compares (num_frames * H * W) against these limits,
        # so we scale per-frame pixels by video_maxlen to prevent stage-2 downscaling.
        video_processor.size = {
            "shortest_edge": model_args.video_min_pixels,
            "longest_edge": model_args.video_max_pixels * model_args.video_maxlen,
        }
    setattr(processor, "video_fps", model_args.video_fps)
    setattr(processor, "video_maxlen", model_args.video_maxlen)
    setattr(processor, "use_audio_in_video", model_args.use_audio_in_video)
    setattr(processor, "audio_sampling_rate", model_args.audio_sampling_rate)
    setattr(processor, "use_stp", model_args.use_stp)
    setattr(processor, "stp_threshold", model_args.stp_threshold)
    setattr(processor, "stp_skip_ratio", model_args.stp_skip_ratio)
    setattr(processor, "stp_large_comp_threshold", model_args.stp_large_comp_threshold)
    setattr(processor, "stp_mode", model_args.stp_mode)
    setattr(processor, "stp_patch_level", model_args.stp_patch_level)
    setattr(processor, "stp_patch_to_token_strategy", model_args.stp_patch_to_token_strategy)
    setattr(processor, "stp_temporal_aggregation", getattr(model_args, "stp_temporal_aggregation", "first"))
    # Temporal Token Pruning (TTP) parameters
    setattr(processor, "use_ttp", getattr(model_args, "use_ttp", False))
    setattr(processor, "ttp_threshold", getattr(model_args, "ttp_threshold", 0.9))
    setattr(processor, "ttp_min_run_length", getattr(model_args, "ttp_min_run_length", 2))
    setattr(processor, "ttp_similarity_metric", getattr(model_args, "ttp_similarity_metric", "cosine"))
    setattr(processor, "ttp_comparison_mode", getattr(model_args, "ttp_comparison_mode", "reference"))
    setattr(processor, "use_raw_frames_in_ttp", getattr(model_args, "use_raw_frames_in_ttp", False))
    setattr(processor, "use_raw_frames_in_stp", getattr(model_args, "use_raw_frames_in_stp", False))


def patch_config(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    init_kwargs: dict[str, Any],
    is_trainable: bool,
) -> None:
    if model_args.compute_dtype is None:  # priority: bf16 > fp16 > fp32
        if model_args.infer_dtype != "auto" and not is_trainable:
            model_args.compute_dtype = getattr(torch, model_args.infer_dtype)
        else:
            model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    configure_attn_implementation(config, model_args)
    configure_rope(config, model_args)
    configure_longlora(config, model_args, is_trainable)
    configure_quantization(config, tokenizer, model_args, is_trainable, init_kwargs)
    configure_moe(config, model_args, is_trainable)
    configure_visual_model(config)
    configure_packing(model_args, is_trainable)
    configure_kv_cache(config, model_args, is_trainable)

    if getattr(config, "model_type", None) == "qwen":
        setattr(config, "use_flash_attn", model_args.flash_attn == "fa2")
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, model_args.compute_dtype == dtype)

    if getattr(config, "model_type", None) == "minicpmo":
        setattr(config, "init_audio", True)
        setattr(config, "init_tts", False)

    # replace the top-k gating method
    if getattr(config, "model_type", None) == "kimi_vl" and is_trainable:
        setattr(config.text_config, "topk_method", "greedy")

    architectures = getattr(config, "architectures", None)
    if isinstance(architectures, list) and "InternVLChatModel" in architectures:
        raise ValueError(
            "Please download the internvl models in a Hugging Face–compatible format "
            "(for example, https://huggingface.co/OpenGVLab/InternVL3-8B-hf)."
        )

    if isinstance(architectures, list) and "LlavaLlamaForCausalLM" in architectures:
        raise ValueError("Please download llava models with hf-compatible format: https://huggingface.co/llava-hf")

    if getattr(config, "model_type", None) == "internlm3" and not is_transformers_version_greater_than("4.47.1"):
        raise RuntimeError("InternLM3 model requires transformers>=4.47.1, please upgrade it.")

    if getattr(config, "model_type", None) == "lfm2_vl" and not is_transformers_version_greater_than("4.58.0"):
        raise RuntimeError(
            "LFM2.5-VL model requires transformers>=4.58.0 or install from commit: "
            "pip install git+https://github.com/huggingface/transformers.git@3c2517727ce28a30f5044e01663ee204deb1cdbe"
        )

    if getattr(config, "model_type", None) == "qwen3_omni_moe":
        patch_qwen3_omni_moe_thinker_text_sparse_moe_block()

    # Fix Qwen2/Qwen2.5/Qwen3-VL generation bug (attention_mask shape mismatch during generation)
    # This bug exists in all transformers versions up to 5.1.0
    if getattr(config, "model_type", None) == "qwen3_vl":
        patch_qwen3_vl_generation()
    elif getattr(config, "model_type", None) == "qwen2_5_vl":
        patch_qwen2_5_vl_generation()
    elif getattr(config, "model_type", None) == "qwen2_vl":
        patch_qwen2_vl_generation()

    # deepspeed zero3 is not compatible with low_cpu_mem_usage
    init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage and (not is_deepspeed_zero3_enabled())

    # fsdp/deepspeed zero3 does not need device map
    if not (is_deepspeed_zero3_enabled() or is_fsdp_enabled()) and init_kwargs["low_cpu_mem_usage"]:
        if "device_map" not in init_kwargs and model_args.device_map:
            init_kwargs["device_map"] = model_args.device_map  # device map requires low_cpu_mem_usage=True

        if init_kwargs.get("device_map", None) == "auto":
            init_kwargs["offload_folder"] = model_args.offload_folder


def patch_model(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    is_trainable: bool,
    add_valuehead: bool,
) -> None:
    gen_config = model.generation_config  # check and fix generation config
    if not gen_config.do_sample and (
        (gen_config.temperature is not None and gen_config.temperature != 1.0)
        or (gen_config.top_p is not None and gen_config.top_p != 1.0)
        or (gen_config.typical_p is not None and gen_config.typical_p != 1.0)
    ):
        gen_config.do_sample = True

    if getattr(model.config, "model_type", None) not in ["minicpmv", "minicpmo"] and "GenerationMixin" not in str(
        model.generate.__func__
    ):
        model.generate = MethodType(GenerationMixin.generate, model)

    if add_valuehead:
        prepare_valuehead_model(model)

    if model_args.resize_vocab:
        resize_embedding_layer(
            model,
            tokenizer,
            new_special_tokens_config=getattr(model_args, "_special_token_descriptions", None),
            init_special_tokens=model_args.init_special_tokens,
        )

    if is_trainable:
        if getattr(model.config, "model_type", None) == "gemma3n":
            setattr(model_args, "disable_gradient_checkpointing", True)

        prepare_model_for_training(model, model_args)
        autocast_projector_dtype(model, model_args)
        add_z3_leaf_module(model)

    # Apply STP token selection based on mode
    # Note: If both STP (forward_removal mode) and TTP are enabled,
    # TTP patch handles both (to avoid conflicting forward patches)
    use_stp = getattr(model_args, "use_stp", False)
    use_ttp = getattr(model_args, "use_ttp", False)
    stp_mode = getattr(model_args, "stp_mode", "preprocess")

    if use_stp:
        if stp_mode == "masking":
            # Zero out skipped embeddings (preserves grid structure)
            patch_stp_visual_encoder_with_masking(model, model_args)
        elif stp_mode == "forward_removal" and not use_ttp:
            # Remove tokens after position IDs computed (true reduction, correct positions)
            # Skip this if TTP is also enabled - TTP patch will handle STP internally
            apply_stp_forward_patch(model, model_args)
        # "preprocess" mode is handled in data collator, no model patching needed

    # Apply Temporal Token Pruning (TTP) for temporal token reduction
    # TTP can work standalone or combined with STP (handles STP internally when both enabled)
    if use_ttp:
        from .model_utils.ttp import apply_ttp_forward_patch

        apply_ttp_forward_patch(model, model_args)

    if not model_args.use_unsloth:
        print_attn_implementation(model.config)

    try:
        model.add_model_tags(["llama-factory"])
    except Exception:
        logger.warning_rank0("Cannot properly tag the model.")


def patch_valuehead_model(model: "AutoModelForCausalLMWithValueHead") -> None:
    def tie_weights(self: "AutoModelForCausalLMWithValueHead") -> None:
        if isinstance(self.pretrained_model, PreTrainedModel):
            self.pretrained_model.tie_weights()

    def get_input_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_input_embeddings()

    def get_output_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_output_embeddings()

    def create_or_update_model_card(self: "AutoModelForCausalLMWithValueHead", output_dir: str) -> None:
        if isinstance(self.pretrained_model, PeftModel):
            self.pretrained_model.create_or_update_model_card(output_dir)

    def get_rope_index_func(self: "AutoModelForCausalLMWithValueHead"):
        if isinstance(self.pretrained_model, PeftModel):
            base_model = self.pretrained_model.base_model.model
        else:
            base_model = self.pretrained_model

        if base_model and hasattr(base_model, "get_rope_index"):
            return base_model.get_rope_index
        elif base_model and hasattr(base_model, "model") and hasattr(base_model.model, "get_rope_index"):
            return base_model.model.get_rope_index
        else:
            return None

    ignore_modules = [name for name, _ in model.named_parameters() if "pretrained_model" in name]
    setattr(model, "_keys_to_ignore_on_save", ignore_modules)
    setattr(model, "tie_weights", MethodType(tie_weights, model))
    setattr(model, "get_input_embeddings", MethodType(get_input_embeddings, model))
    setattr(model, "get_output_embeddings", MethodType(get_output_embeddings, model))
    setattr(model, "get_rope_index", get_rope_index_func(model))
    setattr(model, "create_or_update_model_card", MethodType(create_or_update_model_card, model))
