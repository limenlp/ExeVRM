# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
import re
import uuid
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .metric import ComputeVideoRewardAccuracy


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        data_args: Optional["DataArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        generating_args: Optional["GeneratingArguments"] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        self.model_args = model_args
        self.data_args = data_args
        self.generating_args = generating_args
        self.processor = processor
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        elif finetuning_args.use_eaft_loss:
            from ..trainer_utils import eaft_loss_func

            self.compute_loss_func = lambda outputs, labels, num_items_in_batch=None: eaft_loss_func(
                outputs, labels, num_items_in_batch, finetuning_args.eaft_alpha
            )

        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)


        # Persistent vLLM engine for sleep/wake mode
        self._vllm_engine = None
        self._vllm_engine_sleeping = False
        self._expandable_segments_was_enabled = False  # track runtime state across sleep/wake

        self._train_video_reward_metric: Optional[ComputeVideoRewardAccuracy] = None
        # Keep O(1) memory: accumulate correct/total counts instead of storing a growing buffer.
        self._train_video_reward_correct_count: int = 0
        self._train_video_reward_total_count: int = 0
        self._train_prediction_samples: list[dict[str, Any]] = []
        self._max_prediction_samples: int = 8
        self._enable_prediction_logging = self._check_wandb_table_support()
        if finetuning_args.compute_video_reward_cumulative_accuracy:
            tokenizer_for_metric = getattr(self, "processing_class", None)
            if tokenizer_for_metric is None:
                tokenizer_for_metric = kwargs.get("tokenizer")
            if tokenizer_for_metric is not None:
                self._train_video_reward_metric = ComputeVideoRewardAccuracy(tokenizer=tokenizer_for_metric)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        # Use PairedShuffleSampler for paired_interleave mode
        if (
            self.data_args is not None
            and self.data_args.mix_strategy == "paired_interleave"
            and self.data_args.paired_interleave_group_size is not None
        ):
            from ...data.sampler import DistributedPairedShuffleSampler, PairedShuffleSampler

            group_size = self.data_args.paired_interleave_group_size
            dataset_size = len(self.train_dataset)
            logger.info_rank0(
                f"[paired_interleave] Creating sampler: dataset_size={dataset_size}, "
                f"group_size={group_size}, world_size={self.args.world_size}, "
                f"rank={self.args.process_index}"
            )
            if self.args.world_size > 1:
                sampler = DistributedPairedShuffleSampler(
                    self.train_dataset,
                    group_size=group_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    shuffle=True,
                    seed=self.args.seed,
                    drop_last=self.args.dataloader_drop_last,
                )
                logger.info_rank0(
                    f"[paired_interleave] DistributedPairedShuffleSampler created: "
                    f"len(sampler)={len(sampler)}, "
                    f"num_groups={sampler._num_groups}, "
                    f"groups_per_replica={sampler._num_groups_per_replica}, "
                    f"expected_steps (batch=1, accum=2)={len(sampler) // 2}"
                )
                return sampler
            else:
                sampler = PairedShuffleSampler(
                    self.train_dataset,
                    group_size=group_size,
                )
                logger.info_rank0(
                    f"[paired_interleave] PairedShuffleSampler created: len={len(sampler)}"
                )
                return sampler

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def get_train_dataloader(self):
        """Override to handle paired_interleave sampler correctly.

        For paired_interleave mode, we bypass accelerator.prepare() because our
        DistributedPairedShuffleSampler already handles distribution across GPUs.
        Using accelerator.prepare() would add BatchSamplerShard on top, causing
        double-sharding and reducing the step count to 1/N where N = num_GPUs.

        We cache the DataLoader to avoid creating new workers on each call,
        which can cause "Too many open files" errors with high num_workers.
        """
        # Check if we need special handling for paired_interleave
        use_paired_interleave = (
            self.data_args is not None
            and self.data_args.mix_strategy == "paired_interleave"
            and self.data_args.paired_interleave_group_size is not None
            and self.args.world_size > 1
        )

        if not use_paired_interleave:
            # Use default behavior for non-paired_interleave cases
            return super().get_train_dataloader()

        # Return cached dataloader if available (avoids creating new workers)
        if hasattr(self, "_paired_interleave_train_dataloader") and self._paired_interleave_train_dataloader is not None:
            return self._paired_interleave_train_dataloader

        # For paired_interleave with distributed training, create DataLoader manually
        # to bypass accelerator.prepare() which would add unwanted BatchSamplerShard
        from functools import partial
        from torch.utils.data import DataLoader

        from ...data.sampler import DistributedPairedShuffleSampler

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        # Remove unused columns if using datasets library
        try:
            from datasets import Dataset as HFDataset
            if isinstance(train_dataset, HFDataset):
                train_dataset = self._remove_unused_columns(train_dataset, description="Training")
        except ImportError:
            pass

        # Create our custom distributed sampler and cache it for epoch updates
        group_size = self.data_args.paired_interleave_group_size
        sampler = DistributedPairedShuffleSampler(
            train_dataset,
            group_size=group_size,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            shuffle=True,
            seed=self.args.seed,
            drop_last=self.args.dataloader_drop_last,
        )
        self._paired_interleave_sampler = sampler

        logger.info_rank0(
            f"[paired_interleave] Creating DataLoader WITHOUT accelerator.prepare(): "
            f"dataset_size={len(train_dataset)}, group_size={group_size}, "
            f"world_size={self.args.world_size}, rank={self.args.process_index}, "
            f"sampler_len={len(sampler)}, expected_steps={len(sampler) // (self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps)}"
        )

        # Create DataLoader with our custom sampler (no accelerator.prepare())
        from transformers.trainer_utils import seed_worker

        # Handle persistent_workers: only enable if num_workers > 0
        num_workers = self.args.dataloader_num_workers
        persistent_workers = self.args.dataloader_persistent_workers if num_workers > 0 else False
        # prefetch_factor is only valid when num_workers > 0
        prefetch_factor = self.args.dataloader_prefetch_factor if num_workers > 0 else None

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "sampler": sampler,
            "collate_fn": data_collator,
            "num_workers": num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": persistent_workers,
            "drop_last": self.args.dataloader_drop_last,
            "worker_init_fn": partial(
                seed_worker, num_workers=num_workers, rank=self.args.process_index
            ) if num_workers > 0 else None,
        }
        # Only add prefetch_factor if num_workers > 0
        if prefetch_factor is not None:
            dataloader_params["prefetch_factor"] = prefetch_factor

        dataloader = DataLoader(train_dataset, **dataloader_params)

        # Add set_epoch method to the dataloader so Trainer can update sampler epoch
        # This is needed because we bypass accelerator.prepare() which normally adds this
        def set_epoch(epoch: int):
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
        dataloader.set_epoch = set_epoch

        # Cache the dataloader to avoid recreating workers
        self._paired_interleave_train_dataloader = dataloader

        logger.info_rank0(
            f"[paired_interleave] DataLoader created and cached: "
            f"len(dataloader)={len(dataloader)}, batch_size={self._train_batch_size}"
        )

        return dataloader

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        return_outputs = kwargs.pop("return_outputs", False)
        loss, outputs = super().compute_loss(model, inputs, *args, return_outputs=True, **kwargs)

        # # Debug: Print TTP/STP detection status every 50 steps (only on rank 0)
        # if self.state.global_step % 1 == 0:
        #     try:
        #         import os
        #         if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        #             unwrapped = self.accelerator.unwrap_model(model) if hasattr(self, "accelerator") else model
        #             inner = getattr(unwrapped, "model", unwrapped)
        #             detected_low_res = getattr(inner, "_ttp_detected_low_res", None)
        #             tokens_per_frame = getattr(inner, "_ttp_tokens_per_frame", None)
        #             eff_ttp = getattr(inner, "_ttp_effective_use_ttp", None)
        #             eff_stp = getattr(inner, "_ttp_effective_use_stp", None)
        #             import sys
        #             sys.stderr.write(
        #                 f"[Step {self.state.global_step}] TTP/STP: "
        #                 f"low_res={detected_low_res}, tokens/frame={tokens_per_frame}, "
        #                 f"use_ttp={eff_ttp}, use_stp={eff_stp}\n"
        #             )
        #             sys.stderr.flush()
        #     except Exception:
        #         pass  # Silently ignore any errors

        if self._train_video_reward_metric is not None and "labels" in inputs and outputs is not None:
            logits = outputs.get("logits") if isinstance(outputs, dict) else getattr(outputs, "logits", None)
            if logits is not None:
                # Check if TTP updated the labels (labels may have been shortened due to token removal)
                # Get the underlying model (unwrap DeepSpeed/FSDP wrappers if needed)
                unwrapped_model = self.accelerator.unwrap_model(model) if hasattr(self, "accelerator") else model
                ttp_updated_labels = getattr(unwrapped_model, "_ttp_updated_labels", None)
                if ttp_updated_labels is not None:
                    # Use the updated labels that match the logits shape
                    inputs_for_metric = {**inputs, "labels": ttp_updated_labels}
                    # Clear the stored labels to avoid stale data
                    unwrapped_model._ttp_updated_labels = None
                else:
                    inputs_for_metric = inputs
                self._accumulate_train_video_reward_accuracy(logits, inputs_for_metric)
        if return_outputs:
            return loss, outputs
        return loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        # Use self._gen_kwargs if gen_kwargs is empty (same logic as Seq2SeqTrainer.prediction_step)
        # This must be done BEFORE adding synced_gpus, otherwise the len(gen_kwargs) == 0 check
        # in the parent class will fail and self._gen_kwargs won't be used.
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()

        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
            # Remove video_metadata as it's not a valid argument for model.generate()
            inputs.pop("video_metadata", None)
            # NOTE: Keep _per_video_use_stp and _per_video_use_ttp in inputs.
            # They flow through generate() → prepare_inputs_for_generation() → model.forward()
            # → patched_forward() where they're consumed via kwargs.pop().
            # Without them, TTP/STP defaults to globally enabled, ignoring per-dataset
            # use_stp/use_ttp=false settings in dataset_info.json.
            # Set synced_gpus=True for DDP distributed training to avoid deadlock.
            # In DDP mode, different GPUs may generate sequences of different lengths,
            # which can cause deadlock during gather operations if synced_gpus is not set.
            # transformers only sets this automatically for DeepSpeed Zero3 and FSDP.
            if self.args.world_size > 1 and "synced_gpus" not in gen_kwargs:
                gen_kwargs["synced_gpus"] = True
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

            # Log decoded output on rank 0
            import os
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                for i in range(generated_tokens.size(0)):
                    tokens = generated_tokens[i]
                    tokens = tokens[tokens != self.processing_class.pad_token_id]
                    text = self.processing_class.decode(tokens, skip_special_tokens=False)
                    print(f"[TTP decode] sample {i} ({len(tokens)} tokens): {text[:300]}")

        return loss, generated_tokens, labels

    @override
    def evaluate(
        self,
        eval_dataset: Optional["Dataset"] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
        **kwargs,
    ) -> dict[str, float]:
        if self.finetuning_args.use_vllm_eval and self.args.predict_with_generate:
            return self._vllm_evaluate(eval_dataset, metric_key_prefix)
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix, **kwargs)

    def _vllm_evaluate(
        self,
        eval_dataset: Optional["Dataset"],
        metric_key_prefix: str,
    ) -> dict[str, float]:
        """Dispatch to sleep/wake or legacy vLLM evaluation."""
        if self.finetuning_args.vllm_eval_sleep_mode:
            return self._vllm_evaluate_sleep_wake(eval_dataset, metric_key_prefix)
        return self._vllm_evaluate_legacy(eval_dataset, metric_key_prefix)

    @staticmethod
    def _fix_saved_config_for_vllm(saved_dir: str) -> None:
        """Fix config.json saved by ``save_pretrained`` for vLLM compatibility.

        transformers 4.57.2 has a bug in its mistral-regex check inside
        ``_from_pretrained``: it accesses ``_config.model_type`` on a plain
        dict loaded via ``json.load()``.  Setting ``transformers_version`` to
        a value > 4.57.2 makes the check short-circuit and skip the buggy
        code path entirely.
        """
        config_path = os.path.join(saved_dir, "config.json")
        if not os.path.isfile(config_path):
            return
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        config["transformers_version"] = "99.0.0"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def _create_vllm_engine(self, model_path: str):
        """Create a vLLM engine with sleep mode support.

        The caller (``_vllm_evaluate_sleep_wake``) is responsible for tearing
        down and restoring ``torch.distributed`` — see the comments there.
        This method only handles vLLM-specific env-var isolation.
        """
        from vllm import LLM

        # --- Isolate env vars so vLLM does not pick up training settings ---
        _DIST_ENV_VARS = [
            "MASTER_ADDR", "MASTER_PORT",
            "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
            "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
            "TORCHELASTIC_RUN_ID",
        ]
        saved_env: dict[str, str] = {}
        for var in _DIST_ENV_VARS:
            if var in os.environ:
                saved_env[var] = os.environ.pop(var)

        # Run EngineCore in-process (avoid subprocess that inherits env vars).
        saved_mp = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        # Bust the vllm.envs cache so the new value is picked up.
        try:
            import vllm.envs as _vllm_envs
            if hasattr(_vllm_envs, "disable_envs_cache"):
                _vllm_envs.disable_envs_cache()
        except Exception:
            pass

        # vLLM's CuMemAllocator (required for sleep mode) is incompatible
        # with PyTorch's expandable_segments.  Disable it at both the env-var
        # level (checked by vLLM's assertion) and at the runtime level
        # (checked by torch.cuda.MemPool).
        #
        # Following verl's pattern, expandable_segments stays DISABLED for the
        # entire lifetime of the vLLM engine.  The caller
        # (_vllm_evaluate_sleep_wake) re-enables it after engine sleep.
        saved_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments:True" in saved_alloc_conf:
            new_conf = saved_alloc_conf.replace("expandable_segments:True", "expandable_segments:False")
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = new_conf
        elif "expandable_segments" not in saved_alloc_conf:
            prefix = saved_alloc_conf + "," if saved_alloc_conf else ""
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = prefix + "expandable_segments:False"

        # Also disable at the PyTorch runtime allocator level.
        # Track the original state so the caller can restore it later.
        try:
            snap = torch.cuda.memory._snapshot()
            if snap.get("allocator_settings", {}).get("expandable_segments", False):
                self._expandable_segments_was_enabled = True
                self._set_expandable_segments(False)
                logger.info("vLLM eval: disabled expandable_segments at runtime for CuMemAllocator compatibility")
        except Exception:
            pass

        # Each rank should use its own GPU.  vLLM's UniProcExecutor derives
        # local_rank from ``DeviceConfig.device``.  By default this is just
        # ``torch.device("cuda")`` (no index → local_rank 0).  We temporarily
        # patch ``DeviceConfig.__post_init__`` so it preserves the GPU index.
        local_rank = torch.cuda.current_device()

        from vllm.config.device import DeviceConfig as _DC
        _orig_post_init = _DC.__post_init__

        def _patched_post_init(self_dc):
            self_dc.device_type = "cuda"
            self_dc.device = torch.device(f"cuda:{local_rank}")

        _DC.__post_init__ = _patched_post_init

        try:
            max_model_len = self.finetuning_args.vllm_eval_max_model_len
            # Load tokenizer from the original model directory (not the temp
            # weight dir) to avoid huggingface_hub validation errors on local
            # paths like /dev/shm.  The tokenizer never changes during training.
            tokenizer_path = self.model_args.model_name_or_path if self.model_args else model_path
            # Dynamically cap gpu_memory_utilization based on actual free
            # memory.  DeepSpeed / FSDP may leave optimizer states or
            # buffers on GPU that we couldn't fully offload.
            configured_gpu_util = self.finetuning_args.vllm_eval_gpu_util
            free_mem, total_mem = torch.cuda.mem_get_info()
            # Reserve headroom for vision-encoder workspace (embedding
            # 88K patches needs ~2-3 GiB), attention workspace, and
            # fragmentation.  8 GiB is safe for Qwen3-VL-8B at 720p.
            headroom_bytes = 8 * (2**30)  # 8 GiB
            max_vllm_bytes = max(0, free_mem - headroom_bytes)
            max_safe_util = max_vllm_bytes / total_mem
            effective_gpu_util = min(configured_gpu_util, max(0.20, max_safe_util))
            logger.info(
                "vLLM eval: gpu_memory_utilization=%.2f (configured=%.2f, "
                "free=%.1f GiB, headroom=%.0f GiB)",
                effective_gpu_util, configured_gpu_util,
                free_mem / 2**30, headroom_bytes / 2**30,
            )

            engine_kwargs = {
                "model": model_path,
                "tokenizer": tokenizer_path,
                "trust_remote_code": True,
                "tensor_parallel_size": self.finetuning_args.vllm_eval_tp_size or 1,
                "gpu_memory_utilization": effective_gpu_util,
                "max_model_len": max_model_len,
                "max_num_batched_tokens": max_model_len,
                "enforce_eager": self.finetuning_args.vllm_eval_enforce_eager,
                "disable_log_stats": True,
                "enable_sleep_mode": True,
                "limit_mm_per_prompt": {"image": 4, "video": 2, "audio": 2},
                "allowed_local_media_path": "/",
                "media_io_kwargs": {
                    "video": {
                        "num_frames": getattr(self.model_args, "video_maxlen", 128) if self.model_args else 128,
                    },
                },
            }
            engine = LLM(**engine_kwargs)
        finally:
            _DC.__post_init__ = _orig_post_init
            # Restore env vars for training distributed group.
            os.environ.update(saved_env)
            if saved_mp is not None:
                os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = saved_mp
            else:
                os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
            # Restore the CUDA allocator env-var so it doesn't leak into
            # subprocesses.  The *runtime* setting stays disabled — the
            # caller re-enables expandable_segments after engine sleep.
            if saved_alloc_conf:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = saved_alloc_conf
            else:
                os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

        return engine

    def _load_raw_eval_data(self) -> list[dict[str, Any]]:
        """Load raw eval data directly from JSON/JSONL files.

        Bypasses LLaMA-Factory's data pipeline entirely.  Returns a list
        of dicts ready for ``llm.chat()``, each containing:

        - ``messages``: OpenAI-style chat messages with video_url content parts
        - ``label``:    ground-truth text (e.g. ``\\box{correct}``)
        - ``video_path``: original video file path
        - ``user_instruction``: extracted user task text
        - ``split``:    dataset split name (osworld / scalecua / …)
        """
        if hasattr(self, "_vllm_raw_eval_data"):
            return self._vllm_raw_eval_data

        raw_data: list[dict[str, Any]] = []
        if not self.data_args or not self.data_args.eval_dataset:
            self._vllm_raw_eval_data = raw_data
            return raw_data

        from ...data.parser import get_dataset_list

        eval_attrs = get_dataset_list(self.data_args.eval_dataset, self.data_args.dataset_dir)
        for attr in eval_attrs:
            if attr.load_from != "file":
                continue
            path = os.path.join(self.data_args.dataset_dir, attr.dataset_name)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))

            for item in data:
                # --- video path & split ---
                videos = item.get("videos", [])
                video_path = videos[0] if videos else ""
                split = self._detect_split(video_path)

                # --- conversations → messages + label ---
                human_text = ""
                label = ""
                for conv in item.get("conversations", []):
                    if conv.get("from") == "human":
                        human_text = conv.get("value", "")
                    elif conv.get("from") == "gpt":
                        label = conv.get("value", "")

                # Build OpenAI-style content parts.
                # Remove the ``<video>`` placeholder from text; pass video
                # as a structured content part so vLLM loads it directly.
                text_clean = human_text.replace("<video>", "").strip()
                content_parts: list[dict[str, Any]] = []
                if video_path:
                    content_parts.append({
                        "type": "video_url",
                        "video_url": {"url": f"file://{video_path}"},
                    })
                content_parts.append({"type": "text", "text": text_clean})

                messages: list[dict[str, Any]] = [
                    {"role": "user", "content": content_parts},
                ]

                # --- user instruction ---
                instruction = ""
                marker = "# User Task\n"
                idx = human_text.find(marker)
                if idx != -1:
                    instruction = human_text[idx + len(marker):].strip()
                else:
                    instruction = text_clean

                raw_data.append({
                    "messages": messages,
                    "label": label,
                    "video_path": video_path,
                    "user_instruction": instruction,
                    "split": split,
                })

        self._vllm_raw_eval_data = raw_data
        logger.info("vLLM eval: loaded %d raw eval samples from dataset files", len(raw_data))
        return raw_data

    @staticmethod
    def _convert_weight_keys(
        state_dict: dict[str, "torch.Tensor"], model: "torch.nn.Module"
    ) -> dict[str, "torch.Tensor"]:
        """Convert HF internal parameter names to checkpoint-format names.

        Newer HuggingFace models may rename parameters internally via
        ``_checkpoint_conversion_mapping``.  vLLM's ``load_weights`` expects
        checkpoint-format names, so we reverse the mapping (same as verl).
        """
        if not hasattr(model, "_checkpoint_conversion_mapping"):
            return state_dict

        reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}
        converted: dict[str, "torch.Tensor"] = {}
        for key, value in state_dict.items():
            for pattern, replacement in reverse_key_mapping.items():
                replacement = replacement.lstrip("^")
                replacement = re.sub(r"\(.*\)", "", replacement)
                key, n_replace = re.subn(pattern, replacement, key)
                if n_replace > 0:
                    break
            converted[key] = value
        return converted

    def _sync_weights_to_vllm(self, unwrapped_model) -> None:
        """Transfer weight tensors directly to vLLM engine (verl-style).

        Instead of serializing to safetensors on disk/shm and reloading,
        we pass the state_dict tensors directly to vLLM's ``reload_weights``
        via ``collective_rpc``.  vLLM's ``model.load_weights()`` handles the
        HF-checkpoint-name → vLLM-internal-name mapping and tensor-parallel
        sharding in-place.
        """
        state_dict = unwrapped_model.state_dict()
        state_dict = self._convert_weight_keys(state_dict, unwrapped_model)
        weights_list = list(state_dict.items())
        del state_dict

        self._vllm_engine.collective_rpc(
            "reload_weights",
            kwargs={"weights_iterator": weights_list, "is_checkpoint_format": True},
        )
        self._vllm_engine.reset_prefix_cache()

    def _offload_deepspeed_states_to_cpu(self, rank: int = 0) -> None:
        """Move DeepSpeed ZeRO-2 internal GPU buffers to CPU.

        DeepSpeed ZeRO Stage 2 stores data in internal flat buffers that are
        invisible to ``model.to("cpu")``.  We must move them explicitly:

        1. ``fp16_groups_flat`` – full FP16 model copy per GPU
        2. ``single_partition_of_fp32_groups`` – partitioned FP32 master weights
        3. ``optimizer.state`` – partitioned FP32 momentum + variance
        """
        if not hasattr(self.model, "optimizer") or self.model.optimizer is None:
            return

        zero_opt = self.model.optimizer
        offloaded = 0

        try:
            # 1. FP16 flat parameter groups (full model replica per GPU)
            if hasattr(zero_opt, "fp16_groups_flat"):
                for buf in zero_opt.fp16_groups_flat:
                    if isinstance(buf, torch.Tensor) and buf.is_cuda:
                        buf.data = buf.data.cpu()
                        offloaded += buf.numel() * buf.element_size()

            # Also handle bit16_groups (newer DeepSpeed versions)
            if hasattr(zero_opt, "bit16_groups_flat"):
                for buf in zero_opt.bit16_groups_flat:
                    if isinstance(buf, torch.Tensor) and buf.is_cuda:
                        buf.data = buf.data.cpu()
                        offloaded += buf.numel() * buf.element_size()

            # 2. FP32 master weights (partitioned across ranks)
            if hasattr(zero_opt, "single_partition_of_fp32_groups"):
                for group in zero_opt.single_partition_of_fp32_groups:
                    for i, p in enumerate(group):
                        if isinstance(p, torch.Tensor) and p.is_cuda:
                            group[i] = p.cpu()
                            offloaded += p.numel() * p.element_size()

            # Also handle fp32_groups_flat_partition
            if hasattr(zero_opt, "fp32_groups_flat_partition"):
                for buf in zero_opt.fp32_groups_flat_partition:
                    if isinstance(buf, torch.Tensor) and buf.is_cuda:
                        buf.data = buf.data.cpu()
                        offloaded += buf.numel() * buf.element_size()

            # 3. Underlying optimizer states (momentum + variance)
            inner_opt = getattr(zero_opt, "optimizer", zero_opt)
            if hasattr(inner_opt, "state"):
                for state_vals in inner_opt.state.values():
                    if not isinstance(state_vals, dict):
                        continue
                    for k, v in state_vals.items():
                        if isinstance(v, torch.Tensor) and v.is_cuda:
                            state_vals[k] = v.cpu()
                            offloaded += v.numel() * v.element_size()

            # 4. Gradient buffers
            if hasattr(zero_opt, "grad_partitions_flat_buffer"):
                buf = zero_opt.grad_partitions_flat_buffer
                if isinstance(buf, torch.Tensor) and buf.is_cuda:
                    buf.data = buf.data.cpu()
                    offloaded += buf.numel() * buf.element_size()

            logger.info(
                "vLLM eval [sleep/wake]: rank %d — offloaded %.1f GiB of DeepSpeed state to CPU",
                rank, offloaded / 2**30,
            )
        except Exception as e:
            logger.warning("vLLM eval: DeepSpeed offload partial (freed %.1f GiB): %s", offloaded / 2**30, e)

    def _restore_deepspeed_states_to_gpu(self, device: "torch.device", rank: int = 0) -> None:
        """Move DeepSpeed ZeRO-2 internal buffers back to GPU after vLLM eval.

        IMPORTANT: after moving ``bit16_groups_flat`` back to GPU we must
        rebuild the parameter views via ``_update_model_bit16_weights``.
        The offload step (``buf.data = buf.data.cpu()``) broke the view
        relationship between model parameters and the flat buffer.  Without
        rebuilding, the subsequent ``unwrapped.to(device)`` would allocate
        **separate** GPU tensors for every parameter, doubling model memory
        (~16 GiB for an 8B model) and causing OOM.
        """
        if not hasattr(self.model, "optimizer") or self.model.optimizer is None:
            return

        zero_opt = self.model.optimizer
        restored = 0

        try:
            if hasattr(zero_opt, "fp16_groups_flat"):
                for buf in zero_opt.fp16_groups_flat:
                    if isinstance(buf, torch.Tensor) and not buf.is_cuda:
                        buf.data = buf.data.to(device)
                        restored += buf.numel() * buf.element_size()

            if hasattr(zero_opt, "bit16_groups_flat"):
                for buf in zero_opt.bit16_groups_flat:
                    if isinstance(buf, torch.Tensor) and not buf.is_cuda:
                        buf.data = buf.data.to(device)
                        restored += buf.numel() * buf.element_size()

            # Rebuild parameter-to-flat-buffer views.  DeepSpeed's
            # _update_model_bit16_weights() calls unflatten() which
            # creates views (not copies) of the flat buffer and
            # reassigns them to each model parameter's .data.
            if hasattr(zero_opt, "_update_model_bit16_weights"):
                n_groups = len(
                    getattr(zero_opt, "bit16_groups",
                            getattr(zero_opt, "fp16_groups", []))
                )
                for gi in range(n_groups):
                    zero_opt._update_model_bit16_weights(gi)
                logger.info(
                    "vLLM eval [sleep/wake]: rank %d — rebuilt %d param-group "
                    "views into flat buffers", rank, n_groups,
                )

            if hasattr(zero_opt, "single_partition_of_fp32_groups"):
                for group in zero_opt.single_partition_of_fp32_groups:
                    for i, p in enumerate(group):
                        if isinstance(p, torch.Tensor) and not p.is_cuda:
                            group[i] = p.to(device)
                            restored += p.numel() * p.element_size()

            if hasattr(zero_opt, "fp32_groups_flat_partition"):
                for buf in zero_opt.fp32_groups_flat_partition:
                    if isinstance(buf, torch.Tensor) and not buf.is_cuda:
                        buf.data = buf.data.to(device)
                        restored += buf.numel() * buf.element_size()

            inner_opt = getattr(zero_opt, "optimizer", zero_opt)
            if hasattr(inner_opt, "state"):
                for state_vals in inner_opt.state.values():
                    if not isinstance(state_vals, dict):
                        continue
                    for k, v in state_vals.items():
                        if isinstance(v, torch.Tensor) and not v.is_cuda:
                            state_vals[k] = v.to(device)
                            restored += v.numel() * v.element_size()

            if hasattr(zero_opt, "grad_partitions_flat_buffer"):
                buf = zero_opt.grad_partitions_flat_buffer
                if isinstance(buf, torch.Tensor) and not buf.is_cuda:
                    buf.data = buf.data.to(device)
                    restored += buf.numel() * buf.element_size()

            logger.info(
                "vLLM eval [sleep/wake]: rank %d — restored %.1f GiB of DeepSpeed state to GPU",
                rank, restored / 2**30,
            )
        except Exception as e:
            logger.warning("vLLM eval: DeepSpeed restore partial (%.1f GiB): %s", restored / 2**30, e)

    @staticmethod
    def _aggressive_empty_cache(max_retries: int = 3) -> None:
        """Aggressively free GPU memory (verl-style).

        Runs ``gc.collect()`` → ``torch.cuda.synchronize()`` →
        ``torch.cuda.empty_cache()`` in a retry loop, stopping early when
        less than 1 GiB is freed per iteration.
        """
        import gc

        for _ in range(max_retries):
            before = torch.cuda.memory_reserved()
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            freed = before - torch.cuda.memory_reserved()
            if freed < 2**30:  # < 1 GiB freed → diminishing returns
                break

    @staticmethod
    def _set_expandable_segments(enable: bool) -> None:
        """Toggle PyTorch's expandable_segments allocator setting at runtime.

        ``expandable_segments`` is incompatible with vLLM's CuMemAllocator.
        Following verl's pattern, we disable it for the *entire* duration
        that the vLLM engine is awake, not just during engine creation.
        """
        try:
            torch.cuda.memory._set_allocator_settings(
                f"expandable_segments:{enable}"
            )
        except Exception:
            pass

    def _cleanup_vllm_engine(self) -> None:
        """Destroy the persistent vLLM engine and free resources."""
        if self._vllm_engine is not None:
            # Try to sleep before destroying so CuMemAllocator releases
            # its memory-pool allocations (weights + KV cache).
            try:
                if not self._vllm_engine_sleeping:
                    self._vllm_engine.sleep(level=2)
            except Exception:
                pass
            del self._vllm_engine
            self._vllm_engine = None
            self._vllm_engine_sleeping = False
            self._aggressive_empty_cache()
            logger.info("vLLM eval: persistent engine destroyed")
        # NOTE: Do NOT delete /dev/shm/llamafactory_vllm_weights here.
        # In multi-rank training, one rank may fail (e.g. OOM) and call
        # this cleanup while other ranks still need the processor configs
        # from the shared directory.  The config files are tiny (few KB)
        # and harmless.  They will be overwritten on the next eval or
        # cleaned up on process exit.

    def _pre_init_vllm_parallel_state(self) -> None:
        """Pre-initialise vLLM's distributed parallel state on ALL ranks.

        vLLM's ``GroupCoordinator.__init__`` calls ``torch.distributed.new_group``
        which is **collective** — every rank in the default process group must
        participate.  Normally only rank 0 creates the vLLM engine, so the other
        ranks never call ``new_group`` and the process dead-locks.

        Fix: call ``init_distributed_environment`` and
        ``ensure_model_parallel_initialized`` on **all ranks** *before* the
        engine is created.  The ``new_group()`` calls are now properly
        collective and complete without hanging.  When rank 0 subsequently
        creates the engine, vLLM sees the state is already initialised and
        skips all collective ops.
        """
        from vllm.distributed.parallel_state import (
            _WORLD,
            ensure_model_parallel_initialized,
            init_distributed_environment,
        )

        if _WORLD is not None:
            # Already initialised (e.g. second eval) — nothing to do.
            return

        if not torch.distributed.is_initialized():
            return

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        tp_size = self.finetuning_args.vllm_eval_tp_size or 1

        logger.info(
            "vLLM eval [sleep/wake]: pre-init parallel state on all ranks "
            "(rank=%d, world=%d, tp=%d)",
            rank, world_size, tp_size,
        )

        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            backend="nccl",
        )
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
        )

    def _vllm_evaluate_sleep_wake(
        self,
        eval_dataset: Optional["Dataset"],
        metric_key_prefix: str,
    ) -> dict[str, float]:
        """vLLM evaluation with persistent engines using sleep/wake mode.

        **Manual data parallelism**: every rank creates its own independent
        TP=1 vLLM engine and generates on 1/N of the eval dataset.  Results
        are gathered on rank 0 for metric computation.  This gives ~Nx
        throughput compared to single-GPU eval.

        First eval: save weights to /dev/shm → all ranks create engine → generate → sleep.
        Subsequent evals: all ranks wake → reload weights → generate → sleep.
        """
        import time

        import torch.distributed as dist

        is_training = self.args.do_train
        unwrapped = self.accelerator.unwrap_model(self.model)
        device = next(unwrapped.parameters()).device
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # --- Step 1 (all ranks): free GPU memory for vLLM ---
        # HF Trainer stores the DeepSpeed engine as self.deepspeed (= self.model_wrapped),
        # NOT as self.model.  self.model may be the bare module without the optimizer.
        _ds_engine = getattr(self, "deepspeed", None)
        has_ds = _ds_engine is not None and getattr(_ds_engine, "optimizer", None) is not None

        # Zero out gradient buffers to free GPU memory.
        for param in unwrapped.parameters():
            param.grad = None

        if has_ds:
            # DeepSpeed ZeRO-2: keep training state on GPU.
            #
            # DeepSpeed's internal reference chain (parallel_partitioned_bit16_groups
            # → narrow() views of flat buffer, optimizer.state keys → FP32 master
            # weights, param._hp_mapping.optim_fragment → optimizer state views)
            # makes it impossible to fully offload GPU memory via .data replacement.
            # Attempting to offload leaves ~30 GiB of "zombie" GPU memory that
            # cannot be freed until the references are cleared.
            #
            # Instead, we keep everything on GPU and let vLLM's dynamic
            # gpu_memory_utilization cap (in _create_vllm_engine) use whatever
            # free memory remains.  After vLLM sleep, training resumes
            # immediately without any restore step.
            self._aggressive_empty_cache()
        else:
            # Non-DeepSpeed: simply move model to CPU.
            unwrapped.to("cpu")
            self._aggressive_empty_cache()

        free_mem, total_mem = torch.cuda.mem_get_info()
        logger.info(
            "vLLM eval [sleep/wake]: rank %d — prepared for vLLM "
            "(free=%.1f GiB / total=%.1f GiB, deepspeed=%s)",
            rank, free_mem / 2**30, total_mem / 2**30, has_ds,
        )

        # --- Step 2 (all ranks): pre-initialise vLLM parallel state ---
        self._pre_init_vllm_parallel_state()

        # --- Step 3 (all ranks): disable expandable_segments, create or wake engine ---
        # Following verl's pattern: expandable_segments stays disabled for
        # the entire duration that the vLLM engine is awake, because it
        # conflicts with vLLM's CuMemAllocator virtual-memory allocations.
        self._set_expandable_segments(False)

        metrics: dict[str, float] = {}
        try:
            from vllm import SamplingParams

            if self._vllm_engine is None:
                # --- First eval: create engine on every rank ---
                if is_training:
                    # Rank 0 saves weights to /dev/shm (shared tmpfs).
                    shm_dir = "/dev/shm/llamafactory_vllm_weights"
                    if rank == 0:
                        os.makedirs(shm_dir, exist_ok=True)
                        t0 = time.time()
                        unwrapped.save_pretrained(shm_dir, max_shard_size="100GB", safe_serialization=True)
                        self.processing_class.save_pretrained(shm_dir)
                        if self.processor is not None:
                            self.processor.save_pretrained(shm_dir)
                        self._fix_saved_config_for_vllm(shm_dir)
                        logger.info("vLLM eval [sleep/wake]: saved weights to /dev/shm (%.1fs)", time.time() - t0)
                    dist.barrier()  # wait for rank 0 to finish saving

                    t0 = time.time()
                    self._vllm_engine = self._create_vllm_engine(shm_dir)
                    logger.info("vLLM eval [sleep/wake]: rank %d — engine created (%.1fs)", rank, time.time() - t0)

                    dist.barrier()  # all engines loaded before cleanup
                    if rank == 0:
                        import glob as _glob
                        # Only delete large weight files; keep config/processor
                        # files (preprocessor_config.json, config.json, etc.)
                        # because vLLM lazily loads the HF processor from the
                        # model path during chat(), not during engine creation.
                        for _wf in _glob.glob(os.path.join(shm_dir, "*.safetensors")):
                            os.remove(_wf)
                else:
                    # Eval-only mode: load from original model path.
                    model_path = self.model_args.model_name_or_path if self.model_args else self.args.output_dir
                    t0 = time.time()
                    self._vllm_engine = self._create_vllm_engine(model_path)
                    logger.info("vLLM eval [sleep/wake]: rank %d — engine created from %s (%.1fs)", rank, model_path, time.time() - t0)
                self._vllm_engine_sleeping = False
            else:
                # --- Subsequent eval: wake up and reload weights ---
                # Following verl's pattern: wake weights first, sync, then
                # clean up before allocating KV cache so it has maximum
                # free memory available.
                t0 = time.time()
                self._vllm_engine.wake_up(tags=["weights"])
                self._sync_weights_to_vllm(unwrapped)
                self._aggressive_empty_cache()
                self._vllm_engine.wake_up(tags=["kv_cache"])
                logger.info("vLLM eval [sleep/wake]: rank %d — engine woke + weights reloaded (%.1fs)", rank, time.time() - t0)
                self._vllm_engine_sleeping = False

            # --- Step 4 (all ranks): shard dataset + generate ---
            sampling_params = SamplingParams(
                temperature=self.generating_args.temperature if self.generating_args else 0.95,
                top_p=self.generating_args.top_p if self.generating_args else 0.7,
                top_k=self.generating_args.top_k if self.generating_args else 50,
                max_tokens=self.generating_args.max_new_tokens if self.generating_args else 1024,
                repetition_penalty=self.generating_args.repetition_penalty if self.generating_args else 1.0,
                skip_special_tokens=True,
            )

            raw_data = self._load_raw_eval_data()
            total_len = len(raw_data)
            # Interleaved sharding: rank k gets indices [k, k+N, k+2N, ...]
            shard_indices = list(range(rank, total_len, world_size))
            shard_data = [raw_data[i] for i in shard_indices]
            logger.info("vLLM eval [sleep/wake]: rank %d — generating %d / %d samples", rank, len(shard_data), total_len)

            local_preds: list[str] = []
            local_labels: list[str] = []
            batch_size = self.finetuning_args.vllm_eval_batch_size
            t0 = time.time()
            local_done = 0

            # Chat template kwargs (e.g. enable_thinking for Qwen3).
            # Explicitly pass the flag so vLLM doesn't fall back to the
            # model's default (which may differ from the training config).
            chat_kwargs: dict[str, Any] = {}
            if self.data_args and hasattr(self.data_args, "enable_thinking"):
                chat_kwargs["enable_thinking"] = bool(self.data_args.enable_thinking)

            # Pass video/image processing parameters so vLLM matches
            # the training-time preprocessing (resolution, fps, max frames).
            # NOTE: do_sample_frames must be False.  vLLM's VideoMediaIO
            # already samples frames using the num_frames configured in
            # media_io_kwargs (see _create_vllm_engine).  Setting
            # do_sample_frames=True causes the HF processor to resample
            # from the already-sampled array using the *original* video
            # frame count, producing out-of-bounds indices.
            mm_kwargs: dict[str, Any] = {}
            if self.model_args:
                mm_kwargs["max_pixels"] = getattr(self.model_args, "video_max_pixels", 921600)
                mm_kwargs["min_pixels"] = getattr(self.model_args, "video_min_pixels", 1)

            # Suppress vLLM's verbose logging during generation so the
            # progress bar is not drowned out.
            import logging as _logging
            _vllm_logger = _logging.getLogger("vllm")
            _vllm_prev_level = _vllm_logger.level
            _vllm_logger.setLevel(_logging.WARNING)

            # Rank 0 shows a single tqdm progress bar for all GPUs.
            import sys
            pbar = None
            if rank == 0:
                from tqdm import tqdm
                pbar = tqdm(
                    total=total_len,
                    desc=f"vLLM eval ({world_size} GPUs)",
                    unit="sample",
                    file=sys.stdout,
                    dynamic_ncols=True,
                )
                sys.stdout.flush()

            for i in range(0, len(shard_data), batch_size):
                batch = shard_data[i : min(i + batch_size, len(shard_data))]
                messages_batch = [item["messages"] for item in batch]
                try:
                    results = self._vllm_engine.chat(
                        messages_batch, sampling_params, use_tqdm=False,
                        chat_template_kwargs=chat_kwargs if chat_kwargs else None,
                        mm_processor_kwargs=mm_kwargs if mm_kwargs else None,
                    )
                    local_preds.extend(r.outputs[0].text for r in results)
                    local_labels.extend(item["label"] for item in batch)
                except (torch.OutOfMemoryError, ValueError) as e:
                    # OOM on large videos (e.g. 50 frames @ 720p needs ~2 GiB
                    # workspace for masked_scatter).  Skip this batch rather
                    # than failing the entire eval.
                    logger.warning(
                        "vLLM eval: rank %d — skipping batch %d/%d due to OOM: %s",
                        rank, i // batch_size, len(shard_data) // batch_size, str(e)[:200],
                    )
                    local_preds.extend("" for _ in batch)
                    local_labels.extend(item["label"] for item in batch)
                    torch.cuda.empty_cache()
                local_done += len(batch)

                if pbar is not None:
                    est_total_done = min(local_done * world_size, total_len)
                    pbar.n = est_total_done
                    pbar.refresh()
                    sys.stdout.flush()

            if pbar is not None:
                pbar.n = total_len
                pbar.refresh()
                pbar.close()

            _vllm_logger.setLevel(_vllm_prev_level)

            logger.info(
                "vLLM eval [sleep/wake]: rank %d — generation done (%d samples, %.1fs)",
                rank, len(local_preds), time.time() - t0,
            )

            # --- Step 5 (all ranks): sleep engine ---
            # vLLM's CuMemAllocator (enabled by sleep mode) uses cuMemCreate
            # virtual memory, which IS released back to the system on sleep.
            # This allows training to reclaim the GPU memory.
            if is_training:
                # Clear multimodal encoder cache + prefix cache BEFORE sleep.
                # These caches hold processed video/image features on GPU and
                # are NOT released by sleep(level=2).
                self._vllm_engine.reset_prefix_cache()
                self._vllm_engine.sleep(level=2)
                self._vllm_engine_sleeping = True
                self._aggressive_empty_cache()

                free_mem, total_mem = torch.cuda.mem_get_info()
                logger.info(
                    "vLLM eval [sleep/wake]: rank %d — after sleep: free=%.1f / %.1f GiB",
                    rank, free_mem / 2**30, total_mem / 2**30,
                )

            # --- Step 6: gather predictions on rank 0 ---
            if world_size > 1:
                gathered_preds: list[list[str]] | None = [None] * world_size if rank == 0 else None  # type: ignore[assignment]
                gathered_labels: list[list[str]] | None = [None] * world_size if rank == 0 else None  # type: ignore[assignment]
                dist.gather_object(local_preds, gathered_preds, dst=0)
                dist.gather_object(local_labels, gathered_labels, dst=0)

                if rank == 0:
                    # Reconstruct original order: interleave shards.
                    all_preds = [""] * total_len
                    all_labels = [""] * total_len
                    for src_rank in range(world_size):
                        indices = list(range(src_rank, total_len, world_size))
                        for local_idx, global_idx in enumerate(indices):
                            if local_idx < len(gathered_preds[src_rank]):  # type: ignore[index]
                                all_preds[global_idx] = gathered_preds[src_rank][local_idx]  # type: ignore[index]
                                all_labels[global_idx] = gathered_labels[src_rank][local_idx]  # type: ignore[index]
                    metrics = self._compute_vllm_metrics(all_preds, all_labels, metric_key_prefix)
                    logger.info("vLLM eval [sleep/wake]: %s", metrics)
            else:
                metrics = self._compute_vllm_metrics(local_preds, local_labels, metric_key_prefix)
                logger.info("vLLM eval [sleep/wake]: %s", metrics)

        except Exception as e:
            logger.warning("vLLM eval [sleep/wake] failed on rank %d: %s", rank, e, exc_info=True)
            self._cleanup_vllm_engine()

        # Re-enable expandable_segments now that vLLM is asleep / destroyed.
        if self._expandable_segments_was_enabled:
            self._set_expandable_segments(True)

        self.accelerator.wait_for_everyone()

        # --- Step 7: restore training model to GPU ---
        if has_ds:
            # DeepSpeed ZeRO-2: training state never left GPU — nothing to do.
            pass
        else:
            unwrapped.to(device)

        free_mem, total_mem = torch.cuda.mem_get_info()
        logger.info(
            "vLLM eval [sleep/wake]: rank %d — training model restored to %s "
            "(free=%.1f / %.1f GiB)",
            rank, device, free_mem / 2**30, total_mem / 2**30,
        )

        # Broadcast metrics from rank 0
        if world_size > 1:
            metrics_list = [metrics]
            dist.broadcast_object_list(metrics_list, src=0)
            metrics = metrics_list[0]

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics=metrics)
        return metrics

    def _vllm_evaluate_legacy(
        self,
        eval_dataset: Optional["Dataset"],
        metric_key_prefix: str,
    ) -> dict[str, float]:
        """Legacy vLLM evaluation: create engine → generate → destroy each eval step.

        Flow: save weights → offload training model → create vLLM engine → generate → destroy engine → restore.
        In-process (no subprocess), avoids NCCL/env conflicts.
        """
        import gc
        import shutil
        import time

        unwrapped = self.accelerator.unwrap_model(self.model)
        device = next(unwrapped.parameters()).device
        temp_dir = os.path.join(self.args.output_dir, "_vllm_eval_tmp")

        # --- Step 1: save current training weights (rank 0) ---
        if self.accelerator.is_main_process:
            os.makedirs(temp_dir, exist_ok=True)
            t0 = time.time()
            unwrapped.save_pretrained(temp_dir)
            self.processing_class.save_pretrained(temp_dir)
            if self.processor is not None:
                self.processor.save_pretrained(temp_dir)
            self._fix_saved_config_for_vllm(temp_dir)
            logger.info("vLLM eval: saved weights to %s (%.1fs)", temp_dir, time.time() - t0)
        self.accelerator.wait_for_everyone()

        # --- Step 2: offload training model to CPU (all ranks) ---
        unwrapped.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("vLLM eval: training model offloaded to CPU")

        # --- Step 3: create vLLM engine + generate (rank 0 only) ---
        metrics: dict[str, float] = {}
        if self.accelerator.is_main_process:
            try:
                from vllm import LLM, SamplingParams

                t0 = time.time()
                legacy_max_model_len = self.finetuning_args.vllm_eval_max_model_len
                tokenizer_path = self.model_args.model_name_or_path if self.model_args else temp_dir
                engine_kwargs = {
                    "model": temp_dir,
                    "tokenizer": tokenizer_path,
                    "trust_remote_code": True,
                    "tensor_parallel_size": self.finetuning_args.vllm_eval_tp_size or 1,
                    "gpu_memory_utilization": self.finetuning_args.vllm_eval_gpu_util,
                    "max_model_len": legacy_max_model_len,
                    "max_num_batched_tokens": legacy_max_model_len,
                    "enforce_eager": self.finetuning_args.vllm_eval_enforce_eager,
                    "disable_log_stats": True,
                    "limit_mm_per_prompt": {"image": 4, "video": 2, "audio": 2},
                    "allowed_local_media_path": "/",
                    "media_io_kwargs": {
                        "video": {
                            "num_frames": getattr(self.model_args, "video_maxlen", 128) if self.model_args else 128,
                        },
                    },
                }
                llm = LLM(**engine_kwargs)
                logger.info("vLLM eval: engine created (%.1fs)", time.time() - t0)

                # Prepare sampling params
                sampling_params = SamplingParams(
                    temperature=self.generating_args.temperature if self.generating_args else 0.95,
                    top_p=self.generating_args.top_p if self.generating_args else 0.7,
                    top_k=self.generating_args.top_k if self.generating_args else 50,
                    max_tokens=self.generating_args.max_new_tokens if self.generating_args else 1024,
                    repetition_penalty=self.generating_args.repetition_penalty if self.generating_args else 1.0,
                    skip_special_tokens=True,
                )

                # Chat template kwargs (e.g. enable_thinking for Qwen3).
                # Explicitly pass the flag so vLLM doesn't fall back to the
                # model's default (which may differ from the training config).
                chat_kwargs: dict[str, Any] = {}
                if self.data_args and hasattr(self.data_args, "enable_thinking"):
                    chat_kwargs["enable_thinking"] = bool(self.data_args.enable_thinking)

                # Pass video/image processing parameters so vLLM matches
                # the training-time preprocessing (resolution, fps, max frames).
                mm_kwargs: dict[str, Any] = {}
                if self.model_args:
                    mm_kwargs["max_pixels"] = getattr(self.model_args, "video_max_pixels", 921600)
                    mm_kwargs["min_pixels"] = getattr(self.model_args, "video_min_pixels", 1)
                    mm_kwargs["fps"] = getattr(self.model_args, "video_fps", 2.0)
                    mm_kwargs["max_num_frames"] = getattr(self.model_args, "video_maxlen", 128)

                # Generate
                raw_data = self._load_raw_eval_data()
                all_preds, all_labels = [], []
                batch_size = self.finetuning_args.vllm_eval_batch_size
                t0 = time.time()
                for i in range(0, len(raw_data), batch_size):
                    batch = raw_data[i : min(i + batch_size, len(raw_data))]
                    messages_batch = [item["messages"] for item in batch]
                    results = llm.chat(
                        messages_batch, sampling_params,
                        chat_template_kwargs=chat_kwargs if chat_kwargs else None,
                        mm_processor_kwargs=mm_kwargs if mm_kwargs else None,
                    )
                    all_preds.extend(r.outputs[0].text for r in results)
                    all_labels.extend(item["label"] for item in batch)
                    logger.info("vLLM eval: generated %d / %d", len(all_preds), len(raw_data))

                logger.info("vLLM eval: generation done (%.1fs)", time.time() - t0)

                # Destroy engine to free all GPU memory
                del llm
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("vLLM eval: engine destroyed, GPU memory freed")

                # Compute metrics
                metrics = self._compute_vllm_metrics(all_preds, all_labels, metric_key_prefix)
                logger.info("vLLM eval: %s", metrics)

            except Exception as e:
                logger.warning("vLLM eval failed: %s", e, exc_info=True)

        self.accelerator.wait_for_everyone()

        # --- Step 4: restore training model to GPU (all ranks) ---
        has_ds = hasattr(self.model, "optimizer") and self.model.optimizer is not None
        if has_ds:
            self._restore_deepspeed_states_to_gpu(device)
        else:
            unwrapped.to(device)
        logger.info("vLLM eval: training model restored to %s", device)

        # Cleanup temp dir
        if self.accelerator.is_main_process:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Broadcast metrics from rank 0
        if self.accelerator.num_processes > 1:
            import torch.distributed as dist

            metrics_list = [metrics]
            dist.broadcast_object_list(metrics_list, src=0)
            metrics = metrics_list[0]

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics=metrics)
        return metrics

    # ------------------------------------------------------------------
    #  vLLM eval metric helpers (reuse logic from ComputeVideoRewardAccuracy)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_time_range(text: str) -> Optional[tuple[float, float]]:
        """Parse time range like 'between <11.0 seconds> and <12.0 seconds>'."""
        if not text:
            return None
        pattern = r'(?:between|from)\s*<\s*([\d.:]+)\s*seconds?\s*>\s*(?:and|to)\s*<\s*([\d.:]+)\s*seconds?\s*>'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                def _pv(raw: str) -> float:
                    if ':' in raw:
                        parts = raw.split(':')
                        return float(parts[0]) * 60 + float(parts[1])
                    return float(raw)
                start, end = _pv(match.group(1)), _pv(match.group(2))
                if end >= start:
                    return (start, end)
            except (ValueError, IndexError):
                pass
        return None

    @staticmethod
    def _detect_split(video_path: str) -> str:
        """Detect split from video path (osworld, ubuntu, winmac, scalecua)."""
        lp = video_path.lower()
        basename = lp.rsplit("/", 1)[-1] if "/" in lp else lp
        if basename.startswith("osworld_"):
            return "osworld"
        elif basename.startswith("ubuntu_"):
            return "ubuntu"
        elif basename.startswith("winmac_"):
            return "winmac"
        elif "scalecua" in lp:
            return "scalecua"
        return "unknown"

    @staticmethod
    def _compute_split_metrics(predictions: list[dict]) -> dict[str, Any]:
        """Compute accuracy / precision / recall / f1 for a list of prediction dicts."""
        if not predictions:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "tp": 0, "fp": 0, "fn": 0, "tn": 0, "total": 0}
        tp = sum(1 for p in predictions if p["pred_box"] == "correct" and p["label_box"] == "correct")
        fp = sum(1 for p in predictions if p["pred_box"] == "correct" and p["label_box"] != "correct")
        fn = sum(1 for p in predictions if p["pred_box"] != "correct" and p["label_box"] == "correct")
        tn = sum(1 for p in predictions if p["pred_box"] != "correct" and p["label_box"] != "correct")
        total = tp + fp + fn + tn
        acc = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn, "total": total}

    @staticmethod
    def _compute_range_metrics(predictions: list[dict]) -> dict[str, Any]:
        """Compute temporal range tIoU / MAE for predictions with time ranges."""
        applicable = successful = 0
        tious: list[float] = []
        start_errors: list[float] = []
        end_errors: list[float] = []
        for p in predictions:
            lr = p.get("label_range")
            if lr is None:
                continue
            applicable += 1
            pr = p.get("pred_range")
            if pr is None:
                continue
            successful += 1
            inter_start = max(pr[0], lr[0])
            inter_end = min(pr[1], lr[1])
            intersection = max(0.0, inter_end - inter_start + 1)
            union = (pr[1] - pr[0] + 1) + (lr[1] - lr[0] + 1) - intersection
            tious.append(intersection / union if union > 0 else 0.0)
            start_errors.append(abs(pr[0] - lr[0]))
            end_errors.append(abs(pr[1] - lr[1]))
        return {
            "range_applicable": applicable,
            "range_success": successful,
            "range_success_rate": successful / applicable if applicable > 0 else 0.0,
            "mean_tiou": float(np.mean(tious)) if tious else 0.0,
            "mean_start_mae": float(np.mean(start_errors)) if start_errors else 0.0,
            "mean_end_mae": float(np.mean(end_errors)) if end_errors else 0.0,
        }

    def _compute_vllm_metrics(
        self, preds: list[str], labels: list[str], metric_key_prefix: str
    ) -> dict[str, float]:
        """Compute video reward accuracy from vLLM predictions and save results."""
        raw_data = self._load_raw_eval_data()

        predictions_list: list[dict[str, Any]] = []

        for idx, (pred_text, label_text) in enumerate(zip(preds, labels)):
            pred_box = re.search(r"\\box(?:ed)?\{(.*?)\}", pred_text)
            if not pred_box:
                pred_box = re.search(r"box(?:ed)?\{(.*?)\}", pred_text)
            label_box = re.search(r"\\box(?:ed)?\{(.*?)\}", label_text)

            pred_val = pred_box.group(1) if pred_box else None
            label_val = label_box.group(1) if label_box else None
            matched = int(pred_val is not None and label_val is not None and pred_val == label_val)

            entry: dict[str, Any] = {
                "pred_box": pred_val,
                "label_box": label_val,
                "matched": matched,
                "raw_pred": pred_text,
                "raw_label": label_text,
            }

            # Metadata from raw eval data
            if idx < len(raw_data):
                entry["split"] = raw_data[idx]["split"]
                entry["video_path"] = raw_data[idx]["video_path"]
                entry["user_instruction"] = raw_data[idx]["user_instruction"]

            # Time-range parsing
            label_range = self._parse_time_range(label_text)
            pred_range = self._parse_time_range(pred_text)
            if label_range is not None:
                entry["label_range"] = list(label_range)
            if pred_range is not None:
                entry["pred_range"] = list(pred_range)

            predictions_list.append(entry)

        # Overall metrics
        overall = self._compute_split_metrics(predictions_list)
        total = overall["total"]
        accuracy = overall["accuracy"]

        # Per-split metrics
        per_split: dict[str, dict[str, Any]] = {}
        split_preds: dict[str, list[dict]] = {}
        for p in predictions_list:
            s = p.get("split", "unknown")
            split_preds.setdefault(s, []).append(p)
        for split_name in sorted(split_preds):
            sm = self._compute_split_metrics(split_preds[split_name])
            rm = self._compute_range_metrics(split_preds[split_name])
            if rm["range_applicable"] > 0:
                sm["range"] = rm
            per_split[split_name] = sm

        # Log per-split
        logger.info("=" * 60)
        logger.info("PER-SPLIT EVALUATION METRICS (vLLM)")
        logger.info("=" * 60)
        for sn, sm in per_split.items():
            logger.info(
                "  [%s] (%d) acc=%.4f prec=%.4f rec=%.4f f1=%.4f",
                sn.upper(), sm["total"], sm["accuracy"], sm["precision"], sm["recall"], sm["f1"],
            )
            if "range" in sm:
                rm = sm["range"]
                logger.info(
                    "    range: %d/%d success=%.4f tIoU=%.4f start_mae=%.2f end_mae=%.2f",
                    rm["range_success"], rm["range_applicable"],
                    rm["range_success_rate"], rm["mean_tiou"],
                    rm["mean_start_mae"], rm["mean_end_mae"],
                )
        logger.info("=" * 60)

        # --- Save predictions JSON ---
        if self.finetuning_args.eval_save_predictions:
            from datetime import datetime

            output_dir = self.finetuning_args.eval_predictions_output_dir or self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"eval_predictions_{timestamp}.json")

            output_data: dict[str, Any] = {
                "summary": {
                    "total_samples": total,
                    "correct": overall["tp"] + overall["tn"],
                    "accuracy": accuracy,
                    "precision": overall["precision"],
                    "recall": overall["recall"],
                    "f1": overall["f1"],
                    "tp": overall["tp"], "fp": overall["fp"],
                    "fn": overall["fn"], "tn": overall["tn"],
                },
                "predictions": predictions_list,
            }
            if per_split:
                output_data["per_split_summary"] = per_split

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            logger.info("vLLM eval: saved %d predictions to %s", total, output_file)

        return {
            f"{metric_key_prefix}_vllm_accuracy": accuracy,
            f"{metric_key_prefix}_vllm_correct": overall["tp"] + overall["tn"],
            f"{metric_key_prefix}_vllm_total": total,
        }

    def _accumulate_train_video_reward_accuracy(
        self, logits: "torch.Tensor", inputs: dict[str, Union["torch.Tensor", Any]]
    ) -> None:
        if self.args.prediction_loss_only:
            return
        labels = inputs.get("labels")
        if logits.dim() != 3 or labels is None:
            logger.info_rank0("[ERROR] Logits dim: {}, labels: {}".format(logits.dim(), labels is not None))
            return

        # Check if STP modified the labels (token removal mode)
        # In this case, use the modified labels that match logits length
        if hasattr(self.model, "_stp_modified_labels") and self.model._stp_modified_labels is not None:
            modified_labels = self.model._stp_modified_labels
            if modified_labels.shape[1] == logits.shape[1]:
                labels = modified_labels
            # Clear the stored labels to avoid using stale data
            self.model._stp_modified_labels = None

        tokenizer = None
        if self._train_video_reward_metric is not None:
            tokenizer = self._train_video_reward_metric.tokenizer
        if tokenizer is None or tokenizer.pad_token_id is None:
            return

        input_ids = inputs.get("input_ids")

        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            preds = preds.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            input_texts: Optional[list[str]] = None
            if isinstance(input_ids, torch.Tensor):
                input_texts = tokenizer.batch_decode(
                    input_ids.detach().cpu().numpy(), skip_special_tokens=True
                )

        pad_id = tokenizer.pad_token_id
        preds = np.where(labels_np != IGNORE_INDEX, preds, pad_id)
        labels_np = np.where(labels_np != IGNORE_INDEX, labels_np, pad_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels_np, skip_special_tokens=True)

        for idx, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
            pred_box = re.search(r"\\box(?:ed)?\{(.*?)\}", pred)
            if not pred_box:
                pred_box = re.search(r"box(?:ed)?\{(.*?)\}", pred)
            label_box = re.search(r"\\box(?:ed)?\{(.*?)\}", label)
            matched = 0
            if pred_box and label_box:
                matched = int(pred_box.group(1) == label_box.group(1))
            self._train_video_reward_correct_count += matched
            self._train_video_reward_total_count += 1

            # if (
            #     self._enable_prediction_logging
            #     and len(self._train_prediction_samples) < self._max_prediction_samples
            # ):
            #     prompt = ""
            #     if isinstance(input_ids, torch.Tensor) and input_texts is not None and idx < len(input_texts):
            #         prompt = input_texts[idx]
            #     logger.info_rank0("Prediction: {}\nLabel: {}\nMatch: {}".format(pred, label, matched))
            #     self._train_prediction_samples.append(
            #         {
            #             "id": uuid.uuid4().hex,
            #             "prompt": prompt,
            #             "prediction": pred,
            #             "target": label,
            #             "match": matched,
            #         }
            #     )

    @override
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        if (
            self._train_video_reward_metric is not None
            and self._train_video_reward_total_count > 0
            and "loss" in logs
            and not any(key.startswith("eval_") or key.startswith("predict_") for key in logs)
        ):
            # Compute cumulative accuracy (do not clear counters to track cumulative accuracy).
            logs["video_reward_cumulative_accuracy"] = float(
                self._train_video_reward_correct_count / self._train_video_reward_total_count
            )
            # self._log_predictions_to_wandb_table(stage="train")
        super().log(logs, start_time=start_time)

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

    def _check_wandb_table_support(self) -> bool:
        try:
            import wandb  # type: ignore
        except ImportError:
            return False

        return hasattr(wandb, "Table")

    def _log_predictions_to_wandb_table(self, stage: str) -> None:
        if not self._enable_prediction_logging or not self._train_prediction_samples:
            self._train_prediction_samples.clear()
            return

        try:
            import wandb  # type: ignore
        except ImportError:
            self._enable_prediction_logging = False
            self._train_prediction_samples.clear()
            return

        if getattr(wandb, "run", None) is None:
            self._train_prediction_samples.clear()
            return

        columns = ["id", "step", "stage", "prompt", "prediction", "target", "match"]
        table = wandb.Table(columns=columns)
        step = self.state.global_step if self.state is not None else None

        for sample in self._train_prediction_samples:
            table.add_data(
                sample["id"],
                step,
                stage,
                sample["prompt"],
                sample["prediction"],
                sample["target"],
                sample["match"],
            )

        log_payload = {f"{stage}/predictions": table}
        if step is not None:
            wandb.log(log_payload, step=step, commit=False)
        else:
            wandb.log(log_payload, commit=False)

        self._train_prediction_samples.clear()
