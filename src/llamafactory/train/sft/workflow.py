# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

from typing import TYPE_CHECKING, Optional

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.packages import is_transformers_version_greater_than
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, ComputeVideoRewardAccuracy, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # Set left-padding for generation before dataset processing and training.
    # This ensures consistent padding_side for both training-time eval and standalone eval.
    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"

    # When using vLLM eval, skip eval dataset tokenization — the trainer
    # reads raw JSON files directly via _load_raw_eval_data() and calls
    # llm.chat(), so tokenized eval data is never used.
    saved_eval_dataset = data_args.eval_dataset
    if finetuning_args.use_vllm_eval:
        data_args.eval_dataset = None

    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)

    if finetuning_args.use_vllm_eval:
        data_args.eval_dataset = saved_eval_dataset
        # Trainer.__init__ requires eval_dataset when eval_strategy != "no".
        # Provide a single-row placeholder so the check passes; the actual
        # evaluation reads raw JSON via _load_raw_eval_data() and never
        # touches this dataset.
        if "eval_dataset" not in dataset_module or dataset_module["eval_dataset"] is None:
            from datasets import Dataset as HFDataset
            dataset_module["eval_dataset"] = HFDataset.from_dict(
                {"input_ids": [[0]], "attention_mask": [[1]], "labels": [[0]]}
            )

    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Metric utils
    metric_module = {}
    if model_args.use_kt:
        if training_args.predict_with_generate:
            raise NotImplementedError("`predict_with_generate` is not supported in KTransformers SFT yet.")
        elif finetuning_args.compute_accuracy:
            raise NotImplementedError("`compute_accuracy` is not supported in KTransformers SFT yet.")

    if finetuning_args.compute_video_reward_cumulative_accuracy:
        # Use ComputeVideoRewardAccuracy for evaluation
        eval_output_dir = finetuning_args.eval_predictions_output_dir or training_args.output_dir

        # Resolve eval dataset file paths for per-split metrics (supports multiple datasets)
        eval_dataset_paths = []
        if data_args.eval_dataset:
            import os
            from ...data.parser import get_dataset_list
            eval_dataset_attrs = get_dataset_list(data_args.eval_dataset, data_args.dataset_dir)
            for attr in eval_dataset_attrs:
                if attr.load_from == "file":
                    candidate = os.path.join(data_args.dataset_dir, attr.dataset_name)
                    if os.path.isfile(candidate):
                        eval_dataset_paths.append(candidate)

        metric_module["compute_metrics"] = ComputeVideoRewardAccuracy(
            tokenizer=tokenizer,
            save_predictions=finetuning_args.eval_save_predictions,
            output_dir=eval_output_dir,
            use_generate_mode=training_args.predict_with_generate,
            eval_dataset_paths=eval_dataset_paths if eval_dataset_paths else None,
        )
        # Only use logit processor if not using generate mode
        if not training_args.predict_with_generate:
            metric_module["preprocess_logits_for_metrics"] = eval_logit_processor
    elif training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)

    # Compatible with Transformers v4 and Transformers v5
    if is_transformers_version_greater_than("4.58.0"):
        extra_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
        if not isinstance(extra_ids, list):
            extra_special_tokens = getattr(tokenizer, "_extra_special_tokens", [])
            string_tokens = [str(t) for t in extra_special_tokens]
            extra_ids = tokenizer.convert_tokens_to_ids(string_tokens)
        all_eos_ids = [tokenizer.eos_token_id] + [i for i in extra_ids if i != -1]
        unique_eos_ids = list(dict.fromkeys(all_eos_ids))
        gen_kwargs["eos_token_id"] = unique_eos_ids
    else:
        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # Add thinking budget logits processor for reasoning models
    if generating_args.thinking_budget > 0:
        from ...data.template import ReasoningTemplate
        from ...extras.misc import ThinkingBudgetLogitsProcessor

        if isinstance(template, ReasoningTemplate):
            from transformers import LogitsProcessorList

            think_start_str = template.thought_words[0].strip()
            think_end_str = template.thought_words[1].strip()
            think_start_id = tokenizer.encode(think_start_str, add_special_tokens=False)[0]
            think_end_id = tokenizer.encode(think_end_str, add_special_tokens=False)[0]
            gen_kwargs["logits_processor"] = LogitsProcessorList(
                [ThinkingBudgetLogitsProcessor(think_start_id, think_end_id, generating_args.thinking_budget)]
            )
            logger.info_rank0(
                "Thinking budget enabled: max %d thinking tokens (think_start_id=%d, think_end_id=%d)",
                generating_args.thinking_budget,
                think_start_id,
                think_end_id,
            )

    # Initialize our Trainer
    if model_args.use_kt:
        from ktransformers.sft.lora import KTrainer  # type: ignore
        from ktransformers.util.globals import GLOBAL_CONFIG  # type: ignore

        GLOBAL_CONFIG._config["mod"] = "sft"

        trainer = KTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer_module,
            data_collator=data_collator,
            callbacks=callbacks,
            **dataset_module,
            **metric_module,
        )
        trainer.model_accepts_loss_kwargs = False
        model.config.use_cache = False

    else:
        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            finetuning_args=finetuning_args,
            model_args=model_args,
            data_args=data_args,
            data_collator=data_collator,
            callbacks=callbacks,
            gen_kwargs=gen_kwargs,
            generating_args=generating_args,
            **dataset_module,
            **tokenizer_module,
            **metric_module,
        )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log(train_result.metrics)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
                )
            else:
                keys += ["eval_loss", "eval_accuracy"]

            plot_loss(training_args.output_dir, keys=keys)

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if metrics:
            trainer.log(metrics)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
        else:
            logger.warning_rank0("Evaluation returned empty metrics.")

    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log(predict_results.metrics)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
