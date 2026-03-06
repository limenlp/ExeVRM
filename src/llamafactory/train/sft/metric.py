# Copyright 2025 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
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

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from transformers.utils import is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras import logging
from ...extras.misc import numpify
from ...extras.packages import is_jieba_available, is_rouge_available

logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore


if is_rouge_available():
    from rouge_chinese import Rouge  # type: ignore


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""Compute the token with the largest likelihood to reduce memory footprint."""
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)


@dataclass
class ComputeAccuracy:
    r"""Compute accuracy and support `batch_eval_metrics`."""

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()

@dataclass
class ComputeVideoRewardAccuracy:
    r"""Compute accuracy, F1 score and support `batch_eval_metrics`."""

    tokenizer: "PreTrainedTokenizer"
    save_predictions: bool = False
    output_dir: Optional[str] = None
    use_generate_mode: bool = False  # If True, preds are generated tokens; if False, preds are argmax of logits
    eval_dataset_paths: Optional[list[str]] = None  # List of paths to eval dataset files for per-split metrics

    def _load_split_mapping(self) -> None:
        """Load video paths from the eval dataset files and create a split mapping."""
        self._split_names = []  # split name for each sample, in order
        if not self.eval_dataset_paths:
            return

        import json
        import os

        for dataset_path in self.eval_dataset_paths:
            if not os.path.isfile(dataset_path):
                logger.warning_rank0(f"[ComputeVideoRewardAccuracy] Dataset file not found: {dataset_path}")
                continue

            try:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                # Try JSONL format
                data = []
                with open(dataset_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))

            for item in data:
                videos = item.get("videos", [])
                video_path = videos[0] if videos else ""
                split = self._detect_split(video_path)
                self._split_names.append(split)

            logger.info_rank0(f"[ComputeVideoRewardAccuracy] Loaded {len(data)} samples from {dataset_path}")

        logger.info_rank0(f"[ComputeVideoRewardAccuracy] Total split mapping: {len(self._split_names)} samples")
        from collections import Counter
        split_counts = Counter(self._split_names)
        for split_name, count in sorted(split_counts.items()):
            logger.info_rank0(f"  Split '{split_name}': {count} samples")

    @staticmethod
    def _detect_split(video_path: str) -> str:
        """Detect which split a sample belongs to based on its video path."""
        video_path_lower = video_path.lower()
        # Check filename part for the split prefix pattern
        basename = video_path_lower.rsplit("/", 1)[-1] if "/" in video_path_lower else video_path_lower
        if basename.startswith("osworld_"):
            return "osworld"
        elif basename.startswith("ubuntu_"):
            return "ubuntu"
        elif basename.startswith("winmac_"):
            return "winmac"
        # ScaleCUA videos have UUID filenames but the path contains "scalecua"
        elif "scalecua" in video_path_lower:
            return "scalecua"
        return "unknown"

    @staticmethod
    def _parse_time_value(raw: str) -> float:
        """Parse a time value that may be plain seconds ('11.0') or mm:ss ('1:30')."""
        if ':' in raw:
            parts = raw.split(':')
            return float(parts[0]) * 60 + float(parts[1])
        return float(raw)

    @staticmethod
    def _parse_time_range(text: str) -> Optional[tuple[float, float]]:
        """Parse time range from text like 'between <X seconds> and <Y seconds>'.

        Supports plain seconds (e.g. <11.0 seconds>) and mm:ss (e.g. <1:30 seconds>).

        Returns:
            (start, end) tuple in seconds, or None if not found/parseable.
        """
        if not text:
            return None
        # Match patterns like: between <11.0 seconds> and <12.0 seconds>
        # Also handle: from <X seconds> to <Y seconds>
        # Also handle mm:ss format: <1:30 seconds>
        pattern = r'(?:between|from)\s*<\s*([\d.:]+)\s*seconds?\s*>\s*(?:and|to)\s*<\s*([\d.:]+)\s*seconds?\s*>'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                start = ComputeVideoRewardAccuracy._parse_time_value(match.group(1))
                end = ComputeVideoRewardAccuracy._parse_time_value(match.group(2))
                if end >= start:
                    return (start, end)
            except (ValueError, IndexError):
                pass
        return None

    @staticmethod
    def _compute_range_metrics(predictions: list) -> dict:
        """Compute temporal range metrics for predictions that have time ranges.

        Only considers samples where label has a time range (i.e., scalecua incorrect samples).
        Uses temporal IoU (tIoU) as the primary metric.

        Returns dict with:
            - range_applicable: number of samples where label has a range
            - range_success: number of samples where pred also has a valid range
            - range_success_rate: range_success / range_applicable
            - mean_tiou: mean temporal IoU across successful samples
            - mean_start_mae: mean absolute error of start times (seconds)
            - mean_end_mae: mean absolute error of end times (seconds)
        """
        applicable = 0
        successful = 0
        tious = []
        start_errors = []
        end_errors = []

        for p in predictions:
            label_range = ComputeVideoRewardAccuracy._parse_time_range(p.get("raw_label", ""))
            if label_range is None:
                continue  # This sample doesn't have a label range, skip
            applicable += 1

            pred_range = ComputeVideoRewardAccuracy._parse_time_range(p.get("raw_pred", ""))
            if pred_range is None:
                continue  # Model didn't output a valid range
            successful += 1

            # Compute temporal IoU
            pred_start, pred_end = pred_range
            label_start, label_end = label_range

            # Closed-interval tIoU (+1 convention)
            inter_start = max(pred_start, label_start)
            inter_end = min(pred_end, label_end)
            intersection = max(0.0, inter_end - inter_start + 1)

            pred_len = pred_end - pred_start + 1
            label_len = label_end - label_start + 1
            union = pred_len + label_len - intersection
            tiou = intersection / union if union > 0 else 0.0
            tious.append(tiou)

            # Compute MAE for start and end
            start_errors.append(abs(pred_start - label_start))
            end_errors.append(abs(pred_end - label_end))

        return {
            "range_applicable": applicable,
            "range_success": successful,
            "range_success_rate": successful / applicable if applicable > 0 else 0.0,
            "mean_tiou": float(np.mean(tious)) if tious else 0.0,
            "mean_start_mae": float(np.mean(start_errors)) if start_errors else 0.0,
            "mean_end_mae": float(np.mean(end_errors)) if end_errors else 0.0,
        }

    @staticmethod
    def _compute_split_metrics(predictions: list) -> dict:
        """Compute accuracy, precision, recall, f1 for a list of prediction dicts."""
        if not predictions:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "tp": 0, "fp": 0, "fn": 0, "tn": 0, "total": 0}

        tp = sum(1 for p in predictions if p["pred_box"] == "correct" and p["label_box"] == "correct")
        fp = sum(1 for p in predictions if p["pred_box"] == "correct" and p["label_box"] == "incorrect")
        fn = sum(1 for p in predictions if p["pred_box"] != "correct" and p["label_box"] == "correct")
        tn = sum(1 for p in predictions if p["pred_box"] != "correct" and p["label_box"] == "incorrect")

        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn, "total": total}

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict") and self.score_dict.get("test_set_accuracy"):
            # Only compute mean if we have data
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items() if v}

            # Compute F1 score if save_predictions is enabled
            # For binary classification (correct/incorrect), we treat "correct" as the positive class
            if self.save_predictions and hasattr(self, "predictions_list") and self.predictions_list:
                overall_metrics = self._compute_split_metrics(self.predictions_list)

                result["test_set_precision"] = overall_metrics["precision"]
                result["test_set_recall"] = overall_metrics["recall"]
                result["test_set_f1"] = overall_metrics["f1"]

                # Store metrics for saving to JSON
                self._metrics_for_json = {
                    "tp": overall_metrics["tp"],
                    "fp": overall_metrics["fp"],
                    "fn": overall_metrics["fn"],
                    "tn": overall_metrics["tn"],
                    "precision": overall_metrics["precision"],
                    "recall": overall_metrics["recall"],
                    "f1": overall_metrics["f1"],
                    "accuracy": result.get("test_set_accuracy", 0.0),
                }

                # Compute and log per-split metrics
                self._per_split_metrics = {}
                if self._split_names:
                    split_predictions = {}
                    n_real = len(self._split_names)
                    n_total = len(self.predictions_list)
                    if n_total > n_real:
                        logger.info_rank0(
                            f"[ComputeVideoRewardAccuracy] {n_total} predictions > {n_real} dataset samples "
                            f"(DDP padding detected, {n_total - n_real} extra samples will be excluded from per-split stats)"
                        )
                    # Only use the first n_real predictions for per-split stats
                    for idx in range(min(n_total, n_real)):
                        pred = self.predictions_list[idx]
                        split = self._split_names[idx]
                        if split not in split_predictions:
                            split_predictions[split] = []
                        split_predictions[split].append(pred)

                    logger.info_rank0("=" * 70)
                    logger.info_rank0("PER-SPLIT EVALUATION METRICS")
                    logger.info_rank0("=" * 70)
                    for split_name in sorted(split_predictions.keys()):
                        preds = split_predictions[split_name]
                        sm = self._compute_split_metrics(preds)
                        self._per_split_metrics[split_name] = sm

                        # Add per-split metrics to result dict
                        result[f"test_set_{split_name}_accuracy"] = sm["accuracy"]
                        result[f"test_set_{split_name}_precision"] = sm["precision"]
                        result[f"test_set_{split_name}_recall"] = sm["recall"]
                        result[f"test_set_{split_name}_f1"] = sm["f1"]

                        logger.info_rank0(f"\n  [{split_name.upper()}] ({sm['total']} samples)")
                        logger.info_rank0(f"    Accuracy:  {sm['accuracy']:.4f} ({sm['accuracy']*100:.2f}%)")
                        logger.info_rank0(f"    Precision: {sm['precision']:.4f}")
                        logger.info_rank0(f"    Recall:    {sm['recall']:.4f}")
                        logger.info_rank0(f"    F1:        {sm['f1']:.4f}")
                        logger.info_rank0(f"    TP={sm['tp']}, FP={sm['fp']}, FN={sm['fn']}, TN={sm['tn']}")

                        # Compute temporal range metrics for splits that have range labels
                        rm = self._compute_range_metrics(preds)
                        if rm["range_applicable"] > 0:
                            self._per_split_metrics[split_name]["range"] = rm
                            result[f"test_set_{split_name}_range_tiou"] = rm["mean_tiou"]
                            result[f"test_set_{split_name}_range_success_rate"] = rm["range_success_rate"]
                            result[f"test_set_{split_name}_range_start_mae"] = rm["mean_start_mae"]
                            result[f"test_set_{split_name}_range_end_mae"] = rm["mean_end_mae"]

                            logger.info_rank0(f"    --- Temporal Range Metrics ({rm['range_applicable']} applicable) ---")
                            logger.info_rank0(f"    Range Success Rate: {rm['range_success']}/{rm['range_applicable']} = {rm['range_success_rate']:.4f}")
                            logger.info_rank0(f"    Mean tIoU:          {rm['mean_tiou']:.4f}")
                            logger.info_rank0(f"    Mean Start MAE:     {rm['mean_start_mae']:.2f}s")
                            logger.info_rank0(f"    Mean End MAE:       {rm['mean_end_mae']:.2f}s")

                    logger.info_rank0("=" * 70)

                # Save predictions to JSON
                self._save_predictions_to_json()

        self.score_dict = {"test_set_accuracy": []}
        self.predictions_list = []
        return result

    def _save_predictions_to_json(self) -> None:
        if not self.output_dir or not self.predictions_list:
            return

        import json
        import os
        from datetime import datetime

        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"eval_predictions_{timestamp}.json")

        # Compute summary statistics
        total = len(self.predictions_list)
        correct = sum(1 for p in self.predictions_list if p["matched"] == 1)

        # Get metrics from _dump if available
        metrics = getattr(self, "_metrics_for_json", {})

        output_data = {
            "summary": {
                "total_samples": total,
                "correct": correct,
                "accuracy": metrics.get("accuracy", correct / total if total > 0 else 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1": metrics.get("f1", 0.0),
                "tp": metrics.get("tp", 0),
                "fp": metrics.get("fp", 0),
                "fn": metrics.get("fn", 0),
                "tn": metrics.get("tn", 0),
            },
            "predictions": self.predictions_list
        }

        # Add per-split metrics to JSON output
        per_split_metrics = getattr(self, "_per_split_metrics", {})
        if per_split_metrics:
            output_data["per_split_summary"] = per_split_metrics

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info_rank0(f"[ComputeVideoRewardAccuracy] Saved predictions to {output_file}")
        logger.info_rank0(f"[ComputeVideoRewardAccuracy] Metrics: accuracy={metrics.get('accuracy', 0.0):.4f}, "
                         f"precision={metrics.get('precision', 0.0):.4f}, recall={metrics.get('recall', 0.0):.4f}, "
                         f"f1={metrics.get('f1', 0.0):.4f}")

    def __post_init__(self):
        self._load_split_mapping()
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        logger.info_rank0(f"[ComputeVideoRewardAccuracy] Called with preds shape: {preds.shape}, labels shape: {labels.shape}, use_generate_mode: {self.use_generate_mode}")

        decoded_preds = []
        decoded_labels = []

        if self.use_generate_mode:
            # In generate mode, preds are generated tokens (prompt replaced with pad_token_id)
            # Labels are the full labels including prompt (IGNORE_INDEX) and target
            preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
            labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        else:
            # In logits mode, preds are argmax of logits, need shift alignment
            # logits at position i predict token at position i+1
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

            for i in range(len(preds)):
                # Shift: pred[:-1] predicts label[1:]
                pred_shifted = preds[i, :-1]
                label_shifted = labels[i, 1:]

                # Handle case where preds and labels have different lengths
                min_len = min(len(pred_shifted), len(label_shifted))
                pred_shifted = pred_shifted[:min_len]
                label_shifted = label_shifted[:min_len]

                # Only keep positions where label is not IGNORE_INDEX (i.e., actual target tokens)
                label_mask = label_shifted != IGNORE_INDEX

                # Extract only the valid tokens (where label is not IGNORE_INDEX)
                pred_tokens = pred_shifted[label_mask].astype(np.int64).tolist()
                label_tokens = label_shifted[label_mask].astype(np.int64).tolist()

                # Filter out any invalid token IDs (negative or too large)
                vocab_size = self.tokenizer.vocab_size
                pred_tokens = [t if 0 <= t < vocab_size else pad_token_id for t in pred_tokens]
                label_tokens = [t if 0 <= t < vocab_size else pad_token_id for t in label_tokens]

                decoded_pred = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
                decoded_label = self.tokenizer.decode(label_tokens, skip_special_tokens=True)

                decoded_preds.append(decoded_pred)
                decoded_labels.append(decoded_label)

        logger.info_rank0(f"[ComputeVideoRewardAccuracy] Decoded {len(decoded_preds)} predictions")

        # capture the content in \box{} and compute the reward accuracy
        # Same parsing logic as compute_video_reward_cumulative_accuracy in trainer.py
        for pred, label in zip(decoded_preds, decoded_labels):
            # Parse prediction box with fallback (same as training)
            pred_box_match = re.search(r"\\box(?:ed)?\{(.*?)\}", pred)
            if not pred_box_match:
                pred_box_match = re.search(r"box(?:ed)?\{(.*?)\}", pred)
            # Parse ground truth box (same as training, no fallback)
            label_box_match = re.search(r"\\box(?:ed)?\{(.*?)\}", label)

            pred_box = pred_box_match.group(1) if pred_box_match else None
            label_box = label_box_match.group(1) if label_box_match else None

            matched = 0
            if pred_box is not None and label_box is not None:
                matched = int(pred_box == label_box)
            self.score_dict["test_set_accuracy"].append(matched)

            # Save prediction details if enabled
            if self.save_predictions:
                pred_entry = {
                    "pred_box": pred_box,
                    "label_box": label_box,
                    "matched": matched,
                    "raw_pred": pred,
                    "raw_label": label,
                }
                # Add split info if available (use modulo for DDP padding)
                sample_idx = len(self.predictions_list)
                if self._split_names:
                    pred_entry["split"] = self._split_names[sample_idx % len(self._split_names)]

                # Parse and store time range info if present
                label_range = self._parse_time_range(label)
                pred_range = self._parse_time_range(pred)
                if label_range is not None:
                    pred_entry["label_range"] = list(label_range)
                    pred_entry["pred_range"] = list(pred_range) if pred_range is not None else None

                self.predictions_list.append(pred_entry)

        logger.info_rank0(f"[ComputeVideoRewardAccuracy] Accumulated {len(self.score_dict['test_set_accuracy'])} samples, compute_result={compute_result}")

        if compute_result:
            result = self._dump()
            logger.info_rank0(f"[ComputeVideoRewardAccuracy] Returning result: {result}")
            return result


@dataclass
class ComputeSimilarity:
    r"""Compute text similarity scores and support `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        if compute_result:
            return self._dump()
