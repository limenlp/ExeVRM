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

from __future__ import annotations


class _Dummy:
    def to_dict(self):
        return {}


class _DummyTrainingArgs:
    def __init__(self, metric_for_best_model: str | None = None):
        self.metric_for_best_model = metric_for_best_model


def _make_callback():
    from llamafactory.train.callbacks import ReporterCallback

    return ReporterCallback(_Dummy(), _Dummy(), _Dummy(), _Dummy())


def test_format_metrics_for_wandb_eval_default_group_and_eval_result_top_level():
    cb = _make_callback()

    logs = cb._format_metrics_for_wandb(
        {
            "eval_test_set_accuracy": 0.5,
            "video_reward_cumulative_accuracy": 0.9,  # unprefixed key
            "eval_result": 0.5,
        },
        default_group="eval",
    )

    assert logs["eval/test_set_accuracy"] == 0.5
    assert logs["eval/video_reward_cumulative_accuracy"] == 0.9
    assert logs["eval_result"] == 0.5


def test_format_metrics_for_wandb_train_default_group_keeps_backward_compat():
    cb = _make_callback()

    logs = cb._format_metrics_for_wandb({"foo": 1.0})
    assert logs == {"train/foo": 1.0}


def test_infer_eval_result_prefers_test_set_accuracy_then_metric_for_best_model():
    cb = _make_callback()

    assert cb._infer_eval_result({"eval_test_set_accuracy": 0.7, "eval_loss": 1.0}, _DummyTrainingArgs()) == 0.7
    assert cb._infer_eval_result({"test_set_accuracy": 0.3}, _DummyTrainingArgs()) == 0.3
    assert cb._infer_eval_result({"eval_custom": 0.42}, _DummyTrainingArgs(metric_for_best_model="custom")) == 0.42


def test_on_evaluate_logs_eval_only_metrics_and_keeps_eval_result_top_level():
    cb = _make_callback()

    class _DummyWandb:
        def __init__(self):
            self.run = object()
            self.logged: dict[str, float] | None = None

        def log(self, payload: dict[str, float]):
            self.logged = payload

    class _DummyState:
        is_world_process_zero = True
        global_step = 123

    cb._wandb_enabled = True
    cb._wandb = _DummyWandb()

    cb.on_evaluate(
        _DummyTrainingArgs(),
        _DummyState(),
        control=None,  # unused
        metrics={
            "eval_test_set_accuracy": 0.5,
            "eval_loss": 1.0,
            "video_reward_cumulative_accuracy": 0.9,  # should NOT be logged during eval
        },
    )

    assert cb._wandb.logged is not None
    assert cb._wandb.logged["eval/test_set_accuracy"] == 0.5
    assert cb._wandb.logged["eval_result"] == 0.5
    assert "eval/video_reward_cumulative_accuracy" not in cb._wandb.logged