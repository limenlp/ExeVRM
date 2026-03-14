# ExeVRM: Execution Video Reward Modeling

Official implementation for the paper `Video-Based Reward Modeling for Computer-Use Agents`

ExeVRM is a training framework for execution video reward models, built on top of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). It fine-tunes vision-language models (e.g., Qwen3-VL) to judge whether a computer-use agent's video trajectory successfully completes a given task.

ExeVRM introduces two key token pruning techniques — **STP (Spatial Token Pruning)** and **TTP (Temporal Token Pruning)** — that dramatically reduce the number of visual tokens in long video inputs, enabling efficient training and inference on execution videos with up to 50+ frames at 720p resolution.

## Quick Start

### Installation

```bash
git clone https://github.com/lime-nlp/ExeVRM.git
cd ExeVRM
pip install -e ".[torch,metrics,deepspeed,liger-kernel,vllm]"
```

### Dataset preparation

- Download dataset from [`lime-nlp/ExeVR-53K`](https://huggingface.co/datasets/lime-nlp/ExeVR-53k) and follow the instructions to reassemble the training set videos.
- Use `replace_video_prefix.py` to update all video paths in the annotation files (`exevr53k.jsonl` and `exevrbench.jsonl`):

```bash
# Replace the default prefix (/export/home/ExeVR_53k) with a new one
python replace_video_prefix.py /new/path/to/ExeVR_53k

# Or specify a custom old prefix
python replace_video_prefix.py --old_prefix /old/path/to/ExeVR_53k /new/path/to/ExeVR_53k
```

### Training

```bash
llamafactory-cli train qwen3vl.yaml
```

### Evaluation

```bash
llamafactory-cli train qwen3vl_test.yaml
```

## Core Concepts: STP & TTP

Computer-use execution videos are highly redundant — most screen regions remain static across frames, and consecutive frames are nearly identical during periods of inactivity. ExeVRM exploits these two types of redundancy through spatial and temporal token pruning.

### STP (Spatial Token Pruning)

STP reduces the number of visual tokens **within each frame** by identifying and merging spatially similar patches.

**How it works:**

1. **Build a UI graph**: For each frame, STP compares adjacent patches (at either patch-level 48x48 or token-level 24x24 resolution). Patches with L2 distance below `stp_threshold` are connected.
2. **Connected component analysis**: A Union-Find algorithm groups connected patches into components. Each component represents a visually uniform UI region (e.g., a toolbar, background area, or blank space).
3. **Selective pruning**: Components larger than `stp_large_comp_threshold` are aggressively pruned (they represent large, uniform regions like backgrounds). Within remaining components, tokens are sampled according to `stp_skip_ratio`.

**Key parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_stp` | Enable spatial token pruning | `false` |
| `stp_mode` | Reduction strategy: `forward_removal` (recommended), `masking`, or `preprocess` | `forward_removal` |
| `stp_threshold` | L2 distance threshold for patch similarity. Higher = more aggressive pruning | `3.0` |
| `stp_skip_ratio` | Fraction of tokens to skip within each component (0.0-1.0) | `0.0` |
| `stp_large_comp_threshold` | Components larger than this are fully pruned (0 = disabled) | `10` |
| `stp_patch_level` | Analyze at patch level for finer granularity | `true` |
| `use_raw_frames_in_stp` | Apply STP on raw temporal frames and OR keep masks | `true` |

**STP modes explained:**

- **`forward_removal`** (recommended): Tokens are physically removed after position IDs are computed. Produces correct positional encoding and true memory savings.
- **`masking`**: Pruned token embeddings are zeroed out but not removed. Preserves grid structure but no memory savings in attention.
- **`preprocess`**: Applies 2x2 pooling in the data pipeline, reducing grid dimensions. Simplest but least flexible.

### TTP (Temporal Token Pruning)

TTP reduces the number of visual tokens **across frames** by detecting and removing temporally duplicated patches.

**How it works:**

1. **Frame-by-frame comparison**: For each spatial position, TTP computes the similarity between the current frame's patch and a reference frame's patch.
2. **Duplicate detection**: If the similarity exceeds `ttp_threshold`, the patch is marked as a temporal duplicate.
3. **Token removal**: Duplicate patches are removed from the sequence, keeping only the first occurrence.

**Key parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_ttp` | Enable temporal token pruning | `false` |
| `ttp_threshold` | Similarity threshold. For cosine: higher = more aggressive removal | `0.9999` |
| `ttp_similarity_metric` | Similarity metric: `cosine` (recommended), `l2`, or `l1` | `cosine` |
| `ttp_comparison_mode` | `reference` (compare with last kept frame, more aggressive) or `consecutive` (adjacent frames only) | `reference` |
| `ttp_min_run_length` | Minimum consecutive duplicates to trigger removal (only for `consecutive` mode) | `2` |

### Combining STP + TTP

STP and TTP are complementary and can be used together for maximum token reduction:

- **STP** removes spatial redundancy within each frame (e.g., large uniform backgrounds)
- **TTP** removes temporal redundancy across frames (e.g., static screen regions between actions)

When combined, they can achieve 40-60% token reduction while maintaining reward prediction quality.

## Configuration Reference

### Example Training Config (`qwen3vl.yaml`)

```yaml
### Model
model_name_or_path: Qwen/Qwen3-VL-8B-Instruct
image_max_pixels: 921600     # 1280x720
video_fps: 1.0
video_maxlen: 50
video_max_pixels: 921600     # 1280x720
trust_remote_code: true

### STP (Spatial Token Pruning)
use_stp: true
stp_mode: forward_removal
stp_patch_level: true
stp_threshold: 3
stp_skip_ratio: 0
stp_large_comp_threshold: 10
use_raw_frames_in_stp: true

### TTP (Temporal Token Pruning)
use_ttp: true
ttp_threshold: 0.9999
ttp_similarity_metric: cosine

### Training
stage: sft
do_train: true
finetuning_type: full
freeze_vision_tower: true
freeze_multi_modal_projector: true
deepspeed: examples/deepspeed/ds_z2_config.json

### Dataset
dataset: osworld_agentnet_reward,scalecua_reward
template: qwen3_vl_nothink
cutoff_len: 128000

### Hyperparameters
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 5.0e-6
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
flash_attn: fa2
```

### Dataset Format

Training data follows the ShareGPT conversation format. Each sample contains a video of an agent's screen recording and a binary reward label (`\box{correct}` or `\box{incorrect}`):

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "<video>Given a user task and a computer-using video recording, evaluate whether the user completes the task or not. Reply your judgement in the \\box{}.\nIf the video correctly completes the task, reply \\box{correct}. Otherwise, reply \\box{incorrect}.\n\n# User Task\nChange the slide background to purple. Put the title in the Notes section.\n"
    },
    {
      "from": "gpt",
      "value": "\\box{incorrect}"
    }
  ],
  "videos": ["/path/to/agent_trajectory_video.mp4"]
}
```

A successful execution example:

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "<video>Given a user task and a computer-using video recording, evaluate whether the user completes the task or not. Reply your judgement in the \\box{}.\nIf the video correctly completes the task, reply \\box{correct}. Otherwise, reply \\box{incorrect}.\n\n# User Task\nHelp me change the default save folder for my recordings to the Desktop\n"
    },
    {
      "from": "gpt",
      "value": "\\box{correct}"
    }
  ],
  "videos": ["/path/to/agent_trajectory_video.mp4"]
}
```

ScaleCUA data additionally includes a timestamp range indicating where the agent deviates from the instruction:

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "<video>Given a user task and a computer-using video recording, evaluate whether the user completes the task or not. Reply your judgement in the \\box{}.\nIf the video correctly completes the task, reply \\box{correct}. Otherwise, reply \\box{incorrect}. \nIf the video does not complete the task (i.e., incorrect), please provide the timestemp range, i.e., from <[time_start] seconds> to <[time_end] seconds>, of the video that deviates from the user's instruction.\n\n# User Task\nFind the best-rated restaurant around CMU main campus\n"
    },
    {
      "from": "gpt",
      "value": "\\box{incorrect}\nThe video deviates from the user's instruction between <3.0 seconds> and <4.0 seconds>."
    }
  ],
  "videos": ["/path/to/agent_trajectory_video.mp4"]
}
```

Datasets are registered in `data/dataset_info.json`. Per-dataset STP/TTP overrides are supported via the `use_stp` and `use_ttp` fields (set to `null` to inherit from the global config, or `false` to disable for evaluation sets).

## Project Structure

```
ExeVRM/
├── qwen3vl.yaml                  # Training config (STP + TTP enabled)
├── qwen3vl_test.yaml             # Evaluation config
├── src/llamafactory/
│   ├── model/model_utils/
│   │   ├── stp.py                # Spatial Token Pruning implementation
│   │   └── ttp.py                # Temporal Token Pruning implementation
│   ├── model/patcher.py          # Model patching for STP/TTP integration
│   ├── data/mm_plugin.py         # Vision data processing pipeline
│   ├── hparams/model_args.py     # STP/TTP parameter definitions
│   └── train/sft/trainer.py      # Training loop with STP/TTP label handling
├── data/
│   ├── dataset_info.json         # Dataset registry
│   ├── osworld_agentnet_training/
│   ├── scalecua_training/
│   └── *.jsonl                   # Test/eval sets
└── save_models/                  # Saved checkpoints
```

## Debugging

Enable debug logging for token removal diagnostics:

```yaml
debug_token_removal: true
```

Per-GPU logs are written to `/tmp/ttp_forward_debug_rank{rank}.log`.

## Acknowledgements

ExeVRM is built on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) by hiyouga et al.

## Citation
If you use ExeVRM/ExeVR-53k in your research, please cite our work:
```
@misc{song2026videobasedrewardmodelingcomputeruse,
      title={Video-Based Reward Modeling for Computer-Use Agents}, 
      author={Linxin Song and Jieyu Zhang and Huanxin Sheng and Taiwei Shi and Gupta Rahul and Yang Liu and Ranjay Krishna and Jian Kang and Jieyu Zhao},
      year={2026},
      eprint={2603.10178},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.10178}, 
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
