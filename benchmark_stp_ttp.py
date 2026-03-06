#!/usr/bin/env python3
"""
Benchmark: STP vs TTP vs STP+TTP × frame count for Qwen3-VL training.

Uses a single 101-frame video and varies:
  - Config: STP Only / TTP Only / STP + TTP
  - Frame count: 2, 10, 20, 50

Measures per-step training time and peak GPU memory consumption.

Usage:
    python benchmark_stp_ttp.py
    python benchmark_stp_ttp.py --num-gpus 8 --max-steps 5
    python benchmark_stp_ttp.py --dry-run
"""

import yaml
import subprocess
import os
import sys
import time
import re
import threading
import json
import argparse
import signal
from datetime import datetime
from copy import deepcopy


# ─── Configuration ───────────────────────────────────────────────────────────

BASE_CONFIG_PATH = "/export/home/VideoRM/train/qwen3vl-8B/qwen3vl_8B_rm_original_stp_ttp_highframes.yaml"
DEEPSPEED_ABSOLUTE = "/export/home/LLaMA-Factory/examples/deepspeed/ds_z2_config.json"
BENCHMARK_DATASET = "benchmark_single_video"

CONFIGS = [
    {
        "key": "stp_only",
        "name": "STP Only",
        "overrides": {
            "use_stp": True,
            "use_ttp": False,
        },
    },
    {
        "key": "ttp_only",
        "name": "TTP Only",
        "overrides": {
            "use_stp": False,
            "use_ttp": True,
        },
    },
    {
        "key": "stp_ttp",
        "name": "STP + TTP",
        "overrides": {
            "use_stp": True,
            "use_ttp": True,
        },
    },
]

FRAME_COUNTS = [2, 10, 20, 50]

# Keys to strip from config during benchmark (eval-related)
EVAL_KEYS_TO_REMOVE = [
    "eval_dataset", "eval_strategy", "eval_steps",
    "per_device_eval_batch_size", "eval_accumulation_steps",
    "use_vllm_eval", "vllm_eval_max_model_len",
    "vllm_eval_gpu_util", "vllm_eval_tp_size", "vllm_eval_batch_size",
    "predict_with_generate",
]


# ─── GPU Memory Monitor ─────────────────────────────────────────────────────

class GPUMemoryMonitor:
    """Background thread polling nvidia-smi. Records per-GPU peak memory (MiB)."""

    def __init__(self, num_gpus: int, interval: float = 0.3):
        self.num_gpus = num_gpus
        self.interval = interval
        self.peak_memory: dict[int, int] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self.peak_memory.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[int, int]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return dict(self.peak_memory)

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=index,memory.used",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=3,
                )
                for line in out.strip().splitlines():
                    idx_s, mem_s = line.split(",")
                    idx, mem = int(idx_s.strip()), int(mem_s.strip())
                    if idx < self.num_gpus:
                        self.peak_memory[idx] = max(self.peak_memory.get(idx, 0), mem)
            except Exception:
                pass


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_baseline_gpu_memory(num_gpus: int) -> dict[int, int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        baseline = {}
        for line in out.strip().splitlines():
            idx_s, mem_s = line.split(",")
            idx, mem = int(idx_s.strip()), int(mem_s.strip())
            if idx < num_gpus:
                baseline[idx] = mem
        return baseline
    except Exception:
        return {}


def clear_gpu_cache():
    subprocess.run(
        [sys.executable, "-c",
         "import torch; [torch.cuda.empty_cache() for _ in range(torch.cuda.device_count())]"],
        capture_output=True, timeout=30,
    )
    time.sleep(3)


def load_base_config() -> dict:
    with open(BASE_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def make_benchmark_config(base: dict, config: dict, num_frames: int,
                          output_root: str, max_steps: int) -> dict:
    """Clone base config, apply config overrides + frame count."""
    cfg = deepcopy(base)
    cfg.update(config["overrides"])

    # Frame count control
    cfg["video_maxlen"] = num_frames

    # Use single-video benchmark dataset
    cfg["dataset"] = BENCHMARK_DATASET

    # Benchmark tweaks
    cfg["max_steps"] = max_steps
    cfg["do_eval"] = False
    cfg["report_to"] = "none"
    cfg["plot_loss"] = False
    cfg["save_steps"] = 999999
    cfg["save_strategy"] = "no"
    cfg["logging_steps"] = 1
    cfg["overwrite_output_dir"] = True
    cfg["deepspeed"] = DEEPSPEED_ABSOLUTE

    exp_key = f"{config['key']}_{num_frames}f"
    cfg["output_dir"] = os.path.join(output_root, exp_key)
    cfg["run_name"] = f"benchmark_{exp_key}"

    for k in EVAL_KEYS_TO_REMOVE:
        cfg.pop(k, None)

    return cfg


def parse_step_times(output: str) -> list[float]:
    """Extract per-step wall times from tqdm progress bars."""
    times_s = re.findall(r"\d+/\d+\s+\[[\d:]+<[\d:]+,\s*([\d.]+)s/it\]", output)
    if times_s:
        return [float(t) for t in times_s]
    rates = re.findall(r"\d+/\d+\s+\[[\d:]+<[\d:]+,\s*([\d.]+)it/s\]", output)
    if rates:
        return [1.0 / float(r) for r in rates]
    return []


def parse_train_runtime(output: str) -> float | None:
    m = re.search(r"'train_runtime':\s*([\d.]+)", output)
    return float(m.group(1)) if m else None


def parse_train_loss(output: str) -> float | None:
    m = re.search(r"'train_loss':\s*([\d.]+)", output)
    return float(m.group(1)) if m else None


# ─── Run one experiment ──────────────────────────────────────────────────────

def run_experiment(config_path: str, num_gpus: int, label: str) -> dict:
    env = os.environ.copy()
    env["FORCE_TORCHRUN"] = "1"
    env["NNODES"] = "1"
    env["NPROC_PER_NODE"] = str(num_gpus)

    cmd = ["llamafactory-cli", "train", config_path]

    print(f"\n{'=' * 70}")
    print(f"  Experiment: {label}")
    print(f"  Config:     {config_path}")
    print(f"  GPUs:       {num_gpus}")
    print(f"{'=' * 70}\n")

    gpu_monitor = GPUMemoryMonitor(num_gpus=num_gpus, interval=0.3)
    gpu_monitor.start()
    wall_start = time.monotonic()

    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    lines: list[str] = []
    try:
        for line in proc.stdout:
            lines.append(line)
            sys.stdout.write(line)
            sys.stdout.flush()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=30)
        raise

    proc.wait()
    wall_elapsed = time.monotonic() - wall_start
    peak_mem = gpu_monitor.stop()

    return {
        "returncode": proc.returncode,
        "output": "".join(lines),
        "wall_time_s": round(wall_elapsed, 2),
        "peak_gpu_memory_mib": peak_mem,
    }


# ─── Result formatting ──────────────────────────────────────────────────────

def format_results(results: dict, max_steps: int, baseline_mem: dict):
    """Print a frames × config result table."""
    hline = "=" * 100
    print(f"\n{hline}")
    print(f"{'BENCHMARK RESULTS':^100}")
    print(f"{'(single video: 012b12de...failure.mp4, 101 raw frames, 1080p)':^100}")
    print(f"{hline}")

    # ── Table 1: Step time ──
    print(f"\n{'Per-step training time (seconds)':^100}")
    print("-" * 100)
    header = f"{'Frames':<10}"
    for cfg in CONFIGS:
        header += f"{cfg['name']:<30}"
    print(header)
    print("-" * 100)

    time_data = {}  # (config_key, nframes) -> avg_step
    mem_data = {}   # (config_key, nframes) -> peak_mib

    for nf in FRAME_COUNTS:
        row = f"{nf:<10}"
        for cfg in CONFIGS:
            exp_key = f"{cfg['key']}_{nf}f"
            if exp_key in results and results[exp_key]["returncode"] == 0:
                r = results[exp_key]
                step_times = parse_step_times(r["output"])
                runtime = parse_train_runtime(r["output"])
                if step_times and len(step_times) > 1:
                    avg_step = sum(step_times[1:]) / len(step_times[1:])
                elif runtime:
                    avg_step = runtime / max_steps
                else:
                    avg_step = r["wall_time_s"] / max_steps
                time_data[(cfg["key"], nf)] = avg_step
                row += f"{avg_step:<30.2f}"
            else:
                row += f"{'FAILED':<30}"
        print(row)

    # ── Table 2: GPU memory ──
    print(f"\n{'Peak GPU memory per GPU (MiB)':^100}")
    print("-" * 100)
    header = f"{'Frames':<10}"
    for cfg in CONFIGS:
        header += f"{cfg['name']:<30}"
    print(header)
    print("-" * 100)

    for nf in FRAME_COUNTS:
        row = f"{nf:<10}"
        for cfg in CONFIGS:
            exp_key = f"{cfg['key']}_{nf}f"
            if exp_key in results and results[exp_key]["returncode"] == 0:
                r = results[exp_key]
                peak_mib = max(r["peak_gpu_memory_mib"].values()) if r["peak_gpu_memory_mib"] else 0
                mem_data[(cfg["key"], nf)] = peak_mib
                peak_gib = peak_mib / 1024.0
                row += f"{peak_mib} ({peak_gib:.1f} GiB){'':<10}"
            else:
                row += f"{'FAILED':<30}"
        print(row)

    # ── Table 3: GPU memory in GiB (cleaner) ──
    print(f"\n{'Peak GPU memory per GPU (GiB)':^100}")
    print("-" * 100)
    header = f"{'Frames':<10}"
    for cfg in CONFIGS:
        header += f"{cfg['name']:<30}"
    print(header)
    print("-" * 100)

    for nf in FRAME_COUNTS:
        row = f"{nf:<10}"
        for cfg in CONFIGS:
            exp_key = f"{cfg['key']}_{nf}f"
            if exp_key in results and results[exp_key]["returncode"] == 0:
                r = results[exp_key]
                peak_mib = max(r["peak_gpu_memory_mib"].values()) if r["peak_gpu_memory_mib"] else 0
                row += f"{peak_mib / 1024.0:<30.2f}"
            else:
                row += f"{'FAILED':<30}"
        print(row)

    print(hline)

    if baseline_mem:
        bl = max(baseline_mem.values())
        print(f"\nBaseline (idle) GPU memory: {bl} MiB / {bl/1024:.2f} GiB per GPU")

    return {"time": time_data, "memory": mem_data}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark STP / TTP × frame count")
    parser.add_argument("--num-gpus", type=int, default=8,
                        help="Number of GPUs (default: 8)")
    parser.add_argument("--max-steps", type=int, default=5,
                        help="Training steps per experiment (default: 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate configs only, don't run training")
    parser.add_argument("--cooldown", type=int, default=15,
                        help="Seconds to wait between experiments (default: 15)")
    parser.add_argument("--frames", type=int, nargs="+", default=None,
                        help="Override frame counts (default: 2 10 20 50)")
    parser.add_argument("--configs", type=str, nargs="+", default=None,
                        choices=["stp_only", "ttp_only", "stp_ttp"],
                        help="Run only specific configs")
    args = parser.parse_args()

    frame_counts = args.frames if args.frames else FRAME_COUNTS
    configs = CONFIGS
    if args.configs:
        configs = [c for c in CONFIGS if c["key"] in args.configs]

    # Validate
    if not os.path.exists(BASE_CONFIG_PATH):
        sys.exit(f"Base config not found: {BASE_CONFIG_PATH}")
    if not os.path.exists(DEEPSPEED_ABSOLUTE):
        sys.exit(f"DeepSpeed config not found: {DEEPSPEED_ABSOLUTE}")

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("benchmark_results", f"run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    total_experiments = len(frame_counts) * len(configs)
    print(f"Benchmark output directory: {out_dir}")
    print(f"Experiments: {len(configs)} configs × {len(frame_counts)} frame counts = {total_experiments} runs")
    print(f"Frame counts: {frame_counts}")
    print(f"Configs: {[c['name'] for c in configs]}")

    # Load base config
    base_config = load_base_config()

    # Generate all configs
    config_paths = {}
    for nf in frame_counts:
        for cfg in configs:
            exp_key = f"{cfg['key']}_{nf}f"
            bench_cfg = make_benchmark_config(
                base_config, cfg, nf, out_dir, args.max_steps)
            cfg_path = os.path.join(out_dir, f"{exp_key}.yaml")
            with open(cfg_path, "w") as f:
                yaml.dump(bench_cfg, f, default_flow_style=False, sort_keys=False)
            config_paths[exp_key] = cfg_path
            print(f"  Generated: {cfg_path}")

    if args.dry_run:
        print("\n[Dry run] Configs generated. Exiting.")
        return

    # Pre-flight
    print(f"\nGPU configuration: {args.num_gpus}x GPUs, {args.max_steps} steps/experiment")
    baseline_mem = get_baseline_gpu_memory(args.num_gpus)
    if baseline_mem:
        print(f"Baseline GPU memory: { {f'GPU{k}': f'{v} MiB' for k, v in sorted(baseline_mem.items())} }")

    # Run all experiments: iterate by frame count (outer) × config (inner)
    all_results = {}
    run_idx = 0
    for nf in frame_counts:
        for cfg in configs:
            exp_key = f"{cfg['key']}_{nf}f"
            label = f"{cfg['name']} | {nf} frames"

            if run_idx > 0:
                print(f"\nCooling down for {args.cooldown}s...")
                clear_gpu_cache()
                time.sleep(args.cooldown)

            print(f"\n[{run_idx + 1}/{total_experiments}]")
            result = run_experiment(config_paths[exp_key], args.num_gpus, label)

            if result["returncode"] != 0:
                print(f"\n*** {label} FAILED (exit code {result['returncode']}) ***")
            all_results[exp_key] = result

            # Save log
            with open(os.path.join(out_dir, f"{exp_key}.log"), "w") as f:
                f.write(result["output"])

            # Save result summary (without full output)
            result_summary = {k: v for k, v in result.items() if k != "output"}
            with open(os.path.join(out_dir, f"{exp_key}_result.json"), "w") as f:
                json.dump(result_summary, f, indent=2, default=str)

            run_idx += 1

    # Print final summary table
    tables = format_results(all_results, args.max_steps, baseline_mem)

    # Save summary
    summary = {
        "timestamp": timestamp,
        "num_gpus": args.num_gpus,
        "max_steps": args.max_steps,
        "frame_counts": frame_counts,
        "configs": [c["name"] for c in configs],
        "base_config": BASE_CONFIG_PATH,
        "video": "012b12de67e84411bc68990ad57d839a_failure.mp4 (101 frames, 1080p)",
        "baseline_gpu_memory_mib": baseline_mem,
        "time_data": {f"{k[0]}_{k[1]}f": v for k, v in tables["time"].items()},
        "memory_data": {f"{k[0]}_{k[1]}f": v for k, v in tables["memory"].items()},
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to: {out_dir}/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
