"""
Benchmark: tform.audio.gammatone — Gammatone Filterbank
========================================================
TensorForm uses precomputed frequency-domain gammatone kernels
applied as split-complex matmul (no time-domain convolution).
Baseline: TensorForm CPU sequential reference (no published GPU competitor).

Usage
-----
    python benchmarks/audio/bench_gammatone.py
    python benchmarks/audio/bench_gammatone.py --dataset /path/to/wav --n-filters 64
"""

from __future__ import annotations

import sys
import argparse
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from _bench_core import DEVICE, POWER_W, measure, energy_mj, format_report, save_report
from tensorform.bench.audio._io import collect_wav_trials, compute_trial_stats
from tensorform.bench.core import BenchmarkSuite

DATASET_ROOT = "/Users/augustocamargo/Projects/MelFTF/dataset/celeb_vox/wav"


def run_cpu_reference(trials, args):
    """TensorForm sequential CPU baseline (frame-by-frame split-complex filtering)."""
    from tensorform.audio.gammatone import GammatoneOperator
    op = GammatoneOperator(sample_rate=args.sample_rate, n_fft=args.n_fft,
                           hop_length=args.hop_length, n_filters=args.n_filters,
                           f_min=args.f_min, f_max=args.f_max)
    ms, ss = zip(*[measure(op.legacy_reference, s, iterations=args.iterations, warmup=args.warmup)
                   for s in trials])
    mean, std = float(np.mean(ms)), float(np.mean(ss))
    return {"label": "TF legacy        (CPU)", "device_type": "cpu",
            "mean_ms": mean, "std_ms": std, "energy_mj": energy_mj(mean, "cpu")}, None


def run_tensorform(tensors, args):
    """TensorForm accelerated gammatone filterbank (batched split-complex matmul)."""
    from tensorform.audio.gammatone import GammatoneOperator
    op = GammatoneOperator(sample_rate=args.sample_rate, n_fft=args.n_fft,
                           hop_length=args.hop_length, n_filters=args.n_filters,
                           f_min=args.f_min, f_max=args.f_max)
    ms, ss = zip(*[measure(op.accelerate, t, iterations=args.iterations, warmup=args.warmup)
                   for t in tensors])
    mean, std = float(np.mean(ms)), float(np.mean(ss))
    return {"label": f"TensorForm       ({DEVICE.upper()})", "device_type": DEVICE,
            "mean_ms": mean, "std_ms": std, "energy_mj": energy_mj(mean, DEVICE)}, None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     default=DATASET_ROOT)
    p.add_argument("--max-files",   type=int,   default=50)
    p.add_argument("--iterations",  type=int,   default=20)
    p.add_argument("--warmup",      type=int,   default=5)
    p.add_argument("--sample-rate", type=int,   default=16000)
    p.add_argument("--n-fft",       type=int,   default=1024)
    p.add_argument("--hop-length",  type=int,   default=256)
    p.add_argument("--n-filters",   type=int,   default=32)
    p.add_argument("--f-min",       type=float, default=80.0)
    p.add_argument("--f-max",       type=float, default=8000.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    env = BenchmarkSuite.collect_env_metadata(DEVICE)

    print(f"\nLoading up to {args.max_files} files from: {args.dataset}")
    trials = collect_wav_trials(args.dataset, max_files=args.max_files, target_sr=args.sample_rate)
    if not trials:
        print("No files found."); return

    ds = compute_trial_stats(trials, sample_rate=args.sample_rate)
    ds["source"] = args.dataset
    print(f"Loaded {ds['n_files']} trials  |  "
          f"duration: {ds['duration_mean_s']:.2f} ± {ds['duration_std_s']:.2f} s  |  "
          f"size: {ds['size_mean']} ± {ds['size_std']}")
    print("Profiling implementations...\n")

    tensors = [torch.from_numpy(s).to(DEVICE) for s in trials]
    results = []
    for runner, data in [
        (run_cpu_reference, trials),
        (run_tensorform,    tensors),
    ]:
        r, err = runner(data, args)
        if err:
            print(f"  {err}")
        else:
            print(f"  [✓] {r['label']}")
            results.append(r)

    if not results:
        print("No implementations ran."); return

    config_lines = [
        f"n_filters:      {args.n_filters}",
        f"f_min:          {args.f_min} Hz",
        f"f_max:          {args.f_max} Hz",
        f"n_fft:          {args.n_fft}",
        f"hop_length:     {args.hop_length}",
        f"iterations:     {args.iterations}",
        f"warmup:         {args.warmup}",
    ]
    report = format_report(
        results, env, ds, config_lines,
        benchmark_title="tform.audio.gammatone  (Freq-Domain Split-Complex Matmul)",
        baseline_desc="Baseline: TF CPU sequential  |  No published GPU Gammatone competitor",
        started_at=started_at,
    )
    print(report)
    save_report(report, env, "gammatone")


if __name__ == "__main__":
    main()
