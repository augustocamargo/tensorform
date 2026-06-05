"""
Benchmark: tform.audio.cqt vs contenders — Constant-Q Transform
================================================================
TensorForm uses precomputed frequency-domain kernel matmul.
Contenders: librosa (CPU), nnAudio.

Usage
-----
    python benchmarks/audio/bench_cqt.py
    python benchmarks/audio/bench_cqt.py --dataset /path/to/wav --n-bins 72 --f-min 65.4
"""

from __future__ import annotations

import sys
import math
import argparse
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from _bench_core import DEVICE, POWER_W, measure, energy_mj, gpu_sanity, format_report, save_report
from tensorform.bench.audio._io import collect_wav_trials, compute_trial_stats
from tensorform.bench.core import BenchmarkSuite

DATASET_ROOT = "/Users/augustocamargo/Projects/MelFTF/dataset/celeb_vox/wav"


def run_librosa(trials, args):
    try:
        import librosa
    except ImportError:
        return None, "[–] librosa not installed  →  pip install librosa"
    def _fn(s):
        return np.abs(librosa.cqt(
            s, sr=args.sample_rate, hop_length=args.hop_length,
            n_bins=args.n_bins, bins_per_octave=args.bins_per_octave,
            fmin=args.f_min))
    ms, ss = zip(*[measure(_fn, s, iterations=args.iterations, warmup=args.warmup) for s in trials])
    mean, std = float(np.mean(ms)), float(np.mean(ss))
    return {"label": "librosa CQT      (CPU)", "device_type": "cpu",
            "mean_ms": mean, "std_ms": std, "energy_mj": energy_mj(mean, "cpu")}, None


def run_nnaudio(tensors, args):
    tb = [t.unsqueeze(0) for t in tensors]
    try:
        from nnAudio.Spectrogram import CQT
    except ImportError:
        return None, "[–] nnAudio not installed  →  pip install nnAudio"
    nn = CQT(sr=args.sample_rate, hop_length=args.hop_length, n_bins=args.n_bins,
             bins_per_octave=args.bins_per_octave, fmin=args.f_min).to(DEVICE)
    nn_cpu = CQT(sr=args.sample_rate, hop_length=args.hop_length, n_bins=args.n_bins,
                 bins_per_octave=args.bins_per_octave, fmin=args.f_min)
    gpu_ok = gpu_sanity(nn, nn_cpu, tb)
    dev = DEVICE if gpu_ok else "cpu"
    ms, ss = zip(*[measure(nn, t, iterations=args.iterations, warmup=args.warmup) for t in tb])
    mean, std = float(np.mean(ms)), float(np.mean(ss))
    suffix = f"({DEVICE.upper()})" if gpu_ok else "(CPU fallback)"
    return {"label": f"nnAudio CQT      {suffix}", "device_type": dev,
            "mean_ms": mean, "std_ms": std, "energy_mj": energy_mj(mean, dev)}, None


def run_tensorform(tensors, args):
    from tensorform.audio.cqt import CQTOperator
    op = CQTOperator(sample_rate=args.sample_rate, hop_length=args.hop_length,
                     n_bins=args.n_bins, bins_per_octave=args.bins_per_octave,
                     f_min=args.f_min)
    ms, ss = zip(*[measure(op.accelerate, t, iterations=args.iterations, warmup=args.warmup) for t in tensors])
    mean, std = float(np.mean(ms)), float(np.mean(ss))
    return {"label": f"TensorForm CQT   ({DEVICE.upper()})", "device_type": DEVICE,
            "mean_ms": mean, "std_ms": std, "energy_mj": energy_mj(mean, DEVICE)}, None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",          default=DATASET_ROOT)
    p.add_argument("--max-files",        type=int,   default=50)
    p.add_argument("--iterations",       type=int,   default=20)
    p.add_argument("--warmup",           type=int,   default=5)
    p.add_argument("--sample-rate",      type=int,   default=16000)
    p.add_argument("--hop-length",       type=int,   default=512)
    p.add_argument("--n-bins",           type=int,   default=60)
    p.add_argument("--bins-per-octave",  type=int,   default=12)
    p.add_argument("--f-min",            type=float, default=130.8)
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
        (run_librosa,    trials),
        (run_nnaudio,    tensors),
        (run_tensorform, tensors),
    ]:
        r, err = runner(data, args)
        if err:
            print(f"  {err}")
        else:
            print(f"  [✓] {r['label']}")
            results.append(r)

    if not results:
        print("No implementations ran."); return

    Q = 1.0 / (2.0 ** (1.0 / args.bins_per_octave) - 1.0)
    n_fft = int(2 ** math.ceil(math.log2(Q * args.sample_rate / args.f_min)))
    config_lines = [
        f"n_bins:         {args.n_bins}",
        f"bins_per_octave:{args.bins_per_octave}",
        f"f_min:          {args.f_min} Hz",
        f"hop_length:     {args.hop_length}",
        f"n_fft (auto):   {n_fft}",
        f"iterations:     {args.iterations}",
        f"warmup:         {args.warmup}",
    ]
    report = format_report(
        results, env, ds, config_lines,
        benchmark_title="tform.audio.cqt  (Frequency-Domain Kernel Matmul)",
        baseline_desc="Baselines: variable-window / 1D conv  |  TensorForm: static freq-domain matmul",
        started_at=started_at,
    )
    print(report)
    save_report(report, env, "cqt")


if __name__ == "__main__":
    main()
