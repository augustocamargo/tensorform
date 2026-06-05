"""
Benchmark: tform.audio.melt vs contenders — Mel Spectrogram
============================================================
TensorForm uses Direct Mel Projection (no STFT).
Contenders: librosa (CPU), torchaudio, nnAudio.

Usage
-----
    python benchmarks/audio/bench_mel.py
    python benchmarks/audio/bench_mel.py --dataset /path/to/wav --n-fft 400 --n-mels 80
"""

from __future__ import annotations

import sys
import argparse
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))      # benchmarks/audio/
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
        return librosa.feature.melspectrogram(
            y=s, sr=args.sample_rate, n_fft=args.n_fft,
            hop_length=args.hop_length, n_mels=args.n_mels)
    ms, ss = zip(*[measure(_fn, s, iterations=args.iterations, warmup=args.warmup) for s in trials])
    mean, std = float(np.mean(ms)), float(np.mean(ss))
    return {"label": "librosa          (CPU)", "device_type": "cpu",
            "mean_ms": mean, "std_ms": std, "energy_mj": energy_mj(mean, "cpu")}, None


def run_torchaudio(tensors, args):
    try:
        import torchaudio
    except ImportError:
        return None, "[–] torchaudio not installed  →  pip install torchaudio"
    ta = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sample_rate, n_fft=args.n_fft,
        hop_length=args.hop_length, n_mels=args.n_mels).to(DEVICE)
    ta_cpu = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sample_rate, n_fft=args.n_fft,
        hop_length=args.hop_length, n_mels=args.n_mels)
    gpu_ok = gpu_sanity(ta, ta_cpu, tensors)
    dev = DEVICE if gpu_ok else "cpu"
    ms, ss = zip(*[measure(ta, t, iterations=args.iterations, warmup=args.warmup) for t in tensors])
    mean, std = float(np.mean(ms)), float(np.mean(ss))
    suffix = f"({DEVICE.upper()})" if gpu_ok else "(CPU fallback)"
    return {"label": f"torchaudio       {suffix}", "device_type": dev,
            "mean_ms": mean, "std_ms": std, "energy_mj": energy_mj(mean, dev)}, None


def run_nnaudio(tensors, args):
    tb = [t.unsqueeze(0) for t in tensors]
    try:
        from nnAudio.Spectrogram import MelSpectrogram as nnMel
    except ImportError:
        return None, "[–] nnAudio not installed  →  pip install nnAudio"
    nn = nnMel(sr=args.sample_rate, n_fft=args.n_fft,
               hop_length=args.hop_length, n_mels=args.n_mels).to(DEVICE)
    nn_cpu = nnMel(sr=args.sample_rate, n_fft=args.n_fft,
                   hop_length=args.hop_length, n_mels=args.n_mels)
    gpu_ok = gpu_sanity(nn, nn_cpu, tb)
    dev = DEVICE if gpu_ok else "cpu"
    ms, ss = zip(*[measure(nn, t, iterations=args.iterations, warmup=args.warmup) for t in tb])
    mean, std = float(np.mean(ms)), float(np.mean(ss))
    suffix = f"({DEVICE.upper()})" if gpu_ok else "(CPU fallback)"
    return {"label": f"nnAudio          {suffix}", "device_type": dev,
            "mean_ms": mean, "std_ms": std, "energy_mj": energy_mj(mean, dev)}, None


def run_tensorform(tensors, args):
    from tensorform.audio.melt import MelTOperator
    op = MelTOperator(sample_rate=args.sample_rate, n_fft=args.n_fft,
                      hop_length=args.hop_length, n_mels=args.n_mels)
    ms, ss = zip(*[measure(op.accelerate, t, iterations=args.iterations, warmup=args.warmup) for t in tensors])
    mean, std = float(np.mean(ms)), float(np.mean(ss))
    return {"label": f"TensorForm MelT  ({DEVICE.upper()})", "device_type": DEVICE,
            "mean_ms": mean, "std_ms": std, "energy_mj": energy_mj(mean, DEVICE)}, None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     default=DATASET_ROOT)
    p.add_argument("--max-files",   type=int,   default=50)
    p.add_argument("--iterations",  type=int,   default=20)
    p.add_argument("--warmup",      type=int,   default=5)
    p.add_argument("--sample-rate", type=int,   default=16000)
    p.add_argument("--n-fft",       type=int,   default=512)
    p.add_argument("--hop-length",  type=int,   default=160)
    p.add_argument("--n-mels",      type=int,   default=26)
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
        (run_torchaudio, tensors),
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

    config_lines = [
        f"n_fft:          {args.n_fft}",
        f"hop_length:     {args.hop_length}",
        f"n_mels:         {args.n_mels}",
        f"iterations:     {args.iterations}",
        f"warmup:         {args.warmup}",
    ]
    report = format_report(
        results, env, ds, config_lines,
        benchmark_title="tform.audio.melt  (Direct Mel Projection)",
        baseline_desc="Baselines: STFT+Mel  |  TensorForm: Direct Mel Projection (no STFT)",
        started_at=started_at,
    )
    print(report)
    save_report(report, env, "mel")


if __name__ == "__main__":
    main()
