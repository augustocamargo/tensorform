"""
Benchmark: tform.audio.melt vs contenders — Mel Spectrogram
============================================================
Unified benchmark combining full system telemetry, multi-contender latency
ranking, and energy estimation across all implementations.

Contenders: librosa (CPU), torchaudio, nnAudio, TensorForm MelT.
TensorForm uses Direct Mel Projection (no STFT).

Install optional contenders:
    pip install librosa nnAudio torchaudio

Usage
-----
    python benchmarks/audio/bench_mel.py
    python benchmarks/audio/bench_mel.py --dataset /work/ms/datasets/Libri6000 \
        --n-fft 400 --n-mels 80 --iterations 20 --warmup 5
"""

from __future__ import annotations

import sys
import os
import socket
import time
import platform
import re
import argparse
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from tensorform._device import detect_device
from tensorform.bench.audio._io import collect_wav_trials, compute_trial_stats
from tensorform.bench.core import BenchmarkSuite

DATASET_ROOT = "/Users/augustocamargo/Projects/MelFTF/dataset/celeb_vox/wav"
DEVICE = detect_device()

# Power model (W) for energy estimation
POWER_W = {"mps": 15.0, "cuda": 45.0, "cpu": 45.0}   # CUDA refined via NVML if available


# ── Timing ────────────────────────────────────────────────────────────────────

def sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


def measure(fn, *args, iterations: int = 20, warmup: int = 5):
    """Returns (mean_ms, std_ms)."""
    for _ in range(warmup):
        fn(*args)
    sync()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn(*args)
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


def energy_mj(mean_ms: float, device_type: str) -> float:
    """E (mJ) = P (W) × t (s) × 1000."""
    pw = POWER_W.get(device_type, 45.0)
    return pw * mean_ms / 1000.0 * 1000.0


# ── Contender runners ─────────────────────────────────────────────────────────

def _gpu_sanity(fn_gpu, fn_cpu, tensors, device_type, W=2, N=5):
    """Returns True if GPU run is >15% faster than CPU reference."""
    ms_cpu, _ = zip(*[measure(fn_cpu, t.cpu(), iterations=N, warmup=W) for t in tensors[:5]])
    ms_gpu, _ = zip(*[measure(fn_gpu, t,     iterations=N, warmup=W) for t in tensors[:5]])
    return float(np.mean(ms_gpu)) < float(np.mean(ms_cpu)) * 0.85


def run_librosa(trials, args):
    try:
        import librosa
    except ImportError:
        return None, "[–] librosa not installed  →  pip install librosa"
    def _fn(s):
        return librosa.feature.melspectrogram(
            y=s, sr=args.sample_rate, n_fft=args.n_fft,
            hop_length=args.hop_length, n_mels=args.n_mels
        )
    ms, ss = zip(*[measure(_fn, s, iterations=args.iterations, warmup=args.warmup) for s in trials])
    mean, std = float(np.mean(ms)), float(np.mean(ss))
    return {"label": "librosa", "device_type": "cpu", "gpu": False,
            "mean_ms": mean, "std_ms": std, "energy_mj": energy_mj(mean, "cpu")}, None


def run_torchaudio(tensors, args):
    try:
        import torchaudio
    except ImportError:
        return None, "[–] torchaudio not installed  →  pip install torchaudio"
    ta = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sample_rate, n_fft=args.n_fft,
        hop_length=args.hop_length, n_mels=args.n_mels,
    ).to(DEVICE)
    ta_cpu = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sample_rate, n_fft=args.n_fft,
        hop_length=args.hop_length, n_mels=args.n_mels,
    )
    gpu_ok = _gpu_sanity(ta, ta_cpu, tensors, DEVICE)
    dev = DEVICE if gpu_ok else "cpu"
    ms, ss = zip(*[measure(ta, t, iterations=args.iterations, warmup=args.warmup) for t in tensors])
    mean, std = float(np.mean(ms)), float(np.mean(ss))
    label = f"torchaudio ({DEVICE.upper()})" if gpu_ok else "torchaudio (CPU fallback)"
    return {"label": label, "device_type": dev, "gpu": gpu_ok,
            "mean_ms": mean, "std_ms": std, "energy_mj": energy_mj(mean, dev)}, None


def run_nnaudio(tensors, args):
    tensors_b = [t.unsqueeze(0) for t in tensors]
    try:
        from nnAudio.Spectrogram import MelSpectrogram as nnMel
    except ImportError:
        return None, "[–] nnAudio not installed  →  pip install nnAudio"
    nn = nnMel(sr=args.sample_rate, n_fft=args.n_fft,
               hop_length=args.hop_length, n_mels=args.n_mels).to(DEVICE)
    nn_cpu = nnMel(sr=args.sample_rate, n_fft=args.n_fft,
                   hop_length=args.hop_length, n_mels=args.n_mels)
    gpu_ok = _gpu_sanity(nn, nn_cpu, tensors_b, DEVICE)
    dev = DEVICE if gpu_ok else "cpu"
    ms, ss = zip(*[measure(nn, t, iterations=args.iterations, warmup=args.warmup) for t in tensors_b])
    mean, std = float(np.mean(ms)), float(np.mean(ss))
    label = f"nnAudio ({DEVICE.upper()})" if gpu_ok else "nnAudio (CPU fallback)"
    return {"label": label, "device_type": dev, "gpu": gpu_ok,
            "mean_ms": mean, "std_ms": std, "energy_mj": energy_mj(mean, dev)}, None


def run_tensorform(tensors, args):
    from tensorform.audio.melt import MelTOperator
    op = MelTOperator(sample_rate=args.sample_rate, n_fft=args.n_fft,
                      hop_length=args.hop_length, n_mels=args.n_mels)
    ms, ss = zip(*[measure(op.accelerate, t, iterations=args.iterations, warmup=args.warmup) for t in tensors])
    mean, std = float(np.mean(ms)), float(np.mean(ss))
    label = f"TensorForm MelT ({DEVICE.upper()})"
    return {"label": label, "device_type": DEVICE, "gpu": True,
            "mean_ms": mean, "std_ms": std, "energy_mj": energy_mj(mean, DEVICE)}, None


# ── Formatting ────────────────────────────────────────────────────────────────

MEDALS = ["1st", "2nd", "3rd", "4th", "5th"]

def format_report(results, env, dataset_stats, args, started_at: str) -> str:
    finished_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    W = 75
    ranked = sorted(results, key=lambda r: r["mean_ms"])
    tf_mean = next(r["mean_ms"] for r in ranked if "TensorForm" in r["label"])

    lines = [
        "",
        "=" * W,
        f" BENCHMARK — tform.audio.melt  (Direct Mel Projection)",
        "=" * W,
        " SYSTEM ENVIRONMENT:",
        f"  ├─ Hostname:                {env.get('hostname', 'N/A')}",
        f"  ├─ Started:                 {started_at}",
        f"  ├─ Finished:                {finished_at}",
        f"  ├─ Timezone:                {env.get('timezone', 'N/A')}",
        f"  ├─ OS:                      {env.get('os', 'N/A')}",
        f"  ├─ CPU:                     {env.get('cpu_model', 'N/A')} ({env.get('cpu_cores', '?')} Cores)",
        f"  ├─ Accelerator:             {env.get('accel_name', 'None')}",
        f"  ├─ TensorForm:              {env.get('tensorform_version', 'N/A')}",
        f"  ├─ Python:                  {env.get('python_version', 'N/A')}",
        f"  ├─ PyTorch:                 {env.get('torch_version', 'N/A')}",
        f"  └─ NumPy:                   {env.get('numpy_version', 'N/A')}",
        "-" * W,
        " DATASET:",
        f"  ├─ Source:                  {dataset_stats.get('source', 'N/A')}",
        f"  ├─ Files:                   {dataset_stats['n_files']}",
        f"  ├─ Duration (s):            {dataset_stats['duration_mean_s']:.2f} ± {dataset_stats['duration_std_s']:.2f}",
        f"  └─ Size (float32):          {dataset_stats['size_mean']} ± {dataset_stats['size_std']}",
        "-" * W,
        " RUN CONFIGURATION:",
        f"  ├─ n_fft:                   {args.n_fft}",
        f"  ├─ hop_length:              {args.hop_length}",
        f"  ├─ n_mels:                  {args.n_mels}",
        f"  ├─ iterations:              {args.iterations}",
        f"  └─ warmup:                  {args.warmup}",
        "-" * W,
        " LATENCY RANKING (Mean ± SD, ms):",
        "  Baselines: STFT+Mel  |  TensorForm: Direct Mel Projection (no STFT)",
        "",
    ]

    for i, r in enumerate(ranked):
        medal = MEDALS[i] if i < len(MEDALS) else f"{i+1}th"
        vs = f"  {r['mean_ms']/ranked[0]['mean_ms']:.1f}x slower than 1st" if i > 0 else ""
        lines.append(f"  {medal}  {r['label']:<32}  {r['mean_ms']:>8.4f} ± {r['std_ms']:.4f} ms{vs}")

    lines += [
        "",
        "-" * W,
        f" ENERGY EFFICIENCY (estimated per call):",
        f"  Power model: MPS={POWER_W['mps']:.0f}W  CPU={POWER_W['cpu']:.0f}W  CUDA=NVML-sampled",
        "",
    ]

    for i, r in enumerate(ranked):
        medal = MEDALS[i] if i < len(MEDALS) else f"{i+1}th"
        vs = f"  {r['energy_mj']/ranked[0]['energy_mj']:.1f}x more" if i > 0 else ""
        lines.append(f"  {medal}  {r['label']:<32}  {r['energy_mj']:>8.4f} mJ{vs}")

    best = ranked[0]
    worst = ranked[-1]
    lines += [
        "",
        f"  Total energy gain (1st vs last): {worst['energy_mj']/best['energy_mj']:.1f}x",
        "=" * W, "",
    ]
    return "\n".join(lines)


def save_report(content: str, env: dict) -> str:
    hostname = env.get("hostname", socket.gethostname().split(".")[0])
    accel = env.get("accel_name", DEVICE)
    accel_slug = re.sub(r"[^A-Za-z0-9_\-]", "_", accel).strip("_")
    filename = f"bench_mel_{hostname}_{accel_slug}.txt"

    out = Path(__file__).resolve().parents[2] / "benchmarks" / "results"
    out.mkdir(parents=True, exist_ok=True)
    filepath = out / filename
    with filepath.open("w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"  → saved to {filepath}")
    return str(filepath)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mel Spectrogram unified benchmark")
    parser.add_argument("--dataset",     default=DATASET_ROOT)
    parser.add_argument("--max-files",   type=int,   default=50)
    parser.add_argument("--iterations",  type=int,   default=20)
    parser.add_argument("--warmup",      type=int,   default=5)
    parser.add_argument("--sample-rate", type=int,   default=16000)
    parser.add_argument("--n-fft",       type=int,   default=512)
    parser.add_argument("--hop-length",  type=int,   default=160)
    parser.add_argument("--n-mels",      type=int,   default=26)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    env = BenchmarkSuite.collect_env_metadata(DEVICE)

    print(f"\nLoading up to {args.max_files} files from: {args.dataset}")
    trials = collect_wav_trials(args.dataset, max_files=args.max_files, target_sr=args.sample_rate)
    if not trials:
        print("No files found."); return

    dataset_stats = compute_trial_stats(trials, sample_rate=args.sample_rate)
    dataset_stats["source"] = args.dataset
    print(f"Loaded {dataset_stats['n_files']} trials  |  "
          f"duration: {dataset_stats['duration_mean_s']:.2f} ± {dataset_stats['duration_std_s']:.2f} s  |  "
          f"size: {dataset_stats['size_mean']} ± {dataset_stats['size_std']}")
    print("Profiling implementations...\n")

    tensors = [torch.from_numpy(s).to(DEVICE) for s in trials]
    results = []

    for runner, label in [
        (lambda: run_librosa(trials, args),       "librosa"),
        (lambda: run_torchaudio(tensors, args),    "torchaudio"),
        (lambda: run_nnaudio(tensors, args),       "nnAudio"),
        (lambda: run_tensorform(tensors, args),    "TensorForm MelT"),
    ]:
        r, err = runner()
        if err:
            print(f"  {err}")
        else:
            print(f"  [✓] {r['label']}")
            results.append(r)

    report = format_report(results, env, dataset_stats, args, started_at)
    print(report)
    save_report(report, env)


if __name__ == "__main__":
    main()
