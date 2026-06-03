"""
Comparative benchmark: Constant-Q Transform
============================================
Measures per-trial latency for librosa, nnAudio, and TensorForm
on the same CelebVox audio files and hardware.

Install optional dependencies before running:
    pip install librosa nnAudio

Usage
-----
    python benchmarks/audio/compare_cqt.py
    python benchmarks/audio/compare_cqt.py --max-files 50 --iterations 20 --warmup 5
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import tensorform as tform
from tensorform.bench.audio._io import collect_wav_trials

DATASET_ROOT = "/Users/augustocamargo/Projects/MelFTF/dataset/celeb_vox/wav"

from tensorform._device import detect_device
DEVICE = detect_device()


def sync_device() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


def measure(fn, *args, iterations: int = 20, warmup: int = 5):
    for _ in range(warmup):
        fn(*args)
    sync_device()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn(*args)
        sync_device()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


def run_comparison(trials, tensors, tensors_batched, args):
    results = {}
    W, N = args.warmup, args.iterations
    sr, hop, n_bins, bpo, f_min = (
        args.sample_rate, args.hop_length, args.n_bins, args.bins_per_octave, args.f_min
    )

    # ── librosa (CPU sequential) ──────────────────────────────────────────────
    try:
        import librosa
        def _librosa(s):
            return np.abs(librosa.cqt(
                s, sr=sr, hop_length=hop, n_bins=n_bins,
                bins_per_octave=bpo, fmin=f_min
            ))
        ms, ss = zip(*[measure(_librosa, s, iterations=N, warmup=W) for s in trials])
        results["librosa CQT       (CPU)"] = (np.mean(ms), np.mean(ss))
        print("  [✓] librosa")
    except ImportError:
        print("  [–] librosa not installed  →  pip install librosa")

    # ── nnAudio CQT ───────────────────────────────────────────────────────────
    try:
        from nnAudio.Spectrogram import CQT
        cqt_nn = CQT(
            sr=sr, hop_length=hop, n_bins=n_bins,
            bins_per_octave=bpo, fmin=f_min
        ).to(DEVICE)
        ms, ss = zip(*[measure(cqt_nn, t, iterations=N, warmup=W) for t in tensors_batched])
        results[f"nnAudio CQT      ({DEVICE.upper()})"] = (np.mean(ms), np.mean(ss))
        print("  [✓] nnAudio")
    except ImportError:
        print("  [–] nnAudio not installed  →  pip install nnAudio")

    # ── TensorForm CQT ────────────────────────────────────────────────────────
    from tensorform.audio.cqt import CQTOperator
    op = CQTOperator(
        sample_rate=sr, hop_length=hop, n_bins=n_bins,
        bins_per_octave=bpo, f_min=f_min
    )
    ms, ss = zip(*[measure(op.accelerate, t, iterations=N, warmup=W) for t in tensors])
    results[f"TensorForm CQT   ({DEVICE.upper()})"] = (np.mean(ms), np.mean(ss))
    print("  [✓] TensorForm")

    return results


def print_table(results, args, dataset_stats):
    W = 65
    tf_key = next(k for k in results if "TensorForm" in k)
    lib_key = next((k for k in results if "librosa" in k), None)

    print(f"\n{'='*W}")
    print(f"  CONSTANT-Q TRANSFORM — {dataset_stats['n_files']} files × {args.iterations} iterations")
    print(f"  Duration: {dataset_stats['duration_mean_s']:.2f} ± {dataset_stats['duration_std_s']:.2f} s  |  "
          f"Size: {dataset_stats['size_mean']} ± {dataset_stats['size_std']}")
    print(f"  n_bins={args.n_bins}  bpo={args.bins_per_octave}  f_min={args.f_min}  hop={args.hop_length}  device={DEVICE.upper()}")
    print(f"{'='*W}")
    print(f"  {'Implementation':<32} {'Mean (ms)':>10}  {'SD':>7}  {'vs TF':>8}")
    print(f"  {'-'*(W-2)}")

    tf_mean = results[tf_key][0]

    for name, (mean, std) in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
        ratio = f"{mean/tf_mean:.1f}x" if name != tf_key else "  —"
        print(f"  {name:<32} {mean:>10.4f}  {std:>7.4f}  {ratio:>8}")

    if lib_key:
        lib_mean = results[lib_key][0]
        print(f"\n  TensorForm speedup vs librosa: {lib_mean/tf_mean:.2f}x")
    print(f"{'='*W}\n")


def main():
    parser = argparse.ArgumentParser(description="CQT implementation comparison")
    parser.add_argument("--dataset", default=DATASET_ROOT)
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-bins", type=int, default=72)
    parser.add_argument("--bins-per-octave", type=int, default=12)
    parser.add_argument("--f-min", type=float, default=130.8)
    args = parser.parse_args()

    print(f"\nLoading up to {args.max_files} files from: {args.dataset}")
    trials = collect_wav_trials(args.dataset, max_files=args.max_files, target_sr=args.sample_rate)
    if not trials:
        print("No files found."); return

    tensors = [torch.from_numpy(s).to(DEVICE) for s in trials]
    tensors_batched = [t.unsqueeze(0) for t in tensors]

    from tensorform.bench.audio._io import compute_trial_stats
    dataset_stats = compute_trial_stats(trials, sample_rate=args.sample_rate)
    dataset_stats["source"] = args.dataset
    print(f"Loaded {dataset_stats['n_files']} trials  |  "
          f"duration: {dataset_stats['duration_mean_s']:.2f} ± {dataset_stats['duration_std_s']:.2f} s  |  "
          f"size: {dataset_stats['size_mean']} ± {dataset_stats['size_std']}")
    print("Profiling implementations...\n")
    results = run_comparison(trials, tensors, tensors_batched, args)
    print_table(results, args, dataset_stats)


if __name__ == "__main__":
    main()
