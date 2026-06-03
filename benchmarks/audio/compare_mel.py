"""
Comparative benchmark: Mel Spectrogram
=======================================
Measures per-trial latency for librosa, torchaudio, nnAudio, and TensorForm
on the same CelebVox audio files and hardware.

Install optional dependencies before running:
    pip install librosa nnAudio torchaudio

Usage
-----
    python benchmarks/audio/compare_mel.py
    python benchmarks/audio/compare_mel.py --max-files 100 --iterations 20 --warmup 5
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
    """Time fn(*args), return (mean_ms, std_ms) over iterations."""
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
    sr, n_fft, hop, n_mels = args.sample_rate, args.n_fft, args.hop_length, args.n_mels

    # ── librosa (CPU sequential) ──────────────────────────────────────────────
    try:
        import librosa
        def _librosa(s):
            return librosa.feature.melspectrogram(
                y=s, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
            )
        ms, ss = zip(*[measure(_librosa, s, iterations=N, warmup=W) for s in trials])
        results["librosa          (CPU)"] = (np.mean(ms), np.mean(ss))
        print(f"  [✓] librosa")
    except ImportError:
        print("  [–] librosa not installed  →  pip install librosa")

    # ── torchaudio ────────────────────────────────────────────────────────────
    try:
        import torchaudio
        mel_ta = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
        ).to(DEVICE)
        # Verify model and input are on the target device
        param_device = next(mel_ta.parameters()).device if list(mel_ta.parameters()) else "no params"
        input_device = tensors[0].device
        print(f"  [✓] torchaudio  | model={param_device}  input={input_device}")
        # Sanity: run on CPU too to confirm GPU is faster
        mel_ta_cpu = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
        )
        ms_cpu, _ = zip(*[measure(mel_ta_cpu, t.cpu(), iterations=5, warmup=2) for t in tensors[:5]])
        ms, ss = zip(*[measure(mel_ta, t, iterations=N, warmup=W) for t in tensors])
        gpu_mean, cpu_mean = np.mean(ms), np.mean(ms_cpu)
        gpu_faster = gpu_mean < cpu_mean * 0.85
        status = "✓ GPU confirmed" if gpu_faster else "⚠ CPU fallback detected"
        print(f"      CPU sanity: {cpu_mean:.4f}ms  GPU: {gpu_mean:.4f}ms  {status}")
        label = f"torchaudio       ({DEVICE.upper()})" if gpu_faster else f"torchaudio       (CPU fallback)"
        results[label] = (np.mean(ms), np.mean(ss))
    except ImportError:
        print("  [–] torchaudio not installed  →  pip install torchaudio")

    # ── nnAudio ───────────────────────────────────────────────────────────────
    try:
        from nnAudio.Spectrogram import MelSpectrogram as nnMel
        mel_nn = nnMel(sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels).to(DEVICE)
        # Verify all parameters are on target device
        param_devices = set(str(p.device) for p in mel_nn.parameters())
        input_device = tensors_batched[0].device
        print(f"  [✓] nnAudio     | model params={param_devices}  input={input_device}")
        # Sanity: run on CPU too
        mel_nn_cpu = nnMel(sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
        ms_cpu, _ = zip(*[measure(mel_nn_cpu, t.cpu(), iterations=5, warmup=2) for t in tensors_batched[:5]])
        ms, ss = zip(*[measure(mel_nn, t, iterations=N, warmup=W) for t in tensors_batched])
        gpu_mean, cpu_mean = np.mean(ms), np.mean(ms_cpu)
        gpu_faster = gpu_mean < cpu_mean * 0.85   # must be >15% faster to count as GPU
        status = "✓ GPU confirmed" if gpu_faster else "⚠ CPU fallback detected"
        print(f"      CPU sanity: {cpu_mean:.4f}ms  GPU: {gpu_mean:.4f}ms  {status}")
        label = f"nnAudio          ({DEVICE.upper()})" if gpu_faster else f"nnAudio          (CPU fallback)"
        results[label] = (np.mean(ms), np.mean(ss))
    except ImportError:
        print("  [–] nnAudio not installed  →  pip install nnAudio")

    # ── TensorForm MelT (Direct Mel Projection — no STFT) ────────────────────
    from tensorform.audio.melt import MelTOperator
    op = MelTOperator(sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
    ms, ss = zip(*[measure(op.accelerate, t, iterations=N, warmup=W) for t in tensors])
    results[f"TensorForm MelT  ({DEVICE.upper()})"] = (np.mean(ms), np.mean(ss))
    print(f"  [✓] TensorForm MelT  (Direct Mel Projection — no STFT)")

    return results


def _format_table(results, args, dataset_stats) -> str:
    import socket, re
    W = 65
    tf_key = next(k for k in results if "TensorForm" in k)
    lib_key = next((k for k in results if "librosa" in k), None)
    tf_mean = results[tf_key][0]

    lines = [
        "",
        "=" * W,
        f"  MEL SPECTROGRAM — {dataset_stats['n_files']} files × {args.iterations} iterations",
        f"  Source:   {dataset_stats.get('source', 'N/A')}",
        f"  Duration: {dataset_stats['duration_mean_s']:.2f} ± {dataset_stats['duration_std_s']:.2f} s  |  "
        f"Size: {dataset_stats['size_mean']} ± {dataset_stats['size_std']}",
        f"  n_fft={args.n_fft}  hop={args.hop_length}  n_mels={args.n_mels}  device={DEVICE.upper()}",
        f"  Baselines: STFT+Mel  |  TensorForm: Direct Mel Projection (no STFT)",
        "=" * W,
        f"  {'Implementation':<32} {'Mean (ms)':>10}  {'SD':>7}  {'vs TF':>8}",
        f"  {'-'*(W-2)}",
    ]
    for name, (mean, std) in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
        ratio = f"{mean/tf_mean:.1f}x" if name != tf_key else "  —"
        lines.append(f"  {name:<32} {mean:>10.4f}  {std:>7.4f}  {ratio:>8}")
    if lib_key:
        lines.append(f"\n  TensorForm MelT speedup vs librosa (STFT+Mel): {results[lib_key][0]/tf_mean:.2f}x")
    lines += ["=" * W, ""]
    return "\n".join(lines)


def print_table(results, args, dataset_stats):
    print(_format_table(results, args, dataset_stats))


def save_table(results, args, dataset_stats,
               output_dir: str = None) -> str:
    import socket, re
    from pathlib import Path

    hostname = socket.gethostname().split(".")[0]
    accel_slug = re.sub(r"[^A-Za-z0-9_\-]", "_", DEVICE).strip("_")
    filename = f"compare_mel_{hostname}_{accel_slug}_bench.txt"

    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parents[2] / "benchmarks" / "results")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / filename
    with filepath.open("w", encoding="utf-8") as fh:
        fh.write(_format_table(results, args, dataset_stats))
    print(f"  → saved to {filepath}")
    return str(filepath)


def main():
    parser = argparse.ArgumentParser(description="Mel Spectrogram implementation comparison")
    parser.add_argument("--dataset", default=DATASET_ROOT)
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--n-mels", type=int, default=26)
    args = parser.parse_args()

    print(f"\nLoading up to {args.max_files} files from: {args.dataset}")
    trials = collect_wav_trials(args.dataset, max_files=args.max_files, target_sr=args.sample_rate)
    if not trials:
        print("No files found."); return

    tensors = [torch.from_numpy(s).to(DEVICE) for s in trials]
    tensors_batched = [t.unsqueeze(0) for t in tensors]  # (1, N) for nnAudio

    from tensorform.bench.audio._io import compute_trial_stats
    dataset_stats = compute_trial_stats(trials, sample_rate=args.sample_rate)
    dataset_stats["source"] = args.dataset
    print(f"Loaded {dataset_stats['n_files']} trials  |  "
          f"duration: {dataset_stats['duration_mean_s']:.2f} ± {dataset_stats['duration_std_s']:.2f} s  |  "
          f"size: {dataset_stats['size_mean']} ± {dataset_stats['size_std']}")
    print("Profiling implementations...\n")
    results = run_comparison(trials, tensors, tensors_batched, args)
    print_table(results, args, dataset_stats)
    save_table(results, args, dataset_stats)


if __name__ == "__main__":
    main()
