"""
Scaling Study: latency vs signal duration for each TensorForm audio operator.

For each duration in the grid, ALL implementations receive THE SAME truncated
signals in THE SAME order. Files are loaded once and truncated deterministically.

Usage
-----
    python benchmarks/audio/bench_scaling.py --operator mel
    python benchmarks/audio/bench_scaling.py --operator mfcc --dataset /path/to/wav
    python benchmarks/audio/bench_scaling.py --operator cqt  --durations 1 5 10 20 40 160

Operators: mel, mfcc, cqt, gammatone
"""

from __future__ import annotations

import sys
import re
import socket
import math
import argparse
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from _bench_core import DEVICE, POWER_W, measure, energy_mj, gpu_sanity, sync
from tensorform.bench.core import BenchmarkSuite

# Auto-detect dataset path based on hostname / filesystem
_DATASET_NVIDIA = "/work/ms/datasets/Libri6000_100_160"
_DATASET_MAC    = "/Users/augustocamargo/Projects/MelFTF/dataset/Libri6000_100_160"
DATASET_ROOT    = _DATASET_NVIDIA if Path(_DATASET_NVIDIA).exists() else _DATASET_MAC
DEFAULT_DURATIONS = [1, 3, 5, 10, 20, 40, 60, 80, 160]


# ── Signal preparation ────────────────────────────────────────────────────────

def truncate(trials: List[np.ndarray], duration_s: float, sr: int) -> List[np.ndarray]:
    """Truncate all signals to exactly duration_s seconds. Same array for all impls."""
    n = int(duration_s * sr)
    return [s[:n].copy() for s in trials if len(s) >= n]


# ── Operator-specific runner factories ────────────────────────────────────────

def make_runners_mel(args):
    runners = {}

    try:
        import librosa
        def _lib(s): return librosa.feature.melspectrogram(
            y=s, sr=args.sample_rate, n_fft=args.n_fft,
            hop_length=args.hop_length, n_mels=args.n_mels)
        runners["librosa (CPU)"] = ("cpu", _lib, False)
    except ImportError:
        print("  [–] librosa not installed")

    try:
        import torchaudio
        ta = torchaudio.transforms.MelSpectrogram(
            sample_rate=args.sample_rate, n_fft=args.n_fft,
            hop_length=args.hop_length, n_mels=args.n_mels).to(DEVICE)
        runners[f"torchaudio ({DEVICE.upper()})"] = (DEVICE, ta, True)
    except ImportError:
        print("  [–] torchaudio not installed")

    try:
        from nnAudio.Spectrogram import MelSpectrogram as nnMel
        nn = nnMel(sr=args.sample_rate, n_fft=args.n_fft,
                   hop_length=args.hop_length, n_mels=args.n_mels).to(DEVICE)
        runners[f"nnAudio ({DEVICE.upper()})"] = (DEVICE, lambda t: nn(t.unsqueeze(0)), True)
    except ImportError:
        print("  [–] nnAudio not installed")

    from tensorform.audio.melt import MelTOperator
    op = MelTOperator(sample_rate=args.sample_rate, n_fft=args.n_fft,
                      hop_length=args.hop_length, n_mels=args.n_mels)
    runners[f"TensorForm MelT ({DEVICE.upper()})"] = (DEVICE, op.accelerate, True)
    return runners


def make_runners_mfcc(args):
    runners = {}

    try:
        import librosa
        def _lib(s): return librosa.feature.mfcc(
            y=s, sr=args.sample_rate, n_mfcc=args.n_mfcc,
            n_fft=args.n_fft, hop_length=args.hop_length, n_mels=args.n_mels)
        runners["librosa (CPU)"] = ("cpu", _lib, False)
    except ImportError:
        print("  [–] librosa not installed")

    try:
        import torchaudio
        kw = {"n_fft": args.n_fft, "hop_length": args.hop_length, "n_mels": args.n_mels}
        ta = torchaudio.transforms.MFCC(
            sample_rate=args.sample_rate, n_mfcc=args.n_mfcc, melkwargs=kw).to(DEVICE)
        runners[f"torchaudio ({DEVICE.upper()})"] = (DEVICE, ta, True)
    except ImportError:
        print("  [–] torchaudio not installed")

    from tensorform.audio.mfcct import MFCCTOperator
    op = MFCCTOperator(sample_rate=args.sample_rate, n_fft=args.n_fft,
                       hop_length=args.hop_length, n_mels=args.n_mels, n_mfcct=args.n_mfcc)
    runners[f"TensorForm MFCCT ({DEVICE.upper()})"] = (DEVICE, op.accelerate, True)
    return runners


def make_runners_cqt(args):
    runners = {}

    try:
        import librosa
        def _lib(s): return np.abs(librosa.cqt(
            s, sr=args.sample_rate, hop_length=args.hop_length,
            n_bins=args.n_bins, bins_per_octave=args.bins_per_octave, fmin=args.f_min))
        runners["librosa CQT (CPU)"] = ("cpu", _lib, False)
    except ImportError:
        print("  [–] librosa not installed")

    try:
        from nnAudio.Spectrogram import CQT
        nn = CQT(sr=args.sample_rate, hop_length=args.hop_length, n_bins=args.n_bins,
                 bins_per_octave=args.bins_per_octave, fmin=args.f_min).to(DEVICE)
        runners[f"nnAudio CQT ({DEVICE.upper()})"] = (DEVICE, lambda t: nn(t.unsqueeze(0)), True)
    except ImportError:
        print("  [–] nnAudio not installed")

    from tensorform.audio.cqt import CQTOperator
    op = CQTOperator(sample_rate=args.sample_rate, hop_length=args.hop_length,
                     n_bins=args.n_bins, bins_per_octave=args.bins_per_octave, f_min=args.f_min)
    runners[f"TensorForm CQT ({DEVICE.upper()})"] = (DEVICE, op.accelerate, True)
    return runners


def make_runners_gammatone(args):
    runners = {}

    from tensorform.audio.gammatone import GammatoneOperator
    op = GammatoneOperator(sample_rate=args.sample_rate, n_fft=args.n_fft_gamma,
                           hop_length=args.hop_length_gamma, n_filters=args.n_filters,
                           f_min=80.0, f_max=args.f_max)
    runners["TF legacy (CPU)"] = ("cpu", op.legacy_reference, False)
    runners[f"TensorForm ({DEVICE.upper()})"] = (DEVICE, op.accelerate, True)
    return runners


RUNNER_FACTORIES = {
    "mel":       make_runners_mel,
    "mfcc":      make_runners_mfcc,
    "cqt":       make_runners_cqt,
    "gammatone": make_runners_gammatone,
}

TF_KEYS = {
    "mel":       "TensorForm MelT",
    "mfcc":      "TensorForm MFCCT",
    "cqt":       "TensorForm CQT",
    "gammatone": "TensorForm",
}


# ── Benchmark one (duration, runner) pair ────────────────────────────────────

def run_one(fn, data_list: list, device_type: str, is_gpu: bool,
            iterations: int, warmup: int) -> float:
    """Returns mean_ms over all trials."""
    all_ms = []
    for d in data_list:
        ms, _ = measure(fn, d, iterations=iterations, warmup=warmup)
        all_ms.append(ms)
    return float(np.mean(all_ms))


# ── Formatting ────────────────────────────────────────────────────────────────

def format_scaling_table(
    operator: str,
    durations: List[float],
    results: Dict[float, Dict[str, float]],   # duration → {name: mean_ms}
    tf_key_prefix: str,
    env: dict,
    args,
) -> str:
    finished_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    # Find all implementation names (stable order)
    all_names = []
    for dur_res in results.values():
        for n in dur_res:
            if n not in all_names:
                all_names.append(n)

    tf_name = next((n for n in all_names if tf_key_prefix in n), all_names[-1])
    others = [n for n in all_names if n != tf_name]

    col_w = 10
    header_parts = f"  {'Duration':>10}" + "".join(f"{n[:col_w]:>{col_w+2}}" for n in all_names)
    speedup_headers = "".join(f"{'vs '+o.split('(')[0].strip()[:8]:>12}" for o in others)

    lines = [
        "",
        "=" * 75,
        f" SCALING STUDY — tform.audio.{operator}",
        "=" * 75,
        " SYSTEM ENVIRONMENT:",
        f"  ├─ Hostname:    {env.get('hostname', 'N/A')}",
        f"  ├─ Finished:   {finished_at}",
        f"  ├─ Accelerator:{env.get('accel_name', 'None')}",
        f"  ├─ TensorForm: {env.get('tensorform_version', 'N/A')}",
        f"  ├─ PyTorch:    {env.get('torch_version', 'N/A')}",
        f"  └─ Dataset:    {args.dataset}",
        "-" * 75,
        f" LATENCY (ms) AND SPEEDUP — {args.iterations} iterations × {args.warmup} warmup",
        "-" * 75,
    ]

    # Header row
    hdr = f"  {'Dur (s)':>8} |"
    for n in all_names:
        short = n.split("(")[0].strip()[:14]
        hdr += f" {short:>14} |"
    for o in others:
        short = ("TF/" + o.split("(")[0].strip())[:10]
        hdr += f" {short:>10} |"
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))

    for dur in durations:
        if dur not in results:
            continue
        row = results[dur]
        tf_ms = row.get(tf_name, float("nan"))
        line = f"  {dur:>8.0f} |"
        for n in all_names:
            ms = row.get(n, float("nan"))
            line += f" {ms:>14.4f} |"
        for o in others:
            o_ms = row.get(o, float("nan"))
            ratio = o_ms / tf_ms if tf_ms > 0 else float("nan")
            line += f" {ratio:>9.2f}x |"
        lines.append(line)

    lines += ["=" * 75, ""]
    return "\n".join(lines)


def _result_dir(env: dict) -> Tuple[Path, str, str]:
    hostname = env.get("hostname", socket.gethostname().split(".")[0])
    accel_slug = re.sub(r"[^A-Za-z0-9_\-]", "_", env.get("accel_name", DEVICE)).strip("_")
    return Path(__file__).resolve().parents[2] / "benchmarks" / "results", hostname, accel_slug


def save_scaling_txt(content: str, operator: str, env: dict) -> str:
    out, hostname, accel_slug = _result_dir(env)
    out.mkdir(parents=True, exist_ok=True)
    filepath = out / f"scaling_{operator}_{hostname}_{accel_slug}.txt"
    with filepath.open("w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"  → txt  {filepath}")
    return str(filepath)


def save_scaling_csv(
    operator: str,
    durations: List[float],
    results: Dict[float, Dict[str, float]],
    tf_prefix: str,
    env: dict,
    args,
) -> str:
    """Saves raw data as CSV for plotting and further analysis."""
    import csv, io

    out, hostname, accel_slug = _result_dir(env)
    out.mkdir(parents=True, exist_ok=True)
    filepath = out / f"scaling_{operator}_{hostname}_{accel_slug}.csv"

    # Collect all implementation names
    all_names: List[str] = []
    for dur_res in results.values():
        for n in dur_res:
            if n not in all_names:
                all_names.append(n)
    tf_name = next((n for n in all_names if tf_prefix in n), None)

    rows = []
    for dur in sorted(durations):
        if dur not in results:
            continue
        dur_res = results[dur]
        tf_ms = dur_res.get(tf_name, float("nan"))
        for impl in all_names:
            ms = dur_res.get(impl, float("nan"))
            speedup_vs_tf = ms / tf_ms if tf_ms and tf_ms > 0 else float("nan")
            is_tf = 1 if (tf_name and tf_prefix in impl) else 0
            rows.append({
                "timestamp":      datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
                "operator":       operator,
                "duration_s":     dur,
                "implementation": impl,
                "device":         DEVICE,
                "accel":          env.get("accel_name", "N/A"),
                "hostname":       env.get("hostname", "N/A"),
                "tf_version":     env.get("tensorform_version", "N/A"),
                "torch_version":  env.get("torch_version", "N/A"),
                "mean_ms":        round(ms, 6),
                "speedup_vs_tf":  round(speedup_vs_tf, 4),
                "is_tf":          is_tf,
                "n_fft":          getattr(args, "n_fft", "N/A"),
                "hop_length":     getattr(args, "hop_length", "N/A"),
                "iterations":     args.iterations,
                "warmup":         args.warmup,
            })

    if not rows:
        print(f"  [warn] no data to write to {filepath}")
        return str(filepath)

    with filepath.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  → csv   {filepath}")
    return str(filepath)


def save_file_log(
    source_paths: List[str],
    operator: str,
    env: dict,
    args,
) -> str:
    """Saves the full paths of all source files used in this run."""
    out, hostname, accel_slug = _result_dir(env)
    out.mkdir(parents=True, exist_ok=True)
    filepath = out / f"scaling_{operator}_{hostname}_{accel_slug}_files.txt"

    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        f"# TensorForm scaling study — source file log",
        f"# Operator:   {operator}",
        f"# Timestamp:  {ts}",
        f"# Hostname:   {env.get('hostname', 'N/A')}",
        f"# Accelerator:{env.get('accel_name', 'N/A')}",
        f"# Dataset:    {args.dataset}",
        f"# max-files:  {args.max_files}",
        f"# n_files:    {len(source_paths)}",
        f"# Durations:  {sorted(args.durations)}",
        "#",
        "# All files were loaded once and truncated to each duration point.",
        "# Same file list and same truncation used for ALL implementations.",
        "#",
    ] + source_paths + [""]

    with filepath.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  → files {filepath}")
    return str(filepath)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scaling study: latency vs signal duration")
    p.add_argument("--operator",      choices=list(RUNNER_FACTORIES) + ["all"], default="mel")
    p.add_argument("--dataset",       default=DATASET_ROOT)
    p.add_argument("--max-files",     type=int,   default=100,
                   help="Number of files to use per duration point")
    p.add_argument("--durations",     type=float, nargs="+", default=DEFAULT_DURATIONS)
    p.add_argument("--iterations",    type=int,   default=10)
    p.add_argument("--warmup",        type=int,   default=3)
    p.add_argument("--sample-rate",   type=int,   default=16000)
    # mel / mfcc
    p.add_argument("--n-fft",         type=int,   default=512)
    p.add_argument("--hop-length",    type=int,   default=160)
    p.add_argument("--n-mels",        type=int,   default=26)
    p.add_argument("--n-mfcc",        type=int,   default=13)
    # cqt
    p.add_argument("--n-bins",        type=int,   default=60)
    p.add_argument("--bins-per-octave", type=int, default=12)
    p.add_argument("--f-min",         type=float, default=130.8)
    # gammatone (separate n-fft since default differs from mel/mfcc)
    p.add_argument("--n-filters",     type=int,   default=32)
    p.add_argument("--n-fft-gamma",   type=int,   default=1024,
                   help="n_fft for gammatone (default 1024, not shared with mel/mfcc/cqt)")
    p.add_argument("--hop-length-gamma", type=int, default=256)
    p.add_argument("--f-max",         type=float, default=8000.0)
    return p.parse_args()


def run_operator(operator: str, args, env, all_trials: list, source_paths: List[str]) -> None:
    """Run scaling study for a single operator."""
    sr = args.sample_rate
    tf_prefix = TF_KEYS[operator]

    print(f"\n{'='*60}")
    print(f"  OPERATOR: {operator.upper()}")
    print(f"{'='*60}")
    print(f"Building runners...")
    runners = RUNNER_FACTORIES[operator](args)
    print()

    results: Dict[float, Dict[str, float]] = {}

    for dur in sorted(args.durations):
        n_samples = int(dur * sr)
        signals = [s[:n_samples].copy() for s in all_trials]
        tensors = [torch.from_numpy(s).to(DEVICE) for s in signals]

        print(f"  Duration: {dur:>5.0f}s  ({n_samples:,} samples × {len(signals)} files)")
        dur_results: Dict[str, float] = {}

        tf_runner_name = next((n for n in runners if tf_prefix in n), None)
        for name, (device_type, fn, is_gpu) in runners.items():
            data = tensors if is_gpu else signals
            mean_ms = run_one(fn, data, device_type, is_gpu, args.iterations, args.warmup)
            dur_results[name] = mean_ms
            tf_ms = dur_results.get(tf_runner_name)
            speedup = f"  ({mean_ms / tf_ms:.1f}x vs TF)" if tf_ms and name != tf_runner_name else ""
            print(f"    {name:<38} {mean_ms:>8.4f} ms{speedup}")

        results[dur] = dur_results
        print()

    report = format_scaling_table(operator, sorted(args.durations), results, tf_prefix, env, args)
    print(report)
    save_scaling_txt(report, operator, env)
    save_scaling_csv(operator, sorted(args.durations), results, tf_prefix, env, args)
    save_file_log(source_paths, operator, env, args)


def main() -> None:
    args = parse_args()
    env = BenchmarkSuite.collect_env_metadata(DEVICE)
    sr = args.sample_rate

    # Load source files ONCE — all durations share the same source
    max_dur = max(args.durations)
    required_samples = int(max_dur * sr)
    print(f"\nLoading up to {args.max_files} files (need ≥ {max_dur:.0f}s each) from: {args.dataset}")

    # Collect paths and data together so we can log which files were used
    import os as _os
    raw_paths: List[str] = []
    for dirpath, _, filenames in _os.walk(args.dataset, followlinks=False):
        for fname in sorted(filenames):
            if fname.lower().endswith(".wav"):
                raw_paths.append(_os.path.join(dirpath, fname))
    raw_paths = sorted(raw_paths)

    from tensorform.bench.audio._io import _load_wav
    all_trials: List[np.ndarray] = []
    source_paths: List[str] = []
    for p in raw_paths:
        if len(all_trials) >= args.max_files * 3:
            break
        try:
            s = _load_wav(p, target_sr=sr)
            if len(s) >= required_samples:
                all_trials.append(s)
                source_paths.append(p)
        except Exception as exc:
            print(f"  [warn] skipping {p}: {exc}")

    all_trials = all_trials[:args.max_files]
    source_paths = source_paths[:args.max_files]

    if not all_trials:
        print(f"No files long enough for {max_dur}s. Check --dataset or reduce --durations.")
        return
    print(f"Using {len(all_trials)} source files.")

    operators = list(RUNNER_FACTORIES) if args.operator == "all" else [args.operator]
    for op in operators:
        run_operator(op, args, env, all_trials, source_paths)


if __name__ == "__main__":
    main()
