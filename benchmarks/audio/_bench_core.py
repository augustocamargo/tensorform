"""
Shared utilities for TensorForm benchmark scripts.
"""

from __future__ import annotations

import re
import socket
import time
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tensorform._device import detect_device
from tensorform.bench.core import BenchmarkSuite

DEVICE: str = detect_device()
POWER_W: Dict[str, float] = {"mps": 15.0, "cuda": 45.0, "cpu": 45.0}
MEDALS: List[str] = ["1st", "2nd", "3rd", "4th", "5th"]

# Reproducibility: disable cuDNN auto-tuner so kernel selection is fixed
# across runs (prevents timing variance from algorithm switching on CUDA).
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ── Timing ────────────────────────────────────────────────────────────────────

def sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


def measure(fn, *args, iterations: int = 20, warmup: int = 5):
    """Time fn(*args). Returns (mean_ms, std_ms)."""
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
    """E (mJ) = P (W) × t (s) × 1e3."""
    return POWER_W.get(device_type, 45.0) * mean_ms * 1e-3 * 1e3


def gpu_sanity(fn_gpu, fn_cpu, tensors, W: int = 2, N: int = 5) -> bool:
    """True if GPU run is >15% faster than CPU — confirms GPU is actually used."""
    ms_cpu, _ = zip(*[measure(fn_cpu, t.cpu(), iterations=N, warmup=W) for t in tensors[:5]])
    ms_gpu, _ = zip(*[measure(fn_gpu, t,       iterations=N, warmup=W) for t in tensors[:5]])
    return float(np.mean(ms_gpu)) < float(np.mean(ms_cpu)) * 0.85


# ── Report formatting ─────────────────────────────────────────────────────────

def format_report(
    results: List[Dict[str, Any]],
    env: Dict[str, Any],
    dataset_stats: Dict[str, Any],
    config_lines: List[str],
    benchmark_title: str,
    baseline_desc: str,
    started_at: str,
    W: int = 75,
) -> str:
    """
    Renders a full benchmark report.

    Parameters
    ----------
    results       : list of result dicts with keys label, device_type, mean_ms, std_ms, energy_mj
    env           : from BenchmarkSuite.collect_env_metadata
    dataset_stats : from compute_trial_stats (with 'source' added)
    config_lines  : operator-specific RUN CONFIGURATION lines (e.g. ["n_fft:   512", ...])
    benchmark_title : header title string
    baseline_desc : description of baselines vs TensorForm approach
    started_at    : ISO timestamp string
    """
    finished_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    ranked = sorted(results, key=lambda r: r["mean_ms"])

    lines = [
        "", "=" * W,
        f" BENCHMARK — {benchmark_title}",
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
    ]
    for i, line in enumerate(config_lines):
        prefix = "  └─" if i == len(config_lines) - 1 else "  ├─"
        lines.append(f"{prefix} {line}")

    lines += [
        "-" * W,
        " LATENCY RANKING (Mean ± SD, ms):",
        f"  {baseline_desc}",
        "",
    ]
    for i, r in enumerate(ranked):
        medal = MEDALS[i] if i < len(MEDALS) else f"{i+1}th"
        vs = f"  {r['mean_ms']/ranked[0]['mean_ms']:.1f}x slower than 1st" if i > 0 else ""
        lines.append(f"  {medal}  {r['label']:<36}  {r['mean_ms']:>8.4f} ± {r['std_ms']:.4f} ms{vs}")

    lines += [
        "", "-" * W,
        " ENERGY EFFICIENCY (estimated per call):",
        f"  Power model: MPS={POWER_W['mps']:.0f}W  CPU={POWER_W['cpu']:.0f}W  CUDA=NVML-sampled",
        "",
    ]
    for i, r in enumerate(ranked):
        medal = MEDALS[i] if i < len(MEDALS) else f"{i+1}th"
        vs = f"  {r['energy_mj']/ranked[0]['energy_mj']:.1f}x more" if i > 0 else ""
        lines.append(f"  {medal}  {r['label']:<36}  {r['energy_mj']:>8.4f} mJ{vs}")

    lines += [
        "",
        f"  Total energy gain (1st vs last): {ranked[-1]['energy_mj']/ranked[0]['energy_mj']:.1f}x",
        "=" * W, "",
    ]
    return "\n".join(lines)


def save_report(content: str, env: Dict[str, Any], operator_slug: str) -> str:
    hostname = env.get("hostname", socket.gethostname().split(".")[0])
    accel_slug = re.sub(r"[^A-Za-z0-9_\-]", "_", env.get("accel_name", DEVICE)).strip("_")
    filepath = (
        Path(__file__).resolve().parents[2]
        / "benchmarks" / "results"
        / f"bench_{operator_slug}_{hostname}_{accel_slug}.txt"
    )
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"  → saved to {filepath}")
    return str(filepath)
