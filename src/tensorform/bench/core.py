from __future__ import annotations

import os
import socket
import time
import platform
from datetime import datetime, timezone
import torch
import numpy as np
from typing import Callable, Any, Dict, List, Optional

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class BenchmarkSuite:
    """Core telemetry profile suite tracking runtime statistics, estimates, and device metrics."""

    @staticmethod
    def sync_device(device_type: str) -> None:
        """Enforces execution barriers across valid active tensor runtime structures."""
        if device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device_type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
            torch.mps.synchronize()

    @staticmethod
    def _get_gpu_power(device_type: str) -> float:
        """Samples instantaneous power consumption metrics on NVIDIA platforms."""
        if device_type == "cuda" and NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:
                return 0.0
        return 0.0

    @staticmethod
    def collect_env_metadata(device_type: str) -> Dict[str, Any]:
        """Captures hardware, platform, and software stack metadata."""
        import sys as _sys
        from tensorform import __version__ as _tform_version

        now_local = datetime.now(timezone.utc).astimezone()
        metadata = {
            "hostname":         socket.gethostname().split(".")[0],
            "timezone":         now_local.strftime("%Z (UTC%z)"),
            "os":               f"{platform.system()} {platform.release()}",
            "cpu_model":        platform.processor() or "Unknown CPU",
            "cpu_cores":        os.cpu_count() or 0,
            "cpu_load_pct":     psutil.cpu_percent(interval=None) if psutil else "N/A",
            "ram_total_gb":     round(psutil.virtual_memory().total / (1024**3), 2) if psutil else "N/A",
            "ram_load_pct":     psutil.virtual_memory().percent if psutil else "N/A",
            "accel_name":       "None",
            "accel_type":       device_type,
            "python_version":   _sys.version.split()[0],
            "torch_version":    torch.__version__,
            "numpy_version":    np.__version__,
            "tensorform_version": _tform_version,
        }

        if device_type == "cuda" and torch.cuda.is_available():
            metadata["accel_name"] = torch.cuda.get_device_name(0)
        elif device_type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
            metadata["accel_name"] = "Apple Silicon Integrated GPU"

        return metadata

    @staticmethod
    def profile_execution(
        func: Callable[..., Any],
        *args: Any,
        iterations: int = 50,
        warmup: int = 10,
        device_type: str = "cpu",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Profiles a single computing target under controlled sequential iteration constraints."""
        raw_output = func(*args, **kwargs)
        if isinstance(raw_output, torch.Tensor):
            raw_output = raw_output.detach().cpu().numpy()

        for _ in range(warmup):
            func(*args, **kwargs)

        BenchmarkSuite.sync_device(device_type)
        times = []
        power_samples = []

        for _ in range(iterations):
            if device_type == "cuda":
                power_samples.append(BenchmarkSuite._get_gpu_power(device_type))

            start_time = time.perf_counter()
            func(*args, **kwargs)
            BenchmarkSuite.sync_device(device_type)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        mean_seconds = float(np.mean(times))
        env_meta = BenchmarkSuite.collect_env_metadata(device_type)

        avg_power_watts = 15.0 if device_type == "mps" else 45.0
        if device_type == "cuda" and len(power_samples) > 0:
            avg_power_watts = float(np.mean(power_samples))

        return {
            "mean_seconds": mean_seconds,
            "std_seconds": float(np.std(times)),
            "energy_joules": avg_power_watts * mean_seconds,
            "output": raw_output,
            "env": env_meta,
        }


class BenchmarkReport:
    """Aggregates and formats architectural telemetry metrics from multiple testing trials."""

    def __init__(
        self,
        legacy_trials: List[Dict[str, Any]],
        accel_trials: List[Dict[str, Any]],
        started_at: Optional[str] = None,
    ):
        self.finished_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        self.started_at = started_at or self.finished_at

        if not legacy_trials or not accel_trials:
            raise ValueError(
                "BenchmarkReport requires at least one completed trial. "
                "Check that inputs are valid and longer than n_fft."
            )
        self.num_trials = len(legacy_trials)
        self.env = accel_trials[0]["env"]

        legacy_times = [t["mean_seconds"] for t in legacy_trials]
        accel_times = [t["mean_seconds"] for t in accel_trials]

        self.legacy_mean_ms = float(np.mean(legacy_times)) * 1000
        self.legacy_std_ms = float(np.std(legacy_times)) * 1000
        self.accel_mean_ms = float(np.mean(accel_times)) * 1000
        self.accel_std_ms = float(np.std(accel_times)) * 1000

        legacy_energy = [t["energy_joules"] for t in legacy_trials]
        accel_energy = [t["energy_joules"] for t in accel_trials]

        self.legacy_energy_mj = float(np.mean(legacy_energy)) * 1000
        self.accel_energy_mj = float(np.mean(accel_energy)) * 1000

        cosines, max_errors = [], []
        for l, a in zip(legacy_trials, accel_trials):
            c, e = self._compute_fidelity(l["output"], a["output"])
            cosines.append(c)
            max_errors.append(e)

        self.mean_cosine = float(np.mean(cosines))
        self.worst_max_error = float(np.max(max_errors))

    def _compute_fidelity(self, arr_a: np.ndarray, arr_b: np.ndarray) -> tuple[float, float]:
        flat_a = arr_a.flatten().astype(np.float64)
        flat_b = arr_b.flatten().astype(np.float64)
        denom = np.linalg.norm(flat_a) * np.linalg.norm(flat_b)
        cosine = float(np.dot(flat_a, flat_b) / denom) if denom > 0.0 else 0.0
        return cosine, float(np.max(np.abs(flat_a - flat_b)))

    def _format_summary(
        self,
        operator_name: str,
        run_config: Optional[Dict[str, Any]] = None,
        prior_art: Optional[List[str]] = None,
        dataset_stats: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Renders the report as a plain-text string."""
        speedup = self.legacy_mean_ms / self.accel_mean_ms if self.accel_mean_ms > 0.0 else float("inf")
        energy_gain = self.legacy_energy_mj / self.accel_energy_mj if self.accel_energy_mj > 0.0 else float("inf")
        lines = [
            "",
            "=" * 75,
            f" AGGREGATED BENCHMARK REPORT ({self.num_trials} Evaluation Trials) ",
            f" Operator: {operator_name} ",
            "=" * 75,
            " SYSTEM ENVIRONMENT METADATA:",
            f"  ├─ Hostname:                {self.env.get('hostname', 'N/A')}",
            f"  ├─ Started:                 {self.started_at}",
            f"  ├─ Finished:                {self.finished_at}",
            f"  ├─ Timezone:                {self.env.get('timezone', 'N/A')}",
            f"  ├─ OS:                      {self.env.get('os', 'N/A')}",
            f"  ├─ Host CPU Model:          {self.env['cpu_model']} ({self.env['cpu_cores']} Cores)",
            f"  ├─ Target Accelerator:      {self.env['accel_name']}",
            f"  ├─ TensorForm:              {self.env.get('tensorform_version', 'N/A')}",
            f"  ├─ Python:                  {self.env.get('python_version', 'N/A')}",
            f"  ├─ PyTorch:                 {self.env.get('torch_version', 'N/A')}",
            f"  └─ NumPy:                   {self.env.get('numpy_version', 'N/A')}",
            "-" * 75,
            " EXECUTION LATENCY (Cross-Trial Mean ± SD):",
            f"  ├─ Legacy CPU Reference:    {self.legacy_mean_ms:.4f} ± {self.legacy_std_ms:.2f} ms",
            f"  ├─ Accelerator-Native:      {self.accel_mean_ms:.4f} ± {self.accel_std_ms:.2f} ms",
            f"  └─ Sustained Speedup:       {speedup:.2f}x",
            "-" * 75,
            " ENERGY EFFICIENCY (Mean per Trial):",
            f"  ├─ Legacy CPU Reference:    {self.legacy_energy_mj:.4f} mJ",
            f"  ├─ Accelerator-Native:      {self.accel_energy_mj:.4f} mJ",
            f"  └─ Energy Efficiency Gain:  {energy_gain:.2f}x less energy consumed",
            "-" * 75,
            " MATHEMATICAL FIDELITY ACCUMULATED:",
            f"  ├─ Mean Cosine Similarity:  {self.mean_cosine:.8f}",
            f"  └─ Worst-Case Abs Error:    {self.worst_max_error:.8e}",
        ]
        if dataset_stats:
            d = dataset_stats
            src_line = [f"  ├─ Source:              {d['source']}"] if d.get("source") else []
            lines += [
                "-" * 75,
                " DATASET STATISTICS:",
            ] + src_line + [
                f"  ├─ Files:               {d.get('n_files', 'N/A')}",
                f"  ├─ Duration (s):        {d.get('duration_mean_s', 0):.2f} ± {d.get('duration_std_s', 0):.2f}",
                f"  └─ Size (float32):      {d.get('size_mean', 'N/A')} ± {d.get('size_std', 'N/A')}",
            ]
        if prior_art:
            lines += [
                "-" * 75,
                " PRIOR ART CONTEXT (reported speedups — methodology differs):",
                "  ┆  Hardware, batch size, and reference baseline vary across works.",
                "  ┆  TensorForm reports per-trial single-utterance latency on MPS/CUDA.",
            ]
            for i, entry in enumerate(prior_art):
                prefix = "  └─" if i == len(prior_art) - 1 else "  ├─"
                lines.append(f"{prefix} {entry}")
        if run_config:
            lines += [
                "-" * 75,
                " RUN CONFIGURATION (reproduce with these parameters):",
            ]
            items = list(run_config.items())
            for i, (k, v) in enumerate(items):
                prefix = "  └─" if i == len(items) - 1 else "  ├─"
                lines.append(f"{prefix} {k:<26} {v}")
        lines += ["=" * 75, ""]
        return "\n".join(lines)

    def print_summary(
        self,
        operator_name: str,
        run_config: Optional[Dict[str, Any]] = None,
        prior_art: Optional[List[str]] = None,
        dataset_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Prints the aggregated benchmark report to stdout."""
        print(self._format_summary(operator_name, run_config=run_config, prior_art=prior_art, dataset_stats=dataset_stats))

    def save_report(
        self,
        operator_name: str,
        operator_slug: str,
        run_config: Optional[Dict[str, Any]] = None,
        prior_art: Optional[List[str]] = None,
        dataset_stats: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Saves the benchmark report to a file following the naming convention
        ``{operator_slug}_{hostname}_{accelerator}_bench.txt``.

        Parameters
        ----------
        operator_name : str
            Human-readable operator label included in the report body.
        operator_slug : str
            Short prefix for the filename (e.g. ``"mel"``, ``"mfcc"``, ``"cqt"``).
        output_dir : str, optional
            Directory where the file is written. Defaults to
            ``benchmarks/results/`` relative to the repo root (resolved from
            this module's location). Created if it does not exist.

        Returns
        -------
        str
            Absolute path of the written file.
        """
        import re
        import socket
        from pathlib import Path

        hostname = socket.gethostname().split(".")[0]
        accel = self.env.get("accel_name", "cpu")
        accel_slug = re.sub(r"[^A-Za-z0-9_\-]", "_", accel).strip("_")
        filename = f"{operator_slug}_{hostname}_{accel_slug}_bench.txt"

        if output_dir is None:
            # Resolve relative to repo root: src/tensorform/bench/core.py → parents[3]
            output_dir = str(Path(__file__).resolve().parents[3] / "benchmarks" / "results")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        filepath = out_path / filename

        with filepath.open("w", encoding="utf-8") as fh:
            fh.write(self._format_summary(operator_name, run_config=run_config, prior_art=prior_art, dataset_stats=dataset_stats))

        print(f"  → saved to {filepath}")
        return str(filepath)
