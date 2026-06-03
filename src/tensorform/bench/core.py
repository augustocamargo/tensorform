import os
import time
import platform
from datetime import datetime
import torch
import numpy as np
from typing import Callable, Any, Dict, List

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
        elif device_type == "mps" and hasattr(torch, "mps") and torch.mps.is_available():
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
        """Captures hardware and platform runtime deployment environments."""
        metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "os": f"{platform.system()} {platform.release()}",
            "cpu_model": platform.processor() or "Unknown CPU",
            "cpu_cores": os.cpu_count() or 0,
            "cpu_load_pct": psutil.cpu_percent(interval=None) if psutil else "N/A",
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2) if psutil else "N/A",
            "ram_load_pct": psutil.virtual_memory().percent if psutil else "N/A",
            "accel_name": "None",
            "accel_type": device_type,
        }

        if device_type == "cuda" and torch.cuda.is_available():
            metadata["accel_name"] = torch.cuda.get_device_name(0)
        elif device_type == "mps" and hasattr(torch, "mps") and torch.mps.is_available():
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

    def __init__(self, legacy_trials: List[Dict[str, Any]], accel_trials: List[Dict[str, Any]]):
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
        norm_a = np.linalg.norm(flat_a)
        norm_b = np.linalg.norm(flat_b)
        cosine = float(np.dot(flat_a, flat_b) / (norm_a * norm_b)) if norm_a and norm_b else 1.0
        return cosine, float(np.max(np.abs(flat_a - flat_b)))

    def print_summary(self, operator_name: str) -> None:
        speedup = self.legacy_mean_ms / self.accel_mean_ms
        energy_gain = self.legacy_energy_mj / self.accel_energy_mj

        print("\n" + "=" * 75)
        print(f" AGGREGATED BENCHMARK REPORT ({self.num_trials} Evaluation Trials) ")
        print(f" Operator: {operator_name} ")
        print("=" * 75)
        print(" SYSTEM ENVIRONMENT METADATA:")
        print(f"  ├─ Timestamp:               {self.env['timestamp']}")
        print(f"  ├─ Host CPU Model:          {self.env['cpu_model']} ({self.env['cpu_cores']} Cores)")
        print(f"  └─ Target Accelerator:      {self.env['accel_name']}")
        print("-" * 75)
        print(" EXECUTION LATENCY (Cross-Trial Mean ± SD):")
        print(f"  ├─ Legacy CPU Reference:    {self.legacy_mean_ms:.4f} ± {self.legacy_std_ms:.2f} ms")
        print(f"  ├─ Accelerator-Native:      {self.accel_mean_ms:.4f} ± {self.accel_std_ms:.2f} ms")
        print(f"  └─ Sustained Speedup:       {speedup:.2f}x")
        print("-" * 75)
        print(" ENERGY EFFICIENCY (Mean per Trial):")
        print(f"  ├─ Legacy CPU Reference:    {self.legacy_energy_mj:.4f} mJ")
        print(f"  ├─ Accelerator-Native:      {self.accel_energy_mj:.4f} mJ")
        print(f"  └─ Energy Efficiency Gain:  {energy_gain:.2f}x less energy consumed")
        print("-" * 75)
        print(" MATHEMATICAL FIDELITY ACCUMULATED:")
        print(f"  ├─ Mean Cosine Similarity:  {self.mean_cosine:.8f}")
        print(f"  └─ Worst-Case Abs Error:    {self.worst_max_error:.8e}")
        print("=" * 75 + "\n")
