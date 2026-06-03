import numpy as np
import torch
from datetime import datetime, timezone
from typing import Callable, Optional, List, Union

from tensorform.audio.melt import MelTOperator
from tensorform.bench.audio._io import _load_wav
from tensorform.bench.core import BenchmarkSuite, BenchmarkReport
from tensorform._device import detect_device


def melt(
    inputs: List[Union[str, np.ndarray]],
    legacy_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    accelerate_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    iterations: int = 50,
    warmup: int = 10,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    f_min: float = 80.0,
    f_max: float = 7600.0,
    eps: float = 1e-6,
    target_sr: int = 16000,
) -> BenchmarkReport:
    """
    Orchestrates MelT (Direct Mel Projection) performance profiling.

    Parameters
    ----------
    inputs : List[Union[str, np.ndarray]]
        Trial inputs as waveform arrays or paths to ``.wav`` files.
    legacy_fn : Callable, optional
        CPU reference. Defaults to ``MelTOperator.legacy_reference``.
    accelerate_fn : Callable, optional
        HAC accelerated callable. Defaults to ``MelTOperator.accelerate``.
    iterations : int
        Number of timed iterations per trial.
    warmup : int
        Number of untimed warm-up iterations per trial.
    sample_rate : int
        Audio sample rate in Hz.
    n_fft : int
        Frame length N (paper spectral default: 400 = 25 ms at 16 kHz).
    hop_length : int
        Hop size H in samples (paper default: 160 = 10 ms).
    n_mels : int
        Number of Mel projection bins M (paper spectral default: 80).
    f_min : float
        Lowest analysis frequency in Hz (paper spectral default: 80).
    f_max : float
        Highest analysis frequency in Hz (paper spectral default: 7600).
    eps : float
        Log-compression stability constant ε.
    target_sr : int
        Sample rate used when loading ``.wav`` paths.

    Returns
    -------
    BenchmarkReport
        Aggregated latency, energy, and fidelity metrics.
    """
    default_device = detect_device()

    if legacy_fn is None or accelerate_fn is None:
        op = MelTOperator(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            eps=eps,
            device=default_device,
        )
        legacy_fn = legacy_fn or op.legacy_reference
        accelerate_fn = accelerate_fn or op.accelerate

    started_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    legacy_results, accel_results = [], []

    for item in inputs:
        if isinstance(item, str):
            try:
                signal_np = _load_wav(item, target_sr=target_sr)
            except Exception as exc:
                print(f"[warn] skipping {item}: {exc}")
                continue
        elif isinstance(item, np.ndarray):
            signal_np = item.astype(np.float32)
        else:
            continue

        signal_tensor = torch.from_numpy(signal_np).to(default_device)

        legacy_results.append(BenchmarkSuite.profile_execution(
            legacy_fn, signal_np, iterations=iterations, warmup=warmup, device_type="cpu"
        ))
        accel_results.append(BenchmarkSuite.profile_execution(
            accelerate_fn, signal_tensor, iterations=iterations, warmup=warmup, device_type=default_device
        ))

    return BenchmarkReport(legacy_results, accel_results, started_at=started_at)
