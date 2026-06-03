import numpy as np
import torch
from datetime import datetime, timezone
from typing import Callable, Optional, List, Union

from tensorform.audio.cqt import CQTOperator
from tensorform.bench.audio._io import _load_wav
from tensorform.bench.core import BenchmarkSuite, BenchmarkReport
from tensorform._device import detect_device


def cqt(
    inputs: List[Union[str, np.ndarray]],
    legacy_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    accelerate_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    iterations: int = 50,
    warmup: int = 10,
    sample_rate: int = 16000,
    hop_length: int = 512,
    n_bins: int = 72,
    bins_per_octave: int = 12,
    f_min: float = 130.8,
    filter_scale: float = 1.0,
    target_sr: int = 16000,
) -> BenchmarkReport:
    """
    Orchestrates CQT execution performance profiling over a list of trials.

    Parameters
    ----------
    inputs : List[Union[str, np.ndarray]]
        Trial inputs as waveform arrays or paths to ``.wav`` files.
    legacy_fn : Callable, optional
        Sequential CPU reference. Defaults to ``CQTOperator.legacy_reference``.
    accelerate_fn : Callable, optional
        HAC accelerated callable. Defaults to ``CQTOperator.accelerate``.
    iterations : int
        Number of timed iterations per trial.
    warmup : int
        Number of untimed warm-up iterations per trial.
    sample_rate : int
        Audio sample rate in Hz.
    hop_length : int
        Analysis hop length in samples.
    n_bins : int
        Total number of frequency bins.
    bins_per_octave : int
        Frequency resolution in bins per octave.
    f_min : float
        Lowest center frequency in Hz.
    filter_scale : float
        Filter bandwidth scaling factor (1.0 = standard Q).
    target_sr : int
        Sample rate used when loading ``.wav`` paths.

    Returns
    -------
    BenchmarkReport
        Aggregated latency, energy, and fidelity metrics.
    """
    default_device = detect_device()

    if legacy_fn is None or accelerate_fn is None:
        op = CQTOperator(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            f_min=f_min,
            filter_scale=filter_scale,
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
