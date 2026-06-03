import numpy as np
import torch
from datetime import datetime, timezone
from typing import Callable, Optional, List, Union

from tensorform.audio.mfcc import MFCCOperator
from tensorform.bench.audio._io import _load_wav
from tensorform.bench.core import BenchmarkSuite, BenchmarkReport
from tensorform._device import detect_device


def mfcc(
    inputs: List[Union[str, np.ndarray]],
    legacy_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    accelerate_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    iterations: int = 50,
    warmup: int = 10,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 26,
    n_mfcc: int = 13,
    target_sr: int = 16000,
) -> BenchmarkReport:
    """
    Orchestrates MFCC execution performance profiling over a list of trials.

    Parameters
    ----------
    inputs : List[Union[str, np.ndarray]]
        Trial inputs as waveform arrays or paths to ``.wav`` files.
    legacy_fn : Callable, optional
        Sequential CPU reference. Defaults to ``MFCCOperator.legacy_reference``.
    accelerate_fn : Callable, optional
        HAC accelerated callable. Defaults to ``MFCCOperator.accelerate``.
    iterations : int
        Number of timed iterations per trial.
    warmup : int
        Number of untimed warm-up iterations per trial.
    sample_rate : int
        Audio sample rate in Hz.
    n_fft : int
        FFT window size.
    hop_length : int
        STFT hop length in samples.
    n_mels : int
        Number of Mel filter bank channels.
    n_mfcc : int
        Number of cepstral coefficients.
    target_sr : int
        Sample rate used when loading ``.wav`` paths.

    Returns
    -------
    BenchmarkReport
        Aggregated latency, energy, and fidelity metrics.
    """
    default_device = detect_device()

    if legacy_fn is None or accelerate_fn is None:
        op = MFCCOperator(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
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
