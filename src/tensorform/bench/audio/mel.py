import numpy as np
import torch
from datetime import datetime, timezone
from typing import Callable, Optional, List, Union

from tensorform.audio.mel import MelSpectrogramOperator
from tensorform.bench.audio._io import _load_wav
from tensorform.bench.core import BenchmarkSuite, BenchmarkReport
from tensorform._device import detect_device


def melspectrogram(
    inputs: List[Union[str, np.ndarray]],
    legacy_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    accelerate_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    iterations: int = 50,
    warmup: int = 10,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 26,
    target_sr: int = 16000,
) -> BenchmarkReport:
    """
    Orchestrates execution performance profiling loops over a list of trials.

    Parameters
    ----------
    inputs : List[Union[str, np.ndarray]]
        Trial inputs. Accepts ``np.ndarray`` waveforms directly or ``str``
        paths to individual ``.wav`` files.
    legacy_fn : Callable, optional
        Sequential CPU reference callable. Defaults to
        ``MelSpectrogramOperator.legacy_reference``.
    accelerate_fn : Callable, optional
        Hardware-aligned accelerated callable. Defaults to
        ``MelSpectrogramOperator.accelerate``.
    iterations : int
        Number of timed iterations per trial.
    warmup : int
        Number of untimed warm-up iterations per trial.
    sample_rate : int
        Audio sample rate passed to the operator (Hz).
    n_fft : int
        FFT window size passed to the operator.
    hop_length : int
        STFT hop length in samples passed to the operator.
    n_mels : int
        Number of Mel filter bank channels passed to the operator.
    target_sr : int
        Sample rate used when loading ``.wav`` paths from ``inputs``.
        Should match ``sample_rate`` unless resampling is intentional.

    Returns
    -------
    BenchmarkReport
        Aggregated latency, energy, and fidelity metrics across all trials.
    """
    default_device = detect_device()

    if legacy_fn is None or accelerate_fn is None:
        default_operator = MelSpectrogramOperator(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            device=default_device,
        )
        legacy_fn = legacy_fn or default_operator.legacy_reference
        accelerate_fn = accelerate_fn or default_operator.accelerate

    started_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    legacy_trials_results = []
    accel_trials_results = []

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

        legacy_meta = BenchmarkSuite.profile_execution(
            legacy_fn, signal_np, iterations=iterations, warmup=warmup, device_type="cpu"
        )
        accel_meta = BenchmarkSuite.profile_execution(
            accelerate_fn, signal_tensor, iterations=iterations, warmup=warmup, device_type=default_device
        )

        legacy_trials_results.append(legacy_meta)
        accel_trials_results.append(accel_meta)

    return BenchmarkReport(legacy_trials_results, accel_trials_results, started_at=started_at)
