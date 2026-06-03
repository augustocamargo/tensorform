import os
import numpy as np
import torch
from pathlib import Path
from typing import Callable, Optional, List, Union

from tensorform.audio.mel import MelSpectrogramOperator
from tensorform.bench.core import BenchmarkSuite, BenchmarkReport


def _load_wav(path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Loads a WAV file and returns a mono float32 array, resampling if needed.

    Parameters
    ----------
    path : str
        Absolute or relative path to a ``.wav`` file.
    target_sr : int
        Target sample rate in Hz. Resampling is applied when the file sample
        rate differs from this value.

    Returns
    -------
    np.ndarray
        Mono float32 waveform normalized to ``[-1, 1]``.
    """
    try:
        from scipy.io import wavfile
        from scipy.signal import resample_poly
        from math import gcd
    except ImportError as exc:
        raise ImportError(
            "scipy is required for WAV loading: pip install scipy"
        ) from exc

    sr, data = wavfile.read(path)

    # Convert to float32 in [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    # Downmix to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        g = gcd(target_sr, sr)
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)

    return data


def collect_wav_trials(
    root_dir: str,
    max_files: Optional[int] = None,
    target_sr: int = 16000,
) -> List[np.ndarray]:
    """
    Recursively collects WAV files from a directory tree and loads them.

    Parameters
    ----------
    root_dir : str
        Root directory to search (e.g. ``celeb_vox/wav``).
    max_files : int, optional
        Maximum number of files to load. Loads all when ``None``.
    target_sr : int
        Target sample rate passed to the WAV loader.

    Returns
    -------
    List[np.ndarray]
        Loaded float32 waveform arrays, ready for benchmarking.
    """
    paths = sorted(Path(root_dir).rglob("*.wav"))
    if max_files is not None:
        paths = paths[:max_files]

    trials = []
    for p in paths:
        try:
            trials.append(_load_wav(str(p), target_sr=target_sr))
        except Exception as exc:
            print(f"[warn] skipping {p}: {exc}")

    return trials


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
    if hasattr(torch, "mps") and torch.mps.is_available():
        default_device = "mps"
    elif torch.cuda.is_available():
        default_device = "cuda"
    else:
        default_device = "cpu"

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

    return BenchmarkReport(legacy_trials_results, accel_trials_results)
