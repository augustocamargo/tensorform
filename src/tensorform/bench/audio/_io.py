"""
Shared audio I/O utilities for the bench harness.
"""

from __future__ import annotations

import os
import numpy as np
from pathlib import Path
from typing import List, Optional


def _load_wav(path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Loads a WAV file and returns a mono float32 array, resampling if needed.

    Parameters
    ----------
    path : str
        Absolute or relative path to a ``.wav`` file.
    target_sr : int
        Target sample rate in Hz.

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

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    if data.ndim > 1:
        data = data.mean(axis=1)

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
    # Use os.walk with followlinks=False to avoid entering NFS-mounted
    # or otherwise inaccessible symlinked directories.
    raw_paths: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=False):
        for fname in filenames:
            if fname.lower().endswith(".wav"):
                raw_paths.append(Path(dirpath) / fname)
    paths = sorted(raw_paths)
    if max_files is not None:
        paths = paths[:max_files]

    trials: List[np.ndarray] = []
    for p in paths:
        try:
            trials.append(_load_wav(str(p), target_sr=target_sr))
        except Exception as exc:
            print(f"[warn] skipping {p}: {exc}")

    return trials


def _fmt_bytes(b: float) -> str:
    """Format a byte count to a human-readable string."""
    if b < 1024:
        return f"{b:.0f} B"
    if b < 1024 ** 2:
        return f"{b / 1024:.1f} KB"
    return f"{b / 1024 ** 2:.2f} MB"


def compute_trial_stats(trials: List[np.ndarray], sample_rate: int = 16000) -> dict:
    """
    Computes descriptive statistics over a list of loaded waveform trials.

    Parameters
    ----------
    trials : List[np.ndarray]
        Float32 waveform arrays as returned by ``collect_wav_trials``.
    sample_rate : int
        Sample rate used to convert sample counts to seconds.

    Returns
    -------
    dict
        Keys: ``n_files``, ``duration_mean_s``, ``duration_std_s``,
        ``size_mean``, ``size_std`` (human-readable strings).
    """
    durations = np.array([len(s) / sample_rate for s in trials])
    sizes = np.array([len(s) * 4.0 for s in trials])  # float32 = 4 bytes each

    return {
        "n_files": len(trials),
        "duration_mean_s": float(np.mean(durations)),
        "duration_std_s": float(np.std(durations)),
        "size_mean": _fmt_bytes(float(np.mean(sizes))),
        "size_std": _fmt_bytes(float(np.std(sizes))),
    }
