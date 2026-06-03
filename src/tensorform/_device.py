"""
Hardware accelerator detection utility.
"""

from __future__ import annotations

import torch


def detect_device() -> str:
    """
    Returns the best available accelerator as a string: ``"mps"``, ``"cuda"``, or ``"cpu"``.

    Priority order: Apple Silicon MPS > NVIDIA CUDA > CPU.
    """
    if (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "is_available")
        and torch.mps.is_available()
    ):
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
