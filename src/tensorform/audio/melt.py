from __future__ import annotations

import numpy as np
import torch
from typing import Optional

from tensorform._device import detect_device


class MelTOperator:
    """
    Hardware-Aligned Computational (HAC) of Direct Mel Projection (MelT).

    Eliminates the conventional STFT+Mel pipeline by projecting windowed
    time-domain frames directly onto Mel-spaced NDFT frequency coordinates
    through two precomputed basis matrices W^(r) and W^(i) (Eq. 8).
    The full projection reduces to two dense GEMM operations (Eq. 9),
    yielding a single-stage GEMM-native audio frontend.

    Reference
    ---------
    Camargo & Finger, "MelT: GEMM-Native NDFT for Efficient Single-Stage
    Audio Frontends on Modern Accelerators," arXiv 2026.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 80.0,
        f_max: float = 7600.0,
        eps: float = 1e-6,
        device: Optional[str] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.eps = eps

        if device is None:
            self.device = torch.device(detect_device())
        else:
            self.device = torch.device(device)

        Wr, Wi = self._generate_basis()
        self._Wr_np = Wr                                      # (n_mels, n_fft) float32
        self._Wi_np = Wi
        self.Wr = torch.from_numpy(Wr).to(self.device)       # (n_mels, n_fft)
        self.Wi = torch.from_numpy(Wi).to(self.device)

    def _mel_frequencies(self) -> np.ndarray:
        """
        Mel-spaced analysis frequencies f_m in Hz, Eq. 2–4.

        mu_m = mu_min + (m+1)/(M+1) * (mu_max - mu_min),  m = 0..M-1
        f_m  = 700 * (10^(mu_m / 2595) - 1)
        """
        mu_min = 2595.0 * np.log10(1.0 + self.f_min / 700.0)
        mu_max = 2595.0 * np.log10(1.0 + self.f_max / 700.0)
        m = np.arange(self.n_mels, dtype=np.float64)
        mu_m = mu_min + (m + 1.0) / (self.n_mels + 1.0) * (mu_max - mu_min)
        return 700.0 * (10.0 ** (mu_m / 2595.0) - 1.0)

    def _generate_basis(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Precomputed projection basis matrices W^(r) and W^(i), Eq. 8.

        W^(r)_{m,n} = w[n] * cos(2π f_m n / f_s)
        W^(i)_{m,n} = w[n] * sin(2π f_m n / f_s)

        Window w[n] is absorbed into the basis so no separate windowing step
        is required during inference. Shape: (n_mels, n_fft).
        """
        freqs = self._mel_frequencies()                              # (M,)
        n = np.arange(self.n_fft, dtype=np.float64)                 # (N,)
        window = np.hanning(self.n_fft).astype(np.float64)          # (N,)

        # phase[m, n] = 2π f_m n / f_s,  shape (M, N)
        phase = (
            2.0 * np.pi
            * freqs[:, np.newaxis]
            * n[np.newaxis, :]
            / self.sample_rate
        )

        Wr = (window * np.cos(phase)).astype(np.float32)            # (M, N)
        Wi = (window * np.sin(phase)).astype(np.float32)            # (M, N)
        return Wr, Wi

    def legacy_reference(self, signal: np.ndarray) -> np.ndarray:
        """
        Baseline CPU Direct Mel Projection: frame extraction + numpy matmul.

        Implements Eq. 9–11 on CPU. Frame extraction is sequential; matrix
        projections use numpy BLAS.
        """
        signal = signal.astype(np.float32)
        T = 1 + (len(signal) - self.n_fft) // self.hop_length

        frames = np.stack([
            signal[t * self.hop_length : t * self.hop_length + self.n_fft]
            for t in range(T)
        ])                                           # (T, N)

        R = frames @ self._Wr_np.T                  # (T, M)   Eq. 9
        I = frames @ self._Wi_np.T                  # (T, M)
        S = R ** 2 + I ** 2                         # (T, M)   Eq. 10
        return np.log(S + self.eps).astype(np.float32)          # Eq. 11

    def accelerate(self, signal_tensor: torch.Tensor) -> torch.Tensor:
        """
        Executes GEMM-native Direct Mel Projection on the target accelerator.

        Uses torch.unfold for zero-copy frame extraction followed by two
        batched matmuls against the precomputed basis, Eq. 9–11.
        """
        if signal_tensor.dim() > 1:
            signal_tensor = signal_tensor.squeeze(0)

        frames = signal_tensor.unfold(0, self.n_fft, self.hop_length)  # (T, N)

        R = frames @ self.Wr.T       # (T, M)   Eq. 9
        I = frames @ self.Wi.T       # (T, M)
        S = R ** 2 + I ** 2          # (T, M)   Eq. 10
        return torch.log(S + self.eps)                                  # Eq. 11


def melt(
    signal: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    f_min: float = 80.0,
    f_max: float = 7600.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Exposes the GEMM-native MelT operator to the tform namespace."""
    operator = MelTOperator(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        eps=eps,
        device=str(signal.device),
    )
    return operator.accelerate(signal)
