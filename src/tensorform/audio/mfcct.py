import numpy as np
import torch
from typing import Optional

from tensorform.audio.melt import MelTOperator


class MFCCTOperator:
    """
    Hardware-Aligned Computational (HAC) of MFCCT.

    Cepstral extension of Direct Mel Projection. Applies an orthonormal
    DCT-II matrix D ∈ R^{K×M} to the log-compressed MelT output, Eq. 12:

        C^MFCCT = M^MelT D^T = log(S + ε) D^T

    Defaults match the cepstral configuration in Table 3 of the MelT paper:
    N=1200 (75 ms), M=128, K=13, f_min=0 Hz, f_max=8000 Hz.

    Reference
    ---------
    Camargo & Finger, "MelT: GEMM-Native NDFT for Efficient Single-Stage
    Audio Frontends on Modern Accelerators," arXiv 2026.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1200,
        hop_length: int = 160,
        n_mels: int = 128,
        n_mfcct: int = 13,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        eps: float = 1e-6,
        device: Optional[str] = None,
    ) -> None:
        self.n_mfcct = n_mfcct
        self._melt_op = MelTOperator(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            eps=eps,
            device=device,
        )
        self.device = self._melt_op.device

        dct = self._dct_matrix(n_mels, n_mfcct)
        self._dct_np = dct                                    # (n_mels, K) float32
        self.dct_tensor = torch.from_numpy(dct).to(self.device)

    @staticmethod
    def _dct_matrix(n_mels: int, n_mfcct: int) -> np.ndarray:
        """Orthonormal DCT-II basis matrix of shape (n_mels, n_mfcct)."""
        m = np.arange(n_mels)[:, np.newaxis]
        c = np.arange(n_mfcct)[np.newaxis, :]
        D = np.cos(np.pi / n_mels * (m + 0.5) * c).astype(np.float32)
        D[:, 0] *= np.sqrt(1.0 / n_mels)
        D[:, 1:] *= np.sqrt(2.0 / n_mels)
        return D

    def legacy_reference(self, signal: np.ndarray) -> np.ndarray:
        """
        Baseline CPU MFCCT: Direct Mel Projection + log + DCT-II on CPU.
        """
        melt = self._melt_op.legacy_reference(signal)   # (T, M)
        return (melt @ self._dct_np).astype(np.float32) # (T, K)   Eq. 12

    def accelerate(self, signal_tensor: torch.Tensor) -> torch.Tensor:
        """
        Executes GEMM-native MFCCT computation on the target accelerator.
        """
        melt = self._melt_op.accelerate(signal_tensor)          # (T, M)
        return torch.matmul(melt, self.dct_tensor)              # (T, K)   Eq. 12


def mfcct(
    signal: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 1200,
    hop_length: int = 160,
    n_mels: int = 128,
    n_mfcct: int = 13,
    f_min: float = 0.0,
    f_max: float = 8000.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Exposes the GEMM-native MFCCT operator to the tform namespace."""
    operator = MFCCTOperator(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        n_mfcct=n_mfcct,
        f_min=f_min,
        f_max=f_max,
        eps=eps,
        device=str(signal.device),
    )
    return operator.accelerate(signal)
