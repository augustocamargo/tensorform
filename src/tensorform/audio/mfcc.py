import numpy as np
import torch
from typing import Optional

from tensorform.audio.mel import MelSpectrogramOperator


class MFCCOperator:
    """
    Hardware-Aligned Computational (HAC) of Mel-Frequency Cepstral Coefficients.

    Reformulates the sequential mel → log → DCT pipeline into a static tensor
    computation graph: HAC mel spectrogram, log compression, and orthonormal
    DCT-II matrix projection — all executed as parallelized matmul operations.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 26,
        n_mfcc: int = 13,
        device: Optional[str] = None,
    ) -> None:
        self.n_mfcc = n_mfcc
        self._mel_op = MelSpectrogramOperator(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            device=device,
        )
        self.device = self._mel_op.device
        self.dct_tensor = torch.from_numpy(
            self._dct_matrix(n_mels, n_mfcc)
        ).to(self.device)

    @staticmethod
    def _dct_matrix(n_mels: int, n_mfcc: int) -> np.ndarray:
        """Orthonormal DCT-II basis matrix of shape (n_mels, n_mfcc)."""
        m = np.arange(n_mels)[:, np.newaxis]
        c = np.arange(n_mfcc)[np.newaxis, :]
        D = np.cos(np.pi / n_mels * (m + 0.5) * c).astype(np.float32)
        D[:, 0] *= np.sqrt(1.0 / n_mels)
        D[:, 1:] *= np.sqrt(2.0 / n_mels)
        return D

    def legacy_reference(self, signal: np.ndarray) -> np.ndarray:
        """
        Baseline CPU execution: sequential mel → log → DCT pipeline.
        """
        mel = self._mel_op.legacy_reference(signal)                   # (frames, n_mels) float32
        log_mel = np.log(mel + 1e-6)
        dct = self._dct_matrix(self._mel_op.n_mels, self.n_mfcc)
        return np.dot(log_mel, dct)                                    # (frames, n_mfcc) float32

    def accelerate(self, signal_tensor: torch.Tensor) -> torch.Tensor:
        """
        Executes parallelized hardware-aligned MFCC computation graph.
        """
        mel = self._mel_op.accelerate(signal_tensor)                   # (frames, n_mels)
        log_mel = torch.log(mel + 1e-6)
        return torch.matmul(log_mel, self.dct_tensor)                  # (frames, n_mfcc)


def mfcc(
    signal: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 26,
    n_mfcc: int = 13,
) -> torch.Tensor:
    """Exposes the hardware-aligned MFCC operator to the tform namespace."""
    operator = MFCCOperator(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
        device=str(signal.device),
    )
    return operator.accelerate(signal)
