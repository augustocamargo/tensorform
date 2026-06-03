import numpy as np
import torch
from typing import Optional

from tensorform._device import detect_device


class CQTOperator:
    """
    Hardware-Aligned Computational (HAC) of the Constant-Q Transform.

    Precomputes geometrically-spaced, Hann-windowed complex sinusoid kernels
    in the frequency domain. Reformulates frame-by-frame filtering into a
    single batched split-complex matmul across all STFT frames.

    Both legacy and HAC paths use identical split-complex float32 arithmetic
    to ensure fidelity comparisons are not biased by dtype differences.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 512,
        n_bins: int = 72,
        bins_per_octave: int = 12,
        f_min: float = 130.8,
        filter_scale: float = 1.0,
        device: Optional[str] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.f_min = f_min
        self.Q = filter_scale / (2.0 ** (1.0 / bins_per_octave) - 1.0)

        # n_fft: smallest power of 2 ≥ longest filter (at f_min)
        self.n_fft = int(2 ** np.ceil(np.log2(self.Q * sample_rate / f_min)))

        if device is None:
            self.device = torch.device(detect_device())
        else:
            self.device = torch.device(device)

        kernel = self._generate_kernel()
        self._kernel_real_np = kernel.real  # (n_bins, n_fft//2+1) float32
        self._kernel_imag_np = kernel.imag
        self.kernel_real = torch.from_numpy(self._kernel_real_np).to(self.device)
        self.kernel_imag = torch.from_numpy(self._kernel_imag_np).to(self.device)

    def _generate_kernel(self) -> np.ndarray:
        """
        CQT filter bank in frequency domain, shape (n_bins, n_fft//2+1) complex64.

        Each row is the DFT of a Hann-windowed complex sinusoid tuned to the
        corresponding geometrically spaced center frequency.
        """
        kernel = np.zeros((self.n_bins, self.n_fft // 2 + 1), dtype=np.complex64)
        for k in range(self.n_bins):
            f_k = self.f_min * 2.0 ** (k / self.bins_per_octave)
            N_k = min(int(round(self.Q * self.sample_rate / f_k)), self.n_fft)
            n = np.arange(N_k, dtype=np.float32)
            window = np.hanning(N_k).astype(np.float32)
            basis = (window * np.exp(2j * np.pi * self.Q * n / N_k) / N_k).astype(np.complex64)
            # basis is complex — full FFT, take positive-frequency half, normalize by n_fft
            kernel[k] = (
                np.fft.fft(basis, n=self.n_fft)[: self.n_fft // 2 + 1] * (2.0 / self.n_fft)
            ).astype(np.complex64)
        return kernel

    def _apply_kernel(
        self,
        kr: np.ndarray,
        ki: np.ndarray,
        X_real: np.ndarray,
        X_imag: np.ndarray,
    ) -> np.ndarray:
        """Split-complex matmul: |conj(K) @ X| = |(kr - i·ki) @ (Xr + i·Xi)|.

        CQT analysis requires the conjugate of the synthesis kernel so that
        the filter responses align with positive-frequency signal content.
        """
        out_real = kr @ X_real + ki @ X_imag
        out_imag = kr @ X_imag - ki @ X_real
        return np.sqrt(out_real ** 2 + out_imag ** 2)

    def legacy_reference(self, signal: np.ndarray) -> np.ndarray:
        """
        Baseline CPU execution: frame-by-frame split-complex CQT filtering.
        """
        signal = signal.astype(np.float32)
        num_frames = 1 + (len(signal) - self.n_fft) // self.hop_length
        output = np.zeros((num_frames, self.n_bins), dtype=np.float32)

        for i in range(num_frames):
            start = i * self.hop_length
            X = np.fft.rfft(signal[start:start + self.n_fft]).astype(np.complex64)
            output[i] = self._apply_kernel(
                self._kernel_real_np, self._kernel_imag_np,
                X.real, X.imag,
            ).astype(np.float32)

        return output

    def accelerate(self, signal_tensor: torch.Tensor) -> torch.Tensor:
        """
        Executes parallelized hardware-aligned CQT computation graph.
        """
        if signal_tensor.dim() > 1:
            signal_tensor = signal_tensor.squeeze(0)

        stft_out = torch.stft(
            signal_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.ones(self.n_fft, device=self.device),
            center=False,
            return_complex=True,
        )
        # stft_out: (n_fft//2+1, n_frames)
        Xr, Xi = stft_out.real, stft_out.imag

        # conj(K) @ X: (kr - i·ki)(Xr + i·Xi)
        out_real = self.kernel_real @ Xr + self.kernel_imag @ Xi  # (n_bins, n_frames)
        out_imag = self.kernel_real @ Xi - self.kernel_imag @ Xr

        return torch.sqrt(out_real ** 2 + out_imag ** 2).T          # (n_frames, n_bins)


def cqt(
    signal: torch.Tensor,
    sample_rate: int = 16000,
    hop_length: int = 512,
    n_bins: int = 72,
    bins_per_octave: int = 12,
    f_min: float = 130.8,
    filter_scale: float = 1.0,
) -> torch.Tensor:
    """Exposes the hardware-aligned CQT operator to the tform namespace."""
    operator = CQTOperator(
        sample_rate=sample_rate,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        f_min=f_min,
        filter_scale=filter_scale,
        device=str(signal.device),
    )
    return operator.accelerate(signal)
