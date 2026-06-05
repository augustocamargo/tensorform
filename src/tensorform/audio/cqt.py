import numpy as np
import torch
from typing import Optional

from tensorform._device import detect_device


class CQTOperator:
    """
    Hardware-Aligned Computational (HAC) of the Constant-Q Transform.

    Precomputes geometrically-spaced, Hann-windowed complex sinusoid kernels
    in the frequency domain. The accelerated path is backend-specific:

    - **CUDA**: single complex GEMM (ZGEMM via cuBLAS) on the conjugate kernel.
      Avoids 4× SGEMM + element-wise ops of the split-real path.
    - **MPS / CPU**: split-complex float32 arithmetic (MPS has no complex matmul).

    Legacy reference uses split-real on CPU for all platforms.
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
        self.n_fft = int(2 ** np.ceil(np.log2(self.Q * sample_rate / f_min)))

        if device is None:
            self.device = torch.device(detect_device())
        else:
            self.device = torch.device(device)

        kernel = self._generate_kernel()                              # (n_bins, n_fft//2+1) complex64
        self._kernel_real_np = kernel.real
        self._kernel_imag_np = kernel.imag

        if self.device.type == "cuda":
            # ZGEMM path: store full complex kernel on device
            self.kernel_complex = torch.from_numpy(kernel).to(self.device)
        else:
            # Split-real path: MPS / CPU
            self.kernel_real = torch.from_numpy(self._kernel_real_np).to(self.device)
            self.kernel_imag = torch.from_numpy(self._kernel_imag_np).to(self.device)

    def _generate_kernel(self) -> np.ndarray:
        """
        CQT filter bank in frequency domain, shape (n_bins, n_fft//2+1) complex64.

        Each row is the DFT of a Hann-windowed complex sinusoid at the
        corresponding geometrically spaced center frequency, normalized by n_fft.
        """
        kernel = np.zeros((self.n_bins, self.n_fft // 2 + 1), dtype=np.complex64)
        for k in range(self.n_bins):
            f_k = self.f_min * 2.0 ** (k / self.bins_per_octave)
            N_k = min(int(round(self.Q * self.sample_rate / f_k)), self.n_fft)
            n = np.arange(N_k, dtype=np.float32)
            window = np.hanning(N_k).astype(np.float32)
            basis = (window * np.exp(2j * np.pi * self.Q * n / N_k) / N_k).astype(np.complex64)
            kernel[k] = (
                np.fft.fft(basis, n=self.n_fft)[: self.n_fft // 2 + 1] * (2.0 / self.n_fft)
            ).astype(np.complex64)
        return kernel

    def legacy_reference(self, signal: np.ndarray) -> np.ndarray:
        """Baseline CPU: frame-by-frame split-complex CQT filtering."""
        signal = signal.astype(np.float32)
        num_frames = 1 + (len(signal) - self.n_fft) // self.hop_length
        output = np.zeros((num_frames, self.n_bins), dtype=np.float32)
        kr, ki = self._kernel_real_np, self._kernel_imag_np

        for i in range(num_frames):
            start = i * self.hop_length
            X = np.fft.rfft(signal[start:start + self.n_fft]).astype(np.complex64)
            # conj(K) @ X: (kr - i·ki)(Xr + i·Xi)
            out_real = kr @ X.real + ki @ X.imag
            out_imag = kr @ X.imag - ki @ X.real
            output[i] = np.sqrt(out_real ** 2 + out_imag ** 2).astype(np.float32)

        return output

    def accelerate(self, signal_tensor: torch.Tensor) -> torch.Tensor:
        """
        Executes hardware-specific CQT computation.

        CUDA: conj(kernel_complex) @ stft  — single ZGEMM (cuBLAS).
        MPS/CPU: split-complex float32 matmul.
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

        if self.device.type == "cuda":
            # Single ZGEMM — conj(K) is the analysis kernel
            out = torch.conj(self.kernel_complex) @ stft_out        # (n_bins, n_frames) complex
            return torch.abs(out).T                                  # (n_frames, n_bins)
        else:
            # Split-real: MPS / CPU — conj(K) @ X = (kr+ki·Xi, kr·Xi-ki·Xr)
            Xr, Xi = stft_out.real, stft_out.imag
            out_real = self.kernel_real @ Xr + self.kernel_imag @ Xi
            out_imag = self.kernel_real @ Xi - self.kernel_imag @ Xr
            return torch.sqrt(out_real ** 2 + out_imag ** 2).T


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
