import numpy as np
import torch
from typing import Optional

from tensorform._device import detect_device


class GammatoneOperator:
    """
    Hardware-Aligned Computational (HAC) of the Gammatone Filterbank.

    Models the basilar membrane's frequency selectivity using 4th-order
    gammatone filters at ERB-scale spaced center frequencies. The accelerated
    path is backend-specific:

    - **CUDA**: single complex GEMM (ZGEMM via cuBLAS) on the precomputed kernel.
      Avoids 4× SGEMM + element-wise ops of the split-real path.
    - **MPS / CPU**: split-complex float32 arithmetic (MPS has no complex matmul).

    Kernel impulse responses are computed in float64 for numerical accuracy, then
    cast to float32 before storage.
    """

    _ORDER: int = 4
    _BANDWIDTH_COEF: float = 1.019

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_filters: int = 32,
        f_min: float = 80.0,
        f_max: float = 8000.0,
        device: Optional[str] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_filters = n_filters
        self.f_min = f_min
        self.f_max = f_max
        self.center_freqs = self._erb_spaced_freqs()

        if device is None:
            self.device = torch.device(detect_device())
        else:
            self.device = torch.device(device)

        kernel = self._generate_kernel()                              # (n_filters, n_fft//2+1) complex64
        self._kernel_real_np = kernel.real
        self._kernel_imag_np = kernel.imag

        if self.device.type == "cuda":
            # ZGEMM path: store full complex kernel on device
            self.kernel_complex = torch.from_numpy(kernel).to(self.device)
        else:
            # Split-real path: MPS / CPU
            self.kernel_real = torch.from_numpy(self._kernel_real_np).to(self.device)
            self.kernel_imag = torch.from_numpy(self._kernel_imag_np).to(self.device)

    def _erb_spaced_freqs(self) -> np.ndarray:
        """ERB-scale linearly spaced center frequencies between f_min and f_max."""
        erb_min = 21.4 * np.log10(4.37 * self.f_min / 1000.0 + 1.0)
        erb_max = 21.4 * np.log10(4.37 * self.f_max / 1000.0 + 1.0)
        erb_centers = np.linspace(erb_min, erb_max, self.n_filters)
        return ((10.0 ** (erb_centers / 21.4) - 1.0) / 4.37 * 1000.0).astype(np.float32)

    def _generate_kernel(self) -> np.ndarray:
        """
        Gammatone filter bank in frequency domain, shape (n_filters, n_fft//2+1) complex64.

        Each row is the DFT of a 4th-order gammatone impulse response tuned to
        the corresponding ERB-spaced center frequency.
        """
        kernel = np.zeros((self.n_filters, self.n_fft // 2 + 1), dtype=np.complex64)
        t = np.arange(self.n_fft, dtype=np.float64) / self.sample_rate

        for i, f_c in enumerate(self.center_freqs):
            erb = 24.7 * (4.37 * float(f_c) / 1000.0 + 1.0)
            decay = 2.0 * np.pi * self._BANDWIDTH_COEF * erb
            g = (t ** (self._ORDER - 1)) * np.cos(2.0 * np.pi * f_c * t) * np.exp(-decay * t)
            peak = np.abs(g).max()
            if peak > 0.0:
                g /= peak
            kernel[i] = np.fft.rfft(g.astype(np.float32), n=self.n_fft).astype(np.complex64)

        return kernel

    def legacy_reference(self, signal: np.ndarray) -> np.ndarray:
        """Baseline CPU: frame-by-frame split-complex gammatone filtering."""
        signal = signal.astype(np.float32)
        num_frames = 1 + (len(signal) - self.n_fft) // self.hop_length
        output = np.zeros((num_frames, self.n_filters), dtype=np.float32)
        kr, ki = self._kernel_real_np, self._kernel_imag_np

        for i in range(num_frames):
            start = i * self.hop_length
            X = np.fft.rfft(signal[start:start + self.n_fft]).astype(np.complex64)
            Xr, Xi = X.real, X.imag
            out_real = kr @ Xr - ki @ Xi
            out_imag = kr @ Xi + ki @ Xr
            output[i] = np.sqrt(out_real ** 2 + out_imag ** 2).astype(np.float32)

        return output

    def accelerate(self, signal_tensor: torch.Tensor) -> torch.Tensor:
        """
        Executes hardware-specific gammatone filterbank computation.

        CUDA: kernel_complex @ stft — single ZGEMM (cuBLAS).
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
            # Single ZGEMM
            out = self.kernel_complex @ stft_out                     # (n_filters, n_frames) complex
            return torch.abs(out).T                                  # (n_frames, n_filters)
        else:
            # Split-real: MPS / CPU — K @ X = (kr·Xr - ki·Xi, kr·Xi + ki·Xr)
            Xr, Xi = stft_out.real, stft_out.imag
            out_real = self.kernel_real @ Xr - self.kernel_imag @ Xi
            out_imag = self.kernel_real @ Xi + self.kernel_imag @ Xr
            return torch.sqrt(out_real ** 2 + out_imag ** 2).T


def gammatone(
    signal: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_filters: int = 32,
    f_min: float = 80.0,
    f_max: float = 8000.0,
) -> torch.Tensor:
    """Exposes the hardware-aligned Gammatone operator to the tform namespace."""
    operator = GammatoneOperator(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_filters=n_filters,
        f_min=f_min,
        f_max=f_max,
        device=str(signal.device),
    )
    return operator.accelerate(signal)
