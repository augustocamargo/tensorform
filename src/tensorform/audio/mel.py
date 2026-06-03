import numpy as np
import torch
from typing import Optional

class MelSpectrogramOperator:
    """
    Hardware-Aligned Computational (HAC) of the Mel Spectrogram.
    
    Transforms sequential time-frequency windowing and filtering loops into
    parallelized matrix operations optimized for tensor accelerators.
    """
    
    def __init__(
        self, 
        sample_rate: int = 16000, 
        n_fft: int = 512, 
        hop_length: int = 160, 
        n_mels: int = 26, 
        device: Optional[str] = None
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        if device is None:
            if hasattr(torch, "mps") and torch.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        self.mel_fb_tensor = torch.from_numpy(self._generate_mel_fb()).float().to(self.device)

    def _generate_mel_fb(self) -> np.ndarray:
        """Generates a standard Mel filter bank matrix on the CPU."""
        low_mel = 0.0
        high_mel = 2595.0 * np.log10(1.0 + (self.sample_rate / 2.0) / 700.0)
        mel_points = np.linspace(low_mel, high_mel, self.n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bins = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        fb = np.zeros((self.n_fft // 2 + 1, self.n_mels))
        for m in range(1, self.n_mels + 1):
            for k in range(bins[m - 1], bins[m]):
                fb[k, m - 1] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
            for k in range(bins[m], bins[m + 1]):
                fb[k, m - 1] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
        return fb

    def legacy_reference(self, signal: np.ndarray) -> np.ndarray:
        """
        Baseline CPU execution mimicking conventional DSP sequential loops.

        Operates in float32 to match the accelerated path's precision and
        ensure fidelity comparisons are not biased by dtype differences.
        """
        signal = signal.astype(np.float32)
        num_samples = len(signal)
        num_frames = 1 + int(np.floor((num_samples - self.n_fft) / self.hop_length))
        window = np.hanning(self.n_fft).astype(np.float32)
        fft_segments = np.zeros((num_frames, self.n_fft // 2 + 1), dtype=np.float32)

        for i in range(num_frames):
            start = i * self.hop_length
            end = start + self.n_fft
            frame = signal[start:end] * window
            fft_res = np.fft.rfft(frame, n=self.n_fft)
            fft_segments[i] = np.abs(fft_res).astype(np.float32) ** 2

        return np.dot(fft_segments, self._generate_mel_fb().astype(np.float32))

    def accelerate(self, signal_tensor: torch.Tensor) -> torch.Tensor:
        """
        Executes parallelized hardware-aligned transformation graph.
        """
        if signal_tensor.dim() > 1:
            signal_tensor = signal_tensor.squeeze(0)

        window = torch.hann_window(self.n_fft, periodic=False, device=self.device)

        stft_out = torch.stft(
            signal_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=False,
            return_complex=True,
        )

        # stft_out: (freq, time) → power → (time, freq) → matmul → (time, n_mels)
        power_spectrogram = torch.abs(stft_out) ** 2
        return torch.matmul(power_spectrogram.T, self.mel_fb_tensor)


def melspectrogram(
    signal: torch.Tensor, 
    sample_rate: int = 16000, 
    n_fft: int = 512, 
    hop_length: int = 160, 
    n_mels: int = 26
) -> torch.Tensor:
    """Exposes the hardware-aligned operator to the tform namespace."""
    operator = MelSpectrogramOperator(
        sample_rate=sample_rate, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_mels=n_mels, 
        device=str(signal.device)
    )
    return operator.accelerate(signal)