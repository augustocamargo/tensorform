# TensorForm

**Hardware-Aligned Computational (HAC) reformulation of scientific signal-processing pipelines.**

TensorForm replaces conventional DSP sequential loops with parallelized tensor graph operations that run natively on Apple Silicon (MPS), NVIDIA (CUDA), and CPU — with no code changes required between backends.

---

## Why TensorForm

Classical signal-processing libraries were designed for single-core CPUs. Every operator executes frame by frame in a sequential loop. Modern hardware accelerators cannot leverage this pattern.

TensorForm reformulates each operator as a static tensor computation graph: batched matrix operations that map directly onto the parallel execution model of GPU and NPU silicon. The legacy sequential path is preserved as a fidelity reference.

The central example is **MelT** — a single-stage GEMM-native audio frontend that eliminates the conventional STFT entirely, replacing the two-step STFT+Mel pipeline with a direct Mel-spaced NDFT projection expressed as two dense matrix multiplications.

> Camargo & Finger, *"MelT: GEMM-Native NDFT for Efficient Single-Stage Audio Frontends on Modern Accelerators"*, arXiv 2026.

---

## Features

- **STFT-free Mel frontend** — Direct Mel Projection (MelT / MFCCT) eliminates intermediate spectral tensors
- **Accelerator-native** — auto-detects MPS, CUDA, or CPU at runtime; priority: MPS → CUDA → CPU
- **Fidelity-validated** — every operator ships with a harness measuring cosine similarity and L∞ error
- **Competitive benchmarks** — built-in comparison against librosa, torchaudio, and nnAudio on real datasets
- **Python ≥ 3.8 compatible** — tested on Apple Silicon (MPS) and NVIDIA (CUDA)
- **Open source** — MIT licensed

---

## Installation

```bash
git clone https://github.com/augustocamargo/TensorForm.git
cd TensorForm
pip install -e ".[dev]"
```

Optional contenders for comparison benchmarks:

```bash
pip install librosa torchaudio nnAudio scipy
```

NVIDIA energy telemetry:

```bash
pip install -e ".[nvidia]"
```

---

## Quick Start

```python
import torch
import tensorform as tform

signal = torch.randn(16000)  # 1-second mono audio at 16 kHz

# Direct Mel Projection — no STFT, single-stage GEMM
mel = tform.audio.melt(signal)                    # (98, 26) on MPS/CUDA/CPU

# Cepstral extension
mfcc = tform.audio.mfcct(signal)                  # (98, 13)

# Conventional HAC operators
mel_stft = tform.audio.melspectrogram(signal)     # STFT + Mel filterbank
cqt      = tform.audio.cqt(signal)               # Freq-domain kernel matmul
gamma    = tform.audio.gammatone(signal)          # Split-complex ERB matmul
```

---

## Operators

### `tform.audio`

| Operator | Description | Approach |
|---|---|---|
| `melt(signal, ...)` | Direct Mel Projection | Two GEMMs — **no STFT** |
| `mfcct(signal, ...)` | Direct Mel Projection + DCT-II | Three GEMMs — **no STFT** |
| `melspectrogram(signal, ...)` | Mel Spectrogram | HAC STFT + mel matmul |
| `mfcc(signal, ...)` | MFCC | HAC STFT + mel + DCT |
| `cqt(signal, ...)` | Constant-Q Transform | Freq-domain kernel matmul |
| `gammatone(signal, ...)` | Gammatone Filterbank | Split-complex ERB matmul |

---

## Benchmarks

### Competitive comparison (Apple Silicon MPS, CelebVox)

```
tform.audio.melt  vs  librosa / torchaudio / nnAudio
─────────────────────────────────────────────────────
1st  TensorForm MelT  (MPS)     0.207 ms  —
2nd  torchaudio       (MPS)     0.384 ms  1.9x slower
3rd  nnAudio          (CPU)     1.318 ms  6.4x slower  ← no MPS support
4th  librosa          (CPU)     1.462 ms  7.1x slower

tform.audio.cqt  vs  librosa / nnAudio
─────────────────────────────────────────────────────
1st  TensorForm CQT   (MPS)     0.498 ms  —
2nd  nnAudio CQT      (CPU)     2.346 ms  4.7x slower  ← no MPS support
3rd  librosa CQT      (CPU)     5.158 ms  10.3x slower
```

### Running benchmarks

```bash
# Single operator vs all contenders
python benchmarks/audio/bench_mel.py --dataset /path/to/wav
python benchmarks/audio/bench_mfcc.py
python benchmarks/audio/bench_cqt.py
python benchmarks/audio/bench_gammatone.py

# All operators at once
python benchmarks/audio/run_all.py --dataset /path/to/wav

# Generate a fixed-duration dataset
python benchmarks/tools/gen_dataset.py --src /path/to/wav --dst /path/to/out \
    --n-files 100 --duration 160
```

Results are saved to `benchmarks/results/bench_{operator}_{hostname}_{device}.txt`.

---

## Namespace Conventions

```
tform.<domain>.<operator>          # production operator
tform.bench.<domain>.<operator>    # fidelity harness (mirrors production)
```

---

## Running Tests

```bash
pytest tests/
```

---

## Roadmap

**`tform.audio`**
- MelT / MFCCT (Direct Mel Projection) ✅
- Mel Spectrogram ✅
- MFCC ✅
- CQT ✅
- Gammatone ✅
- MFCC sweep over Mel bins *(planned)*

**`tform.radar`** — Range-Doppler map, CFAR detection *(planned)*

---

## Contributing

All operators must include:

1. A production implementation under `tform.<domain>.<operator>`
2. A mirrored fidelity harness under `tform.bench.<domain>.<operator>`
3. A unit test validating equivalence between the two paths

---

## License

MIT © [Augusto Cesar de Camargo Neto](mailto:augusto.camargo@bluecore.it)
