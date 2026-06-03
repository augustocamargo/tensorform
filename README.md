# TensorForm

**Hardware-Aligned Computational (HAC) reformulation of scientific signal-processing pipelines.**

TensorForm replaces conventional DSP sequential loops with parallelized tensor graph operations that run natively on Apple Silicon (MPS), NVIDIA (CUDA), and CPU — with no code changes required between backends.

---

## Why TensorForm

Classical signal-processing libraries were designed for single-core CPUs. Every operator — FFT windowing, filter bank convolution, spectral aggregation — executes frame by frame in a sequential loop. Modern hardware accelerators cannot leverage this pattern.

TensorForm reformulates each operator as a static tensor computation graph: batched matrix operations that map directly onto the parallel execution model of GPU and NPU silicon. The legacy sequential implementation is preserved as a reference baseline for fidelity validation and benchmarking.

---

## Features

- **Accelerator-native** — auto-detects and targets MPS, CUDA, or CPU at runtime
- **Fidelity-validated** — every operator ships with a telemetry harness that measures cosine similarity and L∞ error between the HAC and legacy paths
- **Backend-agnostic API** — same call signature regardless of hardware
- **Benchmark infrastructure** — multi-trial execution harness with mean ± SD latency, energy estimation, and environment metadata
- **Open source** — MIT licensed

---

## Installation

**From source (development):**

```bash
git clone https://github.com/augustocamargo/TensorForm.git
cd TensorForm
pip install -e ".[dev]"
```

**NVIDIA energy telemetry (optional):**

```bash
pip install -e ".[nvidia]"
```

**Runtime requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, NumPy ≥ 1.22

---

## Quick Start

```python
import torch
import tensorform as tform

signal = torch.randn(16000)  # 1-second mono audio at 16 kHz

mel = tform.audio.melspectrogram(signal)
# → Tensor of shape (97, 26) on the auto-detected accelerator
```

Device selection is automatic: MPS → CUDA → CPU, in that priority order.

---

## Operators

### `tform.audio`

| Operator | Description |
|---|---|
| `melspectrogram(signal, sample_rate, n_fft, hop_length, n_mels)` | HAC Mel Spectrogram |

---

## Benchmarking

TensorForm ships a telemetry harness that profiles each operator against its legacy CPU reference across multiple trials, reporting execution latency, energy estimates, and mathematical fidelity.

### Synthetic benchmark

```python
import numpy as np
import tensorform as tform

trials = [
    np.random.uniform(-1.0, 1.0, 16000 * d).astype(np.float32)
    for d in [1, 2, 3]
]

report = tform.bench.audio.melspectrogram(
    inputs=trials,
    iterations=50,
    warmup=10,
)

report.print_summary("tform.audio.melspectrogram")
```

### Real dataset benchmark

```python
trials = tform.bench.audio.collect_wav_trials(
    root_dir="/path/to/wav/dataset",
    max_files=100,
    target_sr=16000,
)

report = tform.bench.audio.melspectrogram(inputs=trials, iterations=20, warmup=5)
report.print_summary("tform.audio.melspectrogram — CelebVox")
```

Pre-built benchmark scripts are in [`benchmarks/`](benchmarks/).

### Sample report

```
===========================================================================
 AGGREGATED BENCHMARK REPORT (50 Evaluation Trials)
 Operator: tform.audio.melspectrogram
===========================================================================
 SYSTEM ENVIRONMENT METADATA:
  ├─ Timestamp:               2026-06-03 14:22:01
  ├─ Host CPU Model:          Apple M2 Pro (12 Cores)
  └─ Target Accelerator:      Apple Silicon Integrated GPU
---------------------------------------------------------------------------
 EXECUTION LATENCY (Cross-Trial Mean ± SD):
  ├─ Legacy CPU Reference:    8.4231 ± 0.21 ms
  ├─ Accelerator-Native:      0.9184 ± 0.04 ms
  └─ Sustained Speedup:       9.17x
---------------------------------------------------------------------------
 ENERGY EFFICIENCY (Mean per Trial):
  ├─ Legacy CPU Reference:    0.3790 mJ
  ├─ Accelerator-Native:      0.0138 mJ
  └─ Energy Efficiency Gain:  27.46x less energy consumed
---------------------------------------------------------------------------
 MATHEMATICAL FIDELITY ACCUMULATED:
  ├─ Mean Cosine Similarity:  1.00000000
  └─ Worst-Case Abs Error:    3.66210938e-04
===========================================================================
```

---

## Namespace Conventions

TensorForm follows a strict two-root namespace:

```
tform.<domain>.<operator>          # production operator
tform.bench.<domain>.<operator>    # telemetry harness (mirrors production)
```

Example:

```
tform.audio.melspectrogram         # HAC production call
tform.bench.audio.melspectrogram   # benchmark + fidelity report
```

---

## Running Tests

```bash
pytest tests/
```

---

## Roadmap

- `tform.audio` — Mel Spectrogram ✅, MFCC, CQT, Gammatone
- `tform.radar` — Range-Doppler map, CFAR detection
- `tform.io` — Dataset loaders and resampling utilities

---

## Contributing

Contributions are welcome. Please open an issue before submitting a pull request for new operators or backend changes. All operators must include:

1. A production implementation under `tform.<domain>.<operator>`
2. A mirrored telemetry harness under `tform.bench.<domain>.<operator>`
3. A unit test validating fidelity between the two paths

---

## License

MIT © [Augusto Cesar de Camargo Neto](mailto:augusto.camargo@bluecore.it)
