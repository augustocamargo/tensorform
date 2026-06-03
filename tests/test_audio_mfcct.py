import numpy as np
import tensorform as tform


def test_mfcct_equivalence_and_bench() -> None:
    """Validates MFCCT (Direct Mel Projection + DCT-II) HAC vs CPU fidelity."""
    np.random.seed(42)
    sample_rate = 16000

    # Signals must be longer than n_fft=1200 (paper cepstral default)
    mock_trials = [
        np.random.uniform(-1.0, 1.0, sample_rate * duration).astype(np.float32)
        for duration in [1, 2, 3]
    ]

    report = tform.bench.audio.mfcct(
        inputs=mock_trials,
        iterations=5,
        warmup=1,
    )

    assert report.num_trials == 3
    assert report.mean_cosine > 0.999
    assert report.worst_max_error < 1e-3
