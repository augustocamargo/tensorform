import numpy as np
import tensorform as tform


def test_mfcc_equivalence_and_bench() -> None:
    """Validates MFCC algorithmic equivalence between legacy and HAC paths."""
    np.random.seed(42)
    sample_rate = 16000

    mock_trials = [
        np.random.uniform(-1.0, 1.0, sample_rate * duration).astype(np.float32)
        for duration in [1, 2, 3]
    ]

    report = tform.bench.audio.mfcc(
        inputs=mock_trials,
        iterations=5,
        warmup=1,
    )

    assert report.num_trials == 3
    assert report.mean_cosine > 0.999
    assert report.worst_max_error < 1e-3
