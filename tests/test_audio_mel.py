import numpy as np
import tensorform as tform

def test_melspectrogram_equivalence_and_bench() -> None:
    """Validates absolute algorithmic equivalence constraints and telemetry engine."""
    np.random.seed(42)
    sample_rate = 16000
    
    # Generate 3 deterministic array trials mimicking input streams
    mock_trials = [
        np.random.uniform(-1.0, 1.0, sample_rate * duration).astype(np.float32) 
        for duration in [1, 2, 3]
    ]
    
    report = tform.bench.audio.melspectrogram(
        inputs=mock_trials, 
        iterations=5, 
        warmup=1
    )
    
    assert report.num_trials == 3
    assert report.mean_cosine > 0.999
    assert report.worst_max_error < 1e-3