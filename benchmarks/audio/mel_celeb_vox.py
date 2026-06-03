"""
Benchmark: tform.audio.melspectrogram — CelebVox dataset
=========================================================
Runs the HAC Mel Spectrogram operator against real speech recordings from the
CelebVox corpus and prints an aggregated telemetry report.

Usage
-----
    python benchmarks/audio/mel_celeb_vox.py
    python benchmarks/audio/mel_celeb_vox.py --max-files 100 --iterations 20 --warmup 5
"""

import sys
import argparse
from pathlib import Path

# Allow running without installing tensorform (development mode)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import tensorform as tform

DATASET_ROOT = "/Users/augustocamargo/Projects/MelFTF/dataset/celeb_vox/wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HAC MelSpectrogram — CelebVox benchmark")
    parser.add_argument("--dataset", type=str, default=DATASET_ROOT, help="Path to celeb_vox/wav root")
    parser.add_argument("--max-files", type=int, default=50, help="Maximum number of WAV files to load")
    parser.add_argument("--iterations", type=int, default=20, help="Timed iterations per trial")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per trial")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate in Hz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading up to {args.max_files} WAV files from: {args.dataset}")
    trials = tform.bench.audio.collect_wav_trials(
        root_dir=args.dataset,
        max_files=args.max_files,
        target_sr=args.sample_rate,
    )

    if not trials:
        print("No WAV files found. Check --dataset path.")
        return

    print(f"Loaded {len(trials)} trials. Running benchmark...\n")
    report = tform.bench.audio.melspectrogram(
        inputs=trials,
        iterations=args.iterations,
        warmup=args.warmup,
        target_sr=args.sample_rate,
    )

    report.print_summary("tform.audio.melspectrogram — CelebVox")


if __name__ == "__main__":
    main()
