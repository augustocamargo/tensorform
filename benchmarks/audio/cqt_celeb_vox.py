"""
Benchmark: tform.audio.cqt — CelebVox dataset
==============================================

Usage
-----
    python benchmarks/audio/cqt_celeb_vox.py
    python benchmarks/audio/cqt_celeb_vox.py --max-files 100 --iterations 20 --warmup 5
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import tensorform as tform

DATASET_ROOT = "/Users/augustocamargo/Projects/MelFTF/dataset/celeb_vox/wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HAC CQT — CelebVox benchmark")
    parser.add_argument("--dataset", type=str, default=DATASET_ROOT)
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-bins", type=int, default=72)
    parser.add_argument("--bins-per-octave", type=int, default=12)
    parser.add_argument("--f-min", type=float, default=130.8)
    parser.add_argument("--filter-scale", type=float, default=1.0)
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
    dataset_stats = tform.bench.audio.compute_trial_stats(trials, sample_rate=args.sample_rate)
    dataset_stats["source"] = args.dataset
    report = tform.bench.audio.cqt(
        inputs=trials,
        iterations=args.iterations,
        warmup=args.warmup,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        n_bins=args.n_bins,
        bins_per_octave=args.bins_per_octave,
        f_min=args.f_min,
        filter_scale=args.filter_scale,
        target_sr=args.sample_rate,
    )

    run_config = {
        "dataset":          args.dataset,
        "num_trials":       len(trials),
        "iterations":       args.iterations,
        "warmup":           args.warmup,
        "sample_rate":      args.sample_rate,
        "hop_length":       args.hop_length,
        "n_bins":           args.n_bins,
        "bins_per_octave":  args.bins_per_octave,
        "f_min":            args.f_min,
        "filter_scale":     args.filter_scale,
    }
    report.print_summary("tform.audio.cqt — CelebVox", run_config=run_config, dataset_stats=dataset_stats)
    report.save_report("tform.audio.cqt — CelebVox", operator_slug="cqt", run_config=run_config, dataset_stats=dataset_stats)


if __name__ == "__main__":
    main()
