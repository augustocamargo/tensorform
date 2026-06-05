"""
Run all TensorForm audio benchmarks.

By default runs the scaling study (grid [1,3,5,10,20,40,60,80,160]s) for all
operators via bench_scaling.py. Use --fixed to run the single-duration full
telemetry benchmarks (bench_*.py) instead.

Usage
-----
    python benchmarks/audio/run_all.py                      # scaling study (default)
    python benchmarks/audio/run_all.py --fixed              # fixed-duration benchmarks
    python benchmarks/audio/run_all.py --max-files 50       # fewer files (faster)
"""

import sys
import argparse
import subprocess
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
OPERATORS   = ["mel", "mfcc", "cqt", "gammatone"]

# Auto-detect dataset path
_DATASET_NVIDIA  = "/work/ms/datasets/Libri6000_100_160"
_DATASET_MAC     = "/Users/augustocamargo/Projects/MelFTF/dataset/Libri6000_100_160"
_DEFAULT_DATASET = _DATASET_NVIDIA if Path(_DATASET_NVIDIA).exists() else _DATASET_MAC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all TensorForm audio benchmarks")
    parser.add_argument("--fixed",       action="store_true",
                        help="Run fixed-duration bench_*.py instead of scaling study")
    parser.add_argument("--dataset",     type=str, default=_DEFAULT_DATASET)
    parser.add_argument("--max-files",   type=int, default=100)
    parser.add_argument("--iterations",  type=int, default=20)
    parser.add_argument("--warmup",      type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=16000)
    return parser.parse_args()


def run_scaling(args) -> dict:
    """Run bench_scaling.py --operator all."""
    script = SCRIPTS_DIR / "bench_scaling.py"
    cmd = [
        sys.executable, str(script),
        "--operator",   "all",
        "--dataset",    args.dataset,
        "--max-files",  str(args.max_files),
        "--iterations", str(args.iterations),
        "--warmup",     str(args.warmup),
        "--sample-rate", str(args.sample_rate),
    ]
    print(f"\n{'='*75}")
    print(f"  Running: bench_scaling.py --operator all")
    print(f"  Dataset: {args.dataset}")
    print(f"{'='*75}\n")
    ret = subprocess.run(cmd)
    return {"scaling (all operators)": "OK" if ret.returncode == 0 else f"FAILED (exit {ret.returncode})"}


def run_fixed(args) -> dict:
    """Run each bench_{op}.py at the source file duration."""
    results = {}
    for op in OPERATORS:
        script = SCRIPTS_DIR / f"bench_{op}.py"
        cmd = [
            sys.executable, str(script),
            "--dataset",    args.dataset,
            "--max-files",  str(args.max_files),
            "--iterations", str(args.iterations),
            "--warmup",     str(args.warmup),
            "--sample-rate", str(args.sample_rate),
        ]
        print(f"\n{'='*75}")
        print(f"  Running: {script.name}")
        print(f"{'='*75}\n")
        ret = subprocess.run(cmd)
        results[op] = "OK" if ret.returncode == 0 else f"FAILED (exit {ret.returncode})"
    return results


def main() -> None:
    args = parse_args()

    if args.fixed:
        results = run_fixed(args)
    else:
        results = run_scaling(args)

    print(f"\n{'='*75}")
    print("  SUMMARY")
    print(f"{'='*75}")
    for key, status in results.items():
        print(f"  {key:<32} {status}")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
