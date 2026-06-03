"""
Run all TensorForm audio benchmarks sequentially.

Usage
-----
    python benchmarks/audio/run_all.py
    python benchmarks/audio/run_all.py --max-files 100 --iterations 20 --warmup 5
"""

import sys
import argparse
import subprocess
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
OPERATORS = ["mel", "mfcc", "cqt", "gammatone", "melt", "mfcct"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all HAC audio benchmarks")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset path for all operators")
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=16000)
    return parser.parse_args()


def build_cmd(script: Path, args: argparse.Namespace) -> list:
    cmd = [sys.executable, str(script),
           "--max-files", str(args.max_files),
           "--iterations", str(args.iterations),
           "--warmup", str(args.warmup),
           "--sample-rate", str(args.sample_rate)]
    if args.dataset:
        cmd += ["--dataset", args.dataset]
    return cmd


def main() -> None:
    args = parse_args()
    results = {}

    for op in OPERATORS:
        script = SCRIPTS_DIR / f"{op}_celeb_vox.py"
        print(f"\n{'='*75}")
        print(f"  Running: {script.name}")
        print(f"{'='*75}\n")

        ret = subprocess.run(build_cmd(script, args))
        results[op] = "OK" if ret.returncode == 0 else f"FAILED (exit {ret.returncode})"

    print(f"\n{'='*75}")
    print("  SUMMARY")
    print(f"{'='*75}")
    for op, status in results.items():
        print(f"  {op:<12} {status}")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
