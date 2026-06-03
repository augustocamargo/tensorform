"""
Generate a fixed-duration WAV dataset by concatenating source files.

Usage
-----
    python benchmarks/tools/gen_dataset.py
    python benchmarks/tools/gen_dataset.py --src /work/ms/datasets/Libri6000 \
        --dst /work/ms/datasets/Libri6000_100_160 \
        --n-files 100 --duration 160 --sample-rate 16000
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed-duration WAV dataset")
    parser.add_argument("--src",         default="/work/ms/datasets/Libri6000")
    parser.add_argument("--dst",         default="/work/ms/datasets/Libri6000_100_160")
    parser.add_argument("--n-files",     type=int,   default=100)
    parser.add_argument("--duration",    type=float, default=160.0, help="Seconds per output file")
    parser.add_argument("--sample-rate", type=int,   default=16000)
    return parser.parse_args()


def load_wav(path: str, target_sr: int) -> np.ndarray:
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from math import gcd

    sr, data = wavfile.read(path)

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != target_sr:
        g = gcd(target_sr, sr)
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)

    return data


def save_wav(path: str, data: np.ndarray, sr: int) -> None:
    from scipy.io import wavfile
    # Convert to int16 for compact storage
    pcm = np.clip(data * 32767.0, -32768, 32767).astype(np.int16)
    wavfile.write(path, sr, pcm)


def collect_source_paths(src_dir: str) -> list:
    paths = []
    for dirpath, _, filenames in os.walk(src_dir, followlinks=False):
        for fname in filenames:
            if fname.lower().endswith(".wav"):
                paths.append(os.path.join(dirpath, fname))
    return sorted(paths)


def main() -> None:
    args = parse_args()
    target_samples = int(args.duration * args.sample_rate)

    Path(args.dst).mkdir(parents=True, exist_ok=True)

    print(f"Source:      {args.src}")
    print(f"Destination: {args.dst}")
    print(f"Output:      {args.n_files} files × {args.duration}s @ {args.sample_rate} Hz")
    print(f"Target:      {target_samples:,} samples per file\n")

    src_paths = collect_source_paths(args.src)
    print(f"Found {len(src_paths)} source WAV files.")
    if not src_paths:
        print("No source files found. Exiting.")
        sys.exit(1)

    # Cycle through source files as needed
    src_cycle = (src_paths * ((args.n_files * target_samples) // (len(src_paths) * args.sample_rate * 10) + 2))

    buffer = np.zeros(0, dtype=np.float32)
    src_idx = 0
    files_written = 0

    while files_written < args.n_files:
        # Fill buffer until we have enough samples
        while len(buffer) < target_samples:
            if src_idx >= len(src_cycle):
                print("Ran out of source audio. Increase source dataset or reduce --n-files.")
                sys.exit(1)
            path = src_cycle[src_idx]
            src_idx += 1
            try:
                chunk = load_wav(path, args.sample_rate)
                buffer = np.concatenate([buffer, chunk])
            except Exception as exc:
                print(f"  [skip] {Path(path).name}: {exc}")
                continue

        # Slice exactly target_samples
        segment = buffer[:target_samples]
        buffer = buffer[target_samples:]

        out_path = os.path.join(args.dst, f"{files_written:04d}.wav")
        save_wav(out_path, segment, args.sample_rate)
        files_written += 1

        if files_written % 10 == 0 or files_written == args.n_files:
            print(f"  Written {files_written}/{args.n_files}  ({out_path})")

    print(f"\nDone. {files_written} files saved to {args.dst}")


if __name__ == "__main__":
    main()
