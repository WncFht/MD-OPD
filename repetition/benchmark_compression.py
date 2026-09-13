#!/usr/bin/env python3
"""Small latency benchmark for repetition compression settings.

The benchmark measures the same UTF-8/zlib-family operations used by
analyze_and_plot.py. It is intended for relative comparisons, not hardware
independent performance claims.
"""

from __future__ import annotations

import argparse
import bz2
import csv
import gzip
import json
import lzma
import random
import time
import zlib
from pathlib import Path

ALGORITHMS = ("zlib", "gzip", "bz2", "lzma")


def compress(raw: bytes, algorithm: str) -> bytes:
    if algorithm == "zlib":
        return zlib.compress(raw, 9)
    if algorithm == "gzip":
        return gzip.compress(raw, compresslevel=9)
    if algorithm == "bz2":
        return bz2.compress(raw, compresslevel=9)
    if algorithm == "lzma":
        return lzma.compress(raw, preset=9)
    raise ValueError(f"unsupported algorithm: {algorithm}")


def load_texts(eval_root: Path, limit: int, seed: int) -> list[str]:
    paths = sorted(eval_root.glob("**/*.jsonl"))
    rows: list[str] = []
    for path in paths:
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    rows.append(str(json.loads(line).get("response", "")))
    random.Random(seed).shuffle(rows)
    return rows[:limit]


def starts_for(length: int, window_size: int, stride: int) -> list[int]:
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    last = length - window_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "compression_benchmark.csv",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--window-sizes",
        nargs="+",
        type=int,
        default=[256, 512, 768, 1024, 1536, 2048, 4096],
    )
    parser.add_argument("--strides", nargs="+", type=int, default=[64, 128, 256, 384, 512, 768, 1024])
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    args = parser.parse_args()

    texts = load_texts(args.eval_root, args.limit, seed=42)
    if not texts:
        raise SystemExit(f"No JSONL responses found under {args.eval_root}")

    total_chars = sum(len(text) for text in texts)
    total_bytes = sum(len(text.encode("utf-8")) for text in texts)
    rows = []
    for algorithm in args.algorithms:
        for window_size in args.window_sizes:
            for stride in args.strides:
                windows = sum(len(starts_for(len(text), window_size, stride)) for text in texts)
                elapsed = []
                for _ in range(max(args.repeats, 1)):
                    start = time.perf_counter()
                    for text in texts:
                        raw = text.encode("utf-8")
                        for offset in starts_for(len(text), window_size, stride):
                            compress(raw[offset : offset + window_size], algorithm)
                    elapsed.append(time.perf_counter() - start)
                seconds = min(elapsed)
                rows.append(
                    {
                        "algorithm": algorithm,
                        "window_size": window_size,
                        "stride": stride,
                        "responses": len(texts),
                        "input_chars": total_chars,
                        "input_bytes": total_bytes,
                        "windows": windows,
                        "seconds": seconds,
                        "ms_per_response": 1000 * seconds / len(texts),
                        "us_per_window": 1_000_000 * seconds / max(windows, 1),
                        "ms_per_1k_chars": 1_000_000 * seconds / max(total_chars, 1),
                        "mb_per_second": total_bytes / max(seconds, 1e-12) / 1_000_000,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib.pyplot as plt

        figure_dir = args.output.parent / "figures" / "benchmark"
        figure_dir.mkdir(parents=True, exist_ok=True)
        for algorithm in args.algorithms:
            subset = [row for row in rows if row["algorithm"] == algorithm]
            points = sorted(
                (r for r in subset if r["window_size"] == 1024),
                key=lambda r: r["stride"],
            )
            if points:
                plt.figure(figsize=(7, 4.5))
                plt.plot(
                    [r["stride"] for r in points],
                    [r["ms_per_1k_chars"] for r in points],
                    marker="o",
                )
                plt.axvline(256, color="#777", linestyle="--", linewidth=1)
                plt.xlabel("Stride (window=1024)")
                plt.ylabel("Latency (ms / 1k input chars)")
                plt.title(f"{algorithm}: normalized latency vs stride")
                plt.grid(True, linestyle=":", alpha=0.6)
                plt.tight_layout()
                plt.savefig(figure_dir / f"{algorithm}__stride.png", dpi=180)
                plt.close()
            points = sorted(
                (r for r in subset if r["stride"] == 256),
                key=lambda r: r["window_size"],
            )
            if points:
                plt.figure(figsize=(7, 4.5))
                plt.plot(
                    [r["window_size"] for r in points],
                    [r["ms_per_1k_chars"] for r in points],
                    marker="o",
                )
                plt.axvline(1024, color="#777", linestyle="--", linewidth=1)
                plt.xlabel("Window size (stride=256)")
                plt.ylabel("Latency (ms / 1k input chars)")
                plt.title(f"{algorithm}: normalized latency vs window size")
                plt.grid(True, linestyle=":", alpha=0.6)
                plt.tight_layout()
                plt.savefig(figure_dir / f"{algorithm}__window_size.png", dpi=180)
                plt.close()
    except ImportError:
        print("matplotlib unavailable; CSV written without benchmark figures")
    print(f"Benchmarked {len(texts)} responses; wrote {args.output}")


if __name__ == "__main__":
    main()
