#!/usr/bin/env python3
"""Benchmark the D_J pipeline: JSON load, repetition scan, tokenize, 4-grams, pairs.

Reports per-stage latency per 1k tokens on real rollout samples, plus the
implied cost of one (model, bench) combo and of the full run.  Single-process,
warm-cache numbers; the full run additionally pays JSON parsing, process spawn
and memory pressure, so its measured wall time is reported alongside.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, Path(__file__).resolve().parent)

import jaccard_diversity as jd  # noqa: E402

SAMPLES_PER_MODE = 64
GROUP_SIZE = 16  # rollouts per problem in the eval dumps


def load_sample(eval_root: Path, mode: str, model: str, bench: str, limit: int) -> tuple[list[str], list[dict]]:
    """(responses, raw json lines) for latency measurement."""
    path = eval_root / mode / model / f"{bench}_t0.7_p0.95_n16-MNT31744.jsonl"
    texts, raw = [], []
    with path.open() as stream:
        for line in stream:
            if line.strip():
                raw.append(line)
                texts.append(json.loads(line)["response"])
            if len(texts) >= limit:
                break
    return texts, raw


def bench_stage(fn):
    start = time.perf_counter()
    out = fn()
    return time.perf_counter() - start, out


def per_1k(seconds: float, tokens: int) -> str:
    return f"{seconds / tokens * 1e6:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root", type=Path, required=True
    )
    args = parser.parse_args()

    tokenizer = jd.get_tokenizer(str(jd.DEFAULT_TOKENIZER))
    print(f"{'stage':<44}{'ms / 1k tokens':>16}")
    mode_tokens = {}
    encoded_by_mode = {}
    for mode, model in (("thinking", "Qwen3-4B"), ("no-thinking", "Qwen3-1.7B")):
        texts, raw_lines = load_sample(args.eval_root, mode, model, "aime24", SAMPLES_PER_MODE)

        seconds, rows = bench_stage(lambda raw_lines=raw_lines: [json.loads(line) for line in raw_lines])
        mb = sum(len(line) for line in raw_lines) / 1e6
        print(f"json.loads ({mode}, {mb:.1f} MB){'':<24}{seconds / mb:>13.2f} s/MB")

        seconds, encoded = bench_stage(lambda texts=texts: tokenizer.encode_batch(texts))
        tokens = sum(len(e.ids) for e in encoded)
        mode_tokens[mode] = tokens
        encoded_by_mode[mode] = encoded
        print(f"tokenize ({mode}, {tokens / 1e3:,.0f}k tokens){'':<12}{per_1k(seconds, tokens):>16}")

        seconds, _ = bench_stage(lambda texts=texts: [jd.window_ratios(t) for t in texts])
        print(f"repetition scan, zlib windows ({mode}){'':<8}{per_1k(seconds, tokens):>16}")

        seconds, gram_lists = bench_stage(lambda encoded=encoded: [jd.ngram_keys(e.ids) for e in encoded])
        print(f"4-gram lists ({mode}){'':<26}{per_1k(seconds, tokens):>16}")

        seconds, _ = bench_stage(lambda gram_lists=gram_lists: [jd.gram_sets_per_L(g) for g in gram_lists])
        print(f"per-L frozensets, 8 L values ({mode}){'':<11}{per_1k(seconds, tokens):>16}")

    # Pairwise stage: one 16-response group, all 8 L values, thinking lengths.
    group_sets = [jd.gram_sets_per_L(jd.ngram_keys(e.ids)) for e in encoded_by_mode["thinking"][:GROUP_SIZE]]
    group_tokens = sum(len(e.ids) for e in encoded_by_mode["thinking"][:GROUP_SIZE])
    seconds, _ = bench_stage(lambda: [jd.mean_pairwise_distance([sets[L] for sets in group_sets]) for L in jd.L_VALUES])
    print(f"pairwise+mean (16 resp x 8 L, thinking){'':<7}{per_1k(seconds, group_tokens):>16}")

    # Full-run accounting from length_stats.csv.
    stats = list(csv.DictReader(open(Path(__file__).resolve().parent / "results" / "length_stats.csv")))
    run_tokens = sum(int(r["n_responses"]) * float(r["len_mean"]) for r in stats)
    log = Path(__file__).resolve().parent / "compute.log"
    wall = None
    if log.exists():
        st = log.stat()
        wall = st.st_mtime - st.st_birthtime
    print(
        f"\nfull run: {run_tokens / 1e6:,.0f}M tokens over {len(stats)} combos; wall={wall / 60:.0f} min with 6 workers"
        if wall
        else f"\nfull run: {run_tokens / 1e6:,.0f}M tokens"
    )
    if wall:
        print(
            f"end-to-end throughput: {run_tokens / 1e3 / wall:,.0f}k tokens/s with 6 workers "
            f"(~{wall * 6 * 1e6 / run_tokens:.1f} ms / 1k tokens per core)"
        )


if __name__ == "__main__":
    main()
