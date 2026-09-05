#!/usr/bin/env python3
"""Pairwise token-level 4-gram Jaccard diversity D_J(L), following arXiv:2608.31046 App. B.3.

For every (mode, model, bench) rollout dump, responses are grouped by problem
(``example_id``).  Each response is tokenized with the Qwen3 tokenizer, truncated
to its first ``L`` tokens, and reduced to the set of unique token-level 4-grams.
The per-problem score is the mean pairwise Jaccard distance over all response
pairs; the reported D_J(L) is the macro-average over problems.

Two conditions are produced per (model, bench):

- ``all``: every sampled response participates.
- ``no-repetition``: responses flagged as repetition by the repo's zlib window
  detector (window=1024, stride=256, threshold=10, see repetition/README.md)
  are excluded before pairing.  Problems with fewer than two surviving
  responses are skipped for that condition.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

L_VALUES = (256, 512, 1024, 2048, 4096, 8192, 16384, None)  # None means "Full response"
L_LABELS = {
    256: "256",
    512: "512",
    1024: "1024",
    2048: "2048",
    4096: "4096",
    8192: "8192",
    16384: "16384",
    None: "full",
}
N_GRAM = 4
CONDITIONS = ("all", "no-repetition")
# Generation cap of the eval harness (the "MNT31744" suffix in the filenames).
# "Full" therefore mixes naturally-terminated with truncated responses, so the
# cap-hit rate is reported separately in results/length_stats.csv.
CAP_TOKENS = 31744

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER = Path(
    "/Users/fanghaotian/.cache/huggingface/hub/models--llamafactory--tiny-random-qwen3/"
    "snapshots/81d6f5f5e05ed53ea8a1d19431266a486e46bbd8/tokenizer.json"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rep_mod = load_module("jaccard_repetition_detector", REPO_ROOT / "repetition" / "analyze_and_plot.py")
window_ratios = rep_mod.window_ratios  # same detector constants as repetition/README.md
THRESHOLD = rep_mod.THRESHOLD

_TOKENIZER = None


def get_tokenizer(path: str):
    global _TOKENIZER
    if _TOKENIZER is None:
        from tokenizers import Tokenizer

        _TOKENIZER = Tokenizer.from_file(path)
    return _TOKENIZER


def ngram_keys(token_ids: list[int]) -> list[tuple[int, ...]]:
    """Position-ordered 4-grams; ``grams[:L - 3]`` are exactly the grams of tokens[:L]."""
    if len(token_ids) < N_GRAM:
        return []
    return list(zip(token_ids, token_ids[1:], token_ids[2:], token_ids[3:], strict=False))


def gram_sets_per_L(grams: list[tuple[int, ...]]) -> dict[int | None, frozenset]:
    return {L: frozenset(grams if L is None else grams[: max(0, L - (N_GRAM - 1))]) for L in L_VALUES}


def jaccard_distance(set_i: frozenset, set_j: frozenset) -> float:
    if not set_i and not set_j:
        return 0.0  # two ultra-short responses are treated as identical
    return 1.0 - len(set_i & set_j) / len(set_i | set_j)


def mean_pairwise_distance(sets: list[frozenset]) -> tuple[float | None, int]:
    """Mean Jaccard distance over all i<j pairs; (None, 0) when fewer than 2 responses."""
    n = len(sets)
    if n < 2:
        return None, 0
    total = 0.0
    pairs = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            total += jaccard_distance(sets[i], sets[j])
            pairs += 1
    return total / pairs, pairs


def discover_combos(eval_root: Path, modes: tuple[str, ...]) -> list[tuple[str, Path, Path]]:
    combos = []
    for mode in modes:
        mode_root = eval_root / mode
        for model_root in sorted(p for p in mode_root.iterdir() if p.is_dir()):
            for bench_path in sorted(model_root.glob("*.jsonl")):
                combos.append((mode, model_root, bench_path))
    return combos


def process_combo(task: tuple) -> list[dict]:
    mode, model_root, bench_path, tokenizer_path = task
    tokenizer = get_tokenizer(tokenizer_path)

    responses: list[str] = []
    problems: dict[str, list[int]] = defaultdict(list)
    with bench_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            problems[str(row.get("example_id"))].append(len(responses))
            responses.append(str(row.get("response", "")))

    encoded = tokenizer.encode_batch(responses) if responses else []
    token_ids = [enc.ids for enc in encoded]

    # Repetition flags come from the shared zlib window detector (text-based).
    flags = [bool((ratios >= THRESHOLD).any()) for _, ratios in (window_ratios(text) for text in responses)]

    bench = bench_path.name.split("_t", 1)[0]
    rows: list[dict] = []
    n_responses = len(responses)
    repetition_rate = sum(flags) / n_responses if n_responses else 0.0

    lengths = sorted(len(ids) for ids in token_ids)
    n = len(lengths)

    def pct(q: float) -> int:
        return lengths[min(n - 1, int(q * n))] if n else 0

    length_stats = {
        "mode": mode,
        "model": model_root.name,
        "bench": bench,
        "n_responses": n,
        "len_mean": f"{sum(lengths) / n:.1f}" if n else "",
        "len_p50": pct(0.50),
        "len_p75": pct(0.75),
        "len_p90": pct(0.90),
        "len_p99": pct(0.99),
        "len_max": lengths[-1] if n else 0,
        "cap_hit_rate": f"{sum(1 for v in lengths if v >= CAP_TOKENS) / n:.4f}" if n else "",
    }

    per_L = {cond: defaultdict(list) for cond in CONDITIONS}
    used_problems = {cond: 0 for cond in CONDITIONS}
    total_problems = 0

    for example_id in sorted(problems, key=lambda x: (len(x), x)):
        indices = problems[example_id]
        total_problems += 1
        per_response_sets = [gram_sets_per_L(ngram_keys(token_ids[i])) for i in indices]
        for cond in CONDITIONS:
            if cond == "no-repetition":
                survivors = [
                    sets for sets, flag in zip(per_response_sets, (flags[i] for i in indices), strict=True) if not flag
                ]
            else:
                survivors = per_response_sets
            if len(survivors) < 2:
                continue
            used_problems[cond] += 1
            for L in L_VALUES:
                per_L[cond][L].append([sets[L] for sets in survivors])

    for cond in CONDITIONS:
        for L in L_VALUES:
            scores = []
            pair_counts = []
            for sets in per_L[cond][L]:
                score, pairs = mean_pairwise_distance(sets)
                if score is not None:
                    scores.append(score)
                    pair_counts.append(pairs)
            rows.append(
                {
                    "mode": mode,
                    "model": model_root.name,
                    "bench": bench,
                    "condition": cond,
                    "L": L_LABELS[L],
                    "D_J": f"{sum(scores) / len(scores):.6f}" if scores else "",
                    "n_problems": total_problems,
                    "n_problems_used": used_problems[cond],
                    "n_responses": len(responses),
                    "n_pairs_mean": f"{sum(pair_counts) / len(pair_counts):.2f}" if pair_counts else "0",
                    "repetition_rate": f"{repetition_rate:.4f}",
                }
            )
    return rows, length_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root", type=Path, default=Path("/Users/fanghaotian/Desktop/src/OPD/justrl_eval_outputs")
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--tokenizer", type=str, default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--modes", nargs="+", default=["no-thinking", "thinking"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    results_dir = args.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    combos = discover_combos(args.eval_root, tuple(args.modes))
    print(f"Computing D_J for {len(combos)} (mode, model, bench) combos with {args.workers} workers")

    all_rows: list[dict] = []
    all_length_stats: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_combo, (mode, model, bench, args.tokenizer)): f"{mode}/{model.name}/{bench.name}"
            for mode, model, bench in combos
        }
        for future in as_completed(futures):
            label = futures[future]
            rows, length_stats = future.result()
            all_rows.extend(rows)
            all_length_stats.append(length_stats)
            print(f"  done: {label} ({len(rows)} rows)")

    fields = [
        "mode",
        "model",
        "bench",
        "condition",
        "L",
        "D_J",
        "n_problems",
        "n_problems_used",
        "n_responses",
        "n_pairs_mean",
        "repetition_rate",
    ]
    out_path = results_dir / "jaccard_diversity.csv"
    with out_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows to {out_path}")

    length_fields = [
        "mode",
        "model",
        "bench",
        "n_responses",
        "len_mean",
        "len_p50",
        "len_p75",
        "len_p90",
        "len_p99",
        "len_max",
        "cap_hit_rate",
    ]
    stats_path = results_dir / "length_stats.csv"
    with stats_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=length_fields)
        writer.writeheader()
        writer.writerows(sorted(all_length_stats, key=lambda r: (r["mode"], r["model"], r["bench"])))
    print(f"Wrote {len(all_length_stats)} length stats rows to {stats_path}")


if __name__ == "__main__":
    main()
