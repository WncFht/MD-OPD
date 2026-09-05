#!/usr/bin/env python3
"""Count lexical reflection markers in OPD rollout JSONL files.

The script mirrors the repository's repetition detector (overlapping zlib
windows) and excludes flagged rollouts before computing lexical statistics.
It writes per-rollout counts, per-model summaries, and top-k keyword tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path

# Broad candidate inventory assembled from ARLCP, NoWait, CGRS, CyclicReflex,
# and common self-correction/discourse phrases. Categories are reported
# separately so broad markers do not have to be treated as high-precision ones.
LEXICAL_LIST = {
    "core": [
        "wait",
        "hmm",
        "hold on",
        "oh",
        "ah",
        "actually",
        "oops",
        "on second thought",
        "another thought",
        "think again",
        "let me rethink",
        "let me reconsider",
        "reconsider",
        "rethinking",
        "i should rethink",
        "i need to rethink",
    ],
    "branch": [
        "alternatively",
        "alternative",
        "another approach",
        "another way",
        "different approach",
        "different way",
        "instead",
        "rather",
        "on the other hand",
        "or perhaps",
        "maybe",
        "perhaps",
        "however",
        "but then",
        "but actually",
        "but wait",
    ],
    "verification": [
        "verify",
        "verification",
        "check",
        "double-check",
        "double check",
        "recheck",
        "recalculate",
        "confirm",
        "confirmation",
        "validate",
        "make sure",
        "let me check",
        "let's check",
        "lets check",
        "let me verify",
        "let's verify",
        "lets verify",
        "need to check",
        "check again",
        "verify again",
        "ensure",
        "cross-check",
        "cross check",
    ],
    "transition": [
        "again",
        "now",
        "first",
        "second",
        "next",
        "so",
        "therefore",
        "thus",
        "hence",
        "well",
        "let's see",
        "lets see",
        "let me see",
        "i need to",
        "we need to",
        "this suggests",
        "that means",
    ],
}

ALL_KEYWORDS = sorted(
    {kw.lower() for values in LEXICAL_LIST.values() for kw in values},
    key=lambda value: (-len(value), value),
)
KEYWORD_CATEGORY = {kw.lower(): category for category, values in LEXICAL_LIST.items() for kw in values}

# Boundary-aware matching avoids counting "check" inside "checker" while
# allowing punctuation and leading/trailing whitespace around phrases.
KEYWORD_RE = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(kw) for kw in ALL_KEYWORDS) + r")(?!\w)",
    re.IGNORECASE,
)


def compression_ratio(text: str, window_size: int = 1024) -> float:
    raw = text.encode("utf-8")
    if not raw:
        return 0.0
    return len(raw) / max(len(zlib.compress(raw, 9)), 1)


def is_repetition(
    text: str,
    window_size: int = 1024,
    stride: int = 256,
    threshold: float = 10.0,
) -> tuple[bool, float | None, int | None]:
    """Return flag, maximum local compression ratio, and first hit position."""
    if not text:
        return False, 0.0, None
    if len(text) <= window_size:
        starts = [0]
    else:
        starts = list(range(0, len(text) - window_size + 1, stride))
        last = len(text) - window_size
        if starts[-1] != last:
            starts.append(last)
    ratios = [compression_ratio(text[start : start + window_size], window_size) for start in starts]
    hit_indices = [i for i, ratio in enumerate(ratios) if ratio >= threshold]
    return (
        bool(hit_indices),
        max(ratios),
        starts[hit_indices[0]] if hit_indices else None,
    )


def count_keywords(text: str) -> tuple[Counter[str], Counter[str]]:
    counts: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    for match in KEYWORD_RE.finditer(text):
        keyword = match.group(1).lower()
        counts[keyword] += 1
        categories[KEYWORD_CATEGORY[keyword]] += 1
    return counts, categories


def iter_rollouts(root: Path, exclude_checkpoints: bool):
    for mode in ("no-thinking", "thinking"):
        mode_root = root / mode
        if not mode_root.is_dir():
            continue
        for model_root in sorted(path for path in mode_root.iterdir() if path.is_dir()):
            if exclude_checkpoints and re.search(r"global_step_\d+", model_root.name):
                continue
            for path in sorted(model_root.glob("*.jsonl")):
                bench = path.name.split("_t", 1)[0]
                with path.open(encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream):
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        yield {
                            "mode": mode,
                            "model": model_root.name,
                            "bench": bench,
                            "source": str(path),
                            "example_id": str(row.get("example_id", line_number)),
                            "seed": str(row.get("seed", line_number)),
                            "response": str(row.get("response", "")),
                        }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reflection/results"))
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument(
        "--exclude-checkpoints",
        action="store_true",
        help="Skip model directories whose name contains global_step_N.",
    )
    parser.add_argument("--repetition-window", type=int, default=1024)
    parser.add_argument("--repetition-stride", type=int, default=256)
    parser.add_argument("--repetition-threshold", type=float, default=10.0)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    per_rollout: list[dict] = []
    aggregate: dict[tuple[str, str], dict] = defaultdict(
        lambda: {
            "total": 0,
            "kept": 0,
            "repetition": 0,
            "tokens": 0,
            "keywords": Counter(),
            "categories": Counter(),
        }
    )

    for item in iter_rollouts(args.eval_root, args.exclude_checkpoints):
        text = item.pop("response")
        repeated, max_ratio, first_position = is_repetition(
            text,
            args.repetition_window,
            args.repetition_stride,
            args.repetition_threshold,
        )
        key = (item["mode"], item["model"])
        stats = aggregate[key]
        stats["total"] += 1
        if repeated:
            stats["repetition"] += 1
            continue
        counts, categories = count_keywords(text)
        stats["kept"] += 1
        stats["tokens"] += len(text.split())
        stats["keywords"].update(counts)
        stats["categories"].update(categories)
        per_rollout.append(
            {
                **item,
                "response_chars": len(text),
                "word_count": len(text.split()),
                "repetition": False,
                "max_compression_ratio": round(max_ratio or 0.0, 6),
                "first_repetition_position": first_position if first_position is not None else "",
                "lexical_count": sum(counts.values()),
                "lexical_rate_per_1k_words": round(1000 * sum(counts.values()) / max(len(text.split()), 1), 6),
                **{f"count_{category}": categories[category] for category in LEXICAL_LIST},
            }
        )

    write_csv(output_dir / "per_rollout.csv", per_rollout)
    summary_rows = []
    top_rows = []
    for (mode, model), stats in sorted(aggregate.items()):
        kept = stats["kept"]
        summary_rows.append(
            {
                "mode": mode,
                "model": model,
                "rollouts_total": stats["total"],
                "rollouts_repetition": stats["repetition"],
                "rollouts_kept": kept,
                "repetition_rate": round(stats["repetition"] / max(stats["total"], 1), 6),
                "avg_lexical_count": round(
                    sum(r["lexical_count"] for r in per_rollout if r["mode"] == mode and r["model"] == model)
                    / max(kept, 1),
                    6,
                ),
                "avg_lexical_rate_per_1k_words": round(
                    sum(
                        r["lexical_rate_per_1k_words"] for r in per_rollout if r["mode"] == mode and r["model"] == model
                    )
                    / max(kept, 1),
                    6,
                ),
                **{f"count_{category}": stats["categories"][category] for category in LEXICAL_LIST},
            }
        )
        for keyword, count in stats["keywords"].most_common(args.top_k):
            top_rows.append(
                {
                    "mode": mode,
                    "model": model,
                    "keyword": keyword,
                    "category": KEYWORD_CATEGORY[keyword],
                    "count": count,
                    "rate_per_1k_words": round(1000 * count / max(stats["tokens"], 1), 6),
                }
            )

    write_csv(output_dir / "model_summary.csv", summary_rows)
    write_csv(output_dir / "top_keywords.csv", top_rows)
    (output_dir / "lexical_list.json").write_text(
        json.dumps(LEXICAL_LIST, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metadata = {
        "eval_root": str(args.eval_root),
        "top_k": args.top_k,
        "repetition_detector": {
            "window": args.repetition_window,
            "stride": args.repetition_stride,
            "threshold": args.repetition_threshold,
        },
        "matching": "case-insensitive, boundary-aware, longest phrase first",
        "excluded_rollouts": "any rollout with a local zlib compression ratio >= threshold",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"kept rollouts: {len(per_rollout)}")
    print(f"wrote: {output_dir / 'per_rollout.csv'}")
    print(f"wrote: {output_dir / 'model_summary.csv'}")
    print(f"wrote: {output_dir / 'top_keywords.csv'}")


if __name__ == "__main__":
    main()
