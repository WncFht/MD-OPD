#!/usr/bin/env python3
"""Aggregate eval.sh val dumps into report.json + summary.csv.

    python3 eval/aggregate_report.py eval/<arm>/<q>/<seed>/step_300 --data-root data/
    python3 eval/aggregate_report.py root_a root_b --out tables/
    python3 eval/aggregate_report.py <root> --rescore --trainer-log logs/eval-math-<tag>.log

Each EVAL_ROOT holds one directory per eval.sh pass, plus the offline Multi-IF
output when it was run:

    <root>/<bench>/dump/<step>.jsonl   per-sample rows: input/output/gts/score/
                                      acc/data_source/uid + verifier extras
    <root>/<bench>/eval_config.json    resolved pass parameters (recipe/eval.sh)
    <root>/multiif/report.json         eval/run_if_eval.py output (optional)

The dump jsonl is the only per-sample source of truth. Metrics replicate
verl.trainer.ppo.metric_utils.process_validation_metrics: rows group by
(data_source, uid), per-prompt statistics are averaged over prompts. pass@k is
the unbiased Chen et al. 2021 estimator -- the same formula as
metric_utils._pass_at_k, re-implemented on stdlib so this file needs no
torch/numpy. pandas is imported lazily, only to count parquet rows for the
coverage field (filter_overlong_prompts silently drops over-long eval rows --
coverage is the only line of defence).

--rescore re-runs opd_mixed_verify.reward_func on every dumped (output, gts)
pair and reports the rescored mean next to the dumped one; it pulls the real
verl import chain (torch), so it only runs on a box with the train env.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Summary column order follows plots/runs/mdxt04lm__eval.csv (step, per-source
# mean@N columns), extended with the macro aggregates W5 consumes.
SUMMARY_COLUMNS = [
    "step",
    "MATH500",
    "AIME25",
    "AMC23",
    "math_macro",
    "LCBv6",
    "BFCL",
    "MultiIF_turn3",
    "MultiIF_3turn_mean",
]
SOURCE_TO_COLUMN = {"MATH-500": "MATH500", "livecodebench": "LCBv6"}
MATH_SOURCES = ("MATH-500", "AMC23", "AIME25")

# pass@k grid: powers of two up to n plus n itself, matching the trainer's
# gen_ns list with k=1 added (the eval protocol names pass@1 explicitly).
def ks_for(n: int) -> list[int]:
    """pass@k grid for a group of n samples: {1,2,4,...} up to and including n."""
    return sorted({1, n} | {1 << i for i in range(n.bit_length()) if (1 << i) <= n})


def pass_at_k(n: int, num_correct: int, k: int) -> float:
    """Unbiased pass@k: 1 - prod(1 - k/j) for j in (n-c+1 .. n). Chen et al. 2021."""
    if n - num_correct < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / j for j in range(n - num_correct + 1, n + 1))


def prompt_metrics(values: list[float]) -> dict[str, float]:
    """One prompt's n samples -> {mean@n, std@n, pass@k}."""
    n = len(values)
    out = {f"mean@{n}": statistics.fmean(values)}
    if n > 1:
        out[f"std@{n}"] = statistics.pstdev(values)  # population std, np.std semantics
        num_correct = sum(1 for v in values if v == 1.0)
        for k in ks_for(n):
            out[f"pass@{k}"] = pass_at_k(n, num_correct, k)
    return out


def numeric_vars(rows: list[dict]) -> list[str]:
    """Columns worth aggregating: present and numeric on the first row.

    The dump writer emits a fixed key set per file, so row 0's types identify
    the numeric columns; strings (input/output/data_source/uid/pred) and
    `step` are skipped. A later row carrying None under a numeric key fails
    loudly at float() -- a ragged dump is a bug to surface, not to skip.
    """
    return [k for k, v in rows[0].items() if k != "step" and isinstance(v, (int, float))]


def source_metrics(rows: list[dict]) -> dict:
    """Dump rows of one data_source -> {n_prompts, n_samples, n_per_prompt, metrics}."""
    uid2rows: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        uid2rows[str(r.get("uid", r["input"]))].append(r)
    group_sizes = {len(rs) for rs in uid2rows.values()}
    if len(group_sizes) != 1:
        raise SystemExit(
            f"ragged dump: per-prompt sample counts differ {sorted(group_sizes)}; "
            "the dump was truncated or repeats were uneven"
        )
    n = group_sizes.pop()

    metrics: dict[str, dict[str, float]] = {}
    for var in numeric_vars(rows):
        per_metric: dict[str, list[float]] = defaultdict(list)
        for rs in uid2rows.values():
            for name, val in prompt_metrics([float(r[var]) for r in rs]).items():
                per_metric[name].append(val)
        metrics[var] = {name: statistics.fmean(vals) for name, vals in per_metric.items()}
    return {"n_prompts": len(uid2rows), "n_samples": len(rows), "n_per_prompt": n, "metrics": metrics}


def parquet_source_counts(val_files: list[str], data_root: Path) -> dict[str, int]:
    """Expected prompt count per data_source, read from the val parquets."""
    import pandas as pd

    counts: dict[str, int] = defaultdict(int)
    for name in val_files:
        path = Path(name)
        if not path.is_absolute():
            path = data_root / path
        for source, cnt in pd.read_parquet(path, columns=["data_source"])["data_source"].value_counts().items():
            counts[source] += int(cnt)
    return dict(counts)


def read_trainer_metrics(path: Path) -> dict[str, float]:
    """Parse the `Initial validation metrics: {...}` dict a val_only run prints."""
    text = path.read_text()
    marker = "Initial validation metrics:"
    start = text.find(marker)
    if start < 0:
        return {}
    brace = text.find("{", start)
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return ast.literal_eval(text[brace : i + 1])
                except (ValueError, SyntaxError):
                    print(f"[warn] {path}: could not parse validation metrics dict", file=sys.stderr)
                    return {}
    return {}


def rescore_rows(rows: list[dict]) -> list[dict]:
    """Re-run the mixed verifier per row; needs the full verl env (torch)."""
    sys.path.insert(0, str(REPO / "verl"))
    from verl.utils.reward_score.opd_mixed_verify import reward_func

    out = []
    for r in rows:
        res = reward_func(r["data_source"], r["output"], r["gts"])
        out.append(res if isinstance(res, dict) else {"score": float(res), "acc": float(res)})
    return out


def core_var(metrics: dict[str, dict]) -> str:
    """The headline variable, matching _val_metrics_update's acc-then-reward rule."""
    return "acc" if "acc" in metrics else "reward"


def aggregate_bench(bench_dir: Path, data_root: Path, rescore: bool) -> dict:
    """One <root>/<bench> directory -> per-source entries + macro + sampling."""
    config_path = bench_dir / "eval_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    expected = {}
    if config.get("val_files"):
        try:
            expected = parquet_source_counts(config["val_files"], data_root)
        except ImportError:
            print("[warn] pandas unavailable -- coverage fields will be null", file=sys.stderr)

    entries = []
    for dump in sorted((bench_dir / "dump").glob("*.jsonl")):
        rows = [json.loads(line) for line in dump.open()]
        if not rows:
            continue
        if "data_source" not in rows[0]:
            raise SystemExit(
                f"{dump} has no data_source column -- the dump predates the "
                "validation_data_dir patch in ray_trainer; rerun the eval pass"
            )
        source2rows: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            source2rows[r["data_source"]].append(r)

        for source, srows in sorted(source2rows.items()):
            entry = source_metrics(srows)
            entry.update(
                benchmark=config.get("benchmark", bench_dir.name),
                data_source=source,
                dump_step=int(dump.stem) if dump.stem.isdigit() else dump.stem,
            )
            want = expected.get(source)
            entry["coverage"] = entry["n_prompts"] / want if want else None
            if entry["coverage"] is not None and entry["coverage"] != 1.0:
                print(
                    f"[warn] {source}: coverage={entry['coverage']:.3f} "
                    f"({entry['n_prompts']}/{want} prompts) -- <1.0 means filter_overlong_prompts "
                    "dropped eval rows (raise MAX_PROMPT_LENGTH); >1.0 means duplicate dump rows",
                    file=sys.stderr,
                )
            if rescore:
                rescored = rescore_rows(srows)
                var = core_var(entry["metrics"])
                dumped = entry["metrics"][var][f"mean@{entry['n_per_prompt']}"]
                fresh = statistics.fmean(float(r.get(var, r.get("score", 0.0))) for r in rescored)
                entry["rescore_mean"] = fresh
                if abs(fresh - dumped) > 1e-6:
                    print(
                        f"[warn] {source}: rescored mean {fresh:.6f} != dumped {dumped:.6f}",
                        file=sys.stderr,
                    )
            entries.append(entry)
    return {"sampling": config.get("sampling"), "max_prompt_length": config.get("max_prompt_length"),
            "val_files": config.get("val_files"), "entries": entries}


def macro_mean(entries: list[dict], sources: tuple[str, ...] | None = None) -> float | None:
    """Mean of per-source acc mean@N -- the protocol's macro average."""
    vals = []
    for e in entries:
        if sources and e["data_source"] not in sources:
            continue
        var = core_var(e["metrics"])
        vals.append(e["metrics"][var][f"mean@{e['n_per_prompt']}"])
    return statistics.fmean(vals) if vals else None


def infer_step(root: Path, config: dict) -> str:
    """step_N dir name, else global_step_N inside the ckpt path, else the dump file."""
    if root.name.startswith("step_") or root.name == "final":
        return root.name.removeprefix("step_")
    ckpt = config.get("ckpt", "")
    for part in reversed(Path(ckpt).parts):
        if part.startswith("global_step_"):
            return part.removeprefix("global_step_")
    return "0"


def aggregate_root(root: Path, args) -> dict:
    """One EVAL_OUT tree -> the run record consumed by report.json/summary.csv."""
    run: dict = {"eval_out": str(root), "benchmarks": {}}
    for bench_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (bench_dir / "dump").is_dir():
            continue
        run["benchmarks"][bench_dir.name] = aggregate_bench(bench_dir, args.data_root, args.rescore)

    # Model/step identity from the first bench config found.
    config = {}
    for bench_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cfg_path = bench_dir / "eval_config.json"
        if cfg_path.exists():
            config = json.loads(cfg_path.read_text())
            break
    run["model"] = config.get("ckpt")
    run["ckpt_step"] = infer_step(root, config)

    for name, bench in run["benchmarks"].items():
        sources = MATH_SOURCES if name == "math" else None
        bench["macro_mean"] = macro_mean(bench["entries"], sources)

    multiif_report = root / "multiif" / "report.json"
    if multiif_report.exists():
        run["multiif"] = json.loads(multiif_report.read_text())
    return run


def summary_row(run: dict) -> dict[str, str]:
    """One run -> one mdxt04lm-style wide-table row."""
    row = {c: "" for c in SUMMARY_COLUMNS}
    row["step"] = str(run["ckpt_step"])
    for name, bench in run["benchmarks"].items():
        for e in bench["entries"]:
            var = core_var(e["metrics"])
            col = SOURCE_TO_COLUMN.get(e["data_source"], e["data_source"])
            mean_key = f"mean@{e['n_per_prompt']}"
            if col in row:
                row[col] = f"{e['metrics'][var][mean_key]:.6f}"
            else:
                print(
                    f"[warn] data_source {e['data_source']!r} has no summary column -- "
                    "dropped from summary.csv (report.json keeps it)",
                    file=sys.stderr,
                )
        if bench["macro_mean"] is not None:
            if name == "math":
                row["math_macro"] = f"{bench['macro_mean']:.6f}"
            elif name == "bfcl":
                row["BFCL"] = f"{bench['macro_mean']:.6f}"
    headline = run.get("multiif", {}).get("headline", {})
    if "8lang_turn3" in headline:
        row["MultiIF_turn3"] = f"{headline['8lang_turn3']:.4f}"
        row["MultiIF_3turn_mean"] = f"{headline['8lang_3turn_mean']:.4f}"
    return row


def compare_trainer_log(run: dict, log: Path) -> None:
    """Cross-check dump-derived metrics against the trainer's val-core/val-aux keys."""
    truth = read_trainer_metrics(log)
    if not truth:
        print(f"[warn] {log}: no 'Initial validation metrics' line found", file=sys.stderr)
        return
    for bench in run["benchmarks"].values():
        for e in bench["entries"]:
            for var, metric2val in e["metrics"].items():
                for name, val in metric2val.items():
                    for sec in ("val-core", "val-aux"):
                        key = f"{sec}/{e['data_source']}/{var}/{name}"
                        if key in truth and abs(truth[key] - val) > 1e-6:
                            print(
                                f"[warn] {key}: dump {val:.6f} != trainer {truth[key]:.6f}",
                                file=sys.stderr,
                            )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", type=Path, help="EVAL_OUT dirs (one summary row each)")
    p.add_argument("--data-root", type=Path, default=Path("."), help="base for relative val_files paths")
    p.add_argument("--out", type=Path, help="where report.json/summary.csv go (default: the single root, else cwd)")
    p.add_argument("--rescore", action="store_true", help="re-run the verifier on dumped outputs (needs torch env)")
    p.add_argument("--trainer-log", type=Path, action="append", default=[],
                   help="logs/eval-<bench>-<tag>.log to cross-check val-core keys against; repeatable")
    p.add_argument("--expect-full-coverage", action="store_true",
                   help="exit 1 if any data_source lost prompts to filter_overlong_prompts")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    runs = [aggregate_root(root, args) for root in args.roots]

    for log in args.trainer_log:
        for run in runs:
            compare_trainer_log(run, log)

    under_covered = [
        e["data_source"]
        for run in runs
        for bench in run["benchmarks"].values()
        for e in bench["entries"]
        if e["coverage"] is not None and e["coverage"] < 1.0
    ]

    out_dir = args.out or (args.roots[0] if len(args.roots) == 1 else Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps({"runs": runs}, indent=2, ensure_ascii=False))
    with (out_dir / "summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for run in runs:
            writer.writerow(summary_row(run))
    print(f"wrote {out_dir}/report.json and {out_dir}/summary.csv")

    if under_covered:
        print(f"[error] coverage < 1.0 on: {sorted(set(under_covered))}", file=sys.stderr)
        return 1 if args.expect_full_coverage else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
