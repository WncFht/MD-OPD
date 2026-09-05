# Reflection lexical statistics

This directory contains a reproducible lexical reflection counter for the OPD
evaluation rollouts. It scans both `no-thinking` and `thinking`, removes
rollouts flagged by the existing local zlib repetition detector, and reports
per-rollout counts plus per-model top-k keyword frequencies.

## Run

```bash
python3 reflection/lexical_count.py \
  --eval-root /Users/fanghaotian/Desktop/src/OPD/justrl_eval_outputs \
  --output-dir reflection/results \
  --top-k 30
```

Use `--exclude-checkpoints` to skip directories named like
`global_step_100`. By default they are retained and treated as separate model
labels.

## Outputs

- `per_rollout.csv`: one row per non-repetition rollout, with total lexical
  count, count by category, and a length-normalized rate per 1,000 words.
- `model_summary.csv`: total/repetition/kept rollout counts and model-level
  averages.
- `top_keywords.csv`: top-k keyword counts per `(mode, model)`, plus rate per
  1,000 words.
- `lexical_list.json`: the exact broad list used for this run.
- `metadata.json`: detector and matching configuration.

The list is intentionally broad and split into `core`, `branch`,
`verification`, and `transition`. For a high-precision primary metric, use
`count_core` (and optionally `count_verification`); `count_transition` is a
broader auxiliary statistic because words such as `so`, `now`, and `first` can
occur in ordinary exposition.
