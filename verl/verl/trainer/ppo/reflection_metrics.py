# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Lexical reflection metrics computed from decoded response texts.

OPD-style RL degeneration selectively erases self-reflection markers ("wait",
"hmm", "alternatively", ...) before the zlib repetition flags fire, so their
rates act as an early-warning signal alongside ``repetition_metrics``.  The
panel below is a curated subset of the offline inventory in
``reflection/lexical_count.py``: the three high-precision categories and one
word per concept.  The noisy transition fillers ("so", "now", "first") stay
offline-only, while "but" is included deliberately — ordinary narration uses
it heavily, but its collapse is part of the observed degeneration signature.

Every response contributes rates normalized per 1,000 whitespace-delimited
words plus binary presence flags, so the batch/prompt aggregates are macro
averages directly comparable with the offline per-rollout statistics even
when response lengths drift during training.  One deliberate divergence: the
offline analysis excludes repetition-flagged rollouts before counting, the
raw series here do not (a degenerate response carries no markers, so their
inclusion only steepens the observed collapse).  ``compute_grouped_reflection_metrics``
therefore additionally emits ``*_unflagged`` rate variants over non-flagged
samples — those are the series directly comparable with the offline numbers.

Per-response values are scalars so they aggregate with the existing metric
pipelines: mean over a rollout batch for training, and the per-prompt group
channel (``compute_grouped_reflection_metrics``) for validation — the latter
deliberately bypasses the per-sample pass@k / best@N machinery, which only
carries signal for {0,1}-success-valued metrics like acc.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Sequence

import numpy as np

LEXICAL_LIST: dict[str, list[str]] = {
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
        "reconsider",
        "let me reconsider",
        "rethinking",
    ],
    "branch": [
        "but",
        "however",
        "alternatively",
        "alternative",
        "another approach",
        "another way",
        "instead",
        "rather",
        "on the other hand",
        "or perhaps",
        "maybe",
        "perhaps",
        "but then",
        "but wait",
    ],
    "verification": [
        "verify",
        "check",
        "double-check",
        "double check",
        "recheck",
        "recalculate",
        "confirm",
        "validate",
        "make sure",
        "let me check",
        "ensure",
        "cross-check",
        "cross check",
    ],
}

# Words whose individual trajectories must be visible in the panel; the
# import-time checks below guarantee they are all covered.
REQUIRED_PANEL_WORDS = frozenset(
    [
        "wait",
        "alternatively",
        "hold on",
        "another thought",
        "verify",
        "think again",
        "but",
        "however",
        "alternative",
        "check",
        "double-check",
        "oh",
        "hmm",
    ]
)

ALL_KEYWORDS = sorted(
    {kw.lower() for words in LEXICAL_LIST.values() for kw in words},
    key=lambda word: (-len(word), word),
)
KEYWORD_CATEGORY = {kw: category for category, words in LEXICAL_LIST.items() for kw in words}


def _word_key(word: str) -> str:
    """Metric-name-safe panel key.

    Spaces collapse to underscores; hyphens are kept so "double-check" and
    "double check" remain distinct series instead of colliding.
    """
    return word.replace(" ", "_")


WORD_KEYS = {word: _word_key(word) for word in ALL_KEYWORDS}

if len(set(WORD_KEYS.values())) != len(WORD_KEYS):
    raise ValueError("reflection panel words collide after metric-name sanitization")
if not REQUIRED_PANEL_WORDS <= set(ALL_KEYWORDS):
    raise ValueError(f"reflection panel is missing required words: {sorted(REQUIRED_PANEL_WORDS - set(ALL_KEYWORDS))}")

# Boundary-aware matching avoids counting "check" inside "checker" or "wait"
# inside "await" while allowing punctuation around the phrase; alternation is
# longest-first, so "but wait" is counted as one phrase rather than "but"
# plus "wait".  Identical to the offline matcher in reflection/lexical_count.py.
KEYWORD_RE = re.compile(r"(?<!\w)(" + "|".join(re.escape(kw) for kw in ALL_KEYWORDS) + r")(?!\w)", re.IGNORECASE)


def count_keywords(text: str) -> tuple[Counter[str], Counter[str]]:
    """Per-word and per-category occurrence counts of the panel in ``text``."""
    counts: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    for match in KEYWORD_RE.finditer(text):
        keyword = match.group(1).lower()
        counts[keyword] += 1
        categories[KEYWORD_CATEGORY[keyword]] += 1
    return counts, categories


def _zero_metrics() -> dict[str, float]:
    """Complete metric key set at zero, for empty (aborted) responses.

    Every response must carry the same keys: the downstream ``analyze_*``
    helpers take the key list from the first response of the batch.
    """
    metrics: dict[str, float] = {}
    for category in LEXICAL_LIST:
        metrics[f"reflection/{category}/rate_per_1k"] = 0.0
        metrics[f"reflection/{category}/presence"] = 0.0
    for word in ALL_KEYWORDS:
        key = WORD_KEYS[word]
        metrics[f"reflection/word/{key}/rate_per_1k"] = 0.0
        metrics[f"reflection/word/{key}/presence"] = 0.0
    return metrics


def analyze_response(text: str) -> dict[str, float]:
    """Per-response scalar reflection metrics for one decoded response.

    - ``reflection/<category>/rate_per_1k``: category occurrences per 1,000
      whitespace-delimited words of the response
    - ``reflection/<category>/presence``: 1.0 when the category occurs at
      least once
    - ``reflection/word/<word>/...``: the same pair per panel word

    An empty (aborted) response yields all zeros so it can never inflate a
    rate; callers exclude it from the denominators anyway.
    """
    if not text:
        return _zero_metrics()

    n_words = max(len(text.split()), 1)
    counts, categories = count_keywords(text)
    metrics = _zero_metrics()
    for category, count in categories.items():
        metrics[f"reflection/{category}/rate_per_1k"] = 1000.0 * count / n_words
        metrics[f"reflection/{category}/presence"] = float(count > 0)
    for word, count in counts.items():
        key = WORD_KEYS[word]
        metrics[f"reflection/word/{key}/rate_per_1k"] = 1000.0 * count / n_words
        metrics[f"reflection/word/{key}/presence"] = float(count > 0)
    return metrics


def analyze_reflection_responses(texts: list[str]) -> dict[str, list[float]]:
    """Per-response metric lists aligned with the input order.

    The list format plugs directly into validation aggregation, where every
    sample must carry a value (aborted responses are analyzed as empty text).
    """
    per_response = [analyze_response(text) for text in texts]
    keys = list(per_response[0]) if per_response else []
    return {key: [item[key] for item in per_response] for key in keys}


def compute_reflection_metrics(texts: list[str]) -> dict[str, float]:
    """Mean of every per-response reflection metric over a rollout batch.

    Callers should exclude aborted responses beforehand; empty texts are
    analyzed as all-zero but would dilute the means.  The mean of per-response
    rates is a macro average, matching the per-rollout aggregates of
    ``reflection/lexical_count.py`` rather than its corpus-level totals (that
    script excludes repetition-flagged rollouts before counting; these raw
    means include them — the unflagged variants live in
    :func:`compute_grouped_reflection_metrics`).
    """
    if not texts:
        return {}
    per_response = analyze_reflection_responses(texts)
    return {key: float(np.mean(values)) for key, values in per_response.items()}


def compute_grouped_reflection_metrics(
    per_response: dict[str, list[float]],
    uids: Sequence,
    data_sources: Sequence,
    valid: Sequence[bool] | None = None,
    repetition_flags: Sequence[bool] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Group per-response reflection values by uid for the validation channel.

    Returns ``{data_source: {metric: [one value per group]}}`` where each
    value is the mean of the per-response scalars within one prompt's
    ``n`` samples; callers aggregate by averaging over groups.  ``uids`` and
    ``data_sources`` are required and aligned with the per-response values.
    ``valid`` optionally masks out aborted responses (empty texts) so they
    neither dilute the rates nor count as clean samples, matching the
    train-side denominator.

    ``repetition_flags`` optionally carries per-response repetition flags
    (see :mod:`verl.trainer.ppo.repetition_metrics`).  When given, every
    ``*/rate_per_1k`` series gains an ``*_unflagged`` variant computed over
    the group's non-flagged samples — the counterpart directly comparable
    with the offline analysis, which excludes flagged rollouts before
    counting.  A group whose every sample is flagged contributes nothing to
    the unflagged means, so those series average over the groups that can
    contribute; ``reflection/unflagged_groups`` records the share of groups
    that do.

    Adds ``reflection/<category>/all_absent`` — the share of prompt groups
    whose every sample contains zero markers of the category — the reflection
    analogue of ``repetition/all_flagged``.
    """
    if not per_response:
        return {}
    n = len(next(iter(per_response.values())))
    for name, seq in (("uids", uids), ("data_sources", data_sources)):
        if seq is None:
            raise ValueError(f"{name} is required to group reflection values by prompt and data_source.")
        if len(seq) != n:
            raise ValueError(f"{name} has {len(seq)} entries for {n} per-response values.")
    if valid is not None and len(valid) != n:
        raise ValueError(f"valid has {len(valid)} entries for {n} per-response values.")
    if repetition_flags is not None and len(repetition_flags) != n:
        raise ValueError(f"repetition_flags has {len(repetition_flags)} entries for {n} per-response values.")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, uid in enumerate(uids):
        if valid is None or valid[index]:
            groups[str(uid)].append(index)

    results: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for indices in groups.values():
        data_source = str(data_sources[indices[0]])
        for key, values in per_response.items():
            results[data_source][key].append(float(np.mean([values[i] for i in indices])))
        for category in LEXICAL_LIST:
            presence = per_response[f"reflection/{category}/presence"]
            results[data_source][f"reflection/{category}/all_absent"].append(
                float(all(presence[i] == 0.0 for i in indices))
            )
        if repetition_flags is not None:
            unflagged = [i for i in indices if not repetition_flags[i]]
            results[data_source]["reflection/unflagged_groups"].append(1.0 if unflagged else 0.0)
            if unflagged:
                for key, values in per_response.items():
                    if key.endswith("/rate_per_1k"):
                        results[data_source][f"{key}_unflagged"].append(float(np.mean([values[i] for i in unflagged])))
    return results
