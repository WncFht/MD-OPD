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
"""Group-level diversity metrics over validation responses.

Both families are computed per prompt (uid) over the ``n`` sampled responses
and aggregated by averaging over prompts per data_source:

- Jaccard: pairwise token-level n-gram Jaccard distance between the responses
  of one prompt.  Prefix points truncate each response at a relative position
  of its own length (e.g. 0.5 = first half of each response), plus one
  absolute-token anchor and the full response.  Relative points stay
  meaningful when response lengths drift during training; the absolute anchor
  keeps a comparable reading of the shared opening (the regime where
  reasoning templates dominate).  Every point is reported for all usable
  responses and, when repetition flags are available, for the subset not
  flagged by :mod:`verl.trainer.ppo.repetition_metrics` — degenerate loops
  are mutually dissimilar and inflate pairwise distances.

- Answer: diversity of the extracted answers (the ``pred`` strings returned
  by the reward function): the valid ratio, the unique ratio over valid
  answers, and the majority-answer share.  This catches semantic collapse
  that lexical n-grams cannot (same solution reworded).

Per-prompt values are scalars; callers aggregate them directly instead of
routing through the per-sample validation pipeline, so no pseudo pass@k /
bootstrap statistics are produced for group constants.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable, Sequence

DEFAULT_N_GRAM = 4
DEFAULT_RELATIVE_POSITIONS = (0.25, 0.5, 0.75, 1.0)
DEFAULT_ABSOLUTE_PREFIX_TOKENS = 1024


def ngram_keys(token_ids: Sequence[int], n_gram: int) -> list[tuple[int, ...]]:
    """Position-ordered n-grams of a token sequence.

    ``keys[:k - n_gram + 1]`` are exactly the n-grams of ``token_ids[:k]``, so
    every prefix metric is a slice of one full-length construction.
    """
    if len(token_ids) < n_gram:
        return []
    return list(zip(*(token_ids[i:] for i in range(n_gram)), strict=False))


def _prefix_gram_set(grams: Sequence[tuple[int, ...]], prefix_tokens: int, n_gram: int) -> frozenset:
    """Gram set of ``token_ids[:prefix_tokens]`` from the full gram list."""
    if prefix_tokens <= 0:
        return frozenset()
    return frozenset(grams[: prefix_tokens - n_gram + 1])


def _mean_pairwise_jaccard_distance(gram_sets: list[frozenset]) -> float | None:
    """Mean ``1 - |∩| / |∪|`` over all i<j pairs.

    Pairs where either set is empty (response shorter than n_gram tokens) are
    skipped: a sub-n_gram response carries no diversity signal, and counting
    it as "maximally distant" would pollute the mean.  Returns None when no
    valid pair remains.
    """
    total = 0.0
    pairs = 0
    for i in range(len(gram_sets) - 1):
        for j in range(i + 1, len(gram_sets)):
            set_i, set_j = gram_sets[i], gram_sets[j]
            if not set_i or not set_j:
                continue
            union = len(set_i | set_j)
            if union == 0:
                continue
            total += 1.0 - len(set_i & set_j) / union
            pairs += 1
    return total / pairs if pairs else None


def _point_name(relative_position: float) -> str:
    return "full" if relative_position >= 1.0 else f"rel@{relative_position:g}"


def compute_group_jaccard_metrics(
    token_ids_per_response: list[Sequence[int]],
    *,
    n_gram: int = DEFAULT_N_GRAM,
    relative_positions: Sequence[float] = DEFAULT_RELATIVE_POSITIONS,
    absolute_prefix_tokens: int | None = DEFAULT_ABSOLUTE_PREFIX_TOKENS,
    repetition_flags: list[bool] | None = None,
) -> dict[str, float]:
    """Jaccard metric points for one prompt group.

    Metric names are ``all/<point>`` and, when ``repetition_flags`` is given,
    ``filtered/<point>``; a ``<condition>/n_groups`` entry (1.0 / 0.0) records
    whether the condition had at least two usable responses, so the mean over
    prompts reads as the usable-prompt fraction.  Usable means at least
    ``n_gram`` tokens; a point whose pairs are all skipped is omitted.
    """
    for alpha in relative_positions:
        if not 0 < alpha <= 1.0:
            raise ValueError(f"relative_positions must be in (0, 1], got {alpha}.")
    if absolute_prefix_tokens is not None and absolute_prefix_tokens <= 0:
        raise ValueError(f"absolute_prefix_tokens must be positive, got {absolute_prefix_tokens}.")

    usable = [i for i, token_ids in enumerate(token_ids_per_response) if len(token_ids) >= n_gram]
    metrics: dict[str, float] = {}
    if len(usable) < 2:
        return metrics

    grams = {i: ngram_keys(token_ids_per_response[i], n_gram) for i in usable}
    lengths = {i: len(token_ids_per_response[i]) for i in usable}

    points: list[tuple[str, dict[int, int]]] = []
    for alpha in relative_positions:
        ks = {i: lengths[i] if alpha >= 1.0 else int(alpha * lengths[i]) for i in usable}
        points.append((_point_name(alpha), ks))
    if absolute_prefix_tokens:
        ks = {i: min(absolute_prefix_tokens, lengths[i]) for i in usable}
        points.append((f"abs@{absolute_prefix_tokens:g}", ks))

    conditions: dict[str, list[int]] = {"all": usable}
    if repetition_flags is not None:
        conditions["filtered"] = [i for i in usable if not repetition_flags[i]]

    for condition, indices in conditions.items():
        metrics[f"{condition}/n_groups"] = 1.0 if len(indices) >= 2 else 0.0
        if len(indices) < 2:
            continue
        for name, ks in points:
            value = _mean_pairwise_jaccard_distance([_prefix_gram_set(grams[i], ks[i], n_gram) for i in indices])
            if value is not None:
                metrics[f"{condition}/{name}"] = value
    return metrics


def compute_group_answer_metrics(preds: Sequence) -> dict[str, float]:
    """Diversity of the extracted answers for one prompt group.

    ``pred`` values that are None or empty count as invalid (their share is
    ``answer/valid_ratio``); the remaining metrics use valid answers only.
    Answers are compared after whitespace stripping.
    """
    values = [str(pred).strip() if pred is not None else "" for pred in preds]
    if not values:
        return {}
    valid = [value for value in values if value]
    metrics = {"answer/valid_ratio": len(valid) / len(values)}
    if valid:
        counts = Counter(valid)
        metrics["answer/unique_ratio"] = len(counts) / len(valid)
        metrics["answer/top1_ratio"] = max(counts.values()) / len(valid)
    return metrics


def compute_group_diversity_metrics(
    texts: list[str],
    uids: Sequence,
    data_sources: Sequence,
    tokenize_fn: Callable[[list[str]], list[list[int]]],
    *,
    n_gram: int = DEFAULT_N_GRAM,
    relative_positions: Sequence[float] = DEFAULT_RELATIVE_POSITIONS,
    absolute_prefix_tokens: int | None = DEFAULT_ABSOLUTE_PREFIX_TOKENS,
    repetition_flags: list[bool] | None = None,
    preds: Sequence | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Group metrics for one validation batch.

    Responses are grouped by uid (one prompt = one group); groups with fewer
    than two responses are skipped.  Returns
    ``{data_source: {metric_name: [one value per usable group]}}``.  Callers
    aggregate by averaging each metric over the values.  ``preds`` (aligned
    with ``texts``) enables the answer family; pass None when the reward
    function does not expose extracted answers.
    """
    if repetition_flags is not None and len(repetition_flags) != len(texts):
        raise ValueError(f"repetition_flags has {len(repetition_flags)} entries for {len(texts)} texts.")
    if preds is not None and len(preds) != len(texts):
        raise ValueError(f"preds has {len(preds)} entries for {len(texts)} texts.")
    if not texts:
        return {}

    token_ids_per_response = tokenize_fn(texts)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, uid in enumerate(uids):
        groups[str(uid)].append(index)

    results: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for indices in groups.values():
        if len(indices) < 2:
            # Diversity is undefined for a single response; with val_kwargs.n=1
            # every group is skipped and no metric is emitted.
            continue
        group_flags = None
        if repetition_flags is not None:
            group_flags = [bool(repetition_flags[i]) for i in indices]
        group_metrics = compute_group_jaccard_metrics(
            [token_ids_per_response[i] for i in indices],
            n_gram=n_gram,
            relative_positions=relative_positions,
            absolute_prefix_tokens=absolute_prefix_tokens,
            repetition_flags=group_flags,
        )
        if preds is not None:
            group_metrics.update(compute_group_answer_metrics([preds[i] for i in indices]))
        if not group_metrics:
            continue
        data_source = str(data_sources[indices[0]])
        for name, value in group_metrics.items():
            results[data_source][name].append(value)
    return results
