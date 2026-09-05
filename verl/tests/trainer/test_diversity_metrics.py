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
"""Unit tests for the group-level diversity metrics.

The module under test only needs the standard library, so it is loaded by
file path and every test injects a deterministic tokenize_fn; no tokenizer
or torch runtime is required.
"""

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "verl" / "trainer" / "ppo" / "diversity_metrics.py"


def _load_module():
    try:
        from verl.trainer.ppo import diversity_metrics as module

        return module
    except ImportError:  # standalone environment without verl's runtime deps
        spec = importlib.util.spec_from_file_location("diversity_metrics_under_test", _MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


dm = _load_module()


def _tokenizer_from_tables(tables: dict[str, list[int]]):
    return lambda texts: [list(tables[text]) for text in texts]


def test_identical_responses_have_zero_full_distance():
    metrics = dm.compute_group_jaccard_metrics(
        [list(range(20))] * 3, relative_positions=[1.0], absolute_prefix_tokens=None
    )
    assert metrics["all/full"] == 0.0
    assert metrics["all/n_groups"] == 1.0


def test_disjoint_responses_have_full_distance_one():
    metrics = dm.compute_group_jaccard_metrics(
        [list(range(20)), list(range(100, 120)), list(range(200, 220))],
        relative_positions=[1.0],
        absolute_prefix_tokens=None,
    )
    assert metrics["all/full"] == 1.0


def test_relative_and_absolute_prefix_points():
    # A and B share the first 5 tokens and diverge afterwards.
    tokens_a = list(range(10))
    tokens_b = [0, 1, 2, 3, 4, 50, 51, 52, 53, 54]
    metrics = dm.compute_group_jaccard_metrics(
        [tokens_a, tokens_b],
        relative_positions=[0.5, 1.0],
        absolute_prefix_tokens=8,
    )
    # At half length both prefixes are the shared tokens: distance 0.
    assert metrics["all/rel@0.5"] == 0.0
    # Full 10-token responses share only the 2 boundary-spanning grams of 12.
    assert metrics["all/full"] == pytest.approx(1 - 2 / 12)
    # The first 8 tokens still share the opening grams: distance 0.75.
    assert metrics["all/abs@8"] == pytest.approx(1 - 2 / 8)


def test_filtered_condition_excludes_flagged_responses():
    tokens_a = list(range(20))
    tokens_b = list(range(100, 120))
    metrics = dm.compute_group_jaccard_metrics(
        [tokens_a, list(range(20)), tokens_b],
        relative_positions=[1.0],
        absolute_prefix_tokens=None,
        repetition_flags=[False, False, True],
    )
    # all: one identical pair plus two maximally distant pairs.
    assert metrics["all/full"] == pytest.approx(2 / 3)
    # filtered keeps only the identical pair.
    assert metrics["filtered/full"] == 0.0
    assert metrics["filtered/n_groups"] == 1.0


def test_filtered_without_enough_survivors_reports_no_groups():
    metrics = dm.compute_group_jaccard_metrics(
        [list(range(20)), list(range(20))],
        relative_positions=[1.0],
        absolute_prefix_tokens=None,
        repetition_flags=[True, False],
    )
    assert metrics["filtered/n_groups"] == 0.0
    assert "filtered/full" not in metrics
    assert metrics["all/full"] == 0.0


def test_responses_shorter_than_ngram_are_unusable():
    metrics = dm.compute_group_jaccard_metrics(
        [[1, 2], [3, 4], [5, 6]], relative_positions=[1.0], absolute_prefix_tokens=8
    )
    assert metrics == {}


def test_invalid_points_are_rejected():
    with pytest.raises(ValueError):
        dm.compute_group_jaccard_metrics([[1, 2, 3, 4]] * 2, relative_positions=[1.5])
    with pytest.raises(ValueError):
        dm.compute_group_jaccard_metrics([[1, 2, 3, 4]] * 2, relative_positions=[0.0])
    with pytest.raises(ValueError):
        dm.compute_group_jaccard_metrics([[1, 2, 3, 4]] * 2, absolute_prefix_tokens=0)


def test_answer_metrics_count_invalid_predictions():
    metrics = dm.compute_group_answer_metrics(["42", " 42 ", "7", None, ""])
    assert metrics["answer/valid_ratio"] == pytest.approx(3 / 5)
    assert metrics["answer/unique_ratio"] == pytest.approx(2 / 3)
    assert metrics["answer/top1_ratio"] == pytest.approx(2 / 3)


def test_answer_metrics_with_all_invalid():
    metrics = dm.compute_group_answer_metrics([None, "", None])
    assert metrics["answer/valid_ratio"] == 0.0
    assert "answer/unique_ratio" not in metrics
    assert "answer/top1_ratio" not in metrics


def test_batch_grouping_by_uid_and_data_source():
    tables = {
        "a1": list(range(20)),
        "a2": list(range(20)),  # identical to a1
        "b1": list(range(20)),
        "b2": list(range(100, 120)),  # disjoint from b1
    }
    texts = ["a1", "a2", "b1", "b2"]
    uids = ["u1", "u1", "u2", "u2"]
    data_sources = ["d1", "d1", "d2", "d2"]
    results = dm.compute_group_diversity_metrics(
        texts,
        uids,
        data_sources,
        _tokenizer_from_tables(tables),
        relative_positions=[1.0],
        absolute_prefix_tokens=None,
        repetition_flags=[False] * 4,
        preds=["1", "1", "2", None],
    )
    assert results["d1"]["all/full"] == [0.0]
    assert results["d2"]["all/full"] == [1.0]
    # Answer metrics aggregate per data_source over the group's own responses.
    assert results["d1"]["answer/valid_ratio"] == [1.0]
    assert results["d1"]["answer/unique_ratio"] == [0.5]
    assert results["d1"]["answer/top1_ratio"] == [1.0]
    assert results["d2"]["answer/valid_ratio"] == [0.5]


def test_batch_alignment_is_validated():
    with pytest.raises(ValueError):
        dm.compute_group_diversity_metrics(
            ["a", "b"],
            ["u1", "u1"],
            ["d", "d"],
            lambda ts: [[1, 2, 3, 4]] * len(ts),
            repetition_flags=[True],
        )
    with pytest.raises(ValueError):
        dm.compute_group_diversity_metrics(
            ["a", "b"],
            ["u1", "u1"],
            ["d", "d"],
            lambda ts: [[1, 2, 3, 4]] * len(ts),
            preds=["x"],
        )


def test_single_response_groups_are_skipped():
    # With val_kwargs.n=1 every group holds one response; diversity metrics
    # must not emit meaningless all-ones answer values.
    tables = {"a": [1, 2, 3, 4, 5], "b": [10, 11, 12, 13, 14]}
    results = dm.compute_group_diversity_metrics(
        ["a", "b"],
        ["u1", "u2"],
        ["d", "d"],
        _tokenizer_from_tables(tables),
        relative_positions=[1.0],
        absolute_prefix_tokens=None,
        preds=["1", "2"],
    )
    assert results == {}


def test_empty_batch_returns_nothing():
    assert dm.compute_group_diversity_metrics([], [], [], lambda ts: []) == {}
