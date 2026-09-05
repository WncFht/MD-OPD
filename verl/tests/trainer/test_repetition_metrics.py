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
"""Unit tests for the group-level repetition aggregates.

The module under test only needs numpy and the standard library, so it is
loaded by file path; no torch or ray runtime is required.
"""

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "verl" / "trainer" / "ppo" / "repetition_metrics.py"


def _load_module():
    try:
        from verl.trainer.ppo import repetition_metrics as module

        return module
    except ImportError:  # standalone environment without verl's runtime deps
        spec = importlib.util.spec_from_file_location("repetition_metrics_under_test", _MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


rm = _load_module()


def _per_response(repetition, ratios, onsets, posts, comps):
    return {
        "repetition": repetition,
        "max_window_ratio": ratios,
        "onset_frac": onsets,
        "post_onset_frac": posts,
        "compression_ratio": comps,
    }


def test_group_aggregates_on_known_values():
    metrics = rm.compute_group_repetition_metrics(
        _per_response(
            repetition=[1, 0, 1, 1],
            ratios=[2.0, 12.0, 3.0, 4.0],
            onsets=[1.0, 0.5, 1.0, 0.25],
            posts=[0.0, 0.5, 0.0, 0.75],
            comps=[3.0, 10.0, 4.0, 5.0],
        )
    )
    assert metrics["repetition/frac"] == pytest.approx(0.75)
    assert metrics["repetition/all_flagged"] == 0.0
    assert metrics["repetition/max_window_ratio/mean"] == pytest.approx(5.25)
    assert metrics["repetition/max_window_ratio/p90"] == pytest.approx(9.6)  # linear interpolation
    assert metrics["repetition/max_window_ratio/max"] == pytest.approx(12.0)
    assert metrics["repetition/onset_frac/mean"] == pytest.approx(0.6875)
    assert metrics["repetition/post_onset_frac/mean"] == pytest.approx(0.3125)
    assert metrics["compression/ratio/mean"] == pytest.approx(5.5)


def test_all_flagged_group():
    metrics = rm.compute_group_repetition_metrics(
        _per_response([1, 1], [11.0, 12.0], [0.1, 0.2], [0.9, 0.8], [20.0, 30.0])
    )
    assert metrics["repetition/frac"] == 1.0
    assert metrics["repetition/all_flagged"] == 1.0


def test_single_response_group_is_valid():
    metrics = rm.compute_group_repetition_metrics(_per_response([0], [3.0], [1.0], [0.0], [4.0]))
    assert metrics["repetition/frac"] == 0.0
    assert metrics["repetition/all_flagged"] == 0.0
    assert metrics["repetition/max_window_ratio/p90"] == pytest.approx(3.0)


def test_grouped_by_uid_and_data_source():
    per_response = _per_response(
        repetition=[1, 0, 1, 1],
        ratios=[11.0, 3.0, 12.0, 13.0],
        onsets=[0.1, 1.0, 0.2, 0.3],
        posts=[0.9, 0.0, 0.8, 0.7],
        comps=[20.0, 4.0, 21.0, 22.0],
    )
    results = rm.compute_grouped_repetition_metrics(
        per_response,
        uids=["u1", "u1", "u2", "u2"],
        data_sources=["d1", "d1", "d2", "d2"],
    )
    assert results["d1"]["repetition/frac"] == [0.5]
    assert results["d1"]["repetition/all_flagged"] == [0.0]
    assert results["d2"]["repetition/frac"] == [1.0]
    assert results["d2"]["repetition/all_flagged"] == [1.0]


def test_valid_mask_excludes_aborted_responses():
    per_response = _per_response(
        repetition=[1, 0, 1],
        ratios=[11.0, 0.0, 12.0],
        onsets=[0.1, 1.0, 0.2],
        posts=[0.9, 0.0, 0.8],
        comps=[20.0, 0.0, 21.0],
    )
    # The middle response is aborted: it must not dilute the group rates.
    results = rm.compute_grouped_repetition_metrics(
        per_response, uids=["u1", "u1", "u1"], data_sources=["d"] * 3, valid=[True, False, True]
    )
    assert results["d"]["repetition/frac"] == [1.0]
    assert results["d"]["repetition/all_flagged"] == [1.0]

    # A group whose responses are all aborted disappears entirely.
    results = rm.compute_grouped_repetition_metrics(
        per_response, uids=["u1", "u1", "u2"], data_sources=["d"] * 3, valid=[True, False, False]
    )
    assert results["d"]["repetition/frac"] == [1.0]
    assert all(len(vals) == 1 for vals in results["d"].values())


def test_alignment_errors_are_rejected():
    per_response = _per_response([1, 0], [11.0, 3.0], [0.1, 1.0], [0.9, 0.0], [20.0, 4.0])
    with pytest.raises(ValueError):
        rm.compute_grouped_repetition_metrics(per_response, uids=["u1"], data_sources=["d", "d"])
    with pytest.raises(ValueError):
        rm.compute_grouped_repetition_metrics(per_response, uids=["u1", "u1"], data_sources=["d"])
    with pytest.raises(ValueError):
        rm.compute_grouped_repetition_metrics(per_response, uids=["u1", "u1"], data_sources=["d", "d"], valid=[True])


def test_end_to_end_with_analyze_responses():
    texts = ["blah" * 400, "A perfectly normal sentence about the weather.", "blah" * 400]
    per_response = rm.analyze_responses(texts)
    results = rm.compute_grouped_repetition_metrics(per_response, uids=["u1"] * 3, data_sources=["aime24"] * 3)
    assert results["aime24"]["repetition/frac"] == [pytest.approx(2 / 3)]
    assert results["aime24"]["repetition/all_flagged"] == [0.0]


def test_empty_inputs():
    assert rm.compute_group_repetition_metrics({}) == {}
    assert rm.compute_grouped_repetition_metrics({}, [], []) == {}
