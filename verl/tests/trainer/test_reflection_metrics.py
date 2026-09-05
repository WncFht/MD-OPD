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
"""Unit tests for the lexical reflection metrics.

The module under test only needs numpy and the standard library, so it is
loaded by file path; no torch or ray runtime is required.
"""

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "verl" / "trainer" / "ppo" / "reflection_metrics.py"


def _load_module():
    try:
        from verl.trainer.ppo import reflection_metrics as module

        return module
    except ImportError:  # standalone environment without verl's runtime deps
        spec = importlib.util.spec_from_file_location("reflection_metrics_under_test", _MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


rm = _load_module()


def test_lexical_list_invariants():
    all_words = [word for words in rm.LEXICAL_LIST.values() for word in words]
    assert len(all_words) >= 32, "the panel must track at least 32 marker words"
    assert len(set(all_words)) == len(all_words), "words must be unique across categories"
    assert all(word == word.lower() and word for word in all_words)
    assert set(rm.LEXICAL_LIST) == {"core", "branch", "verification"}
    assert all(len(words) > 0 for words in rm.LEXICAL_LIST.values())


def test_required_words_are_covered():
    required = [
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
    assert set(required) <= set(rm.ALL_KEYWORDS)


def test_word_keys_are_unique_metric_names():
    keys = [rm.WORD_KEYS[word] for word in rm.ALL_KEYWORDS]
    assert len(set(keys)) == len(keys)
    assert " " not in "".join(keys), "panel keys must not contain spaces"
    # Hyphen variants stay distinct.
    assert rm.WORD_KEYS["double-check"] == "double-check"
    assert rm.WORD_KEYS["double check"] == "double_check"


def test_matching_is_case_insensitive_and_boundary_aware():
    counts, _ = rm.count_keywords("Wait, WAIT. await awaits. checker checks checked.")
    assert counts["wait"] == 2  # "await"/"awaits" are excluded by the boundary
    counts, _ = rm.count_keywords("Check it. checker checks.")
    assert counts["check"] == 1
    counts, _ = rm.count_keywords("Oh! OH! Ohio.")
    assert counts["oh"] == 2


def test_longest_phrase_wins():
    counts, categories = rm.count_keywords("But wait, let me check the checker. On second thought, no.")
    assert counts["but wait"] == 1
    assert counts["but"] == 0  # consumed by the longer "but wait"
    assert counts["wait"] == 0
    assert counts["let me check"] == 1
    assert counts["check"] == 0
    assert counts["on second thought"] == 1
    assert categories["branch"] == 1
    assert categories["verification"] == 1
    assert categories["core"] == 1


def test_analyze_response_rates_and_presence():
    text = "Let me check this. Wait, actually, hold on - hmm, alternatively, double-check it. But wait, oh no."
    counts, categories = rm.count_keywords(text)
    n_words = len(text.split())
    metrics = rm.analyze_response(text)
    for category, count in categories.items():
        assert metrics[f"reflection/{category}/rate_per_1k"] == pytest.approx(1000.0 * count / n_words)
        assert metrics[f"reflection/{category}/presence"] == 1.0
    # Present words carry rate and presence; the rest stay at zero.
    for word, count in counts.items():
        key = rm.WORD_KEYS[word]
        assert metrics[f"reflection/word/{key}/rate_per_1k"] == pytest.approx(1000.0 * count / n_words)
        assert metrics[f"reflection/word/{key}/presence"] == 1.0
    assert metrics["reflection/word/reconsider/rate_per_1k"] == 0.0
    assert metrics["reflection/word/reconsider/presence"] == 0.0
    # Every response must carry the complete key set regardless of content.
    assert set(metrics) == set(rm.analyze_response("unrelated plain narration"))


def test_empty_text_yields_zero_signal():
    metrics = rm.analyze_response("")
    assert len(metrics) == len(rm.analyze_response("some text"))
    assert all(value == 0.0 for value in metrics.values())


def test_train_aggregate_is_mean_of_per_response():
    texts = ["Wait, let me check it again.", "A plain sentence without any markers whatsoever."]
    per_response = rm.analyze_reflection_responses(texts)
    aggregated = rm.compute_reflection_metrics(texts)
    assert set(aggregated) == set(per_response)
    for key, values in per_response.items():
        assert aggregated[key] == pytest.approx(sum(values) / len(values))
    # Macro average over per-response rates, not a corpus-level ratio.
    first_rate = rm.analyze_response(texts[0])["reflection/word/wait/rate_per_1k"]
    assert aggregated["reflection/word/wait/rate_per_1k"] == pytest.approx(first_rate / 2)


def test_grouped_by_uid_and_data_source():
    texts = ["wait wait wait", "nothing here", "hmm, let me verify", "so be it"]
    per_response = rm.analyze_reflection_responses(texts)
    results = rm.compute_grouped_reflection_metrics(
        per_response,
        uids=["u1", "u1", "u2", "u2"],
        data_sources=["d1", "d1", "d2", "d2"],
    )
    # u1: two responses, one with core markers.
    assert results["d1"]["reflection/core/presence"] == [0.5]
    assert results["d1"]["reflection/core/all_absent"] == [0.0]
    assert results["d1"]["reflection/word/wait/presence"] == [0.5]
    # u2: one response with markers, one without.
    assert results["d2"]["reflection/core/presence"] == [0.5]
    assert results["d2"]["reflection/verification/all_absent"] == [0.0]


def test_all_absent_group():
    texts = ["nothing here", "also nothing"]
    per_response = rm.analyze_reflection_responses(texts)
    results = rm.compute_grouped_reflection_metrics(per_response, uids=["u1", "u1"], data_sources=["d"] * 2)
    assert results["d"]["reflection/core/all_absent"] == [1.0]
    assert results["d"]["reflection/core/presence"] == [0.0]


def test_valid_mask_excludes_aborted_responses():
    texts = ["wait, hmm", "", "hmm, let me verify"]
    per_response = rm.analyze_reflection_responses(texts)
    # The middle response is aborted: it must not dilute the group rates.
    results = rm.compute_grouped_reflection_metrics(
        per_response, uids=["u1", "u1", "u1"], data_sources=["d"] * 3, valid=[True, False, True]
    )
    assert results["d"]["reflection/core/presence"] == [1.0]
    assert results["d"]["reflection/verification/presence"] == [0.5]
    assert results["d"]["reflection/core/all_absent"] == [0.0]

    # A group whose responses are all aborted disappears entirely.
    results = rm.compute_grouped_reflection_metrics(
        per_response, uids=["u1", "u1", "u2"], data_sources=["d"] * 3, valid=[True, False, False]
    )
    assert all(len(vals) == 1 for vals in results["d"].values())


def test_alignment_errors_are_rejected():
    per_response = rm.analyze_reflection_responses(["wait", "hmm"])
    with pytest.raises(ValueError):
        rm.compute_grouped_reflection_metrics(per_response, uids=["u1"], data_sources=["d", "d"])
    with pytest.raises(ValueError):
        rm.compute_grouped_reflection_metrics(per_response, uids=["u1", "u1"], data_sources=["d"])
    with pytest.raises(ValueError):
        rm.compute_grouped_reflection_metrics(per_response, uids=["u1", "u1"], data_sources=["d", "d"], valid=[True])


def test_grouping_inputs_are_required():
    per_response = rm.analyze_reflection_responses(["wait", "hmm"])
    with pytest.raises(ValueError):
        rm.compute_grouped_reflection_metrics(per_response, None, ["d", "d"])
    with pytest.raises(ValueError):
        rm.compute_grouped_reflection_metrics(per_response, ["u1", "u1"], None)


def test_unflagged_rates_exclude_flagged_responses():
    texts = ["wait wait wait", "nothing here at all", "hmm, let me verify"]
    per_response = rm.analyze_reflection_responses(texts)
    results = rm.compute_grouped_reflection_metrics(
        per_response,
        uids=["u1", "u1", "u1"],
        data_sources=["d"] * 3,
        repetition_flags=[True, False, False],
    )
    # The raw series still include the flagged sample.
    assert results["d"]["reflection/word/wait/rate_per_1k"] == [pytest.approx(1000.0 / 3)]
    # The unflagged series come from the two clean samples only.
    assert results["d"]["reflection/word/wait/rate_per_1k_unflagged"] == [0.0]
    assert results["d"]["reflection/word/hmm/rate_per_1k_unflagged"] == [pytest.approx(125.0)]
    assert results["d"]["reflection/core/rate_per_1k_unflagged"] == [pytest.approx(125.0)]
    assert results["d"]["reflection/verification/rate_per_1k_unflagged"] == [pytest.approx(125.0)]
    assert results["d"]["reflection/unflagged_groups"] == [1.0]
    # Presence has no unflagged variant: rates are the comparable series.
    assert "reflection/core/presence_unflagged" not in results["d"]


def test_unflagged_rates_skip_all_flagged_groups():
    texts = ["wait wait wait wait", "hmm, verify this"]
    per_response = rm.analyze_reflection_responses(texts)
    results = rm.compute_grouped_reflection_metrics(
        per_response,
        uids=["u1", "u1"],
        data_sources=["d"] * 2,
        repetition_flags=[True, True],
    )
    assert results["d"]["reflection/unflagged_groups"] == [0.0]
    assert not any(key.endswith("_unflagged") for key in results["d"])
    # Raw series are unaffected by the flags.
    assert results["d"]["reflection/core/presence"] == [1.0]


def test_unflagged_series_require_repetition_flags():
    per_response = rm.analyze_reflection_responses(["wait", "hmm"])
    results = rm.compute_grouped_reflection_metrics(per_response, uids=["u1", "u1"], data_sources=["d"] * 2)
    assert not any(key.endswith("_unflagged") for key in results["d"])
    assert "reflection/unflagged_groups" not in results["d"]


def test_flag_alignment_errors_are_rejected():
    per_response = rm.analyze_reflection_responses(["wait", "hmm"])
    with pytest.raises(ValueError):
        rm.compute_grouped_reflection_metrics(
            per_response, uids=["u1", "u1"], data_sources=["d"] * 2, repetition_flags=[True]
        )


def test_empty_inputs():
    assert rm.compute_reflection_metrics([]) == {}
    assert rm.compute_grouped_reflection_metrics({}, [], []) == {}
