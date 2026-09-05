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
Repetition and compression metrics computed from decoded response texts.

The detector compresses overlapping character windows of each response with
zlib; a window whose compression ratio (raw bytes / compressed bytes) reaches
the threshold marks the response as repetitive. Defaults are window=1024,
stride=256, threshold=10, so the training-time metric stays comparable with
the offline window-scan analysis of eval rollouts.

Per-response values are scalars so they aggregate with the existing metric
pipelines: mean/p90/max over a rollout batch for training, and the
per-data_source mean@N grouping for validation.
"""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np

DEFAULT_WINDOW_SIZE = 1024
DEFAULT_STRIDE = 256
DEFAULT_THRESHOLD = 10.0


def compression_ratio(text: str) -> float:
    """Full-text zlib compression ratio: raw UTF-8 bytes / compressed bytes.

    Returns 0.0 for empty text so aborted responses degrade to "no signal"
    instead of inf/nan in downstream means.
    """
    raw = text.encode("utf-8")
    if not raw:
        return 0.0
    return len(raw) / max(len(zlib.compress(raw, 9)), 1)


def _window_starts(length: int, window_size: int, stride: int) -> list[int]:
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    last = length - window_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def window_ratios(text: str, window_size: int = DEFAULT_WINDOW_SIZE, stride: int = DEFAULT_STRIDE) -> np.ndarray:
    """Compression ratio of each overlapping character window of the text."""
    if not text:
        return np.zeros(1)
    return np.asarray(
        [
            compression_ratio(text[start : start + window_size])
            for start in _window_starts(len(text), window_size, stride)
        ]
    )


def analyze_response(
    text: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Per-response scalar metrics for one decoded response text.

    - repetition: 1.0 when any window ratio >= threshold, else 0.0
    - max_window_ratio: largest local compression ratio (0.0 for empty text)
    - onset_frac: first flagged window start / text length; 1.0 when never
      flagged, so the mean over all responses reads as "relative position
      where degeneration begins (1 = never)"
    - post_onset_frac: character share after the first flagged window start;
      0.0 when never flagged, i.e. the wasted share of the generation
    - compression_ratio: full-text zlib compression ratio
    """
    if not text:
        return {
            "repetition": 0.0,
            "max_window_ratio": 0.0,
            "onset_frac": 1.0,
            "post_onset_frac": 0.0,
            "compression_ratio": 0.0,
        }

    ratios = window_ratios(text, window_size, stride)
    max_ratio = float(np.max(ratios))
    flagged = bool(max_ratio >= threshold)
    if flagged:
        onset = float(_window_starts(len(text), window_size, stride)[int(np.argmax(ratios >= threshold))])
        onset_frac = onset / len(text)
        post_onset_frac = 1.0 - onset_frac
    else:
        onset_frac = 1.0
        post_onset_frac = 0.0

    return {
        "repetition": float(flagged),
        "max_window_ratio": max_ratio,
        "onset_frac": onset_frac,
        "post_onset_frac": post_onset_frac,
        "compression_ratio": compression_ratio(text),
    }


def analyze_responses(texts: list[str], **kwargs: Any) -> dict[str, list[float]]:
    """Per-response metric lists aligned with the input order.

    The list format plugs directly into validation aggregation, where every
    sample must carry a value (aborted responses are analyzed as empty text).
    """
    per_response = [analyze_response(text, **kwargs) for text in texts]
    keys = list(per_response[0]) if per_response else []
    return {key: [item[key] for item in per_response] for key in keys}


def compute_repetition_metrics(texts: list[str], **kwargs: Any) -> dict[str, float]:
    """Aggregate per-response metrics over a rollout batch (train logging).

    Callers should exclude aborted responses beforehand; empty texts are
    analyzed as "no repetition" but would dilute the means.
    """
    if not texts:
        return {}

    per_response = analyze_responses(texts, **kwargs)
    max_ratios = np.asarray(per_response["max_window_ratio"])
    return {
        "repetition/frac": float(np.mean(per_response["repetition"])),
        "repetition/max_window_ratio/mean": float(np.mean(max_ratios)),
        "repetition/max_window_ratio/p90": float(np.percentile(max_ratios, 90)),
        "repetition/max_window_ratio/max": float(np.max(max_ratios)),
        "repetition/onset_frac/mean": float(np.mean(per_response["onset_frac"])),
        "repetition/post_onset_frac/mean": float(np.mean(per_response["post_onset_frac"])),
        "compression/ratio/mean": float(np.mean(per_response["compression_ratio"])),
    }
