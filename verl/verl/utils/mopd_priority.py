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
"""MOPD replay priority math (paper §4.4 module M3, §4.6 property P-C).

The replay sampler draws mini-batch rows under the corrected P-C priority
measure: per-state demand follows the per-state optimal step size
``eta(s) ~ 1 / mu_eff(s)``, where ``mu_eff`` is the marginal absorption rate
of the residual-energy measure ``q ∝ b·d²``. The original instantiation
``∝ G(s)`` was falsified by the distribution-level E19/E20 experiments and
survives only as the explicit negative control; ``uniform`` is the flat
baseline.

Allocation is deterministic and stratified (integer quotas by largest
remainder), never iid-multinomial reweighting: Adam's scale invariance
annihilates slot-count noise, so only exact integer counts carry the
priority signal into the update.

This module is numpy-only by design so the sampler semantics stay
CPU-testable without torch. ``MopdReplayBuffer`` is the thin DataProto
wrapper around it.
"""

import numpy as np

__all__ = [
    "PRIORITY_MODES",
    "position_demand",
    "priority_weights",
    "row_demand",
    "stratified_quota",
]

PRIORITY_MODES = ("marginal_absorption", "proportional_G_negcontrol", "uniform")

#: Floor under mu_eff inside the demand reciprocal; converged states are
#: gated on residual energy before this floor is ever reached.
MU_EFF_FLOOR = 1e-12

#: Batch fields consumed per priority mode. ``response_mask`` is always needed.
MODE_FIELDS = {
    "marginal_absorption": ("state_mu_eff", "state_resid_energy"),
    "proportional_G_negcontrol": ("state_gap",),
    "uniform": (),
}


def position_demand(fields: dict[str, np.ndarray], mode: str) -> np.ndarray:
    """Per-position replay demand ``(B, T)`` under the P-C priority measure.

    ``fields`` carries the per-state batch fields plus ``response_mask``.
    ``marginal_absorption`` demand is ``1/mu_eff`` on positions with residual
    energy and exactly 0 where ``E_b[d^2] = 0``: a converged state has no
    residual measure to absorb, so the inverse rate would otherwise assign
    it a spuriously large share. ``proportional_G_negcontrol`` is
    ``relu(G(s))``, the falsified E19/E20 baseline kept for ablations.
    """
    if mode == "uniform":
        return np.ones_like(fields["response_mask"], dtype=np.float64)
    if mode == "marginal_absorption":
        energy = np.asarray(fields["state_resid_energy"], dtype=np.float64)
        mu_eff = np.asarray(fields["state_mu_eff"], dtype=np.float64)
        demand = np.where(energy > 0.0, 1.0 / np.maximum(mu_eff, MU_EFF_FLOOR), 0.0)
        return np.where(np.isfinite(demand), demand, 0.0)
    if mode == "proportional_G_negcontrol":
        return np.maximum(np.asarray(fields["state_gap"], dtype=np.float64), 0.0)
    raise ValueError(f"unknown MOPD replay priority_mode: {mode!r}; expected one of {PRIORITY_MODES}")


def row_demand(demand: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Aggregate per-position demand into one scalar per row (masked mean).

    A replay slot serves every live position of the row at once, so the
    bundle's demand is the *average* per-state demand — the row-level lift
    of the per-state allocation ``eta(s) ~ 1/mu_eff(s)``.
    """
    mask = np.asarray(mask, dtype=np.float64)
    denom = np.maximum(mask.sum(axis=-1), 1.0)
    return (np.asarray(demand, dtype=np.float64) * mask).sum(axis=-1) / denom


def priority_weights(demand: np.ndarray, n_sampled: np.ndarray, coverage_rarity: float) -> np.ndarray:
    """Blend the priority measure with coverage rarity into draw weights.

    ``demand`` and the rarity score ``r_s = 1/(1 + n_s)`` are each normalized
    to mean 1 over the live buffer, so ``coverage_rarity`` is a true mixture
    weight: 0 replays purely under the priority measure, 1 purely by rarity.
    An all-zero demand map falls back to uniform weights rather than nan.
    """
    weights = np.asarray(demand, dtype=np.float64)
    mean = weights.mean()
    weights = weights / mean if mean > 0 else np.ones_like(weights)
    if coverage_rarity <= 0.0:
        return weights
    rarity = 1.0 / (1.0 + np.asarray(n_sampled, dtype=np.float64))
    rarity = rarity / rarity.mean()
    return (1.0 - coverage_rarity) * weights + coverage_rarity * rarity


def stratified_quota(weights: np.ndarray, n_slots: int) -> np.ndarray:
    """Deterministic stratified allocation of ``n_slots`` draws.

    Floors the exact proportional quota, then distributes the remainder by
    largest fractional part; ties resolve to the lower buffer index via a
    stable argsort, so the result is fully determined by the weights.
    All-zero weights degrade to an exact uniform split.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if n_slots <= 0 or weights.size == 0:
        return np.zeros(weights.shape, dtype=np.int64)
    total = weights.sum()
    share = weights / total if total > 0 else np.full(weights.shape, 1.0 / weights.size)
    exact = share * n_slots
    quota = np.floor(exact).astype(np.int64)
    remainder = int(n_slots - quota.sum())
    if remainder > 0:
        order = np.argsort(-(exact - quota), kind="stable")
        quota[order[:remainder]] += 1
    return quota
