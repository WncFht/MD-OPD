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
"""Unit test for the MOPD replay priority sampler (paper §4.4 M3, §4.6 P-C).

Numpy-only by design: exercises ``verl/utils/mopd_priority.py`` (priority
demand per mode, row aggregation, coverage-rarity blend, stratified quota)
plus a fake in-memory buffer scenario. Loads the module by file path so it
runs without torch:

    uv run --with numpy python3 verl/tests/test_mopd_replay_sampler_on_cpu.py

Also pytest-compatible (``pytest verl/tests/test_mopd_replay_sampler_on_cpu.py``).
"""

import importlib.util
from pathlib import Path

import numpy as np

_MODULE_PATH = Path(__file__).resolve().parent.parent / "verl" / "utils" / "mopd_priority.py"
_spec = importlib.util.spec_from_file_location("mopd_priority", _MODULE_PATH)
mopd_priority = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mopd_priority)

position_demand = mopd_priority.position_demand
row_demand = mopd_priority.row_demand
priority_weights = mopd_priority.priority_weights
stratified_quota = mopd_priority.stratified_quota


def _fields(mask, **kw):
    fields = {"response_mask": np.asarray(mask, dtype=np.float32)}
    fields.update({k: np.asarray(v, dtype=np.float32) for k, v in kw.items()})
    return fields


def test_marginal_absorption_demand():
    # demand = 1/mu_eff where residual energy > 0, else 0.
    mask = [[1, 1], [1, 0]]
    mu_eff = [[0.5, 2.0], [4.0, 0.25]]
    energy = [[1.0, 0.1], [0.0, 1.0]]
    demand = position_demand(_fields(mask, state_mu_eff=mu_eff, state_resid_energy=energy), "marginal_absorption")
    expected = np.array([[2.0, 0.5], [0.0, 4.0]])  # 1/0.5, 1/2.0; energy=0 -> 0; 1/0.25
    np.testing.assert_allclose(demand, expected, rtol=1e-10)
    # inf mu_eff with positive energy -> zero demand (numerically finite anyway)
    demand_inf = position_demand(
        _fields(mask, state_mu_eff=[[np.inf, 2.0], [4.0, 0.25]], state_resid_energy=energy),
        "marginal_absorption",
    )
    assert demand_inf[0, 0] == 0.0
    assert np.isfinite(demand_inf).all()


def test_proportional_g_negcontrol_demand():
    gap = [[0.3, -0.2], [1.5, 0.0]]
    demand = position_demand(_fields([[1, 1], [1, 1]], state_gap=gap), "proportional_G_negcontrol")
    np.testing.assert_allclose(demand, [[0.3, 0.0], [1.5, 0.0]])


def test_uniform_and_unknown_mode():
    mask = [[1, 1, 0], [1, 0, 1]]
    demand = position_demand(_fields(mask), "uniform")
    np.testing.assert_array_equal(demand, np.ones((2, 3)))
    try:
        position_demand(_fields(mask), "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown priority_mode must raise ValueError")


def test_row_demand_masked_mean():
    demand = np.array([[2.0, 4.0, 0.0], [1.0, 3.0, 5.0]])
    mask = np.array([[1, 1, 0], [0, 0, 0]])
    rows = row_demand(demand, mask)
    np.testing.assert_allclose(rows, [3.0, 0.0])  # zero-mask row -> 0, not nan


def test_priority_weights_pure_and_blended():
    demand = np.array([4.0, 1.0, 0.0])
    counts = np.array([0, 0, 0])

    w = priority_weights(demand, counts, coverage_rarity=0.0)
    np.testing.assert_allclose(w, demand / demand.mean())

    # all-zero demand degrades to uniform, never nan
    w0 = priority_weights(np.zeros(3), counts, coverage_rarity=0.0)
    np.testing.assert_allclose(w0, np.ones(3))

    # rarity mixture: rare entries gain share vs the well-sampled one. Entry 0
    # loses absolute share to entry 2 (same rarity, zero demand rescues it from
    # 0), but still gains relative to the over-sampled entry 1.
    counts2 = np.array([0, 9, 0])
    w_blend = priority_weights(demand, counts2, coverage_rarity=0.5)
    w_pure = priority_weights(demand, counts2, coverage_rarity=0.0)
    assert w_blend[2] > w_pure[2]  # starved rare row rescued from zero
    assert w_blend[1] < w_pure[1]  # n=9 -> lowest rarity
    assert w_blend[0] / w_blend[1] > w_pure[0] / w_pure[1]  # rare gains vs sampled
    np.testing.assert_allclose(w_blend, 0.5 * w_pure + 0.5 * (1.0 / (1.0 + counts2)) / np.mean(1.0 / (1.0 + counts2)))


def test_stratified_quota_exact_and_deterministic():
    weights = np.array([0.5, 0.3, 0.2])
    q = stratified_quota(weights, 10)
    assert q.sum() == 10
    np.testing.assert_array_equal(q, [5, 3, 2])  # exact proportions, no remainder

    # largest-remainder: exact = [1.6, 1.6, 0.8] -> floors [1,1,0], remainder 2
    # -> +1 goes to the 0.8 fraction, then to the first 0.6 (tie -> low index).
    q2 = stratified_quota(np.array([0.4, 0.4, 0.2]), 4)
    np.testing.assert_array_equal(q2, [2, 1, 1])
    # determinism across calls and under identical ties (lowest index wins)
    q3a = stratified_quota(np.array([1.0, 1.0, 1.0]), 4)
    q3b = stratified_quota(np.array([1.0, 1.0, 1.0]), 4)
    np.testing.assert_array_equal(q3a, q3b)
    np.testing.assert_array_equal(q3a, [2, 1, 1])
    # fewer slots than entries; all-zero -> exact uniform split
    np.testing.assert_array_equal(stratified_quota(np.zeros(4), 8), [2, 2, 2, 2])
    np.testing.assert_array_equal(stratified_quota(np.array([0.9, 0.1]), 2), [2, 0])


def test_end_to_end_buffer_scenario():
    # Two windows of 2 rows each; row demand precomputed the way
    # MopdReplayBuffer.append does (masked mean of position demand).
    mask_w1 = np.array([[1, 1], [1, 1]])
    mask_w2 = np.array([[1, 1], [1, 1]])
    # window1: slow absorber (mu_eff 0.05) vs fast absorber (mu_eff 1.0)
    dem_w1 = row_demand(
        position_demand(
            _fields(
                mask_w1,
                state_mu_eff=[[0.05, 0.05], [1.0, 1.0]],
                state_resid_energy=[[1.0, 1.0], [1.0, 1.0]],
            ),
            "marginal_absorption",
        ),
        mask_w1,
    )
    # window2: one converged row (energy 0) + one moderate row
    dem_w2 = row_demand(
        position_demand(
            _fields(
                mask_w2,
                state_mu_eff=[[0.5, 0.5], [0.2, 0.2]],
                state_resid_energy=[[0.0, 0.0], [1.0, 1.0]],
            ),
            "marginal_absorption",
        ),
        mask_w2,
    )
    np.testing.assert_allclose(dem_w1, [20.0, 1.0], rtol=1e-6)
    np.testing.assert_allclose(dem_w2, [0.0, 5.0], rtol=1e-6)

    demand = np.concatenate([dem_w1, dem_w2])
    counts = np.zeros(4)

    # Pure priority: the slow absorber leads, the moderate row second, the
    # fast absorber third; the converged row gets nothing.
    quota = stratified_quota(priority_weights(demand, counts, 0.0), 8)
    np.testing.assert_array_equal(quota, [6, 0, 0, 2])  # exact [6.15, .31, 0, 1.54]
    assert quota.sum() == 8

    # Coverage rarity rescues the starved converged row.
    quota_cov = stratified_quota(priority_weights(demand, counts, 0.8), 8)
    assert quota_cov[2] >= 1

    # Direction check vs the falsified baseline: marginal_absorption ranks the
    # low-mu_eff row first, while proportional_G ranks by gap magnitude —
    # the E19/E20 correction is an ordering inversion, not a rescaling.
    dem_g = row_demand(
        position_demand(_fields(mask_w1, state_gap=[[0.01, 0.01], [1.0, 1.0]]), "proportional_G_negcontrol"),
        mask_w1,
    )
    quota_g = stratified_quota(priority_weights(dem_g, counts[:2], 0.0), 8)
    assert quota_g[0] < quota_g[1]  # ∝G starves the small-gap row
    assert dem_w1[0] > dem_w1[1]  # 1/mu_eff feeds it the most


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
