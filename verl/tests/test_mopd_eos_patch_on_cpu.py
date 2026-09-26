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
"""Unit test for the MOPD EOS semantic-class patch (paper M4).

Numpy-only by design: exercises ``verl/utils/mopd_eos.py`` — class id
validation, ``log p(E)`` fields, sampled-token merge, merged-space
expectations, and the merged residual energy the anchor chunk computes —
against brute-force computations on the explicitly collapsed action space.
Loads the module by file path so it runs without torch:

    uv run --with numpy python3 verl/tests/test_mopd_eos_patch_on_cpu.py

Also pytest-compatible (``pytest verl/tests/test_mopd_eos_patch_on_cpu.py``).
"""

import importlib.util
from pathlib import Path

import numpy as np

_MODULE_PATH = Path(__file__).resolve().parent.parent / "verl" / "utils" / "mopd_eos.py"
_spec = importlib.util.spec_from_file_location("mopd_eos", _MODULE_PATH)
mopd_eos = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mopd_eos)

eos_class_lse = mopd_eos.eos_class_lse
eos_class_log_probs = mopd_eos.eos_class_log_probs
eos_member_log_probs = mopd_eos.eos_member_log_probs
merge_sampled_log_probs = mopd_eos.merge_sampled_log_probs
merged_class_dot = mopd_eos.merged_class_dot
merged_expectation = mopd_eos.merged_expectation
semantic_eos_mask = mopd_eos.semantic_eos_mask
validate_eos_class_ids = mopd_eos.validate_eos_class_ids

RNG = np.random.default_rng(20260927)
V = 50
EOS = [3, 17, 41]


def _logits(*shape):
    return RNG.normal(0.0, 3.0, size=(*shape, V))


def _lse(z):
    m = z.max(axis=-1)
    return m + np.log(np.exp(z - np.expand_dims(m, -1)).sum(-1))


def _probs(z):
    return np.exp(z - np.expand_dims(_lse(z), -1))


def _merged_coords(vocab_size, eos_ids):
    """Non-class coordinates of the merged action space, in vocab order."""
    keep = np.ones(vocab_size, dtype=bool)
    keep[np.asarray(eos_ids)] = False
    return np.nonzero(keep)[0]


def _merged_probs(p, eos_ids):
    """Collapse p over V into the merged space: members of E -> one mass."""
    keep = _merged_coords(p.shape[-1], eos_ids)
    return np.concatenate([p[..., keep], p[..., list(eos_ids)].sum(-1, keepdims=True)], axis=-1)


def test_validate_eos_class_ids():
    assert validate_eos_class_ids([151643, 151645]) == [151643, 151645]
    assert validate_eos_class_ids((7,)) == [7]
    assert validate_eos_class_ids(np.array([1, 2])) == [1, 2]
    for bad in (None, [], [5, 5], [1, 2, 1]):
        try:
            validate_eos_class_ids(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid class ids {bad!r} must raise ValueError")


def test_semantic_eos_mask():
    sampled = np.array([[3, 0, 17], [41, 2, 3]])
    mask = semantic_eos_mask(sampled, EOS)
    np.testing.assert_array_equal(mask, [[True, False, True], [True, False, True]])
    assert mask.dtype == np.bool_


def test_eos_class_lse_and_log_probs():
    z = _logits(4, 6)
    lse = _lse(z)
    np.testing.assert_allclose(eos_class_lse(z, EOS), _lse(z[..., EOS]), rtol=2e-7)

    p = _probs(z)
    expected = np.log(p[..., EOS].sum(-1))
    np.testing.assert_allclose(eos_class_log_probs(z, EOS), expected, rtol=2e-6, atol=1e-7)
    # Precomputed vocab lse gives the identical result.
    np.testing.assert_allclose(eos_class_log_probs(z, EOS, lse=lse), expected, rtol=2e-6, atol=1e-7)


def test_eos_member_log_probs():
    z = _logits(2, 3)
    member_lp = eos_member_log_probs(z, EOS)
    assert member_lp.shape == (2, 3, len(EOS))
    # Per-member log probs; their mass sums to the class mass.
    np.testing.assert_allclose(member_lp, z[..., EOS] - np.expand_dims(_lse(z), -1), rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(
        np.log(np.exp(member_lp).sum(-1)), eos_class_log_probs(z, EOS), rtol=2e-6, atol=1e-7
    )


def test_merge_sampled_log_probs():
    z = _logits(2, 4)
    p = _probs(z)
    sampled = np.array([[0, 3, 17, 1], [41, 2, 0, 3]])
    raw = np.log(np.take_along_axis(p, sampled[..., None], axis=-1)[..., 0]).astype(np.float32)
    merged = merge_sampled_log_probs(z, sampled, raw, EOS)

    mask = semantic_eos_mask(sampled, EOS)
    class_lp = eos_class_log_probs(z, EOS)
    np.testing.assert_array_equal(merged[~mask], raw[~mask])  # untouched, bitwise
    np.testing.assert_allclose(merged[mask], np.broadcast_to(class_lp, merged.shape)[mask], rtol=2e-6)

    # Class mass is what the sampled-token reward should see at terminal
    # positions: >= each member's own log prob.
    member_lp = eos_member_log_probs(z, EOS)
    for k, e in enumerate(EOS):
        at_e = sampled == e
        if at_e.any():
            assert (merged[at_e] >= member_lp[..., k][at_e]).all()


def test_merge_single_member_and_boundary_ids():
    # Single-member class: merge is the identity at every position.
    z = _logits(V)  # V positions; position v samples token v
    p = _probs(z)
    sampled = np.arange(V)
    raw = np.log(np.take_along_axis(p, sampled[:, None], axis=-1)[:, 0]).astype(np.float32)
    merged1 = merge_sampled_log_probs(z, sampled, raw, [7])
    np.testing.assert_allclose(merged1, raw, rtol=2e-6, atol=1e-7)

    # Boundary ids (first / last vocab index) merge like any other.
    eos = [0, V - 1]
    z2 = _logits(3)
    p2 = _probs(z2)
    sampled2 = np.array([0, V - 1, 5])
    raw3 = np.log(np.take_along_axis(p2, sampled2[:, None], axis=-1)[:, 0]).astype(np.float32)
    merged3 = merge_sampled_log_probs(z2, sampled2, raw3, eos)
    expected_cls = np.log(p2[:, 0] + p2[:, V - 1])
    np.testing.assert_allclose(merged3[:2], expected_cls[:2], rtol=2e-6)
    assert merged3[2] == raw3[2]


def test_merged_class_dot():
    b = _probs(_logits(4))
    x = RNG.normal(0.0, 2.0, size=(4, V))
    x_class = RNG.normal(0.0, 2.0, size=4)
    got = merged_class_dot(b, x, EOS, x_class)

    keep = _merged_coords(V, EOS)
    b_m = _merged_probs(b, EOS)
    x_m = np.concatenate([x[..., keep], x_class[..., None]], axis=-1)
    np.testing.assert_allclose(got, (b_m * x_m).sum(-1), rtol=2e-10, atol=1e-10)


def test_merged_expectation_matches_merged_space():
    z_b = _logits(5)
    z_p = _logits(5)
    b = _probs(z_b)
    got = merged_expectation(b, z_p, _lse(z_p), EOS)

    # Brute force: E_b[log p] on the collapsed space.
    b_m = _merged_probs(b, EOS)
    logp_m = np.log(_merged_probs(_probs(z_p), EOS))
    np.testing.assert_allclose(got, (b_m * logp_m).sum(-1), rtol=2e-6, atol=1e-6)

    # KL(b'||p') = E_b[log b]_m - E_b[log p]_m >= 0, and = 0 iff b' = p'.
    kl = merged_expectation(b, z_b, _lse(z_b), EOS) - got
    assert (kl >= -1e-6).all()
    kl_same = merged_expectation(b, z_b, _lse(z_b), EOS) - merged_expectation(
        b, z_b, _lse(z_b), EOS
    )
    np.testing.assert_allclose(kl_same, 0.0, atol=1e-10)


def test_merged_residual_energy():
    # Mirror the anchor chunk's merged branch, then compare against the
    # residual computed directly on the collapsed action space.
    z_b, z_t = _logits(6), _logits(6)
    b = _probs(z_b)
    lse_b_E = _lse(z_b[..., EOS])
    lse_t_E = _lse(z_t[..., EOS])
    x = z_b - z_t  # member coords: raw logit gap
    x_E = lse_b_E - lse_t_E  # class coord: raw-lse convention

    eb_x = merged_class_dot(b, x, EOS, x_E)
    d = eb_x[..., None] - x
    d_E = eb_x - x_E
    energy = merged_class_dot(b, d * d, EOS, d_E * d_E)
    mu_num = merged_class_dot(b * b, d * d, EOS, d_E * d_E)  # b_E^2 handled below
    # merged_class_dot uses sum b_e^2 for the class weight; the chunk uses
    # b_E^2. Fix the class coordinate explicitly for the check.
    b_E = b[..., EOS].sum(-1)
    mu_num = mu_num - (b[..., EOS] ** 2).sum(-1) * d_E**2 + b_E**2 * d_E**2

    # Direct merged-space residual: x' = log b' - log pT' (shifted version of
    # x; the shift cancels in d' = <b,x'>_m - x').
    keep = _merged_coords(V, EOS)
    b_m = _merged_probs(b, EOS)
    t_m = _merged_probs(_probs(z_t), EOS)
    x_m = np.log(b_m) - np.log(t_m)
    eb_x_m = (b_m * x_m).sum(-1)
    d_m = eb_x_m[..., None] - x_m
    energy_m = (b_m * d_m * d_m).sum(-1)
    mu_num_m = (b_m * b_m * d_m * d_m).sum(-1)

    np.testing.assert_allclose(energy, energy_m, rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(mu_num, mu_num_m, rtol=2e-6, atol=1e-6)
    # Non-class residuals match the merged-space ones coordinatewise.
    np.testing.assert_allclose(d[..., keep], d_m[..., : keep.size], rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(d_E, d_m[..., -1], rtol=2e-6, atol=1e-6)
    # Residual energy is non-negative and vanishes when b' = pT'.
    assert (energy >= -1e-7).all()
    same = merged_class_dot(b, (x - x) ** 2, EOS, np.zeros(6))
    np.testing.assert_allclose(same, 0.0, atol=1e-12)


def test_merge_idempotent_on_class_positions():
    # Sampling e2 from the student is impossible pre-patch; post-patch the
    # field only ever sees the merged value once — merging a second time with
    # the merged field as 'raw' changes nothing at class positions.
    z = _logits(2, 2)
    sampled = np.array([[EOS[0], 1], [EOS[1], 2]])
    raw = np.zeros_like(sampled, dtype=np.float32)  # degenerate raw field
    m1 = merge_sampled_log_probs(z, sampled, raw, EOS)
    m2 = merge_sampled_log_probs(z, sampled, m1, EOS)
    np.testing.assert_allclose(m1, m2, rtol=0, atol=0)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
