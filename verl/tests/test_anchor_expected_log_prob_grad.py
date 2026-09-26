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
"""Grad-parity test for ``anchor_expected_log_prob`` (B5 diff-review F2).

The custom ``AnchorExpectedLogProbFn`` recomputes softmaxes in backward
instead of letting autograd retain per-chunk fp32 (N, V) temporaries. This
test asserts value+gradient parity against the naive autodiff formula it
replaces — unmerged and merged (EOS class) paths, plus a non-unit anchor
temperature — on CPU, small vocab:

    python3 verl/tests/test_anchor_expected_log_prob_grad.py

pytest-compatible; skipped when torch is unavailable (the module needs the
real package import for ``verl.utils.mopd_eos``).
"""

import numpy as np

try:
    import torch

    from verl.utils.opd_anchor import anchor_expected_log_prob
except ImportError:  # pragma: no cover - torch-free machines (Mac) skip
    torch = None

try:
    import pytest

    pytestmark = pytest.mark.skipif(torch is None, reason="torch unavailable")
except ImportError:
    pass


def naive_forward(anchor_logits, student_logits, anchor_temperature, eos_ids):
    """The pre-F2 chunked formula, written dense so autograd can tape it."""
    b = torch.softmax(anchor_logits.float() / anchor_temperature, dim=-1)
    z = student_logits.float()
    lse_z = torch.logsumexp(z, dim=-1)
    if not eos_ids:
        return (b * z).sum(-1) - lse_z
    idx = torch.as_tensor(list(eos_ids), device=z.device, dtype=torch.long)
    z_e = z.index_select(-1, idx)
    b_e = b.index_select(-1, idx)
    return (b * z).sum(-1) - (b_e * z_e).sum(-1) + b_e.sum(-1) * torch.logsumexp(z_e, -1) - lse_z


def check_parity(eos_ids, anchor_temperature=1.0, n=37, v=53, seed=0):
    g = torch.Generator().manual_seed(seed)
    anchor = torch.randn(n, v, generator=g)
    student_ref = torch.randn(n, v, generator=g)
    student_fn = student_ref.clone().requires_grad_(True)
    student_naive = student_ref.clone().requires_grad_(True)

    out_fn = anchor_expected_log_prob(anchor, student_fn, anchor_temperature, eos_ids)
    out_naive = naive_forward(anchor, student_naive, anchor_temperature, eos_ids)
    np.testing.assert_allclose(out_fn.detach().numpy(), out_naive.detach().numpy(), rtol=1e-6, atol=1e-6)

    weights = torch.randn(n, generator=g)
    grad_fn = torch.autograd.grad((out_fn * weights).sum(), student_fn)[0]
    grad_naive = torch.autograd.grad((out_naive * weights).sum(), student_naive)[0]
    np.testing.assert_allclose(grad_fn.numpy(), grad_naive.numpy(), rtol=1e-5, atol=1e-6)


def test_unmerged_grad_parity():
    check_parity(eos_ids=None)


def test_merged_grad_parity():
    check_parity(eos_ids=[3, 17, 41])


def test_anchor_temperature_grad_parity():
    check_parity(eos_ids=[3, 17, 41], anchor_temperature=1.5)


if __name__ == "__main__":
    if torch is None:
        print("torch unavailable -- skipping")
        raise SystemExit(0)
    test_unmerged_grad_parity()
    test_merged_grad_parity()
    test_anchor_temperature_grad_parity()
    print("3/3 grad-parity checks passed")
