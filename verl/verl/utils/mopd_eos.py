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
"""Terminal-token semantic-class merge for OPD/MOPD (paper M4, EOS patch).

A base student terminates on one token id ``e1`` (e.g. ``<|endoftext|>``)
while a post-trained teacher terminates on another, ``e2`` (e.g.
``<|im_end|>``). Without the patch the teacher's terminal mass lands on a
token the student never samples, so the reverse-KL reward can never teach the
student to stop: sampled-token rewards see ``log p_T(y_t)`` only at positions
the student actually emitted. The patch collapses both ids into one semantic
class ``E = {e1, e2}`` and evaluates every log-prob field on the *merged*
action space:

* rollout: generation stops on any member of ``E`` (see
  ``rollout.extra_stop_token_ids`` -> ``SamplingParams.stop_token_ids``).
* sampled-token fields: at positions where ``y_t in E``, ``log p(y_t)`` is
  replaced by ``log p(E) = logsumexp(z_E) - logsumexp(z)`` for the teacher,
  the student policy (old_log_probs *and* the update forward), and the
  anchor's ``log b(y_t)``.
* per-state terms of the MOPD residual: ``E_b[f]`` becomes
  ``sum_{v not in E} b_v f_v + b_E * f_E`` where ``f_E`` is the
  ``logsumexp``-merged coordinate — applied to ``KL(b||pT)``, ``E_b[log b]``,
  ``E_b[log pi_theta]`` and the residual energy ``E_b[d^2]`` /
  ``E_b[b d^2]`` with merged center ``<b,x>_m`` and merged class residual
  ``d_E = <b,x>_m - x_E``.

This mirrors ``opd-eos``'s ``semantic_class`` mode (``map_teacher_sampled_
log_probs`` / ``student_policy_log_probs`` in the reference fork), extended
to the anchor state terms that reference does not have. ``teacher_map`` /
``canonical`` are deliberately not ported: the paper runs M4 in class mode.

Everything here is dual-backend: torch tensors run the torch ops (autograd
flows through the student logits for ``anchor_eb_logp``), numpy arrays run
the same math for CPU tests, so ``verl/tests/test_mopd_eos_patch_on_cpu.py``
exercises the shipped functions directly. Torch is never imported at module
level; ``verl.utils.topk_logits`` is only needed (lazily) to bound the FP32
chunk width on the torch path.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "eos_class_lse",
    "eos_class_log_probs",
    "eos_member_log_probs",
    "merge_sampled_log_probs",
    "merged_class_dot",
    "merged_expectation",
    "semantic_eos_mask",
    "validate_eos_class_ids",
]


def _is_torch(x) -> bool:
    return type(x).__module__.split(".")[0] == "torch"


def _as_id_index(like, token_ids: Sequence[int]):
    """Class ids as a last-dim index tensor/array matching ``like``'s backend."""
    ids = [int(i) for i in token_ids]
    if _is_torch(like):
        import torch

        return torch.as_tensor(ids, dtype=torch.long, device=like.device)
    return np.asarray(ids, dtype=np.int64)


def _lse_lastdim(x):
    """logsumexp over the last dim; float32/float64 accumulation."""
    if _is_torch(x):
        return x.logsumexp(-1)
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=-1)
    return m + np.log(np.exp(x - np.expand_dims(m, axis=-1)).sum(axis=-1))


def _where(cond, a, b):
    if _is_torch(a) or _is_torch(cond) or _is_torch(b):
        import torch

        return torch.where(cond, a, b)
    return np.where(cond, a, b)


def _flat_chunk_rows(x):
    """Row budget for FP32 (N, V) temporaries on the torch path."""
    from verl.utils.topk_logits import top_k_logits_chunk_rows

    return top_k_logits_chunk_rows(x)


def validate_eos_class_ids(token_ids) -> list[int]:
    """Normalize the merged terminal class; must be non-empty distinct ints."""
    ids = [int(i) for i in token_ids] if token_ids is not None else []
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(
            "[MOPD-EOS] semantic_eos_token_ids must be a non-empty list of distinct token ids, "
            f"got {token_ids!r}"
        )
    return ids


def semantic_eos_mask(token_ids, eos_ids: Sequence[int]):
    """Bool ``(...)`` mask: True where the sampled token is a class member."""
    idx = _as_id_index(token_ids, eos_ids)
    return (token_ids.unsqueeze(-1) == idx).any(-1) if _is_torch(token_ids) else (
        np.expand_dims(token_ids, axis=-1) == idx
    ).any(axis=-1)


def eos_class_lse(logits, eos_ids: Sequence[int]):
    """``logsumexp(z_E)`` per leading position, FP32.

    The vocab gather is the only vocab-indexed op; logits stay in their
    original dtype. This is *not* normalized — subtract the vocab
    logsumexp for ``log p(E)``.
    """
    ids = validate_eos_class_ids(eos_ids)
    idx = _as_id_index(logits, ids)
    if _is_torch(logits):
        return logits.index_select(-1, idx).float().logsumexp(-1)
    z = np.asarray(logits, dtype=np.float64)
    return _lse_lastdim(z[..., np.asarray(ids)]).astype(np.float32)


def eos_class_log_probs(logits, eos_ids: Sequence[int], lse=None):
    """``log p(E) = logsumexp(z_E) - logsumexp(z)`` per leading position, FP32.

    ``lse`` is the precomputed ``logsumexp(z)`` when the caller already has it
    (the anchor state-terms chunk does); otherwise it is computed here, in
    FP32 chunks on the torch path so no full ``(N, V)`` FP32 tensor exists.
    """
    ids = validate_eos_class_ids(eos_ids)
    if _is_torch(logits):
        import torch

        idx = _as_id_index(logits, ids)
        flat = logits.reshape(-1, logits.shape[-1])
        class_lse = torch.empty(flat.shape[0], dtype=torch.float32, device=logits.device)
        vocab_lse = None if lse is not None else torch.empty_like(class_lse)
        for start in range(0, flat.shape[0], _flat_chunk_rows(flat)):
            end = min(start + _flat_chunk_rows(flat), flat.shape[0])
            chunk = flat[start:end].float()
            class_lse[start:end] = chunk.index_select(-1, idx).logsumexp(-1)
            if vocab_lse is not None:
                vocab_lse[start:end] = chunk.logsumexp(-1)
        if lse is not None:
            vocab_lse = lse.reshape(-1).float()
        return (class_lse - vocab_lse).view(*logits.shape[:-1])

    z = np.asarray(logits, dtype=np.float64)
    idx = np.asarray(ids, dtype=np.int64)
    vocab_lse = _lse_lastdim(z) if lse is None else np.asarray(lse, dtype=np.float64)
    return (_lse_lastdim(z[..., idx]) - vocab_lse).astype(np.float32)


def eos_member_log_probs(logits, eos_ids: Sequence[int], lse=None):
    """``log p(v)`` for each ``v in E``: ``(..., K)`` FP32 diagnostic field.

    Emitted as ``teacher_eos_log_probs`` / ``anchor_eos_log_probs`` so the
    per-member split of the class mass is visible in logs — the pathology M4
    fixes is exactly that split being ~0 on the teacher's native terminal id.
    """
    ids = validate_eos_class_ids(eos_ids)
    if _is_torch(logits):
        import torch

        idx = _as_id_index(logits, ids)
        flat = logits.reshape(-1, logits.shape[-1])
        out = torch.empty(flat.shape[0], len(ids), dtype=torch.float32, device=logits.device)
        if lse is not None:
            lse = lse.reshape(-1).float()
        for start in range(0, flat.shape[0], _flat_chunk_rows(flat)):
            end = min(start + _flat_chunk_rows(flat), flat.shape[0])
            chunk = flat[start:end].float()
            denom = chunk.logsumexp(-1, keepdim=True) if lse is None else lse[start:end].unsqueeze(-1)
            out[start:end] = chunk.index_select(-1, idx) - denom
        return out.view(*logits.shape[:-1], len(ids))

    z = np.asarray(logits, dtype=np.float64)
    idx = np.asarray(ids, dtype=np.int64)
    denom = _lse_lastdim(z) if lse is None else np.asarray(lse, dtype=np.float64)
    return (z[..., idx] - np.expand_dims(denom, axis=-1)).astype(np.float32)


def merge_sampled_log_probs(logits, sampled_ids, raw_log_probs, eos_ids: Sequence[int]):
    """Semantic-class merge of a sampled-token log-prob field.

    At positions where ``sampled_ids in E`` returns ``log p(E)``; every other
    position keeps ``raw_log_probs``. Identical math for the teacher's
    ``teacher_log_probs``, the student's ``log_probs`` (old_log_probs and the
    update forward share the call site), and the anchor's ``log b(y_t)``.
    """
    class_lp = eos_class_log_probs(logits, eos_ids).reshape(-1)
    mask = semantic_eos_mask(sampled_ids.reshape(-1), eos_ids)
    if _is_torch(class_lp):
        raw_flat = raw_log_probs.reshape(-1).to(class_lp.dtype)
    else:
        raw_flat = np.asarray(raw_log_probs, dtype=np.float32).reshape(-1)
    return _where(mask, class_lp, raw_flat).reshape(raw_log_probs.shape)


def merged_class_dot(b_probs, x, eos_ids: Sequence[int], x_class):
    """``sum_{v not in E} b_v * x_v + b_E * x_class`` per leading position.

    This is the merged-coordinate version of ``<b, x>``: member coordinates
    of ``E`` are dropped from the weighted sum and replaced by the class mass
    ``b_E`` times the merged value ``x_class`` (e.g. ``logsumexp(z_E)`` for
    log-prob expectations, ``x_E = log b_E - log p_E`` for residual centers).

    Shapes: ``b_probs``, ``x`` are ``(..., V)`` FP32; ``x_class`` is ``(...)``.
    """
    idx = _as_id_index(b_probs, eos_ids)
    if _is_torch(b_probs):
        b_e = b_probs.index_select(-1, idx)
        x_e = x.index_select(-1, idx)
        return (b_probs * x).sum(-1) - (b_e * x_e).sum(-1) + b_e.sum(-1) * x_class
    b = np.asarray(b_probs, dtype=np.float64)
    xv = np.asarray(x, dtype=np.float64)
    i = np.asarray([int(v) for v in eos_ids], dtype=np.int64)
    b_e = b[..., i]
    return (b * xv).sum(-1) - (b_e * xv[..., i]).sum(-1) + b_e.sum(-1) * np.asarray(x_class, dtype=np.float64)


def merged_expectation(b_probs, z, lse_z, eos_ids: Sequence[int]):
    """Merged ``E_b[log p]`` where ``log p = z - lse_z`` per leading position.

    ``E_b[log p]_merged = <b, z>_m - lse_z`` with the class coordinate
    contributing ``b_E * logsumexp(z_E)`` — the form ``anchor_eb_logp``,
    ``state_anchor_logb`` and ``E_b[log pT]`` all take under the patch.
    """
    return merged_class_dot(b_probs, z, eos_ids, eos_class_lse(z, eos_ids)) - lse_z
