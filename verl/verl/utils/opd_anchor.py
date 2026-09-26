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
"""OPD anchor (frozen generator snapshot b) helpers.

Two things live here, and both are needed by more than one worker:

1. A per-process registry for the anchor and actor FSDP modules. The OPD
   teacher worker and the actor worker are colocated inside the same Ray
   WorkerDict process (``create_colocated_worker_cls`` in
   ``verl.single_controller.ray.base``), so in-process module handles can be
   shared directly. The actor worker publishes its module so the teacher worker
   can re-snapshot it into the anchor at window boundaries; the teacher worker
   publishes the anchor so the actor's own engine can run it inside its forward
   pass, where the full-vocabulary student logits are materialized.
2. Memory-bounded pure-tensor math for the per-state quantities of the MOPD
   residual ``d_t = log pT(y_t) - log pS(y_t) + KL(b||pT)(s_t) - KL(b||pS)(s_t)``.
   All expectations over b are computed with the ``<b, z> - logsumexp(z)``
   identity, which avoids materializing a second full-vocabulary log-softmax:
   only per-chunk FP32 temporaries bounded by
   :func:`~verl.utils.topk_logits.top_k_logits_chunk_rows` are allocated.

Nothing here imports verl workers or engines, so this module stays a leaf that
both sides can import without cycles.
"""

from collections.abc import Sequence

import torch

from verl.utils.mopd_eos import (
    merged_class_dot,
    merged_expectation,
    semantic_eos_mask,
    validate_eos_class_ids,
)
from verl.utils.topk_logits import top_k_logits_chunk_rows

__all__ = [
    "anchor_expected_log_prob",
    "compute_anchor_state_terms",
    "get_opd_actor_module",
    "get_opd_anchor_module",
    "set_opd_actor_module",
    "set_opd_anchor_module",
]

# Per-process module handles. Each Ray worker process hosts exactly one fused
# WorkerDict, so a single anchor and a single actor module exist per process.
opd_anchor_module = None
opd_actor_module = None


def set_opd_anchor_module(module: torch.nn.Module) -> None:
    """Publish the frozen anchor module built by the teacher worker."""
    global opd_anchor_module
    opd_anchor_module = module


def get_opd_anchor_module() -> torch.nn.Module | None:
    """Return the anchor module visible in this process, or None."""
    return opd_anchor_module


def set_opd_actor_module(module: torch.nn.Module) -> None:
    """Publish the trainable actor module so the anchor can snapshot it."""
    global opd_actor_module
    opd_actor_module = module


def get_opd_actor_module() -> torch.nn.Module | None:
    """Return the actor module visible in this process, or None."""
    return opd_actor_module


def anchor_expected_log_prob(
    anchor_logits: torch.Tensor,
    student_logits: torch.Tensor,
    anchor_temperature: float = 1.0,
    eos_token_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Per-position ``E_b[log pS]`` from paired full-vocab logits.

    ``b`` is the anchor distribution ``softmax(anchor_logits / anchor_temperature)``
    and ``student_logits`` are the already-tempered student logits ``z``. Uses

        E_b[log pS] = <b, z> - logsumexp(z)

    which keeps grad flowing through ``z`` (the backward is ``b - pS``) while
    the anchor side stays detached. Rows are processed in chunks so the FP32
    temporaries stay bounded independent of ``N``; the chunked results are
    concatenated, which preserves autograd. Shapes: ``(N, V)`` -> ``(N,)``,
    float32. CPU-testable.

    When ``eos_token_ids`` names the merged terminal class ``E`` (M4), the
    expectation runs on the merged action space instead:

        E_b[log pS]_merged = sum_{v not in E} b_v z_v + b_E * lse(z_E) - lse(z)

    and the backward spreads the class coordinate's gradient over the member
    logits, ``b_E * pi_v / pi_E`` — exactly what the merged-space residual
    requires of ``anchor_eb_logp``.
    """
    if anchor_logits.shape != student_logits.shape:
        raise ValueError(
            f"anchor and student logits must have identical shape, got {tuple(anchor_logits.shape)} "
            f"and {tuple(student_logits.shape)}"
        )
    eos_ids = validate_eos_class_ids(eos_token_ids) if eos_token_ids else None
    parts = []
    for start in range(0, anchor_logits.shape[0], top_k_logits_chunk_rows(anchor_logits)):
        end = min(start + top_k_logits_chunk_rows(anchor_logits), anchor_logits.shape[0])
        b_logits = anchor_logits[start:end].float() / anchor_temperature
        b_probs = torch.softmax(b_logits, dim=-1)
        z_chunk = student_logits[start:end].float()
        lse_z = torch.logsumexp(z_chunk, dim=-1)
        if eos_ids is not None:
            parts.append(merged_expectation(b_probs, z_chunk, lse_z, eos_ids))
        else:
            parts.append((b_probs * z_chunk).sum(dim=-1) - lse_z)
    return torch.cat(parts, dim=0)


def compute_anchor_state_terms(
    anchor_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    responses: torch.Tensor,
    eos_token_ids: Sequence[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Per-state MOPD anchor terms from paired full-vocab response logits.

    Both logit tensors are ``(B, T, V)`` and already tempered: position ``t``
    holds the distribution over the next token given the prefix ``s_t``.
    ``responses`` is ``(B, T)`` with the sampled token ids ``y_t``.

    Returns float32 tensors of shape ``(B, T)``:

    - ``anchor_log_probs_fwd``: ``log b(y_t | s_t)`` gathered at the sampled token.
    - ``state_anchor_logb``: ``E_b[log b](s_t) = <b, z_b> - lse(z_b) = -H(b|s_t)``,
      the b-only half of ``KL(b || pS)`` at train time.
    - ``state_gap``: ``G(s_t) = KL(b || pT)(s_t) = E_b[log b] - E_b[log pT]``.
    - ``state_resid_energy``: ``E_b[d^2](s_t) = x^T F_b x`` with ``x = z_b - z_t``
      and ``d = <b, x> - x`` — the initial residual energy of the P-C substrate
      (``F_b`` annihilates constants, so raw logits and log-probs give the same
      ``d``; logsumexp terms never enter).
    - ``state_mu_eff``: ``mu_eff(s_t) = x^T F_b^2 x / x^T F_b x = E_b[b d^2] / E_b[d^2]``,
      the marginal absorption rate of the corrected P-C priority measure
      (paper §4.6: ``eta(s) ~ 1/mu_eff(s)``). ``+inf`` where the residual energy
      is exactly zero — a converged state has no absorption rate and the
      downstream demand formula reads ``inf`` as zero demand.

    When ``eos_token_ids`` names the merged terminal class ``E`` (M4), every
    term is evaluated on the merged action space: ``anchor_log_probs_fwd``
    carries ``log b_E`` at positions where ``y_t in E``; the two log-moments
    become ``sum_{v not in E} b_v z_v + b_E * lse(z_E) - lse(z)``; and the
    residual uses the merged center ``<b,x>_m`` with the single class
    coordinate ``d_E = <b,x>_m - x_E`` where ``x_E = lse(z_b_E) - lse(z_t_E)``
    (raw-lse convention — ``F_b`` annihilates the shared normalization shift
    exactly as it does for the member coordinates).

    All expectations use the ``<b, z> - lse(z)`` identity under chunking, so no
    full ``(B, T, V)`` float32 tensor ever exists. CPU-testable.
    """
    if anchor_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"anchor and teacher logits must have identical shape, got {tuple(anchor_logits.shape)} "
            f"and {tuple(teacher_logits.shape)}"
        )
    eos_ids = validate_eos_class_ids(eos_token_ids) if eos_token_ids else None
    eos_idx = None
    if eos_ids is not None:
        eos_idx = torch.as_tensor(eos_ids, dtype=torch.long, device=anchor_logits.device)
    bsz, seqlen = anchor_logits.shape[:2]
    z_b = anchor_logits.reshape(bsz * seqlen, -1)
    z_t = teacher_logits.reshape(bsz * seqlen, -1)
    y = responses.reshape(-1)

    anchor_log_probs = torch.empty(bsz * seqlen, dtype=torch.float32, device=z_b.device)
    state_anchor_logb = torch.empty(bsz * seqlen, dtype=torch.float32, device=z_b.device)
    state_gap = torch.empty(bsz * seqlen, dtype=torch.float32, device=z_b.device)
    state_mu_eff = torch.empty(bsz * seqlen, dtype=torch.float32, device=z_b.device)
    state_resid_energy = torch.empty(bsz * seqlen, dtype=torch.float32, device=z_b.device)
    anchor_eos_log_probs = None
    if eos_idx is not None:
        # Per-member log b(v) for v in E — the class-mass split is the direct
        # diagnostic of the EOS-inflation pathology M4 fixes.
        anchor_eos_log_probs = torch.empty(bsz * seqlen, len(eos_ids), dtype=torch.float32, device=z_b.device)

    chunk_rows = top_k_logits_chunk_rows(z_b)
    for start in range(0, z_b.shape[0], chunk_rows):
        end = min(start + chunk_rows, z_b.shape[0])
        z_b_chunk = z_b[start:end].float()
        z_t_chunk = z_t[start:end].float()
        b_probs = torch.softmax(z_b_chunk, dim=-1)
        lse_b = torch.logsumexp(z_b_chunk, dim=-1)
        lse_t = torch.logsumexp(z_t_chunk, dim=-1)

        if eos_idx is not None:
            # Merged terminal class: per-member lse over E gives the class
            # coordinate's merged value; b_E is the class mass.
            lse_b_E = z_b_chunk.index_select(-1, eos_idx).logsumexp(-1)
            lse_t_E = z_t_chunk.index_select(-1, eos_idx).logsumexp(-1)
            log_b_E = lse_b_E - lse_b
            x_E = lse_b_E - lse_t_E
            b_e_vec = b_probs.index_select(-1, eos_idx)
            b_E = b_e_vec.sum(-1)
            anchor_eos_log_probs[start:end] = z_b_chunk.index_select(-1, eos_idx) - lse_b.unsqueeze(-1)
            eb_logb = merged_expectation(b_probs, z_b_chunk, lse_b, eos_ids)
            eb_logt = merged_expectation(b_probs, z_t_chunk, lse_t, eos_ids)
        else:
            eb_logb = (b_probs * z_b_chunk).sum(dim=-1) - lse_b
            eb_logt = (b_probs * z_t_chunk).sum(dim=-1) - lse_t

        sampled_lp = z_b_chunk.gather(-1, y[start:end].unsqueeze(-1)).squeeze(-1) - lse_b
        if eos_idx is not None:
            anchor_log_probs[start:end] = torch.where(
                semantic_eos_mask(y[start:end], eos_ids), log_b_E, sampled_lp
            )
        else:
            anchor_log_probs[start:end] = sampled_lp
        state_anchor_logb[start:end] = eb_logb
        state_gap[start:end] = eb_logb - eb_logt

        # P-C substrate: d_v = <b, x> - x_v with x = z_b - z_t. The d, d^2,
        # b*d^2 and b^2*d^2 stages all fold in place onto one buffer so the
        # chunk peak stays at one extra (rows, V) fp32 temporary.
        resid = torch.sub(z_b_chunk, z_t_chunk)
        if eos_idx is not None:
            # Merged space: the center is <b,x>_m = sum_{v not in E} b_v x_v +
            # b_E x_E. Member coordinates inside E keep their (meaningless)
            # member-level d_v in the buffer; each energy stage then subtracts
            # its member contribution and adds the single class term.
            eb_x = merged_class_dot(b_probs, resid, eos_ids, x_E).unsqueeze(-1)
        else:
            eb_x = (b_probs * resid).sum(dim=-1, keepdim=True)
        resid.neg_().add_(eb_x)  # d_v = <b, x>_m - x_v
        resid.square_()  # d^2
        if eos_idx is not None:
            d2_e = resid.index_select(-1, eos_idx)  # member-level d_v^2 (to drop)
            d_E = eb_x.squeeze(-1) - x_E  # class residual
        resid.mul_(b_probs)  # b * d^2
        resid_energy = resid.sum(dim=-1)
        if eos_idx is not None:
            resid_energy = resid_energy - (b_e_vec * d2_e).sum(-1) + b_E * d_E.square()
        resid.mul_(b_probs)  # b^2 * d^2
        mu_num = resid.sum(dim=-1)
        if eos_idx is not None:
            mu_num = mu_num - (b_e_vec.square() * d2_e).sum(-1) + b_E.square() * d_E.square()
        state_resid_energy[start:end] = resid_energy
        state_mu_eff[start:end] = torch.where(
            resid_energy > 0,
            mu_num / resid_energy.clamp_min(torch.finfo(torch.float32).tiny),
            torch.inf,
        )

    result = {
        "anchor_log_probs_fwd": anchor_log_probs.view(bsz, seqlen),
        "state_anchor_logb": state_anchor_logb.view(bsz, seqlen),
        "state_gap": state_gap.view(bsz, seqlen),
        "state_mu_eff": state_mu_eff.view(bsz, seqlen),
        "state_resid_energy": state_resid_energy.view(bsz, seqlen),
    }
    if anchor_eos_log_probs is not None:
        result["anchor_eos_log_probs"] = anchor_eos_log_probs.view(bsz, seqlen, -1)
    return result
