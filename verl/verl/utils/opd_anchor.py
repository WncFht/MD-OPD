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

import torch

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
    """
    if anchor_logits.shape != student_logits.shape:
        raise ValueError(
            f"anchor and student logits must have identical shape, got {tuple(anchor_logits.shape)} "
            f"and {tuple(student_logits.shape)}"
        )
    parts = []
    for start in range(0, anchor_logits.shape[0], top_k_logits_chunk_rows(anchor_logits)):
        end = min(start + top_k_logits_chunk_rows(anchor_logits), anchor_logits.shape[0])
        b_logits = anchor_logits[start:end].float() / anchor_temperature
        b_probs = torch.softmax(b_logits, dim=-1)
        z_chunk = student_logits[start:end].float()
        parts.append((b_probs * z_chunk).sum(dim=-1) - torch.logsumexp(z_chunk, dim=-1))
    return torch.cat(parts, dim=0)


def compute_anchor_state_terms(
    anchor_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    responses: torch.Tensor,
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

    All expectations use the ``<b, z> - lse(z)`` identity under chunking, so no
    full ``(B, T, V)`` float32 tensor ever exists. CPU-testable.
    """
    if anchor_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"anchor and teacher logits must have identical shape, got {tuple(anchor_logits.shape)} "
            f"and {tuple(teacher_logits.shape)}"
        )
    bsz, seqlen = anchor_logits.shape[:2]
    z_b = anchor_logits.reshape(bsz * seqlen, -1)
    z_t = teacher_logits.reshape(bsz * seqlen, -1)
    y = responses.reshape(-1)

    anchor_log_probs = torch.empty(bsz * seqlen, dtype=torch.float32, device=z_b.device)
    state_anchor_logb = torch.empty(bsz * seqlen, dtype=torch.float32, device=z_b.device)
    state_gap = torch.empty(bsz * seqlen, dtype=torch.float32, device=z_b.device)

    chunk_rows = top_k_logits_chunk_rows(z_b)
    for start in range(0, z_b.shape[0], chunk_rows):
        end = min(start + chunk_rows, z_b.shape[0])
        z_b_chunk = z_b[start:end].float()
        z_t_chunk = z_t[start:end].float()
        b_probs = torch.softmax(z_b_chunk, dim=-1)
        lse_b = torch.logsumexp(z_b_chunk, dim=-1)
        lse_t = torch.logsumexp(z_t_chunk, dim=-1)

        eb_logb = (b_probs * z_b_chunk).sum(dim=-1) - lse_b
        eb_logt = (b_probs * z_t_chunk).sum(dim=-1) - lse_t
        anchor_log_probs[start:end] = z_b_chunk.gather(-1, y[start:end].unsqueeze(-1)).squeeze(-1) - lse_b
        state_anchor_logb[start:end] = eb_logb
        state_gap[start:end] = eb_logb - eb_logt

    return {
        "anchor_log_probs_fwd": anchor_log_probs.view(bsz, seqlen),
        "state_anchor_logb": state_anchor_logb.view(bsz, seqlen),
        "state_gap": state_gap.view(bsz, seqlen),
    }
