"""Low-rank spectral acceleration for OPD.

This module is deliberately trainer-agnostic.  It implements the two pieces
needed for a first prototype:

* checkpoint-level modal extrapolation, which modifies a checkpoint like
  EffOPD but assigns a separate multiplier to each learned direction;
* optimizer-level modal stepping, which applies a damped finite-time
  gradient-flow step in a low-rank subspace and leaves the spectral tail to
  the ordinary optimizer.

All vectors are represented as flattened torch tensors.  A caller integrating
this with a sharded model should flatten/unflatten parameters outside this
module and perform the SVD/HVP computations on an appropriate local or global
representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

Tensor = torch.Tensor


@dataclass
class SpectralConfig:
    """Controls the conservative defaults for spectral acceleration."""

    rank: int = 8
    tau: float = 1.0
    damping: float = 1e-8
    min_gain: float = 0.5
    max_gain: float = 8.0
    tail_lr: float = 1e-6


def orthonormal_update_basis(
    deltas: Tensor,
    rank: int,
) -> tuple[Tensor, Tensor]:
    """Return a low-rank basis from checkpoint displacements.

    Args:
        deltas: Matrix with shape ``[num_checkpoints, num_parameters]``.
        rank: Number of right-singular directions to retain.

    Returns:
        ``(basis, singular_values)`` where basis has shape
        ``[num_parameters, rank_kept]`` and orthonormal columns.
    """
    if deltas.ndim != 2:
        raise ValueError("deltas must have shape [num_checkpoints, num_parameters]")
    if rank < 1:
        raise ValueError("rank must be positive")
    _, singular_values, vh = torch.linalg.svd(deltas, full_matrices=False)
    kept = min(rank, vh.shape[0])
    return vh[:kept].transpose(0, 1).contiguous(), singular_values[:kept]


def rayleigh_curvatures(
    basis: Tensor,
    hvp: Callable[[Tensor], Tensor],
    *,
    floor: float = 0.0,
) -> Tensor:
    """Estimate ``u_i^T A u_i`` for each basis vector using an HVP callback."""
    if basis.ndim != 2:
        raise ValueError("basis must have shape [num_parameters, rank]")
    values = []
    for i in range(basis.shape[1]):
        u = basis[:, i]
        au = hvp(u)
        values.append(torch.dot(u, au) / torch.dot(u, u).clamp_min(1e-30))
    return torch.stack(values).clamp_min(floor)


def finite_time_gains(
    curvatures: Tensor,
    *,
    tau: float,
    damping: float = 0.0,
    min_gain: Optional[float] = None,
    max_gain: Optional[float] = None,
    reference_lr: Optional[float] = None,
) -> Tensor:
    """Compute modal gains for a finite-time gradient-flow step.

    The unscaled modal coefficient is

        (1 - exp(-tau * lambda)) / (lambda + damping).

    If ``reference_lr`` is supplied, return a multiplier relative to an
    ordinary ``reference_lr`` gradient step.  This is useful for comparing
    directly with a checkpoint displacement or an Adam update direction.
    """
    lam = curvatures.to(dtype=torch.float32).clamp_min(0.0)
    denom = lam + float(damping)
    raw = torch.where(
        denom > 0,
        -torch.expm1(-float(tau) * lam) / denom,
        torch.full_like(lam, float(tau)),
    )
    if reference_lr is not None:
        raw = raw / float(reference_lr)
    if min_gain is not None or max_gain is not None:
        raw = raw.clamp(
            min=min_gain if min_gain is not None else -float("inf"),
            max=max_gain if max_gain is not None else float("inf"),
        )
    return raw


def checkpoint_modal_extrapolation(
    current: Tensor,
    displacement: Tensor,
    basis: Tensor,
    curvatures: Tensor,
    *,
    tau: float,
    reference_lr: float,
    damping: float = 0.0,
    min_gain: float = 0.5,
    max_gain: float = 8.0,
    tail_gain: float = 1.0,
) -> Tensor:
    """Apply a checkpoint-level, direction-dependent extrapolation.

    ``displacement`` is the recent checkpoint difference (the analogue of
    EffOPD's ``Delta_n``).  The head is projected onto ``basis`` and each
    coefficient receives its own finite-time-flow gain.  The orthogonal tail
    is retained with ``tail_gain``.  No optimizer state is touched.
    """
    if current.ndim != 1 or displacement.shape != current.shape:
        raise ValueError("current and displacement must be flat vectors of equal shape")
    if basis.shape[0] != current.numel():
        raise ValueError("basis has incompatible parameter dimension")
    if curvatures.numel() != basis.shape[1]:
        raise ValueError("one curvature is required per basis vector")

    coeff = basis.transpose(0, 1) @ displacement
    gains = finite_time_gains(
        curvatures,
        tau=tau,
        damping=damping,
        min_gain=min_gain,
        max_gain=max_gain,
        reference_lr=reference_lr,
    ).to(displacement)
    head = basis @ (gains * coeff)
    tail = displacement - basis @ coeff
    return current + head + float(tail_gain) * tail


def modal_gradient_step(
    params: Tensor,
    grad: Tensor,
    basis: Tensor,
    curvatures: Tensor,
    *,
    config: SpectralConfig | None = None,
) -> Tensor:
    """Apply a low-rank spectral optimizer step.

    The principal subspace receives

        -(1 - exp(-tau*lambda)) / (lambda + damping) * grad_i,

    while the orthogonal complement receives an ordinary ``-tail_lr * grad``
    step.  This is a functional update; callers that use Adam should apply it
    instead of, or in a separately ablated branch from, the Adam parameter
    update.  It does not update Adam moments.
    """
    if config is None:
        config = SpectralConfig()
    if params.ndim != 1 or grad.shape != params.shape:
        raise ValueError("params and grad must be flat vectors of equal shape")
    if basis.shape[0] != params.numel():
        raise ValueError("basis has incompatible parameter dimension")
    if curvatures.numel() != basis.shape[1]:
        raise ValueError("one curvature is required per basis vector")

    # Ensure numerical orthogonality is not amplified by projection roundoff.
    u = basis.to(dtype=grad.dtype, device=grad.device)
    head_grad = u @ (u.transpose(0, 1) @ grad)
    tail_grad = grad - head_grad
    modal_step = finite_time_gains(
        curvatures.to(device=grad.device),
        tau=config.tau,
        damping=config.damping,
    ).to(dtype=grad.dtype, device=grad.device)
    head_step = -u @ (modal_step * (u.transpose(0, 1) @ grad))
    tail_step = -float(config.tail_lr) * tail_grad
    return params + head_step + tail_step


__all__ = [
    "SpectralConfig",
    "orthonormal_update_basis",
    "rayleigh_curvatures",
    "finite_time_gains",
    "checkpoint_modal_extrapolation",
    "modal_gradient_step",
]
