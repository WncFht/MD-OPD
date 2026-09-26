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
"""Advantage estimators and policy losses for On-Policy Distillation (OPD).

OPD replaces the usual scalar outcome reward with a *dense, token-level* reward
derived from a teacher model's log-probabilities over the student's own rollout:

    r_t = -(log p_student(y_t) - log p_teacher(y_t))       (reverse KL, k1)

Because that reward is already per-token and already on the right scale, the
advantage is simply the reward itself (masked) -- there is no baseline to
subtract and no discounting to apply. That is ``token_reward_direct``.

``token_reward_direct_plus_grpo`` additionally mixes in a conventional GRPO
outcome advantage computed from the *verifier* score, so the student is pulled
both toward the teacher's distribution and toward actually solving the task.

This module is imported at the bottom of ``verl.trainer.ppo.core_algos`` purely
so that the ``@register_adv_est`` / ``@register_policy_loss`` decorators run.
"""

from typing import Any, Optional

import torch

from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_grpo_outcome_advantage,
    register_adv_est,
    register_policy_loss,
)
from verl.utils.torch_functional import masked_mean
from verl.workers.config import ActorConfig

__all__ = [
    "compute_token_reward_direct_advantage",
    "compute_token_reward_direct_plus_grpo_advantage",
    "compute_discounted_token_reward_advantage",
    "compute_policy_loss_vanilla_topk",
    "compute_policy_loss_opd_noratio",
    "compute_policy_loss_mopd_resid",
    "compute_policy_loss_mopd_grpo",
    "MOPD_POLICY_LOSS_FIELDS",
    "MOPD_ANCHOR_EB_LOSSES",
]

#: Batch fields (written by the M1 anchor pipeline in
#: ``OPDTrainerMixin.compute_opd_token_reward``) that each MOPD per-state loss
#: consumes beyond the shared policy-loss interface. ``ppo_loss`` validates and
#: selects them by loss mode. ``anchor_eb_logp`` is deliberately absent here: it
#: must arrive via ``model_output`` because only the in-forward value carries
#: grad against the *current* student.
MOPD_POLICY_LOSS_FIELDS: dict[str, tuple[str, ...]] = {
    "mopd_resid": ("teacher_log_probs", "anchor_log_probs", "state_gap", "state_anchor_logb"),
    "mopd_grpo": ("teacher_log_probs", "anchor_log_probs", "state_gap"),
}

#: MOPD losses that additionally require ``anchor_eb_logp`` = E_b[log pi_theta]
#: from the actor's own train forward (``anchor.expect_in_train=True``).
MOPD_ANCHOR_EB_LOSSES: frozenset = frozenset({"mopd_resid"})


@register_adv_est("token_reward_direct")
def compute_token_reward_direct_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the token-level teacher reward directly as the advantage.

    Args:
        token_level_rewards: ``(bs, response_length)`` for scalar-per-token OPD,
            or ``(bs, response_length, K)`` when top-K distillation is enabled.
        response_mask: ``(bs, response_length)``.
        config: unused; accepted to match the estimator calling convention.

    Returns:
        ``(advantages, returns)``, both the same shape as ``token_level_rewards``.
    """
    with torch.no_grad():
        mask = response_mask
        if token_level_rewards.dim() == 3:
            # top-K rewards are (bs, T, K); broadcast the (bs, T) mask over K.
            mask = response_mask.unsqueeze(-1)
        advantages = token_level_rewards * mask
        returns = advantages.clone()
    return advantages, returns


@register_adv_est("discounted_token_reward")
def compute_discounted_token_reward_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use a critic-free discounted reward-to-go as the token advantage.

    This is a Monte-Carlo suffix return, not GAE: the implicit value baseline is
    zero and no critic is needed.  For a response reward ``r_t`` it computes

        A_t = r_t + gamma * A_{t+1}

    along the response dimension.  A trailing top-K candidate dimension is
    preserved, so the same estimator can be used with top-K OPD rewards.
    """
    gamma = 1.0
    if config is not None:
        gamma = float(config.get("gamma", 1.0))
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"discounted_token_reward requires gamma in [0, 1], got {gamma}")

    mask = response_mask.to(dtype=token_level_rewards.dtype)
    if token_level_rewards.dim() == 3:
        mask = mask.unsqueeze(-1)
    rewards = token_level_rewards * mask

    returns = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[:, -1])
    for t in range(rewards.shape[1] - 1, -1, -1):
        running = rewards[:, t] + gamma * running
        running = running * mask[:, t]
        returns[:, t] = running

    return returns, returns.clone()


@register_adv_est("token_reward_direct_plus_grpo")
def compute_token_reward_direct_plus_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense teacher advantage plus a GRPO outcome advantage from the verifier.

    The GRPO term is computed from ``true_token_level_scores`` when the trainer
    supplies it (the rule-based verifier reward, which OPD otherwise displaces
    from ``token_level_rewards``). If absent, it falls back to
    ``token_level_rewards`` so the estimator still runs standalone.

    ``algorithm.grpo_outcome_weight`` scales the GRPO term (default 1.0).
    """
    direct_adv, _ = compute_token_reward_direct_advantage(token_level_rewards, response_mask, config, **kwargs)

    rewards_for_grpo = kwargs.get("true_token_level_scores")
    if rewards_for_grpo is None:
        rewards_for_grpo = token_level_rewards
    if rewards_for_grpo.dim() == 3:
        # GRPO needs a scalar outcome per token position; collapse the K axis.
        rewards_for_grpo = rewards_for_grpo.sum(dim=-1)

    norm_adv_by_std_in_grpo = True
    weight = 1.0
    if config is not None:
        norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
        weight = config.get("grpo_outcome_weight", 1.0)

    grpo_adv, _ = compute_grpo_outcome_advantage(
        token_level_rewards=rewards_for_grpo,
        response_mask=response_mask,
        index=index,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )
    if direct_adv.dim() == 3:
        grpo_adv = grpo_adv.unsqueeze(-1)

    combined = direct_adv + weight * grpo_adv
    return combined, combined.clone()


@register_policy_loss("vanilla_topk")
def compute_policy_loss_vanilla_topk(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """PPO clipped objective over top-K candidate tokens per position.

    With top-K distillation the log-probs and advantages carry an extra trailing
    axis of size K: ``(bs, T, K)``. The importance ratio and clipping are applied
    per (position, candidate), then summed over K to yield a per-position loss,
    which is aggregated exactly as the scalar case.

    Falls back to upstream's ``vanilla`` loss when the tensors are 2-D, so the
    same ``policy_loss.loss_mode`` works whether or not top-K is switched on.
    """
    assert config is not None

    if log_prob.dim() != 3 or old_log_prob.dim() != 3:
        from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

        return compute_policy_loss_vanilla(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=rollout_is_weights,
        )

    clip_ratio = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    mask_3d = response_mask.unsqueeze(-1)

    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange_low, 1.0 + cliprange_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights.unsqueeze(-1)

    # Sum over the K axis -> (bs, T), then aggregate with the standard helper.
    # global_batch_info is forwarded exactly as the vanilla loss does, so the
    # normalization (and hence the loss scale) is identical across the two modes.
    pg_loss = agg_loss(
        loss_mat=torch.sum(pg_losses, dim=-1),
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    denom = mask_3d.sum().clamp(min=1)
    pg_clipfrac = (torch.gt(pg_losses2, pg_losses1).float() * mask_3d).sum() / denom
    ppo_kl = (-negative_approx_kl * mask_3d).sum() / denom

    # Keys carry the same "actor/" prefix the vanilla loss uses; ppo_loss passes
    # them through untouched, so dropping it would rename the metrics.
    return pg_loss, {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": 0.0,
    }


@register_policy_loss("opd_noratio")
def compute_policy_loss_opd_noratio(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Pure OPD policy-gradient loss: no PPO ratio and no rollout IS.

    ``old_log_prob`` is accepted for the common actor-loss interface but is
    deliberately unused.  The advantage is detached because OPD computes the
    teacher-minus-current-student gap before entering the policy-gradient
    loss; differentiating through that gap would add an unintended second
    term.  This is the loss used by the frozen-state experiment.
    """
    if rollout_is_weights is not None:
        raise ValueError("opd_noratio does not support rollout importance-sampling weights")
    if config is None:
        raise ValueError("opd_noratio requires ActorConfig")

    pg_losses = -advantages.detach() * log_prob
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )
    return pg_loss, {
        "actor/pg_clipfrac": 0.0,
        "actor/ppo_kl": 0.0,
        "actor/pg_clipfrac_lower": 0.0,
    }


def _mopd_is_floor(
    log_prob: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    anchor_log_probs: torch.Tensor,
    is_cap: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """IS-under-b estimate of ``-E_{pi_T}[log pi_theta]`` — the theta-dependent
    half of the ``lam * KL(pi_T || pi_theta)`` support floor.

    The exact floor needs ``E_{pi_T}[log pi_theta]`` over the full vocabulary;
    M1 stores only sampled-token teacher log-probs, so each sampled token is
    weighted by ``w = pi_T(y)/b(y)`` capped at ``is_cap``. The estimate is
    unbiased over b's support but cannot deliver mass to b-dead tokens — the
    dead-support recovery the floor exists for still needs the deferred
    gather-at-teacher-ids engine path.
    """
    is_w = (teacher_log_probs - anchor_log_probs).exp().clamp_max(is_cap)
    return -is_w * log_prob, {
        "mopd/floor_is_w_mean": is_w,
        "mopd/floor_is_cap_frac": (is_w >= is_cap).to(is_w.dtype),
    }


def _mopd_floor_term(
    config: ActorConfig,
    log_prob: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    anchor_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-token floor contribution ``-lam * w * log pi_theta``, or zero."""
    lam = float(config.policy_loss.get("mopd_lambda", 0.0))
    if lam <= 0.0:
        return torch.zeros((), dtype=log_prob.dtype, device=log_prob.device), {}
    floor_mat, floor_metrics = _mopd_is_floor(
        log_prob, teacher_log_probs, anchor_log_probs, float(config.policy_loss.get("mopd_is_cap", 20.0))
    )
    return lam * floor_mat, floor_metrics


@register_policy_loss("mopd_resid")
def compute_policy_loss_mopd_resid(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    *,
    teacher_log_probs: torch.Tensor,
    anchor_log_probs: torch.Tensor,
    state_gap: torch.Tensor,
    state_anchor_logb: torch.Tensor,
    anchor_eb_logp: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """MOPD per-state residual loss ``E_b[d^2]`` (paper form 2; eta=1 main form).

    Fused residual ``d_v = <b, x> - x_v + (eta-1) A`` with
    ``x = log pi_theta - log pi_T``: the ``E_b[log b]`` halves of the two
    centering KLs cancel analytically inside ``<b, x>``, so the loss never
    subtracts the two large per-state constants (the fp16-fragile pattern E25
    measured). ``<b, x> = anchor_eb_logp + state_gap - state_anchor_logb``
    carries grad only through ``anchor_eb_logp`` (backward ``b - pi_theta``)
    and ``x_v`` only through ``log_prob`` (backward ``pi_theta - e_y``), so
    ``nabla_z d = b - e_y`` and autodiff of ``d^2`` gives the unbiased
    single-sample update ``2 d (b - e_y)`` — no manual stop-gradient. Every
    other field is a detached constant and all logprob-derived quantities are
    fp32 per the E25 precision contract.

    ``old_log_prob``/``advantages`` are accepted for the shared policy-loss
    interface and unused: the M1 anchor fields carry the whole signal.
    """
    if rollout_is_weights is not None:
        raise ValueError("mopd_resid does not support rollout importance-sampling weights")
    if config is None:
        raise ValueError("mopd_resid requires ActorConfig")

    eta = float(config.policy_loss.get("mopd_eta", 1.0))

    log_prob = log_prob.float()
    anchor_eb_logp = anchor_eb_logp.float()
    teacher_lp = teacher_log_probs.float().detach()
    anchor_lp = anchor_log_probs.float().detach()
    gap = state_gap.float().detach()
    eblogb = state_anchor_logb.float().detach()

    # <b, x> = E_b[log pi_theta] - E_b[log pi_T]:
    # anchor_eb_logp is E_b[log pi_theta]; state_gap - state_anchor_logb = -E_b[log pi_T].
    resid = anchor_eb_logp + (gap - eblogb) - (log_prob - teacher_lp)
    if eta != 1.0:
        # d^eta = d^1 + (eta - 1) * A, with A = log pi_T - log b + KL(b || pi_T).
        resid = resid + (eta - 1.0) * (teacher_lp - anchor_lp + gap)

    floor_mat, floor_metrics = _mopd_floor_term(config, log_prob, teacher_lp, anchor_lp)
    loss_mat = resid.square() + floor_mat
    pg_loss = agg_loss(
        loss_mat=loss_mat,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    resid = resid.detach()
    metrics = {
        "actor/pg_clipfrac": 0.0,
        "actor/pg_clipfrac_lower": 0.0,
        "actor/ppo_kl": masked_mean(old_log_prob.float() - log_prob.detach(), response_mask).detach().item(),
        # E_b[d] = 0 per state, so a nonzero masked mean flags plumbing or
        # dtype drift; resid_rms is the LS objective's own scale.
        "mopd/resid_mean": masked_mean(resid, response_mask).item(),
        "mopd/resid_rms": masked_mean(resid.square(), response_mask).sqrt().item(),
    }
    for key, val in floor_metrics.items():
        metrics[key] = masked_mean(val, response_mask).item()
    return pg_loss, metrics


@register_policy_loss("mopd_grpo")
def compute_policy_loss_mopd_grpo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    *,
    teacher_log_probs: torch.Tensor,
    anchor_log_probs: torch.Tensor,
    state_gap: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Practical MOPD loss (paper form 3): the BPO-shaped weighted SFT form.

    ``L = -2 eta sg[A] M min{sg[omega], C} log pi_theta(y|s)`` with
    ``A = log pi_T - log b + KL(b || pi_T)`` — exactly centered per state, so
    no group normalization — and ``omega = (1 + eps - b) / (1 + eps - pi)``
    the binary-KL complement ratio. ``M`` is the BPO clip-rule mask applied to
    ``omega``: off when ``A > 0`` and ``omega > 1 + clip_high`` (positive
    signal, token already saturated) or ``A < 0`` and ``omega < 1 - clip_low``
    (negative signal, already pushed below b); the clip bounds default to the
    actor's ``clip_ratio_low``/``clip_ratio_high``.

    Only ``log_prob`` carries grad: ``A``, ``omega`` and ``M`` are all
    stop-gradiented, so the gradient is the intended weighted-SFT direction.
    """
    if rollout_is_weights is not None:
        raise ValueError("mopd_grpo does not support rollout importance-sampling weights")
    if config is None:
        raise ValueError("mopd_grpo requires ActorConfig")

    eta = float(config.policy_loss.get("mopd_eta", 1.0))
    eps = float(config.policy_loss.get("mopd_epsilon", 0.05))
    cap = float(config.policy_loss.get("mopd_cap", 3.0))
    clip_low = config.policy_loss.get("mopd_clip_low")
    clip_high = config.policy_loss.get("mopd_clip_high")
    if clip_low is None:
        clip_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    if clip_high is None:
        clip_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
    if clip_low is None or clip_high is None:
        raise ValueError(
            "mopd_grpo clip bounds unset: configure policy_loss mopd_clip_low/mopd_clip_high "
            "or actor clip_ratio(_low/_high)"
        )
    clip_low, clip_high = float(clip_low), float(clip_high)

    log_prob = log_prob.float()
    teacher_lp = teacher_log_probs.float().detach()
    anchor_lp = anchor_log_probs.float().detach()
    gap = state_gap.float().detach()

    adv = teacher_lp - anchor_lp + gap
    pi_prob = log_prob.detach().exp()
    b_prob = anchor_lp.exp()
    # Both numerator and denominator stay in [eps, 1+eps], so omega > 0 and the
    # clamp only guards the saturated tail pi -> 1 (omega ~ (1+eps-b)/eps).
    omega = ((1.0 + eps - b_prob) / (1.0 + eps - pi_prob).clamp_min(1e-8)).detach()
    weight = omega.clamp_max(cap)
    keep = ~(((adv > 0) & (omega > 1.0 + clip_high)) | ((adv < 0) & (omega < 1.0 - clip_low)))

    floor_mat, floor_metrics = _mopd_floor_term(config, log_prob, teacher_lp, anchor_lp)
    loss_mat = -2.0 * eta * adv * keep.to(log_prob.dtype) * weight * log_prob + floor_mat
    pg_loss = agg_loss(
        loss_mat=loss_mat,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    clip_frac_high = masked_mean(((adv > 0) & (omega > 1.0 + clip_high)).to(log_prob.dtype), response_mask)
    clip_frac_low = masked_mean(((adv < 0) & (omega < 1.0 - clip_low)).to(log_prob.dtype), response_mask)
    metrics = {
        "actor/pg_clipfrac": (clip_frac_high + clip_frac_low).detach().item(),
        "actor/pg_clipfrac_lower": clip_frac_low.detach().item(),
        "actor/ppo_kl": masked_mean(old_log_prob.float() - log_prob.detach(), response_mask).detach().item(),
        "mopd/adv_mean": masked_mean(adv, response_mask).item(),
        "mopd/adv_abs_mean": masked_mean(adv.abs(), response_mask).item(),
        "mopd/omega_mean": masked_mean(omega, response_mask).item(),
        "mopd/omega_cap_frac": masked_mean((omega > cap).to(log_prob.dtype), response_mask).item(),
    }
    for key, val in floor_metrics.items():
        metrics[key] = masked_mean(val, response_mask).item()
    return pg_loss, metrics

