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
"""OPD extensions for :class:`~verl.trainer.ppo.ray_trainer.RayPPOTrainer`.

Everything OPD adds on the trainer side lives here as a mixin so the upstream
trainer keeps its shape -- ``RayPPOTrainer`` inherits it and calls into it from
three short hook sites, rather than being forked into a second trainer class.
That is what keeps ``python -m verl.trainer.main_ppo`` the single entry point
for all four scenarios (OPD / MOPD / 1-shot OPD / template).

* :meth:`OPDTrainerMixin.register_opd_teacher` -- add the teacher worker to the
  actor's colocated worker set, so the teacher shares the actor's GPUs instead
  of demanding a second resource pool.
* :meth:`OPDTrainerMixin.init_opd_teacher` -- load the teacher weights, after
  the actor/rollout group is up so vLLM's KV-cache estimate is unaffected.
* :meth:`OPDTrainerMixin.compute_opd_token_reward` -- run the teacher over the
  student rollout and install the resulting dense per-token reward as
  ``token_level_scores``, preserving the verifier's own score under
  ``true_token_level_scores`` for ``token_reward_direct_plus_grpo``.
* :meth:`OPDTrainerMixin.log_template_metrics` -- filter statistics for
  template mode's self-generated queries.

All of it is inert unless ``distillation.fsdp_teacher.enabled=True`` (or
``template.enabled=True``), so an unmodified PPO/GRPO run behaves exactly as
upstream does.
"""

import logging
import os
from collections import Counter
from typing import Any

import numpy as np
import torch

from verl.protocol import DataProto
from verl.trainer.ppo.opd_core_algos import MOPD_ANCHOR_EB_LOSSES, MOPD_POLICY_LOSS_FIELDS
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.mopd_priority import PRIORITY_MODES
from verl.workers.config import FsdpTeacherConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = ["OPDTrainerMixin", "compute_mdpo_eta", "opd_teacher_enabled", "template_enabled"]


def _mean_metric(value):
    """Average a worker metric that may be a scalar or a DP-concat list."""
    if value is None:
        return None
    if isinstance(value, list):
        nums = [float(x) for x in value if x is not None]
        return sum(nums) / max(len(nums), 1) if nums else None
    return float(value)


#: Key under which the teacher worker is registered in the colocated worker set.
OPD_TEACHER_KEY = "opd_teacher"


def opd_teacher_enabled(config) -> bool:
    """Whether the OPD FSDP teacher should be built for this run."""
    distillation = config.get("distillation", None)
    if distillation is None:
        return False
    fsdp_teacher = distillation.get("fsdp_teacher", None)
    if fsdp_teacher is None:
        return False
    return bool(fsdp_teacher.get("enabled", False))


def template_enabled(config) -> bool:
    """Whether template mode (student writes its own query) is on."""
    template = config.get("template", None)
    return bool(template is not None and template.get("enabled", False))


def compute_mdpo_eta(
    global_step: int,
    total_training_steps: int,
    critic_warmup: int,
    eta_start: float,
    eta_end: float,
    schedule: str,
) -> float:
    """Return the MDPO mirror step for one actor outer update."""
    if schedule == "constant":
        return eta_start

    first_actor_step = max(1, critic_warmup)
    actor_update_count = total_training_steps - first_actor_step + 1
    progress = (global_step - first_actor_step) / max(actor_update_count - 1, 1)
    progress = min(max(progress, 0.0), 1.0)
    return eta_start + progress * (eta_end - eta_start)


class OPDTrainerMixin:
    """Teacher-reward and template-mode hooks for the PPO trainer."""

    # ------------------------------------------------------------------
    # teacher worker lifecycle
    # ------------------------------------------------------------------
    def register_opd_teacher(self, resource_pool) -> None:
        """Colocate the teacher worker with the actor on ``resource_pool``.

        The teacher is forward-only and its params are CPU-offloaded between
        micro-batches, so sharing the actor's GPUs costs far less than standing
        up a second resource pool would.
        """
        import ray

        from verl.single_controller.ray import RayClassWithInitArgs
        from verl.workers.fsdp_teacher_worker import TeacherWorker

        self.opd_teacher_config: FsdpTeacherConfig = omega_conf_to_dataclass(self.config.distillation.fsdp_teacher)
        teacher_cls = RayClassWithInitArgs(cls=ray.remote(TeacherWorker), config=self.config)
        self.resource_pool_to_cls[resource_pool][OPD_TEACHER_KEY] = teacher_cls

    def init_opd_teacher(self, all_wg: dict) -> None:
        """Pick the teacher group out of the spawned workers and load its weights."""
        self.opd_teacher_wg = all_wg[OPD_TEACHER_KEY]
        self.opd_teacher_wg.init_model()

    # ------------------------------------------------------------------
    # top-K: reward mode vs metric-only mode
    # ------------------------------------------------------------------
    def opd_student_top_k(self) -> int:
        """K to extract from the student, or 0 to skip top-K entirely.

        Two independent knobs decide this, and they mean different things:

        * ``log_prob_top_k > 0`` -- top-K *drives the reward*. The token reward
          becomes the (B, T, K) top-K distillation reward.
        * ``metric_top_k > 0`` -- top-K is computed for *observability only*.
          The reward stays the plain (B, T) reverse KL; the top-K tensors are
          used solely to log student/teacher agreement.

        ``log_prob_top_k`` wins when both are set, because you cannot train on
        one K while reporting another without the metric describing a different
        distribution than the reward. Metric mode exists so a run can measure
        how much top-K would change the signal before switching a real training
        run over to it.
        """
        if not getattr(self, "use_opd_teacher", False):
            return 0
        teacher = self.config.distillation.fsdp_teacher
        train_top_k = int(teacher.get("log_prob_top_k", 0) or 0)
        if train_top_k > 0:
            return train_top_k
        return int(teacher.get("metric_top_k", 0) or 0)

    def opd_top_k_is_reward(self) -> bool:
        """True when top-K should replace the reward, not merely be measured."""
        if not getattr(self, "use_opd_teacher", False):
            return False
        return int(self.config.distillation.fsdp_teacher.get("log_prob_top_k", 0) or 0) > 0

    def validate_opd_top_k_config(self) -> None:
        """Reject top-K reward setups whose candidate set would drift under training.

        The training forward re-derives the student's top-K from the current
        logits; the advantage it is multiplied against was built from the top-K
        taken at rollout time. Those two agree only while the actor has not been
        updated since the rollout -- i.e. one epoch over one mini-batch that spans
        the whole rollout batch. Any further update makes the ratio compare
        log-probs of two different candidate sets, which is silently wrong rather
        than loudly broken, so it is refused up front.
        """
        if not self.opd_top_k_is_reward():
            return
        actor = self.config.actor_rollout_ref.actor
        ppo_epochs = int(actor.ppo_epochs)
        mini_batch_size = int(actor.ppo_mini_batch_size)
        train_batch_size = int(self.config.data.train_batch_size)
        if ppo_epochs != 1 or mini_batch_size != train_batch_size:
            raise ValueError(
                "OPD top-K reward mode (distillation.fsdp_teacher.log_prob_top_k > 0) requires strictly "
                "on-policy updates: actor.ppo_epochs=1 and actor.ppo_mini_batch_size == data.train_batch_size. "
                f"Got ppo_epochs={ppo_epochs}, ppo_mini_batch_size={mini_batch_size}, "
                f"train_batch_size={train_batch_size}. Off-policy updates re-derive a different top-K "
                "candidate set than the one the advantage was built from. Use metric_top_k instead if you "
                "only want the agreement statistics."
            )

    # ------------------------------------------------------------------
    # MOPD anchor (frozen generator snapshot b)
    # ------------------------------------------------------------------
    def opd_anchor_enabled(self) -> bool:
        """Whether the anchor snapshot module is active for this run."""
        if not getattr(self, "use_opd_teacher", False):
            return False
        anchor = self.config.distillation.fsdp_teacher.get("anchor", None)
        return bool(anchor is not None and anchor.get("enabled", False))

    def opd_anchor_expect_in_train(self) -> bool:
        """Whether the actor forward must also emit ``anchor_eb_logp`` = E_b[log pi_theta].

        That expectation is the only term of the MOPD residual that depends on
        the *current* student, so it has to be computed where full-vocab student
        logits exist — inside the actor's train forward, dotted against the
        colocated frozen anchor.
        """
        if not self.opd_anchor_enabled():
            return False
        return bool(self.config.distillation.fsdp_teacher.anchor.get("expect_in_train", False))

    def opd_anchor_refresh_interval(self) -> int:
        """Window length in global steps between b <- pi_theta re-snapshots; 0 disables."""
        if not self.opd_anchor_enabled():
            return 0
        return int(self.config.distillation.fsdp_teacher.anchor.get("refresh_interval", 0) or 0)

    def assign_anchor_expectation_flag(self, batch_td) -> None:
        """Mark the actor forward to emit ``anchor_eb_logp`` for this batch."""
        from verl.utils import tensordict_utils as tu

        if not self.opd_anchor_expect_in_train():
            return
        anchor = self.config.distillation.fsdp_teacher.anchor
        tu.assign_non_tensor(
            batch_td,
            opd_anchor_expectation=True,
            opd_anchor_temperature=float(anchor.get("temperature", 1.0)),
        )

    def validate_opd_anchor_config(self) -> None:
        """Reject anchor setups the colocated in-engine forward cannot honour.

        ``expect_in_train`` runs the anchor inside the actor's forward on the
        exact same packed inputs, so the two modules must share that input
        layout: remove-padding on, fused kernels off (full logits must
        materialize for the ``<b, z>`` dot), and identical ulysses
        sequence-parallel sizes so the SP-sliced inputs are valid for both.
        """
        if not self.opd_anchor_expect_in_train():
            return
        model_cfg = self.config.actor_rollout_ref.model
        if not model_cfg.get("use_remove_padding", False):
            raise ValueError("anchor.expect_in_train requires actor_rollout_ref.model.use_remove_padding=True")
        if model_cfg.get("use_fused_kernels", False):
            raise ValueError("anchor.expect_in_train requires actor_rollout_ref.model.use_fused_kernels=False")
        # The anchor was patched with the teacher's settings; it must accept the
        # same packed inputs the actor engine builds.
        if not self.config.distillation.fsdp_teacher.get("use_remove_padding", True):
            raise ValueError("anchor.expect_in_train requires distillation.fsdp_teacher.use_remove_padding=True")
        teacher_sp = int(self.config.distillation.fsdp_teacher.get("ulysses_sequence_parallel_size", 1) or 1)
        actor_sp = int(self.config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) or 1)
        if teacher_sp != actor_sp:
            raise ValueError(
                "anchor.expect_in_train requires matching ulysses_sequence_parallel_size on "
                f"actor ({actor_sp}) and fsdp_teacher ({teacher_sp}): the anchor forward "
                "consumes the SP-sliced packed inputs the actor engine builds."
            )

    def validate_mopd_loss_config(self) -> None:
        """Refuse mopd_* policy losses whose anchor inputs would be missing.

        The per-state residual losses read the M1 anchor batch fields
        (``teacher_log_probs``, ``anchor_log_probs``, ``state_gap``,
        ``state_anchor_logb``), so they are only defined when the anchor module
        is on; ``mopd_resid`` additionally needs ``anchor_eb_logp`` emitted by
        the actor's own train forward (``anchor.expect_in_train``). They are
        sampled-token losses: top-K reward mode's (B, T, K) plumbing does not
        apply to them.
        """
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if loss_mode not in MOPD_POLICY_LOSS_FIELDS:
            return
        if not self.opd_anchor_enabled():
            raise ValueError(
                f"actor.policy_loss.loss_mode={loss_mode} requires "
                "distillation.fsdp_teacher.anchor.enabled=True: the M1 anchor pipeline writes "
                "the b-side fields (anchor_log_probs, state_gap, state_anchor_logb) it consumes."
            )
        if loss_mode in MOPD_ANCHOR_EB_LOSSES and not self.opd_anchor_expect_in_train():
            raise ValueError(
                f"actor.policy_loss.loss_mode={loss_mode} requires "
                "distillation.fsdp_teacher.anchor.expect_in_train=True: the residual needs "
                "anchor_eb_logp = E_b[log pi_theta] from the actor's train forward."
            )
        if self.opd_top_k_is_reward():
            raise ValueError(
                f"actor.policy_loss.loss_mode={loss_mode} is a sampled-token MOPD loss; "
                "distillation.fsdp_teacher.log_prob_top_k>0 (top-K reward mode) is incompatible."
            )

    # ------------------------------------------------------------------
    # MOPD replay (cross-window buffer + P-C priority sampler)
    # ------------------------------------------------------------------
    def mopd_replay_enabled(self) -> bool:
        """Whether the M3 cross-window replay buffer drives the actor update."""
        if not getattr(self, "use_opd_teacher", False):
            return False
        replay = self.config.distillation.get("mopd_replay", None)
        return bool(replay is not None and replay.get("enabled", False))

    def validate_mopd_replay_config(self) -> None:
        """Refuse replay setups whose priority inputs would be missing.

        ``marginal_absorption`` reads ``state_mu_eff``/``state_resid_energy``
        and ``proportional_G_negcontrol`` reads ``state_gap`` — all written by
        the anchor paired forward, so a non-uniform replay without the anchor
        would silently sample uniform. ``frozen_rollout`` already replays a
        single cached window forever; stacking another buffer on top of it is
        not a meaningful configuration.
        """
        if not self.mopd_replay_enabled():
            return
        replay = self.config.distillation.mopd_replay
        mode = replay.get("priority_mode", "marginal_absorption")
        if mode not in PRIORITY_MODES:
            raise ValueError(f"distillation.mopd_replay.priority_mode must be one of {PRIORITY_MODES}, got {mode!r}")
        if int(replay.get("window", 0)) < 1:
            raise ValueError("distillation.mopd_replay.window must be >= 1")
        coverage_rarity = float(replay.get("coverage_rarity", 0.0))
        if not 0.0 <= coverage_rarity <= 1.0:
            raise ValueError("distillation.mopd_replay.coverage_rarity must be in [0, 1]")
        if self.frozen_rollout:
            raise ValueError(
                "distillation.mopd_replay is incompatible with algorithm.frozen_rollout: "
                "frozen mode already replays one cached window on every step."
            )
        if mode != "uniform" and not self.opd_anchor_enabled():
            raise ValueError(
                f"distillation.mopd_replay.priority_mode={mode} requires "
                "distillation.fsdp_teacher.anchor.enabled=True: the priority fields "
                "(state_mu_eff, state_resid_energy, state_gap) are written by the "
                "anchor paired forward."
            )

    def init_mopd_replay(self):
        """Build the driver-side replay buffer, or None when disabled."""
        if not self.mopd_replay_enabled():
            return None
        from verl.trainer.ppo.mopd_replay_buffer import MopdReplayBuffer

        replay = self.config.distillation.mopd_replay
        buffer = MopdReplayBuffer(
            window=int(replay.get("window", 4)),
            priority_mode=replay.get("priority_mode", "marginal_absorption"),
            coverage_rarity=float(replay.get("coverage_rarity", 0.0)),
        )
        print(
            "[mopd-replay] enabled: "
            f"window={buffer.window}, priority_mode={buffer.priority_mode}, "
            f"coverage_rarity={buffer.coverage_rarity}",
            flush=True,
        )
        return buffer

    def validate_mdpo_config(self) -> None:
        """Reject training modes whose policy anchors differ from MDPO's old policy."""
        actor = self.config.actor_rollout_ref.actor
        if not actor.mdpo_kl_loss:
            return
        if not self.use_opd_teacher:
            raise ValueError("mdpo_kl_loss requires OPD teacher distillation")
        if actor.policy_loss.get("loss_mode", "vanilla") != "vanilla":
            raise ValueError("mdpo_kl_loss requires policy_loss.loss_mode=vanilla")
        if self.config.algorithm.use_kl_in_reward:
            raise ValueError("mdpo_kl_loss does not support algorithm.use_kl_in_reward=True")
        if self.frozen_rollout:
            raise ValueError("mdpo_kl_loss does not support algorithm.frozen_rollout=True")
        if int(self.config.distillation.fsdp_teacher.get("log_prob_top_k", 0) or 0) > 0:
            raise ValueError("mdpo_kl_loss supports sampled-token OPD only (log_prob_top_k=0)")
        rollout_corr = self.config.algorithm.get("rollout_correction", {}) or {}
        if (
            rollout_corr.get("bypass_mode", False)
            or rollout_corr.get("rollout_is") is not None
            or rollout_corr.get("rollout_rs") is not None
        ):
            raise ValueError("mdpo_kl_loss does not support rollout correction or bypass mode")

    # ------------------------------------------------------------------
    # teacher reward
    # ------------------------------------------------------------------
    def compute_opd_token_reward(self, batch: DataProto, metrics: dict[str, Any]) -> DataProto:
        """Replace the outcome reward with the teacher's dense per-token reward.

        ``token_level_scores`` normally holds the verifier's sparse score. OPD
        needs the dense teacher reward there instead, because that is what the
        advantage estimator consumes. The verifier's own tensor is kept under
        ``true_token_level_scores`` so ``token_reward_direct_plus_grpo`` can
        still build a GRPO outcome term from it, and so the usual
        ``critic/score`` metrics stay meaningful.

        When top-K runs in metric-only mode the teacher still computes the
        overlap statistics, but ``rm_scores`` stays the plain (B, T) reverse KL.
        """
        batch.meta_info["opd_top_k_is_reward"] = self.opd_top_k_is_reward()

        # MOPD anchor window lifecycle: re-snapshot b <- pi_theta at window
        # starts. Steps are 1-indexed and scoring precedes the actor update, so
        # window starts are global_steps in {L+1, 2L+1, ...} — at that point the
        # actor holds exactly the weights the window's first trajectory was
        # sampled from. Step 1 already got its snapshot at init when
        # anchor.from_actor=True.
        refresh_interval = self.opd_anchor_refresh_interval()
        if refresh_interval > 0 and self.global_steps > 1 and (self.global_steps - 1) % refresh_interval == 0:
            logger.info(
                "[MOPD] refreshing anchor at global_steps=%d (interval=%d)",
                self.global_steps,
                refresh_interval,
            )
            self.opd_teacher_wg.refresh_anchor()

        # Frozen-state mode caches only the rollout tokens/states.  The teacher
        # remains frozen, but we intentionally run one teacher forward per
        # optimizer step so metric-only top-K uses the *current* student's
        # candidate set and reports fresh overlap/mass statistics.
        teacher_output = self.opd_teacher_wg.compute_rm_score(batch)

        if "token_level_scores" in batch.batch.keys():
            batch.batch["true_token_level_scores"] = batch.batch["token_level_scores"]

        response_mask = batch.batch["response_mask"]
        rm_scores = teacher_output.batch["rm_scores"].to(response_mask.device)
        batch.batch["token_level_scores"] = rm_scores
        batch.batch["teacher_log_probs"] = teacher_output.batch["teacher_log_probs"].to(response_mask.device)

        n_tokens = max(int(response_mask.sum().item()), 1)
        # Keep the diagnostic statistics on the plain sampled-token advantage,
        # even when top-K reward mode is enabled.  This makes the metrics
        # comparable across metric-only and top-K training runs.
        teacher_log_probs = batch.batch["teacher_log_probs"]
        student_log_probs = batch.batch.get("current_student_log_probs", batch.batch["old_log_probs"]).to(
            teacher_log_probs.device
        )
        token_mask = response_mask.to(teacher_log_probs.dtype)
        advantage = teacher_log_probs - student_log_probs
        denom = token_mask.sum().clamp_min(1.0)
        adv_mean = (advantage * token_mask).sum() / denom
        adv_second = (advantage.square() * token_mask).sum() / denom
        adv_abs_mean = (advantage.abs() * token_mask).sum() / denom
        adv_pos_mean = (advantage.clamp_min(0) * token_mask).sum() / denom
        adv_neg_abs_mean = (advantage.clamp_max(0).abs() * token_mask).sum() / denom
        metrics["opd/adv_mean"] = float(adv_mean.item())
        metrics["opd/adv_abs_mean"] = float(adv_abs_mean.item())
        metrics["opd/adv_rms"] = float(adv_second.clamp_min(0).sqrt().item())
        metrics["opd/adv_std"] = float((adv_second - adv_mean.square()).clamp_min(0).sqrt().item())
        metrics["opd/adv_pos_mean"] = float(adv_pos_mean.item())
        metrics["opd/adv_neg_abs_mean"] = float(adv_neg_abs_mean.item())
        metrics["opd/adv_pos_frac"] = float(
            ((advantage > 0).to(token_mask.dtype) * token_mask).sum().item() / denom.item()
        )

        # These are strict same-batch sampled-token aggregates: all terms use
        # the response mask from this exact teacher-scoring batch.  The CE term
        # is the teacher NLL on sampled student tokens; it is deliberately
        # named explicitly rather than presented as a full-vocabulary KL.
        metrics["opd/teacher_nll_mean"] = float(((-teacher_log_probs) * token_mask).sum().item() / denom.item())
        metrics["opd/student_nll_mean"] = float(((-student_log_probs) * token_mask).sum().item() / denom.item())
        entropys = batch.batch.get("entropys")
        if entropys is not None:
            entropys = entropys.to(teacher_log_probs.device, dtype=teacher_log_probs.dtype)
            student_entropy_mean = (entropys * token_mask).sum() / denom
            metrics["opd/student_entropy_mean"] = float(student_entropy_mean.item())
            metrics["opd/ce_minus_entropy_proxy"] = float(
                metrics["opd/teacher_nll_mean"] - metrics["opd/student_entropy_mean"]
            )

        # rm_scores may be (B, T) or (B, T, K) when top-K is on; broadcast the mask.
        mask = response_mask if rm_scores.dim() == 2 else response_mask.unsqueeze(-1)
        metrics["opd/token_reward_mean"] = float((rm_scores * mask).sum().item()) / n_tokens
        metrics["opd/teacher_logp_mean"] = (
            float((batch.batch["teacher_log_probs"] * response_mask).sum().item()) / n_tokens
        )

        teachers = teacher_output.non_tensor_batch.get("mopd_teacher")
        if teachers is not None:
            counts = dict(sorted(Counter(str(t) for t in teachers).items()))
            # The running MOPD job greps for exactly this line.
            print(f"[MOPD] batch routing: {counts}", flush=True)
            logger.info("[MOPD] batch routing: %s", counts)
            for name, count in counts.items():
                metrics[f"mopd/routed/{name}"] = count

        # Worker scalars live under meta_info["metrics"] so DataProto.concat
        # (which special-cases that key) can aggregate across DP ranks.
        worker_metrics = teacher_output.meta_info.get("metrics") or {}
        overlap = _mean_metric(worker_metrics.get("top_k_overlap_ratio"))
        if overlap is not None:
            metrics["opd/top_k_overlap_ratio"] = float(overlap)
            metrics["opd/top_k_mode"] = 1.0 if self.opd_top_k_is_reward() else 0.0
        for key in ("top_k_student_mass", "top_k_teacher_mass", "top_k_reward_mean"):
            value = _mean_metric(worker_metrics.get(key))
            if value is not None:
                metrics[f"opd/{key}"] = float(value)

        if self.opd_anchor_enabled():
            response_device = response_mask.device
            anchor_fwd = teacher_output.batch["anchor_log_probs_fwd"].to(response_device)
            batch.batch["state_gap"] = teacher_output.batch["state_gap"].to(response_device)
            batch.batch["state_anchor_logb"] = teacher_output.batch["state_anchor_logb"].to(response_device)
            batch.batch["state_mu_eff"] = teacher_output.batch["state_mu_eff"].to(response_device)
            batch.batch["state_resid_energy"] = teacher_output.batch["state_resid_energy"].to(response_device)
            batch.batch["anchor_log_probs_fwd"] = anchor_fwd

            # anchor_log_probs carries each token's *actual generator* logprob:
            # student-generated rows take the rollout's own recorded logprob
            # (the true b), teacher-generated rows take the routed teacher's
            # logprob. The pure anchor-forward value stays under
            # anchor_log_probs_fwd so the two can be compared.
            generator_lp = batch.batch.get("rollout_log_probs")
            if generator_lp is None:
                generator_lp = batch.batch["old_log_probs"]
            generator_lp = generator_lp.to(response_device).float()
            anchor_log_probs = generator_lp.clone()
            anchor_source = np.full(len(batch), "rollout", dtype=object)
            generators = batch.non_tensor_batch.get("generator")
            if generators is not None:
                teacher_rows = np.asarray([str(g) == "teacher" for g in generators])
                if teacher_rows.any():
                    teacher_idx = torch.as_tensor(teacher_rows, device=response_device)
                    anchor_log_probs[teacher_idx] = batch.batch["teacher_log_probs"][teacher_idx]
                    anchor_source[teacher_rows] = "teacher"
            batch.batch["anchor_log_probs"] = anchor_log_probs
            batch.non_tensor_batch["anchor_source"] = anchor_source

            if "anchor_top_k_ids" in teacher_output.batch.keys():
                batch.batch["anchor_top_k_ids"] = teacher_output.batch["anchor_top_k_ids"].to(response_device)
                batch.batch["anchor_top_k_log_probs"] = teacher_output.batch["anchor_top_k_log_probs"].to(
                    response_device
                )

            gap_mean = _mean_metric(worker_metrics.get("anchor_state_gap_mean"))
            if gap_mean is not None:
                metrics["opd/anchor_state_gap_mean"] = float(gap_mean)
            logb_mean = _mean_metric(worker_metrics.get("anchor_logb_mean"))
            if logb_mean is not None:
                metrics["opd/anchor_logb_mean"] = float(logb_mean)
            for key in ("anchor_mu_eff_mean", "anchor_resid_energy_mean"):
                value = _mean_metric(worker_metrics.get(key))
                if value is not None:
                    metrics[f"opd/{key}"] = float(value)
            # Sanity signal: the anchor's own recomputation of log b(y_t) should
            # sit near zero difference against the recorded generator logprob;
            # a persistent gap means engine-vs-FSDP numerics or a stale anchor.
            metrics["opd/anchor_vs_rollout_mean"] = float(
                ((anchor_fwd - generator_lp) * token_mask).sum().item() / denom.item()
            )

        return batch

    # ------------------------------------------------------------------
    # template mode
    # ------------------------------------------------------------------
    def log_template_metrics(self, batch: DataProto, metrics: dict[str, Any]) -> None:
        """Report how many self-generated queries would pass the content filter.

        This is monitoring only -- rows are not dropped here, because dropping
        rows mid-step would desynchronise the DP-balanced mini-batching. Set
        ``template.oversample_ratio`` above 1.0 to trade compute for quality
        instead.
        """
        from verl.utils.template_prompt import decode_responses, filter_valid_texts

        template_config = self.config.get("template", {})
        try:
            texts = decode_responses(
                self.tokenizer,
                batch.batch["responses"],
                batch.batch["response_mask"],
            )
            result = filter_valid_texts(
                texts,
                min_char_length=template_config.get("min_char_length", 20),
                task=template_config.get("task", "math"),
            )
        except Exception as e:  # monitoring must never break training
            logger.warning("[Template] filter monitoring failed: %s", e)
            return

        metrics["template/valid_ratio"] = result.valid_ratio
        metrics["template/valid_count"] = result.valid_count
        metrics["template/rejected_count"] = result.rejected_count
        for reason, count in result.reject_reasons.items():
            metrics[f"template/reject_{reason}"] = count

        print(
            f"[Template] step {self.global_steps}: valid={result.valid_count}/{result.total_count} "
            f"({result.valid_ratio:.1%}) rejects={result.reject_reasons}",
            flush=True,
        )

    @staticmethod
    def _masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> float:
        return (tensor * mask).sum().item() / mask.sum().clamp(min=1).item()
