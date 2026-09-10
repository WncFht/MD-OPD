# One-Shot OPD

This context defines the training-objective vocabulary for one-shot and full-data on-policy distillation experiments.

## Language

**MDPO mirror step (\(\eta\))**:
The proximal step-size parameter controlling how far the updated policy may move from the rollout policy within one MDPO update. Its KL coefficient is \(1 / \eta\).
_Avoid_: optimizer learning rate, Adam learning rate

**MDPO old-policy KL**:
A KL regularizer between the policy being optimized and the fixed policy that scored the current rollout batch. It is distinct from a KL regularizer against an independently frozen reference model.
_Avoid_: reference KL, reward KL

**MDPO eta schedule**:
The outer-update schedule for the MDPO mirror step. A linear schedule decreases eta from 1.0 to 0.01, corresponding to an old-policy KL coefficient increasing from 1 to 100; a constant schedule keeps eta fixed.
_Avoid_: optimizer learning-rate schedule, inner-epoch schedule

**MDPO KL estimator**:
The sampled-token KL estimator selected by the existing KL type setting, such as low-variance k3. It determines how divergence is estimated, while the MDPO eta schedule alone determines its strength.
_Avoid_: MDPO KL coefficient

**MDPO update**:
One actor update on a newly collected, sampled-token OPD rollout batch. Eta progresses once per MDPO update, excluding critic-warmup batches; reference-KL, reward-KL, top-K reward, rollout-correction, and frozen-rollout modes are outside this training mode.
_Avoid_: PPO inner epoch, training batch

**Optimizer learning rate (\(\alpha\))**:
The parameter-update step size used by the optimizer while solving the policy objective. It is independent of the MDPO mirror step.
_Avoid_: MDPO eta, KL coefficient
