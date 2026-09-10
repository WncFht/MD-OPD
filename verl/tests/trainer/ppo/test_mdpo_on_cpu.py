# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""CPU coverage for MDPO's old-policy KL and eta schedule."""

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss_vanilla, kl_penalty
from verl.trainer.ppo.opd_trainer_mixin import compute_mdpo_eta
from verl.utils import tensordict_utils as tu
from verl.workers.config.actor import ActorConfig, PolicyLossConfig
from verl.workers.utils import losses


def actor_config(**kwargs) -> ActorConfig:
    """Build the smallest valid actor configuration for MDPO loss tests."""
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        policy_loss=PolicyLossConfig(loss_mode="vanilla"),
        **kwargs,
    )


def test_mdpo_linear_eta_excludes_critic_warmup():
    assert compute_mdpo_eta(5, 10, 5, 1.0, 0.01, "linear") == pytest.approx(1.0)
    assert compute_mdpo_eta(10, 10, 5, 1.0, 0.01, "linear") == pytest.approx(0.01)
    assert compute_mdpo_eta(7, 10, 5, 1.0, 0.01, "linear") == pytest.approx(0.604)
    assert compute_mdpo_eta(7, 10, 5, 1.0, 0.01, "constant") == pytest.approx(1.0)


def test_mdpo_loss_uses_fixed_old_policy_without_reference(monkeypatch):
    monkeypatch.setattr(losses, "no_padding_2_padding", lambda tensor, data: tensor)
    config = actor_config(mdpo_kl_loss=True, kl_loss_type="k3+")
    model_output = {"log_probs": torch.tensor([[-0.8, -0.9]], requires_grad=True)}
    data = TensorDict(
        {
            "response_mask": torch.tensor([[1.0, 1.0]]),
            "old_log_probs": torch.tensor([[-1.0, -0.7]]),
            "advantages": torch.tensor([[2.0, -1.0]]),
            "dp_size": 1,
            "batch_num_tokens": None,
            "global_batch_size": None,
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(data, mdpo_eta=0.5)

    loss, metrics = losses.ppo_loss(config, model_output, data)
    response_mask = data["response_mask"].bool()
    old_log_prob = data["old_log_probs"]
    log_prob = model_output["log_probs"]
    pg_loss, _ = compute_policy_loss_vanilla(
        old_log_prob, log_prob, data["advantages"], response_mask, "token-mean", config
    )
    mdpo_kl_loss = agg_loss(
        kl_penalty(log_prob, old_log_prob, "k3+"), response_mask, "token-mean", **config.global_batch_info
    )
    torch.testing.assert_close(loss, pg_loss + 2.0 * mdpo_kl_loss)
    assert metrics["mdpo_eta"] == 0.5
    assert metrics["mdpo_kl_coef"] == 2.0


def test_mdpo_and_reference_kl_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        actor_config(mdpo_kl_loss=True, use_kl_loss=True)
