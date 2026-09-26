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
"""MOPD cross-window replay buffer and P-C priority sampler (paper §4.4 M3).

v0 ``RayPPOTrainer`` has no replay buffer at all (the v1 TransferQueue path
carries none of the OPD code), so the cross-window buffer lives here on the
driver: every fully-scored generation batch is appended once, retained for
the last ``window`` generations, and each step the actor's update batch is
re-drawn under the P-C priority measure instead of being strictly on-policy.

The tuple semantics of paper §4.4 -- ``(s, y, log b, log pi_T, G, provenance)``
-- map onto the row granularity the actor update consumes: a buffered row is
a whole scored trajectory whose per-state quantities live in the M1 batch
fields (``anchor_log_probs``, ``teacher_log_probs``, ``state_gap``,
``state_mu_eff``, ``state_resid_energy``) plus the ``anchor_source``
provenance column. Per-state priority demand is aggregated to one scalar per
row by masked mean — the row-level lift of ``eta(s) ~ 1/mu_eff(s)`` (see
``verl.utils.mopd_priority`` for the measure itself, kept numpy-only so the
semantics are CPU-testable).
"""

import copy
import logging
import os

import numpy as np

from verl.protocol import DataProto
from verl.utils.mopd_priority import MODE_FIELDS, position_demand, priority_weights, row_demand, stratified_quota

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = ["MopdReplayBuffer"]

#: meta_info keys describing one batch's row composition; they would collide
#: across windows inside DataProto.concat and are rebuilt on the drawn batch.
VOLATILE_META_KEYS = ("global_token_num", "images_seqlens")


class MopdReplayBuffer:
    """Retains the last ``window`` scored generation batches; draws the actor's
    update batch by deterministic stratified quota under the P-C priority
    measure plus coverage-rarity resampling.
    """

    def __init__(self, window: int, priority_mode: str, coverage_rarity: float):
        self.window = int(window)
        self.priority_mode = str(priority_mode)
        self.coverage_rarity = float(coverage_rarity)
        self.windows: list[dict] = []

    def append(self, batch: DataProto, gen_step: int) -> None:
        """Store one scored window and its per-row priority demand.

        The demand scalar is fixed at append time: the priority measure reads
        the anchor's frozen generation-time terms, which do not move as the
        buffer ages. The proto is deep-copied because the live batch object
        keeps flowing through the rest of the step.
        """
        cuda_fields = [k for k, v in batch.batch.items() if v.is_cuda]
        if cuda_fields:
            raise ValueError(
                f"mopd replay expects CPU-resident batches on the driver, got CUDA tensors "
                f"in {cuda_fields}; every retained window would pin a full GPU copy"
            )
        fields = {"response_mask": batch.batch["response_mask"].detach().cpu().numpy()}
        for key in MODE_FIELDS[self.priority_mode]:
            fields[key] = batch.batch[key].detach().cpu().numpy()
        demand = row_demand(position_demand(fields, self.priority_mode), fields["response_mask"])
        self.windows.append(
            {
                "proto": copy.deepcopy(batch),
                "gen_step": int(gen_step),
                "demand": demand,
                "n_sampled": np.zeros(len(batch), dtype=np.int64),
            }
        )
        self.evict(gen_step)

    def evict(self, current_step: int) -> None:
        """Drop generations older than the last ``window`` fills."""
        cutoff = int(current_step) - self.window + 1
        self.windows = [w for w in self.windows if w["gen_step"] >= cutoff]

    def draw(self, n_rows: int, current_step: int) -> tuple[DataProto, dict]:
        """Draw the actor's update batch under the priority measure.

        Returns the drawn DataProto and a flat ``mopd_replay/*`` metrics dict.
        Repeats are real repeats — a state replayed ``q`` times contributes
        ``q`` slots to the batch, which is the only allocation granularity
        Adam cannot normalize away.
        """
        self.evict(current_step)
        demand = np.concatenate([w["demand"] for w in self.windows])
        counts = np.concatenate([w["n_sampled"] for w in self.windows])
        weights = priority_weights(demand, counts, self.coverage_rarity)
        quota = stratified_quota(weights, n_rows)

        parts = []
        offset = 0
        for w in self.windows:
            n = w["demand"].shape[0]
            q = quota[offset : offset + n]
            offset += n
            w["n_sampled"] += q
            if q.sum() == 0:
                continue
            idxs = np.repeat(np.arange(n), q)
            part = w["proto"].select_idxs(idxs)
            part.meta_info = {
                key: value for key, value in w["proto"].meta_info.items() if key not in VOLATILE_META_KEYS
            }
            parts.append(part)

        drawn = DataProto.concat(parts) if len(parts) > 1 else parts[0]
        # Row-composition meta_info refers to the freshly drawn batch.
        drawn.meta_info["global_token_num"] = drawn.batch["attention_mask"].sum(dim=-1).tolist()
        if "multi_modal_inputs" in drawn.non_tensor_batch:
            images_seqlens_all = []
            for multi_modal_input in drawn.non_tensor_batch["multi_modal_inputs"]:
                if "image_grid_thw" not in multi_modal_input.keys():
                    continue
                images_seqlens_all.extend(multi_modal_input["image_grid_thw"].tolist())
            drawn.meta_info["images_seqlens"] = images_seqlens_all

        metrics = {
            "mopd_replay/buffer_rows": int(demand.size),
            "mopd_replay/buffer_windows": len(self.windows),
            "mopd_replay/draw_unique": int((quota > 0).sum()),
            "mopd_replay/draw_repeat_slots": int(n_rows - (quota > 0).sum()),
            "mopd_replay/coverage_frac": float((counts + quota > 0).mean()),
            "mopd_replay/weight_max_share": float(weights.max() / weights.sum()),
        }
        provenance = drawn.non_tensor_batch.get("anchor_source")
        if provenance is not None:
            metrics["mopd_replay/source_teacher_frac"] = float(np.mean([str(g) == "teacher" for g in provenance]))
        return drawn, metrics
