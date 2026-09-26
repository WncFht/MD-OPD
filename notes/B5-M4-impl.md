# B5 — M4 EOS semantic patch implementation

M4 ports the `semantic_class` terminal-token merge from `opd-eos`[^eos26] into the v0 link-B path used by `recipe/mopd_m*.sh`. The pathology: the base student terminates on `e1` (Qwen `<|endoftext|>` = 151643) while post-trained teachers terminate on `e2` (`<|im_end|>` = 151645); teacher terminal mass lands on a token the student never samples, so sampled-token rewards can never teach the student to stop. The patch collapses both ids into one semantic class `E` and evaluates every log-prob field — teacher, student (old_log_probs and the update forward), and anchor — on the merged action space.

Commits on `mopd-m1` (base `6ec4516`): `326794e` rollout passthrough, `c1224a3` merge utils, `1f95a3d` wiring + config, `6a5d173` recipe defaults, `c9ce352` CPU test.

## Fidelity vs the reference

Ported `semantic_class` semantics verbatim from `verl/utils/eos_semantics.py` in the reference fork: stop ids become the union of `E`, and at positions where `y_t ∈ E` the sampled-token log-prob is replaced by `log p(E) = logsumexp(z_E) − logsumexp(z)`. The reference's `teacher_map` / `canonical` modes are deliberately not ported — the paper runs M4 in class mode throughout.

Two extensions the reference does not have, both required by MOPD's anchor residual. First, per-state expectations run on the merged space: `E_b[f]_merged = Σ_{v∉E} b_v f_v + b_E f_E` with the class coordinate contributing `b_E·logsumexp(z_E)` — applied to `KL(b‖pT)`, `E_b[log b]`, and `E_b[log π_θ]` (`anchor_eb_logp`, which keeps autograd through the student logits). Second, the residual energy uses the merged center `eb_x_m` with raw-lse class coordinate `x_E = lse(z_b_E) − lse(z_t_E)`; the dropped normalization shift is common to all coordinates and annihilates under `d = eb_x − x`, so `d_E = eb_x_m − x_E` matches the direct merged-space residual exactly (verified numerically).

`anchor_log_probs` is a hybrid by construction: on student rows at class positions the rollout's recorded logprob cannot express merged mass (vLLM returns per-token `log p(y_t)` only), so the field takes `anchor_log_probs_fwd` there — the anchor IS the formal `b`, and its forward computes `log b_E` directly. Teacher rows keep the merged `teacher_log_probs`.

`mopd_resid` / `mopd_grpo` kernels needed no changes: `resid = anchor_eb_logp + (gap − eblogb) − (log_prob − teacher_lp)` composes correctly once every input is merged.

## Config surface

`distillation.mopd_eos_patch.enabled` (bool, default `False`) and `distillation.mopd_eos_patch.semantic_eos_token_ids` (list[int], required non-empty distinct when enabled) — dataclass `MopdEosPatchConfig` at `verl/verl/workers/config/fsdp_teacher.py:181`, field on `DistillationConfig` at `verl/verl/workers/config/distillation.py:276`, yaml block at `verl/verl/trainer/config/distillation/distillation.yaml:264`.

`actor_rollout_ref.rollout.extra_stop_token_ids` (list[int], default `[]`) — field at `verl/verl/workers/config/rollout.py:200`, yaml at `verl/verl/trainer/config/rollout/rollout.yaml:56`. Applies to train and val rollouts alike; `SamplingParams` accepts `stop_token_ids` natively and the tool agent loop already unions it with tool stop ids.

Recipes default the patch ON per the paper's 全程开启： `mopd_eos_args` (`recipe/_common.sh:494`) emits `enabled=${MOPD_EOS_PATCH_ENABLED:-True}` and `ids=[${MOPD_EOS_TOKEN_IDS:-151643,151645}]` for both the class and `extra_stop_token_ids`; wired into `mopd.sh`, `mopd_m1.sh`, `mopd_m2.sh`, `mopd_m3.sh`. `opd.sh` is out of scope.

Startup validation (`validate_mopd_eos_config`, `verl/verl/trainer/ppo/opd_trainer_mixin.py:372`) rejects: no `fsdp_teacher` (nothing to merge against), `use_fused_kernels=True` (fused path can't expose merged fields), `rollout.ignore_eos=True`, class ids not covered by `extra_stop_token_ids ∪ {tokenizer.eos_token_id}`, `log_prob_top_k>0` (top-K reward mode), and `rollout_correction.bypass_mode`. `metric_top_k` is allowed — it only observes raw stats.

## File:line map

| Site | Location |
|---|---|
| Merge leaf (dual-backend) | `verl/verl/utils/mopd_eos.py` — `validate_eos_class_ids`:105, `semantic_eos_mask`:116, `eos_class_lse`:124, `eos_class_log_probs`:139, `eos_member_log_probs`:170, `merge_sampled_log_probs`:199, `merged_class_dot`:216, `merged_expectation`:238 |
| Anchor merged terms | `verl/verl/utils/opd_anchor.py` — `anchor_expected_log_prob`:84 (merged branch :125), `compute_anchor_state_terms`:132 (merged branch :206–256, `anchor_eos_log_probs` emit :215) |
| Teacher worker | `verl/verl/workers/fsdp_teacher_worker.py` — `eos_patch_ids` init :291–295, merge in `_forward_micro_batch` :452–456, anchor call :483, scatter + metrics :544, :585–591, :634 |
| Student lp merge (both old_log_probs and update forward share `prepare_model_outputs`) | `verl/verl/workers/engine/fsdp/transformer_impl.py` — flag read :1344, fused-kernel guard :1376, rmpad anchor call :1469, class lp :1477, sampled merge :1495, emit :1532–1534, non-rmpad :1607 |
| Trainer mixin | `verl/verl/trainer/ppo/opd_trainer_mixin.py` — `mopd_eos_patch_enabled`:345, `mopd_eos_class_ids`:352, `assign_eos_patch_flag`:358 (tuple → scalar NonTensorData), `validate_mopd_eos_config`:372, `teacher_eos_log_probs`/`anchor_eos_log_probs` batch fields :522, eos fixup :626–637, `mopd_eos/*` metrics :666–677 |
| Trainer call sites | `verl/verl/trainer/ppo/ray_trainer.py` — validator :403, flag assign :1582 & :1686, `student_eos_class_log_probs` :1618 |
| Rollout stop ids | `verl/verl/experimental/agent_loop/agent_loop.py:631` |

New batch fields: `teacher_eos_log_probs` (B,T,K), `anchor_eos_log_probs` (B,T,K) — per-member log-prob diagnostics; `student_eos_class_log_probs` (B,T) — class mass on student rows. New metrics: `mopd_eos/class_sample_frac`, `mopd_eos/teacher_class_mass`, `mopd_eos/anchor_class_mass`, `mopd_eos/student_class_mass`. Non-tensor flag `mopd_eos_token_ids` (tuple) carries the class into `prepare_model_outputs`.

## Test results

`uv run --no-project --with numpy python3 tests/test_mopd_eos_patch_on_cpu.py` — 10/10 pass. Coverage: id validation accept/reject (incl. the numpy-array truthiness fix at `mopd_eos.py:107` the test caught), `log p(E)` vs direct `log Σ p_E` with and without precomputed lse, member splits summing to class mass, merge semantics at class/non-class/single-member/boundary-id positions plus idempotence, `merged_expectation` vs brute-force collapsed-space `E_b[log p]` (KL ≥ 0, =0 iff `b′=p′`), and merged residual energy + `mu_num` vs direct merged-space residuals including coordinatewise `d` equality. `py_compile` and `ruff` clean on all touched files; `bash -n` clean on all recipes.

## GPU-risk list (untested paths — no torch on this box)

- `anchor_eb_logp` autograd: `merged_expectation` inside `anchor_expected_log_prob` must carry grad through `b_probs`/`z` — the merged branch calls `index_select`/`logsumexp`, both differentiable, but the flow is unverified end-to-end.
- `torch.where(mask, log_b_E, sampled_lp)` broadcast at `opd_anchor.py:223` assumes `log_b_E` shape (chunk,) vs `sampled_lp` (chunk,) — same, fine; but `anchor_log_probs` dtype vs `log_b_E` (float32 vs logits dtype) is unverified.
- `eos_member_log_probs`/`eos_class_log_probs` chunked loop reuses `_flat_chunk_rows(flat)` — the lazy `top_k_logits_chunk_rows` import happens inside the call, unexercised on CPU.
- `NestedTensor` emit `model_output["eos_class_log_probs"]` at `transformer_impl.py:1532` and its read at `ray_trainer.py:1618` (`no_padding_2_padding`) — jagged gather/unpad correctness unverified.
- `compute_rm_score` scatter of (B,T,K) tensors into micro-batch-sized empties at `fsdp_teacher_worker.py:585` assumes the same chunking the (B,T) fields use — K-dim allocation verified by reading, not by running.
- The eos fixup at `opd_trainer_mixin.py:636` indexes `anchor_log_probs`/`anchor_fwd` with a bool mask of shape (B,T) — requires both already on `response_device`; `anchor_fwd` device unverified.
- `teacher_rows` when `generators is None` treats all rows as student rows — matches the M1–M3 assumption, still worth one look under a real multi-teacher batch.

## Smoke-checklist additions

- `OPD_DRY_RUN=1 bash recipe/mopd_m1.sh` — confirm `distillation.mopd_eos_patch.enabled=True`, `semantic_eos_token_ids=[151643,151645]`, `actor_rollout_ref.rollout.extra_stop_token_ids=[151643,151645]` in the resolved config.
- Negative: `MOPD_EOS_TOKEN_IDS=999999 bash recipe/mopd_m1.sh` must fail validation (ids not covered by stop set).
- Negative: `actor_rollout_ref.rollout.ignore_eos=True` + patch on must fail validation.
- GPU smoke (2–3 steps): check logs for `[MOPD-EOS] semantic terminal class active`; `mopd_eos/class_sample_frac` > 0 on math rows; `mopd_eos/teacher_class_mass` ≈ teacher terminal mass (expect ≫0 at sequence end); `mopd_eos/student_class_mass` near 0 early — that gap IS the pathology being patched.
- Loss parity: with patch on vs `MOPD_EOS_PATCH_ENABLED=False`, `mopd_resid` loss should differ only at/near terminal positions; non-terminal `state_gap`/`state_resid_energy` should shift by exactly the class-coordinate contribution.
- Rollout: generations should terminate on either member; a forced `<|im_end|>`-only teacher corpus should no longer produce unterminated student responses.

### 参考文献

[^eos26]: opd-eos reference fork. Terminal-token semantic merging for OPD. arXiv 2609.20511, 2026. [github.com/UNCSciML/opd-eos](https://github.com/UNCSciML/opd-eos)
