#!/bin/bash
# MOPD + M3 跨窗口 replay buffer + P-C 优先级 sampler：在 mopd_m2.sh 的
# 逐状态残差 loss 之上，把 actor 更新批从“严格 on-policy”换成
# “按优先级量从最近 L 个生成窗口里确定性分层抽取”。
#
# MOPD_PRIORITY_MODE=marginal_absorption（默认）：P-C 修正形式，逐状态
#   demand ~ 1/mu_eff(s)，mu_eff = E_b[b d^2]/E_b[d^2] 为边际吸收率
#   （E19/E20 后非总缺口 G）。无残余能量的位置 demand=0。
# MOPD_PRIORITY_MODE=proportional_G_negcontrol：被证伪的 ∝G 反向对照（消融 A3/A7）。
# MOPD_PRIORITY_MODE=uniform：平坦对照，不需要 anchor（但本 recipe 仍开 anchor）。
# MOPD_REPLAY_WINDOW=L：缓冲区保留最近 L 个生成批（默认 4）。
# MOPD_COVERAGE_RARITY=α：覆盖率稀有度混合权重，r=1/(1+n_sampled)，默认 0。
#
# 用法：
#     bash recipe/mopd_m3.sh                                        # 默认 marginal_absorption
#     MOPD_PRIORITY_MODE=proportional_G_negcontrol bash recipe/mopd_m3.sh
#     OPD_DRY_RUN=1 bash recipe/mopd_m3.sh                          # 只打配置

export POLICY_LOSS_MODE=${POLICY_LOSS_MODE:-mopd_resid}

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [ "$MODE" = "template" ]; then
    echo "MODE=template is single-teacher only; use opd.sh" >&2
    exit 2
fi

# mopd_resid 的残差需要带梯度的 anchor_eb_logp；mopd_grpo 只用落盘字段。
if [ "$POLICY_LOSS_MODE" = "mopd_resid" ]; then
    export ANCHOR_EXPECT_IN_TRAIN=${ANCHOR_EXPECT_IN_TRAIN:-True}
fi

export PROJECT_NAME=${PROJECT_NAME:-One-Shot-OPD}

export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${MODEL_BASE}/DeepSeek-R1-Distill-Qwen-1.5B}
export TEACHER_MATH_PATH=${TEACHER_MATH_PATH:-${MODEL_BASE}/JustRL-DeepSeek-1.5B}
export TEACHER_CODE_PATH=${TEACHER_CODE_PATH:-${MODEL_BASE}/Nemotron-Research-Reasoning-Qwen-1.5B-v2}
export TEACHER_IF_PATH=${TEACHER_IF_PATH:-${MODEL_BASE}/UltraData-IF-1.5B}
ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")

export ABILITY_TO_TEACHER=${ABILITY_TO_TEACHER:-'{math:math,code:code,instruction_following:if}'}
export STRICT_ROUTING=${STRICT_ROUTING:-True}

if [ "$MODE" = "oneshot" ]; then
    # 只有 math 提供 1-shot corpus。若需要这种组合，请显式传入 CODE_TRAIN/IF_TRAIN。
    MATH_TRAIN=${MATH_TRAIN:-${DATA_BASE}/math/DAPO-oneshot/t22.parquet}
else
    MATH_TRAIN=${MATH_TRAIN:-${DATA_BASE}/math/DAPO/train.parquet}
fi

CODE_TRAIN=${CODE_TRAIN:-${DATA_BASE}/code/TACO/train.parquet}
IF_TRAIN=${IF_TRAIN:-${DATA_BASE}/if/MultiIF/train.parquet}
export TRAIN_FILES=${TRAIN_FILES:-["$MATH_TRAIN","$CODE_TRAIN","$IF_TRAIN"]}

export VAL_FILES=${VAL_FILES:-["${DATA_BASE}/math/MATH-500/test.parquet","${DATA_BASE}/math/AMC23/test.parquet","${DATA_BASE}/math/AIME25/test.parquet","${DATA_BASE}/code/LCBv6/test.parquet"]}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${DATE}-mopd_m3_${MODE}_${ACTOR_MODEL_NAME}-${POLICY_LOSS_MODE}_${MOPD_PRIORITY_MODE:-marginal_absorption}-win${MOPD_REPLAY_WINDOW:-4}-${TIME_TAG}}

echo "Mode:         $MODE"
echo "Actor:        $ACTOR_MODEL_PATH"
echo "Policy loss:  $POLICY_LOSS_MODE (eta=${MOPD_ETA:-1.0} eps=${MOPD_EPSILON:-0.05} C=${MOPD_CAP:-3.0} lambda=${MOPD_LAMBDA:-0.0})"
echo "Anchor:       from_actor=${ANCHOR_FROM_ACTOR:-True} refresh_interval=${ANCHOR_REFRESH_INTERVAL:-0} expect_in_train=${ANCHOR_EXPECT_IN_TRAIN:-False}"
echo "Replay:       enabled=${MOPD_REPLAY_ENABLED:-True} window=${MOPD_REPLAY_WINDOW:-4} priority=${MOPD_PRIORITY_MODE:-marginal_absorption} coverage_rarity=${MOPD_COVERAGE_RARITY:-0.0}"
echo "EOS patch:    enabled=${MOPD_EOS_PATCH_ENABLED:-True} ids=[${MOPD_EOS_TOKEN_IDS:-151643,151645}]"

launch "$EXPERIMENT_NAME" \
    $(common_args) \
    $(mode_args) \
    $(multi_teacher_args \
        "math:${TEACHER_MATH_PATH}" \
        "code:${TEACHER_CODE_PATH}" \
        "if:${TEACHER_IF_PATH}") \
    $(anchor_args) \
    $(mopd_loss_args) \
    $(mopd_replay_args) \
    $(mopd_eos_args) \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    "$@"
