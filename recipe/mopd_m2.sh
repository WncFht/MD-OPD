#!/bin/bash
# MOPD + M2 逐状态残差 loss：在 mopd_m1.sh 的 anchor 存盘之上切换到
# paper §4.3 的两个等价目标。
#
# POLICY_LOSS_MODE=mopd_resid（默认）：精确 E_b[d^2]，d = log pi_T - log pi_theta
#   + KL(b||pi_T) - KL(b||pi_theta)。梯度 2d(b - e_y) 经 anchor_eb_logp 与
#   log_prob 自动微分得到 —— 需要 anchor.expect_in_train=True（每 actor
#   micro-batch 多一次 anchor 前向）。
# POLICY_LOSS_MODE=mopd_grpo：BPO 同形实用损失
#   -2 eta sg[A] M min{sg[omega],C} log pi_theta，A = log pi_T - log b +
#   KL(b||pi_T) 逐状态精确居中（免组归一化）；只消费落盘字段，可关
#   ANCHOR_EXPECT_IN_TRAIN 省掉那次前向。
# MOPD_LAMBDA>0 开 lambda*KL(pi_T||pi_theta) 支撑地板的 b-IS 近似
# （只覆盖 b 的支撑，不能复活 b-死 token；cap 由 MOPD_IS_CAP 控制）。
#
# 用法：
#     bash recipe/mopd_m2.sh                                  # 完整数据 + mopd_resid
#     POLICY_LOSS_MODE=mopd_grpo bash recipe/mopd_m2.sh       # BPO 同形损失
#     OPD_DRY_RUN=1 bash recipe/mopd_m2.sh                    # 只打配置

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

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${DATE}-mopd_m2_${MODE}_${ACTOR_MODEL_NAME}-${POLICY_LOSS_MODE}_eta${MOPD_ETA:-1.0}-anchorL${ANCHOR_REFRESH_INTERVAL:-0}-${TIME_TAG}}

echo "Mode:         $MODE"
echo "Actor:        $ACTOR_MODEL_PATH"
echo "Policy loss:  $POLICY_LOSS_MODE (eta=${MOPD_ETA:-1.0} eps=${MOPD_EPSILON:-0.05} C=${MOPD_CAP:-3.0} lambda=${MOPD_LAMBDA:-0.0})"
echo "Anchor:       from_actor=${ANCHOR_FROM_ACTOR:-True} refresh_interval=${ANCHOR_REFRESH_INTERVAL:-0} expect_in_train=${ANCHOR_EXPECT_IN_TRAIN:-False}"
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
    $(mopd_eos_args) \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    "$@"
