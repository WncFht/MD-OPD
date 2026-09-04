#!/bin/bash
# MOPD：按 ability 路由的多 teacher OPD。
#
# 蒸馏目标与 opd.sh 相同，但使用多个冻结 teacher。每行按 `ability` 列路由到
# 对应 teacher；属于同一 teacher 的行组成一个分组计算，再散回原 batch 顺序，
# 因此 reward tensor 的顺序保持不变。每一步都会记录
#     [MOPD] batch routing: {teacher: count}
#
# 用法：
#     bash recipe/mopd.sh                      # 完整数据
#     MODE=oneshot bash recipe/mopd.sh         # 1-shot prompt
#
# 增加或移除 teacher 只需修改参数：向 multi_teacher_args 传入 name:path，
# 并扩展 ABILITY_TO_TEACHER。

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [ "$MODE" = "template" ]; then
    echo "MODE=template is single-teacher only; use opd.sh" >&2
    exit 2
fi

export PROJECT_NAME=${PROJECT_NAME:-One-Shot-OPD}

export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${MODEL_BASE}/DeepSeek-R1-Distill-Qwen-1.5B}
export TEACHER_MATH_PATH=${TEACHER_MATH_PATH:-${MODEL_BASE}/JustRL-DeepSeek-1.5B}
export TEACHER_CODE_PATH=${TEACHER_CODE_PATH:-${MODEL_BASE}/Nemotron-Research-Reasoning-Qwen-1.5B-v2}
export TEACHER_IF_PATH=${TEACHER_IF_PATH:-${MODEL_BASE}/UltraData-IF-1.5B}
ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")

# Keep experiment names focused on settings that change the training signal:
# metric-only top-K is intentionally omitted because it does not change reward.
if [ "${LOG_PROB_TOP_K}" -gt 0 ]; then
    OPD_SIGNAL_TAG="topk${LOG_PROB_TOP_K}_${TOP_K_STRATEGY}_${REWARD_WEIGHT_MODE}"
else
    OPD_SIGNAL_TAG="sampled"
fi
case "${ADV_ESTIMATOR}" in
    token_reward_direct) OPD_ADV_TAG="direct" ;;
    discounted_token_reward) OPD_ADV_TAG="discounted_g${GAMMA}" ;;
    *) OPD_ADV_TAG="${ADV_ESTIMATOR}" ;;
esac

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


export VAL_FILES=${VAL_FILES:-["${DATA_BASE}/math/MATH-500/test.parquet","${DATA_BASE}/code/LCBv6/test.parquet"]}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${DATE}-mopd_${MODE}_${ACTOR_MODEL_NAME}-${OPD_SIGNAL_TAG}_adv_${OPD_ADV_TAG}-T_${TEMPERATURE}-${TIME_TAG}}

echo "Mode:         $MODE"
echo "Actor:        $ACTOR_MODEL_PATH"
echo "Teacher math: $TEACHER_MATH_PATH"
echo "Teacher code: $TEACHER_CODE_PATH"
echo "Teacher if:   $TEACHER_IF_PATH"

launch "$EXPERIMENT_NAME" \
    $(common_args) \
    $(mode_args) \
    $(multi_teacher_args \
        "math:${TEACHER_MATH_PATH}" \
        "code:${TEACHER_CODE_PATH}" \
        "if:${TEACHER_IF_PATH}") \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    "$@"
