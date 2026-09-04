#!/bin/bash
# GRPO：参考 baseline，仅使用 verifier reward，不使用 teacher。
#
# 使用与 opd.sh 相同的入口、数据和长度设置；reward 改为规则 verifier，
# 并基于每个 prompt 的 N_RESPONSES 个样本计算 GRPO advantage。不加载 teacher。
#
# 用法：
#     bash recipe/grpo.sh                      # 完整数据
#     MODE=oneshot bash recipe/grpo.sh         # 1-shot prompt

# 由 _common.sh 读取，因此必须在 source 之前设置。
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
export N_RESPONSES=${N_RESPONSES:-8}

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [ "$MODE" = "template" ]; then
    echo "MODE=template needs a teacher to score the generated prompts; use opd.sh" >&2
    exit 2
fi

export PROJECT_NAME=${PROJECT_NAME:-One-Shot-OPD}

export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${MODEL_BASE}/DeepSeek-R1-Distill-Qwen-1.5B}
ACTOR_MODEL_NAME=$(basename "$ACTOR_MODEL_PATH")

export VAL_FILES=${VAL_FILES:-["${DATA_BASE}/math/MATH-500/test.parquet"]}
if [ "$MODE" = "oneshot" ]; then
    export TRAIN_FILES=${TRAIN_FILES:-["${DATA_BASE}/math/DAPO-oneshot/t22.parquet"]}
else
    export TRAIN_FILES=${TRAIN_FILES:-["${DATA_BASE}/math/DAPO/train.parquet"]}
fi

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${DATE}-grpo_${MODE}_${ACTOR_MODEL_NAME}-T_${TEMPERATURE}-${TIME_TAG}}

echo "Mode:  $MODE"
echo "Actor: $ACTOR_MODEL_PATH"
echo "Train: $TRAIN_FILES  (n=${N_RESPONSES} per prompt, adv=${ADV_ESTIMATOR})"

launch "$EXPERIMENT_NAME" \
    $(common_args) \
    $(mode_args) \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    "$@"
