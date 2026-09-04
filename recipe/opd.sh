#!/bin/bash
# OPD：使用一个冻结 teacher 的 on-policy distillation。
#
# teacher 对 student 自己生成的 rollout 打分；逐 token reverse KL
# -(logp_student - logp_teacher) 作为 token-level reward。梯度不使用 verifier
# reward（该 baseline 见 grpo.sh）。
#
# 用法：
#     bash recipe/opd.sh                       # math，完整数据
#     DOMAIN=code bash recipe/opd.sh           # code，完整数据
#     MODE=oneshot bash recipe/opd.sh          # 1-shot prompt
#     MODE=template bash recipe/opd.sh         # student 自己编写输入
#
# DOMAIN 选择 teacher 和 parquet；MODE 选择读取哪种 prompt。
# 命令行末尾追加的参数会原样传给 Hydra。

# 设置 DISCOUNTED_TOKEN_REWARD=True 可使用无 critic 的 discounted reward-to-go：
# A_t = r_t + GAMMA * A_{t+1}。默认仍使用 token_reward_direct。
export DISCOUNTED_TOKEN_REWARD=${DISCOUNTED_TOKEN_REWARD:-False}
if [ "$DISCOUNTED_TOKEN_REWARD" = "True" ]; then
    export ADV_ESTIMATOR=${ADV_ESTIMATOR:-discounted_token_reward}
fi

# --- domain 默认值 ----------------------------------------------------------
# 在 source _common.sh 之前运行：_common.sh 在 source 时将 MAX_PROMPT_LENGTH /
# MAX_RESP_LENGTH 解析为 MAX_MODEL_LEN 和 SEQ_LEN_TOKENS，因此需要不同预算的
# domain 必须在这里先设置。此处只选择名称；MODEL_BASE 和 DATA_BASE 尚未建立。
DOMAIN=${DOMAIN:-math}
case "$DOMAIN" in
    math)
        DOMAIN_TEACHER=JustRL-DeepSeek-1.5B
        DOMAIN_TRAIN=math/DAPO/train.parquet
        DOMAIN_VAL="math/MATH-500/test.parquet
                    math/AMC23/test.parquet
                    math/AIME25/test.parquet"
        ;;
    code)
        DOMAIN_TEACHER=DeepCoder-1.5B-Preview
        # 使用 TACO，而不是 LCBv6：LCBv6 是 validation 集，没有 train split。
        DOMAIN_TRAIN=code/TACO/train.parquet
        DOMAIN_VAL=code/LCBv6/test.parquet
        ;;
    fc)
        # 该 domain 的 student 不使用共享的 R1 distill 模型。
        DOMAIN_ACTOR=Qwen2.5-Coder-1.5B-Instruct
        DOMAIN_TEACHER=Hammer2.1-1.5b
        DOMAIN_TRAIN=agentic/xlam/train.parquet
        DOMAIN_VAL="agentic/BFCL/simple/test.parquet
                    agentic/BFCL/multiple/test.parquet
                    agentic/BFCL/parallel/test.parquet
                    agentic/BFCL/parallel_multiple/test.parquet
                    agentic/BFCL/live_simple/test.parquet"
        export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
        export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-2048}
        ;;
    if)
        DOMAIN_TEACHER=UltraData-IF-1.5B
        DOMAIN_TRAIN=if/MultiIF/train.parquet
        # 特意留空：Multi-IF 有三轮，而 validation loop 只生成一次，因此该 domain
        # 的指标唯一来源是 eval/run_if_eval.py。
        DOMAIN_VAL=
        export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
        export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-7168}
        ;;
    *)
        echo "unknown DOMAIN='$DOMAIN' (expected: math | code | fc | if)" >&2
        exit 1
        ;;
esac

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

export PROJECT_NAME=${PROJECT_NAME:-One-Shot-OPD}

export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${MODEL_BASE}/${DOMAIN_ACTOR:-DeepSeek-R1-Distill-Qwen-1.5B}}
export TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-${MODEL_BASE}/${DOMAIN_TEACHER}}
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
    discounted_token_reward) OPD_ADV_TAG="discounted" ;;
    *) OPD_ADV_TAG="${ADV_ESTIMATOR}" ;;
esac

# DOMAIN_VAL 是以空白分隔的 DATA_BASE 相对路径列表；Hydra 要求一个带方括号的
# 列表。空值保持为空，见下方 `if` 分支。
if [ -z "${VAL_FILES:-}" ]; then
    _val_list=
    for _p in $DOMAIN_VAL; do
        _val_list="${_val_list:+${_val_list},}${DATA_BASE}/${_p}"
    done
    export VAL_FILES="[${_val_list}]"
fi
case "$MODE" in
    full)
        export TRAIN_FILES=${TRAIN_FILES:-[${DATA_BASE}/${DOMAIN_TRAIN}]}
        ;;
    oneshot)
        # t22 是 DAPO problem 22 的 64 次重复：共 64 行但只有一个不同 prompt。
        # 由 data/prep/prepare_dapo_oneshot.py 构建。
        if [ -z "${TRAIN_FILES:-}" ] && [ "$DOMAIN" != "math" ]; then
            echo "MODE=oneshot has no corpus for DOMAIN=$DOMAIN -- only math" \
                 "ships one (math/DAPO-oneshot/). Pass TRAIN_FILES to override." >&2
            exit 1
        fi
        export TRAIN_FILES=${TRAIN_FILES:-[${DATA_BASE}/math/DAPO-oneshot/t22.parquet]}
        ;;
    template)
        # prompt 是生成的，不从文件读取；但 Hydra 仍要求提供该 key。
        export TRAIN_FILES=${TRAIN_FILES:-$VAL_FILES}
        ;;
esac

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${DATE}-opd_${DOMAIN}_${MODE}_${ACTOR_MODEL_NAME}-${OPD_SIGNAL_TAG}_adv_${OPD_ADV_TAG}-T_${TEMPERATURE}-${TIME_TAG}}

echo "Domain:  $DOMAIN"
echo "Mode:    $MODE"
echo "Actor:   $ACTOR_MODEL_PATH"
echo "Teacher: $TEACHER_MODEL_PATH"
echo "Train:   $TRAIN_FILES"
echo "Val:     $VAL_FILES"

launch "$EXPERIMENT_NAME" \
    $(common_args) \
    $(mode_args) \
    $(teacher_args "$TEACHER_MODEL_PATH") \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    "$@"
