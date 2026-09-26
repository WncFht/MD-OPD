#!/bin/bash
# MOPD + M1 anchor：在 mopd.sh 的多 teacher 路由之上启用冻结生成器快照 b。
#
# anchor 挂在 fsdp_teacher worker 上、独立于 teachers 路由表，对全部行做一次
# 配对前向，落盘 MOPD 残差所需的逐 token / 逐状态量：
#     anchor_log_probs      (B,T) 逐 token 真实生成器 logprob
#     anchor_log_probs_fwd  (B,T) anchor 自身前向重算的 log b(y_t)
#     state_gap             (B,T) G(s_t) = KL(b || pi_T)(s_t)
#     state_anchor_logb     (B,T) E_b[log b](s_t) = -H(b|s_t)
#     anchor_source         (B,)  {"rollout","teacher"} 填充来源
#     anchor_top_k_*        (B,T,K) 可选截断支持集（消融专用）
# 非张量开关 ANCHOR_EXPECT_IN_TRAIN=True 时，actor 训练前向内联跑一次 anchor
# 并输出 anchor_eb_logp = E_b[log pi_theta]（每 micro-batch +1 次 anchor 前向；
# 要求 use_remove_padding=True、use_fused_kernels=False、ulysses 与 actor 一致）。
#
# 用法：
#     bash recipe/mopd_m1.sh                      # 完整数据
#     MODE=oneshot bash recipe/mopd_m1.sh         # 1-shot prompt
#     OPD_DRY_RUN=1 bash recipe/mopd_m1.sh        # 只打配置
#
# anchor 窗口语义：refresh_interval=L 时每 L 步在窗口起点重采样 b <- pi_theta；
# 0 表示整个训练冻结在初始快照。anchor.micro_batch_size_per_gpu 控制配对前向的
# 行粒度（两份 (rows,T,V) logits 同时在手），显存紧时减半。

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

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-${DATE}-mopd_m1_${MODE}_${ACTOR_MODEL_NAME}-${OPD_SIGNAL_TAG}_adv_${OPD_ADV_TAG}-T_${TEMPERATURE}-anchorL${ANCHOR_REFRESH_INTERVAL:-0}-${TIME_TAG}}

echo "Mode:         $MODE"
echo "Actor:        $ACTOR_MODEL_PATH"
echo "Teacher math: $TEACHER_MATH_PATH"
echo "Teacher code: $TEACHER_CODE_PATH"
echo "Teacher if:   $TEACHER_IF_PATH"
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
    $(mopd_eos_args) \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    "$@"
