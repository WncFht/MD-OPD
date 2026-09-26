#!/bin/bash
# eval.sh — 按基准分开的 val_only 评测驱动（B5-M5）。
#
# 协议给每个族定了不同的 avg@N 和输出 cap（数学 avg@16 cap 31744、LCBv6
# avg@3 cap 65536、BFCL avg@8 cap 16384），而 rollout.val_kwargs.* 是单次
# val 的全局单值，所以每族必须跑一次独立的 val_only pass。本脚本就是那一次
# pass 的入口；与训练内 val 共享 rollout/验证器/指标/dump 全部语义。
#
# 用法：
#     EVAL_CKPT=checkpoints/<exp>/global_step_300/actor/huggingface \
#     EVAL_OUT=eval/<arm>/<query>/<seed>/step_300 \
#     EVAL_BENCH=math bash recipe/eval.sh
#     EVAL_BENCH=math ... OPD_DRY_RUN=1 bash recipe/eval.sh   # 只打配置
#
# EVAL_BENCH ∈ math | math32 | code | bfcl。math32 = math 同表 n=32 档，供
# 协议 pass@k 的 k=32 读数（math 预设 n=16 只够到 pass@16）；写独立 bench
# 目录，不会覆盖 math 的 n=16 dump。Multi-IF 三轮不在此处——训练内 val
# 结构性做不了多轮，走 eval/run_if_eval.py 离线通道。
#
# 产物：
#     $EVAL_OUT/$EVAL_BENCH/dump/<step>.jsonl   逐样本（input/output/gts/
#                                             score/acc/data_source/uid/…）
#     $EVAL_OUT/$EVAL_BENCH/eval_config.json    本 pass 的解析后参数与数据清单
# 聚合：python3 eval/aggregate_report.py $EVAL_OUT --data-root data/
#
# 评测不是训练：不传 teacher/anchor/replay/mopd_loss 任何配置块，
# distillation.* 各项保持 yaml 默认关闭；EOS 补丁显式 enabled=False（其启动
# 校验要求 fsdp_teacher 在场），extra_stop_token_ids 单独保留——任何 ckpt
# 都该在 <|endoftext|> 与 <|im_end|> 上同时收尾。

# ---- 入口参数 ---------------------------------------------------------------
EVAL_BENCH=${EVAL_BENCH:?set EVAL_BENCH to one of: math, code, bfcl}
EVAL_CKPT=${EVAL_CKPT:?set EVAL_CKPT to the checkpoint HF dir (actor/huggingface)}
EVAL_OUT=${EVAL_OUT:?set EVAL_OUT to the eval output root}

[ -d "$EVAL_CKPT" ] || { echo "EVAL_CKPT=$EVAL_CKPT is not a directory" >&2; exit 2; }
[ -f "$EVAL_CKPT/config.json" ] || \
    echo "[warn] $EVAL_CKPT/config.json missing -- is this really an HF model dir?" >&2

# ---- bench 预设（必须先于 source：这些变量在 _common.sh 里定稿） -------------
# MAX_MODEL_LEN 会被 _common.sh 设为 MAX_PROMPT_LENGTH + MAX_VAL_RESP_LENGTH。
case "$EVAL_BENCH" in
    math)
        export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
        export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-31744}
        export VAL_N=${VAL_N:-16}
        ;;
    code)
        export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
        export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-65536}
        export VAL_N=${VAL_N:-3}
        ;;
    bfcl)
        # BFCL prompt 带 tool schema，明显偏长；压到 4096 会被
        # filter_overlong_prompts 静默丢行。
        export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
        export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-16384}
        export VAL_N=${VAL_N:-8}
        ;;
    math32)
        export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
        export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-31744}
        export VAL_N=${VAL_N:-32}
        ;;
    *)
        echo "EVAL_BENCH=$EVAL_BENCH is not one of: math, math32, code, bfcl" >&2
        exit 2
        ;;
esac

# ---- 固定开关：val_only pass 不训练 -----------------------------------------
export VAL_ONLY=True
export POLICY_LOSS_MODE=vanilla USE_KL=False MDPO_KL=False
export LOG_PROB_TOP_K=0 METRIC_TOP_K=0
export TEST_FREQ=-1 SAVE_FREQ=-1 TOTAL_EPOCHS=1

# M4 训练出的学生可能只以 <|im_end|>(151645) 收尾；stop 集合作 rollout 参数
# 单独 emit，对所有 ckpt 都正确。
export EVAL_STOP_TOKEN_IDS=${EVAL_STOP_TOKEN_IDS:-151643,151645}

# ckpt 标签：.../global_step_300/actor/huggingface -> global_step_300
if [ "$(basename "$EVAL_CKPT")" = "huggingface" ] && \
   [ "$(basename "$(dirname "$EVAL_CKPT")")" = "actor" ]; then
    CKPT_TAG=$(basename "$(dirname "$(dirname "$EVAL_CKPT")")")
else
    CKPT_TAG=$(basename "$EVAL_CKPT")
fi
TAG=${TAG:-$CKPT_TAG}

BENCH_DIR="${EVAL_OUT}/${EVAL_BENCH}"
export VAL_DATA_DIR="${BENCH_DIR}/dump"
export ACTOR_MODEL_PATH="$EVAL_CKPT"

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# ---- 数据清单（需要 _common.sh 的 DATA_BASE） --------------------------------
# train dataloader 必须至少有 1 个 batch（drop_last=True，默认 batch=64），
# val_only 不消费它，用仓库内 smoke parquet 占位。
export TRAIN_FILES=${TRAIN_FILES:-["${DATA_BASE}/smoke/mixed_3domain_192.parquet"]}

case "$EVAL_BENCH" in
    math|math32)
        export VAL_FILES=${VAL_FILES:-["${DATA_BASE}/math/MATH-500/test.parquet","${DATA_BASE}/math/AMC23/test.parquet","${DATA_BASE}/math/AIME25/test.parquet"]}
        ;;
    code)
        export VAL_FILES=${VAL_FILES:-["${DATA_BASE}/code/LCBv6/test.parquet"]}
        ;;
    bfcl)
        export VAL_FILES=${VAL_FILES:-["${DATA_BASE}/agentic/BFCL/simple/test.parquet","${DATA_BASE}/agentic/BFCL/multiple/test.parquet","${DATA_BASE}/agentic/BFCL/parallel/test.parquet","${DATA_BASE}/agentic/BFCL/parallel_multiple/test.parquet","${DATA_BASE}/agentic/BFCL/live_simple/test.parquet"]}
        ;;
esac

# ---- 规格快照：aggregator 靠它把 dump 行数对回 parquet 并回填采样参数 --------
# VAL_FILES 到这一步是裸 [a,b]（:- 默认展开会吃掉内层引号）；eval_config.json
# 要合法 JSON，重新加引号。路径本就不能带逗号/空格——Hydra override 按空白分词。
VAL_FILES_JSON=$(printf '%s' "$VAL_FILES" | tr -d '[]"' | tr ',' '\n' | sed 's/.*/"&"/' | paste -sd, -)
mkdir -p "$BENCH_DIR"
cat > "${BENCH_DIR}/eval_config.json" <<EOF
{
  "benchmark": "${EVAL_BENCH}",
  "ckpt": "$(cd "$EVAL_CKPT" && pwd)",
  "ckpt_tag": "${TAG}",
  "val_files": [${VAL_FILES_JSON}],
  "sampling": {
    "n": ${VAL_N},
    "temperature": ${VAL_TEMPERATURE},
    "top_p": ${VAL_TOP_P},
    "max_tokens": ${MAX_VAL_RESP_LENGTH},
    "stop_token_ids": [${EVAL_STOP_TOKEN_IDS}]
  },
  "max_prompt_length": ${MAX_PROMPT_LENGTH}
}
EOF

export PROJECT_NAME=${PROJECT_NAME:-One-Shot-OPD-eval}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-eval-${EVAL_BENCH}-${TAG}}

echo "Eval bench:  $EVAL_BENCH"
echo "Checkpoint:  $EVAL_CKPT"
echo "Val files:   $VAL_FILES  (n=${VAL_N} per prompt, cap=${MAX_VAL_RESP_LENGTH})"
echo "Dump dir:    $VAL_DATA_DIR"

launch "$EXPERIMENT_NAME" \
    $(common_args) \
    $(mode_args) \
    distillation.mopd_eos_patch.enabled=False \
    "actor_rollout_ref.rollout.extra_stop_token_ids=[${EVAL_STOP_TOKEN_IDS}]" \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    "$@"
