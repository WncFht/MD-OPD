#!/bin/bash
# 本 recipe 中所有启动脚本共用的设置。该文件会被 source，而不是直接执行。
#
#   grpo.sh   仅使用 verifier reward，不使用 teacher（baseline）
#   opd.sh    一个冻结 teacher 给 student 打分           MODE=full|oneshot|template
#   mopd.sh   按 ability 列路由到多个 teacher           MODE=full|oneshot
#
# 三者都通过 `python -m verl.trainer.main_ppo` 启动，并设置 trainer.use_v1=False。

set -x

# ============ 路径 ============
_OPD_RECIPE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OPD_RECIPE_DIR=${OPD_RECIPE_DIR:-$_OPD_RECIPE_DIR}
export VERL_ROOT=${VERL_ROOT:-$(cd "${_OPD_RECIPE_DIR}/../verl" && pwd)}
# OPD_ROOT 是 data/、logs/ 和 checkpoints/ 所在目录。
OPD_ROOT=${OPD_ROOT:-$(cd "${_OPD_RECIPE_DIR}/.." && pwd)}
export OPD_ROOT
export PYTHONPATH=${VERL_ROOT}:${PYTHONPATH}

export MODEL_BASE=${MODEL_BASE:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/model}
export DATA_BASE=${DATA_BASE:-${OPD_ROOT}/data}

# 可使用 conda、venv 或集群预装 Python。保留 CONDA_ENV_BIN 兼容旧配置，但
# 不再把默认占位路径或空字符串写入 PATH。
export CONDA_ENV_BIN=${CONDA_ENV_BIN:-}
if [[ -n "$CONDA_ENV_BIN" ]]; then
    export PATH="$CONDA_ENV_BIN:$PATH"
fi

# ============ 集群 Python/CUDA 环境适配 ============
# 不假设提交节点激活了 conda；按显式覆盖、当前 conda、项目 .venv 和系统
# Python 的顺序寻找真正包含训练依赖的解释器。
opd_sanitize_ld_library_path() {
    local old_ifs="$IFS" entry sanitized=""
    IFS=":"
    for entry in ${LD_LIBRARY_PATH:-}; do
        case "$entry" in
            ""|/usr/local/cuda/compat|/usr/local/cuda/compat/*|\
            /usr/local/lib/python3.10/site-packages/torch/lib|\
            /usr/local/lib/python3.10/site-packages/torch/lib/*) continue ;;
        esac
        [[ -d "$entry" ]] || continue
        sanitized="${sanitized:+${sanitized}:}${entry}"
    done
    IFS="$old_ifs"
    export LD_LIBRARY_PATH="$sanitized"
}

opd_python_usable() {
    local candidate="$1"
    [[ "$candidate" = */* ]] || candidate="$(command -v "$candidate" 2>/dev/null || true)"
    [[ -x "$candidate" ]] || return 1
    env -u PYTHONPATH "$candidate" -c \
        'import torch, ray, transformers, vllm, hydra, omegaconf' >/dev/null 2>&1
}

opd_setup_train_env() {
    opd_sanitize_ld_library_path
    local candidate configured="${PYTHON_BIN:-}" conda_prefix="${CONDA_PREFIX:-}" virtual_env="${VIRTUAL_ENV:-}"
    if [[ -n "$configured" ]] && opd_python_usable "$configured"; then
        :
    else
        [[ -z "$configured" ]] || echo "[warn] unusable PYTHON_BIN: $configured" >&2
        PYTHON_BIN=""
        [[ -z "$conda_prefix" ]] || for candidate in "$conda_prefix/bin/python" "$conda_prefix/bin/python3"; do
            opd_python_usable "$candidate" && { PYTHON_BIN="$candidate"; break; }
        done
        if [[ -z "$PYTHON_BIN" ]]; then
            local custom_path
            IFS=":" read -r -a _opd_custom_paths <<< "${OPD_PYTHON_CANDIDATES:-}"
            for custom_path in "${_opd_custom_paths[@]}"; do
                [[ -n "$custom_path" ]] || continue
                opd_python_usable "$custom_path" && { PYTHON_BIN="$custom_path"; break; }
            done
        fi
        if [[ -z "$PYTHON_BIN" ]]; then
            for candidate in "$CONDA_ENV_BIN/python" "$CONDA_ENV_BIN/python3" \
                "$virtual_env/bin/python" "$virtual_env/bin/python3" \
                "$OPD_ROOT/.venv/bin/python" "$OPD_ROOT/.venv/bin/python3" \
                "$VERL_ROOT/.venv/bin/python" "$VERL_ROOT/.venv/bin/python3" \
                "$HOME/conda/bin/python" "$HOME/conda/bin/python3" \
                /usr/local/bin/python3 /usr/local/bin/python \
                /usr/bin/python3 /usr/bin/python /bin/python3 /bin/python; do
                opd_python_usable "$candidate" && { PYTHON_BIN="$candidate"; break; }
            done
        fi
        if [[ -z "$PYTHON_BIN" ]]; then
            for candidate in python3 python; do
                opd_python_usable "$candidate" && { PYTHON_BIN="$candidate"; break; }
            done
        fi
    fi
    [[ -n "$PYTHON_BIN" ]] || { echo "[error] No Python with torch/ray/transformers/vllm/hydra found. Set PYTHON_BIN." >&2; return 2; }
    export PYTHON_BIN PYTHONNOUSERSITE=1
    export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
}

opd_print_train_env() {
    echo "[env] hostname=$(hostname)" "date=$(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "[env] OPD_ROOT=$OPD_ROOT VERL_ROOT=$VERL_ROOT PYTHON_BIN=$PYTHON_BIN"
    echo "[env] python_version=$($PYTHON_BIN --version 2>&1)"
    echo "[env] PYTHONPATH=${PYTHONPATH:-<unset>} LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
    echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
}

opd_require_train_dependencies() {
    "$PYTHON_BIN" - <<'PY'
import importlib
required = ("torch", "ray", "transformers", "vllm", "hydra", "omegaconf")
for name in required:
    module = importlib.import_module(name)
    print(f"[env] dependency_{name}={getattr(module, '__version__', 'ok')}")
PY
}

opd_require_cuda() {
    "$PYTHON_BIN" - <<'PY'
import os, sys, torch
print("[env] dependency_python:", sys.executable)
print("[env] torch:", torch.__version__, "cuda:", torch.version.cuda)
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("CUDA preflight failed: torch CUDA runtime is unavailable.")
torch.cuda.synchronize()
print("[env] cuda_devices:", torch.cuda.device_count(), torch.cuda.get_device_name(0))
PY
}

# ============ 模式 ============
export MODE=${MODE:-full}
[ "$MODE" = "input" ] && export MODE=template
case "$MODE" in
    full|oneshot|template) ;;
    *)
        # 必须 exit，不能 return；return 只会离开当前被 source 的文件。
        echo "MODE=${MODE} is not one of: full, oneshot, template (alias: input)" >&2
        exit 2
        ;;
esac

# ============ 序列长度 ============
# 4096 足够容纳 code/IF/agentic；math launcher 会自行设置为 1024。不要在这里
# 全局降低：data.filter_overlong_prompts=True 会丢弃超长行，而不是截断它们。
if [ "$MODE" = "oneshot" ]; then
    export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
fi
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-7168}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-31744}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-$(( MAX_PROMPT_LENGTH + MAX_VAL_RESP_LENGTH ))}

if [ "${MAX_MODEL_LEN}" -lt "$(( MAX_PROMPT_LENGTH + MAX_RESP_LENGTH ))" ]; then
    echo "MAX_MODEL_LEN=${MAX_MODEL_LEN} is shorter than the sequence it must" \
         "hold (MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH} + MAX_RESP_LENGTH=" \
         "${MAX_RESP_LENGTH}). Raise MAX_VAL_RESP_LENGTH to at least" \
         "${MAX_RESP_LENGTH}, or set MAX_MODEL_LEN explicitly." >&2
    exit 1
fi

# ============ 训练超参数 ============
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
# 同一份 rollout 重复更新几次。不是 trainer.total_epochs（整份数据扫几遍）。
# LOG_PROB_TOP_K>0 时必须保持 1，否则 top-K candidate 会和 advantage 错位。
export PPO_EPOCHS=${PPO_EPOCHS:-1}
export TEMPERATURE=${TEMPERATURE:-1.0}
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
export N_RESPONSES=${N_RESPONSES:-1}
export PARALLEL_SIZE=${PARALLEL_SIZE:-1}
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-token_reward_direct}
export GAMMA=${GAMMA:-1.0}
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
export MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
export USE_KL=${USE_KL:-False}
export MDPO_KL=${MDPO_KL:-False}
export MDPO_ETA_SCHEDULE=${MDPO_ETA_SCHEDULE:-linear}
export MDPO_ETA_START=${MDPO_ETA_START:-1.0}
export MDPO_ETA_END=${MDPO_ETA_END:-0.01}

# 默认只保存 Hugging Face 格式模型到 checkpoint/huggingface/，不保存
# optimizer 或 extra 状态。此模式适合作为推理/发布产物，不支持从该 checkpoint
# 恢复 optimizer 状态继续训练；如需断点续训，可设置 SAVE_CONTENTS。
export SAVE_CONTENTS=${SAVE_CONTENTS:-"['hf_model']"}
# 默认不从 HF-only 产物恢复训练状态；需要断点续训时同时覆盖 SAVE_CONTENTS 和
# LOAD_CONTENTS（例如保存并加载 model/optimizer/extra）。
export LOAD_CONTENTS=${LOAD_CONTENTS:-"[]"}
# HF-only checkpoint 不能用于自动续训，因此默认关闭自动 resume；完整断点续训
# 时可设置 RESUME_MODE=auto 或 RESUME_MODE=resume_path，并覆盖上面的保存/加载内容。
export RESUME_MODE=${RESUME_MODE:-disable}

# ============ top-K 蒸馏 ============
#   LOG_PROB_TOP_K > 0  top-K 会成为训练 reward（token reward 变为 3-D）
#                      同时会计算并记录 top-k metrics；不需要再开启 METRIC_TOP_K。
#   METRIC_TOP_K   > 0  top-K 仅用于指标；reward 仍是逐 token reverse KL
# 两者同时设置时 LOG_PROB_TOP_K 优先（METRIC_TOP_K 被忽略）。
# 只想观测 top-k 而不改变 reward 时，保持 LOG_PROB_TOP_K=0 并设置 METRIC_TOP_K>0。
# 两者都为 0 时完全关闭 top-K。
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-0}
export METRIC_TOP_K=${METRIC_TOP_K:-16}
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}
# reward 模式会增加 candidate 维度，原生 `vanilla` loss 无法对其广播。
if [ "${LOG_PROB_TOP_K}" -gt 0 ]; then
    export POLICY_LOSS_MODE=${POLICY_LOSS_MODE:-vanilla_topk}
else
    export POLICY_LOSS_MODE=${POLICY_LOSS_MODE:-vanilla}
fi

# ============ template 模式（MODE=template） ============
export PREFIX_TEMPLATE=${PREFIX_TEMPLATE:-'{bos}{user}<think>\n'}
export PREFIX_SUFFIX=${PREFIX_SUFFIX:-''}
export PREFIX_MODE=${PREFIX_MODE:-template}
export MIN_CHAR_LENGTH=${MIN_CHAR_LENGTH:-20}
export TEMPLATE_TASK=${TEMPLATE_TASK:-math}
export TEMPLATE_DATASET_LENGTH=${TEMPLATE_DATASET_LENGTH:-1000000}
export TEMPLATE_DATA_SOURCE=${TEMPLATE_DATA_SOURCE:-template}
export OVERSAMPLE_RATIO=${OVERSAMPLE_RATIO:-1.0}

# ============ 集群 ============
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export NNODES=${NNODES:-1}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export SAVE_FREQ=${SAVE_FREQ:-50}
export TEST_FREQ=${TEST_FREQ:-20}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}

# VAL_ONLY=True 使用当前权重对 data.val_files 打分后退出。
export VAL_ONLY=${VAL_ONLY:-False}
[ "$VAL_ONLY" = "True" ] && export VAL_BEFORE_TRAIN=True

# ============ validation 采样 ============
# 使用 avg@N，而不是 greedy。所有参考结果都在这些设置下产生。
export VAL_N=${VAL_N:-16}
export VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.7}
export VAL_TOP_P=${VAL_TOP_P:-0.9}
# validation 时记录到 tracker 的 rollout 数；TRAINER_LOGGER 包含 wandb 时
# 会写入 `val/generations` 表。设为 0 可关闭，仍可用 VAL_N 生成评估样本。
export VAL_LOG_GENERATIONS=${VAL_LOG_GENERATIONS:-8}

# ============ W&B ============
# 远程节点默认离线记录；如需上传云端，可设置 WANDB_MODE=online。
export WANDB_MODE=${WANDB_MODE:-offline}
# offline 模式不需要 API key，记录 console 和 W&B。
export TRAINER_LOGGER="['console','wandb']"

# ============ 运行时环境 ============
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true
export RAY_memory_usage_threshold=0.99
# ray.init 的 `uv run` 探测会遍历祖先进程，并可能因 NoSuchProcess 退出。
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}
# `local` 强制创建新集群；自动发现可能连接到别人的 GCS。
export RAY_ADDRESS=${RAY_ADDRESS:-local}
# ray、vllm 和 FSDP 会耗尽 1024 的 soft fd 限制，而且失败时可能没有明显报错。
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
# 如果 flash-attn 无法针对当前 torch/CUDA 构建，可覆盖为 sdpa。
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
# DataLoader worker 会 fork trainer；配合 FSDP CPU offload 时可能收到 SIGKILL。
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

opd_setup_train_env || exit $?
opd_print_train_env
opd_require_train_dependencies || exit $?
if [ -z "${OPD_DRY_RUN:-}" ]; then
    opd_require_cuda || exit $?
fi

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M)
export DATE TIME_TAG

mkdir -p "${OPD_ROOT}/logs"

# dynamic batch size 的 token 预算必须能容纳一条完整序列。
SEQ_LEN_TOKENS=$(( MAX_PROMPT_LENGTH + MAX_RESP_LENGTH ))
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$(( SEQ_LEN_TOKENS > 32768 ? SEQ_LEN_TOKENS : 32768 ))}
if [ "${PPO_MAX_TOKEN_LEN_PER_GPU}" -lt "${SEQ_LEN_TOKENS}" ]; then
    echo "PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU} cannot hold one" \
         "sequence (${MAX_PROMPT_LENGTH}+${MAX_RESP_LENGTH}=${SEQ_LEN_TOKENS})" >&2
    exit 1
fi
export PPO_MAX_TOKEN_LEN_PER_GPU

if [ "$USE_KL" = "True" ]; then
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.005 \
        actor_rollout_ref.actor.kl_loss_type=k3+"
else
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=False"
fi
export KL_ARGS

if [ "$MDPO_KL" = "True" ] && [ "$USE_KL" = "True" ]; then
    echo "MDPO_KL=True requires USE_KL=False because MDPO anchors KL to the rollout policy." >&2
    exit 1
fi

MDPO_ARGS="actor_rollout_ref.actor.mdpo_kl_loss=${MDPO_KL} \
    actor_rollout_ref.actor.mdpo_eta_schedule=${MDPO_ETA_SCHEDULE} \
    actor_rollout_ref.actor.mdpo_eta_start=${MDPO_ETA_START} \
    actor_rollout_ref.actor.mdpo_eta_end=${MDPO_ETA_END}"
export MDPO_ARGS

# common_args：所有 launcher 共用的 Hydra override。这里故意按空白分词
# （Hydra 要求每个 override 一个 token），因此路径不能包含空格。
common_args() {
    echo "\
        --config-name=ppo_trainer \
        trainer.use_v1=False \
        algorithm.adv_estimator=${ADV_ESTIMATOR} \
        algorithm.gamma=${GAMMA} \
        data.train_batch_size=$(( MINI_BATCH_SIZE * PARALLEL_SIZE )) \
        data.max_prompt_length=${MAX_PROMPT_LENGTH} \
        data.max_response_length=${MAX_RESP_LENGTH} \
        data.filter_overlong_prompts=True \
        data.truncation=error \
        data.return_raw_chat=True \
        data.dataloader_num_workers=${DATALOADER_NUM_WORKERS} \
        actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        ++actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPLEMENTATION} \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
        actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS} \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${PARALLEL_SIZE} \
        actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE} \
        actor_rollout_ref.actor.policy_loss.loss_mode=${POLICY_LOSS_MODE} \
        ${KL_ARGS} \
        ${MDPO_ARGS} \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${PARALLEL_SIZE} \
        actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.75} \
        actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
        actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN} \
        actor_rollout_ref.rollout.max_num_seqs=${MINI_BATCH_SIZE} \
        actor_rollout_ref.rollout.n=${N_RESPONSES} \
        actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH} \
        actor_rollout_ref.rollout.response_length=${MAX_RESP_LENGTH} \
        actor_rollout_ref.rollout.calculate_log_probs=True \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
        actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
        actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
        actor_rollout_ref.rollout.val_kwargs.max_tokens=${MAX_VAL_RESP_LENGTH} \
        trainer.log_val_generations=${VAL_LOG_GENERATIONS} \
        reward.custom_reward_function.path=${VERL_ROOT}/verl/utils/reward_score/opd_mixed_verify.py \
        reward.custom_reward_function.name=reward_func \
        trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
        trainer.nnodes=${NNODES} \
        trainer.total_epochs=${TOTAL_EPOCHS} \
        trainer.save_freq=${SAVE_FREQ} \
        trainer.test_freq=${TEST_FREQ} \
        trainer.resume_mode=${RESUME_MODE} \
        trainer.val_before_train=${VAL_BEFORE_TRAIN} \
        trainer.val_only=${VAL_ONLY} \
        actor_rollout_ref.actor.checkpoint.save_contents="${SAVE_CONTENTS}" \
        actor_rollout_ref.actor.checkpoint.load_contents="${LOAD_CONTENTS}" \
        trainer.logger=${TRAINER_LOGGER}"
}

# mode_args：MODE=full / oneshot / template 之间的差异。
#
# prefix 值使用字面单引号包裹：其中的 `{`、`}` 和 `<` 否则会被 Hydra
# override lexer 当作字典/列表语法解析。
mode_args() {
    if [ "$MODE" != "template" ]; then
        echo "data.shuffle=True"
        return
    fi
    echo "\
        data.shuffle=False \
        data.custom_cls.path=${VERL_ROOT}/verl/utils/dataset/template_dataset.py \
        data.custom_cls.name=TemplatePromptDataset \
        template.enabled=True \
        template.min_char_length=${MIN_CHAR_LENGTH} \
        template.prefix_template='${PREFIX_TEMPLATE}' \
        template.prefix_suffix='${PREFIX_SUFFIX}' \
        template.prefix_mode=${PREFIX_MODE} \
        template.task=${TEMPLATE_TASK} \
        template.dataset_length=${TEMPLATE_DATASET_LENGTH} \
        template.data_source=${TEMPLATE_DATA_SOURCE} \
        template.oversample_ratio=${OVERSAMPLE_RATIO}"
}

# teacher_args <path>：单 teacher OPD reward 配置块。
teacher_args() {
    echo "\
        $(_teacher_common_args) \
        distillation.fsdp_teacher.model_path=$1"
}

# multi_teacher_args <name:path> [name:path ...]：MOPD reward 配置块。
#
# teachers/ability_to_teacher 不在结构化配置中，因此需要 `++` 和 Hydra 的裸键
# 语法；JSON 引号会导致解析失败。这里不允许空格。
multi_teacher_args() {
    local entries="" spec
    for spec in "$@"; do
        entries="${entries}${entries:+,}{name:${spec%%:*},path:${spec#*:}}"
    done
    echo "\
        $(_teacher_common_args) \
        distillation.fsdp_teacher.strict_routing=${STRICT_ROUTING:-True} \
        ++distillation.fsdp_teacher.teachers=[${entries}] \
        ++distillation.fsdp_teacher.ability_to_teacher=${ABILITY_TO_TEACHER}"
}

_teacher_common_args() {
    echo "\
        distillation.fsdp_teacher.enabled=True \
        distillation.fsdp_teacher.micro_batch_size_per_gpu=${TEACHER_MICRO_BSZ:-24} \
        distillation.fsdp_teacher.dtype=${MODEL_DTYPE} \
        distillation.fsdp_teacher.teacher_temperature=${TEACHER_TEMPERATURE} \
        distillation.fsdp_teacher.log_prob_top_k=${LOG_PROB_TOP_K} \
        distillation.fsdp_teacher.metric_top_k=${METRIC_TOP_K} \
        distillation.fsdp_teacher.top_k_strategy=${TOP_K_STRATEGY} \
        distillation.fsdp_teacher.attn_implementation=${ATTN_IMPLEMENTATION} \
        distillation.fsdp_teacher.reward_weight_mode=${REWARD_WEIGHT_MODE} \
        distillation.fsdp_teacher.fsdp_config.param_offload=True"
}

# anchor_args：MOPD anchor（冻结生成器快照 b）配置块。
#
# anchor.* 已在 distillation.yaml 结构化声明，直接 . 覆盖，不走 ++。
# ANCHOR_PATH 为空时默认 actor_rollout_ref.model.path（窗口 0 快照）；
# ANCHOR_TEMPERATURE 应与 rollout 采样温度一致（b 就是该分布）。
anchor_args() {
    local args="\
        distillation.fsdp_teacher.anchor.enabled=True \
        distillation.fsdp_teacher.anchor.from_actor=${ANCHOR_FROM_ACTOR:-True} \
        distillation.fsdp_teacher.anchor.refresh_interval=${ANCHOR_REFRESH_INTERVAL:-0} \
        distillation.fsdp_teacher.anchor.temperature=${ANCHOR_TEMPERATURE:-${TEMPERATURE}} \
        distillation.fsdp_teacher.anchor.micro_batch_size_per_gpu=${ANCHOR_MICRO_BSZ:-0} \
        distillation.fsdp_teacher.anchor.expect_in_train=${ANCHOR_EXPECT_IN_TRAIN:-False} \
        distillation.fsdp_teacher.anchor.top_k=${ANCHOR_TOP_K:-0}"
    if [ -n "${ANCHOR_PATH:-}" ]; then
        args="${args} distillation.fsdp_teacher.anchor.path=${ANCHOR_PATH}"
    fi
    echo "$args"
}

# mopd_loss_args：MOPD 逐状态残差 loss（paper §4.3 形式二/三）的超参块。
#
# POLICY_LOSS_MODE=mopd_resid 用精确 E_b[d^2]（需 anchor.expect_in_train=True），
# mopd_grpo 用 BPO 同形的 -2 eta sg[A] M min{sg[omega],C} log pi_theta
# （只需 anchor.enabled=True）。mopd_* 已在 actor.yaml 结构化声明，直接 . 覆盖。
mopd_loss_args() {
    local args="\
        actor_rollout_ref.actor.policy_loss.mopd_eta=${MOPD_ETA:-1.0} \
        actor_rollout_ref.actor.policy_loss.mopd_epsilon=${MOPD_EPSILON:-0.05} \
        actor_rollout_ref.actor.policy_loss.mopd_cap=${MOPD_CAP:-3.0} \
        actor_rollout_ref.actor.policy_loss.mopd_lambda=${MOPD_LAMBDA:-0.0} \
        actor_rollout_ref.actor.policy_loss.mopd_is_cap=${MOPD_IS_CAP:-20.0}"
    if [ -n "${MOPD_CLIP_LOW:-}" ]; then
        args="${args} actor_rollout_ref.actor.policy_loss.mopd_clip_low=${MOPD_CLIP_LOW}"
    fi
    if [ -n "${MOPD_CLIP_HIGH:-}" ]; then
        args="${args} actor_rollout_ref.actor.policy_loss.mopd_clip_high=${MOPD_CLIP_HIGH}"
    fi
    echo "$args"
}

# mopd_replay_args：M3 跨窗口 replay buffer + P-C 优先级 sampler（paper §4.4/§4.6）。
#
# mopd_replay.* 已在 distillation.yaml 结构化声明，直接 . 覆盖。
# MOPD_PRIORITY_MODE=marginal_absorption 需 anchor.enabled（state_mu_eff/
# state_resid_energy/state_gap 都来自 anchor 配对前向）；uniform 可裸跑。
mopd_replay_args() {
    echo "\
        distillation.mopd_replay.enabled=${MOPD_REPLAY_ENABLED:-True} \
        distillation.mopd_replay.window=${MOPD_REPLAY_WINDOW:-4} \
        distillation.mopd_replay.priority_mode=${MOPD_PRIORITY_MODE:-marginal_absorption} \
        distillation.mopd_replay.coverage_rarity=${MOPD_COVERAGE_RARITY:-0.0}"
}

# launch <experiment name> <override>...：运行 main_ppo 并将日志 tee 到文件。
#
# OPD_DRY_RUN=1 只组合并打印配置，不启动 ray。`--cfg job` 必须放在前面：
# Hydra 的 argparse 遇到第一个 flag 后会停止收集 override。
launch() {
    local exp="$1"
    shift
    set -- "$@" \
        trainer.project_name="$PROJECT_NAME" \
        trainer.experiment_name="$exp" \
        trainer.default_local_dir="${CKPT_PATH:-${OPD_ROOT}/checkpoints/${exp}}"

    if [ -n "${OPD_DRY_RUN:-}" ]; then
        "$PYTHON_BIN" -m verl.trainer.main_ppo --cfg job "$@"
        return
    fi

    "$PYTHON_BIN" -m verl.trainer.main_ppo "$@" \
        2>&1 | tee "${OPD_ROOT}/logs/${exp}.log"
}
