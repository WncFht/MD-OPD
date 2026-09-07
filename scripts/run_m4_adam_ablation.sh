#!/usr/bin/env bash
# M4 Adam 消融 wrapper（docs/2026-09-06-absorption-rate-spectral-theory.md §6 M4）。
#
# 在 §6.0 基线（math、full-data、sampled-token、AdamW lr=1e-6 betas=(0.9,0.999)
# eps=1e-8）之上，每个 arm 只改一个优化器参数。学生/教师/数据/rollout 全部沿用
# recipe 的 math 默认值（即基线 run mdxt04lm 的 config），不在这里覆盖模型路径。
# 每个 arm 是一次独立 run（≈13.5 h @ 8 GPU）；all 模式顺序执行，单个 arm 失败
# 不阻塞其余 arm。
#
# 用法：
#     bash scripts/run_m4_adam_ablation.sh                    # 全部 7 个 arm
#     bash scripts/run_m4_adam_ablation.sh c_lr_5e7           # 只跑一个 arm
#     bash scripts/run_m4_adam_ablation.sh --dry-run          # 只打印组合配置，不启动 ray
#     bash scripts/run_m4_adam_ablation.sh --dry-run b_eps_1e4
#
# arm 一览（delta vs §6.0，判读对照见文档 M4 表）：
#     a_beta2_0p95   betas (0.9,0.999) -> (0.9,0.95)   # (a) H3，判读须配 (c) 折叠臂
#     b_eps_1e6      eps 1e-8(torch 默认) -> 1e-6      # (b) H3
#     b_eps_1e4      eps -> 1e-4                      # (b) H3
#     c_lr_5e7       lr 1e-6 -> 5e-7                  # (c) H1 折叠复验 + H3a
#     c_lr_2e6       lr 1e-6 -> 2e-6                  # (c) H1 折叠复验 + H3a
#     c_lr_5e6       lr 1e-6 -> 5e-6                  # (c) H1 折叠复验 + H3a，lr 往大延伸
#     c_lr_1e5       lr 1e-6 -> 1e-5                  # (c) H1 折叠复验 + H3a，lr 往大延伸

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
RECIPE="${REPO_ROOT}/recipe/opd.sh"

if [[ ! -f "${RECIPE}" ]]; then
    echo "[error] recipe not found: ${RECIPE}" >&2
    exit 1
fi

ARM="all"
for a in "$@"; do
    case "$a" in
        --dry-run) export OPD_DRY_RUN=1 ;;
        *) ARM="$a" ;;
    esac
done

TAG=$(date +%m%d-%H%M)

# run <arm> <hydra overrides...>：实验名 m4_<arm>_<日期>，启动 opd.sh。
# override 追加在 common_args 之后，Hydra 同键后者覆盖。
run() {
    local arm="$1"; shift
    echo "[m4] arm=${arm} overrides: $*"
    EXPERIMENT_NAME="m4_${arm}_${TAG}" bash "${RECIPE}" "$@"
}

arm_a_beta2_0p95() {
    run a_beta2_0p95 actor_rollout_ref.actor.optim.betas='[0.9,0.95]'
}

# eps 走 override_optimizer_config 注入 torch AdamW kwargs（optim config 无独立 eps 键）。
arm_b_eps_1e6() {
    run b_eps_1e6 'actor_rollout_ref.actor.optim.override_optimizer_config={eps:1e-6}'
}

arm_b_eps_1e4() {
    run b_eps_1e4 'actor_rollout_ref.actor.optim.override_optimizer_config={eps:1e-4}'
}

arm_c_lr_5e7() {
    run c_lr_5e7 actor_rollout_ref.actor.optim.lr=5e-7
}

arm_c_lr_2e6() {
    run c_lr_2e6 actor_rollout_ref.actor.optim.lr=2e-6
}

arm_c_lr_5e6() {
    run c_lr_5e6 actor_rollout_ref.actor.optim.lr=5e-6
}

arm_c_lr_1e5() {
    run c_lr_1e5 actor_rollout_ref.actor.optim.lr=1e-5
}

ALL_ARMS=(a_beta2_0p95 b_eps_1e6 b_eps_1e4 c_lr_5e7 c_lr_2e6 c_lr_5e6 c_lr_1e5)

if [[ "$ARM" == "all" ]]; then
    failed=()
    for arm in "${ALL_ARMS[@]}"; do
        echo "[m4] === arm ${arm} ==="
        if ! "arm_${arm}"; then
            echo "[m4] [warn] arm ${arm} failed, continuing with the rest" >&2
            failed+=("${arm}")
        fi
    done
    if (( ${#failed[@]} > 0 )); then
        echo "[m4] failed arms: ${failed[*]}" >&2
        exit 1
    fi
else
    known=""
    for arm in "${ALL_ARMS[@]}"; do
        [[ "${arm}" == "${ARM}" ]] && known=1
    done
    if [[ -z "${known}" ]]; then
        echo "[error] unknown arm '${ARM}' (choose one of: ${ALL_ARMS[*]}, or 'all')" >&2
        exit 1
    fi
    "arm_${ARM}"
fi
