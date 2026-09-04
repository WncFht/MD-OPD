#!/usr/bin/env bash
# Qwen3-1.7B student + Qwen3-4B teacher OPD wrapper。

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
RECIPE="${REPO_ROOT}/recipe/opd.sh"

if [[ ! -f "${RECIPE}" ]]; then
    echo "[error] recipe not found: ${RECIPE}" >&2
    exit 1
fi

export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_BASE:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/model}/Qwen3-1.7B}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${MODEL_BASE:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/model}/Qwen3-4B}"
export DOMAIN="${DOMAIN:-math}"
export MODE="${MODE:-full}"
export ENABLE_THINKING="${ENABLE_THINKING:-True}"

extra_args=(
    "++data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING}"
)
extra_args+=("$@")

exec bash "${RECIPE}" "${extra_args[@]}"
