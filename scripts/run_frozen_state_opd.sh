#!/usr/bin/env bash
# Frozen-state OPD (§5.3): cache the first rollout pool, then on every step
# recompute current-student log-probs on those fixed response tokens.  The
# Supports both no-ratio OPD and frozen-PPO-anchor runs; rollout IS/RS are disabled.
#
# Usage:
#   bash scripts/run_frozen_state_opd.sh                 # all learning rates
#   bash scripts/run_frozen_state_opd.sh lr_5e7           # one no-ratio arm
#   bash scripts/run_frozen_state_opd.sh --ppo lr_1e6     # frozen PPO anchor
#   bash scripts/run_frozen_state_opd.sh --no-ratio all  # explicit default
#   bash scripts/run_frozen_state_opd.sh --dry-run

set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
RECIPE="${REPO_ROOT}/recipe/opd.sh"

ARM="all"
FROZEN_LOSS_MODE="${FROZEN_LOSS_MODE:-opd_noratio}"
for arg in "$@"; do
    case "$arg" in
        --dry-run) export OPD_DRY_RUN=1 ;;
        --ppo) FROZEN_LOSS_MODE=vanilla ;;
        --no-ratio) FROZEN_LOSS_MODE=opd_noratio ;;
        *) ARM="$arg" ;;
    esac
done

TAG=$(date +%m%d-%H%M)

run() {
    local name="$1"; shift
    local mode_tag="noratio"
    [[ "$FROZEN_LOSS_MODE" == "vanilla" ]] && mode_tag="ppo"
    echo "[frozen-opd] mode=${mode_tag} ${name}: $*"
    export POLICY_LOSS_MODE="$FROZEN_LOSS_MODE"
    DOMAIN=math MODE=oneshot \
    MINI_BATCH_SIZE=64 N_RESPONSES=1 \
    LOG_PROB_TOP_K=0 METRIC_TOP_K=16 \
    EXPERIMENT_NAME="frozen_${mode_tag}_${name}_${TAG}" \
    bash "${RECIPE}" \
      algorithm.frozen_rollout=True \
      algorithm.rollout_correction.rollout_is=null \
      algorithm.rollout_correction.rollout_rs=null \
      algorithm.rollout_correction.bypass_mode=False \
      trainer.total_training_steps=${TOTAL_STEPS:-300} \
      "$@"
}

case "$ARM" in
    all)
        run lr_5e7 actor_rollout_ref.actor.optim.lr=5e-7
        run lr_1e6 actor_rollout_ref.actor.optim.lr=1e-6
        run lr_2e6 actor_rollout_ref.actor.optim.lr=2e-6
        run lr_5e6 actor_rollout_ref.actor.optim.lr=5e-6
        ;;
    lr_5e7) run lr_5e7 actor_rollout_ref.actor.optim.lr=5e-7 ;;
    lr_1e6) run lr_1e6 actor_rollout_ref.actor.optim.lr=1e-6 ;;
    lr_2e6) run lr_2e6 actor_rollout_ref.actor.optim.lr=2e-6 ;;
    lr_5e6) run lr_5e6 actor_rollout_ref.actor.optim.lr=5e-6 ;;
    *) echo "unknown arm: $ARM (choose all, lr_5e7, lr_1e6, lr_2e6, lr_5e6)" >&2; exit 2 ;;
esac
