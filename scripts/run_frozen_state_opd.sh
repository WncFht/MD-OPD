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
#   bash scripts/run_frozen_state_opd.sh --mode-tag paper53 lr_1e6
#   MODE_TAG=paper53 bash scripts/run_frozen_state_opd.sh lr_1e6
#   bash scripts/run_frozen_state_opd.sh --dry-run

set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
RECIPE="${REPO_ROOT}/recipe/opd.sh"

ARM="all"
FROZEN_LOSS_MODE="${FROZEN_LOSS_MODE:-opd_noratio}"
MODE_TAG="${MODE_TAG:-}"
DRY_RUN=0
ARGS=()
while (($#)); do
    arg="$1"
    case "$arg" in
        --dry-run)
            export OPD_DRY_RUN=1
            DRY_RUN=1
            ;;
        --ppo)
            FROZEN_LOSS_MODE=vanilla
            ;;
        --no-ratio)
            FROZEN_LOSS_MODE=opd_noratio
            ;;
        --mode-tag=*)
            MODE_TAG="${arg#*=}"
            ;;
        --mode-tag)
            shift
            (($#)) || { echo "--mode-tag requires a value" >&2; exit 2; }
            MODE_TAG="$1"
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
    shift
done
if ((${#ARGS[@]} > 1)); then
    echo "expected at most one arm argument, got: ${ARGS[*]}" >&2
    exit 2
fi
if ((${#ARGS[@]} == 1)); then
    ARM="${ARGS[0]}"
fi
if [[ -z "$MODE_TAG" ]]; then
    MODE_TAG="noratio"
    [[ "$FROZEN_LOSS_MODE" == "vanilla" ]] && MODE_TAG="ppo"
fi
if [[ ! "$MODE_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "invalid mode tag '$MODE_TAG' (use letters, digits, ., _, or -)" >&2
    exit 2
fi

TAG=$(date +%m%d-%H%M)

run() {
    local name="$1"; shift
    echo "[frozen-opd] mode=${MODE_TAG} loss=${FROZEN_LOSS_MODE} ${name}: $*"
    DOMAIN=math MODE=oneshot \
    MINI_BATCH_SIZE=64 N_RESPONSES=1 \
    LOG_PROB_TOP_K=0 METRIC_TOP_K=16 \
    EXPERIMENT_NAME="frozen_${MODE_TAG}_${name}_${TAG}" \
    bash "${RECIPE}" \
      algorithm.frozen_rollout=True \
      algorithm.rollout_correction.rollout_is=null \
      algorithm.rollout_correction.rollout_rs=null \
      algorithm.rollout_correction.bypass_mode=False \
      actor_rollout_ref.actor.policy_loss.loss_mode=${FROZEN_LOSS_MODE} \
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
