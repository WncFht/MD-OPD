.DEFAULT_GOAL := help

WANDB_DIR ?= wandb
SWANLAB_DIR ?= swanlab
LOG_SNAP_DIR ?= log_snap
LOG_ACTION ?= unpack
LOG_ARCHIVE ?=

.PHONY: help sync-log sync_log sync-log-pack sync_log_pack sync-log-unpack sync_log_unpack

help:
	@echo "Logs:"
	@echo "  make sync-log LOG_ACTION=pack|unpack [LOG_ARCHIVE=...]"
	@echo "  make sync-log-pack [LOG_ARCHIVE=...]"
	@echo "  make sync-log-unpack [LOG_ARCHIVE=...]"

sync-log sync_log:
	@case "$(LOG_ACTION)" in pack|unpack) ;; *) echo "LOG_ACTION must be pack or unpack" >&2; exit 2 ;; esac; \
	OPD_WANDB_DIR="$(WANDB_DIR)" OPD_SWANLAB_DIR="$(SWANLAB_DIR)" OPD_LOG_SNAP_DIR="$(LOG_SNAP_DIR)" \
	bash scripts/sync-log.sh "$(LOG_ACTION)" $(if $(LOG_ARCHIVE),"$(LOG_ARCHIVE)",)

sync-log-pack sync_log_pack:
	@$(MAKE) sync-log LOG_ACTION=pack LOG_ARCHIVE="$(LOG_ARCHIVE)" WANDB_DIR="$(WANDB_DIR)" SWANLAB_DIR="$(SWANLAB_DIR)" LOG_SNAP_DIR="$(LOG_SNAP_DIR)"

sync-log-unpack sync_log_unpack:
	@$(MAKE) sync-log LOG_ACTION=unpack LOG_ARCHIVE="$(LOG_ARCHIVE)" WANDB_DIR="$(WANDB_DIR)" SWANLAB_DIR="$(SWANLAB_DIR)" LOG_SNAP_DIR="$(LOG_SNAP_DIR)"
