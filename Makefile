.DEFAULT_GOAL := help

WANDB_DIR ?= wandb
SWANLAB_DIR ?= swanlab
LOG_SNAP_DIR ?= log_snap
LOG_ACTION ?= unpack
LOG_ARCHIVE ?=
ARCHIVE ?=
ARCHIVE_SOURCES ?= Makefile recipe scripts verl data README.md
RESULTS_REMOTE_DIR ?= /mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/OPD-Shot-OPD/results
RESULTS_LOCAL_DIR ?= results
RESULTS_ARCHIVE ?=
ACTION ?= unpack
HOPE_DIR ?= hope/opdonly
HOPE_FILE ?= opdonly.hope

.PHONY: help sync-log sync_log sync-log-pack sync_log_pack sync-log-unpack sync_log_unpack
.PHONY: archive archive-pack archive-unpack opd-submit sync-results sync_results sync-results-pack sync_results_unpack

help:
	@echo "Logs:"
	@echo "  make sync-log LOG_ACTION=pack|unpack [LOG_ARCHIVE=...]"
	@echo "  make sync-log-pack [LOG_ARCHIVE=...]"
	@echo "  make sync-log-unpack [LOG_ARCHIVE=...]"
	@echo "  make opd-submit [HOPE_DIR=hope/opdonly] [HOPE_FILE=opdonly.hope]"
	@echo "  make sync-results ACTION=pack|unpack [RESULTS_ARCHIVE=...]"
	@echo "  make archive-pack [ARCHIVE=archive.tar.gz]"
	@echo "  make archive-unpack ARCHIVE=archive.tar.gz [DEST=output-dir]"

sync-log sync_log:
	@case "$(LOG_ACTION)" in pack|unpack) ;; *) echo "LOG_ACTION must be pack or unpack" >&2; exit 2 ;; esac; \
	OPD_WANDB_DIR="$(WANDB_DIR)" OPD_SWANLAB_DIR="$(SWANLAB_DIR)" OPD_LOG_SNAP_DIR="$(LOG_SNAP_DIR)" \
	bash scripts/sync-log.sh "$(LOG_ACTION)" $(if $(LOG_ARCHIVE),"$(LOG_ARCHIVE)",)

sync-log-pack sync_log_pack:
	@$(MAKE) sync-log LOG_ACTION=pack LOG_ARCHIVE="$(LOG_ARCHIVE)" WANDB_DIR="$(WANDB_DIR)" SWANLAB_DIR="$(SWANLAB_DIR)" LOG_SNAP_DIR="$(LOG_SNAP_DIR)"

sync-log-unpack sync_log_unpack:
	@$(MAKE) sync-log LOG_ACTION=unpack LOG_ARCHIVE="$(LOG_ARCHIVE)" WANDB_DIR="$(WANDB_DIR)" SWANLAB_DIR="$(SWANLAB_DIR)" LOG_SNAP_DIR="$(LOG_SNAP_DIR)"

archive archive-pack:
	@archive="$(ARCHIVE)"; \
	if [ -z "$$archive" ]; then archive="archive_$$(date +%Y%m%d_%H%M%S).tar.gz"; fi; \
	bash scripts/archive.sh pack "$$archive" $(ARCHIVE_SOURCES)

archive-unpack:
	bash scripts/archive.sh unpack "$(ARCHIVE)" "$${DEST:-.}"

opd-submit:
	@test -f "$(HOPE_DIR)/$(HOPE_FILE)" || { echo "Hope file not found: $(HOPE_DIR)/$(HOPE_FILE)" >&2; exit 1; }
	@command -v hope >/dev/null 2>&1 || { echo "hope CLI not found" >&2; exit 1; }
	@cd "$(HOPE_DIR)" && hope run "$(HOPE_FILE)"

sync-results sync_results:
	@case "$(ACTION)" in pack|unpack) ;; *) echo "ACTION must be pack or unpack" >&2; exit 2 ;; esac; \
	OPD_REMOTE_RESULTS_DIR="$(RESULTS_REMOTE_DIR)" OPD_LOCAL_RESULTS_DIR="$(RESULTS_LOCAL_DIR)" \
	bash scripts/sync_results_from_remote.sh "$(ACTION)" $(if $(RESULTS_ARCHIVE),"$(RESULTS_ARCHIVE)",)

sync-results-pack:
	@$(MAKE) sync-results ACTION=pack RESULTS_ARCHIVE="$(RESULTS_ARCHIVE)" RESULTS_REMOTE_DIR="$(RESULTS_REMOTE_DIR)" RESULTS_LOCAL_DIR="$(RESULTS_LOCAL_DIR)"

sync-results-unpack:
	@$(MAKE) sync-results ACTION=unpack RESULTS_ARCHIVE="$(RESULTS_ARCHIVE)" RESULTS_REMOTE_DIR="$(RESULTS_REMOTE_DIR)" RESULTS_LOCAL_DIR="$(RESULTS_LOCAL_DIR)"
