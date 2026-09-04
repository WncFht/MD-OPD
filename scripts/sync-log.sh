#!/usr/bin/env bash
set -eo pipefail

# Transfer offline W&B runs without SSH: pack remotely, then unpack locally.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WANDB_DIR="${OPD_WANDB_DIR:-$REPO_ROOT/wandb}"
SNAP_DIR="${OPD_LOG_SNAP_DIR:-$REPO_ROOT/log_snap}"
CHUNK_SIZE="${OPD_LOG_CHUNK_SIZE:-10485760}"
KEEP_ARCHIVE_AFTER_UNPACK="${KEEP_ARCHIVE_AFTER_UNPACK:-0}"

[[ "$WANDB_DIR" = /* ]] || WANDB_DIR="$REPO_ROOT/$WANDB_DIR"
[[ "$SNAP_DIR" = /* ]] || SNAP_DIR="$REPO_ROOT/$SNAP_DIR"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/sync-log.sh pack [archive_name]
  bash scripts/sync-log.sh unpack [archive_or_part]

pack runs on the remote checkout and creates log_One-Shot-OPD_*.tar.gz (or parts).
After a successful pack, older log archives in the snapshot directory are removed.
After downloading the archive, unpack runs locally and syncs W&B.
With no input, unpack searches ~/Downloads and the current directory.
USAGE
}

run_changed() {
  local run="$1" last_file="$2"
  [[ ! -f "$last_file" ]] && return 0
  [[ "$run" -nt "$last_file" ]] && return 0
  find "$run" -type f -newer "$last_file" -print -quit | grep -q . && return 0
  return 1
}

remove_old_log_archives() {
  local archive_path="$1" archive_dir candidate removed=0
  local -a candidates=()
  archive_dir="$(dirname -- "$archive_path")"
  shopt -s nullglob
  candidates=(
    "$archive_dir"/log_One-Shot-OPD_*.tar.gz
    "$archive_dir"/log_One-Shot-OPD_*.tar.gz.part.*
  )
  shopt -u nullglob
  for candidate in "${candidates[@]}"; do
    [[ "$candidate" == "$archive_path" ]] && continue
    [[ "$candidate" == "${archive_path}.part."* ]] && continue
    rm -f -- "$candidate"
    removed=$((removed + 1))
  done
  if (( removed > 0 )); then
    echo ">>> Removed $removed old log archive file(s) from: $archive_dir"
  fi
}

pack_logs() {
  cd "$REPO_ROOT"
  [[ -d "$WANDB_DIR" ]] || {
    echo "Error: W&B directory does not exist: $WANDB_DIR" >&2
    exit 1
  }
  [[ "$CHUNK_SIZE" =~ ^[0-9]+$ ]] && (( CHUNK_SIZE > 0 )) || {
    echo "Error: chunk size must be positive" >&2
    exit 2
  }
  mkdir -p "$SNAP_DIR"
  local last_file="$SNAP_DIR/.last_sync_logs" tmp_dir archive_path archive_arg found=0 run
  local -a files=()
  for run in "$WANDB_DIR"/offline-run-*; do
    [[ -d "$run" ]] || continue
    if run_changed "$run" "$last_file"; then
      files+=("$run")
      [[ "$run" == "$WANDB_DIR"/* ]] && [[ -f "$run.wandb" ]] && files+=("$run.wandb")
      found=1
    fi
  done
  if (( found == 0 )); then
    echo "Nothing to package: no new or updated offline runs."
    return 0
  fi
  echo ">>> Packaging updated W&B runs:"; printf '%s\n' "${files[@]}"
  tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/opd_logs_pack.XXXXXX")"
  trap 'rm -rf -- "$tmp_dir"' EXIT
  for run in "${files[@]}"; do
    if [[ "$run" == "$WANDB_DIR"/* ]]; then
      mkdir -p "$tmp_dir/wandb"
      cp -a "$run" "$tmp_dir/wandb/"
      [[ -f "$run.wandb" ]] && cp -a "$run.wandb" "$tmp_dir/wandb/"
    fi
  done
  archive_arg="${1:-log_One-Shot-OPD_$(date +%F-%H%M%S).tar.gz}"
  [[ "$archive_arg" == *.tar.gz ]] || archive_arg="${archive_arg}.tar.gz"
  archive_path="$SNAP_DIR/$(basename -- "$archive_arg")"
  rm -f "$archive_path" "$archive_path.part."*
  COPYFILE_DISABLE=1 tar --exclude='.DS_Store' --exclude='._*' --exclude='__MACOSX' -czf "$archive_path" -C "$tmp_dir" .
  local bytes; bytes="$(wc -c < "$archive_path" | tr -d '[:space:]')"
  if (( bytes > CHUNK_SIZE )); then
    split -b "$CHUNK_SIZE" -d -a 3 "$archive_path" "$archive_path.part."
    rm -f "$archive_path"; echo ">>> Created split archive parts:"; ls -lh "$archive_path.part."*
  else
    echo ">>> Created archive: $archive_path ($bytes bytes)"
  fi
  remove_old_log_archives "$archive_path"
  touch "$last_file"
}

find_archive() {
  local dir="$1"; local -a candidates=()
  [[ -d "$dir" ]] || return 1
  shopt -s nullglob; candidates=("$dir"/log_One-Shot-OPD_*.tar.gz "$dir"/log_One-Shot-OPD_*.tar.gz.part.000); shopt -u nullglob
  (( ${#candidates[@]} )) || return 1
  ls -1t "${candidates[@]}" | head -n 1
}

unpack_logs() {
  local input="${1:-}" archive temp_archive="" extract_dir
  if [[ -z "$input" ]]; then
    input="$(find_archive "${HOME:-}/Downloads" || find_archive "$(pwd)")" || {
      echo "Error: no log_One-Shot-OPD archive found" >&2; exit 1;
    }
  fi
  [[ "$input" = /* ]] || input="$(pwd)/$input"
  [[ -f "$input" ]] || { echo "Error: archive not found: $input" >&2; exit 1; }
  if [[ "$input" == *.part.* ]]; then
    local prefix="${input%[0-9][0-9][0-9]}"; local -a parts=()
    shopt -s nullglob; parts=("${prefix}"[0-9][0-9][0-9]); shopt -u nullglob
    (( ${#parts[@]} )) || { echo "Error: archive parts not found" >&2; exit 1; }
    temp_archive="$(mktemp "${TMPDIR:-/tmp}/opd_logs_unpack.XXXXXX.tar.gz")"; cat "${parts[@]}" > "$temp_archive"; archive="$temp_archive"
  else archive="$input"; fi
  extract_dir="$(mktemp -d "${TMPDIR:-/tmp}/opd_logs_extract.XXXXXX")"
  trap 'rm -rf -- "$extract_dir" "$temp_archive"' EXIT
  tar -xzf "$archive" -C "$extract_dir"
  mkdir -p "$WANDB_DIR"
  local sync_rc=0
  if [[ -d "$extract_dir/wandb" ]]; then
    local -a runs_to_sync=()
    local extracted_run
    shopt -s nullglob
    for extracted_run in "$extract_dir/wandb"/offline-run-*; do
      [[ -d "$extracted_run" ]] || continue
      runs_to_sync+=("$WANDB_DIR/$(basename -- "$extracted_run")")
    done
    shopt -u nullglob

    cp -a "$extract_dir/wandb/." "$WANDB_DIR/"
    if command -v wandb >/dev/null 2>&1; then
      for run in "${runs_to_sync[@]}"; do
        echo ">>> wandb sync $run"
        wandb sync "$run" || sync_rc=$?
      done
    else echo ">>> wandb command not found; skipping W&B sync" >&2; fi
  fi
  [[ "$KEEP_ARCHIVE_AFTER_UNPACK" == 1 ]] || {
    if [[ "$input" == *.part.* ]]; then rm -f "${input%[0-9][0-9][0-9]}"[0-9][0-9][0-9]; else rm -f "$input"; fi
  }
  (( sync_rc == 0 )) || exit "$sync_rc"
}

[[ $# -ge 1 ]] || { usage; exit 2; }
case "$1" in
  pack) shift; pack_logs "$@" ;;
  unpack) shift; unpack_logs "$@" ;;
  -h|--help) usage ;;
  *) echo "Error: unknown command: $1" >&2; usage; exit 2 ;;
esac
