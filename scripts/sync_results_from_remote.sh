#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REMOTE_REPO_DIR="${OPD_REMOTE_REPO_DIR:-/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-poistar/fanghaotian/OPD-Shot-OPD}"
REMOTE_RESULTS_DIR="${OPD_REMOTE_RESULTS_DIR:-${REMOTE_REPO_DIR}/justrl_eval_outputs}"
RESULT_SNAP_DIR="${OPD_RESULT_SNAP_DIR:-${REMOTE_REPO_DIR}/result_snap}"
LOCAL_RESULTS_DIR="${OPD_LOCAL_RESULTS_DIR:-${LOCAL_REPO_DIR}/results}"

MAX_FILE_MB=100
EXCLUDE_BASENAMES_CSV=""
CHUNK_SIZE=10485760

usage() {
  cat <<'USAGE'
Usage:
  # Step 1 (run on remote server): pack OPD results
  bash scripts/sync_results_from_remote.sh pack [archive_name]

  # Step 2 (after manual download, run locally): unpack and merge local OPD results
  bash scripts/sync_results_from_remote.sh unpack [archive_or_part]

Description:
  - No SSH sync. This script only supports manual transfer.
  - pack:
      1) packs new or modified files since the previous pack
      2) ignores files larger than MAX_FILE_MB
      3) optionally ignores basenames in EXCLUDE_BASENAMES_CSV
      4) builds tar.gz; if final archive > CHUNK_SIZE, splits to .part.000/.001/...
  - unpack:
      1) auto picks latest OPD_results*.tar.gz from ~/Downloads, then current dir, when input omitted
      2) extracts archive (or rebuilds from .part.000 + parts)
      3) incrementally merges files into local justrl_eval_outputs
USAGE
}

tar_supports_flag() {
  tar --help 2>&1 | grep -Fq -- "$1"
}

file_signature() {
  local file="$1"
  if stat -c '%s:%y' -- "$file" >/dev/null 2>&1; then
    stat -c '%s:%y' -- "$file"
  else
    stat -f '%z:%m' "$file"
  fi
}

find_latest_results_archive() {
  local search_dir="$1"
  local -a candidate_archives=()
  local latest_archive

  shopt -s nullglob
  # A split archive is represented by its first part; unpack will collect all
  # matching parts after selecting the .part.000 file.
  candidate_archives=("$search_dir"/OPD_results*.tar.gz "$search_dir"/OPD_results*.tar.gz.part.000)
  shopt -u nullglob

  if [[ ${#candidate_archives[@]} -eq 0 ]]; then
    return 1
  fi

  latest_archive="$(ls -1t "${candidate_archives[@]}" 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest_archive" ]]; then
    return 1
  fi

  printf '%s\n' "$latest_archive"
}

find_latest_results_archive_in_dirs() {
  local search_dir latest_archive
  for search_dir in "$@"; do
    [[ -z "$search_dir" ]] && continue
    if latest_archive="$(find_latest_results_archive "$search_dir")"; then
      printf '%s\n' "$latest_archive"
      return 0
    fi
  done

  return 1
}

pack_results() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  if [[ $# -gt 1 ]]; then
    usage
    exit 1
  fi

  if ! [[ "$MAX_FILE_MB" =~ ^[0-9]+$ ]] || (( MAX_FILE_MB <= 0 )); then
    echo "Error: MAX_FILE_MB must be a positive integer, got: $MAX_FILE_MB" >&2
    exit 1
  fi
  if ! [[ "$CHUNK_SIZE" =~ ^[0-9]+$ ]] || (( CHUNK_SIZE <= 0 )); then
    echo "Error: CHUNK_SIZE must be a positive integer (bytes), got: $CHUNK_SIZE" >&2
    exit 1
  fi

  if [[ ! -d "$REMOTE_REPO_DIR" ]]; then
    echo "Error: remote repo dir not found: $REMOTE_REPO_DIR" >&2
    exit 1
  fi
  if [[ ! -d "$REMOTE_RESULTS_DIR" ]]; then
    echo "Error: remote results dir not found: $REMOTE_RESULTS_DIR" >&2
    exit 1
  fi

  local archive_arg archive_path
  archive_arg="${1:-OPD_results_latest.tar.gz}"

  if [[ "$archive_arg" = /* ]]; then
    archive_path="$archive_arg"
  else
    archive_path="$RESULT_SNAP_DIR/$(basename -- "$archive_arg")"
  fi

  if [[ "$archive_path" != *.tar.gz ]]; then
    archive_path="${archive_path}.tar.gz"
  fi

  local archive_dir
  archive_dir="$(dirname "$archive_path")"
  mkdir -p "$archive_dir"
  rm -f "$archive_path" "${archive_path}.part."[0-9][0-9][0-9]

  local remote_parent
  remote_parent="$(dirname "$REMOTE_REPO_DIR")"

  local -a exclude_basenames=()
  if [[ -n "$EXCLUDE_BASENAMES_CSV" ]]; then
    local -a raw_exclude_names
    local raw_name trimmed_name
    IFS=',' read -r -a raw_exclude_names <<< "$EXCLUDE_BASENAMES_CSV"
    for raw_name in "${raw_exclude_names[@]}"; do
      trimmed_name="${raw_name#"${raw_name%%[![:space:]]*}"}"
      trimmed_name="${trimmed_name%"${trimmed_name##*[![:space:]]}"}"
      if [[ -n "$trimmed_name" ]]; then
        exclude_basenames+=("$trimmed_name")
      fi
    done
  fi

  local max_file_bytes
  max_file_bytes=$((MAX_FILE_MB * 1024 * 1024))

  local tmp_file_list all_files_list next_state_file
  tmp_file_list="/tmp/OPD_results_pack_files_$(date +%s)_$$.txt"
  all_files_list="/tmp/OPD_results_pack_all_files_$(date +%s)_$$.txt"
  next_state_file="/tmp/OPD_results_pack_state_$(date +%s)_$$.txt"
  rm -f "$tmp_file_list" "$all_files_list" "$next_state_file"

  echo "Scanning source files..."
  find "$REMOTE_RESULTS_DIR" -type f | LC_ALL=C sort -u > "$all_files_list"

  local total_files kept_files skipped_large_files skipped_name_files
  total_files=$(wc -l < "$all_files_list" | tr -d '[:space:]')
  kept_files=0
  skipped_large_files=0
  skipped_name_files=0

  mkdir -p "$RESULT_SNAP_DIR"
  local state_file="$RESULT_SNAP_DIR/results_state_OPD.tsv"
  local abs_file rel_file base_name excluded_name file_bytes is_excluded signature state_entry
  while IFS= read -r abs_file; do
    [[ -z "$abs_file" ]] && continue

    base_name="$(basename "$abs_file")"
    is_excluded=0
    for excluded_name in "${exclude_basenames[@]:-}"; do
      if [[ "$base_name" == "$excluded_name" ]]; then
        is_excluded=1
        break
      fi
    done

    if (( is_excluded == 1 )); then
      skipped_name_files=$((skipped_name_files + 1))
      continue
    fi

    file_bytes=$(wc -c < "$abs_file" | tr -d '[:space:]')
    if (( file_bytes > max_file_bytes )); then
      skipped_large_files=$((skipped_large_files + 1))
      continue
    fi

    rel_file="${abs_file#"$remote_parent/"}"
    if [[ "$rel_file" == "$abs_file" ]]; then
      rm -f "$tmp_file_list" "$all_files_list" "$next_state_file"
      echo "Error: file is not under remote parent dir: $abs_file" >&2
      exit 1
    fi

    signature="$(file_signature "$abs_file")"
    state_entry="${rel_file}"$'\t'"${signature}"
    printf '%s\n' "$state_entry" >> "$next_state_file"
    if [[ -f "$state_file" ]] && grep -Fqx -- "$state_entry" "$state_file"; then
      continue
    fi

    printf '%s\n' "$rel_file" >> "$tmp_file_list"
    kept_files=$((kept_files + 1))
  done < "$all_files_list"

  rm -f "$all_files_list"

  if (( kept_files == 0 )); then
    rm -f "$tmp_file_list" "$next_state_file"
    echo "Nothing to package: no new or updated files matched pack filters."
    return 0
  fi

  local -a tar_create_opts
  tar_create_opts=()
  if tar_supports_flag "--no-xattrs"; then
    tar_create_opts+=(--no-xattrs)
  fi
  if tar_supports_flag "--no-acls"; then
    tar_create_opts+=(--no-acls)
  fi
  if tar_supports_flag "--no-mac-metadata"; then
    tar_create_opts+=(--no-mac-metadata)
  fi
  if tar_supports_flag "--disable-copyfile"; then
    tar_create_opts+=(--disable-copyfile)
  fi

  echo "Packing: $REMOTE_RESULTS_DIR"
  echo "Files: total=$total_files kept=$kept_files skipped_large=$skipped_large_files skipped_by_name=$skipped_name_files"
  echo "Archive: $archive_path"

  if (( ${#tar_create_opts[@]} > 0 )); then
    COPYFILE_DISABLE=1 tar "${tar_create_opts[@]}" -czf "$archive_path" -C "$remote_parent" -T "$tmp_file_list"
  else
    COPYFILE_DISABLE=1 tar -czf "$archive_path" -C "$remote_parent" -T "$tmp_file_list"
  fi

  rm -f "$tmp_file_list"

  local archive_bytes
  archive_bytes=$(wc -c < "$archive_path" | tr -d '[:space:]')

  if (( archive_bytes <= CHUNK_SIZE )); then
    echo "Created archive: $archive_path"
    echo "Archive size: ${archive_bytes} bytes"
    echo "OUTPUT_ARCHIVE=$archive_path"
    mv "$next_state_file" "$state_file"
    echo "Next: manually download this archive to your local machine, then run unpack."
    return
  fi

  if ! command -v split >/dev/null 2>&1; then
    echo "Error: archive size ${archive_bytes} bytes exceeds CHUNK_SIZE=${CHUNK_SIZE}, but 'split' is not available." >&2
    echo "Generated unsplit archive: $archive_path" >&2
    exit 1
  fi

  local part_prefix
  part_prefix="${archive_path}.part."
  rm -f "${part_prefix}"*
  split -b "$CHUNK_SIZE" -d -a 3 "$archive_path" "$part_prefix"
  rm -f "$archive_path"
  mv "$next_state_file" "$state_file"

  local part_count
  part_count=$(ls -1 "${part_prefix}"* | wc -l | tr -d '[:space:]')
  echo "Archive size ${archive_bytes} bytes > CHUNK_SIZE=${CHUNK_SIZE}, split into ${part_count} parts:"
  ls -lh "${part_prefix}"*
  echo "OUTPUT_PART_PREFIX=${part_prefix}"
  echo "Next: manually download all .part.* files, then run unpack with .part.000 (or prefix path)."
}

unpack_results() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  if [[ $# -gt 1 ]]; then
    usage
    exit 1
  fi

  local input_arg="" input_path=""
  if [[ $# -eq 0 ]]; then
    local downloads_dir=""
    if [[ -n "${HOME:-}" ]]; then
      downloads_dir="${HOME}/Downloads"
    fi

    if ! input_path="$(find_latest_results_archive_in_dirs "$downloads_dir" "$(pwd)")"; then
      if [[ -n "$downloads_dir" ]]; then
        echo "Error: no OPD_results archive or split parts found in ~/Downloads or current dir: $(pwd)" >&2
      else
        echo "Error: no OPD_results archive or split parts found in current dir: $(pwd)" >&2
      fi
      exit 1
    fi
    echo "No unpack input provided. Using latest archive: $input_path"
  else
    input_arg="$1"
    if [[ "$input_arg" = /* ]]; then
      input_path="$input_arg"
    else
      input_path="$(pwd)/$input_arg"
    fi

    if [[ "$input_arg" != *.tar.gz && ! "$input_arg" =~ \.part\.[0-9]{3}$ ]]; then
      local -a matching_parts=()
      shopt -s nullglob
      matching_parts=("${input_path}"[0-9][0-9][0-9])
      shopt -u nullglob

      if [[ ${#matching_parts[@]} -eq 0 ]]; then
        if ! input_path="$(find_latest_results_archive "$(pwd)")"; then
          echo "Error: input is not an archive or split part, and no OPD_results archive was found in current dir: $(pwd)" >&2
          exit 1
        fi
        echo "Input '$input_arg' is not .tar.gz. Using latest archive: $input_path"
      fi
    fi
  fi

  local temp_archive=""
  local temp_extract_dir=""
  local archive_path=""
  local -a source_archives=()

  cleanup() {
    [[ -n "${temp_archive:-}" && -f "${temp_archive:-}" ]] && rm -f "${temp_archive:-}"
    [[ -n "${temp_extract_dir:-}" && -d "${temp_extract_dir:-}" ]] && rm -rf "${temp_extract_dir:-}"
    return 0
  }
  trap cleanup EXIT

  if [[ -f "$input_path" && "$input_path" == *.tar.gz ]]; then
    archive_path="$input_path"
    source_archives=("$input_path")
  else
    local part_prefix
    if [[ -f "$input_path" && "$input_path" =~ \.part\.[0-9]{3}$ ]]; then
      part_prefix="${input_path%[0-9][0-9][0-9]}"
    else
      part_prefix="$input_path"
    fi

    local -a part_files
    shopt -s nullglob
    part_files=("${part_prefix}"[0-9][0-9][0-9])
    shopt -u nullglob

    if [[ ${#part_files[@]} -eq 0 ]]; then
      echo "Error: cannot find archive or split parts from input: $input_arg" >&2
      exit 1
    fi

    temp_archive="/tmp/OPD_results_unpack_$(date +%s)_$$.tar.gz"
    cat "${part_files[@]}" > "$temp_archive"
    archive_path="$temp_archive"
    source_archives=("${part_files[@]}")
    echo "Rebuilt archive from ${#part_files[@]} part files."
  fi

  temp_extract_dir="/tmp/OPD_results_extract_$(date +%s)_$$"
  mkdir -p "$temp_extract_dir"

  local -a tar_extract_opts
  tar_extract_opts=()
  if tar_supports_flag "--warning"; then
    tar_extract_opts+=(--warning=no-unknown-keyword)
  fi

  if (( ${#tar_extract_opts[@]} > 0 )); then
    tar "${tar_extract_opts[@]}" -xzf "$archive_path" -C "$temp_extract_dir"
  else
    tar -xzf "$archive_path" -C "$temp_extract_dir"
  fi

  local extracted_results
  extracted_results="$temp_extract_dir/$(basename "$REMOTE_REPO_DIR")/results"
  [[ -d "$extracted_results" ]] || extracted_results="$temp_extract_dir/$(basename "$REMOTE_REPO_DIR")/justrl_eval_outputs"

  if [[ ! -d "$extracted_results" ]]; then
    echo "Error: cannot find extracted OPD results in archive." >&2
    exit 1
  fi

  mkdir -p "$LOCAL_RESULTS_DIR"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --update "$extracted_results"/ "$LOCAL_RESULTS_DIR"/
  else
    cp -a "$extracted_results"/. "$LOCAL_RESULTS_DIR"/
  fi

  local result_files
  result_files=$(find "$LOCAL_RESULTS_DIR" -type f | wc -l | tr -d '[:space:]')

  echo "Unpack complete."
  echo "Merged local dir:"
  echo "  - $LOCAL_RESULTS_DIR (files: $result_files)"

  if [[ ${#source_archives[@]} -gt 0 ]]; then
    local archive_file removed_source_archives=0
    for archive_file in "${source_archives[@]}"; do
      if [[ -f "$archive_file" ]]; then
        rm -f "$archive_file"
        removed_source_archives=$((removed_source_archives + 1))
      fi
    done

    if (( removed_source_archives > 0 )); then
      echo "Removed source archive file(s): $removed_source_archives"
    fi
  fi
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

case "$1" in
  pack)
    shift
    pack_results "$@"
    ;;
  unpack)
    shift
    unpack_results "$@"
    ;;
  -h|--help)
    usage
    ;;
  *)
    echo "Error: unknown command: $1" >&2
    usage
    exit 1
    ;;
esac
