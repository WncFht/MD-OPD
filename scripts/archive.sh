#!/usr/bin/env bash
set -eo pipefail

CHUNK_SIZE=$((10 * 1024 * 1024))
KEEP_ARCHIVE_AFTER_UNPACK="${KEEP_ARCHIVE_AFTER_UNPACK:-0}"
WORK_DIR=""
EXTRACT_DIR=""

cleanup() {
  [[ -n "$WORK_DIR" ]] && rm -rf -- "$WORK_DIR"
  [[ -n "$EXTRACT_DIR" ]] && rm -rf -- "$EXTRACT_DIR"
  return 0
}
trap cleanup EXIT

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/archive.sh pack <archive.tar.gz> <path> [path ...]
  bash scripts/archive.sh unpack [archive-or-part] [destination]

When archive-or-part is omitted, unpack first searches ~/Downloads, then the
current directory, and uses the newest archive_*.tar.gz or .part.000 there.
Pack excludes macOS metadata files and extended attributes by default.
Unpack replaces each archived top-level path instead of merging with existing files.
Successfully unpacked archives are deleted unless KEEP_ARCHIVE_AFTER_UNPACK=1.
USAGE
}

tar_supports_option() {
  local option="$1"
  tar "$option" -cf /dev/null --files-from /dev/null >/dev/null 2>&1
}

print_supported_tar_options() {
  local option
  for option in "$@"; do
    if tar_supports_option "$option"; then
      printf '%s\n' "$option"
    fi
  done
}

tar_create_metadata_options() {
  print_supported_tar_options \
    --no-xattrs \
    --no-mac-metadata \
    --no-acls \
    --no-fflags \
    --disable-copyfile
}

tar_extract_metadata_options() {
  print_supported_tar_options --warning=no-unknown-keyword
}

expand_home() {
  local path="$1"
  case "$path" in
    "~") printf '%s\n' "$HOME" ;;
    "~/"*) printf '%s/%s\n' "$HOME" "${path#\~/}" ;;
    *) printf '%s\n' "$path" ;;
  esac
}

find_latest_archive_in_dir() {
  local search_dir="$1"
  local -a candidates=()

  [[ -d "$search_dir" ]] || return 1

  shopt -s nullglob
  candidates=(
    "$search_dir"/archive_*.tar.gz
    "$search_dir"/archive_*.tar.gz.part.000
  )
  shopt -u nullglob

  if [[ ${#candidates[@]} -eq 0 ]]; then
    return 1
  fi

  ls -1t "${candidates[@]}" | sed -n '1p'
}

find_latest_archive() {
  local search_dir latest_archive
  for search_dir in "${HOME}/Downloads" "$(pwd)"; do
    if latest_archive="$(find_latest_archive_in_dir "$search_dir")"; then
      printf '%s\n' "$latest_archive"
      return 0
    fi
  done

  echo "Error: no archive_*.tar.gz or archive_*.tar.gz.part.000 found under ~/Downloads or $(pwd)" >&2
  return 1
}

remove_old_archives() {
  local archive_path="$1"
  local archive_dir
  local candidate
  local removed=0
  local -a old_archives=()
  archive_dir="$(dirname -- "$archive_path")"

  shopt -s nullglob
  old_archives=(
    "$archive_dir"/archive_*.tar.gz
    "$archive_dir"/archive_*.tar.gz.part.*
  )
  shopt -u nullglob

  for candidate in "${old_archives[@]}"; do
    [[ "$candidate" == "$archive_path" ]] && continue
    [[ "$candidate" == "${archive_path}.part."* ]] && continue
    rm -f -- "$candidate"
    removed=$((removed + 1))
  done

  if (( removed > 0 )); then
    echo "Removed $removed old archive file(s) from: $archive_dir"
  fi
}

remove_older_archives() {
  local archive_path="$1"
  shift
  local archive_dir
  local candidate
  local source_archive
  local removed=0
  local -a candidates=()

  archive_dir="$(dirname -- "$archive_path")"
  shopt -s nullglob
  candidates=(
    "$archive_dir"/archive_*.tar.gz
    "$archive_dir"/archive_*.tar.gz.part.*
  )
  shopt -u nullglob

  for candidate in "${candidates[@]}"; do
    for source_archive in "$@"; do
      [[ "$candidate" == "$source_archive" ]] && continue 2
    done
    if [[ "$candidate" -ot "$archive_path" ]]; then
      rm -f -- "$candidate"
      removed=$((removed + 1))
    fi
  done

  if (( removed > 0 )); then
    echo "Removed $removed older archive file(s) from: $archive_dir"
  fi
}

pack() {
  if [[ $# -lt 2 ]]; then
    usage >&2
    exit 2
  fi

  local archive_path
  archive_path="$(expand_home "$1")"
  shift
  [[ "$archive_path" == *.tar.gz ]] || archive_path="${archive_path}.tar.gz"

  local source_path source_name source_parent existing_name
  local -a source_names=()
  local -a tar_args=()
  for source_path in "$@"; do
    if [[ ! -e "$source_path" ]]; then
      echo "Error: path not found: $source_path" >&2
      exit 1
    fi

    source_name="$(basename -- "$source_path")"
    for existing_name in "${source_names[@]:-}"; do
      if [[ "$source_name" == "$existing_name" ]]; then
        echo "Error: duplicate top-level name in archive: $source_name" >&2
        exit 1
      fi
    done
    source_names+=("$source_name")
    source_parent="$(cd -- "$(dirname -- "$source_path")" && pwd)"
    tar_args+=(-C "$source_parent" "$source_name")
  done

  local archive_dir archive_name temp_archive archive_bytes
  local -a create_tar_args=()
  local create_tar_option
  archive_dir="$(dirname -- "$archive_path")"
  archive_name="$(basename -- "$archive_path")"
  mkdir -p "$archive_dir"
  archive_dir="$(cd -- "$archive_dir" && pwd)"
  archive_path="${archive_dir}/${archive_name}"
  WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/opd4rec_archive.XXXXXX")"
  temp_archive="${WORK_DIR}/${archive_name}"

  create_tar_args+=(
    --exclude='.DS_Store'
    --exclude='*/.DS_Store'
    --exclude='._*'
    --exclude='*/._*'
    --exclude='__MACOSX'
    --exclude='__MACOSX/*'
    --exclude='*/__MACOSX'
    --exclude='*/__MACOSX/*'
    --exclude='.AppleDouble'
    --exclude='.AppleDouble/*'
    --exclude='*/.AppleDouble'
    --exclude='*/.AppleDouble/*'
    --exclude='.LSOverride'
    --exclude='*/.LSOverride'
    --exclude='.Spotlight-V100'
    --exclude='.Spotlight-V100/*'
    --exclude='*/.Spotlight-V100'
    --exclude='*/.Spotlight-V100/*'
    --exclude='.TemporaryItems'
    --exclude='.TemporaryItems/*'
    --exclude='*/.TemporaryItems'
    --exclude='*/.TemporaryItems/*'
    --exclude='.Trashes'
    --exclude='.Trashes/*'
    --exclude='*/.Trashes'
    --exclude='*/.Trashes/*'
    --exclude='.fseventsd'
    --exclude='.fseventsd/*'
    --exclude='*/.fseventsd'
    --exclude='*/.fseventsd/*'
  )
  while IFS= read -r create_tar_option; do
    [[ -n "$create_tar_option" ]] && create_tar_args+=("$create_tar_option")
  done < <(tar_create_metadata_options)

  echo "Packing ${#source_names[@]} path(s) into: $archive_path"
  COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 \
    tar "${create_tar_args[@]}" -czf "$temp_archive" "${tar_args[@]}"
  archive_bytes="$(wc -c < "$temp_archive" | tr -d '[:space:]')"

  rm -f "$archive_path" "${archive_path}.part."*
  if (( archive_bytes <= CHUNK_SIZE )); then
    mv "$temp_archive" "$archive_path"
    echo "Created: $archive_path (${archive_bytes} bytes)"
    remove_old_archives "$archive_path"
    return
  fi

  split -b "$CHUNK_SIZE" -d -a 3 "$temp_archive" "${WORK_DIR}/${archive_name}.part."

  local part
  for part in "${WORK_DIR}/${archive_name}.part."*; do
    mv "$part" "$archive_dir/"
  done
  echo "Archive size ${archive_bytes} bytes exceeds 10 MiB; created parts:"
  ls -lh "${archive_path}.part."*
  remove_old_archives "$archive_path"
}

unpack() {
  if [[ $# -gt 2 ]]; then
    usage >&2
    exit 2
  fi

  local input_path
  local destination
  if [[ -n "${1:-}" ]]; then
    input_path="$(expand_home "$1")"
  else
    input_path="$(find_latest_archive)"
    echo "Auto-selected: $input_path"
  fi
  destination="$(expand_home "${2:-.}")"
  local archive_path="$input_path"
  local -a source_archives=()

  if [[ ! -f "$input_path" ]]; then
    echo "Error: archive or part not found: $input_path" >&2
    exit 1
  fi

  if [[ "$input_path" == *.part.* ]]; then
    local part_prefix="${input_path%.*}."
    local -a parts=("${part_prefix}"*)
    if [[ ! -f "${parts[0]}" ]]; then
      echo "Error: no archive parts found for: $input_path" >&2
      exit 1
    fi

    WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/opd4rec_unarchive.XXXXXX")"
    archive_path="${WORK_DIR}/archive.tar.gz"
    cat "${parts[@]}" > "$archive_path"
    source_archives=("${parts[@]}")
    echo "Joined ${#parts[@]} archive parts."
  else
    source_archives=("$input_path")
  fi

  mkdir -p "$destination"
  destination="$(cd -- "$destination" && pwd)"

  local archive_member normalized_member top_level_path existing_top_level
  local -a top_level_paths=()
  while IFS= read -r archive_member; do
    normalized_member="$archive_member"
    while [[ "$normalized_member" == ./* ]]; do
      normalized_member="${normalized_member#./}"
    done
    normalized_member="${normalized_member%/}"
    [[ -z "$normalized_member" ]] && continue

    if [[ "$normalized_member" == /* || "/$normalized_member/" == */../* ]]; then
      echo "Error: unsafe path in archive: $archive_member" >&2
      exit 1
    fi

    top_level_path="${normalized_member%%/*}"
    [[ -z "$top_level_path" || "$top_level_path" == "." || "$top_level_path" == ".." ]] && {
      echo "Error: unsafe top-level path in archive: $archive_member" >&2
      exit 1
    }

    for existing_top_level in "${top_level_paths[@]:-}"; do
      [[ "$top_level_path" == "$existing_top_level" ]] && continue 2
    done
    top_level_paths+=("$top_level_path")
  done < <(tar -tzf "$archive_path")

  if [[ ${#top_level_paths[@]} -eq 0 ]]; then
    echo "Error: archive contains no files or directories." >&2
    exit 1
  fi

  EXTRACT_DIR="$(mktemp -d "${destination}/.opd4rec_unpack.XXXXXX")"
  local -a extract_tar_args=()
  local extract_tar_option
  while IFS= read -r extract_tar_option; do
    [[ -n "$extract_tar_option" ]] && extract_tar_args+=("$extract_tar_option")
  done < <(tar_extract_metadata_options)

  COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 \
    tar "${extract_tar_args[@]}" -xzf "$archive_path" -C "$EXTRACT_DIR"

  for top_level_path in "${top_level_paths[@]}"; do
    if [[ ! -e "$EXTRACT_DIR/$top_level_path" && ! -L "$EXTRACT_DIR/$top_level_path" ]]; then
      echo "Error: extracted top-level path not found: $top_level_path" >&2
      exit 1
    fi
  done

  for top_level_path in "${top_level_paths[@]}"; do
    rm -rf -- "$destination/$top_level_path"
    mv -- "$EXTRACT_DIR/$top_level_path" "$destination/$top_level_path"
    echo "Replaced: $destination/$top_level_path"
  done
  rm -rf -- "$EXTRACT_DIR"
  EXTRACT_DIR=""
  echo "Archive paths now match: $destination"

  remove_older_archives "$input_path" "${source_archives[@]}"

  if [[ "$KEEP_ARCHIVE_AFTER_UNPACK" == "1" ]]; then
    echo "Kept source archive file(s) because KEEP_ARCHIVE_AFTER_UNPACK=1"
  else
    local source_archive
    for source_archive in "${source_archives[@]}"; do
      rm -f -- "$source_archive"
    done
    echo "Removed source archive file(s): ${#source_archives[@]}"
  fi
}

case "${1:-}" in
  pack)
    shift
    pack "$@"
    ;;
  unpack)
    shift
    unpack "$@"
    ;;
  -h|--help|"")
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
