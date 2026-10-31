#!/usr/bin/env bash
# Shared by the WSL installer. Source this file; do not run it directly.
set -euo pipefail
NGV_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
NGV_PYTHON_VERSION='3.12.15'
NGV_PYTHON_ARCHIVE='cpython-3.12.15+20261003-x86_64-unknown-linux-gnu-install_only.tar.gz'
NGV_PYTHON_URL='https://github.com/astral-sh/python-build-standalone/releases/download/20261003/cpython-3.12.15%2B20261003-x86_64-unknown-linux-gnu-install_only.tar.gz'
NGV_PYTHON_SHA='f937814031eab4698ca6d07ec606ede1825768f3f3e99af76d9db3900bee03c5'
NGV_BUNDLE="$NGV_ROOT/.cache/wsl_packages"
ngv_fail() { printf 'STOP: %s\n' "$*" >&2; exit 1; }
ngv_user() { [[ "$EUID" -ne 0 ]] || ngv_fail 'Run as your ordinary WSL user. Do not put sudo before bash.'; }
ngv_archive_check() {
  local archive="$1" actual
  [[ -f "$archive" ]] || ngv_fail "Missing Python archive: $archive. Run: bash setup_laptop.sh"
  actual="$(sha256sum -- "$archive")"; actual="${actual%% *}"
  [[ "$actual" == "$NGV_PYTHON_SHA" ]] || ngv_fail 'Python archive checksum mismatch. Run bash setup_laptop.sh to download and verify a fresh package cache.'
}
ngv_extract_python() {
  local archive="$1" destination="$2"
  ngv_archive_check "$archive"
  mkdir -p -- "$destination"
  tar -xzf "$archive" --no-same-owner --strip-components=1 -C "$destination"
  "$destination/bin/python3" -I -c 'import sys,ssl,sqlite3,venv; assert sys.version_info[:3]==(3,12,15); print("Portable Python:",sys.version.split()[0])'
}
ngv_python() {
  local marker="$NGV_ROOT/.python/.ngv_archive_sha256" stage
  if [[ -e "$NGV_ROOT/.python" ]]; then
    [[ -f "$marker" && "$(cat "$marker")" == "$NGV_PYTHON_SHA" ]] || ngv_fail 'An unrelated .python already exists. Rename it to .python.before_R5 (do not delete it), then retry.'
    [[ -x "$NGV_ROOT/.python/bin/python3" ]] || ngv_fail 'Portable Python is incomplete. Rename .python to .python.before_R5 and retry.'
    "$NGV_ROOT/.python/bin/python3" -I -c 'import sys; assert sys.version_info[:3]==(3,12,15)' || ngv_fail 'Portable Python does not run on this host.'
    return
  fi
  stage="$(mktemp -d "$NGV_ROOT/.python-staging.XXXXXX")"
  if (ngv_extract_python "$NGV_BUNDLE/python/$NGV_PYTHON_ARCHIVE" "$stage"); then
    printf '%s\n' "$NGV_PYTHON_SHA" > "$stage/.ngv_archive_sha256"
    mv -- "$stage" "$NGV_ROOT/.python"
  else
    rm -rf -- "$stage"
    ngv_fail 'Python extraction failed; the existing project files were preserved.'
  fi
}
ngv_venv() {
  local destination="$1"
  if [[ -e "$destination" ]]; then
    [[ -x "$destination/bin/python" ]] || ngv_fail "$destination is incomplete. Rename it with a .before_R5 suffix, then retry."
    "$destination/bin/python" -I -c 'import pathlib,sys; assert sys.version_info[:3]==(3,12,15); assert pathlib.Path(sys.base_prefix).resolve()==pathlib.Path(sys.argv[1]).resolve()' "$NGV_ROOT/.python" || ngv_fail "Existing $destination uses another Python. Deactivate it and rename it with a .before_R5 suffix, then retry. Models/data are unaffected."
  else
    "$NGV_ROOT/.python/bin/python3" -I -m venv "$destination"
  fi
}
