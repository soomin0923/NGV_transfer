#!/usr/bin/env bash
# Internal download step used by setup_laptop.sh. No system Python is required.
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/wsl_common.sh"
ngv_user
[[ $# -eq 0 ]] || ngv_fail 'Usage: bash tools/prepare_online_packages.sh'
bash "$NGV_ROOT/tools/check_wsl_host.sh" --require-libraries
for tool in curl tar gzip sha256sum; do
  command -v "$tool" >/dev/null || ngv_fail "Missing $tool. Run: bash setup_laptop.sh"
done
mkdir -p "$NGV_ROOT/.cache"
bootstrap="$(mktemp -d "$NGV_ROOT/.cache/.python-bootstrap.XXXXXX")"
stage=''
previous=''
cleanup() {
  [[ -z "$stage" ]] || rm -rf -- "$stage"
  [[ -z "$bootstrap" ]] || rm -rf -- "$bootstrap"
  if [[ -n "$previous" && ! -e "$NGV_BUNDLE" && -d "$previous" ]]; then mv -- "$previous" "$NGV_BUNDLE"; fi
}
trap cleanup EXIT

# Reuse only a complete, SHA-verified cache for this code's requirements.
cached_archive="$NGV_BUNDLE/python/$NGV_PYTHON_ARCHIVE"
if [[ -f "$NGV_BUNDLE/bundle.json" && -f "$cached_archive" ]]; then
  if (ngv_extract_python "$cached_archive" "$bootstrap") && \
      "$bootstrap/bin/python3" -I "$NGV_ROOT/tools/wsl_bundle.py" verify; then
    printf '\nREUSE: verified internet-download cache: %s\n' "$NGV_BUNDLE"
    exit 0
  fi
  echo 'The previous package cache is incomplete or changed. Downloading a new verified cache.'
  rm -rf -- "$bootstrap"
  bootstrap="$(mktemp -d "$NGV_ROOT/.cache/.python-bootstrap.XXXXXX")"
fi

stage="$(mktemp -d "$NGV_ROOT/.cache/.packages-staging.XXXXXX")"
mkdir -p "$stage/python" "$stage/wheels/wsl_tf" "$stage/wheels/wsl_onnx"
echo 'Downloading official isolated CPython (about 64 MiB); checking its pinned SHA256.'
curl --fail --location --retry 3 --proto '=https' --tlsv1.2 \
  --output "$stage/python/$NGV_PYTHON_ARCHIVE" "$NGV_PYTHON_URL"
ngv_extract_python "$stage/python/$NGV_PYTHON_ARCHIVE" "$bootstrap"
python="$bootstrap/bin/python3"
platform_args=()
for (( abi_minor=27; abi_minor>=5; abi_minor-- )); do platform_args+=(--platform "manylinux_2_${abi_minor}_x86_64"); done
platform_args+=(--platform manylinux2014_x86_64 --platform manylinux2010_x86_64 --platform manylinux1_x86_64)
for group in wsl_tf wsl_onnx; do
  if [[ "$group" == wsl_tf ]]; then requirements="$NGV_ROOT/wsl/requirements.txt"; else requirements="$NGV_ROOT/day3/requirements_tools.txt"; fi
  echo "Downloading $group: CPython 3.12 / x86_64 / glibc <= 2.27 compatible packages."
  "$python" -I -m pip --isolated download --index-url https://pypi.org/simple \
    --only-binary=:all: --implementation cp --python-version 3.12 --abi cp312 --abi abi3 --abi none \
    "${platform_args[@]}" -r "$requirements" -d "$stage/wheels/$group"
done
"$python" -I "$NGV_ROOT/tools/wsl_bundle.py" build --bundle "$stage"
"$python" -I "$NGV_ROOT/tools/wsl_bundle.py" verify --bundle "$stage"
for group in wsl_tf wsl_onnx; do
  "$python" -I -m pip --isolated install --dry-run --ignore-installed --no-index \
    --find-links "$stage/wheels/$group" --require-hashes -r "$stage/$group.lock.txt"
done
if [[ -e "$NGV_BUNDLE" ]]; then
  previous="$NGV_ROOT/.cache/wsl_packages.previous_$(date -u +%Y%m%dT%H%M%S)_$$"
  mv -- "$NGV_BUNDLE" "$previous"
fi
mv -- "$stage" "$NGV_BUNDLE"
stage=''
printf '\nONLINE PACKAGES READY: %s\n' "$NGV_BUNDLE"
[[ -z "$previous" ]] || printf 'Previous cache preserved at: %s\n' "$previous"
