#!/usr/bin/env bash
# Existing Ubuntu: independent CPython; never modify /usr/bin/python3.
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/../tools/wsl_common.sh"
ngv_user
[[ $# -le 1 && ( "${1:-}" == '' || "${1:-}" == '--prepared' ) ]] || ngv_fail 'Use the complete laptop installer: bash setup_laptop.sh'
cd "$NGV_ROOT"
if [[ "${1:-}" != '--prepared' ]]; then
  bash tools/install_wsl_os_deps.sh
  bash tools/prepare_online_packages.sh
fi
bash tools/check_wsl_host.sh --require-libraries
[[ -f "$NGV_BUNDLE/bundle.json" ]] || ngv_fail 'Internet package download is incomplete. Run: bash setup_laptop.sh'
ngv_python
.python/bin/python3 -I tools/wsl_bundle.py verify
ngv_venv "$NGV_ROOT/.venv"
.venv/bin/python -I -m pip --isolated install --no-index --find-links "$NGV_BUNDLE/wheels/wsl_tf" \
  --require-hashes -r "$NGV_BUNDLE/wsl_tf.lock.txt"
.venv/bin/python -I -m pip check
mkdir -p dataset_raw artifacts models
.venv/bin/python -I -m pip freeze > wsl/installed_versions.txt
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python tools/check_installation.py --role train
printf '\nCPU INSTALL PASS. Next: source day3/env_day2.sh\n'
