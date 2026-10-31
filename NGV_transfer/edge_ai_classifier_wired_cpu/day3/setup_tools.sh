#!/usr/bin/env bash
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/../tools/wsl_common.sh"
ngv_user
[[ $# -le 1 && ( "${1:-}" == '' || "${1:-}" == '--prepared' ) ]] || ngv_fail 'Use the complete laptop installer: bash setup_laptop.sh'
cd "$NGV_ROOT"
bash tools/check_wsl_host.sh --require-libraries
[[ -x .venv/bin/python ]] || ngv_fail 'Install the complete laptop environment first: bash setup_laptop.sh'
if [[ "${1:-}" != '--prepared' ]]; then bash tools/prepare_online_packages.sh; fi
[[ -f "$NGV_BUNDLE/bundle.json" ]] || ngv_fail 'Internet package download is incomplete. Run: bash setup_laptop.sh'
ngv_python
.python/bin/python3 -I tools/wsl_bundle.py verify
ngv_venv "$NGV_ROOT/.venv"
ngv_venv "$NGV_ROOT/.venv_day3_tools"
.venv_day3_tools/bin/python -I -m pip --isolated install --no-index \
  --find-links "$NGV_BUNDLE/wheels/wsl_onnx" --require-hashes -r "$NGV_BUNDLE/wsl_onnx.lock.txt"
.venv_day3_tools/bin/python -I -m pip check
mkdir -p day3/out/environment
.venv_day3_tools/bin/python -I -m pip freeze > day3/out/environment/tools_installed.lock.txt
.venv_day3_tools/bin/python tools/check_installation.py --role tools
printf '\nONNX CPU INSTALL PASS. Next: source day3/env_tools.sh\n'
