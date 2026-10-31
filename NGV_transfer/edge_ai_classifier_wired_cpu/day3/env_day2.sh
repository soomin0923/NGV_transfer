#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then echo 'Use: source day3/env_day2.sh'; exit 2; fi
_EDGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$_EDGE_ROOT/.venv/bin/activate" ]] || { echo 'Run bash wsl/setup_wsl.sh first.'; return 1; }
if declare -F deactivate >/dev/null; then deactivate; fi
source "$_EDGE_ROOT/.venv/bin/activate"
export CUDA_VISIBLE_DEVICES=-1
export TF_ENABLE_ONEDNN_OPTS=0
unset _EDGE_ROOT
