#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then echo 'Use: source day3/env_tools.sh'; exit 2; fi
_EDGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$_EDGE_ROOT/.venv_day3_tools/bin/activate" ]] || { echo 'Run bash day3/setup_tools.sh first.'; return 1; }
if declare -F deactivate >/dev/null; then deactivate; fi
source "$_EDGE_ROOT/.venv_day3_tools/bin/activate"
export CUDA_VISIBLE_DEVICES=-1
unset _EDGE_ROOT
