#!/usr/bin/env bash
# Online installation step: the installed Ubuntu chooses its own apt package versions.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$EUID" -ne 0 ]] || { echo 'Run as your ordinary WSL user, without sudo before bash.'; exit 1; }
bash "$ROOT/tools/check_wsl_host.sh"
echo 'Installing base OS packages from THIS Ubuntu release. System Python and the Ubuntu release are unchanged.'
if ! sudo apt-get update; then
  echo 'STOP: apt repositories could not be refreshed. Resolve the displayed network/repository errors, then rerun bash setup_laptop.sh. For an old/EOL Ubuntu ask the administrator to repair its official repositories; this script does not rewrite repository URLs.' >&2
  exit 1
fi
sudo apt-get install -y ca-certificates curl tar gzip libgomp1 libstdc++6
bash "$ROOT/tools/check_wsl_host.sh" --require-libraries
