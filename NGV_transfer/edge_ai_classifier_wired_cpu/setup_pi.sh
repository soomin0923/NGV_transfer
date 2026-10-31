#!/usr/bin/env bash
# One online installation on each Raspberry Pi, without sudo before bash.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/raspberry_pi/setup_pi.sh" "$@"
