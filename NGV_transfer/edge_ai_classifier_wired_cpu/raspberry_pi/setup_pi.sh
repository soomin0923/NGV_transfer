#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$EUID" -ne 0 ]] || { echo 'STOP: run as your ordinary Pi user, without sudo before bash.'; exit 1; }
[[ "$(uname -m)" == aarch64 ]] || { echo 'STOP: install Raspberry Pi OS 64-bit (aarch64), not the 32-bit image.'; exit 1; }
[[ $# -eq 0 ]] || { echo 'Usage: bash setup_pi.sh'; exit 2; }
trap 'echo "STOP: Pi setup did not finish. Read the error above; fix that issue and rerun: bash setup_pi.sh" >&2' ERR
echo '[1/3] Installing Python and camera libraries from this Pi OS release.'
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip python3-opencv python3-numpy python3-pil v4l-utils ca-certificates libgomp1
cd "$ROOT"
echo '[2/3] Creating the Pi environment and downloading its compatible LiteRT package.'
/usr/bin/python3 -c "import sys; assert sys.version_info[:2] in [(3,11),(3,12),(3,13)], 'Pi requires Python 3.11-3.13 for the pinned LiteRT package; report the OS and Python version to the instructor.'"
if [[ ! -e .venv ]]; then /usr/bin/python3 -m venv --system-site-packages .venv; fi
.venv/bin/python - <<'CHECK'
import sys, subprocess
from pathlib import Path
assert sys.version_info[:2] in [(3,11),(3,12),(3,13)], 'This pinned course uses Python 3.11-3.13 on Pi; check LiteRT wheel support before selecting a different OS.'
v=subprocess.check_output(['/usr/bin/python3','-c','import sys; print(sys.version_info[:2])'],text=True).strip()
assert str(sys.version_info[:2])==v, 'System/venv Python mismatch; recreate only .venv.'
assert 'include-system-site-packages = true' in Path('.venv/pyvenv.cfg').read_text(), 'Pi venv must use --system-site-packages'
import cv2,numpy,PIL
print('System cv2/NumPy/Pillow:',cv2.__version__,numpy.__version__)
CHECK
# Pin system NumPy/Pillow versions so pip cannot silently shadow apt OpenCV dependencies.
.venv/bin/python - <<'CONSTRAINTS'
import numpy,PIL
from pathlib import Path
Path('raspberry_pi/system_constraints.txt').write_text(f'numpy=={numpy.__version__}\nPillow=={PIL.__version__}\n')
CONSTRAINTS
.venv/bin/python -m pip --isolated install --index-url https://pypi.org/simple --only-binary=:all: -c raspberry_pi/system_constraints.txt -r raspberry_pi/requirements.txt
mkdir -p dataset_raw models day2_extra/comparison day3_image/results
echo '[3/3] Checking required Pi library imports and versions.'
# PI_FIX_20261008: validate course imports/versions below, not unrelated apt packages.
.venv/bin/python tools/check_installation.py --role pi
.venv/bin/python -m pip freeze > raspberry_pi/installed_versions.txt
if ! id -nG | tr ' ' '\n' | grep -qx video; then
  sudo usermod -aG video "$(id -un)"
  echo 'VIDEO GROUP ADDED: exit this SSH session and reconnect before opening the camera.'
fi
printf '\nPI INSTALL PASS. Next: source .venv/bin/activate\n'
