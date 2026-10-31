#!/usr/bin/env bash
# One online installation per laptop. Run from existing WSL Ubuntu, without sudo.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$ROOT/tools/wsl_common.sh"
ngv_user
[[ $# -eq 0 ]] || ngv_fail 'Usage: bash setup_laptop.sh'
cd "$ROOT"
trap 'echo "STOP: laptop setup did not finish. Read the error above; fix that issue and rerun: bash setup_laptop.sh" >&2' ERR
case "$ROOT" in
  /mnt/[a-z]/*) ngv_fail 'This is the Windows drive copy. In File Explorer, copy the whole project folder into your WSL Linux home, then run bash setup_laptop.sh there.' ;;
esac

echo '[1/6] Checking this existing Ubuntu and the seven USB data/model files.'
bash tools/check_wsl_host.sh
echo 'Python plan: install project-private Python 3.12.15 in .python; keep Ubuntu and /usr/bin/python3 unchanged.'
missing=0
for name in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5 weights_mobilenet_v3_small_224_1.0_float.h5 imagenet_class_index.json; do
  if [[ ! -s "offline/raw/$name" ]]; then
    printf 'Missing: offline/raw/%s\n' "$name" >&2
    missing=1
  fi
done
[[ "$missing" -eq 0 ]] || ngv_fail 'Copy the seven original files into offline/raw using File Explorer, then rerun bash setup_laptop.sh. Keep the .gz files compressed.'

echo '[2/6] Installing base libraries from this Ubuntu release (sudo password may be requested).'
bash tools/install_wsl_os_deps.sh
echo '[3/6] Downloading compatible private Python and TensorFlow/ONNX packages from the internet.'
bash tools/prepare_online_packages.sh
echo '[4/6] Installing the CPU TensorFlow and TFLite environment.'
bash wsl/setup_wsl.sh --prepared
echo '[5/6] Installing the CPU ONNX environment.'
bash day3/setup_tools.sh --prepared
echo '[6/6] Verifying and importing the seven USB data/model files.'
.venv/bin/python tools/prepare_offline.py --from-raw
.venv/bin/python tools/check_offline.py
printf '\nLAPTOP SETUP PASS. Ubuntu and system Python were not replaced.\nNext: source day3/env_day2.sh\n'
