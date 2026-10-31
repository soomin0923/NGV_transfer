#!/usr/bin/env bash
# Read-only: no Python, apt, sudo, or network required.
set -euo pipefail
fail() { printf 'STOP: %s\n' "$*" >&2; exit 1; }
[[ "$(uname -s)" == Linux ]] || fail 'Run this command INSIDE WSL Ubuntu.'
[[ -r /etc/os-release ]] || fail 'Cannot identify the Linux distribution.'
. /etc/os-release
printf 'OS: %s\nArchitecture: %s\nKernel: %s\n' "${PRETTY_NAME:-unknown}" "$(uname -m)" "$(uname -r)"
[[ "${ID:-}" == ubuntu ]] || fail 'This installer supports existing Ubuntu on x86_64 WSL2. Do not run it inside Windows PowerShell, Pi, or another distribution.'
[[ "$(uname -m)" == x86_64 ]] || fail 'This installer is for x86_64 laptops. ARM64 WSL needs a separate compatible ML stack.'
libc="$(getconf GNU_LIBC_VERSION 2>/dev/null || true)"
[[ "$libc" =~ ^glibc[[:space:]]+([0-9]+)\.([0-9]+) ]] || fail 'Cannot determine glibc version.'
major="${BASH_REMATCH[1]}"; minor="${BASH_REMATCH[2]}"
printf 'C library: %s\n' "$libc"
(( major > 2 || (major == 2 && minor >= 27) )) || fail 'glibc 2.27 or later is required by the selected ML packages. This older OS cannot use this package set; installing a different Python alone cannot fix it. Ask the instructor for a compatible prepared machine.'
grep -Eq '^flags[[:space:]]*:.*[[:space:]]avx([[:space:]]|$)' /proc/cpuinfo || fail 'AVX CPU instructions are required by this TensorFlow build. This machine cannot use the selected TensorFlow package.'
printf 'CPU AVX: present\n'
if ! grep -qi microsoft /proc/sys/kernel/osrelease; then
  printf 'NOTE: this appears to be ordinary Linux. In class, Windows must show VERSION 2 in wsl --list --verbose.\n'
fi
if [[ "${1:-}" == '--require-libraries' ]]; then
  cache="$(/sbin/ldconfig -p 2>/dev/null || ldconfig -p 2>/dev/null || true)"
  gomp="$(awk '/libgomp.so.1 .*x86-64/{print $NF;exit}' <<< "$cache")"
  stdcpp="$(awk '/libstdc\+\+.so.6 .*x86-64/{print $NF;exit}' <<< "$cache")"
  [[ -n "$gomp" && -f "$gomp" && -n "$stdcpp" && -f "$stdcpp" ]] || fail 'Missing CPU runtime libraries. Run: bash setup_laptop.sh'
  grep -aFq 'GLIBCXX_3.4.22' "$stdcpp" || fail 'libstdc++ is too old. Run tools/install_wsl_os_deps.sh with working Ubuntu repositories; do not replace libc manually.'
  printf 'CPU runtime libraries: libgomp + libstdc++ (GLIBCXX_3.4.22) present\n'
elif [[ "${1:-}" != '' ]]; then
  fail 'Usage: bash tools/check_wsl_host.sh [--require-libraries]'
fi
printf 'HOST CHECK PASS: Ubuntu release number need not be 24.04. System Python is not used or changed.\n'
