#!/bin/bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Install the checkout's pinned CPython 3.10 PTOAS wheel in its own environment.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/pypto-venv.sh"
PYPTO_HOME="${PYPTO_HOME:-/opt/pypto}"
PYPTO_SRC="${PYPTO_SRC:-$(cd "$here/../.." && pwd)}"
PTOAS_ROOT="${PTOAS_ROOT:-$PYPTO_HOME/ptoas}"
# Prefer the mounted checkout pin. The image copy bootstraps the deps stage
# before a PyPTO checkout exists.
toolchain_env="$PYPTO_SRC/toolchain/versions.env"
if [ ! -f "$toolchain_env" ]; then
    toolchain_env="$PYPTO_HOME/toolchain/versions.env"
fi
force=false
while [ $# -gt 0 ]; do
    case "$1" in
        --toolchain-env)
            [ $# -ge 2 ] || { echo "--toolchain-env requires a path" >&2; exit 2; }
            toolchain_env="$2"; shift 2 ;;
        --force) force=true; shift ;;
        -h|--help)
            echo "Usage: $0 [--toolchain-env FILE] [--force]"
            echo 'Environment: PYPTO_SRC, PYPTO_HOME, PTOAS_ROOT, PYPTO_PYTHON'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

[ -r "$toolchain_env" ] || { echo "Missing toolchain pin: $toolchain_env" >&2; exit 1; }
version="$(sed -n 's/^PTOAS_VERSION=//p' "$toolchain_env")"
[[ "$version" =~ ^v[0-9]+(\.[0-9]+)+$ ]] || { echo "Invalid PTOAS_VERSION: $version" >&2; exit 1; }
arch="$(uname -m)"
case "$arch" in
    aarch64) sha_key=PTOAS_SHA256_AARCH64 ;;
    x86_64) sha_key=PTOAS_SHA256_X86_64 ;;
    *) echo "Unsupported PTOAS architecture: $arch" >&2; exit 1 ;;
esac
sha="$(sed -n "s/^${sha_key}=//p" "$toolchain_env")"
[[ "$sha" =~ ^[0-9a-f]{64}$ ]] || { echo "Invalid $sha_key in $toolchain_env" >&2; exit 1; }

# Normalize the executable banner ("ptoas [version] [v]X.Y") before comparing
# it with the v-prefixed pin; comparing raw strings reinstalls on every update.
# The standalone installer also runs in older images without our activation.
# Ignore inherited CANN/PyPTO import paths so both probes inspect this venv's wheel.
if ! $force && pypto_venv_is_compatible "$PTOAS_ROOT" &&
    output="$(PYTHONPATH= "$PTOAS_ROOT/bin/ptoas" --version 2>/dev/null)" &&
    [[ "$output" =~ ptoas[[:space:]]+(version[[:space:]]+)?v?([0-9]+(\.[0-9]+)+) ]] &&
    [ "${BASH_REMATCH[2]}" = "${version#v}" ]; then
    echo "PTOAS $version already installed at $PTOAS_ROOT"
    exit 0
fi

python="${PYPTO_PYTHON:-$PYPTO_HOME/venv/bin/python}"
pypto_python_is_310 "$python" || {
    echo "PTOAS wheels require CPython 3.10 at $python" >&2; exit 1;
}
command -v uv >/dev/null || { echo "uv is required; run pypto-setup.sh first" >&2; exit 1; }

download_dir="$(mktemp -d)"
trap 'rm -rf "$download_dir"' EXIT
wheel="ptoas-${version#v}-cp310-cp310-manylinux_2_34_${arch}.whl"
curl --fail --location --retry 3 --retry-all-errors \
    "https://github.com/hw-native-sys/PTOAS/releases/download/${version}/${wheel}" \
    -o "$download_dir/$wheel"
printf '%s  %s\n' "$sha" "$download_dir/$wheel" | sha256sum -c -
# Verify the download before replacing an incompatible/incomplete environment.
pypto_ensure_venv "$PTOAS_ROOT" "$python"
uv pip install --python "$PTOAS_ROOT/bin/python" --force-reinstall "$download_dir/$wheel"
PYTHONPATH= "$PTOAS_ROOT/bin/ptoas" --version
