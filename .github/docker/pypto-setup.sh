#!/bin/bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Provision an existing CANN container using this checkout's scripts and pins.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYPTO_HOME="${PYPTO_HOME:-/opt/pypto}"
export PYPTO_SRC="${PYPTO_SRC:-$(cd "$here/../.." && pwd)}"
torch_version="${TORCH_VERSION:-2.10.0}"
torch_index="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
uv_version="${UV_VERSION:-latest}"
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            echo "Usage: $0 [--pypto-home PATH] [--pypto-src PATH] [--proxy URL]"
            echo '  --http-proxy URL --https-proxy URL --no-proxy HOSTS --pypi URL'
            echo '  --torch-index URL --torch-version VERSION --uv-version VERSION --uv-mirror URL'
            exit 0 ;;
        --pypto-home|--pypto-src|--proxy|--http-proxy|--https-proxy|--no-proxy|--pypi|--pip-index|--torch-index|--torch-version|--uv-version|--uv-mirror)
            [ $# -ge 2 ] && [ -n "$2" ] || { echo "$1 requires a value" >&2; exit 2; }
            case "$1" in
                --pypto-home) export PYPTO_HOME="$2" ;;
                --pypto-src) export PYPTO_SRC="$2" ;;
                --proxy) export HTTP_PROXY="$2" HTTPS_PROXY="$2" http_proxy="$2" https_proxy="$2" ;;
                --http-proxy) export HTTP_PROXY="$2" http_proxy="$2" ;;
                --https-proxy) export HTTPS_PROXY="$2" https_proxy="$2" ;;
                --no-proxy) export NO_PROXY="$2" no_proxy="$2" ;;
                --pypi|--pip-index) export PIP_INDEX_URL="$2" UV_INDEX_URL="$2" ;;
                --torch-index) torch_index="$2" ;;
                --torch-version) torch_version="$2" ;;
                --uv-version) uv_version="$2" ;;
                --uv-mirror) export UV_PYTHON_INSTALL_MIRROR="$2" ;;
            esac
            shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

toolchain_env="$PYPTO_SRC/toolchain/versions.env"
constraints="$PYPTO_SRC/build-constraints.txt"
if [ ! -f "$toolchain_env" ]; then toolchain_env="$PYPTO_HOME/toolchain/versions.env"; fi
if [ ! -f "$constraints" ]; then constraints="$PYPTO_HOME/toolchain/build-constraints.txt"; fi
[ -r "$toolchain_env" ] && [ -r "$constraints" ] || {
    echo "Set PYPTO_SRC to a checkout containing toolchain/versions.env and build-constraints.txt" >&2
    exit 1
}
export PATH="$PYPTO_HOME/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    case "$(uname -m)" in
        aarch64|x86_64) asset="uv-$(uname -m)-unknown-linux-gnu" ;;
        *) echo "Unsupported host architecture: $(uname -m)" >&2; exit 1 ;;
    esac
    if [ "$uv_version" = latest ]; then
        url="https://github.com/astral-sh/uv/releases/latest/download/$asset.tar.gz"
    else
        url="https://github.com/astral-sh/uv/releases/download/$uv_version/$asset.tar.gz"
    fi
    download_dir="$(mktemp -d)"
    trap 'rm -rf "$download_dir"' EXIT
    curl --fail --location --retry 3 --retry-all-errors "$url" -o "$download_dir/uv.tar.gz"
    tar -xzf "$download_dir/uv.tar.gz" -C "$download_dir"
    install -d "$PYPTO_HOME/bin"
    install -m 0755 "$download_dir/$asset/uv" "$PYPTO_HOME/bin/uv"
fi
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$PYPTO_HOME/python}"
export PIP_USER=0
uv python install 3.10
if [ ! -x "$PYPTO_HOME/venv/bin/python" ]; then
    uv venv --python 3.10 "$PYPTO_HOME/venv"
fi
python="$PYPTO_HOME/venv/bin/python"
uv pip install --python "$python" "torch==$torch_version" --index-url "$torch_index"
uv pip install --python "$python" --constraint "$constraints" \
    'scikit-build-core>=0.10.0' 'nanobind>=2.0.0,<3' 'ninja>=1.11.0' 'cmake>=3.15' \
    'numpy>=2.0' 'pytest>=7.0.0' 'pytest-forked>=1.0' 'pytest-xdist==3.8.0' \
    'pytest-timeout>=2.3' 'pyright==1.1.407' 'ruff==0.14.8' 'clang-tidy==21.1.0' \
    'cloudpickle>=2.2' 'PyYAML>=6.0'
"$here/pypto-install-ptoas.sh" --toolchain-env "$toolchain_env" --force

# Install shared sources rather than maintaining embedded copies of the helpers.
install -d "$PYPTO_HOME/bin" "$PYPTO_HOME/toolchain" "$PYPTO_HOME/cache"
for helper in pypto-build pypto-doctor pypto-update.sh pypto-setup.sh pypto-install-ptoas.sh; do
    if [ ! "$here/$helper" -ef "$PYPTO_HOME/bin/$helper" ]; then
        install -m 0755 "$here/$helper" "$PYPTO_HOME/bin/$helper"
    fi
done
if [ ! "$here/pypto-activate.sh" -ef "$PYPTO_HOME/bin/pypto-activate.sh" ]; then
    install -m 0644 "$here/pypto-activate.sh" "$PYPTO_HOME/bin/pypto-activate.sh"
fi
if [ ! "$toolchain_env" -ef "$PYPTO_HOME/toolchain/versions.env" ]; then
    install -m 0644 "$toolchain_env" "$PYPTO_HOME/toolchain/versions.env"
fi
if [ ! "$constraints" -ef "$PYPTO_HOME/toolchain/build-constraints.txt" ]; then
    install -m 0644 "$constraints" "$PYPTO_HOME/toolchain/build-constraints.txt"
fi
echo "Setup complete. Activate with: source $PYPTO_HOME/bin/pypto-activate.sh"
