# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Environment for the PyPTO on-device development image. Sourced as the
# interactive shell's rcfile (ENTRYPOINT) and as BASH_ENV for non-interactive
# shells, so `docker exec ... bash -c '...'` gets the same environment as a
# login shell. Safe to source more than once.
#
# shellcheck shell=bash

# Imports and diagnostic Git probes must not leave root-owned state in sources.
export PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0

export PYPTO_HOME="${PYPTO_HOME:-/opt/pypto}"

# Where the bind-mounted checkouts live. Overridable so a host that mounts
# somewhere else needs no rebuild.
export PYPTO_SRC="${PYPTO_SRC:-/workspace/pypto}"
export PYPTO_LIB_SRC="${PYPTO_LIB_SRC:-/workspace/pypto-lib}"

# The launcher selects a platform explicitly. Direct image runs derive it from
# the CANN tag retained by the Dockerfile; custom environments can set it too.
export PYPTO_PLATFORM="${PYPTO_PLATFORM:-}"
if [ -z "$PYPTO_PLATFORM" ]; then
    case "${PYPTO_CANN_TAG:-}" in
        *-a2-*|*-a3-*) export PYPTO_PLATFORM=a2a3 ;;
        *-950-*) export PYPTO_PLATFORM=a5 ;;
    esac
fi

_pypto_model_command() {
    if [ -n "${PYPTO_PLATFORM:-}" ]; then
        printf '    python $PYPTO_LIB_SRC/models/qwen3_14b/decode_fwd.py -p %s -d ${PYPTO_DEVICE:-0}\n' "$PYPTO_PLATFORM"
    else
        echo '    Set PYPTO_PLATFORM=a2a3 or a5 for model commands.'
    fi
}

# --- CANN -----------------------------------------------------------------
# First, because set_env.sh prepends its own dirs to PATH / LD_LIBRARY_PATH /
# PYTHONPATH and everything below must land in front of it.
_pypto_cann_env="${ASCEND_TOOLKIT_ROOT:-/usr/local/Ascend/ascend-toolkit}/set_env.sh"
if [ -f "$_pypto_cann_env" ]; then
    # shellcheck disable=SC1090
    source "$_pypto_cann_env"
fi
unset _pypto_cann_env

# --- PyPTO's own Python (uv-managed CPython 3.10) --------------------------
# shellcheck disable=SC1091
source "$PYPTO_HOME/venv/bin/activate"

# --- ptoas ----------------------------------------------------------------
export PTOAS_ROOT="${PTOAS_ROOT:-$PYPTO_HOME/ptoas}"
# Keep Python and package installation in PyPTO's venv. PTOAS also has a Python
# executable, so put its bin directory after that venv but before CANN's tools.
export PATH="$PYPTO_HOME/venv/bin:$PTOAS_ROOT/bin:$PATH"

# PYTHONPATH is searched before either interpreter's site-packages, so selecting
# the pinned executable alone still lets it import CANN's bundled ptoas. Keep
# PyPTO's packages first (including numpy), then the pinned PTOAS packages, then
# CANN. Rebuild this order on every source because set_env.sh prepends CANN again.
_pypto_site="$("$PYPTO_HOME/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null)"
_ptoas_site=""
if [ -x "$PTOAS_ROOT/bin/python" ]; then
    _ptoas_site="$("$PTOAS_ROOT/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null)"
fi
if [ -n "$_ptoas_site" ]; then
    export PYTHONPATH="$_ptoas_site${PYTHONPATH:+:$PYTHONPATH}"
fi
if [ -n "$_pypto_site" ]; then
    export PYTHONPATH="$_pypto_site${PYTHONPATH:+:$PYTHONPATH}"
fi
unset _pypto_site _ptoas_site

# --- Compilers ------------------------------------------------------------
# The base image's g++ (11) is the right default: simpler's runtime is C++17,
# in-core kernels go through CANN's CCEC, and device code through CANN's
# aarch64 cross-g++. g++-15 matters only for the CPU simulator's kernels, so
# it is picked up when the `sim` image layer provided it and ignored otherwise.
if command -v g++-15 >/dev/null 2>&1; then
    export CC="${CC:-gcc-15}" CXX="${CXX:-g++-15}"
else
    export CC="${CC:-gcc}" CXX="${CXX:-g++}"
fi
if command -v ccache >/dev/null 2>&1; then
    export CMAKE_C_COMPILER_LAUNCHER="${CMAKE_C_COMPILER_LAUNCHER:-ccache}"
    export CMAKE_CXX_COMPILER_LAUNCHER="${CMAKE_CXX_COMPILER_LAUNCHER:-ccache}"
fi

# --- The built PyPTO ------------------------------------------------------
# A native (non-scikit-build) CMake build puts pypto_core.*.so straight into
# $PYPTO_SRC/python/pypto, which is the tree `import pypto` must find. pypto-lib
# is a plain source checkout and is imported the same way.
# Reinsert checkout paths on every activation: the venv was just prepended above
# and may contain an installed PyPTO that would otherwise shadow the mounted tree.
_pypto_pp=""
_pypto_ifs="$IFS"; IFS=:
for _pypto_path in ${PYTHONPATH:-}; do
    case "$_pypto_path" in
        ""|"$PYPTO_SRC/python"|"$PYPTO_LIB_SRC") continue ;;
    esac
    _pypto_pp="${_pypto_pp:+$_pypto_pp:}$_pypto_path"
done
IFS="$_pypto_ifs"
export PYTHONPATH="$_pypto_pp"
for _pypto_path in "$PYPTO_SRC/python" "$PYPTO_LIB_SRC"; do
    [ -d "$_pypto_path" ] || continue
    export PYTHONPATH="$_pypto_path${PYTHONPATH:+:$PYTHONPATH}"
done
unset _pypto_path _pypto_pp _pypto_ifs

export PATH="$PYPTO_HOME/bin:$PATH"
# CANN and virtualenv activation both prepend paths. Keep the first occurrence,
# including the empty component (current directory), without shell glob expansion.
_pypto_deduplicate_path() {
    local name="$1" value rest="${!1}" result="" separator="" done=false
    local -A seen=()
    while ! $done; do
        if [[ "$rest" = *:* ]]; then value="${rest%%:*}"; rest="${rest#*:}"
        else value="$rest"; done=true; fi
        if [ -z "${seen["x$value"]:-}" ]; then
            result+="$separator$value"; separator=:
            seen["x$value"]=1
        fi
    done
    printf -v "$name" '%s' "$result"
    export "$name"
}
_pypto_deduplicate_path PATH
_pypto_deduplicate_path PYTHONPATH
if [ -n "${LD_LIBRARY_PATH:-}" ]; then _pypto_deduplicate_path LD_LIBRARY_PATH; fi
unset -f _pypto_deduplicate_path

if [ -n "${PS1:-}" ] && [ -z "${PYPTO_MOTD_SHOWN:-}" ]; then
    export PYPTO_MOTD_SHOWN=1
    echo "PyPTO dev environment — pypto-doctor (check the current checkout) / pypto-build (build or refresh it)"
    echo "After the checks pass, run a model:"
    _pypto_model_command
    # This describes image provenance. A bind mount can hide the image's built
    # checkout, so the manifest cannot establish readiness of the current tree.
    if [ -f "$PYPTO_HOME/dist-manifest" ]; then
        echo "Image build revisions: $PYPTO_HOME/dist-manifest; mounted sources may differ."
    fi
fi
