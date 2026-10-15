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

export PYPTO_HOME="${PYPTO_HOME:-/opt/pypto}"

# Where the bind-mounted checkouts live. Overridable so a host that mounts
# somewhere else needs no rebuild.
export PYPTO_SRC="${PYPTO_SRC:-/workspace/pypto}"
export PYPTO_LIB_SRC="${PYPTO_LIB_SRC:-/workspace/pypto-lib}"

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

# Put the venv's site-packages at the FRONT of PYTHONPATH. CANN's set_env.sh
# puts its own there, and it carries Python packages — built for the base
# image's 3.11/3.12 — that would otherwise shadow what is installed here.
#
# Prepended unconditionally, with no "already present" guard: this file is
# sourced more than once per session and each source re-runs set_env.sh, which
# re-prepends CANN's dir. A guard that skipped on finding the venv dir anywhere
# would leave that fresh CANN copy in front of it. The duplicates left behind
# are harmless and bounded by the number of sources.
_pypto_site="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null)"
if [ -n "$_pypto_site" ]; then
    export PYTHONPATH="$_pypto_site${PYTHONPATH:+:$PYTHONPATH}"
fi
unset _pypto_site

# --- ptoas ----------------------------------------------------------------
# Its own venv: ptoas is invoked as an executable, so only bin/ is exposed and
# its dependency closure never mixes with PyPTO's.
export PTOAS_ROOT="${PTOAS_ROOT:-$PYPTO_HOME/ptoas}"
# CANN prepends its online-build assembler on every activation. Prepend ours
# again even when it already appears later in PATH.
export PATH="$PTOAS_ROOT/bin:$PATH"

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
for _pypto_path in "$PYPTO_SRC/python" "$PYPTO_LIB_SRC"; do
    [ -d "$_pypto_path" ] || continue
    case ":$PYTHONPATH:" in
        *":$_pypto_path:"*) ;;
        *) export PYTHONPATH="$_pypto_path${PYTHONPATH:+:$PYTHONPATH}" ;;
    esac
done
unset _pypto_path

export PATH="$PYPTO_HOME/bin:$PATH"

if [ -n "${PS1:-}" ] && [ -z "${PYPTO_MOTD_SHOWN:-}" ]; then
    export PYPTO_MOTD_SHOWN=1
    # The `dist` image arrives already built, so telling its user to build first
    # would send them off to repeat work that is done. Only that image writes
    # the manifest, so its absence means this is the bring-your-own-source one.
    if [ -f "$PYPTO_HOME/dist-manifest" ]; then
        echo "PyPTO — already built. Run a model:"
        echo "    python \$PYPTO_LIB_SRC/models/qwen3_14b/decode_fwd.py -p a2a3 -d \${PYPTO_DEVICE:-0}"
        echo "Update source and PTOAS with pypto-update.sh; keep edits with --no-pull."
        echo "pypto-doctor checks the environment; $PYPTO_HOME/dist-manifest records what shipped."
    else
        echo "PyPTO dev image — pypto-doctor (check the environment) / pypto-build (build pypto + simpler)"
    fi
fi
