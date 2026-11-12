#!/usr/bin/env bash
# Prepare a machine to run the PyPTO demos, then print the environment to use.
#
# Usage:  source scripts/setup-demo-env.sh
#
# Idempotent: re-running skips the steps that are already done. Everything it
# creates lives outside the repository (~/.local/shims, .venv) except the build
# directories, which are normal CMake output.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
PTOAS_VERSION=${PTOAS_VERSION:-0.72}
DEMO_ENV=${DEMO_ENV:-$REPO/.demo-env}
SHIMS=${SHIMS:-$DEMO_ENV/shims}
PTOAS_ENV=${PTOAS_ENV:-$DEMO_ENV/ptoas}

say() { printf '\n== %s\n' "$*"; }

# The OpenHarmony toolchain that some IDEs put on PATH ships a `diff` that always
# exits 0. autoconf takes that to mean "config.h is unchanged" and writes nothing,
# so libbacktrace then fails with 'config.h' file not found. Strip that directory
# for every command this script runs, and tell the caller to keep it stripped.
sanitized_path() {
  printf '%s' "$PATH" | tr ':' '\n' | grep -v 'DevEco-Studio' | paste -sd: -
}

say "1/5 C++ compiler shim"
# The compiler is invoked as g++-15. Map it to the system clang through a wrapper
# script: a symlink does not work, because clang looks for its own driver under
# the name it was invoked as, and /tmp is not executable on this platform.
mkdir -p "$SHIMS"
printf '#!/bin/sh\nexec /usr/bin/clang++ "$@"\n' > "$SHIMS/g++-15"
chmod +x "$SHIMS/g++-15"
"$SHIMS/g++-15" --version | head -1

say "2/5 Python environment and pypto"
[ -d "$REPO/.venv" ] || "$PYTHON" -m venv --system-site-packages "$REPO/.venv"
VENV_PY="$REPO/.venv/bin/python"
"$VENV_PY" -m pip install -q --upgrade pip scikit-build-core nanobind ninja cloudpickle
(cd "$REPO" && PATH="$(sanitized_path)" "$VENV_PY" -m pip install -q -e . --no-build-isolation)
"$VENV_PY" -c 'import pypto; print("pypto", pypto.__version__)'

say "3/5 runtime bindings (_task_interface)"
# The vendored `simpler` runtime needs its nanobind extension, which is a separate
# CMake project under runtime/ and must be built against this venv's Python.
CM=$(command -v cmake)
(cd "$REPO/runtime" && PATH="$(sanitized_path)" "$CM" -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DPython_EXECUTABLE="$VENV_PY" >/dev/null &&
  PATH="$(sanitized_path)" "$CM" --build build --parallel >/dev/null)
ls "$REPO"/runtime/python/_task_interface*.so | head -1

say "4/5 PTOAS v$PTOAS_VERSION"
# PyPTO requires ptoas >= 0.72 and refuses older ones. The release ships wheels for
# Python 3.10-3.12 only, so install the CLI in its own interpreter and point
# PTOAS_ROOT at it; the demos only execute the binary, so the caller's Python
# version does not matter.
if [ ! -x "$PTOAS_ENV/bin/ptoas" ]; then
  WHEEL="ptoas-${PTOAS_VERSION}-cp312-cp312-macosx_26_0_arm64.whl"
  URL="https://github.com/hw-native-sys/PTOAS/releases/download/v${PTOAS_VERSION}/${WHEEL}"
  TMP=$(mktemp -d)
  curl -fsSL -o "$TMP/$WHEEL" "$URL"
  python3.12 -m venv "$PTOAS_ENV"
  "$PTOAS_ENV/bin/pip" install -q "$TMP/$WHEEL"
  rm -rf "$TMP"
fi
"$PTOAS_ENV/bin/ptoas" --version

say "5/5 done"
# Everything the demos need lives under one directory, so sourcing this script is
# the only setup step and nothing depends on the caller's shell environment.
cat > "$DEMO_ENV/activate.sh" <<ENV
export PATH="$SHIMS:$PTOAS_ENV/bin:$(printf '%s' "\$PATH" | tr ':' '\\n' | grep -v DevEco-Studio | paste -sd: -)"
export PTOAS_ROOT="$PTOAS_ENV"
export PYTHONPATH="$REPO/runtime/python:$REPO/runtime"
ENV
if [ "${BASH_SOURCE[0]}" != "\$0" ]; then
  . "$DEMO_ENV/activate.sh"
  echo "environment exported; run: $VENV_PY examples/beginner/01_hello_world.py"
else
  echo "wrote $DEMO_ENV/activate.sh; source it to export the environment"
fi
