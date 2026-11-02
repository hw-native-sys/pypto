# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Shared checks for setup's dedicated PyPTO and PTOAS environments. A working
# interpreter alone does not establish that a virtual environment is complete.
pypto_python_is_310() {
    local version
    version="$("$1" -I -S -c 'import sys; print(".".join(map(str, sys.version_info[:2])) if sys.implementation.name == "cpython" else "unsupported")' 2>/dev/null)" || return 1
    [ "$version" = 3.10 ]
}

# The main environment builds native extensions; PTOAS only needs an interpreter.
pypto_python_has_development_files() {
    "$1" -I -c '
import os
import sys
import sysconfig
from pathlib import Path

def readable(path):
    return path.is_file() and os.access(path, os.R_OK)

headers = [Path(sysconfig.get_path(kind)) / name for kind, name in
           [("include", "Python.h"), ("platinclude", "pyconfig.h")]]
libraries = [Path(directory) / name
             for key in ("LIBDIR", "LIBPL")
             if (directory := sysconfig.get_config_var(key))
             for variable in ("LDLIBRARY", "LIBRARY")
             if (name := sysconfig.get_config_var(variable))]
sys.exit(not (all(map(readable, headers)) and any(map(readable, libraries))))
' 2>/dev/null
}

pypto_venv_is_compatible() {
    [ -f "$1/pyvenv.cfg" ] && [ -r "$1/pyvenv.cfg" ] &&
        [ -f "$1/bin/activate" ] && [ -r "$1/bin/activate" ] &&
        pypto_python_is_310 "$1/bin/python" || return 1
    if [ "${2:-runtime}" = development ]; then
        pypto_python_has_development_files "$1/bin/python" || return 1
    fi
}

pypto_ensure_venv() {
    local prefix="$1" requested_python="$2" purpose="${3:-runtime}"
    local options=()
    if [ "$purpose" = development ]; then options+=(--managed-python); fi
    if pypto_venv_is_compatible "$prefix" "$purpose"; then return 0; fi
    # Only callers' dedicated installation prefixes are passed here. Removing
    # the final symlink replaces the prefix without deleting its external target.
    rm -rf -- "$prefix" || return 1
    uv venv "${options[@]}" --python "$requested_python" "$prefix" || return 1
    pypto_venv_is_compatible "$prefix" "$purpose" || {
        echo "Requires a complete CPython 3.10 virtual environment ($purpose) at $prefix" >&2
        return 1
    }
}
