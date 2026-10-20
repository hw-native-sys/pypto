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

pypto_venv_is_compatible() {
    [ -f "$1/pyvenv.cfg" ] && [ -r "$1/pyvenv.cfg" ] &&
        [ -f "$1/bin/activate" ] && [ -r "$1/bin/activate" ] &&
        pypto_python_is_310 "$1/bin/python"
}

pypto_ensure_venv() {
    local prefix="$1" requested_python="$2"
    if pypto_venv_is_compatible "$prefix"; then return 0; fi
    # Only callers' dedicated installation prefixes are passed here. Removing
    # the final symlink replaces the prefix without deleting its external target.
    rm -rf -- "$prefix" || return 1
    uv venv --python "$requested_python" "$prefix" || return 1
    pypto_venv_is_compatible "$prefix" || {
        echo "Requires a complete CPython 3.10 virtual environment at $prefix" >&2
        return 1
    }
}
