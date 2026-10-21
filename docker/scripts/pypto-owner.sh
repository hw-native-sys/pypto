#!/bin/bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Shared ownership boundary for commands that write into bind-mounted checkouts.
# Source this file; callers keep their privileges for device diagnostics.

pypto_as_checkout_owner() {
    local checkout="$1"; shift
    local uid gid owner_home
    uid="$(stat -c %u "$checkout")" || return 1
    if [ "$(id -u)" != 0 ] || [ "$uid" = 0 ]; then
        "$@"
        return
    fi
    command -v setpriv >/dev/null || {
        echo 'setpriv (util-linux) is required to preserve checkout ownership' >&2
        return 1
    }
    gid="$(stat -c %g "$checkout")" || return 1
    owner_home="${PYPTO_HOME:-/opt/pypto}/users/$uid"
    install -d -m 0755 "${PYPTO_HOME:-/opt/pypto}/users" || return 1
    install -d -m 0700 -o "$uid" -g "$gid" "$owner_home" || return 1
    # Numeric IDs work even when the host account has no /etc/passwd entry.
    # Do not inherit root's HOME, Git config, or package cache into that account.
    setpriv --reuid="$uid" --regid="$gid" --clear-groups \
        env HOME="$owner_home" PYTHONDONTWRITEBYTECODE=1 "$@"
}
