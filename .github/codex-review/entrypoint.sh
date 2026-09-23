# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

install -m 0600 /usr/local/share/codex-review/config.toml "${CODEX_HOME}/config.toml"

# Read authentication from stdin into tmpfs; it is never bind-mounted. Wait
# until the access watcher is active before starting Codex, then remove the
# file as soon as Codex's initial authentication read closes.
cat >"${CODEX_HOME}/auth.json"
chmod 0600 "${CODEX_HOME}/auth.json"
coproc AUTH_WATCHER {
    inotifywait --event close_nowrite "${CODEX_HOME}/auth.json" 2>&1
}
exec 3<&"${AUTH_WATCHER[0]}"
IFS= read -r _setup_line <&3
IFS= read -r _ready_line <&3
(
    IFS= read -r _auth_event <&3
    sleep 1
    rm -f "${CODEX_HOME}/auth.json"
) &
exec 3<&-

exec codex "$@" </dev/null
