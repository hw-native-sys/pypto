# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared PTOAS invocation configuration for compilation and debug replay."""

import os

PTOAS_TIMEOUT_ENV = "PYPTO_PTOAS_TIMEOUT_SECONDS"
PTOAS_BUFFER_ID_SYNC_MARKER = "// pypto.ptoas_buffer_id_sync"
PTOAS_MANUAL_SYNC_MARKER = "// pypto.ptoas_manual_sync"


def sync_flags(*, buffer_id_sync: bool, manual_sync: bool) -> list[str]:
    """Automatic event insertion, automatic buffer IDs, and authored synchronization are exclusive."""
    if manual_sync and buffer_id_sync:
        raise ValueError("ptoas_manual_sync cannot be combined with ptoas_buffer_id_sync")
    if manual_sync:
        return []
    return ["--enable-bufid_sync" if buffer_id_sync else "--enable-insert-sync"]


def get_ptoas_timeout() -> int:
    """Read the per-kernel assembly deadline at invocation time, in seconds."""
    value = os.environ.get(PTOAS_TIMEOUT_ENV, "60")
    try:
        timeout = int(value)
    except ValueError:
        raise ValueError(
            f"{PTOAS_TIMEOUT_ENV} must be a positive integer in seconds, got {value!r}"
        ) from None
    if timeout <= 0:
        raise ValueError(f"{PTOAS_TIMEOUT_ENV} must be a positive integer in seconds, got {value!r}")
    return timeout
