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
