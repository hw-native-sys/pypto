# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Environment variable caching utility.

Provides :func:`ensure` to validate and cache required environment variables
and :func:`get` to retrieve previously validated values.
"""

import os

_cache: dict[str, str | None] = {}


def get(name: str) -> str | None:
    """Return the cached value for *name*. ``None`` if not yet ensured or env var was absent."""
    return _cache.get(name)


def ensure(name: str) -> str:
    """Fetch env var, cache it, raise ``EnvironmentError`` if unset/empty."""
    cached = _cache.get(name)
    if cached is not None:
        return cached
    value = os.environ.get(name)
    if not value:
        raise OSError(f"Environment variable '{name}' is not set.")
    _cache[name] = value
    return value
