# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Environment variable caching utility.

Provides :func:`ensure` to validate and cache required environment variables,
:func:`get` to retrieve previously validated values, and
:func:`get_simpler_root` to locate the simpler ``runtime/`` submodule.
"""

import os
from pathlib import Path

_cache: dict[str, str | None] = {}

_simpler_root: Path | None = None


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


def get_simpler_root() -> Path:
    """Return the ``simpler`` submodule root directory.

    Resolution order:

    1. Editable / source-tree install: ``<pypto-repo>/simpler/`` relative to
       this file (``python/pypto/runtime/env_manager.py`` → 3 parents up).
    2. Non-editable (pip) install: walk up from ``cwd()`` looking for a git
       repository that contains a ``simpler/`` submodule.

    The result is cached after the first successful lookup.
    """
    global _simpler_root  # noqa: PLW0603
    if _simpler_root is not None:
        return _simpler_root

    candidate = Path(__file__).resolve().parents[3] / "runtime"
    if candidate.is_dir():
        _simpler_root = candidate
        return _simpler_root

    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / "runtime"
        if (parent / ".git").exists() and candidate.is_dir():
            _simpler_root = candidate
            return _simpler_root

    raise OSError(
        "Cannot find runtime/ submodule directory.\n"
        "Ensure you are running from within the pypto repository with "
        "submodules initialized:\n"
        "  git submodule update --init"
    )
