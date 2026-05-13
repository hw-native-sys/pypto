# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Single user-facing knob for simpler runtime log level.

PyPTO's C++ logger and simpler's Python logger are intentionally independent
subsystems. This module exposes one helper that drives only simpler by default,
with an opt-in flag to mirror the band onto PyPTO's coarser enum.

Three entry points:

* :func:`configure_log` — programmatic API (used in user code / tests).
* ``PYPTO_RUNTIME_LOG`` env var — bootstrapped by :func:`_ensure_configured`
  at ``pypto.runtime`` import time. ``PYPTO_RUNTIME_LOG_SYNC=1`` flips the
  default value of ``sync_pypto``.
* :func:`current_level` — read back the effective threshold.

Level parsing delegates to :mod:`simpler_setup.log_config` (the canonical
CLI level table); simpler imports are lazy so ``import pypto.runtime`` still
works in offline-compile environments where simpler is not installed.
"""

import os
from typing import Final

_ENV_LEVEL: Final[str] = "PYPTO_RUNTIME_LOG"
_ENV_SYNC: Final[str] = "PYPTO_RUNTIME_LOG_SYNC"

_configured: bool = False


def configure_log(level: int | str, *, sync_pypto: bool = False) -> None:
    """Set simpler's log threshold (and optionally PyPTO's C++ logger too).

    Args:
        level: Python ``logging`` int (e.g. ``20``) or string (``"debug"``,
            ``"v0".."v9"``, ``"info"``, ``"warn"``, ``"error"``, ``"null"``).
            Case-insensitive. See :data:`simpler_setup.log_config.LOG_LEVEL_CHOICES`.
        sync_pypto: When ``True``, also push the closest PyPTO ``LogLevel`` to
            the C++ side so both subsystems display the same band. Defaults to
            ``False`` because the two loggers are intentionally independent.
    """
    global _configured

    from simpler_setup.log_config import parse_level  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
    from simpler._log import get_logger as _simpler_logger  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    threshold = parse_level(level)
    _simpler_logger().setLevel(threshold)

    if sync_pypto:
        _sync_to_pypto(threshold)

    _configured = True


def _sync_to_pypto(threshold: int) -> None:
    """Map the unified threshold onto PyPTO's coarser LogLevel enum.

    Bands mirror :func:`simpler._log._split_threshold`:
    ``<=14`` DEBUG, ``15..24`` (V0..V9) INFO, ``25..39`` WARN,
    ``40..59`` ERROR, ``>=60`` NUL/NONE.
    """
    from pypto import LogLevel, set_log_level  # noqa: PLC0415

    if threshold <= 14:
        set_log_level(LogLevel.DEBUG)
    elif threshold <= 24:
        set_log_level(LogLevel.INFO)
    elif threshold <= 39:
        set_log_level(LogLevel.WARN)
    elif threshold <= 59:
        set_log_level(LogLevel.ERROR)
    else:
        set_log_level(LogLevel.NONE)


def _ensure_configured() -> None:
    """Idempotent env-var bootstrap, called once at ``pypto.runtime`` import.

    If the user later calls :func:`configure_log` explicitly, that wins; if
    the env var was unset, this is a no-op beyond flipping the cache flag.
    """
    global _configured
    if _configured:
        return
    raw = os.environ.get(_ENV_LEVEL)
    if raw is None:
        # Mark configured so we do not re-check the env on every call.
        _configured = True
        return
    sync = os.environ.get(_ENV_SYNC) == "1"
    configure_log(raw, sync_pypto=sync)


def current_level() -> int:
    """Return simpler's effective Python ``logging`` threshold."""
    from simpler._log import get_logger as _simpler_logger  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
    return _simpler_logger().getEffectiveLevel()
