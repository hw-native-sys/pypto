# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Optimization config entries for ``pl.at(..., optimizations=[...])``.

Each entry is an orthogonal optimization hint applied to the enclosing scope.
The entries can be combined freely in the ``optimizations=`` list.

Available entries:
    - ``pl.split(mode)`` — Cross-core data-transfer split hint, consumed by
      the ``ExpandMixedKernel`` pass. Lowers the scope to ``InCore`` with
      ``split_=mode``.
    - ``pl.auto_chunk`` — Request compiler-driven outlining of chunked
      parallel loops. Lowers the scope to ``AutoInCore`` so that the
      ``InterchangeChunkLoops`` pass can interchange and outline chunked
      loops within it.

These two entries are independent and may be combined::

    with pl.at(level=pl.Level.CORE_GROUP,
               optimizations=[pl.auto_chunk, pl.split(pl.SplitMode.UP_DOWN)]):
        ...
"""

from __future__ import annotations

from dataclasses import dataclass

from pypto.pypto_core.ir import SplitMode


class Optimization:
    """Base class for ``pl.at(..., optimizations=[...])`` entries."""


@dataclass(frozen=True)
class Split(Optimization):
    """Cross-core data-transfer split hint.

    Sets ``ScopeStmt::split_`` on the enclosing ``pl.at`` scope; that metadata
    is consumed by the ``ExpandMixedKernel`` pass via the outlined function's
    ``SplitMode``. The split hint is independent of the resulting scope kind:

    - ``optimizations=[pl.split(mode)]`` → ``ScopeKind::InCore`` (split metadata).
    - ``optimizations=[pl.auto_chunk, pl.split(mode)]`` → ``ScopeKind::AutoInCore``
      (split metadata still attached).

    Args:
        mode: Split mode (``SplitMode.NONE``, ``SplitMode.UP_DOWN``, or
            ``SplitMode.LEFT_RIGHT``).
        ring_slots: Optional override for the cross-core pipe ring depth (number
            of slots in the C2V / V2C ring). When unset, ``ExpandMixedKernel``
            falls back to its built-in heuristic (8 slots for unidirectional,
            4 slots for bidirectional). Only meaningful with a non-``NONE``
            ``mode``; must be a positive ``int``.
    """

    mode: SplitMode
    ring_slots: int | None = None


@dataclass(frozen=True)
class AutoChunk(Optimization):
    """Request compiler-driven outlining of chunked parallel loops.

    Lowers the enclosing ``pl.at`` scope to ``ScopeKind::AutoInCore`` so the
    ``InterchangeChunkLoops`` pass can interchange chunked parallel loops
    and outline the inner sequential portion into ``InCore`` scopes.

    Only valid with ``level=pl.Level.CORE_GROUP``.
    """


def split(mode: SplitMode, *, ring_slots: int | None = None) -> Split:
    """Create a ``Split`` optimization entry.

    Args:
        mode: Split mode. May be ``SplitMode.NONE``,
            ``SplitMode.UP_DOWN``, or ``SplitMode.LEFT_RIGHT``.
        ring_slots: Optional override for the cross-core pipe ring depth.
            When unset, ``ExpandMixedKernel`` uses its built-in heuristic
            (8 for unidirectional, 4 for bidirectional). Must be a positive
            ``int`` and is only meaningful when ``mode != SplitMode.NONE``.

    Returns:
        ``Split`` instance for use in ``pl.at(..., optimizations=[...])``.

    Raises:
        ValueError: If ``ring_slots`` is not a positive ``int``, or if
            ``ring_slots`` is given alongside ``mode == SplitMode.NONE``.
    """
    if ring_slots is not None:
        if isinstance(ring_slots, bool) or not isinstance(ring_slots, int):
            raise ValueError(f"ring_slots must be a positive int, got {ring_slots!r}")
        if ring_slots <= 0:
            raise ValueError(f"ring_slots must be a positive int, got {ring_slots}")
        if mode == SplitMode.NONE:
            raise ValueError(
                "ring_slots is only meaningful with a non-NONE split mode; "
                "use pl.split(pl.SplitMode.UP_DOWN, ring_slots=N) or "
                "pl.split(pl.SplitMode.LEFT_RIGHT, ring_slots=N)"
            )
    return Split(mode=mode, ring_slots=ring_slots)


auto_chunk: AutoChunk = AutoChunk()
"""Sentinel for the ``AutoChunk`` optimization.

Use as ``pl.auto_chunk`` in ``pl.at(..., optimizations=[pl.auto_chunk, ...])``.
"""


__all__ = [
    "Optimization",
    "Split",
    "AutoChunk",
    "split",
    "auto_chunk",
]
