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
        ring_slots: Optional override for the cross-core consumer-side ring
            depth (a.k.a. PTOAS ``local_slot_num``). Sizes the UB
            ``reserve_buffer`` as ``ring_slots * slot_size`` and lets a
            UB-constrained scope trade pipelining depth for footprint.
            Must be in ``[1, 8]`` for unidirectional pipes and ``[1, 4]``
            for bidirectional. ``None`` keeps the platform default
            (8 single-direction / 4 bidirectional).
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
        ring_slots: Optional cross-core consumer ring depth override
            (lowers to PTOAS ``local_slot_num``). Range ``[1, 8]`` for
            unidirectional pipes, ``[1, 4]`` for bidirectional. ``None``
            uses the platform default (8 / 4).

    Returns:
        ``Split`` instance for use in ``pl.at(..., optimizations=[...])``.

    """
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
