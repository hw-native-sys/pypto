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
      ``split_=mode``::

          with pl.at(level=pl.Level.CORE_GROUP,
                     optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
              ...

    - ``pl.cross_core_slot(slot_num=N)`` — Slot count (ring depth) for the
      automatic cross-core pipe. Orthogonal to splitting; combine freely::

          with pl.at(level=pl.Level.CORE_GROUP,
                     optimizations=[pl.split(pl.SplitMode.UP_DOWN),
                                    pl.cross_core_slot(slot_num=4)]):
              ...
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from pypto.pypto_core.ir import SplitMode

# Shared by the ``pl.split()`` factory and the AST parser's ``pl.split(...)``
# matcher so both deprecation paths say exactly the same thing.
SPLIT_SLOT_NUM_DEPRECATION = (
    "pl.split(slot_num=...) is deprecated: the cross-core slot count is orthogonal to the "
    "split mode, and spelling it here forces a split mode you may not want (the "
    "pl.split(pl.SplitMode.NONE, slot_num=N) idiom). Use "
    "optimizations=[pl.cross_core_slot(slot_num=N)] instead, optionally alongside "
    "pl.split(MODE)."
)


def _validate_slot_num(slot_num: int, api: str) -> None:
    """Validate a cross-core slot count.

    Args:
        slot_num: Candidate slot count.
        api: API name used in the error message (e.g. ``"pl.cross_core_slot"``).

    Raises:
        ValueError: If ``slot_num`` is not a positive integer.
    """
    # bool is a subclass of int — reject it so True/False can't pose as a count.
    if not isinstance(slot_num, int) or isinstance(slot_num, bool):
        raise ValueError(f"{api} slot_num must be a positive integer, got {slot_num!r}")
    if slot_num <= 0:
        raise ValueError(f"{api} slot_num must be a positive integer, got {slot_num}")


class Optimization:
    """Base class for ``pl.at(..., optimizations=[...])`` entries."""


@dataclass(frozen=True)
class Split(Optimization):
    """Cross-core data-transfer split hint.

    Sets ``ScopeStmt::split_`` on the enclosing ``pl.at`` scope; that metadata
    is consumed by the ``ExpandMixedKernel`` pass via the outlined function's
    ``SplitMode``. ``optimizations=[pl.split(mode)]`` lowers the scope to
    ``ScopeKind::InCore`` with the split metadata attached.

    Args:
        mode: Split mode (``SplitMode.NONE``, ``SplitMode.UP_DOWN``, or
            ``SplitMode.LEFT_RIGHT``).
        slot_num: **Deprecated** — use ``pl.cross_core_slot(slot_num=N)``, which
            carries the same value without naming a split mode. Kept as an alias
            so existing kernels keep working; see :class:`CrossCoreSlot`.
    """

    mode: SplitMode
    slot_num: int | None = None


def split(mode: SplitMode, *, slot_num: int | None = None) -> Split:
    """Create a ``Split`` optimization entry.

    Args:
        mode: Split mode. May be ``SplitMode.NONE``,
            ``SplitMode.UP_DOWN``, or ``SplitMode.LEFT_RIGHT``.
        slot_num: **Deprecated** — use ``pl.cross_core_slot(slot_num=N)``.
            Must be positive when set. Emits a ``DeprecationWarning``.

    Returns:
        ``Split`` instance for use in ``pl.at(..., optimizations=[...])``.

    Raises:
        ValueError: If ``slot_num`` is set but not positive.
    """
    if slot_num is not None:
        _validate_slot_num(slot_num, "pl.split")
        warnings.warn(SPLIT_SLOT_NUM_DEPRECATION, DeprecationWarning, stacklevel=2)
    return Split(mode=mode, slot_num=slot_num)


@dataclass(frozen=True)
class CrossCoreSlot(Optimization):
    """Slot count (ring depth) for the automatic cross-core pipe.

    Orthogonal to splitting — it sizes a data channel, it does not partition
    work. Sets the ``slot_num`` scope attr, which ``OutlineIncoreScopes``
    propagates to the outlined function and ``ExpandMixedKernel`` reads to size
    **both** the reserved buffer (``slot_size * slot_num``) and the emitted
    ``initialize_pipe`` ``slot_num`` attribute, in whichever directions the
    scope actually uses (cube→vector, vector→cube, or both).

    Omitting the entry keeps the default depth of 2 per live direction — enough
    to double-buffer the handoff while leaving on-chip room for the tiles. The
    value is ignored when the outlined scope ends up with no cross-core ops.

    Args:
        slot_num: Ring depth. Must be a positive integer.
    """

    slot_num: int


def cross_core_slot(*, slot_num: int) -> CrossCoreSlot:
    """Create a ``CrossCoreSlot`` optimization entry.

    Args:
        slot_num: Cross-core pipe slot count (ring depth). Must be positive.

    Returns:
        ``CrossCoreSlot`` instance for use in ``pl.at(..., optimizations=[...])``.

    Raises:
        ValueError: If ``slot_num`` is not a positive integer.

    Examples:
        >>> # Deepen the auto-inserted ring so the producing core can run further ahead
        >>> with pl.at(level=pl.Level.CORE_GROUP,
        ...            optimizations=[pl.cross_core_slot(slot_num=4)]):
        ...     ...
    """
    _validate_slot_num(slot_num, "pl.cross_core_slot")
    return CrossCoreSlot(slot_num=slot_num)


__all__ = [
    "CrossCoreSlot",
    "Optimization",
    "Split",
    "cross_core_slot",
    "split",
]
