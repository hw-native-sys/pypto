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
    ``SplitMode``. ``optimizations=[pl.split(mode)]`` lowers the scope to
    ``ScopeKind::InCore`` with the split metadata attached.

    Args:
        mode: Split mode (``SplitMode.NONE``, ``SplitMode.UP_DOWN``, or
            ``SplitMode.LEFT_RIGHT``).
        slot_num: Optional cross-core ring-buffer depth for the automatic
            cube→vector pipe inserted by ``ExpandMixedKernel``. ``None`` keeps
            the PTOAS-derived default (8 unidirectional, 4 per direction
            bidirectional). When set, both the reserved buffer
            (``slot_size * slot_num``) and the emitted ``initialize_pipe``
            ``slot_num`` attribute use this value. Valid with any ``mode``,
            including ``SplitMode.NONE``: a NONE mixed kernel still drives a
            cube→vector pipe (on Ascend910B via dual-AIV dispatch), so
            ``slot_num`` sizes that ring regardless of split mode. It is ignored
            only when the outlined scope ends up with no cross-core ops.
    """

    mode: SplitMode
    slot_num: int | None = None


def split(mode: SplitMode, *, slot_num: int | None = None) -> Split:
    """Create a ``Split`` optimization entry.

    Args:
        mode: Split mode. May be ``SplitMode.NONE``,
            ``SplitMode.UP_DOWN``, or ``SplitMode.LEFT_RIGHT``.
        slot_num: Optional cross-core ring-buffer depth for the automatic
            cube→vector pipe. Must be positive when set. Omit to keep the
            PTOAS-derived default (8 unidirectional, 4 bidirectional).

    Returns:
        ``Split`` instance for use in ``pl.at(..., optimizations=[...])``.

    Raises:
        ValueError: If ``slot_num`` is set but not positive.
    """
    if slot_num is not None:
        # bool is a subclass of int — reject it so True/False can't pose as a depth.
        if not isinstance(slot_num, int) or isinstance(slot_num, bool):
            raise ValueError(f"pl.split slot_num must be a positive integer, got {slot_num!r}")
        if slot_num <= 0:
            raise ValueError(f"pl.split slot_num must be a positive integer, got {slot_num}")
    return Split(mode=mode, slot_num=slot_num)


__all__ = [
    "Optimization",
    "Split",
    "split",
]
