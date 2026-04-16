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

Available entries:
    - ``pl.split(mode)`` — Cross-core data-transfer split hint, consumed by
      the ``ExpandMixedKernel`` pass. Only valid at ``Level::CORE_GROUP``;
      sets ``split`` on the enclosing ``HierarchyScopeStmt``.
"""

from __future__ import annotations

from dataclasses import dataclass

from pypto.pypto_core.ir import SplitMode


class Optimization:
    """Base class for ``pl.at(..., optimizations=[...])`` entries."""


@dataclass(frozen=True)
class Split(Optimization):
    """Cross-core data-transfer split hint.

    Sets ``HierarchyScopeStmt::split_`` on the enclosing ``pl.at`` scope.
    Only valid at ``Level::CORE_GROUP``; consumed by the ``ExpandMixedKernel``
    pass via the outlined function's ``SplitMode``.

    Args:
        mode: Split mode (``SplitMode.UP_DOWN`` or ``SplitMode.LEFT_RIGHT``).
            ``SplitMode.NONE`` is rejected — omit the entry instead to
            indicate "no split".
    """

    mode: SplitMode


def split(mode: SplitMode) -> Split:
    """Create a ``Split`` optimization entry.

    Args:
        mode: Split mode. Must be ``SplitMode.UP_DOWN`` or
            ``SplitMode.LEFT_RIGHT``. ``SplitMode.NONE`` is rejected — omit
            the entry instead to indicate "no split".

    Returns:
        ``Split`` instance for use in ``pl.at(..., optimizations=[...])``.

    Raises:
        ValueError: if ``mode`` is ``SplitMode.NONE``.
    """
    if mode == SplitMode.NONE:
        raise ValueError(
            "pl.split(pl.SplitMode.NONE) is not supported; omit the entry instead to indicate 'no split'."
        )
    return Split(mode=mode)


__all__ = [
    "Optimization",
    "Split",
    "split",
]
