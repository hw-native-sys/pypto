# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fixture for a dep whose layout annotation is *postponed*.

``from __future__ import annotations`` (PEP 563) keeps every annotation as
source text, so ``LAYOUT`` is resolved from this module's globals when
``@pl.jit`` reads the signature — not at ``def`` time. Rebinding
``<module>.LAYOUT`` therefore changes the layout the dep declares while leaving
the source text, and so the source hash, identical.

Loading this file twice under two module names gives two distinct ``helper``
functions that agree on ``__name__`` and can carry different layouts — the
shape that made the name-keyed ``dep_layouts`` cache component collapse.
"""

from __future__ import annotations  # noqa: I001 — the point of the fixture

import pypto.language as pl
from pypto.jit.decorator import jit

LAYOUT = pl.ND


@jit.incore
def helper(src: pl.Tensor[[64, 64], pl.FP16, LAYOUT], dst: pl.Out[pl.Tensor]) -> pl.Tensor:
    tile = pl.load(src, [0, 0], [64, 64])
    pl.store(tile, [0, 0], dst)
    return dst
