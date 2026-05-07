# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime scope context managers for the PyPTO Language DSL."""


class manual_scope:
    """Context manager for a manual-dependency runtime scope.

    Inside this block, the simpler runtime skips OverlapMap dependency
    tracking and TensorMap insert. Dependencies are emitted explicitly by
    the compiler:

      1. Auto-derived from SSA data flow: each tensor argument referencing a
         Var produced by a prior kernel call in the same ``manual_scope``
         becomes an explicit dep edge.
      2. User-supplied via the ``deps=[var1, var2, ...]`` kwarg on any
         ``self.kernel(...)`` call inside this block.
      3. ``pl.no_dep(arg)`` (Phase 2) suppresses the auto-derived edge for
         that single argument.

    Usage::

        with pl.manual_scope():
            sij = self.qk_matmul(qi, kj, sij_buf)
            pij = self.softmax(sij)                       # auto edge: sij → softmax
            oi = self.pv_matmul(pij, vj, oi_buf,
                                deps=[sij])               # auto + user edge
            other = self.peek(pl.no_dep(pij))             # no auto edge for pij

    Restrictions:
      - Must appear inside an Orchestration function (not InCore).
      - Cannot be nested inside another ``manual_scope`` (runtime forbids).
      - Each kernel call's total deps (auto + user) must not exceed the
        runtime cap of 16; the compiler reports an error otherwise.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


__all__ = ["manual_scope"]
