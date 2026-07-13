# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fixture for signature-mode ``compile()`` closure-var resolution (issue #1996).

This module uses ``from __future__ import annotations`` so every parameter
annotation is a *string* at runtime. :func:`make_closure_kernel` returns a
``@pl.jit`` kernel defined inside a function whose tensor annotations reference a
``pl.dynamic`` var captured as a **closure free variable** (not a module global).
Resolving those string annotations therefore requires
``typing.get_type_hints(..., globalns=<globals + closure free-vars>)`` — the
exact path signature-mode compile exercises.
"""

from __future__ import annotations

import pypto.language as pl
from pypto.jit.decorator import jit


def make_closure_kernel():
    """Return a JIT kernel whose annotations reference a closure-scope dynvar.

    ``ROWS`` is referenced in each body (via ``bind_dynamic``), so Python
    captures it as a closure free variable — the case where resolving the
    string annotation needs the closure bindings, not just module globals.
    """
    ROWS = pl.dynamic("ROWS")  # closure free var, not a module global

    @jit.incore
    def _closure_incore(a: pl.Tensor[[ROWS, 64], pl.FP32], c: pl.Out[pl.Tensor[[ROWS, 64], pl.FP32]]):
        a.bind_dynamic(0, ROWS)
        c.bind_dynamic(0, ROWS)
        t = pl.load(a, [0, 0], [64, 64])
        pl.store(t, [0, 0], c)
        return c

    @jit
    def closure_kernel(a: pl.Tensor[[ROWS, 64], pl.FP32], c: pl.Out[pl.Tensor[[ROWS, 64], pl.FP32]]):
        a.bind_dynamic(0, ROWS)
        c.bind_dynamic(0, ROWS)
        c = _closure_incore(a, c)
        return c

    return closure_kernel
