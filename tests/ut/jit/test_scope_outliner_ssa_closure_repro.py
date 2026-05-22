# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression repro for ScopeOutliner freshened Var SSA closure.

This file intentionally keeps the witness independent from phase-fence passes:
the diamond-inline program is compiled with the upstream default pipeline, then
the resulting whole program is checked for SSAForm explicitly.
"""

import pypto.language as pl
import pytest
from pypto.jit.decorator import jit
from pypto.pypto_core import passes


def test_inline_diamond_outlined_body_uses_freshened_signature_vars():
    torch = pytest.importorskip("torch")

    @jit.inline
    def shared(a: pl.Tensor, out: pl.Out[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP):
            tile = pl.load(a, [0, 0], [32, 32])
            pl.store(tile, [0, 0], out)
        return out

    @jit.inline
    def a_helper(a: pl.Tensor, out: pl.Out[pl.Tensor]):
        out = shared(a, out)
        return out

    @jit.inline
    def b_helper(a: pl.Tensor, out: pl.Out[pl.Tensor]):
        out = shared(a, out)
        return out

    @jit
    def entry_diamond(a: pl.Tensor, out: pl.Out[pl.Tensor]):
        out = a_helper(a, out)
        out = b_helper(a, out)
        return out

    post_pass = entry_diamond.compile_for_test(torch.randn(32, 32), torch.empty(32, 32))

    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.SSAForm)
    passes.verify_properties(props, post_pass, "inline_diamond_scope_outliner_ssa_repro")
