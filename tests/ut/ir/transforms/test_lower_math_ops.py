# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the LowerMathOps pass skeleton.

The LowerMathOps pass will eventually decompose ``tile.sin`` / ``tile.cos`` into
primitive arithmetic tile ops (Cody-Waite range reduction + degree-9 Horner
polynomial).  This commit adds only the skeleton: the pass exists, is wired
into the build, bound to Python, and slotted into the pass manager — but it
performs no decomposition yet.  The actual lowering logic lands in a follow-up.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes


def test_lower_math_ops_pass_factory_exists():
    """The factory returns a Pass instance with the expected name."""
    p = passes.lower_math_ops()
    assert p is not None
    assert p.get_name() == "LowerMathOps"


def test_lower_math_ops_noop_on_no_trig():
    """Skeleton: pass must be safe to run on programs without sin/cos and leave them unchanged."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def main_incore_0(
            self,
            x: pl.Tensor[[16, 16], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            x_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
            y_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.exp(x_tile)
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.store(y_tile, [0, 0], out_0)
            return out_0

        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
            r: pl.Tensor[[16, 16], pl.FP32] = self.main_incore_0(x, out_0)
            return r

    After = passes.lower_math_ops()(Before)
    ir.assert_structural_equal(After, Before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
