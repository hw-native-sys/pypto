# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the TileTypeCoherence property verifier.

The verifier asserts the canonical encoding of TileType: an implicit-for-(shape,
memory_space) tile_view is stored as None. The TileType constructor enforces
this invariant; the verifier catches passes that mutate the public ``tile_view_``
field directly without going through the constructor.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes


def _verify(program: ir.Program) -> list:
    """Run the TileTypeCoherence verifier and return its diagnostics."""
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.TileTypeCoherence)
    return passes.PropertyVerifierRegistry.verify(props, program)


def test_canonical_program_has_no_diagnostics():
    """A program with canonical TileTypes (implicit views stored as None) verifies clean."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main_incore_0(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            out_0: pl.Out[pl.Tensor[[16, 128], pl.BF16]],
        ) -> pl.Tensor[[16, 128], pl.BF16]:
            x_tile: pl.Tile[[16, 128], pl.BF16, pl.MemorySpace.Mat] = pl.load(
                x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
            )
            out_0: pl.Tensor[[16, 128], pl.BF16] = pl.store(x_tile, [0, 0], out_0)
            return out_0

    diagnostics = _verify(Program)
    assert len(diagnostics) == 0


def test_constructor_canonicalizes_implicit_acc_view():
    """An Acc-implicit tile_view passed to the constructor is collapsed to None.

    This pins the canonical-encoding invariant the verifier asserts. If this test
    fails, the verifier itself becomes meaningless.
    """
    shape = [16, 128]
    span = ir.Span.unknown()
    valid_shape = [ir.ConstInt(16, ir.DataType.INDEX, span), ir.ConstInt(128, ir.DataType.INDEX, span)]
    implicit_acc_view = ir.TileView(
        valid_shape=valid_shape,
        blayout=ir.TileLayout.col_major,
        slayout=ir.TileLayout.row_major,
        fractal=1024,
    )
    t = ir.TileType(shape, ir.DataType.FP32, None, implicit_acc_view, ir.MemorySpace.Acc)
    assert t.tile_view is None  # canonicalized away

    # The effective view is still col_major / row_major / 1024 — semantic layout is preserved.
    eff = t.get_effective_tile_view()
    assert eff.blayout == ir.TileLayout.col_major
    assert eff.slayout == ir.TileLayout.row_major
    assert eff.fractal == 1024


def test_explicit_non_implicit_view_is_preserved():
    """A view that does not match the implicit-for-(shape, memory_space) form stays present."""
    shape = [16, 128]
    span = ir.Span.unknown()
    # Non-Acc-implicit: blayout=row_major (Vec-style) on an Acc tile is non-canonical.
    weird_view = ir.TileView(
        valid_shape=[ir.ConstInt(16, ir.DataType.INDEX, span), ir.ConstInt(128, ir.DataType.INDEX, span)],
        blayout=ir.TileLayout.row_major,
        slayout=ir.TileLayout.none_box,
        fractal=512,
    )
    t = ir.TileType(shape, ir.DataType.FP32, None, weird_view, ir.MemorySpace.Acc)
    assert t.tile_view is not None
    assert t.tile_view.blayout == ir.TileLayout.row_major


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
