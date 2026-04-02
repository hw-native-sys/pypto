# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for SubstituteTiles pass."""

import pypto.language as pl
from pypto import ir, passes


class OpAndLoopCollector(ir.IRVisitor):
    def __init__(self):
        super().__init__()
        self.op_names: list[str] = []
        self.for_count = 0

    def visit_call(self, op):
        self.op_names.append(op.op.name)
        super().visit_call(op)

    def visit_for_stmt(self, op):
        self.for_count += 1
        super().visit_for_stmt(op)


class TestSubstituteTiles:
    """Tests for SubstituteTiles pass."""

    def test_expand_clone_broadcast_axis(self):
        """tile.expand_clone with one broadcast axis becomes tile.create + tile.assemble + for loop."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self) -> pl.Tile[[4, 8, 16], pl.FP32]:
                tile_src: pl.Tile[[4, 1, 16], pl.FP32] = pl.tile.create_tile([4, 1, 16], dtype=pl.FP32)
                tile_dst: pl.Tile[[4, 8, 16], pl.FP32] = pl.tile.expand_clone(tile_src, [4, 8, 16])
                return tile_dst

        After = passes.substitute_tiles()(Before)

        collector = OpAndLoopCollector()
        collector.visit_program(After)

        assert "tile.expand_clone" not in collector.op_names
        assert "tile.assemble" in collector.op_names
        assert collector.for_count == 1

    def test_expand_clone_no_broadcast(self):
        """tile.expand_clone with no broadcast axis becomes tile.create + tile.assemble without loops."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self) -> pl.Tile[[4, 1, 16], pl.FP32]:
                tile_src: pl.Tile[[4, 1, 16], pl.FP32] = pl.tile.create_tile([4, 1, 16], dtype=pl.FP32)
                tile_dst: pl.Tile[[4, 1, 16], pl.FP32] = pl.tile.expand_clone(tile_src, [4, 1, 16])
                return tile_dst

        After = passes.substitute_tiles()(Before)

        collector = OpAndLoopCollector()
        collector.visit_program(After)

        assert "tile.expand_clone" not in collector.op_names
        assert "tile.assemble" in collector.op_names
        assert collector.for_count == 0

    def test_expand_clone_structural_equal(self):
        """Substitute tile.expand_clone into create+assemble+SSA loop and compare IR."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(self) -> pl.Tile[[4, 8, 16], pl.FP32]:
                tile_src: pl.Tile[[4, 1, 16], pl.FP32] = pl.tile.create_tile([4, 1, 16], dtype=pl.FP32)
                tile_dst: pl.Tile[[4, 8, 16], pl.FP32] = pl.tile.expand_clone(tile_src, [4, 8, 16])
                return tile_dst

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True, type=pl.FunctionType.InCore)
            def main_incore_0(self) -> pl.Tile[[4, 8, 16], pl.FP32]:
                tile_src: pl.Tile[[4, 1, 16], pl.FP32] = pl.tile.create_tile([4, 1, 16], dtype=pl.FP32)
                tile_init: pl.Tile[[4, 8, 16], pl.FP32] = pl.tile.create_tile([4, 8, 16], dtype=pl.FP32)
                for i, (tile_iter,) in pl.range(8, init_values=(tile_init,)):
                    tile_next: pl.Tile[[4, 8, 16], pl.FP32] = pl.tile.assemble(tile_iter, tile_src, [0, i, 0])
                    tile_rv = pl.yield_(tile_next)
                return tile_rv

        After = passes.substitute_tiles()(Before)
        ir.assert_structural_equal(After, Expected)
