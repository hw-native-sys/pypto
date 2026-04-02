# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for NormalizeReturnOrder pass."""

import pypto.language as pl
import pytest
from pypto import ir, passes


def _run_normalize(program):
    """Run normalize_return_order without property verification."""
    pipeline = passes.PassPipeline()
    pipeline.add_pass(passes.normalize_return_order())
    ctx = passes.PassContext([], passes.VerificationLevel.NONE)
    with ctx:
        return pipeline.run(program)


class TestNormalizeReturnOrder:
    """Tests for the NormalizeReturnOrder pass."""

    def test_swapped_returns_reordered(self):
        """Two Out params with returns in wrong order → reordered + call site TupleGetItem updated."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out_a: pl.Out[pl.Tensor[[16], pl.FP32]],
                out_b: pl.Out[pl.Tensor[[16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
                x_tile: pl.Tile[[16], pl.FP32] = pl.load(x, [0], [16])
                a_tile: pl.Tile[[16], pl.FP32] = pl.tile.add(x_tile, x_tile)
                b_tile: pl.Tile[[16], pl.FP32] = pl.tile.mul(x_tile, x_tile)
                out_b_store: pl.Tensor[[16], pl.FP32] = pl.store(b_tile, [0], out_b)
                out_a_store: pl.Tensor[[16], pl.FP32] = pl.store(a_tile, [0], out_a)
                return (out_b_store, out_a_store)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out_a: pl.Out[pl.Tensor[[16], pl.FP32]],
                out_b: pl.Out[pl.Tensor[[16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
                ret: tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]] = self.kernel(x, out_a, out_b)
                a: pl.Tensor[[16], pl.FP32] = ret[0]
                b: pl.Tensor[[16], pl.FP32] = ret[1]
                return (a, b)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out_a: pl.Out[pl.Tensor[[16], pl.FP32]],
                out_b: pl.Out[pl.Tensor[[16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
                x_tile: pl.Tile[[16], pl.FP32] = pl.load(x, [0], [16])
                a_tile: pl.Tile[[16], pl.FP32] = pl.tile.add(x_tile, x_tile)
                b_tile: pl.Tile[[16], pl.FP32] = pl.tile.mul(x_tile, x_tile)
                out_b_store: pl.Tensor[[16], pl.FP32] = pl.store(b_tile, [0], out_b)
                out_a_store: pl.Tensor[[16], pl.FP32] = pl.store(a_tile, [0], out_a)
                return (out_a_store, out_b_store)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out_a: pl.Out[pl.Tensor[[16], pl.FP32]],
                out_b: pl.Out[pl.Tensor[[16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
                ret: tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]] = self.kernel(x, out_a, out_b)
                a: pl.Tensor[[16], pl.FP32] = ret[1]
                b: pl.Tensor[[16], pl.FP32] = ret[0]
                return (a, b)

        After = _run_normalize(Before)
        ir.assert_structural_equal(After, Expected)

    def test_already_ordered_noop(self):
        """Two Out params with returns already in Out-param order → no change."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out_a: pl.Out[pl.Tensor[[16], pl.FP32]],
                out_b: pl.Out[pl.Tensor[[16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
                x_tile: pl.Tile[[16], pl.FP32] = pl.load(x, [0], [16])
                a_tile: pl.Tile[[16], pl.FP32] = pl.tile.add(x_tile, x_tile)
                b_tile: pl.Tile[[16], pl.FP32] = pl.tile.mul(x_tile, x_tile)
                out_a_store: pl.Tensor[[16], pl.FP32] = pl.store(a_tile, [0], out_a)
                out_b_store: pl.Tensor[[16], pl.FP32] = pl.store(b_tile, [0], out_b)
                return (out_a_store, out_b_store)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out_a: pl.Out[pl.Tensor[[16], pl.FP32]],
                out_b: pl.Out[pl.Tensor[[16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
                ret: tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]] = self.kernel(x, out_a, out_b)
                a: pl.Tensor[[16], pl.FP32] = ret[0]
                b: pl.Tensor[[16], pl.FP32] = ret[1]
                return (a, b)

        After = _run_normalize(Before)
        ir.assert_structural_equal(After, Before)

    def test_single_return_noop(self):
        """Single Out param with single return → no reorder needed."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out_0: pl.Out[pl.Tensor[[16], pl.FP32]],
            ) -> pl.Tensor[[16], pl.FP32]:
                x_tile: pl.Tile[[16], pl.FP32] = pl.load(x, [0], [16])
                y_tile: pl.Tile[[16], pl.FP32] = pl.tile.add(x_tile, x_tile)
                out_0_store: pl.Tensor[[16], pl.FP32] = pl.store(y_tile, [0], out_0)
                return out_0_store

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out_0: pl.Out[pl.Tensor[[16], pl.FP32]],
            ) -> pl.Tensor[[16], pl.FP32]:
                result: pl.Tensor[[16], pl.FP32] = self.kernel(x, out_0)
                return result

        After = _run_normalize(Before)
        ir.assert_structural_equal(After, Before)

    def test_non_incore_unchanged(self):
        """Program with only non-InCore functions → unchanged."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
                y: pl.Tensor[[16], pl.FP32] = pl.add(x, x)
                return y

        After = _run_normalize(Before)
        ir.assert_structural_equal(After, Before)

    def test_three_returns_scrambled(self):
        """Three Out params with return order [c, a, b] → normalized to [a, b, c]."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out_a: pl.Out[pl.Tensor[[16], pl.FP32]],
                out_b: pl.Out[pl.Tensor[[16], pl.FP32]],
                out_c: pl.Out[pl.Tensor[[16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
                x_tile: pl.Tile[[16], pl.FP32] = pl.load(x, [0], [16])
                a_tile: pl.Tile[[16], pl.FP32] = pl.tile.add(x_tile, x_tile)
                b_tile: pl.Tile[[16], pl.FP32] = pl.tile.mul(x_tile, x_tile)
                c_tile: pl.Tile[[16], pl.FP32] = pl.tile.sub(x_tile, x_tile)
                out_c_store: pl.Tensor[[16], pl.FP32] = pl.store(c_tile, [0], out_c)
                out_a_store: pl.Tensor[[16], pl.FP32] = pl.store(a_tile, [0], out_a)
                out_b_store: pl.Tensor[[16], pl.FP32] = pl.store(b_tile, [0], out_b)
                return (out_c_store, out_a_store, out_b_store)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out_a: pl.Out[pl.Tensor[[16], pl.FP32]],
                out_b: pl.Out[pl.Tensor[[16], pl.FP32]],
                out_c: pl.Out[pl.Tensor[[16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
                ret: tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]] = (
                    self.kernel(x, out_a, out_b, out_c)
                )
                c: pl.Tensor[[16], pl.FP32] = ret[0]
                a: pl.Tensor[[16], pl.FP32] = ret[1]
                b: pl.Tensor[[16], pl.FP32] = ret[2]
                return (c, a, b)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out_a: pl.Out[pl.Tensor[[16], pl.FP32]],
                out_b: pl.Out[pl.Tensor[[16], pl.FP32]],
                out_c: pl.Out[pl.Tensor[[16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
                x_tile: pl.Tile[[16], pl.FP32] = pl.load(x, [0], [16])
                a_tile: pl.Tile[[16], pl.FP32] = pl.tile.add(x_tile, x_tile)
                b_tile: pl.Tile[[16], pl.FP32] = pl.tile.mul(x_tile, x_tile)
                c_tile: pl.Tile[[16], pl.FP32] = pl.tile.sub(x_tile, x_tile)
                out_c_store: pl.Tensor[[16], pl.FP32] = pl.store(c_tile, [0], out_c)
                out_a_store: pl.Tensor[[16], pl.FP32] = pl.store(a_tile, [0], out_a)
                out_b_store: pl.Tensor[[16], pl.FP32] = pl.store(b_tile, [0], out_b)
                return (out_a_store, out_b_store, out_c_store)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out_a: pl.Out[pl.Tensor[[16], pl.FP32]],
                out_b: pl.Out[pl.Tensor[[16], pl.FP32]],
                out_c: pl.Out[pl.Tensor[[16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
                ret: tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]] = (
                    self.kernel(x, out_a, out_b, out_c)
                )
                c: pl.Tensor[[16], pl.FP32] = ret[2]
                a: pl.Tensor[[16], pl.FP32] = ret[0]
                b: pl.Tensor[[16], pl.FP32] = ret[1]
                return (c, a, b)

        After = _run_normalize(Before)
        ir.assert_structural_equal(After, Expected)

    def test_2d_tensor_reorder(self):
        """2D tensors: tile.store offset args don't affect param detection (offsets are MakeTuple)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[4, 16], pl.FP32],
                out_a: pl.Out[pl.Tensor[[4, 16], pl.FP32]],
                out_b: pl.Out[pl.Tensor[[4, 16], pl.FP32]],
            ) -> tuple[pl.Tensor[[4, 16], pl.FP32], pl.Tensor[[4, 16], pl.FP32]]:
                x_tile: pl.Tile[[4, 16], pl.FP32] = pl.load(x, [0, 0], [4, 16])
                a_tile: pl.Tile[[4, 16], pl.FP32] = pl.tile.add(x_tile, x_tile)
                b_tile: pl.Tile[[4, 16], pl.FP32] = pl.tile.mul(x_tile, x_tile)
                out_b_store: pl.Tensor[[4, 16], pl.FP32] = pl.store(b_tile, [0, 0], out_b)
                out_a_store: pl.Tensor[[4, 16], pl.FP32] = pl.store(a_tile, [0, 0], out_a)
                return (out_b_store, out_a_store)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[4, 16], pl.FP32],
                out_a: pl.Out[pl.Tensor[[4, 16], pl.FP32]],
                out_b: pl.Out[pl.Tensor[[4, 16], pl.FP32]],
            ) -> tuple[pl.Tensor[[4, 16], pl.FP32], pl.Tensor[[4, 16], pl.FP32]]:
                ret: tuple[pl.Tensor[[4, 16], pl.FP32], pl.Tensor[[4, 16], pl.FP32]] = self.kernel(
                    x, out_a, out_b
                )
                a: pl.Tensor[[4, 16], pl.FP32] = ret[0]
                b: pl.Tensor[[4, 16], pl.FP32] = ret[1]
                return (a, b)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[4, 16], pl.FP32],
                out_a: pl.Out[pl.Tensor[[4, 16], pl.FP32]],
                out_b: pl.Out[pl.Tensor[[4, 16], pl.FP32]],
            ) -> tuple[pl.Tensor[[4, 16], pl.FP32], pl.Tensor[[4, 16], pl.FP32]]:
                x_tile: pl.Tile[[4, 16], pl.FP32] = pl.load(x, [0, 0], [4, 16])
                a_tile: pl.Tile[[4, 16], pl.FP32] = pl.tile.add(x_tile, x_tile)
                b_tile: pl.Tile[[4, 16], pl.FP32] = pl.tile.mul(x_tile, x_tile)
                out_b_store: pl.Tensor[[4, 16], pl.FP32] = pl.store(b_tile, [0, 0], out_b)
                out_a_store: pl.Tensor[[4, 16], pl.FP32] = pl.store(a_tile, [0, 0], out_a)
                return (out_a_store, out_b_store)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[4, 16], pl.FP32],
                out_a: pl.Out[pl.Tensor[[4, 16], pl.FP32]],
                out_b: pl.Out[pl.Tensor[[4, 16], pl.FP32]],
            ) -> tuple[pl.Tensor[[4, 16], pl.FP32], pl.Tensor[[4, 16], pl.FP32]]:
                ret: tuple[pl.Tensor[[4, 16], pl.FP32], pl.Tensor[[4, 16], pl.FP32]] = self.kernel(
                    x, out_a, out_b
                )
                a: pl.Tensor[[4, 16], pl.FP32] = ret[1]
                b: pl.Tensor[[4, 16], pl.FP32] = ret[0]
                return (a, b)

        After = _run_normalize(Before)
        ir.assert_structural_equal(After, Expected)

    def test_inout_param_reorder(self):
        """InOut params also participate in return reordering."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16], pl.FP32],
                a: pl.InOut[pl.Tensor[[16], pl.FP32]],
                b: pl.InOut[pl.Tensor[[16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
                x_tile: pl.Tile[[16], pl.FP32] = pl.load(x, [0], [16])
                a_tile: pl.Tile[[16], pl.FP32] = pl.tile.add(x_tile, x_tile)
                b_tile: pl.Tile[[16], pl.FP32] = pl.tile.mul(x_tile, x_tile)
                b_store: pl.Tensor[[16], pl.FP32] = pl.store(b_tile, [0], b)
                a_store: pl.Tensor[[16], pl.FP32] = pl.store(a_tile, [0], a)
                return (b_store, a_store)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16], pl.FP32],
                a: pl.InOut[pl.Tensor[[16], pl.FP32]],
                b: pl.InOut[pl.Tensor[[16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
                ret: tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]] = self.kernel(x, a, b)
                ra: pl.Tensor[[16], pl.FP32] = ret[0]
                rb: pl.Tensor[[16], pl.FP32] = ret[1]
                return (ra, rb)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16], pl.FP32],
                a: pl.InOut[pl.Tensor[[16], pl.FP32]],
                b: pl.InOut[pl.Tensor[[16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
                x_tile: pl.Tile[[16], pl.FP32] = pl.load(x, [0], [16])
                a_tile: pl.Tile[[16], pl.FP32] = pl.tile.add(x_tile, x_tile)
                b_tile: pl.Tile[[16], pl.FP32] = pl.tile.mul(x_tile, x_tile)
                b_store: pl.Tensor[[16], pl.FP32] = pl.store(b_tile, [0], b)
                a_store: pl.Tensor[[16], pl.FP32] = pl.store(a_tile, [0], a)
                return (a_store, b_store)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16], pl.FP32],
                a: pl.InOut[pl.Tensor[[16], pl.FP32]],
                b: pl.InOut[pl.Tensor[[16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
                ret: tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]] = self.kernel(x, a, b)
                ra: pl.Tensor[[16], pl.FP32] = ret[1]
                rb: pl.Tensor[[16], pl.FP32] = ret[0]
                return (ra, rb)

        After = _run_normalize(Before)
        ir.assert_structural_equal(After, Expected)


class TestNormalizeReturnOrderProperties:
    """Verify pass metadata and properties."""

    def test_pass_name(self):
        p = passes.normalize_return_order()
        assert p.get_name() == "NormalizeReturnOrder"

    def test_required_properties(self):
        p = passes.normalize_return_order()
        required = p.get_required_properties()
        assert required.contains(passes.IRProperty.SplitIncoreOrch)
        assert required.contains(passes.IRProperty.IncoreTileOps)

    def test_no_produced_properties(self):
        p = passes.normalize_return_order()
        produced = p.get_produced_properties()
        assert produced.empty()

    def test_no_invalidated_properties(self):
        p = passes.normalize_return_order()
        invalidated = p.get_invalidated_properties()
        assert invalidated.empty()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
