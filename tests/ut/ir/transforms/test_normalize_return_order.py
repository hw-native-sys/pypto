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
    """Run normalize_return_order via a single-pass pipeline."""
    pipeline = passes.PassPipeline()
    pipeline.add_pass(passes.normalize_return_order())
    return pipeline.run(program)


def _run_normalize_direct(program):
    """Run normalize_return_order via a direct pass invocation.

    ``manual_scope``/``submit`` programs trip the pipeline's PostPipeline
    perf-hint diagnostic (it needs a configured backend handler), so the
    Submit-bearing cases call the pass object directly rather than through
    a ``PassPipeline``. The single-pass behaviour is identical; only the
    pipeline's post-run diagnostic checks are skipped.
    """
    return passes.normalize_return_order()(program)


class TestNormalizeReturnOrder:
    """Tests for the NormalizeReturnOrder pass."""

    def test_swapped_returns_reordered(self):
        """Two Out params with returns in wrong order → reordered + canonicalized to param Vars
        + call site TupleGetItem updated."""

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
                out_b_store: pl.Tensor[[16], pl.FP32] = pl.store(b_tile, [0], out_b)  # noqa: F841
                out_a_store: pl.Tensor[[16], pl.FP32] = pl.store(a_tile, [0], out_a)  # noqa: F841
                return (out_a, out_b)

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
        """Two Out params with returns already in Out-param order → only
        return values canonicalized to the param Vars; call sites unchanged."""

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
                out_a_store: pl.Tensor[[16], pl.FP32] = pl.store(a_tile, [0], out_a)  # noqa: F841
                out_b_store: pl.Tensor[[16], pl.FP32] = pl.store(b_tile, [0], out_b)  # noqa: F841
                return (out_a, out_b)

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
        ir.assert_structural_equal(After, Expected)

    def test_group_wrapper_declaring_pl_tuple_return_stays_one_value(self):
        """A Group wrapper declaring a single ``pl.Tuple[...]`` return keeps its ONE return value.

        ``-> pl.Tuple[A, B]`` declares ONE return type (a TupleType); ``-> tuple[A, B]``
        declares two. The forwarded-tuple expansion — which turns a wrapper's single
        ``return packed`` into N explicit param returns — must therefore NOT fire here:
        the wrapper has one declared return position, and expanding it would leave a
        two-value ReturnStmt that its one-entry ``return_types_`` cannot describe.

        Only ``kernel`` (which declares two flat positions) is canonicalized; the Group
        wrapper is left exactly as written.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV)
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

            @pl.function(type=pl.FunctionType.Group)
            def group_func(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out_a: pl.Out[pl.Tensor[[16], pl.FP32]],
                out_b: pl.Out[pl.Tensor[[16], pl.FP32]],
            ) -> pl.Tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
                packed: tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]] = self.kernel(
                    x, out_a, out_b
                )
                return packed

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out_a: pl.Out[pl.Tensor[[16], pl.FP32]],
                out_b: pl.Out[pl.Tensor[[16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
                x_tile: pl.Tile[[16], pl.FP32] = pl.load(x, [0], [16])
                a_tile: pl.Tile[[16], pl.FP32] = pl.tile.add(x_tile, x_tile)
                b_tile: pl.Tile[[16], pl.FP32] = pl.tile.mul(x_tile, x_tile)
                out_a_store: pl.Tensor[[16], pl.FP32] = pl.store(a_tile, [0], out_a)  # noqa: F841
                out_b_store: pl.Tensor[[16], pl.FP32] = pl.store(b_tile, [0], out_b)  # noqa: F841
                return (out_a, out_b)

            @pl.function(type=pl.FunctionType.Group)
            def group_func(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out_a: pl.Out[pl.Tensor[[16], pl.FP32]],
                out_b: pl.Out[pl.Tensor[[16], pl.FP32]],
            ) -> pl.Tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
                packed: tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]] = self.kernel(
                    x, out_a, out_b
                )
                return packed

        After = _run_normalize(Before)
        ir.assert_structural_equal(After, Expected)

        # The wrapper's ReturnStmt arity must still match its ONE declared TupleType return.
        group_func = After.get_function("group_func")
        assert group_func is not None
        assert len(group_func.return_types) == 1
        assert isinstance(group_func.return_types[0], ir.TupleType)

    def test_single_return_noop(self):
        """Single Out param with single return → no reorder; return canonicalized to the param Var."""

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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out_0: pl.Out[pl.Tensor[[16], pl.FP32]],
            ) -> pl.Tensor[[16], pl.FP32]:
                x_tile: pl.Tile[[16], pl.FP32] = pl.load(x, [0], [16])
                y_tile: pl.Tile[[16], pl.FP32] = pl.tile.add(x_tile, x_tile)
                out_0_store: pl.Tensor[[16], pl.FP32] = pl.store(y_tile, [0], out_0)  # noqa: F841
                return out_0

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out_0: pl.Out[pl.Tensor[[16], pl.FP32]],
            ) -> pl.Tensor[[16], pl.FP32]:
                result: pl.Tensor[[16], pl.FP32] = self.kernel(x, out_0)
                return result

        After = _run_normalize(Before)
        ir.assert_structural_equal(After, Expected)

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
                out_c_store: pl.Tensor[[16], pl.FP32] = pl.store(c_tile, [0], out_c)  # noqa: F841
                out_a_store: pl.Tensor[[16], pl.FP32] = pl.store(a_tile, [0], out_a)  # noqa: F841
                out_b_store: pl.Tensor[[16], pl.FP32] = pl.store(b_tile, [0], out_b)  # noqa: F841
                return (out_a, out_b, out_c)

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
                out_b_store: pl.Tensor[[4, 16], pl.FP32] = pl.store(b_tile, [0, 0], out_b)  # noqa: F841
                out_a_store: pl.Tensor[[4, 16], pl.FP32] = pl.store(a_tile, [0, 0], out_a)  # noqa: F841
                return (out_a, out_b)

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
                b_store: pl.Tensor[[16], pl.FP32] = pl.store(b_tile, [0], b)  # noqa: F841
                a_store: pl.Tensor[[16], pl.FP32] = pl.store(a_tile, [0], a)  # noqa: F841
                return (a, b)

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


class TestNormalizeReturnOrderSubmit:
    """Step B must remap ``TupleGetItemExpr`` indices on a ``pl.submit`` result
    just as it does for a plain ``self.kernel(...)`` Call.

    A ``pl.submit(self.kernel, ...)`` inside ``pl.manual_scope()`` desugars to
    a ``Submit`` node whose flat return type is
    ``Tuple[<kernel return>..., Scalar[TASK_ID]]``; the unpack
    ``(a, b), tid = pl.submit(...)`` becomes ``_submit_tmp[0]`` / ``[1]`` /
    ``[2]``. When Step A reorders the InCore kernel's returns, those
    projection indices must be permuted in lockstep so the same physical
    output buffer still flows into the same name (doc
    ``27-normalize_return_order.md`` §"Step B"; pass principle in
    ``.claude/rules/pass-submit-awareness.md``).
    """

    @pytest.mark.parametrize("out_b_rows", [16, 32])
    def test_submit_swapped_returns_remapped(self, out_b_rows):
        """InCore kernel returns swapped + result consumed via ``pl.submit`` →
        kernel returns reordered AND the submit-result projection indices
        permuted so ``a``/``b`` keep binding the same buffers.

        Derivation: original kernel ``return (out_b_store, out_a_store)`` maps
        return[0]→out_b (param 2), return[1]→out_a (param 1). With
        out_indices ``[1, 2]`` that yields permutation ``[1, 0]`` — kernel
        becomes ``return (out_a_store, out_b_store)``. Step B then rewrites the
        caller's projections by ``permutation[old_index]``: ``a`` was
        ``_submit_tmp[0]`` → ``_submit_tmp[1]``; ``b`` was ``_submit_tmp[1]`` →
        ``_submit_tmp[0]``; ``tid`` at index 2 (>= perm size) is untouched.
        The ``(b, a), tid`` unpack in Expected encodes exactly that:
        ``b = _submit_tmp[0]`` and ``a = _submit_tmp[1]``.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out_a: pl.Out[pl.Tensor[[16], pl.FP32]],
                out_b: pl.Out[pl.Tensor[[out_b_rows], pl.FP32]],
            ) -> tuple[pl.Tensor[[out_b_rows], pl.FP32], pl.Tensor[[16], pl.FP32]]:
                x_tile: pl.Tile[[16], pl.FP32] = pl.load(x, [0], [16])
                a_tile: pl.Tile[[16], pl.FP32] = pl.tile.add(x_tile, x_tile)
                b_tile: pl.Tile[[16], pl.FP32] = pl.tile.mul(x_tile, x_tile)
                out_b_store: pl.Tensor[[out_b_rows], pl.FP32] = pl.store(b_tile, [0], out_b)
                out_a_store: pl.Tensor[[16], pl.FP32] = pl.store(a_tile, [0], out_a)
                return (out_b_store, out_a_store)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out_a: pl.Out[pl.Tensor[[16], pl.FP32]],
                out_b: pl.Out[pl.Tensor[[out_b_rows], pl.FP32]],
            ) -> tuple[pl.Tensor[[out_b_rows], pl.FP32], pl.Tensor[[16], pl.FP32]]:
                with pl.manual_scope():
                    (a, b), tid = pl.submit(self.kernel, x, out_a, out_b)  # noqa: F841
                return (a, b)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out_a: pl.Out[pl.Tensor[[16], pl.FP32]],
                out_b: pl.Out[pl.Tensor[[out_b_rows], pl.FP32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[out_b_rows], pl.FP32]]:
                x_tile: pl.Tile[[16], pl.FP32] = pl.load(x, [0], [16])
                a_tile: pl.Tile[[16], pl.FP32] = pl.tile.add(x_tile, x_tile)
                b_tile: pl.Tile[[16], pl.FP32] = pl.tile.mul(x_tile, x_tile)
                out_b_store: pl.Tensor[[out_b_rows], pl.FP32] = pl.store(b_tile, [0], out_b)  # noqa: F841
                out_a_store: pl.Tensor[[16], pl.FP32] = pl.store(a_tile, [0], out_a)  # noqa: F841
                return (out_a, out_b)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out_a: pl.Out[pl.Tensor[[16], pl.FP32]],
                out_b: pl.Out[pl.Tensor[[out_b_rows], pl.FP32]],
            ) -> tuple[pl.Tensor[[out_b_rows], pl.FP32], pl.Tensor[[16], pl.FP32]]:
                with pl.manual_scope():
                    # The pass remaps the submit-result projection indices IN
                    # PLACE (statement order preserved), exactly like the Call
                    # path in test_swapped_returns_reordered: with permutation
                    # [1, 0], `a` now reads _submit_tmp[1] (out_b, moved to slot
                    # 1) and `b` reads _submit_tmp[0] (out_a, moved to slot 0);
                    # `tid` at index 2 is past the permutation and untouched.
                    # This is the explicit-subscript form the pass emits — a
                    # (b, a) tuple-unpack would instead reorder the statements
                    # (b before a), which the in-place remap does not do.
                    _submit_tmp = pl.submit(self.kernel, x, out_a, out_b)
                    a: pl.Tensor[[out_b_rows], pl.FP32] = _submit_tmp[1]
                    b: pl.Tensor[[16], pl.FP32] = _submit_tmp[0]
                    tid: pl.Scalar[pl.TASK_ID] = _submit_tmp[2]  # noqa: F841
                return (a, b)

        After = _run_normalize_direct(Before)
        ir.assert_structural_equal(After, Expected)

    def test_submit_already_ordered_noop(self):
        """A ``pl.submit`` of a kernel whose returns already match Out-param
        order needs no permutation → Step A produces no permutation, Step B
        never fires, and the Submit-bearing caller is left untouched. The only
        change is the kernel's returns being canonicalized to the param Vars.
        """

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
                with pl.manual_scope():
                    (a, b), tid = pl.submit(self.kernel, x, out_a, out_b)  # noqa: F841
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
                out_a_store: pl.Tensor[[16], pl.FP32] = pl.store(a_tile, [0], out_a)  # noqa: F841
                out_b_store: pl.Tensor[[16], pl.FP32] = pl.store(b_tile, [0], out_b)  # noqa: F841
                return (out_a, out_b)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out_a: pl.Out[pl.Tensor[[16], pl.FP32]],
                out_b: pl.Out[pl.Tensor[[16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
                with pl.manual_scope():
                    (a, b), tid = pl.submit(self.kernel, x, out_a, out_b)  # noqa: F841
                return (a, b)

        After = _run_normalize_direct(Before)
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

    def test_produced_properties(self):
        p = passes.normalize_return_order()
        produced = p.get_produced_properties()
        assert produced.contains(passes.IRProperty.ReturnParamsExplicit)

    def test_no_invalidated_properties(self):
        p = passes.normalize_return_order()
        invalidated = p.get_invalidated_properties()
        assert invalidated.empty()


class TestReturnParamsAreStructurallyExplicit:
    """The pass makes the return->param map readable off the ReturnStmt alone.

    Consumers downstream of this pass (orchestration codegen, ClassifyIterArgCarry)
    read return position `j` -> param `i` by pointer identity instead of tracing
    SSA lineage across functions, so the canonical form is load-bearing.
    """

    def _return_values(self, program, func_name):
        for _gv, func in program.functions.items():
            if func.name != func_name:
                continue
            body = func.body
            stmts = list(body.stmts) if isinstance(body, ir.SeqStmts) else [body]
            ret = stmts[-1]
            assert isinstance(ret, ir.ReturnStmt), f"'{func_name}' body does not end in a ReturnStmt"
            return list(ret.value), list(func.params)
        raise AssertionError(f"function '{func_name}' not found")

    def test_branch_writeback_and_wrapper_return_reference_param(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(
                self, out: pl.InOut[pl.Tensor[[4, 16], pl.FP32]], condition: pl.Scalar[pl.BOOL]
            ) -> pl.Tensor[[4, 16], pl.FP32]:
                if condition:
                    value = pl.tile.full([1, 16], pl.FP32, 1.0)
                    out = pl.store(value, [0, 0], out)
                else:
                    value = pl.tile.full([1, 16], pl.FP32, 0.0)
                    out = pl.store(value, [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Spmd)
            def wrapper(
                self, out: pl.InOut[pl.Tensor[[4, 16], pl.FP32]], condition: pl.Scalar[pl.BOOL]
            ) -> pl.Tensor[[4, 16], pl.FP32]:
                result = self.kernel(out, condition)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
            def kernel(
                self, out: pl.InOut[pl.Tensor[[4, 16], pl.FP32]], condition: pl.Scalar[pl.BOOL]
            ) -> pl.Tensor[[4, 16], pl.FP32]:
                if condition:
                    value_then = pl.tile.full([1, 16], pl.FP32, 1.0)
                    out_then = pl.store(value_then, [0, 0], out)
                    _merged_out, _merged_value = pl.yield_(out_then, value_then)
                else:
                    value_else = pl.tile.full([1, 16], pl.FP32, 0.0)
                    out_else = pl.store(value_else, [0, 0], out)
                    _merged_out, _merged_value = pl.yield_(out_else, value_else)
                return out

            @pl.function(type=pl.FunctionType.Spmd, strict_ssa=True)
            def wrapper(
                self, out: pl.InOut[pl.Tensor[[4, 16], pl.FP32]], condition: pl.Scalar[pl.BOOL]
            ) -> pl.Tensor[[4, 16], pl.FP32]:
                _result = self.kernel(out, condition)
                return out

        Before = passes.convert_to_ssa()(Before)
        values, params = self._return_values(Before, "kernel")
        assert values[0] is not params[0]
        After = _run_normalize_direct(Before)
        ir.assert_structural_equal(After, Expected)
        for name in ("kernel", "wrapper"):
            values, params = self._return_values(After, name)
            assert values[0] is params[0]

    def test_dynamic_branch_call_returns_reference_caller_param(self):
        """Merge metadata must use caller types forwarded by validated callees."""
        program_source = """
@pl.program
class {name}:
    @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
    def write(
        self, inner_n: pl.Scalar[pl.INDEX], out: pl.InOut[pl.Tensor[[inner_n, 16], pl.FP32]]
    ) -> pl.Tensor[[inner_n, 16], pl.FP32]:
        return out

    @pl.function(type=pl.FunctionType.Spmd, strict_ssa=True)
    def wrapper(
        self, outer_n: pl.Scalar[pl.INDEX], out: pl.InOut[pl.Tensor[[outer_n, 16], pl.FP32]],
        condition: pl.Scalar[pl.BOOL]
    ) -> pl.Tensor[[outer_n, 16], pl.FP32]:
        if condition:
            then_out = self.write(outer_n, out)
            merged = pl.yield_(then_out)
        else:
            else_out = self.write(outer_n, out)
            merged = pl.yield_(else_out)
        return {returned}
"""
        Before = pl.parse(program_source.replace("{name}", "Before").replace("{returned}", "merged"))
        Expected = pl.parse(program_source.replace("{name}", "Expected").replace("{returned}", "out"))
        assert isinstance(Before, ir.Program)
        assert isinstance(Expected, ir.Program)

        def retain_callee_shape_vars(program: ir.Program) -> ir.Program:
            callee = program.get_function("write")
            assert callee is not None
            replacements: dict[int, ir.Var] = {}

            class RetainCalleeShapeVars(ir.IRMutator):
                def visit_var(self, op: ir.Var) -> ir.Expr:
                    if op.name_hint != "merged":
                        return op
                    if op.unique_id not in replacements:
                        replacements[op.unique_id] = ir.Var(op.name_hint, callee.params[1].type, op.span)
                    return replacements[op.unique_id]

            return RetainCalleeShapeVars().visit_program(program)

        # Generated calls/phis can retain callee-local shape Vars until launch
        # legalization. Model that intermediate type without changing its roots.
        Before = retain_callee_shape_vars(Before)
        Expected = retain_callee_shape_vars(Expected)
        # Launch legalization remaps these provisional type Vars; this test
        # isolates the return-lineage query that must work before that remap.
        with passes.PassContext([], passes.VerificationLevel.NONE):
            After = _run_normalize_direct(Before)
        # Callee-local shape symbols are open references in this provisional wrapper.
        ir.assert_structural_equal(After, Expected, enable_auto_mapping=True)
        values, params = self._return_values(After, "wrapper")
        assert values[0] is params[1]

    @pytest.mark.parametrize("other_source", ["parameter", "fresh", "view", "valid_shape"])
    def test_branch_return_with_unproven_identity_is_not_canonicalized(self, other_source):
        source = {
            "parameter": "other",
            "fresh": "pl.create_tensor([4, 16], dtype=pl.FP32)",
            "view": "pl.tensor.view(out, [4, 16])",
            "valid_shape": "pl.tensor.set_validshape(out, 2, 16)",
        }[other_source]
        then_source = source if other_source in ("view", "valid_shape") else "pl.store(value, [0, 0], out)"
        Before = pl.parse(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(
        self,
        out: pl.InOut[pl.Tensor[[4, 16], pl.FP32]],
        other: pl.InOut[pl.Tensor[[4, 16], pl.FP32]],
        condition: pl.Scalar[pl.BOOL],
    ) -> pl.Tensor[[4, 16], pl.FP32]:
        if condition:
            value = pl.tile.full([1, 16], pl.FP32, 1.0)
            result = {then_source}
        else:
            result = {source}
        return result
""")

        assert isinstance(Before, ir.Program)
        Before = passes.convert_to_ssa()(Before)
        # These cases must leave the complete input program unchanged.
        Expected = Before
        before_values, _ = self._return_values(Before, "kernel")
        After = _run_normalize_direct(Before)
        ir.assert_structural_equal(After, Expected)
        values, params = self._return_values(After, "kernel")
        assert values[0] is before_values[0]
        assert all(values[0] is not param for param in params)

    @pytest.mark.parametrize("narrowed_first", [False, True])
    def test_writeback_after_branch_merge_preserves_returned_valid_shape(self, narrowed_first):
        """Final type checks also apply when the merge root is already cached."""

        program_source = """
@pl.program
class {name}:
    @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
    def kernel(
        self, out: pl.InOut[pl.Tensor[[4, 16], pl.FP32]], condition: pl.Scalar[pl.BOOL]
    ) -> tuple[pl.Tensor[[4, 16], pl.FP32], pl.Tensor[[4, 16], pl.FP32]]:
        if condition:
            then_value = pl.tile.full([1, 16], pl.FP32, 1.0)
            then_out = pl.store(then_value, [0, 0], out)
            merged_out = pl.yield_(then_out)
        else:
            else_value = pl.tile.full([1, 16], pl.FP32, 0.0)
            else_out = pl.store(else_value, [0, 0], out)
            merged_out = pl.yield_(else_out)
        result = pl.tensor.set_validshape(merged_out, 2, 16)
        return {returned}
"""
        before_return = "result, merged_out" if narrowed_first else "merged_out, result"
        expected_return = "result, out" if narrowed_first else "out, result"
        Before = pl.parse(program_source.replace("{name}", "Before").replace("{returned}", before_return))
        Expected = pl.parse(
            program_source.replace("{name}", "Expected").replace("{returned}", expected_return)
        )
        assert isinstance(Before, ir.Program)
        assert isinstance(Expected, ir.Program)
        After = _run_normalize_direct(Before)
        ir.assert_structural_equal(After, Expected)
        values, params = self._return_values(After, "kernel")
        assert values[int(narrowed_first)] is params[0]
        assert values[int(not narrowed_first)] is not params[0]

    @pytest.mark.parametrize("branch_narrowing", [False, True])
    @pytest.mark.parametrize("restored_first", [False, True])
    def test_restored_writeback_metadata_references_param(self, restored_first, branch_narrowing):
        """Validate final metadata rather than rejecting an intermediate narrowing."""
        if branch_narrowing:
            narrowing = """        if condition:
            then_out = pl.tensor.set_validshape(out, 2, 16)
            narrowed = pl.yield_(then_out)
        else:
            else_out = pl.tensor.set_validshape(out, 2, 16)
            narrowed = pl.yield_(else_out)"""
        else:
            narrowing = "        narrowed = pl.tensor.set_validshape(out, 2, 16)"
        program_source = """
@pl.program
class {name}:
    @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
    def kernel(
        self, out: pl.InOut[pl.Tensor[[4, 16], pl.FP32]], tail: pl.Tensor[[2, 16], pl.FP32],
        condition: pl.Scalar[pl.BOOL]
    ) -> tuple[pl.Tensor[[4, 16], pl.FP32], pl.Tensor[[4, 16], pl.FP32]]:
{narrowing}
        restored = pl.tensor.assemble(narrowed, tail, [2, 0])
        return {returned}
"""
        program_source = program_source.replace("{narrowing}", narrowing)
        before_return = "restored, narrowed" if restored_first else "narrowed, restored"
        expected_return = "out, narrowed" if restored_first else "narrowed, out"
        Before = pl.parse(program_source.replace("{name}", "Before").replace("{returned}", before_return))
        Expected = pl.parse(
            program_source.replace("{name}", "Expected").replace("{returned}", expected_return)
        )
        assert isinstance(Before, ir.Program)
        assert isinstance(Expected, ir.Program)
        After = _run_normalize_direct(Before)
        ir.assert_structural_equal(After, Expected)
        values, params = self._return_values(After, "kernel")
        assert values[int(not restored_first)] is params[0]
        assert values[int(restored_first)] is not params[0]

    def test_sequential_branch_writebacks_share_lineage(self):
        """A linear sequence of merges must not trigger an exponential lineage walk."""
        updates = "\n".join(
            f"""        if condition:
            then_value_{i} = pl.tile.full([1, 16], pl.FP32, 1.0)
            context = pl.store(then_value_{i}, [0, 0], context)
        else:
            else_value_{i} = pl.tile.full([1, 16], pl.FP32, 0.0)
            context = pl.store(else_value_{i}, [0, 0], context)"""
            for i in range(48)
        )
        program_source = f"""
@pl.program
class {{name}}:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(
        self, out: pl.InOut[pl.Tensor[[4, 16], pl.FP32]], condition: pl.Scalar[pl.BOOL]
    ) -> pl.Tensor[[4, 16], pl.FP32]:
        context = out
{updates}
        return {{returned}}
"""
        Before = pl.parse(program_source.replace("{name}", "Before").replace("{returned}", "context"))
        Expected = pl.parse(program_source.replace("{name}", "Expected").replace("{returned}", "out"))
        assert isinstance(Before, ir.Program)
        assert isinstance(Expected, ir.Program)
        Before = passes.convert_to_ssa()(Before)
        Expected = passes.convert_to_ssa()(Expected)
        After = _run_normalize_direct(Before)
        ir.assert_structural_equal(After, Expected)
        values, params = self._return_values(After, "kernel")
        assert values[0] is params[0]

    def test_multi_out_kernel_returns_reference_their_params(self):
        """Each returned tensor IS the param object, not an SSA alias of it."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[64], pl.FP32],
                out_0: pl.Out[pl.Tensor[[64], pl.FP32]],
                out_1: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                t: pl.Tile[[64], pl.FP32] = pl.load(a, [0], [64])
                r0: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out_0)
                r1: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out_1)
                return r0, r1

        values, params = self._return_values(Before, "kernel")
        # Before: the returns are the SSA aliases r0 / r1, not the params.
        before_vars = [v for v in values if isinstance(v, ir.Var)]
        assert len(before_vars) == len(values)
        assert [v.name_hint for v in before_vars] == ["r0", "r1"]

        After = _run_normalize(Before)
        values, params = self._return_values(After, "kernel")
        # After: pointer identity with params_[1] and params_[2].
        assert values[0] is params[1]
        assert values[1] is params[2]

    def test_unreturned_inout_param_does_not_shift_the_mapping(self):
        """An InOut param written in place but not returned must not shift positions.

        The naive "tail-align returns onto the trailing Out/InOut params"
        heuristic mis-binds here (#1573); reading the ReturnStmt cannot.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                inout_t: pl.InOut[pl.Tensor[[64], pl.FP32]],
                out_a: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(inout_t, [0], [64])
                _w: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], inout_t)
                r: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out_a)
                return r

        After = _run_normalize(Before)
        values, params = self._return_values(After, "kernel")
        # The single return binds to out_a (index 1), not to the unreturned
        # InOut param at index 0.
        assert len(values) == 1
        assert values[0] is params[1]

    def test_graph_returns_reference_their_params(self):
        """A Graph is a callee whose returns codegen aliases, so it needs the form too.

        A Graph body is written ``out = core(..., out, ...); return out``, which
        reaches this pass as a ``TupleGetItem`` rebind rather than the param.
        While Graph was excluded from Step A0, orchestration codegen got an
        all-``nullopt`` map, fell back to "return j is the j-th Out/InOut param",
        and silently bound every result to the wrong call-site tensor whenever the
        return order differed from the parameter order (#2601).
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def core(
                self,
                a: pl.Tensor[[64], pl.FP32],
                x: pl.InOut[pl.Tensor[[64], pl.FP32]],
                y: pl.InOut[pl.Tensor[[64], pl.FP32]],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                t: pl.Tile[[64], pl.FP32] = pl.load(a, [0], [64])
                r0: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], x)
                r1: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], y)
                return r0, r1

            @pl.function(type=pl.FunctionType.Graph)
            def layer(
                self,
                a: pl.Tensor[[64], pl.FP32],
                x: pl.InOut[pl.Tensor[[64], pl.FP32]],
                y: pl.InOut[pl.Tensor[[64], pl.FP32]],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                x2, y2 = self.core(a, x, y)
                # Reversed relative to the parameter order.
                return y2, x2

        values, _params = self._return_values(Before, "layer")
        # Before: SSA renames produced by the tuple unpack, not the params.
        assert all(isinstance(v, ir.Var) for v in values)
        assert not any(v is p for v in values for p in _params)

        After = _run_normalize_direct(Before)
        values, params = self._return_values(After, "layer")
        # After: position 0 is param `y` (index 2), position 1 is param `x` (index 1).
        # Getting this backwards is exactly the #2601 mis-binding.
        assert values[0] is params[2]
        assert values[1] is params[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
