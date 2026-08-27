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
from pypto import DataType, ir, passes


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

    def test_if_phi_returns_reordered_and_call_projections_remapped(self):
        """IfStmt phi values retain their Out-param identity through normalization (#2392).

        ``b`` is deliberately declared before ``a`` while the IfStmt yields
        ``(a, b)``.  Both branches preserve that semantic identity, so the
        pass must trace each phi back to its parameter, canonicalize and
        reorder the kernel return to parameter order ``(b, a)``, then remap
        the caller's tuple projections in lockstep.

        Different element types make a crossed mapping visible instead of
        allowing it to hide behind two structurally identical tensor types.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(
                self,
                cond: pl.Scalar[pl.INT32],
                b: pl.Out[pl.Tensor[[16], pl.FP32]],
                a: pl.Out[pl.Tensor[[16], pl.INT32]],
            ) -> tuple[pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.FP32]]:
                if cond > 0:
                    a_phi, b_phi = pl.yield_(a, b)
                else:
                    a_phi, b_phi = pl.yield_(a, b)
                return a_phi, b_phi

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                cond: pl.Scalar[pl.INT32],
                b: pl.Out[pl.Tensor[[16], pl.FP32]],
                a: pl.Out[pl.Tensor[[16], pl.INT32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.INT32]]:
                ret: tuple[pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.FP32]] = self.kernel(cond, b, a)
                a_result: pl.Tensor[[16], pl.INT32] = ret[0]
                b_result: pl.Tensor[[16], pl.FP32] = ret[1]
                return b_result, a_result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(
                self,
                cond: pl.Scalar[pl.INT32],
                b: pl.Out[pl.Tensor[[16], pl.FP32]],
                a: pl.Out[pl.Tensor[[16], pl.INT32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.INT32]]:
                if cond > 0:
                    a_phi, b_phi = pl.yield_(a, b)  # noqa: F841
                else:
                    a_phi, b_phi = pl.yield_(a, b)  # noqa: F841
                return b, a

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                cond: pl.Scalar[pl.INT32],
                b: pl.Out[pl.Tensor[[16], pl.FP32]],
                a: pl.Out[pl.Tensor[[16], pl.INT32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.INT32]]:
                ret: tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.INT32]] = self.kernel(cond, b, a)
                a_result: pl.Tensor[[16], pl.INT32] = ret[1]
                b_result: pl.Tensor[[16], pl.FP32] = ret[0]
                return b_result, a_result

        After = _run_normalize(Before)
        ir.assert_structural_equal(After, Expected)

    def test_if_phi_with_conflicting_branch_roots_stays_unmapped(self):
        """A phi is not attributed when its branches come from different params.

        Each result can refer to either output at runtime, so assigning either
        phi a fixed return-to-param identity would be an unsafe guess.  The
        pass must conservatively leave the function unchanged.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(
                self,
                cond: pl.Scalar[pl.INT32],
                first: pl.Out[pl.Tensor[[16], pl.FP32]],
                second: pl.Out[pl.Tensor[[16], pl.FP32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.FP32]]:
                if cond > 0:
                    result_0, result_1 = pl.yield_(first, second)
                else:
                    result_0, result_1 = pl.yield_(second, first)
                return result_0, result_1

        After = _run_normalize(Before)
        ir.assert_structural_equal(After, Before)

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


class TestNormalizeReturnOrderStepBSafety:
    """Step B rejects uses that cannot be remapped element-wise before rewriting."""

    def test_tuple_alias_is_rejected_before_callee_permutation(self):
        """A whole-tuple alias is rejected before it can change the caller contract."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                b: pl.Out[pl.Tensor[[16], pl.FP32]],
                a: pl.Out[pl.Tensor[[16], pl.INT32]],
            ) -> tuple[pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.FP32]]:
                a_alias: pl.Tensor[[16], pl.INT32] = a
                b_alias: pl.Tensor[[16], pl.FP32] = b
                return a_alias, b_alias

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                b: pl.Out[pl.Tensor[[16], pl.FP32]],
                a: pl.Out[pl.Tensor[[16], pl.INT32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.INT32]]:
                direct: pl.Tuple[pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.FP32]] = self.kernel(b, a)
                alias: pl.Tuple[pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.FP32]] = direct
                a_result: pl.Tensor[[16], pl.INT32] = alias[0]
                b_result: pl.Tensor[[16], pl.FP32] = alias[1]
                return b_result, a_result

        snapshot = pl.parse_program(Before.as_python())
        with pytest.raises(ValueError, match="used as a whole tuple"):
            _run_normalize(Before)
        ir.assert_structural_equal(Before, snapshot)

    def test_incore_caller_is_rejected_with_actionable_hint(self):
        """Direct projections cannot be remapped inside an InCore caller."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def callee(
                self,
                b: pl.Out[pl.Tensor[[16], pl.FP32]],
                a: pl.Out[pl.Tensor[[16], pl.INT32]],
            ) -> tuple[pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.FP32]]:
                a_alias: pl.Tensor[[16], pl.INT32] = a
                b_alias: pl.Tensor[[16], pl.FP32] = b
                return a_alias, b_alias

            @pl.function(type=pl.FunctionType.InCore)
            def caller(
                self,
                b: pl.Out[pl.Tensor[[16], pl.FP32]],
                a: pl.Out[pl.Tensor[[16], pl.INT32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.INT32]]:
                result: pl.Tuple[pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.FP32]] = self.callee(b, a)
                a_result: pl.Tensor[[16], pl.INT32] = result[0]
                b_result: pl.Tensor[[16], pl.FP32] = result[1]
                return b_result, a_result

        snapshot = pl.parse_program(Before.as_python())
        with pytest.raises(ValueError) as exc_info:
            _run_normalize(Before)

        message = str(exc_info.value)
        assert "called from an InCore body" in message
        assert "non-InCore caller" in message
        ir.assert_structural_equal(Before, snapshot)

    def test_if_phi_tuple_flow_is_rejected_before_callee_permutation(self):
        """Yielding a call tuple through an If phi is rejected before reordering."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                b: pl.Out[pl.Tensor[[16], pl.FP32]],
                a: pl.Out[pl.Tensor[[16], pl.INT32]],
            ) -> tuple[pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.FP32]]:
                a_alias: pl.Tensor[[16], pl.INT32] = a
                b_alias: pl.Tensor[[16], pl.FP32] = b
                return a_alias, b_alias

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                cond: pl.Scalar[pl.INT32],
                b: pl.Out[pl.Tensor[[16], pl.FP32]],
                a: pl.Out[pl.Tensor[[16], pl.INT32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.INT32]]:
                if cond > 0:
                    then_result: pl.Tuple[pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.FP32]] = self.kernel(
                        b, a
                    )
                    selected: pl.Tuple[pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.FP32]] = pl.yield_(
                        then_result  # pyright: ignore[reportArgumentType]
                    )
                else:
                    else_result: pl.Tuple[pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.FP32]] = self.kernel(
                        b, a
                    )
                    selected: pl.Tuple[pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.FP32]] = pl.yield_(
                        else_result  # pyright: ignore[reportArgumentType]
                    )
                a_result: pl.Tensor[[16], pl.INT32] = selected[0]
                b_result: pl.Tensor[[16], pl.FP32] = selected[1]
                return b_result, a_result

        snapshot = pl.parse_program(Before.as_python())
        with pytest.raises(ValueError, match="used as a whole tuple"):
            _run_normalize(Before)
        ir.assert_structural_equal(Before, snapshot)

    def test_discarded_call_result_allows_callee_permutation(self):
        """An EvalStmt Call has no caller-visible tuple contract to remap."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                b: pl.Out[pl.Tensor[[16], pl.FP32]],
                a: pl.Out[pl.Tensor[[16], pl.INT32]],
            ) -> tuple[pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.FP32]]:
                a_alias: pl.Tensor[[16], pl.INT32] = a
                b_alias: pl.Tensor[[16], pl.FP32] = b
                return a_alias, b_alias

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                b: pl.Out[pl.Tensor[[16], pl.FP32]],
                a: pl.Out[pl.Tensor[[16], pl.INT32]],
            ):
                self.kernel(b, a)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                b: pl.Out[pl.Tensor[[16], pl.FP32]],
                a: pl.Out[pl.Tensor[[16], pl.INT32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.INT32]]:
                a_alias: pl.Tensor[[16], pl.INT32] = a  # noqa: F841
                b_alias: pl.Tensor[[16], pl.FP32] = b  # noqa: F841
                return b, a

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                b: pl.Out[pl.Tensor[[16], pl.FP32]],
                a: pl.Out[pl.Tensor[[16], pl.INT32]],
            ):
                self.kernel(b, a)

        After = _run_normalize_direct(Before)
        ir.assert_structural_equal(After, Expected)

    def test_discarded_submit_result_allows_callee_permutation(self):
        """A fire-and-forget Submit remains safe when its callee is reordered."""

        span = ir.Span.unknown()
        fp32 = ir.TensorType([16], DataType.FP32)
        int32 = ir.TensorType([16], DataType.INT32)
        task_id = ir.ScalarType(DataType.TASK_ID)

        b = ir.Var("b", fp32, span)
        a = ir.Var("a", int32, span)
        a_alias = ir.Var("a_alias", int32, span)
        kernel = ir.Function(
            "kernel",
            [(b, ir.ParamDirection.Out), (a, ir.ParamDirection.Out)],
            [int32, fp32],
            ir.SeqStmts([ir.AssignStmt(a_alias, a, span), ir.ReturnStmt([a_alias, b], span)], span),
            span,
            ir.FunctionType.InCore,
        )

        main_b = ir.Var("b", fp32, span)
        main_a = ir.Var("a", int32, span)
        submit_type = ir.TupleType([int32, fp32, task_id])
        submit = ir.Submit(ir.GlobalVar("kernel"), [main_b, main_a], [], submit_type, span)
        main = ir.Function(
            "main",
            [(main_b, ir.ParamDirection.Out), (main_a, ir.ParamDirection.Out)],
            [],
            ir.SeqStmts([ir.EvalStmt(submit, span), ir.ReturnStmt([], span)], span),
            span,
            ir.FunctionType.Orchestration,
        )
        Before = ir.Program([kernel, main], "discarded_submit", span)

        # Bare Submit is representable in IR but intentionally has no DSL
        # syntax, so replace the ambient print/parse instrument for this case.
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
            After = _run_normalize_direct(Before)
        funcs = {func.name: func for _, func in After.functions.items()}
        assert list(funcs["kernel"].return_types) == [fp32, int32]
        kernel_body = funcs["kernel"].body
        assert isinstance(kernel_body, ir.SeqStmts)
        kernel_return = kernel_body.stmts[-1]
        assert isinstance(kernel_return, ir.ReturnStmt)
        assert list(kernel_return.value) == list(funcs["kernel"].params)

        main_body = funcs["main"].body
        assert isinstance(main_body, ir.SeqStmts)
        eval_stmt = main_body.stmts[0]
        assert isinstance(eval_stmt, ir.EvalStmt)
        assert isinstance(eval_stmt.expr, ir.Submit)
        result_type = eval_stmt.expr.type
        assert isinstance(result_type, ir.TupleType)
        assert list(result_type.types) == [fp32, int32, task_id]

    @pytest.mark.parametrize("caller_type", [pl.FunctionType.Group, pl.FunctionType.Spmd])
    def test_wrapper_multi_value_tuple_forward_gets_inverse_permutation_adapter(self, caller_type):
        """A wrapper adapts a forwarded tuple even when returning other values beside it."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                b: pl.Out[pl.Tensor[[16], pl.FP32]],
                a: pl.Out[pl.Tensor[[16], pl.INT32]],
            ) -> tuple[pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.FP32]]:
                a_alias: pl.Tensor[[16], pl.INT32] = a
                b_alias: pl.Tensor[[16], pl.FP32] = b
                return a_alias, b_alias

            @pl.function(type=pl.FunctionType.Group)
            def wrapper(
                self,
                b: pl.Out[pl.Tensor[[16], pl.FP32]],
                a: pl.Out[pl.Tensor[[16], pl.INT32]],
            ) -> pl.Tuple[pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.FP32]]:
                packed: pl.Tuple[pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.FP32]] = self.kernel(b, a)
                return packed

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                b: pl.Out[pl.Tensor[[16], pl.FP32]],
                a: pl.Out[pl.Tensor[[16], pl.INT32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.INT32]]:
                wrapped: pl.Tuple[pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.FP32]] = self.wrapper(b, a)
                a_result: pl.Tensor[[16], pl.INT32] = wrapped[0]
                b_result: pl.Tensor[[16], pl.FP32] = wrapped[1]
                return b_result, a_result

        parsed_funcs = {func.name: func for _, func in Before.functions.items()}
        parsed_wrapper = parsed_funcs["wrapper"]
        parsed_body = parsed_wrapper.body
        assert isinstance(parsed_body, ir.SeqStmts)
        packed_assign = parsed_body.stmts[0]
        assert isinstance(packed_assign, ir.AssignStmt)
        parsed_return = parsed_body.stmts[-1]
        assert isinstance(parsed_return, ir.ReturnStmt)
        multi_value_return = ir.ReturnStmt(
            [packed_assign.var, parsed_wrapper.params[0], parsed_wrapper.params[1]], parsed_return.span
        )
        wrapper_body = ir.SeqStmts([*parsed_body.stmts[:-1], multi_value_return], parsed_body.span)
        wrapper_params = list(zip(parsed_wrapper.params, parsed_wrapper.param_directions, strict=True))
        wrapper = ir.Function(
            parsed_wrapper.name,
            wrapper_params,
            [packed_assign.var.type, parsed_wrapper.params[0].type, parsed_wrapper.params[1].type],
            wrapper_body,
            parsed_wrapper.span,
            caller_type,
            attrs=parsed_wrapper.attrs,
            requires_runtime_binding=parsed_wrapper.requires_runtime_binding,
        )
        Before = ir.Program([parsed_funcs["kernel"], wrapper], Before.name, Before.span)

        After = _run_normalize_direct(Before)
        funcs = {func.name: func for _, func in After.functions.items()}

        kernel = funcs["kernel"]
        kernel_returns = list(kernel.return_types)
        assert all(isinstance(ret, ir.TensorType) for ret in kernel_returns)
        assert [ret.dtype for ret in kernel_returns if isinstance(ret, ir.TensorType)] == [
            DataType.FP32,
            DataType.INT32,
        ]
        kernel_body = kernel.body
        assert isinstance(kernel_body, ir.SeqStmts)
        kernel_return = kernel_body.stmts[-1]
        assert isinstance(kernel_return, ir.ReturnStmt)
        assert list(kernel_return.value) == list(kernel.params)

        wrapper = funcs["wrapper"]
        wrapper_body = wrapper.body
        assert isinstance(wrapper_body, ir.SeqStmts)
        packed_assign = wrapper_body.stmts[0]
        assert isinstance(packed_assign, ir.AssignStmt)
        packed_type = packed_assign.var.type
        assert isinstance(packed_type, ir.TupleType)
        packed_items = list(packed_type.types)
        assert all(isinstance(item, ir.TensorType) for item in packed_items)
        assert [item.dtype for item in packed_items if isinstance(item, ir.TensorType)] == [
            DataType.FP32,
            DataType.INT32,
        ]
        wrapper_return = wrapper_body.stmts[-1]
        assert isinstance(wrapper_return, ir.ReturnStmt)
        assert len(wrapper_return.value) == 3
        adapter = wrapper_return.value[0]
        assert isinstance(adapter, ir.MakeTuple)
        elements = list(adapter.elements)
        assert all(isinstance(element, ir.TupleGetItemExpr) for element in elements)
        assert [element.index for element in elements if isinstance(element, ir.TupleGetItemExpr)] == [1, 0]


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
    ``26-normalize_return_order.md`` §"Step B"; pass principle in
    ``.claude/rules/pass-submit-awareness.md``).
    """

    def test_submit_swapped_returns_remapped(self):
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
                    a: pl.Tensor[[16], pl.FP32] = _submit_tmp[1]
                    b: pl.Tensor[[16], pl.FP32] = _submit_tmp[0]
                    tid: pl.Scalar[pl.TASK_ID] = _submit_tmp[2]  # noqa: F841
                return (a, b)

        After = _run_normalize_direct(Before)
        ir.assert_structural_equal(After, Expected)

    def test_submit_distinct_result_types_reordered_before_task_id(self):
        """Submit result types follow the kernel permutation; TASK_ID stays last.

        Unlike the same-dtype case above, swapping these two tensor results
        requires updating both the Submit expression and its result Var from
        ``Tuple[INT32, FP32, TASK_ID]`` to
        ``Tuple[FP32, INT32, TASK_ID]``.  The task ID is submit metadata, not a
        kernel return, so it remains at index 2 while only indices 0 and 1 are
        permuted.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                b: pl.Out[pl.Tensor[[16], pl.FP32]],
                a: pl.Out[pl.Tensor[[16], pl.INT32]],
            ) -> tuple[pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.FP32]]:
                a_alias: pl.Tensor[[16], pl.INT32] = a
                b_alias: pl.Tensor[[16], pl.FP32] = b
                return a_alias, b_alias

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                b: pl.Out[pl.Tensor[[16], pl.FP32]],
                a: pl.Out[pl.Tensor[[16], pl.INT32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.INT32]]:
                with pl.manual_scope():
                    (a_result, b_result), tid = pl.submit(self.kernel, b, a)  # noqa: F841
                return b_result, a_result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                b: pl.Out[pl.Tensor[[16], pl.FP32]],
                a: pl.Out[pl.Tensor[[16], pl.INT32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.INT32]]:
                a_alias: pl.Tensor[[16], pl.INT32] = a  # noqa: F841
                b_alias: pl.Tensor[[16], pl.FP32] = b  # noqa: F841
                return b, a

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                b: pl.Out[pl.Tensor[[16], pl.FP32]],
                a: pl.Out[pl.Tensor[[16], pl.INT32]],
            ) -> tuple[pl.Tensor[[16], pl.FP32], pl.Tensor[[16], pl.INT32]]:
                with pl.manual_scope():
                    _submit_tmp: pl.Tuple[
                        pl.Tensor[[16], pl.FP32],
                        pl.Tensor[[16], pl.INT32],
                        pl.Scalar[pl.TASK_ID],
                    ] = pl.submit(self.kernel, b, a)
                    a_result: pl.Tensor[[16], pl.INT32] = _submit_tmp[1]
                    b_result: pl.Tensor[[16], pl.FP32] = _submit_tmp[0]
                    tid: pl.Scalar[pl.TASK_ID] = _submit_tmp[2]  # noqa: F841
                return b_result, a_result

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
