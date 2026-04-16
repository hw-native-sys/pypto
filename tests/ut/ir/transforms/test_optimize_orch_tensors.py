# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for OptimizeOrchTensors pass.

Each test uses explicit Before (post-ConvertTensorToTileOps tile-level IR)
and Expected (optimized) programs in @pl.program style.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes


class TestIterArgReuse:
    """Pattern 1: Merge Out params into In params via iter-arg feedback."""

    def test_simple_single_return(self):
        """Single-return InCore in ForStmt: Out param merged into InOut."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                x: pl.Tensor[[64], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                acc__tile: pl.Tile[[64], pl.FP32] = pl.load(acc, [0], [64])
                x__tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y__tile: pl.Tile[[64], pl.FP32] = pl.tile.add(acc__tile, x__tile)
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(y__tile, [0], ret0__out)
                return ret0__store

            @pl.function
            def main(
                self, acc0: pl.Tensor[[64], pl.FP32], x: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                for i, (acc,) in pl.range(10, init_values=(acc0,)):
                    ret0__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                    result: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, x, ret0__out)
                    new_acc = pl.yield_(result)
                return new_acc

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                acc: pl.InOut[pl.Tensor[[64], pl.FP32]],
                x: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                acc__tile: pl.Tile[[64], pl.FP32] = pl.load(acc, [0], [64])
                x__tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y__tile: pl.Tile[[64], pl.FP32] = pl.tile.add(acc__tile, x__tile)
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(y__tile, [0], acc)
                return ret0__store

            @pl.function
            def main(
                self, acc0: pl.Tensor[[64], pl.FP32], x: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                for i, (acc,) in pl.range(10, init_values=(acc0,)):
                    result: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, x)
                    new_acc = pl.yield_(result)
                return new_acc

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_multi_return_iter_arg(self):
        """Multi-return InCore with two iter-arg-fed Out params: both merged to InOut."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
                ret1__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                a__tile: pl.Tile[[64], pl.FP32] = pl.load(a, [0], [64])
                b__tile: pl.Tile[[64], pl.FP32] = pl.load(b, [0], [64])
                y__tile: pl.Tile[[64], pl.FP32] = pl.tile.add(a__tile, b__tile)
                z__tile: pl.Tile[[64], pl.FP32] = pl.tile.mul(a__tile, b__tile)
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(y__tile, [0], ret0__out)
                ret1__store: pl.Tensor[[64], pl.FP32] = pl.store(z__tile, [0], ret1__out)
                return ret0__store, ret1__store

            @pl.function
            def main(
                self,
                a0: pl.Tensor[[64], pl.FP32],
                b0: pl.Tensor[[64], pl.FP32],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                for i, (a, b) in pl.range(3, init_values=(a0, b0)):
                    ret0__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                    ret1__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                    result: tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]] = self.main_incore_0(
                        a, b, ret0__out, ret1__out
                    )
                    new_a: pl.Tensor[[64], pl.FP32] = result[0]
                    new_b: pl.Tensor[[64], pl.FP32] = result[1]
                    out_a, out_b = pl.yield_(new_a, new_b)
                return out_a, out_b

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.InOut[pl.Tensor[[64], pl.FP32]],
                b: pl.InOut[pl.Tensor[[64], pl.FP32]],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                a__tile: pl.Tile[[64], pl.FP32] = pl.load(a, [0], [64])
                b__tile: pl.Tile[[64], pl.FP32] = pl.load(b, [0], [64])
                y__tile: pl.Tile[[64], pl.FP32] = pl.tile.add(a__tile, b__tile)
                z__tile: pl.Tile[[64], pl.FP32] = pl.tile.mul(a__tile, b__tile)
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(y__tile, [0], a)
                ret1__store: pl.Tensor[[64], pl.FP32] = pl.store(z__tile, [0], b)
                return ret0__store, ret1__store

            @pl.function
            def main(
                self,
                a0: pl.Tensor[[64], pl.FP32],
                b0: pl.Tensor[[64], pl.FP32],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                for i, (a, b) in pl.range(3, init_values=(a0, b0)):
                    result: tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]] = self.main_incore_0(
                        a, b
                    )
                    new_a: pl.Tensor[[64], pl.FP32] = result[0]
                    new_b: pl.Tensor[[64], pl.FP32] = result[1]
                    out_a, out_b = pl.yield_(new_a, new_b)
                return out_a, out_b

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_multi_return_with_if_branch(self):
        """Multi-return InCore with IfStmt branch: Out params merged to InOut."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
                ret1__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                a__tile: pl.Tile[[64], pl.FP32] = pl.load(a, [0], [64])
                b__tile: pl.Tile[[64], pl.FP32] = pl.load(b, [0], [64])
                if n == 0:
                    ra: pl.Tile[[64], pl.FP32] = a__tile
                    rb: pl.Tile[[64], pl.FP32] = b__tile
                    phi_a, phi_b = pl.yield_(ra, rb)
                else:
                    ra__tile: pl.Tile[[64], pl.FP32] = pl.tile.add(a__tile, b__tile)
                    rb__tile: pl.Tile[[64], pl.FP32] = pl.tile.mul(a__tile, b__tile)
                    phi_a, phi_b = pl.yield_(ra__tile, rb__tile)
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(phi_a, [0], ret0__out)
                ret1__store: pl.Tensor[[64], pl.FP32] = pl.store(phi_b, [0], ret1__out)
                return ret0__store, ret1__store

            @pl.function
            def main(
                self,
                a0: pl.Tensor[[64], pl.FP32],
                b0: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                for i, (a, b) in pl.range(3, init_values=(a0, b0)):
                    ret0__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                    ret1__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                    result: tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]] = self.main_incore_0(
                        a, b, n, ret0__out, ret1__out
                    )
                    new_a: pl.Tensor[[64], pl.FP32] = result[0]
                    new_b: pl.Tensor[[64], pl.FP32] = result[1]
                    out_a, out_b = pl.yield_(new_a, new_b)
                return out_a, out_b

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.InOut[pl.Tensor[[64], pl.FP32]],
                b: pl.InOut[pl.Tensor[[64], pl.FP32]],
                n: pl.Scalar[pl.INT64],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                a__tile: pl.Tile[[64], pl.FP32] = pl.load(a, [0], [64])
                b__tile: pl.Tile[[64], pl.FP32] = pl.load(b, [0], [64])
                if n == 0:
                    ra: pl.Tile[[64], pl.FP32] = a__tile
                    rb: pl.Tile[[64], pl.FP32] = b__tile
                    phi_a, phi_b = pl.yield_(ra, rb)
                else:
                    ra__tile: pl.Tile[[64], pl.FP32] = pl.tile.add(a__tile, b__tile)
                    rb__tile: pl.Tile[[64], pl.FP32] = pl.tile.mul(a__tile, b__tile)
                    phi_a, phi_b = pl.yield_(ra__tile, rb__tile)
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(phi_a, [0], a)
                ret1__store: pl.Tensor[[64], pl.FP32] = pl.store(phi_b, [0], b)
                return ret0__store, ret1__store

            @pl.function
            def main(
                self,
                a0: pl.Tensor[[64], pl.FP32],
                b0: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                for i, (a, b) in pl.range(3, init_values=(a0, b0)):
                    result: tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]] = self.main_incore_0(
                        a, b, n
                    )
                    new_a: pl.Tensor[[64], pl.FP32] = result[0]
                    new_b: pl.Tensor[[64], pl.FP32] = result[1]
                    out_a, out_b = pl.yield_(new_a, new_b)
                return out_a, out_b

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_standalone_call_merges_in_out(self):
        """Standalone InCore call with an iter_arg chain (remainder-kernel shape):
        In + tensor.create Out pair merges to InOut even without an enclosing loop.

        Regression for #928: pl.parallel remainder kernel lost inout accumulation
        because Pattern 1 only matched calls inside an iter-arg loop.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                x: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                acc__tile: pl.Tile[[64], pl.FP32] = pl.load(acc, [0], [64])
                x__tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                for i, (a,) in pl.range(n, init_values=(acc__tile,)):
                    new_a__tile: pl.Tile[[64], pl.FP32] = pl.tile.add(a, x__tile)
                    final = pl.yield_(new_a__tile)
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(final, [0], ret0__out)
                return ret0__store

            @pl.function
            def main(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                x: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
            ) -> pl.Tensor[[64], pl.FP32]:
                ret0__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                result: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, x, n, ret0__out)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                acc: pl.InOut[pl.Tensor[[64], pl.FP32]],
                x: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
            ) -> pl.Tensor[[64], pl.FP32]:
                acc__tile: pl.Tile[[64], pl.FP32] = pl.load(acc, [0], [64])
                x__tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                for i, (a,) in pl.range(n, init_values=(acc__tile,)):
                    new_a__tile: pl.Tile[[64], pl.FP32] = pl.tile.add(a, x__tile)
                    final = pl.yield_(new_a__tile)
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(final, [0], acc)
                return ret0__store

            @pl.function
            def main(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                x: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
            ) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, x, n)
                return result

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_standalone_call_in_arg_reused_not_merged(self):
        """Safety: when the In arg is read again after the call, do NOT merge.

        Merging would clobber the original value the later use expects.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                acc__tile: pl.Tile[[64], pl.FP32] = pl.load(acc, [0], [64])
                for i, (a,) in pl.range(n, init_values=(acc__tile,)):
                    next_a: pl.Tile[[64], pl.FP32] = pl.tile.add(a, a)
                    final = pl.yield_(next_a)
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(final, [0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.InCore)
            def reader(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                acc__tile: pl.Tile[[64], pl.FP32] = pl.load(acc, [0], [64])
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(acc__tile, [0], ret0__out)
                return ret0__store

            @pl.function
            def main(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
            ) -> pl.Tensor[[64], pl.FP32]:
                ret0__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                _unused: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, n, ret0__out)
                ret1__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                result: pl.Tensor[[64], pl.FP32] = self.reader(acc, ret1__out)
                return result

        After = passes.optimize_orch_tensors()(Before)
        # acc is read again by reader — merging main_incore_0's In/Out would
        # corrupt it. Expected: Before is unchanged.
        ir.assert_structural_equal(After, Before)

    def test_standalone_call_unsafe_sibling_blocks_merge(self):
        """When the same callee has multiple standalone call sites, the merge
        must only apply if EVERY site is safe. One unsafe sibling (here: the
        second call reuses `acc` after a later call) must block the rewrite —
        otherwise the rewrite corrupts the sibling's In arg.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                acc__tile: pl.Tile[[64], pl.FP32] = pl.load(acc, [0], [64])
                for i, (a,) in pl.range(n, init_values=(acc__tile,)):
                    next_a: pl.Tile[[64], pl.FP32] = pl.tile.add(a, a)
                    final = pl.yield_(next_a)
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(final, [0], ret0__out)
                return ret0__store

            @pl.function
            def main(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
            ) -> pl.Tensor[[64], pl.FP32]:
                # First call: acc is read again below → unsafe to merge.
                ret_a: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                _first: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, n, ret_a)
                # Second call: uses acc again (this is the "unsafe" sibling).
                ret_b: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                result: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, n, ret_b)
                return result

        After = passes.optimize_orch_tensors()(Before)
        # Any rewrite here would silently corrupt at least one of the two
        # callers, so the pass must leave Before untouched.
        ir.assert_structural_equal(After, Before)

    def test_standalone_call_without_iter_arg_chain_not_merged(self):
        """A standalone call whose callee is a plain load→store (no iter_arg
        chain) is NOT merged: we require semantic evidence (an iter_arg chain)
        that the In/Out were intended to alias.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_copy(
                self,
                src: pl.Tensor[[64], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(src, [0], [64])
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], ret0__out)
                return ret0__store

            @pl.function
            def main(self, src: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                ret0__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                result: pl.Tensor[[64], pl.FP32] = self.kernel_copy(src, ret0__out)
                return result

        After = passes.optimize_orch_tensors()(Before)
        # kernel_copy has no iter_arg loop → no merge expected.
        ir.assert_structural_equal(After, Before)

    def test_no_iter_arg_no_change(self):
        """InCore call not in iter-arg loop: no optimization, Out params remain."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x__tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y__tile: pl.Tile[[64], pl.FP32] = pl.tile.add(x__tile, x__tile)
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(y__tile, [0], ret0__out)
                return ret0__store

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                ret0__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, ret0__out)
                return y

        After = passes.optimize_orch_tensors()(Before)
        # No iter-arg loop → should be unchanged
        ir.assert_structural_equal(After, Before)


class TestLoopHoisting:
    """Loop hoisting (disabled — breaks scope-based alloc_tensors batching)."""

    def test_tensor_create_stays_inside_loop(self):
        """tensor.create stays inside loop to preserve scope-based memory batching."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x__tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y__tile: pl.Tile[[64], pl.FP32] = pl.tile.add(x__tile, x__tile)
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(y__tile, [0], ret0__out)
                return ret0__store

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(10):
                    ret0__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                    y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, ret0__out)
                return y

        After = passes.optimize_orch_tensors()(Before)
        # Loop hoisting disabled: tensor.create should remain unchanged
        ir.assert_structural_equal(After, Before)


class TestAssembleParentStrides:
    """Pattern 2: Attach parent-derived strides to Out params for assemble patterns."""

    def test_out_param_gets_parent_stride(self):
        """When InCore result feeds tensor.assemble in orch, Out param gets parent strides."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                mb: pl.Scalar[pl.INDEX],
                nb: pl.Scalar[pl.INDEX],
                ret0__out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                a__tile: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [mb, nb], [32, 32])
                ret0__store: pl.Tensor[[32, 32], pl.FP32] = pl.store(a__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for mb, (c_iter,) in pl.range(0, 128, 32, init_values=(c,)):
                    for nb, (c_iter2,) in pl.range(0, 128, 32, init_values=(c_iter,)):
                        ret0__out: pl.Tensor[[32, 32], pl.FP32] = pl.create_tensor([32, 32], dtype=pl.FP32)
                        result: pl.Tensor[[32, 32], pl.FP32] = self.main_incore_0(a, mb, nb, ret0__out)
                        c_next: pl.Tensor[[128, 128], pl.FP32] = pl.assemble(c_iter2, result, [mb, nb])
                        c_rv = pl.yield_(c_next)
                    c_rv2 = pl.yield_(c_rv)
                return c_rv2

        # fmt: off
        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                mb: pl.Scalar[pl.INDEX],
                nb: pl.Scalar[pl.INDEX],
                ret0__out: pl.Out[  # noqa: E501
                    pl.Tensor[[32, 32], pl.FP32, pl.TensorView(stride=[128, 1], layout=pl.TensorLayout.ND)]
                ],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                a__tile: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [mb, nb], [32, 32])
                ret0__store: pl.Tensor[  # noqa: E501
                    [32, 32], pl.FP32, pl.TensorView(stride=[128, 1], layout=pl.TensorLayout.ND)
                ] = pl.store(a__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for mb, (c_iter,) in pl.range(0, 128, 32, init_values=(c,)):
                    for nb, (c_iter2,) in pl.range(0, 128, 32, init_values=(c_iter,)):
                        ret0__out: pl.Tensor[[32, 32], pl.FP32] = pl.create_tensor(
                            [32, 32], dtype=pl.FP32
                        )
                        result: pl.Tensor[[32, 32], pl.FP32] = self.main_incore_0(
                            a, mb, nb, ret0__out
                        )
                        c_next: pl.Tensor[[128, 128], pl.FP32] = pl.assemble(
                            c_iter2, result, [mb, nb]
                        )
                        c_rv = pl.yield_(c_next)
                    c_rv2 = pl.yield_(c_rv)
                return c_rv2
        # fmt: on

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_3d_parent_out_param_gets_trailing_stride(self):
        """When parent tensor is 3D and output tile is 2D, only trailing strides are applied."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def proj_incore_0(
                self,
                x: pl.Tensor[[16, 5120], pl.FP32],
                q0: pl.Scalar[pl.INDEX],
                ret0__out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                x__tile: pl.Tile[[16, 64], pl.FP32] = pl.load(x, [0, q0], [16, 64])
                ret0__store: pl.Tensor[[16, 64], pl.FP32] = pl.store(x__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def proj(
                self,
                x: pl.Tensor[[16, 5120], pl.FP32],
                q_proj: pl.Out[pl.Tensor[[4, 128, 5120], pl.FP32]],
            ) -> pl.Tensor[[4, 128, 5120], pl.FP32]:
                for b in pl.range(4):
                    for p0 in pl.range(0, 128, 16):
                        for q0, (q_iter,) in pl.range(0, 5120, 64, init_values=(q_proj,)):
                            ret0__out: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor(
                                [16, 64], dtype=pl.FP32
                            )
                            result: pl.Tensor[[16, 64], pl.FP32] = self.proj_incore_0(x, q0, ret0__out)
                            q_next: pl.Tensor[[4, 128, 5120], pl.FP32] = pl.assemble(
                                q_iter, result, [b, p0, q0]
                            )
                            q_rv = pl.yield_(q_next)
                return q_rv

        # fmt: off
        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def proj_incore_0(
                self,
                x: pl.Tensor[[16, 5120], pl.FP32],
                q0: pl.Scalar[pl.INDEX],
                ret0__out: pl.Out[  # noqa: E501
                    pl.Tensor[[16, 64], pl.FP32, pl.TensorView(stride=[5120, 1], layout=pl.TensorLayout.ND)]
                ],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                x__tile: pl.Tile[[16, 64], pl.FP32] = pl.load(x, [0, q0], [16, 64])
                ret0__store: pl.Tensor[  # noqa: E501
                    [16, 64], pl.FP32, pl.TensorView(stride=[5120, 1], layout=pl.TensorLayout.ND)
                ] = pl.store(x__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def proj(
                self,
                x: pl.Tensor[[16, 5120], pl.FP32],
                q_proj: pl.Out[pl.Tensor[[4, 128, 5120], pl.FP32]],
            ) -> pl.Tensor[[4, 128, 5120], pl.FP32]:
                for b in pl.range(4):
                    for p0 in pl.range(0, 128, 16):
                        for q0, (q_iter,) in pl.range(0, 5120, 64, init_values=(q_proj,)):
                            ret0__out: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor(
                                [16, 64], dtype=pl.FP32
                            )
                            result: pl.Tensor[[16, 64], pl.FP32] = self.proj_incore_0(
                                x, q0, ret0__out
                            )
                            q_next: pl.Tensor[[4, 128, 5120], pl.FP32] = pl.assemble(
                                q_iter, result, [b, p0, q0]
                            )
                            q_rv = pl.yield_(q_next)
                return q_rv
        # fmt: on

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Expected)


class TestAssembleLoopRewrite:
    """Pattern 3: Rewrite tile.assemble loops to tile.store loops."""

    def test_assemble_loop_to_store_loop(self):
        """ForStmt with tile.assemble rewritten to tile.store with Out param as iter-arg init."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[1, 32], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                buf__tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.create(
                    [1, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                for i, (acc,) in pl.range(2, init_values=(buf__tile,)):
                    off: pl.Scalar[pl.INDEX] = i * 32
                    chunk__tile: pl.Tile[[1, 32], pl.FP32] = pl.load(x, [0, 0], [1, 32])
                    acc_next__tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.assemble(acc, chunk__tile, [0, off])
                    result: pl.Tile[[1, 64], pl.FP32] = pl.yield_(acc_next__tile)
                ret0__store: pl.Tensor[[1, 64], pl.FP32] = pl.store(result, [0, 0], ret0__out)
                return ret0__store

            @pl.function
            def main(self, x: pl.Tensor[[1, 32], pl.FP32]) -> pl.Tensor[[1, 64], pl.FP32]:
                ret0__out: pl.Tensor[[1, 64], pl.FP32] = pl.create_tensor([1, 64], dtype=pl.FP32)
                y: pl.Tensor[[1, 64], pl.FP32] = self.main_incore_0(x, ret0__out)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[1, 32], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                for i, (acc,) in pl.range(2, init_values=(ret0__out,)):
                    off: pl.Scalar[pl.INDEX] = i * 32
                    chunk__tile: pl.Tile[[1, 32], pl.FP32] = pl.load(x, [0, 0], [1, 32])
                    acc_next: pl.Tensor[[1, 64], pl.FP32] = pl.store(chunk__tile, [0, off], acc)
                    result = pl.yield_(acc_next)
                return result

            @pl.function
            def main(self, x: pl.Tensor[[1, 32], pl.FP32]) -> pl.Tensor[[1, 64], pl.FP32]:
                ret0__out: pl.Tensor[[1, 64], pl.FP32] = pl.create_tensor([1, 64], dtype=pl.FP32)
                y: pl.Tensor[[1, 64], pl.FP32] = self.main_incore_0(x, ret0__out)
                return y

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Expected)


class TestSliceInputStrides:
    """Pattern 4: Attach parent-derived strides to In params for slice patterns."""

    def test_in_param_gets_parent_stride_from_slice(self):
        """When orch slices a 2D parent and passes result to InCore In param, param gets parent strides."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[32, 32], pl.FP32],
                mb: pl.Scalar[pl.INDEX],
                nb: pl.Scalar[pl.INDEX],
                ret0__out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                a__tile: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                ret0__store: pl.Tensor[[32, 32], pl.FP32] = pl.store(a__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for mb in pl.range(0, 128, 32):
                    for nb, (c_iter,) in pl.range(0, 128, 32, init_values=(c,)):
                        chunk: pl.Tensor[[32, 32], pl.FP32] = pl.slice(data, [32, 32], [mb, nb])
                        ret0__out: pl.Tensor[[32, 32], pl.FP32] = pl.create_tensor([32, 32], dtype=pl.FP32)
                        result: pl.Tensor[[32, 32], pl.FP32] = self.main_incore_0(chunk, mb, nb, ret0__out)
                        c_next: pl.Tensor[[128, 128], pl.FP32] = pl.assemble(c_iter, result, [mb, nb])
                        c_rv = pl.yield_(c_next)
                return c_rv

        # fmt: off
        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[  # noqa: E501
                    [32, 32], pl.FP32, pl.TensorView(stride=[128, 1], layout=pl.TensorLayout.ND)
                ],
                mb: pl.Scalar[pl.INDEX],
                nb: pl.Scalar[pl.INDEX],
                ret0__out: pl.Out[  # noqa: E501
                    pl.Tensor[[32, 32], pl.FP32, pl.TensorView(stride=[128, 1], layout=pl.TensorLayout.ND)]
                ],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                a__tile: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                ret0__store: pl.Tensor[  # noqa: E501
                    [32, 32], pl.FP32, pl.TensorView(stride=[128, 1], layout=pl.TensorLayout.ND)
                ] = pl.store(a__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for mb in pl.range(0, 128, 32):
                    for nb, (c_iter,) in pl.range(0, 128, 32, init_values=(c,)):
                        chunk: pl.Tensor[[32, 32], pl.FP32] = pl.slice(data, [32, 32], [mb, nb])
                        ret0__out: pl.Tensor[[32, 32], pl.FP32] = pl.create_tensor(
                            [32, 32], dtype=pl.FP32
                        )
                        result: pl.Tensor[[32, 32], pl.FP32] = self.main_incore_0(
                            chunk, mb, nb, ret0__out
                        )
                        c_next: pl.Tensor[[128, 128], pl.FP32] = pl.assemble(
                            c_iter, result, [mb, nb]
                        )
                        c_rv = pl.yield_(c_next)
                return c_rv
        # fmt: on

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_3d_parent_in_param_gets_trailing_stride(self):
        """When parent tensor is 3D and input slice is 2D, only trailing strides are applied."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def proj_incore_0(
                self,
                x: pl.Tensor[[16, 64], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                x__tile: pl.Tile[[16, 64], pl.FP32] = pl.load(x, [0, 0], [16, 64])
                ret0__store: pl.Tensor[[16, 64], pl.FP32] = pl.store(x__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def proj(
                self,
                data: pl.Tensor[[4, 128, 5120], pl.FP32],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                chunk: pl.Tensor[[16, 64], pl.FP32] = pl.slice(data, [16, 64], [0, 0, 0])
                ret0__out: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
                result: pl.Tensor[[16, 64], pl.FP32] = self.proj_incore_0(chunk, ret0__out)
                return result

        # fmt: off
        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def proj_incore_0(
                self,
                x: pl.Tensor[  # noqa: E501
                    [16, 64], pl.FP32, pl.TensorView(stride=[5120, 1], layout=pl.TensorLayout.ND)
                ],
                ret0__out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                x__tile: pl.Tile[[16, 64], pl.FP32] = pl.load(x, [0, 0], [16, 64])
                ret0__store: pl.Tensor[[16, 64], pl.FP32] = pl.store(
                    x__tile, [0, 0], ret0__out
                )
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def proj(
                self,
                data: pl.Tensor[[4, 128, 5120], pl.FP32],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                chunk: pl.Tensor[[16, 64], pl.FP32] = pl.slice(data, [16, 64], [0, 0, 0])
                ret0__out: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor(
                    [16, 64], dtype=pl.FP32
                )
                result: pl.Tensor[[16, 64], pl.FP32] = self.proj_incore_0(chunk, ret0__out)
                return result
        # fmt: on

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_multiple_sliced_in_params(self):
        """Multiple In params from different parents each get correct strides."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def gemm_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.FP32],
                b: pl.Tensor[[128, 64], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                a__tile: pl.Tile[[16, 128], pl.FP32] = pl.load(a, [0, 0], [16, 128])
                b__tile: pl.Tile[[128, 64], pl.FP32] = pl.load(b, [0, 0], [128, 64])
                c__tile: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(a__tile, b__tile)
                ret0__store: pl.Tensor[[16, 64], pl.FP32] = pl.store(c__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def gemm(
                self,
                attn_out: pl.Tensor[[16, 8192], pl.FP32],
                wo: pl.Tensor[[8192, 8192], pl.FP32],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                a_chunk: pl.Tensor[[16, 128], pl.FP32] = pl.slice(attn_out, [16, 128], [0, 0])
                w_chunk: pl.Tensor[[128, 64], pl.FP32] = pl.slice(wo, [128, 64], [0, 0])
                ret0__out: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
                result: pl.Tensor[[16, 64], pl.FP32] = self.gemm_incore_0(a_chunk, w_chunk, ret0__out)
                return result

        # fmt: off
        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def gemm_incore_0(
                self,
                a: pl.Tensor[  # noqa: E501
                    [16, 128], pl.FP32, pl.TensorView(stride=[8192, 1], layout=pl.TensorLayout.ND)
                ],
                b: pl.Tensor[  # noqa: E501
                    [128, 64], pl.FP32, pl.TensorView(stride=[8192, 1], layout=pl.TensorLayout.ND)
                ],
                ret0__out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                a__tile: pl.Tile[[16, 128], pl.FP32] = pl.load(a, [0, 0], [16, 128])
                b__tile: pl.Tile[[128, 64], pl.FP32] = pl.load(b, [0, 0], [128, 64])
                c__tile: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(a__tile, b__tile)
                ret0__store: pl.Tensor[[16, 64], pl.FP32] = pl.store(
                    c__tile, [0, 0], ret0__out
                )
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def gemm(
                self,
                attn_out: pl.Tensor[[16, 8192], pl.FP32],
                wo: pl.Tensor[[8192, 8192], pl.FP32],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                a_chunk: pl.Tensor[[16, 128], pl.FP32] = pl.slice(
                    attn_out, [16, 128], [0, 0]
                )
                w_chunk: pl.Tensor[[128, 64], pl.FP32] = pl.slice(wo, [128, 64], [0, 0])
                ret0__out: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor(
                    [16, 64], dtype=pl.FP32
                )
                result: pl.Tensor[[16, 64], pl.FP32] = self.gemm_incore_0(
                    a_chunk, w_chunk, ret0__out
                )
                return result
        # fmt: on

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_non_sliced_in_param_unchanged(self):
        """In params that are not from tensor.slice remain unchanged."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 32], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                x__tile: pl.Tile[[32, 32], pl.FP32] = pl.load(x, [0, 0], [32, 32])
                ret0__store: pl.Tensor[[32, 32], pl.FP32] = pl.store(x__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[32, 32], pl.FP32],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                ret0__out: pl.Tensor[[32, 32], pl.FP32] = pl.create_tensor([32, 32], dtype=pl.FP32)
                result: pl.Tensor[[32, 32], pl.FP32] = self.main_incore_0(data, ret0__out)
                return result

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Before)


class TestEdgeCases:
    """Edge cases: pass should not modify programs that don't match any pattern."""

    def test_no_incore_functions(self):
        """Programs with no InCore functions pass through unchanged."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
