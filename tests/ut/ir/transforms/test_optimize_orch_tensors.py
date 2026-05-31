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
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def _run_to_optimize_orch_tensors(program):
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    result = program
    for pass_name, pass_obj in zip(pm.pass_names, pm.passes, strict=True):
        result = pass_obj(result)
        if pass_name == "OptimizeOrchTensors":
            return result
    raise AssertionError("Default pipeline did not run OptimizeOrchTensors")


def _get_function(program, name: str):
    func = program.get_function(name)
    assert func is not None
    return func


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
                acc__tile = pl.load(acc, [0], [64])
                x__tile = pl.load(x, [0], [64])
                y__tile = pl.tile.add(acc__tile, x__tile)
                ret0__store = pl.store(y__tile, [0], acc)
                return ret0__store

            @pl.function
            def main(
                self, acc0: pl.Tensor[[64], pl.FP32], x: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                for i, (acc,) in pl.range(10, init_values=(acc0,)):
                    result = self.main_incore_0(acc, x)
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
                a__tile = pl.load(a, [0], [64])
                b__tile = pl.load(b, [0], [64])
                y__tile = pl.tile.add(a__tile, b__tile)
                z__tile = pl.tile.mul(a__tile, b__tile)
                ret0__store = pl.store(y__tile, [0], a)
                ret1__store = pl.store(z__tile, [0], b)
                return ret0__store, ret1__store

            @pl.function
            def main(
                self,
                a0: pl.Tensor[[64], pl.FP32],
                b0: pl.Tensor[[64], pl.FP32],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                for i, (a, b) in pl.range(3, init_values=(a0, b0)):
                    result = self.main_incore_0(a, b)
                    new_a = result[0]
                    new_b = result[1]
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
                a__tile = pl.load(a, [0], [64])
                b__tile = pl.load(b, [0], [64])
                if n == 0:
                    ra: pl.Tile[[64], pl.FP32] = a__tile
                    rb: pl.Tile[[64], pl.FP32] = b__tile
                    phi_a, phi_b = pl.yield_(ra, rb)
                else:
                    ra__tile: pl.Tile[[64], pl.FP32] = pl.tile.add(a__tile, b__tile)
                    rb__tile: pl.Tile[[64], pl.FP32] = pl.tile.mul(a__tile, b__tile)
                    phi_a, phi_b = pl.yield_(ra__tile, rb__tile)
                ret0__store = pl.store(phi_a, [0], a)
                ret1__store = pl.store(phi_b, [0], b)
                return ret0__store, ret1__store

            @pl.function
            def main(
                self,
                a0: pl.Tensor[[64], pl.FP32],
                b0: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                for i, (a, b) in pl.range(3, init_values=(a0, b0)):
                    result = self.main_incore_0(a, b, n)
                    new_a = result[0]
                    new_b = result[1]
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
                acc__tile = pl.load(acc, [0], [64])
                x__tile = pl.load(x, [0], [64])
                for i, (a,) in pl.range(n, init_values=(acc__tile,)):
                    new_a__tile = pl.tile.add(a, x__tile)
                    final = pl.yield_(new_a__tile)
                ret0__store = pl.store(final, [0], acc)
                return ret0__store

            @pl.function
            def main(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                x: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
            ) -> pl.Tensor[[64], pl.FP32]:
                result = self.main_incore_0(acc, x, n)
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
                a__tile = pl.load(a, [mb, nb], [32, 32])
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
                        ret0__out = pl.create_tensor(
                            [32, 32], dtype=pl.FP32
                        )
                        result = self.main_incore_0(
                            a, mb, nb, ret0__out
                        )
                        c_next = pl.assemble(
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
                x__tile = pl.load(x, [0, q0], [16, 64])
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
                            ret0__out = pl.create_tensor(
                                [16, 64], dtype=pl.FP32
                            )
                            result = self.proj_incore_0(
                                x, q0, ret0__out
                            )
                            q_next = pl.assemble(
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
                    chunk__tile = pl.load(x, [0, 0], [1, 32])
                    acc_next = pl.store(chunk__tile, [0, off], acc)
                    result = pl.yield_(acc_next)
                return result

            @pl.function
            def main(self, x: pl.Tensor[[1, 32], pl.FP32]) -> pl.Tensor[[1, 64], pl.FP32]:
                ret0__out = pl.create_tensor([1, 64], dtype=pl.FP32)
                y = self.main_incore_0(x, ret0__out)
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
                a__tile = pl.load(a, [0, 0], [32, 32])
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
                        chunk = pl.slice(data, [32, 32], [mb, nb])
                        ret0__out = pl.create_tensor(
                            [32, 32], dtype=pl.FP32
                        )
                        result = self.main_incore_0(
                            chunk, mb, nb, ret0__out
                        )
                        c_next = pl.assemble(
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
                x__tile = pl.load(x, [0, 0], [16, 64])
                ret0__store = pl.store(
                    x__tile, [0, 0], ret0__out
                )
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def proj(
                self,
                data: pl.Tensor[[4, 128, 5120], pl.FP32],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                chunk = pl.slice(data, [16, 64], [0, 0, 0])
                ret0__out = pl.create_tensor(
                    [16, 64], dtype=pl.FP32
                )
                result = self.proj_incore_0(chunk, ret0__out)
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
                a__tile = pl.load(a, [0, 0], [16, 128])
                b__tile = pl.load(b, [0, 0], [128, 64])
                c__tile = pl.tile.matmul(a__tile, b__tile)
                ret0__store = pl.store(
                    c__tile, [0, 0], ret0__out
                )
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def gemm(
                self,
                attn_out: pl.Tensor[[16, 8192], pl.FP32],
                wo: pl.Tensor[[8192, 8192], pl.FP32],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                a_chunk = pl.slice(
                    attn_out, [16, 128], [0, 0]
                )
                w_chunk = pl.slice(wo, [128, 64], [0, 0])
                ret0__out = pl.create_tensor(
                    [16, 64], dtype=pl.FP32
                )
                result = self.gemm_incore_0(
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

        After = _run_to_optimize_orch_tensors(Before)
        ir.assert_structural_equal(After, Before)


class TestOutWindowExternalizer:
    """Pattern 5: static out-window externalization."""

    def test_direct_out_call_rewrites_to_windowed_clone(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_stripe(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                bias: pl.Scalar[pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                result: pl.Tile[[64, 64], pl.FP32] = pl.add(tile, bias)
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(result, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                row: pl.Scalar[pl.INDEX] = 64
                out_next: pl.Tensor[[256, 64], pl.FP32] = self.kernel_stripe(data, row, 1.0, out)
                return out_next

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_stripe(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                bias: pl.Scalar[pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    data, [row_offset, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                result: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(tile, bias)
                ret = pl.tile.store(result, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def kernel_stripe__windowed(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                bias: pl.Scalar[pl.FP32],
                out: pl.Out[
                    pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)]
                ],
            ) -> pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)]:
                tile: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    data, [row_offset, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                result: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(tile, bias)
                ret: pl.Tensor[
                    [64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)
                ] = pl.tile.store(result, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                out__window = pl.tensor.slice(out, [64, 64], [64, 0])
                out_next__windowed: pl.Tensor[
                    [64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)
                ] = self.kernel_stripe__windowed(data, 64, 1.0, out__window)
                out_next = pl.tensor.assemble(out, out_next__windowed, [64, 0])
                return out_next

        After = _run_to_optimize_orch_tensors(Before)
        ir.assert_structural_equal(After, Expected)

    def test_phase_fence_auto_nested_loop_shape_rewrites(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_stripe(
                self,
                data: pl.Tensor[[1024, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                bias: pl.Scalar[pl.FP32],
                out: pl.Out[pl.Tensor[[1024, 64], pl.FP32]],
            ) -> pl.Tensor[[1024, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(
                    data, [row_offset, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
                )
                result: pl.Tile[[64, 64], pl.FP32] = pl.tile.adds(tile, bias)
                ret: pl.Tensor[[1024, 64], pl.FP32] = pl.tile.store(result, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[1024, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[1024, 64], pl.FP32]],
            ) -> pl.Tensor[[1024, 64], pl.FP32]:
                for phase, (out_phase,) in pl.range(4, init_values=(out,)):
                    for branch, (out_branch,) in pl.parallel(4, init_values=(out_phase,)):
                        row: pl.Scalar[pl.INDEX] = (phase * 4 + branch) * 64
                        out_next: pl.Tensor[[1024, 64], pl.FP32] = self.kernel_stripe(
                            data, row, 1.0, out_branch
                        )
                        out_branch_next = pl.yield_(out_next)
                    out_phase_next = pl.yield_(out_branch_next)
                return out_phase_next

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_stripe(
                self,
                data: pl.Tensor[[1024, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                bias: pl.Scalar[pl.FP32],
                out: pl.Out[pl.Tensor[[1024, 64], pl.FP32]],
            ) -> pl.Tensor[[1024, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    data, [row_offset, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                result: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(tile, bias)
                ret = pl.tile.store(result, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def kernel_stripe__windowed(
                self,
                data: pl.Tensor[[1024, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                bias: pl.Scalar[pl.FP32],
                out: pl.Out[
                    pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)]
                ],
            ) -> pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)]:
                tile: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    data, [row_offset, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                result: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(tile, bias)
                ret: pl.Tensor[
                    [64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)
                ] = pl.tile.store(result, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[1024, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[1024, 64], pl.FP32]],
            ) -> pl.Tensor[[1024, 64], pl.FP32]:
                for phase, (out_phase,) in pl.range(4, init_values=(out,)):
                    for branch, (out_branch,) in pl.parallel(4, init_values=(out_phase,)):
                        row: pl.Scalar[pl.INDEX] = (phase * 4 + branch) * 64
                        out_branch__window = pl.tensor.slice(out_branch, [64, 64], [row, 0])
                        out_next__windowed: pl.Tensor[
                            [64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)
                        ] = self.kernel_stripe__windowed(data, row, 1.0, out_branch__window)
                        out_next = pl.tensor.assemble(out_branch, out_next__windowed, [row, 0])
                        out_branch_next = pl.yield_(out_next)
                    out_phase_next = pl.yield_(out_branch_next)
                return out_phase_next

        After = _run_to_optimize_orch_tensors(Before)
        ir.assert_structural_equal(After, Expected)

    def test_multi_out_final_store_rewrites_both_outputs(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kv_stripe(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                k_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
                v_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32]]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                k_next: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile, [row_offset, 0], k_out)
                v_tile: pl.Tile[[64, 64], pl.FP32] = pl.add(tile, tile)
                v_next: pl.Tensor[[256, 64], pl.FP32] = pl.store(v_tile, [row_offset, 0], v_out)
                return k_next, v_next

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                k_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
                v_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32]]:
                row: pl.Scalar[pl.INDEX] = 64
                result: tuple[pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32]] = self.kv_stripe(
                    data, row, k_out, v_out
                )
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kv_stripe(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                k_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
                v_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32]]:
                tile: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    data, [row_offset, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                k_next = pl.tile.store(tile, [row_offset, 0], k_out)
                v_tile: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(tile, tile)
                v_next = pl.tile.store(v_tile, [row_offset, 0], v_out)
                return k_next, v_next

            @pl.function(type=pl.FunctionType.InCore)
            def kv_stripe__windowed(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                k_out: pl.Out[
                    pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)]
                ],
                v_out: pl.Out[
                    pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)]
                ],
            ) -> tuple[
                pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)],
                pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)],
            ]:
                tile: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    data, [row_offset, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                k_next: pl.Tensor[
                    [64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)
                ] = pl.tile.store(tile, [0, 0], k_out)
                v_tile: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(tile, tile)
                v_next: pl.Tensor[
                    [64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)
                ] = pl.tile.store(v_tile, [0, 0], v_out)
                return k_next, v_next

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                k_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
                v_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32]]:
                k_out__window = pl.tensor.slice(k_out, [64, 64], [64, 0])
                v_out__window = pl.tensor.slice(v_out, [64, 64], [64, 0])
                result__windowed: pl.Tuple[
                    pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)],
                    pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)],
                ] = self.kv_stripe__windowed(data, 64, k_out__window, v_out__window)
                result__windowed_0: pl.Tensor[
                    [64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)
                ] = result__windowed[0]
                result__assembled_0 = pl.tensor.assemble(k_out, result__windowed_0, [64, 0])
                result__windowed_1: pl.Tensor[
                    [64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)
                ] = result__windowed[1]
                result__assembled_1 = pl.tensor.assemble(v_out, result__windowed_1, [64, 0])
                result: pl.Tuple[pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32]] = [
                    result__assembled_0,
                    result__assembled_1,
                ]
                return result

        After = _run_to_optimize_orch_tensors(Before)
        ir.assert_structural_equal(After, Expected)

    def test_return_reordered_multi_out_later_parent_read_stays_baseline(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kv_stripe(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                k_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
                v_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32]]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                k_next: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile, [row_offset, 0], k_out)
                v_next: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile, [row_offset, 0], v_out)
                return v_next, k_next

            @pl.function(type=pl.FunctionType.InCore)
            def consume_full(
                self,
                k: pl.Tensor[[256, 64], pl.FP32],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                return k

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                k_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
                v_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                row: pl.Scalar[pl.INDEX] = 64
                result: tuple[pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32]] = self.kv_stripe(
                    data, row, k_out, v_out
                )
                k_next: pl.Tensor[[256, 64], pl.FP32] = result[1]
                return self.consume_full(k_next)

        After = _run_to_optimize_orch_tensors(Before)

        assert After.get_function("kv_stripe__windowed") is None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "pl.tensor.slice(k_out" not in printed_main
        assert "pl.tensor.slice(v_out" not in printed_main

    def test_tensor_full_root_later_parent_read_stays_baseline(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_stripe(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def consume_full(
                self,
                out: pl.Tensor[[256, 64], pl.FP32],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                out: pl.Tensor[[256, 64], pl.FP32] = pl.full([256, 64], dtype=pl.FP32, value=0.0)
                row: pl.Scalar[pl.INDEX] = 64
                out_next: pl.Tensor[[256, 64], pl.FP32] = self.kernel_stripe(data, row, out)
                return self.consume_full(out_next)

        After = _run_to_optimize_orch_tensors(Before)

        assert After.get_function("kernel_stripe__windowed") is None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "pl.tensor.slice(out" not in printed_main

    def test_loop_returned_output_later_parent_read_stays_baseline(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_rows(
                self,
                out: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
                row_base: pl.Scalar[pl.INDEX],
                data: pl.Tensor[[128, 64], pl.FP32],
            ) -> pl.Tensor[[128, 64], pl.FP32]:
                for i, (out_iter,) in pl.range(2, init_values=(out,)):
                    row: pl.Scalar[pl.INDEX] = row_base + i * 32
                    tile: pl.Tile[[32, 64], pl.FP32] = pl.tile.load(data, [row, 0], [32, 64], [32, 64])
                    out_next: pl.Tensor[[128, 64], pl.FP32] = pl.tile.store(tile, [row, 0], out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

            @pl.function(type=pl.FunctionType.InCore)
            def consume_full(
                self,
                out: pl.Tensor[[128, 64], pl.FP32],
            ) -> pl.Tensor[[128, 64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[128, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
            ) -> pl.Tensor[[128, 64], pl.FP32]:
                row_base: pl.Scalar[pl.INDEX] = 32
                out_next: pl.Tensor[[128, 64], pl.FP32] = self.kernel_rows(out, row_base, data)
                return self.consume_full(out_next)

        After = _run_to_optimize_orch_tensors(Before)

        assert After.get_function("kernel_rows__windowed") is None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "pl.tensor.slice(out" not in printed_main

    def test_multi_out_final_store_all_or_nothing_stays_baseline(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kv_stripe(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                k_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
                v_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32]]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                k_next: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile, [row_offset, 0], k_out)
                passthrough: pl.Tensor[[256, 64], pl.FP32] = v_out
                pl.store(tile, [row_offset, 0], v_out)
                return k_next, passthrough

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                k_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
                v_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32]]:
                row: pl.Scalar[pl.INDEX] = 64
                result: tuple[pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32]] = self.kv_stripe(
                    data, row, k_out, v_out
                )
                return result

        After = _run_to_optimize_orch_tensors(Before)

        assert After.get_function("kv_stripe__windowed") is None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "pl.tensor.slice(k_out" not in printed_main
        assert "pl.tensor.slice(v_out" not in printed_main

    def test_callee_local_kv_loop_without_callsite_window_stays_baseline(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kv_proj(
                self,
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
                wv: pl.Tensor[[512, 512], pl.BF16],
                k_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                v_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]]:
                b0: pl.Scalar[pl.INDEX] = 0
                layer_hidden_base: pl.Scalar[pl.INDEX] = 0
                for ob_chunk in pl.range(0, 8, 4):
                    for ob in pl.range(ob_chunk, ob_chunk + 4):
                        kv0: pl.Scalar[pl.INDEX] = ob * 64
                        tile_a: pl.Tensor[[16, 128], pl.BF16] = pl.slice(normed_tile, [16, 128], [0, 0])
                        tile_wk: pl.Tensor[[128, 64], pl.BF16] = pl.slice(
                            wk, [128, 64], [layer_hidden_base, kv0]
                        )
                        k_acc: pl.Tensor[[16, 64], pl.FP32] = pl.matmul(tile_a, tile_wk, out_dtype=pl.FP32)
                        for kb in pl.range(1, 4):
                            k0: pl.Scalar[pl.INDEX] = kb * 128
                            tile_a_i: pl.Tensor[[16, 128], pl.BF16] = pl.slice(
                                normed_tile, [16, 128], [0, k0]
                            )
                            tile_wk_i: pl.Tensor[[128, 64], pl.BF16] = pl.slice(
                                wk, [128, 64], [layer_hidden_base + k0, kv0]
                            )
                            k_acc = pl.matmul_acc(k_acc, tile_a_i, tile_wk_i)
                        k_proj = pl.assemble(k_proj, k_acc, [b0, kv0])

                        tile_a = pl.slice(normed_tile, [16, 128], [0, 0])
                        tile_wv: pl.Tensor[[128, 64], pl.BF16] = pl.slice(
                            wv, [128, 64], [layer_hidden_base, kv0]
                        )
                        v_acc: pl.Tensor[[16, 64], pl.FP32] = pl.matmul(tile_a, tile_wv, out_dtype=pl.FP32)
                        for kb in pl.range(1, 4):
                            k0 = kb * 128
                            tile_a_i = pl.slice(normed_tile, [16, 128], [0, k0])
                            tile_wv_i: pl.Tensor[[128, 64], pl.BF16] = pl.slice(
                                wv, [128, 64], [layer_hidden_base + k0, kv0]
                            )
                            v_acc = pl.matmul_acc(v_acc, tile_a_i, tile_wv_i)
                        v_proj = pl.assemble(v_proj, v_acc, [b0, kv0])
                return k_proj, v_proj

            @pl.function
            def main(
                self,
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
                wv: pl.Tensor[[512, 512], pl.BF16],
                k_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                v_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]]:
                result: tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]] = self.kv_proj(
                    normed_tile, wk, wv, k_proj, v_proj
                )
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kv_proj(
                self,
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
                wv: pl.Tensor[[512, 512], pl.BF16],
                k_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                v_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]]:
                for ob_chunk, (k_proj_iter, v_proj_iter) in pl.range(0, 8, 4, init_values=(k_proj, v_proj)):
                    for ob, (k_proj_iter2, v_proj_iter2) in pl.range(
                        ob_chunk, ob_chunk + 4, init_values=(k_proj_iter, v_proj_iter)
                    ):
                        kv0: pl.Scalar[pl.INDEX] = ob * 64
                        tile_a: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                            normed_tile,
                            [0, 0],
                            [16, 128],
                            [16, 128],
                            target_memory=pl.Mem.Mat,
                            transpose=False,
                        )
                        tile_wk: pl.Tile[[128, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                            wk, [0, kv0], [128, 64], [128, 64], target_memory=pl.Mem.Mat, transpose=False
                        )
                        k_acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(tile_a, tile_wk)
                        for kb, (k_acc_iter,) in pl.range(1, 4, init_values=(k_acc,)):
                            k0: pl.Scalar[pl.INDEX] = kb * 128
                            tile_a_i: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                                normed_tile,
                                [0, k0],
                                [16, 128],
                                [16, 128],
                                target_memory=pl.Mem.Mat,
                                transpose=False,
                            )
                            tile_wk_i: pl.Tile[[128, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                                wk, [k0, kv0], [128, 64], [128, 64], target_memory=pl.Mem.Mat, transpose=False
                            )
                            k_acc_next: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(
                                k_acc_iter, tile_a_i, tile_wk_i
                            )
                            k_acc_rv: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(k_acc_next)
                        k_proj_tile = pl.tile.store(k_acc_rv, [0, kv0], k_proj_iter2)
                        tile_a_2: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                            normed_tile,
                            [0, 0],
                            [16, 128],
                            [16, 128],
                            target_memory=pl.Mem.Mat,
                            transpose=False,
                        )
                        tile_wv: pl.Tile[[128, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                            wv, [0, kv0], [128, 64], [128, 64], target_memory=pl.Mem.Mat, transpose=False
                        )
                        v_acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(tile_a_2, tile_wv)
                        for kb2, (v_acc_iter,) in pl.range(1, 4, init_values=(v_acc,)):
                            k0_2: pl.Scalar[pl.INDEX] = kb2 * 128
                            tile_a_i_2: pl.Tile[[16, 128], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                                normed_tile,
                                [0, k0_2],
                                [16, 128],
                                [16, 128],
                                target_memory=pl.Mem.Mat,
                                transpose=False,
                            )
                            tile_wv_i: pl.Tile[[128, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                                wv,
                                [k0_2, kv0],
                                [128, 64],
                                [128, 64],
                                target_memory=pl.Mem.Mat,
                                transpose=False,
                            )
                            v_acc_next: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(
                                v_acc_iter, tile_a_i_2, tile_wv_i
                            )
                            v_acc_rv: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(v_acc_next)
                        v_proj_tile = pl.tile.store(v_acc_rv, [0, kv0], v_proj_iter2)
                        k_proj_rv2, v_proj_rv2 = pl.yield_(k_proj_tile, v_proj_tile)
                    k_proj_rv, v_proj_rv = pl.yield_(k_proj_rv2, v_proj_rv2)
                return k_proj_rv, v_proj_rv

            @pl.function
            def main(
                self,
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
                wv: pl.Tensor[[512, 512], pl.BF16],
                k_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                v_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]]:
                result: pl.Tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]] = self.kv_proj(
                    normed_tile, wk, wv, k_proj, v_proj
                )
                return result

        After = _run_to_optimize_orch_tensors(Before)
        ir.assert_structural_equal(After, Expected)

    def test_post_outline_kv_dynamic_start_aggregate_shape_rewrites(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kv_proj(
                self,
                k_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                v_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                ob_chunk: pl.Scalar[pl.INDEX],
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
                wv: pl.Tensor[[512, 512], pl.BF16],
            ) -> tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]]:
                for ob, (k_proj_iter, v_proj_iter) in pl.range(
                    ob_chunk, ob_chunk + 4, init_values=(k_proj, v_proj)
                ):
                    kv0: pl.Scalar[pl.INDEX] = ob * 64
                    tile_a: pl.Tile[[16, 128], pl.BF16] = pl.tile.load(
                        normed_tile, [0, 0], [16, 128], [16, 128]
                    )
                    tile_wk: pl.Tile[[128, 64], pl.BF16] = pl.tile.load(wk, [0, kv0], [128, 64], [128, 64])
                    k_acc: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(tile_a, tile_wk)
                    k_proj_next: pl.Tensor[[16, 512], pl.FP32] = pl.tile.store(k_acc, [0, kv0], k_proj_iter)

                    tile_wv: pl.Tile[[128, 64], pl.BF16] = pl.tile.load(wv, [0, kv0], [128, 64], [128, 64])
                    v_acc: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(tile_a, tile_wv)
                    v_proj_next: pl.Tensor[[16, 512], pl.FP32] = pl.tile.store(v_acc, [0, kv0], v_proj_iter)
                    k_proj_rv, v_proj_rv = pl.yield_(k_proj_next, v_proj_next)
                return k_proj_rv, v_proj_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
                wv: pl.Tensor[[512, 512], pl.BF16],
                k_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                v_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]]:
                for ob_chunk, (k_proj_iter, v_proj_iter) in pl.range(0, 8, 4, init_values=(k_proj, v_proj)):
                    result: tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]] = (
                        self.kv_proj(k_proj_iter, v_proj_iter, ob_chunk, normed_tile, wk, wv)
                    )
                    k_proj_next: pl.Tensor[[16, 512], pl.FP32] = result[0]
                    v_proj_next: pl.Tensor[[16, 512], pl.FP32] = result[1]
                    k_proj_rv, v_proj_rv = pl.yield_(k_proj_next, v_proj_next)
                return k_proj_rv, v_proj_rv

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kv_proj(
                self,
                k_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                v_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                ob_chunk: pl.Scalar[pl.INDEX],
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
                wv: pl.Tensor[[512, 512], pl.BF16],
            ) -> tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]]:
                for ob, (k_proj_iter, v_proj_iter) in pl.range(
                    ob_chunk, ob_chunk + 4, init_values=(k_proj, v_proj)
                ):
                    kv0: pl.Scalar[pl.INDEX] = ob * 64
                    tile_a: pl.Tile[[16, 128], pl.BF16, pl.Mem.Vec] = pl.tile.load(
                        normed_tile, [0, 0], [16, 128], [16, 128], target_memory=pl.Mem.Vec, transpose=False
                    )
                    tile_wk: pl.Tile[[128, 64], pl.BF16, pl.Mem.Vec] = pl.tile.load(
                        wk, [0, kv0], [128, 64], [128, 64], target_memory=pl.Mem.Vec, transpose=False
                    )
                    k_acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(tile_a, tile_wk)
                    k_proj_next = pl.tile.store(k_acc, [0, kv0], k_proj_iter)
                    tile_wv: pl.Tile[[128, 64], pl.BF16, pl.Mem.Vec] = pl.tile.load(
                        wv, [0, kv0], [128, 64], [128, 64], target_memory=pl.Mem.Vec, transpose=False
                    )
                    v_acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(tile_a, tile_wv)
                    v_proj_next = pl.tile.store(v_acc, [0, kv0], v_proj_iter)
                    k_proj_rv, v_proj_rv = pl.yield_(k_proj_next, v_proj_next)
                return k_proj_rv, v_proj_rv

            @pl.function(type=pl.FunctionType.InCore)
            def kv_proj__windowed(
                self,
                k_proj: pl.Out[
                    pl.Tensor[[16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)]
                ],
                v_proj: pl.Out[
                    pl.Tensor[[16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)]
                ],
                ob_chunk: pl.Scalar[pl.INDEX],
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
                wv: pl.Tensor[[512, 512], pl.BF16],
            ) -> tuple[
                pl.Tensor[[16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)],
                pl.Tensor[[16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)],
            ]:
                for ob, (k_proj_iter, v_proj_iter) in pl.range(
                    ob_chunk, ob_chunk + 4, init_values=(k_proj, v_proj)
                ):
                    kv0: pl.Scalar[pl.INDEX] = ob * 64
                    tile_a: pl.Tile[[16, 128], pl.BF16, pl.Mem.Vec] = pl.tile.load(
                        normed_tile, [0, 0], [16, 128], [16, 128], target_memory=pl.Mem.Vec, transpose=False
                    )
                    tile_wk: pl.Tile[[128, 64], pl.BF16, pl.Mem.Vec] = pl.tile.load(
                        wk, [0, kv0], [128, 64], [128, 64], target_memory=pl.Mem.Vec, transpose=False
                    )
                    k_acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(tile_a, tile_wk)
                    k_proj_next: pl.Tensor[
                        [16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)
                    ] = pl.tile.store(k_acc, [0, kv0 - ob_chunk * 64], k_proj_iter)
                    tile_wv: pl.Tile[[128, 64], pl.BF16, pl.Mem.Vec] = pl.tile.load(
                        wv, [0, kv0], [128, 64], [128, 64], target_memory=pl.Mem.Vec, transpose=False
                    )
                    v_acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(tile_a, tile_wv)
                    v_proj_next: pl.Tensor[
                        [16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)
                    ] = pl.tile.store(v_acc, [0, kv0 - ob_chunk * 64], v_proj_iter)
                    k_proj_rv, v_proj_rv = pl.yield_(k_proj_next, v_proj_next)
                return k_proj_rv, v_proj_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
                wv: pl.Tensor[[512, 512], pl.BF16],
                k_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                v_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]]:
                for ob_chunk, (k_proj_iter, v_proj_iter) in pl.range(0, 8, 4, init_values=(k_proj, v_proj)):
                    k_proj_iter__window = pl.tensor.slice(k_proj_iter, [16, 256], [0, ob_chunk * 64])
                    v_proj_iter__window = pl.tensor.slice(v_proj_iter, [16, 256], [0, ob_chunk * 64])
                    result__windowed: pl.Tuple[
                        pl.Tensor[
                            [16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)
                        ],
                        pl.Tensor[
                            [16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)
                        ],
                    ] = self.kv_proj__windowed(
                        k_proj_iter__window, v_proj_iter__window, ob_chunk, normed_tile, wk, wv
                    )
                    result__windowed_0: pl.Tensor[
                        [16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)
                    ] = result__windowed[0]
                    result__assembled_0 = pl.tensor.assemble(
                        k_proj_iter, result__windowed_0, [0, ob_chunk * 64]
                    )
                    result__windowed_1: pl.Tensor[
                        [16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)
                    ] = result__windowed[1]
                    result__assembled_1 = pl.tensor.assemble(
                        v_proj_iter, result__windowed_1, [0, ob_chunk * 64]
                    )
                    result: pl.Tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]] = [
                        result__assembled_0,
                        result__assembled_1,
                    ]
                    k_proj_next = result__assembled_0
                    v_proj_next = result__assembled_1
                    k_proj_rv, v_proj_rv = pl.yield_(k_proj_next, v_proj_next)
                return k_proj_rv, v_proj_rv

        After = _run_to_optimize_orch_tensors(Before)
        ir.assert_structural_equal(After, Expected)

    def test_post_outline_kv_nested_loop_local_parent_rewrites(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kv_proj(
                self,
                k_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                v_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                ob_chunk: pl.Scalar[pl.INDEX],
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
                wv: pl.Tensor[[512, 512], pl.BF16],
            ) -> tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]]:
                for ob, (k_proj_iter, v_proj_iter) in pl.range(
                    ob_chunk, ob_chunk + 4, init_values=(k_proj, v_proj)
                ):
                    kv0: pl.Scalar[pl.INDEX] = ob * 64
                    tile_a: pl.Tile[[16, 128], pl.BF16] = pl.tile.load(
                        normed_tile, [0, 0], [16, 128], [16, 128]
                    )
                    tile_wk: pl.Tile[[128, 64], pl.BF16] = pl.tile.load(wk, [0, kv0], [128, 64], [128, 64])
                    k_acc: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(tile_a, tile_wk)
                    k_proj_next: pl.Tensor[[16, 512], pl.FP32] = pl.tile.store(k_acc, [0, kv0], k_proj_iter)

                    tile_wv: pl.Tile[[128, 64], pl.BF16] = pl.tile.load(wv, [0, kv0], [128, 64], [128, 64])
                    v_acc: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(tile_a, tile_wv)
                    v_proj_next: pl.Tensor[[16, 512], pl.FP32] = pl.tile.store(v_acc, [0, kv0], v_proj_iter)
                    k_proj_rv, v_proj_rv = pl.yield_(k_proj_next, v_proj_next)
                return k_proj_rv, v_proj_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
                wv: pl.Tensor[[512, 512], pl.BF16],
            ) -> tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]]:
                final_k: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.create(
                    [16, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                final_v: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.create(
                    [16, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                for layer_idx, (final_k_iter, final_v_iter) in pl.range(40, init_values=(final_k, final_v)):
                    k_proj: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.create(
                        [16, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND
                    )
                    v_proj: pl.Tensor[[16, 512], pl.FP32] = pl.tensor.create(
                        [16, 512], dtype=pl.FP32, layout=pl.TensorLayout.ND
                    )
                    for ob_chunk, (k_proj_iter, v_proj_iter) in pl.parallel(
                        0, 8, 4, init_values=(k_proj, v_proj)
                    ):
                        result: tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]] = (
                            self.kv_proj(k_proj_iter, v_proj_iter, ob_chunk, normed_tile, wk, wv)
                        )
                        k_proj_next: pl.Tensor[[16, 512], pl.FP32] = result[0]
                        v_proj_next: pl.Tensor[[16, 512], pl.FP32] = result[1]
                        k_proj_rv, v_proj_rv = pl.yield_(k_proj_next, v_proj_next)
                    final_k_next: pl.Tensor[[16, 512], pl.FP32] = k_proj_rv
                    final_v_next: pl.Tensor[[16, 512], pl.FP32] = v_proj_rv
                    final_k_rv, final_v_rv = pl.yield_(final_k_next, final_v_next)
                return final_k_rv, final_v_rv

        After = _run_to_optimize_orch_tensors(Before)

        assert After.get_function("kv_proj__windowed") is not None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "pl.tensor.slice(k_proj_iter" in printed_main
        assert "pl.tensor.slice(v_proj_iter" in printed_main
        assert "kv_proj__windowed(k_proj_iter__window, v_proj_iter__window" in printed_main

    def test_post_outline_kv_direct_tuple_use_remains_defined(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kv_proj(
                self,
                k_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                v_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                ob_chunk: pl.Scalar[pl.INDEX],
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
                wv: pl.Tensor[[512, 512], pl.BF16],
            ) -> tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]]:
                for ob, (k_proj_iter, v_proj_iter) in pl.range(
                    ob_chunk, ob_chunk + 4, init_values=(k_proj, v_proj)
                ):
                    kv0: pl.Scalar[pl.INDEX] = ob * 64
                    tile_a: pl.Tile[[16, 128], pl.BF16] = pl.tile.load(
                        normed_tile, [0, 0], [16, 128], [16, 128]
                    )
                    tile_wk: pl.Tile[[128, 64], pl.BF16] = pl.tile.load(wk, [0, kv0], [128, 64], [128, 64])
                    k_acc: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(tile_a, tile_wk)
                    k_proj_next: pl.Tensor[[16, 512], pl.FP32] = pl.tile.store(k_acc, [0, kv0], k_proj_iter)

                    tile_wv: pl.Tile[[128, 64], pl.BF16] = pl.tile.load(wv, [0, kv0], [128, 64], [128, 64])
                    v_acc: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(tile_a, tile_wv)
                    v_proj_next: pl.Tensor[[16, 512], pl.FP32] = pl.tile.store(v_acc, [0, kv0], v_proj_iter)
                    k_proj_rv, v_proj_rv = pl.yield_(k_proj_next, v_proj_next)
                return k_proj_rv, v_proj_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
                wv: pl.Tensor[[512, 512], pl.BF16],
                k_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                v_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]]:
                ob_chunk: pl.Scalar[pl.INDEX] = 0
                result: tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]] = self.kv_proj(
                    k_proj, v_proj, ob_chunk, normed_tile, wk, wv
                )
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kv_proj(
                self,
                k_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                v_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                ob_chunk: pl.Scalar[pl.INDEX],
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
                wv: pl.Tensor[[512, 512], pl.BF16],
            ) -> tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]]:
                for ob, (k_proj_iter, v_proj_iter) in pl.range(
                    ob_chunk, ob_chunk + 4, init_values=(k_proj, v_proj)
                ):
                    kv0: pl.Scalar[pl.INDEX] = ob * 64
                    tile_a: pl.Tile[[16, 128], pl.BF16, pl.Mem.Vec] = pl.tile.load(
                        normed_tile, [0, 0], [16, 128], [16, 128], target_memory=pl.Mem.Vec, transpose=False
                    )
                    tile_wk: pl.Tile[[128, 64], pl.BF16, pl.Mem.Vec] = pl.tile.load(
                        wk, [0, kv0], [128, 64], [128, 64], target_memory=pl.Mem.Vec, transpose=False
                    )
                    k_acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(tile_a, tile_wk)
                    k_proj_next = pl.tile.store(k_acc, [0, kv0], k_proj_iter)
                    tile_wv: pl.Tile[[128, 64], pl.BF16, pl.Mem.Vec] = pl.tile.load(
                        wv, [0, kv0], [128, 64], [128, 64], target_memory=pl.Mem.Vec, transpose=False
                    )
                    v_acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(tile_a, tile_wv)
                    v_proj_next = pl.tile.store(v_acc, [0, kv0], v_proj_iter)
                    k_proj_rv, v_proj_rv = pl.yield_(k_proj_next, v_proj_next)
                return k_proj_rv, v_proj_rv

            @pl.function(type=pl.FunctionType.InCore)
            def kv_proj__windowed(
                self,
                k_proj: pl.Out[
                    pl.Tensor[[16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)]
                ],
                v_proj: pl.Out[
                    pl.Tensor[[16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)]
                ],
                ob_chunk: pl.Scalar[pl.INDEX],
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
                wv: pl.Tensor[[512, 512], pl.BF16],
            ) -> tuple[
                pl.Tensor[[16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)],
                pl.Tensor[[16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)],
            ]:
                for ob, (k_proj_iter, v_proj_iter) in pl.range(
                    ob_chunk, ob_chunk + 4, init_values=(k_proj, v_proj)
                ):
                    kv0: pl.Scalar[pl.INDEX] = ob * 64
                    tile_a: pl.Tile[[16, 128], pl.BF16, pl.Mem.Vec] = pl.tile.load(
                        normed_tile, [0, 0], [16, 128], [16, 128], target_memory=pl.Mem.Vec, transpose=False
                    )
                    tile_wk: pl.Tile[[128, 64], pl.BF16, pl.Mem.Vec] = pl.tile.load(
                        wk, [0, kv0], [128, 64], [128, 64], target_memory=pl.Mem.Vec, transpose=False
                    )
                    k_acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(tile_a, tile_wk)
                    k_proj_next: pl.Tensor[
                        [16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)
                    ] = pl.tile.store(k_acc, [0, kv0 - ob_chunk * 64], k_proj_iter)
                    tile_wv: pl.Tile[[128, 64], pl.BF16, pl.Mem.Vec] = pl.tile.load(
                        wv, [0, kv0], [128, 64], [128, 64], target_memory=pl.Mem.Vec, transpose=False
                    )
                    v_acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(tile_a, tile_wv)
                    v_proj_next: pl.Tensor[
                        [16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)
                    ] = pl.tile.store(v_acc, [0, kv0 - ob_chunk * 64], v_proj_iter)
                    k_proj_rv, v_proj_rv = pl.yield_(k_proj_next, v_proj_next)
                return k_proj_rv, v_proj_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
                wv: pl.Tensor[[512, 512], pl.BF16],
                k_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                v_proj: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]]:
                k_proj__window = pl.tensor.slice(k_proj, [16, 256], [0, 0])
                v_proj__window = pl.tensor.slice(v_proj, [16, 256], [0, 0])
                result__windowed: pl.Tuple[
                    pl.Tensor[[16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)],
                    pl.Tensor[[16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)],
                ] = self.kv_proj__windowed(k_proj__window, v_proj__window, 0, normed_tile, wk, wv)
                result__windowed_0: pl.Tensor[
                    [16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)
                ] = result__windowed[0]
                result__assembled_0 = pl.tensor.assemble(k_proj, result__windowed_0, [0, 0])
                result__windowed_1: pl.Tensor[
                    [16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)
                ] = result__windowed[1]
                result__assembled_1 = pl.tensor.assemble(v_proj, result__windowed_1, [0, 0])
                result: pl.Tuple[pl.Tensor[[16, 512], pl.FP32], pl.Tensor[[16, 512], pl.FP32]] = [
                    result__assembled_0,
                    result__assembled_1,
                ]
                return result

        After = _run_to_optimize_orch_tensors(Before)
        ir.assert_structural_equal(After, Expected)

    def test_post_outline_kv_descending_loop_aggregate_shape_rewrites(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def k_proj(
                self,
                k_out: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                ob_chunk: pl.Scalar[pl.INDEX],
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
            ) -> pl.Tensor[[16, 512], pl.FP32]:
                for ob, (k_iter,) in pl.range(ob_chunk + 3, ob_chunk - 1, -1, init_values=(k_out,)):
                    kv0: pl.Scalar[pl.INDEX] = ob * 64
                    tile_a: pl.Tile[[16, 128], pl.BF16] = pl.tile.load(
                        normed_tile, [0, 0], [16, 128], [16, 128]
                    )
                    tile_wk: pl.Tile[[128, 64], pl.BF16] = pl.tile.load(wk, [0, kv0], [128, 64], [128, 64])
                    k_acc: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(tile_a, tile_wk)
                    k_next: pl.Tensor[[16, 512], pl.FP32] = pl.tile.store(k_acc, [0, kv0], k_iter)
                    k_rv = pl.yield_(k_next)
                return k_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
                k_out: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
            ) -> pl.Tensor[[16, 512], pl.FP32]:
                for ob_chunk, (k_iter,) in pl.range(0, 8, 4, init_values=(k_out,)):
                    k_next: pl.Tensor[[16, 512], pl.FP32] = self.k_proj(k_iter, ob_chunk, normed_tile, wk)
                    k_rv = pl.yield_(k_next)
                return k_rv

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def k_proj(
                self,
                k_out: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                ob_chunk: pl.Scalar[pl.INDEX],
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
            ) -> pl.Tensor[[16, 512], pl.FP32]:
                for ob, (k_iter,) in pl.range(ob_chunk + 3, ob_chunk - 1, -1, init_values=(k_out,)):
                    kv0: pl.Scalar[pl.INDEX] = ob * 64
                    tile_a: pl.Tile[[16, 128], pl.BF16, pl.Mem.Vec] = pl.tile.load(
                        normed_tile, [0, 0], [16, 128], [16, 128], target_memory=pl.Mem.Vec, transpose=False
                    )
                    tile_wk: pl.Tile[[128, 64], pl.BF16, pl.Mem.Vec] = pl.tile.load(
                        wk, [0, kv0], [128, 64], [128, 64], target_memory=pl.Mem.Vec, transpose=False
                    )
                    k_acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(tile_a, tile_wk)
                    k_next = pl.tile.store(k_acc, [0, kv0], k_iter)
                    k_rv = pl.yield_(k_next)
                return k_rv

            @pl.function(type=pl.FunctionType.InCore)
            def k_proj__windowed(
                self,
                k_out: pl.Out[
                    pl.Tensor[[16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)]
                ],
                ob_chunk: pl.Scalar[pl.INDEX],
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
            ) -> pl.Tensor[[16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)]:
                for ob, (k_iter,) in pl.range(ob_chunk + 3, ob_chunk - 1, -1, init_values=(k_out,)):
                    kv0: pl.Scalar[pl.INDEX] = ob * 64
                    tile_a: pl.Tile[[16, 128], pl.BF16, pl.Mem.Vec] = pl.tile.load(
                        normed_tile, [0, 0], [16, 128], [16, 128], target_memory=pl.Mem.Vec, transpose=False
                    )
                    tile_wk: pl.Tile[[128, 64], pl.BF16, pl.Mem.Vec] = pl.tile.load(
                        wk, [0, kv0], [128, 64], [128, 64], target_memory=pl.Mem.Vec, transpose=False
                    )
                    k_acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(tile_a, tile_wk)
                    k_next: pl.Tensor[
                        [16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)
                    ] = pl.tile.store(k_acc, [0, kv0 - ob_chunk * 64], k_iter)
                    k_rv: pl.Tensor[
                        [16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)
                    ] = pl.yield_(k_next)
                return k_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
                k_out: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
            ) -> pl.Tensor[[16, 512], pl.FP32]:
                for ob_chunk, (k_iter,) in pl.range(0, 8, 4, init_values=(k_out,)):
                    k_iter__window = pl.tensor.slice(k_iter, [16, 256], [0, ob_chunk * 64])
                    k_next__windowed: pl.Tensor[
                        [16, 256], pl.FP32, pl.TensorView(stride=[512, 1], layout=pl.TensorLayout.ND)
                    ] = self.k_proj__windowed(k_iter__window, ob_chunk, normed_tile, wk)
                    k_next = pl.tensor.assemble(k_iter, k_next__windowed, [0, ob_chunk * 64])
                    k_rv = pl.yield_(k_next)
                return k_rv

        After = _run_to_optimize_orch_tensors(Before)
        ir.assert_structural_equal(After, Expected)

    def test_aggregate_out_with_bypass_read_stays_baseline(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def k_proj(
                self,
                k_out: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                ob_chunk: pl.Scalar[pl.INDEX],
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
            ) -> pl.Tensor[[16, 512], pl.FP32]:
                k_acc: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(k_out, [0, 0], [16, 64], [16, 64])
                for ob, (k_iter,) in pl.range(ob_chunk, ob_chunk + 4, init_values=(k_out,)):
                    kv0: pl.Scalar[pl.INDEX] = ob * 64
                    tile_a: pl.Tile[[16, 128], pl.BF16] = pl.tile.load(
                        normed_tile, [0, 0], [16, 128], [16, 128]
                    )
                    tile_wk: pl.Tile[[128, 64], pl.BF16] = pl.tile.load(wk, [0, kv0], [128, 64], [128, 64])
                    matmul: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(tile_a, tile_wk)
                    k_acc = pl.tile.add(k_acc, matmul)
                    k_next: pl.Tensor[[16, 512], pl.FP32] = pl.tile.store(k_acc, [0, kv0], k_iter)
                    k_rv = pl.yield_(k_next)
                return k_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
                k_out: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
            ) -> pl.Tensor[[16, 512], pl.FP32]:
                for ob_chunk, (k_iter,) in pl.range(0, 8, 4, init_values=(k_out,)):
                    k_next: pl.Tensor[[16, 512], pl.FP32] = self.k_proj(k_iter, ob_chunk, normed_tile, wk)
                    k_rv = pl.yield_(k_next)
                return k_rv

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def k_proj(
                self,
                k_out: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
                ob_chunk: pl.Scalar[pl.INDEX],
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
            ) -> pl.Tensor[[16, 512], pl.FP32]:
                k_acc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    k_out, [0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                for ob, (k_iter, k_acc_iter) in pl.range(ob_chunk, ob_chunk + 4, init_values=(k_out, k_acc)):
                    kv0: pl.Scalar[pl.INDEX] = ob * 64
                    tile_a: pl.Tile[[16, 128], pl.BF16, pl.Mem.Vec] = pl.tile.load(
                        normed_tile, [0, 0], [16, 128], [16, 128], target_memory=pl.Mem.Vec, transpose=False
                    )
                    tile_wk: pl.Tile[[128, 64], pl.BF16, pl.Mem.Vec] = pl.tile.load(
                        wk, [0, kv0], [128, 64], [128, 64], target_memory=pl.Mem.Vec, transpose=False
                    )
                    matmul: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(tile_a, tile_wk)
                    k_acc_next: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(k_acc_iter, matmul)
                    k_next = pl.tile.store(k_acc_next, [0, kv0], k_iter)
                    k_rv, k_acc_rv = pl.yield_(k_next, k_acc_next)
                return k_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                normed_tile: pl.Tensor[[16, 512], pl.BF16],
                wk: pl.Tensor[[512, 512], pl.BF16],
                k_out: pl.Out[pl.Tensor[[16, 512], pl.FP32]],
            ) -> pl.Tensor[[16, 512], pl.FP32]:
                for ob_chunk, (k_iter,) in pl.range(0, 8, 4, init_values=(k_out,)):
                    k_next = self.k_proj(k_iter, ob_chunk, normed_tile, wk)
                    k_rv = pl.yield_(k_next)
                return k_rv

        After = _run_to_optimize_orch_tensors(Before)
        ir.assert_structural_equal(After, Expected)

    def test_overlapping_sequential_windows_stay_baseline(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_stripe(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                for i in pl.range(4):
                    row: pl.Scalar[pl.INDEX] = i * 32
                    out = self.kernel_stripe(data, row, out)
                return out

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Before)

    def test_callsite_in_while_stays_baseline(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_stripe(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                row: pl.Scalar[pl.INDEX] = 0
                for row_iter, out_iter in pl.while_(init_values=(row, out)):
                    pl.cond(row_iter < n)
                    out_next: pl.Tensor[[256, 64], pl.FP32] = self.kernel_stripe(data, row_iter, out_iter)
                    row_next: pl.Scalar[pl.INDEX] = row_iter + 64
                    row_rv, out_rv = pl.yield_(row_next, out_next)
                return out_rv

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Before)

    def test_full_shape_zero_offset_window_stays_baseline(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_full(
                self,
                data: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [0, 0], [64, 64])
                ret: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                result: pl.Tensor[[64, 64], pl.FP32] = self.kernel_full(data, out)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_full(
                self,
                data: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    data, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                ret = pl.tile.store(tile, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                result = self.kernel_full(data, out)
                return result

        After = _run_to_optimize_orch_tensors(Before)
        ir.assert_structural_equal(After, Expected)


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


class TestPattern3WhileLoop:
    """Pattern 3 (AssembleLoopRewriter) is ForStmt-only.

    The rewriter (LoopRewriteMutator) only overrides VisitStmt_(ForStmtPtr)
    (src ~line 1328); there is no WhileStmt branch. So a while-carried
    tile.assemble accumulation must stay baseline: the tile.create buffer is
    kept, the iter-arg init stays the buffer (not the Out param), and the
    tile.assemble is NOT rewritten to tile.store. This is the dual of the
    passing ForStmt case in TestAssembleLoopRewrite.test_assemble_loop_to_store_loop.
    """

    def test_while_assemble_loop_not_rewritten(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[1, 32], pl.FP32],
                n: pl.Scalar[pl.INDEX],
                ret0__out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                buf__tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.create(
                    [1, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                i0: pl.Scalar[pl.INDEX] = 0
                for acc, ii in pl.while_(init_values=(buf__tile, i0)):
                    pl.cond(ii < n)
                    off: pl.Scalar[pl.INDEX] = ii * 32
                    chunk__tile: pl.Tile[[1, 32], pl.FP32] = pl.load(x, [0, 0], [1, 32])
                    acc_next__tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.assemble(acc, chunk__tile, [0, off])
                    ii_next: pl.Scalar[pl.INDEX] = ii + 1
                    acc_rv, ii_rv = pl.yield_(acc_next__tile, ii_next)
                ret0__store: pl.Tensor[[1, 64], pl.FP32] = pl.store(acc_rv, [0, 0], ret0__out)
                return ret0__store

            @pl.function
            def main(
                self, x: pl.Tensor[[1, 32], pl.FP32], n: pl.Scalar[pl.INDEX]
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                ret0__out: pl.Tensor[[1, 64], pl.FP32] = pl.create_tensor([1, 64], dtype=pl.FP32)
                y: pl.Tensor[[1, 64], pl.FP32] = self.main_incore_0(x, n, ret0__out)
                return y

        After = passes.optimize_orch_tensors()(Before)
        # Pattern 3 only matches ForStmt; the WhileStmt assemble loop is left
        # untouched. (Patterns 1/4 also do not fire: the In param x is sliced
        # nowhere, and there is no iter-arg-fed In/Out merge.)
        ir.assert_structural_equal(After, Before)


class TestOutWindowMultiOutAllOrNothing:
    """Pattern 5 multi-Out policy is all-or-nothing (doc line 95, src ~line 1815).

    AnalyzeFinalStore rejects an Out whose only store covers the full tensor at
    zero offset (src ~line 3115). When the FinalStore analysis encounters such
    an Out among several, `all_final` is cleared and the whole callee falls back
    to baseline — no per-Out partial windowing.
    """

    def test_one_full_shape_out_blocks_whole_callee(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kv_stripe(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                k_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
                v_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32]]:
                # k_out: local 64x64 window at [row_offset, 0] -> windowable.
                ktile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                k_next: pl.Tensor[[256, 64], pl.FP32] = pl.store(ktile, [row_offset, 0], k_out)
                # v_out: full 256x64 write at [0, 0] -> NOT a window (full-shape,
                # zero-offset). This blocks the all-or-nothing multi-Out rewrite.
                vtile: pl.Tile[[256, 64], pl.FP32] = pl.load(data, [0, 0], [256, 64])
                v_next: pl.Tensor[[256, 64], pl.FP32] = pl.store(vtile, [0, 0], v_out)
                return k_next, v_next

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                k_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
                v_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32]]:
                row: pl.Scalar[pl.INDEX] = 64
                result: tuple[pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32]] = self.kv_stripe(
                    data, row, k_out, v_out
                )
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kv_stripe(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                k_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
                v_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32]]:
                ktile: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    data, [row_offset, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                k_next = pl.tile.store(ktile, [row_offset, 0], k_out)
                vtile: pl.Tile[[256, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    data, [0, 0], [256, 64], [256, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                v_next = pl.tile.store(vtile, [0, 0], v_out)
                return k_next, v_next

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                k_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
                v_out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32]]:
                row: pl.Scalar[pl.INDEX] = 64
                result = self.kv_stripe(data, row, k_out, v_out)
                return result

        After = passes.optimize_orch_tensors()(Before)
        # all-or-nothing: v_out is a full-shape store, so neither Out is
        # externalized and no __windowed clone is emitted.
        assert After.get_function("kv_stripe__windowed") is None
        ir.assert_structural_equal(After, Expected)


class TestOutWindowSubmitCall:
    """Pattern 5 IsSubmitCall branch (TASK_ID return augmentation).

    TryRewriteCall has an IsSubmitCall branch (src ~line 2464) that, for a
    task-launch call whose return type is augmented with a trailing
    Scalar[TASK_ID], must keep the TASK_ID in the windowed call's return type
    and route through the tuple-projection tail (the single-output FinalStore
    shortcut is gated by `!is_submit_call`). Per pass-submit-awareness rule 1
    ("when walking calls, walk Submit too"), a windowable kernel launched via
    pl.submit inside pl.manual_scope SHOULD be externalized just like the
    plain-call form in
    TestOutWindowExternalizer.test_direct_out_call_rewrites_to_windowed_clone.
    """

    @pytest.mark.xfail(
        reason="suspected bug: OrchRewriter dispatches via As<Call>, which is "
        "exact-kind and never matches a Submit node; the IsSubmitCall branch is "
        "unreachable from the pl.submit DSL form, so the windowable submit stays "
        "baseline instead of being externalized",
        strict=False,
    )
    def test_submit_windowable_kernel_is_externalized(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_stripe(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                with pl.manual_scope():
                    row: pl.Scalar[pl.INDEX] = 64
                    out_next, tid = pl.submit(self.kernel_stripe, data, row, out)
                return out_next

        After = passes.optimize_orch_tensors()(Before)
        # A statically provable 64x64 window write at [64, 0] must be
        # externalized: the windowed clone exists and the orchestration call
        # site slices the Out param before the (still task-launching) call.
        windowed = After.get_function("kernel_stripe__windowed")
        assert windowed is not None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "pl.tensor.slice(out" in printed_main
        assert "kernel_stripe__windowed" in printed_main


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
