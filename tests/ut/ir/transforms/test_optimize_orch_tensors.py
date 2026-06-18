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

from typing import Literal, cast

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

OptimizeWindowPolicy = Literal["auto", "all", "off"]


def _run_to_optimize_orch_tensors(
    program,
    *,
    window_policy: str = "auto",
):
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    result = program
    for pass_name, pass_obj in zip(pm.pass_names, pm.passes, strict=True):
        if pass_name == "OptimizeOrchTensors":
            result = passes.optimize_orch_tensors(
                window_policy=cast(OptimizeWindowPolicy, window_policy),
            )(result)
            return result
        result = pass_obj(result)
    raise AssertionError("Default pipeline did not run OptimizeOrchTensors")


def _run_aggressive_exact_to_optimize_orch_tensors(program):
    return _run_to_optimize_orch_tensors(program, window_policy="all")


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

        After = _run_to_optimize_orch_tensors(Before)
        ir.assert_structural_equal(After, Expected)

    def test_iter_arg_merge_preserves_dump_vars(self):
        """The Out->InOut merge rewrites the incore call site; a ``kAttrDumpVars``
        tag on a surviving (non-merged) In arg must ride through the rewrite.

        Regression: ``CallSiteRewriter::VisitStmt_`` rebuilt the call with a Call
        constructor that drops ``attrs_``, so ``pl.dump_tag``-seeded ``dump_vars``
        was lost. ``x`` is loop-invariant (same Var across iterations) and is
        consumed by the in-loop dispatch but is NOT the merged Out param, so its
        dump tag must survive the merge."""

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

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self, acc0: pl.Tensor[[64], pl.FP32], x: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                pl.dump_tag(x)
                for i, (acc,) in pl.range(10, init_values=(acc0,)):
                    ret0__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                    result: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, x, ret0__out)
                    new_acc = pl.yield_(result)
                return new_acc

        After = _run_to_optimize_orch_tensors(Before)

        dump_var_names: list[str] = []

        class _Collector(ir.IRVisitor):
            def visit_call(self, op):
                name = getattr(getattr(op, "op", None), "name", "")
                if name == "main_incore_0":
                    dv = (op.attrs or {}).get("dump_vars")
                    if dv:
                        dump_var_names.extend(v.name_hint.split("__", 1)[0] for v in dv)
                super().visit_call(op)

        _Collector().visit_program(After)
        assert "x" in dump_var_names, (
            f"dump_vars dropped by the iter-arg-merge call rewrite; got {dump_var_names}"
        )

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

    def test_coalesce_inserts_mlp_silu_runtime_current_before_full_down_proj_read(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def silu(
                self,
                src: pl.Tensor[[64, 256], pl.FP32],
                col: pl.Scalar[pl.INDEX],
                mlp_silu_tile: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
            ) -> pl.Tensor[[64, 256], pl.FP32]:
                tile: pl.Tile[[64, 128], pl.FP32] = pl.tile.load(src, [0, col], [64, 128], [64, 128])
                return pl.tile.store(tile, [0, col], mlp_silu_tile)

            @pl.function(type=pl.FunctionType.InCore)
            def down_proj(
                self,
                mlp_silu_tile: pl.Tensor[[64, 256], pl.FP32],
                col: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
            ) -> tuple[pl.Tensor[[64, 256], pl.FP32], pl.Tensor[[64, 256], pl.FP32]]:
                tile: pl.Tile[[64, 128], pl.FP32] = pl.tile.load(
                    mlp_silu_tile, [0, col], [64, 128], [64, 128]
                )
                next_out: pl.Tensor[[64, 256], pl.FP32] = pl.tile.store(tile, [0, col], out)
                return next_out, mlp_silu_tile

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                src: pl.Tensor[[64, 256], pl.FP32],
                mlp_silu_tile: pl.Tensor[[64, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
            ) -> pl.Tensor[[64, 256], pl.FP32]:
                for i, (mlp_silu_tile_iter,) in pl.range(0, 2, init_values=(mlp_silu_tile,)):
                    col: pl.Scalar[pl.INDEX] = i * 128
                    mlp_silu_tile_next = self.silu(src, col, mlp_silu_tile_iter)
                    mlp_silu_tile_rv = pl.yield_(mlp_silu_tile_next)

                for j, (out_iter,) in pl.range(0, 2, init_values=(out,)):
                    col: pl.Scalar[pl.INDEX] = j * 128
                    result = self.down_proj(mlp_silu_tile_rv, col, out_iter)
                    out_next: pl.Tensor[[64, 256], pl.FP32] = result[0]
                    out_rv = pl.yield_(out_next)
                return out_rv

        After = _run_to_optimize_orch_tensors(Before, window_policy="all")

        assert After.get_function("__pypto_runtime_current_barrier") is None
        assert "__runtime_current" not in ir.python_print(_get_function(After, "main"))

    def test_coalesce_inserts_attention_runtime_current_before_full_out_proj_read(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def attention_writeback(
                self,
                ctx: pl.Tensor[[1, 128], pl.BF16],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
                attn_tile: pl.Out[pl.Tensor[[2, 256], pl.BF16]],
            ) -> pl.Tensor[[2, 256], pl.BF16]:
                tile: pl.Tile[[1, 128], pl.BF16] = pl.tile.load(ctx, [0, 0], [1, 128], [1, 128])
                return pl.tile.store(tile, [row, col], attn_tile)

            @pl.function(type=pl.FunctionType.InCore)
            def out_proj(
                self,
                attn_tile: pl.Tensor[[2, 256], pl.BF16],
                col: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[2, 256], pl.FP32]],
            ) -> tuple[pl.Tensor[[2, 256], pl.FP32], pl.Tensor[[2, 256], pl.BF16]]:
                lhs0: pl.Tile[[2, 128], pl.BF16] = pl.tile.load(attn_tile, [0, 0], [2, 128], [2, 128])
                lhs1: pl.Tile[[2, 128], pl.BF16] = pl.tile.load(attn_tile, [0, 128], [2, 128], [2, 128])
                lhs: pl.Tile[[2, 128], pl.BF16] = pl.tile.add(lhs0, lhs1)
                cast: pl.Tile[[2, 128], pl.FP32] = pl.tile.cast(lhs, target_type=pl.FP32)
                out_next: pl.Tensor[[2, 256], pl.FP32] = pl.tile.store(cast, [0, col], out)
                return out_next, attn_tile

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                ctx: pl.Tensor[[1, 128], pl.BF16],
                attn_tile: pl.Tensor[[2, 256], pl.BF16],
                out: pl.Out[pl.Tensor[[2, 256], pl.FP32]],
            ) -> pl.Tensor[[2, 256], pl.FP32]:
                for row, (attn_tile_iter,) in pl.range(0, 2, init_values=(attn_tile,)):
                    col: pl.Scalar[pl.INDEX] = row * 128
                    attn_tile_next = self.attention_writeback(ctx, row, col, attn_tile_iter)
                    attn_tile_rv = pl.yield_(attn_tile_next)

                for j, (out_iter,) in pl.range(0, 2, init_values=(out,)):
                    col: pl.Scalar[pl.INDEX] = j * 128
                    result = self.out_proj(attn_tile_rv, col, out_iter)
                    out_next: pl.Tensor[[2, 256], pl.FP32] = result[0]
                    out_rv = pl.yield_(out_next)
                return out_rv

        After = _run_to_optimize_orch_tensors(Before, window_policy="all")

        assert After.get_function("__pypto_runtime_current_barrier") is None
        assert "__runtime_current" not in ir.python_print(_get_function(After, "main"))

    def test_pure_input_window_consumer_rewrites_to_windowed_clone(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def consume(
                self,
                score: pl.Tensor[[64, 128], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[32, 64], pl.FP32]:
                block: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.slice(score, [32, 64], [row_offset, 0])
                return block

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                score: pl.Tensor[[64, 128], pl.FP32],
            ) -> pl.Tensor[[32, 64], pl.FP32]:
                row: pl.Scalar[pl.INDEX] = 32
                return self.consume(score, row)

        After = _run_to_optimize_orch_tensors(Before)

        assert After.get_function("consume__windowed") is not None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "pl.tensor.slice(score" in printed_main
        assert "consume__windowed(score__ssa_v0__window, 32, ret0__out)" in printed_main

        printed_windowed = ir.python_print(_get_function(After, "consume__windowed"))
        assert "pl.Tensor[[32, 64], pl.FP32, pl.TensorView(stride=[128, 1]" in printed_windowed
        assert "pl.tile.load(score__ssa_v0, [0, 0]" in printed_windowed

        none_policy = _run_to_optimize_orch_tensors(Before, window_policy="off")
        assert none_policy.get_function("consume__windowed") is None

    def test_input_window_rejects_unrecoverable_dynamic_tensor_view_stride(self):
        M = pl.dynamic("M")
        N = pl.dynamic("N")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def tile_add(
                self,
                a: pl.Tensor[[M * 2, N], pl.FP32],
                b: pl.Tensor[[M * 2, N], pl.FP32],
                f: pl.Out[pl.Tensor[[M * 2, N], pl.FP32]],
            ) -> pl.Tensor[[M * 2, N], pl.FP32]:
                tile_f = pl.add(pl.load(a, [0, 0], [128, 128]), pl.load(b, [0, 0], [128, 128]))
                return pl.store(tile_f, [0, 0], f)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[M * 2, N], pl.FP32],
                b: pl.Tensor[[M * 2, N], pl.FP32],
                f: pl.Out[pl.Tensor[[M * 2, N], pl.FP32]],
            ) -> pl.Tensor[[M * 2, N], pl.FP32]:
                return self.tile_add(a, b, f)

        After = _run_to_optimize_orch_tensors(Before)
        assert After.get_function("tile_add__windowed") is None

    def test_topk_name_does_not_block_eligible_input_window(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def topk_like(
                self,
                score: pl.Tensor[[64, 128], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[32, 64], pl.FP32]:
                block: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.slice(score, [32, 64], [row_offset, 0])
                return block

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                score: pl.Tensor[[64, 128], pl.FP32],
            ) -> pl.Tensor[[32, 64], pl.FP32]:
                row: pl.Scalar[pl.INDEX] = 32
                return self.topk_like(score, row)

        After = _run_to_optimize_orch_tensors(Before)

        assert After.get_function("topk_like__windowed") is not None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "pl.tensor.slice(score" in printed_main
        assert "topk_like__windowed(score__ssa_v0__window, 32, ret0__out)" in printed_main

    def test_input_window_rewrite_accepts_loop_carried_affine_offset(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def consume(
                self,
                score: pl.Tensor[[64, 128], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[32, 64], pl.FP32]:
                block: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.slice(score, [32, 64], [row_offset, 0])
                return block

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                score: pl.Tensor[[64, 128], pl.FP32],
                seed: pl.Tensor[[32, 64], pl.FP32],
            ) -> pl.Tensor[[32, 64], pl.FP32]:
                for i, (result,) in pl.range(2, init_values=(seed,)):
                    row: pl.Scalar[pl.INDEX] = i * 32
                    block: pl.Tensor[[32, 64], pl.FP32] = self.consume(score, row)
                    result_rv = pl.yield_(block)
                return result_rv

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def consume(
                self,
                score: pl.Tensor[[64, 128], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                ret0__out: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
            ) -> pl.Tensor[[32, 64], pl.FP32]:
                block__tile: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    score, [row_offset, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                ret0__store: pl.Tensor[[32, 64], pl.FP32] = pl.tile.store(block__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.InCore)
            def consume__windowed(
                self,
                score: pl.Tensor[
                    [32, 64], pl.FP32, pl.TensorView(stride=[128, 1], layout=pl.TensorLayout.ND)
                ],
                row_offset: pl.Scalar[pl.INDEX],
                ret0__out: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
            ) -> pl.Tensor[[32, 64], pl.FP32]:
                block__tile: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    score, [0, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                ret0__store: pl.Tensor[[32, 64], pl.FP32] = pl.tile.store(block__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                score: pl.Tensor[[64, 128], pl.FP32],
                seed: pl.Tensor[[32, 64], pl.FP32],
            ) -> pl.Tensor[[32, 64], pl.FP32]:
                for i, (result,) in pl.range(2, init_values=(seed,)):
                    row: pl.Scalar[pl.INDEX] = i * 32
                    ret0__out: pl.Tensor[[32, 64], pl.FP32] = pl.create_tensor([32, 64], dtype=pl.FP32)
                    score__window: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.slice(score, [32, 64], [row, 0])
                    block: pl.Tensor[[32, 64], pl.FP32] = self.consume__windowed(
                        score__window, row, ret0__out
                    )
                    result_rv = pl.yield_(block)
                return result_rv

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)
        ir.assert_structural_equal(After, Expected)

    def test_final_store_keeps_already_detected_input_window(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def mix_window(
                self,
                data: pl.Tensor[[64, 128], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                tile: pl.Tile[[32, 64], pl.FP32] = pl.tile.load(data, [row_offset, 0], [32, 64], [32, 64])
                result: pl.Tensor[[64, 128], pl.FP32] = pl.tile.store(tile, [row_offset, 0], out)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[64, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                row: pl.Scalar[pl.INDEX] = 32
                return self.mix_window(data, row, out)

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "mix_window__windowed" in printed_main
        assert "pl.tensor.slice(data" in printed_main
        assert "pl.tensor.slice(out" in printed_main

        printed_windowed = ir.python_print(_get_function(After, "mix_window__windowed"))
        assert "pl.tile.load(data__ssa_v0, [0, 0]" in printed_windowed
        assert "pl.tile.store(tile__ssa_v0, [0, 0]" in printed_windowed

    def test_input_full_read_blocks_input_window_rewrite(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def consume(
                self,
                score: pl.Tensor[[64, 128], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
            ) -> tuple[pl.Tensor[[32, 64], pl.FP32], pl.Tensor[[64, 128], pl.FP32]]:
                block: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.slice(score, [32, 64], [row_offset, 0])
                return block, score

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                score: pl.Tensor[[64, 128], pl.FP32],
            ) -> tuple[pl.Tensor[[32, 64], pl.FP32], pl.Tensor[[64, 128], pl.FP32]]:
                row: pl.Scalar[pl.INDEX] = 32
                return self.consume(score, row)

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        assert After.get_function("consume__windowed") is None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "pl.tensor.slice(score" not in printed_main

    def test_no_return_input_consumer_stays_full_tensor(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def fence(
                self,
                src: pl.Tensor[[64, 128], pl.FP32],
                dummy: pl.Out[pl.Tensor[[64, 1], pl.FP32]],
            ):
                block: pl.Tile[[64, 1], pl.FP32] = pl.load(src, [0, 0], [64, 1])
                _next: pl.Tensor[[64, 1], pl.FP32] = pl.store(block, [0, 0], dummy)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                src: pl.Tensor[[64, 128], pl.FP32],
            ) -> pl.Scalar[pl.TASK_ID]:
                with pl.manual_scope():
                    dummy: pl.Tensor[[64, 1], pl.FP32] = pl.create_tensor([64, 1], dtype=pl.FP32)
                    _tid = pl.submit(self.fence, src, dummy)
                return _tid

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)
        assert After.get_function("fence__windowed") is None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "pl.tensor.slice(src" not in printed_main
        assert "pl.submit(fence, src__ssa_v0, dummy__ssa_v0)" in printed_main

    def test_indexer_score_writes_window_but_topk_score_read_stays_full(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def score_init(
                self,
                score: pl.Out[pl.Tensor[[4, 16], pl.FP32]],
                t0: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[4, 16], pl.FP32]:
                init_tile: pl.Tile[[2, 16], pl.FP32] = pl.tile.full([2, 16], dtype=pl.FP32, value=-1.0)
                score_next: pl.Tensor[[4, 16], pl.FP32] = pl.tile.store(init_tile, [t0, 0], score)
                return score_next

            @pl.function(type=pl.FunctionType.InCore)
            def score_writer(
                self,
                score: pl.Out[pl.Tensor[[4, 16], pl.FP32]],
                t0: pl.Scalar[pl.INDEX],
                cache0: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[4, 16], pl.FP32]:
                score_tile: pl.Tile[[2, 4], pl.FP32] = pl.tile.full([2, 4], dtype=pl.FP32, value=1.0)
                score_next: pl.Tensor[[4, 16], pl.FP32] = pl.tile.store(score_tile, [t0, cache0], score)
                return score_next

            @pl.function(type=pl.FunctionType.InCore)
            def topk_like(
                self,
                topk: pl.Out[pl.Tensor[[4, 16], pl.INT32]],
                t0: pl.Scalar[pl.INDEX],
                score: pl.Tensor[[4, 16], pl.FP32],
            ) -> pl.Tensor[[4, 16], pl.INT32]:
                invalid: pl.Tile[[1, 16], pl.INT32] = pl.tile.full([1, 16], dtype=pl.INT32, value=-1)
                topk_init: pl.Tensor[[4, 16], pl.INT32] = pl.tile.store(invalid, [t0, 0], topk)
                score_row: pl.Tile[[1, 16], pl.FP32] = pl.tile.load(score, [t0, 0], [1, 16], [1, 16])
                idx_tile: pl.Tile[[1, 16], pl.INT32] = pl.tile.cast(score_row, target_type=pl.INT32)
                topk_next: pl.Tensor[[4, 16], pl.INT32] = pl.tile.store(idx_tile, [t0, 0], topk_init)
                return topk_next

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                score: pl.Out[pl.Tensor[[4, 16], pl.FP32]],
                topk: pl.Out[pl.Tensor[[4, 16], pl.INT32]],
            ) -> tuple[pl.Tensor[[4, 16], pl.FP32], pl.Tensor[[4, 16], pl.INT32]]:
                for b, (score_iter, topk_iter) in pl.parallel(2, init_values=(score, topk)):
                    t0: pl.Scalar[pl.INDEX] = b * 2
                    score_init_next: pl.Tensor[[4, 16], pl.FP32] = self.score_init(score_iter, t0)
                    for cb, (score_iter2,) in pl.parallel(4, init_values=(score_init_next,)):
                        cache0: pl.Scalar[pl.INDEX] = cb * 4
                        score_next: pl.Tensor[[4, 16], pl.FP32] = self.score_writer(score_iter2, t0, cache0)
                        score_rv = pl.yield_(score_next)
                    topk_next: pl.Tensor[[4, 16], pl.INT32] = self.topk_like(topk_iter, t0, score_rv)
                    score_out, topk_out = pl.yield_(score_rv, topk_next)
                return score_out, topk_out

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "score_init__windowed" in printed_main
        assert "score_init__windowed(score_iter__window" in printed_main
        assert "score_writer__windowed" in printed_main
        assert "score_writer__windowed(score_iter2__window" in printed_main
        assert "topk_like(topk_iter, t0__ssa_v0, score_rv)" in printed_main
        assert "topk_like__windowed" not in printed_main
        assert "score_rv__window" not in printed_main

    def test_dynamic_indexed_reader_after_loop_carried_writer_keeps_full_parent(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def cache_write(
                self,
                cache: pl.Out[pl.Tensor[[1024, 64], pl.FP32]],
                data: pl.Tensor[[1, 64], pl.FP32],
                slot: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[1024, 64], pl.FP32]:
                src: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(data, [0, 0], [1, 64], [1, 64])
                result: pl.Tensor[[1024, 64], pl.FP32] = pl.tile.store(src, [slot, 0], cache)
                return result

            @pl.function(type=pl.FunctionType.InCore)
            def cache_read(
                self,
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
                cache: pl.Tensor[[1024, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
            ) -> pl.Tensor[[4, 64], pl.FP32]:
                for sb, (out_iter,) in pl.range(0, 4, init_values=(out,)):
                    pbid_i32: pl.Scalar[pl.INT32] = pl.tensor.read(block_table, [sb])
                    pbid: pl.Scalar[pl.INDEX] = pl.cast(pbid_i32, target_type=pl.INDEX)
                    row: pl.Scalar[pl.INDEX] = pbid * 128
                    cache_tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(cache, [row, 0], [1, 64], [1, 64])
                    out_next: pl.Tensor[[4, 64], pl.FP32] = pl.tile.store(cache_tile, [sb, 0], out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                cache: pl.Out[pl.Tensor[[1024, 64], pl.FP32]],
                data: pl.Tensor[[1, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[1024, 64], pl.FP32], pl.Tensor[[4, 64], pl.FP32]]:
                for _, (cache_iter,) in pl.range(1, init_values=(cache,)):
                    cache_next: pl.Tensor[[1024, 64], pl.FP32] = self.cache_write(cache_iter, data, 7)
                    cache_rv = pl.yield_(cache_next)
                result: pl.Tensor[[4, 64], pl.FP32] = self.cache_read(out, cache_rv, block_table)
                return cache_rv, result

        After = _run_to_optimize_orch_tensors(Before)

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "cache_read(out__ssa_v0, cache_next__ssa_v0, block_table__ssa_v0)" in printed_main
        assert "cache_read__windowed" not in printed_main
        assert "pl.tensor.slice(cache_next__ssa_v0, " not in printed_main

    def test_dynamic_indexed_reader_after_singleton_loop_writer_rematerializes_carrier(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def cache_write(
                self,
                cache: pl.Out[pl.Tensor[[1024, 64], pl.FP32]],
                data: pl.Tensor[[2, 64], pl.FP32],
                slot: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[1024, 64], pl.FP32]:
                for ki, (cache_iter,) in pl.range(0, 2, init_values=(cache,)):
                    src: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(data, [ki, 0], [1, 64], [1, 64])
                    row: pl.Scalar[pl.INDEX] = slot + ki * 128
                    cache_next: pl.Tensor[[1024, 64], pl.FP32] = pl.tile.store(src, [row, 0], cache_iter)
                    cache_rv = pl.yield_(cache_next)
                return cache_rv

            @pl.function(type=pl.FunctionType.InCore)
            def cache_read(
                self,
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
                cache: pl.Tensor[[1024, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
            ) -> pl.Tensor[[4, 64], pl.FP32]:
                for sb, (out_iter,) in pl.range(0, 4, init_values=(out,)):
                    pbid_i32: pl.Scalar[pl.INT32] = pl.tensor.read(block_table, [sb])
                    pbid: pl.Scalar[pl.INDEX] = pl.cast(pbid_i32, target_type=pl.INDEX)
                    row: pl.Scalar[pl.INDEX] = pbid * 128
                    cache_tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(cache, [row, 0], [1, 64], [1, 64])
                    out_next: pl.Tensor[[4, 64], pl.FP32] = pl.tile.store(cache_tile, [sb, 0], out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                cache: pl.Out[pl.Tensor[[1024, 64], pl.FP32]],
                data: pl.Tensor[[2, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
                slot_block: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[1024, 64], pl.FP32], pl.Tensor[[4, 64], pl.FP32]]:
                for _, (cache_iter,) in pl.parallel(slot_block, slot_block + 8, 8, init_values=(cache,)):
                    cache_next: pl.Tensor[[1024, 64], pl.FP32] = self.cache_write(cache_iter, data, 7)
                    cache_rv = pl.yield_(cache_next)
                result: pl.Tensor[[4, 64], pl.FP32] = self.cache_read(out, cache_rv, block_table)
                return cache_rv, result

        After = _run_to_optimize_orch_tensors(Before, window_policy="all")

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "cache_write__windowed" in printed_main
        assert "__carrier_scan_" not in printed_main
        assert "__carrier_current_remat" not in printed_main

    def test_dynamic_indexed_reader_after_non_singleton_loop_writer_keeps_full_parent(self):
        """v5: dynamic reader without carrier keeps full parent (no private dynamic reader)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def cache_write(
                self,
                cache: pl.Out[pl.Tensor[[1024, 64], pl.FP32]],
                data: pl.Tensor[[2, 64], pl.FP32],
                slot: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[1024, 64], pl.FP32]:
                src: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(data, [0, 0], [1, 64], [1, 64])
                result: pl.Tensor[[1024, 64], pl.FP32] = pl.tile.store(src, [slot, 0], cache)
                return result

            @pl.function(type=pl.FunctionType.InCore)
            def cache_read(
                self,
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
                cache: pl.Tensor[[1024, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
            ) -> pl.Tensor[[4, 64], pl.FP32]:
                for sb, (out_iter,) in pl.range(0, 4, init_values=(out,)):
                    pbid_i32: pl.Scalar[pl.INT32] = pl.tensor.read(block_table, [sb])
                    pbid: pl.Scalar[pl.INDEX] = pl.cast(pbid_i32, target_type=pl.INDEX)
                    row: pl.Scalar[pl.INDEX] = pbid * 128
                    cache_tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(cache, [row, 0], [1, 64], [1, 64])
                    out_next: pl.Tensor[[4, 64], pl.FP32] = pl.tile.store(cache_tile, [sb, 0], out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                cache: pl.Out[pl.Tensor[[1024, 64], pl.FP32]],
                data: pl.Tensor[[2, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
                slot_block: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[1024, 64], pl.FP32], pl.Tensor[[4, 64], pl.FP32]]:
                for i, (cache_iter,) in pl.range(slot_block, slot_block + 16, 8, init_values=(cache,)):
                    slot: pl.Scalar[pl.INDEX] = i * 128 + 7
                    cache_next: pl.Tensor[[1024, 64], pl.FP32] = self.cache_write(cache_iter, data, slot)
                    cache_rv = pl.yield_(cache_next)
                result: pl.Tensor[[4, 64], pl.FP32] = self.cache_read(out, cache_rv, block_table)
                return cache_rv, result

        After = _run_to_optimize_orch_tensors(Before, window_policy="all")

        printed_main = ir.python_print(_get_function(After, "main"))
        # v5: dynamic reader without carrier keeps full parent (no private dynamic reader).
        assert "cache_read__windowed" not in printed_main
        assert "__carrier_current_remat" not in printed_main

    def test_dynamic_reader_fallback_is_parent_local(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def cache_write(
                self,
                cache: pl.Out[pl.Tensor[[1024, 64], pl.FP32]],
                data: pl.Tensor[[1, 64], pl.FP32],
                slot: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[1024, 64], pl.FP32]:
                src: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(data, [0, 0], [1, 64], [1, 64])
                result: pl.Tensor[[1024, 64], pl.FP32] = pl.tile.store(src, [slot, 0], cache)
                return result

            @pl.function(type=pl.FunctionType.InCore)
            def unrelated_write(
                self,
                other: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
                data: pl.Tensor[[1, 64], pl.FP32],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                src: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(data, [0, 0], [1, 64], [1, 64])
                result: pl.Tensor[[16, 64], pl.FP32] = pl.tile.store(src, [3, 0], other)
                return result

            @pl.function(type=pl.FunctionType.InCore)
            def cache_read(
                self,
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
                cache: pl.Tensor[[1024, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
            ) -> pl.Tensor[[4, 64], pl.FP32]:
                for sb, (out_iter,) in pl.range(0, 4, init_values=(out,)):
                    pbid_i32: pl.Scalar[pl.INT32] = pl.tensor.read(block_table, [sb])
                    pbid: pl.Scalar[pl.INDEX] = pl.cast(pbid_i32, target_type=pl.INDEX)
                    row: pl.Scalar[pl.INDEX] = pbid * 128
                    cache_tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(cache, [row, 0], [1, 64], [1, 64])
                    out_next: pl.Tensor[[4, 64], pl.FP32] = pl.tile.store(cache_tile, [sb, 0], out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                cache: pl.Out[pl.Tensor[[1024, 64], pl.FP32]],
                other: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
                data: pl.Tensor[[1, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
            ) -> tuple[
                pl.Tensor[[1024, 64], pl.FP32],
                pl.Tensor[[16, 64], pl.FP32],
                pl.Tensor[[4, 64], pl.FP32],
            ]:
                cache_next: pl.Tensor[[1024, 64], pl.FP32] = self.cache_write(cache, data, 7)
                other_next: pl.Tensor[[16, 64], pl.FP32] = self.unrelated_write(other, data)
                result: pl.Tensor[[4, 64], pl.FP32] = self.cache_read(out, cache_next, block_table)
                return cache_next, other_next, result

        After = _run_to_optimize_orch_tensors(Before)

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "cache_read(out__ssa_v0, cache_next__ssa_v0, block_table__ssa_v0)" in printed_main
        assert "cache_read__windowed" not in printed_main
        assert "unrelated_write__windowed" in printed_main
        assert "other__ssa_v0__window" in printed_main
        assert "pl.tensor.slice(other__ssa_v0" in printed_main

    def test_same_dense_region_writer_reader_share_carrier_current(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def cache_write(
                self,
                cache: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
                data: pl.Tensor[[1, 64], pl.FP32],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                src: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(data, [0, 0], [1, 64], [1, 64])
                result: pl.Tensor[[16, 64], pl.FP32] = pl.tile.store(src, [3, 0], cache)
                return result

            @pl.function(type=pl.FunctionType.InCore)
            def cache_read(
                self,
                out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
                cache: pl.Tensor[[16, 64], pl.FP32],
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(cache, [3, 0], [1, 64], [1, 64])
                result: pl.Tensor[[1, 64], pl.FP32] = pl.tile.store(tile, [0, 0], out)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                cache: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
                data: pl.Tensor[[1, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 64], pl.FP32], pl.Tensor[[1, 64], pl.FP32]]:
                cache_next: pl.Tensor[[16, 64], pl.FP32] = self.cache_write(cache, data)
                result: pl.Tensor[[1, 64], pl.FP32] = self.cache_read(out, cache_next)
                return cache_next, result

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "cache_write__windowed" in printed_main
        assert "cache_read__windowed" in printed_main
        assert "cache__ssa_v0__window" in printed_main
        assert "cache_next__ssa_v0__windowed" in printed_main
        assert "cache_read__windowed(out__ssa_v0, cache_next__ssa_v0__windowed)" in printed_main
        assert "pl.tensor.slice(cache_next__ssa_v0" not in printed_main

    def test_dynamic_indexed_reader_reuses_coalesced_writer_carrier_current(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def cache_write(
                self,
                cache: pl.Out[pl.Tensor[[1024, 64], pl.FP32]],
                data: pl.Tensor[[2, 64], pl.FP32],
                slot: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[1024, 64], pl.FP32]:
                for ki, (cache_iter,) in pl.range(0, 2, init_values=(cache,)):
                    src: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(data, [ki, 0], [1, 64], [1, 64])
                    row: pl.Scalar[pl.INDEX] = slot + ki * 128
                    cache_next: pl.Tensor[[1024, 64], pl.FP32] = pl.tile.store(src, [row, 0], cache_iter)
                    cache_rv = pl.yield_(cache_next)
                return cache_rv

            @pl.function(type=pl.FunctionType.InCore)
            def cache_read(
                self,
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
                cache: pl.Tensor[[1024, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
            ) -> pl.Tensor[[4, 64], pl.FP32]:
                for sb, (out_iter,) in pl.range(0, 4, init_values=(out,)):
                    pbid_i32: pl.Scalar[pl.INT32] = pl.tensor.read(block_table, [sb])
                    pbid: pl.Scalar[pl.INDEX] = pl.cast(pbid_i32, target_type=pl.INDEX)
                    row: pl.Scalar[pl.INDEX] = pbid * 128
                    cache_tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(cache, [row, 0], [1, 64], [1, 64])
                    out_next: pl.Tensor[[4, 64], pl.FP32] = pl.tile.store(cache_tile, [sb, 0], out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                cache: pl.Out[pl.Tensor[[1024, 64], pl.FP32]],
                data: pl.Tensor[[2, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[1024, 64], pl.FP32], pl.Tensor[[4, 64], pl.FP32]]:
                cache_next: pl.Tensor[[1024, 64], pl.FP32] = self.cache_write(cache, data, 7)
                result: pl.Tensor[[4, 64], pl.FP32] = self.cache_read(out, cache_next, block_table)
                return cache_next, result

        After = _run_to_optimize_orch_tensors(Before, window_policy="all")

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "cache_write__windowed" in printed_main
        assert "__carrier_scan_" not in printed_main

    def test_auto_dynamic_indexed_reader_without_shared_carrier_current_keeps_full_parent(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def cache_write(
                self,
                cache: pl.Out[pl.Tensor[[1024, 64], pl.FP32]],
                data: pl.Tensor[[2, 64], pl.FP32],
                slot: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[1024, 64], pl.FP32]:
                for ki, (cache_iter,) in pl.range(0, 2, init_values=(cache,)):
                    src: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(data, [ki, 0], [1, 64], [1, 64])
                    row: pl.Scalar[pl.INDEX] = slot + ki * 128
                    cache_next: pl.Tensor[[1024, 64], pl.FP32] = pl.tile.store(src, [row, 0], cache_iter)
                    cache_rv = pl.yield_(cache_next)
                return cache_rv

            @pl.function(type=pl.FunctionType.InCore)
            def cache_read(
                self,
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
                cache: pl.Tensor[[1024, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
            ) -> pl.Tensor[[4, 64], pl.FP32]:
                for sb, (out_iter,) in pl.range(0, 4, init_values=(out,)):
                    pbid_i32: pl.Scalar[pl.INT32] = pl.tensor.read(block_table, [sb])
                    pbid: pl.Scalar[pl.INDEX] = pl.cast(pbid_i32, target_type=pl.INDEX)
                    row: pl.Scalar[pl.INDEX] = pbid * 128
                    cache_tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(cache, [row, 0], [1, 64], [1, 64])
                    out_next: pl.Tensor[[4, 64], pl.FP32] = pl.tile.store(cache_tile, [sb, 0], out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                cache: pl.Out[pl.Tensor[[1024, 64], pl.FP32]],
                data: pl.Tensor[[2, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[1024, 64], pl.FP32], pl.Tensor[[4, 64], pl.FP32]]:
                result: pl.Tensor[[4, 64], pl.FP32] = self.cache_read(out, cache, block_table)
                cache_next: pl.Tensor[[1024, 64], pl.FP32] = self.cache_write(cache, data, 7)
                return cache_next, result

        After = _run_to_optimize_orch_tensors(Before)

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "cache_read__windowed" not in printed_main
        assert "cache__ssa_v0__carrier_window" not in printed_main
        assert "__carrier_scan_" not in printed_main

    def test_exact_all_dynamic_indexed_reader_without_shared_carrier_current_uses_standalone_window(self):
        """v5: dynamic reader without carrier keeps full parent (no private dynamic reader)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def cache_read(
                self,
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
                cache: pl.Tensor[[1024, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
            ) -> pl.Tensor[[4, 64], pl.FP32]:
                for sb, (out_iter,) in pl.range(0, 4, init_values=(out,)):
                    pbid_i32: pl.Scalar[pl.INT32] = pl.tensor.read(block_table, [sb])
                    pbid: pl.Scalar[pl.INDEX] = pl.cast(pbid_i32, target_type=pl.INDEX)
                    row: pl.Scalar[pl.INDEX] = pbid * 128
                    cache_tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(cache, [row, 0], [1, 64], [1, 64])
                    out_next: pl.Tensor[[4, 64], pl.FP32] = pl.tile.store(cache_tile, [sb, 0], out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                cache: pl.Tensor[[1024, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
            ) -> pl.Tensor[[4, 64], pl.FP32]:
                result: pl.Tensor[[4, 64], pl.FP32] = self.cache_read(out, cache, block_table)
                return result

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        printed_main = ir.python_print(_get_function(After, "main"))
        # v5: dynamic reader without carrier keeps full parent (no private dynamic reader).
        assert "cache_read__windowed" not in printed_main

    def test_dynamic_indexed_reader_unknown_trip_count_keeps_full_parent(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def cache_read(
                self,
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
                cache: pl.Tensor[[1024, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
                valid_blocks: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[4, 64], pl.FP32]:
                for sb, (out_iter,) in pl.range(0, valid_blocks, init_values=(out,)):
                    pbid_i32: pl.Scalar[pl.INT32] = pl.tensor.read(block_table, [sb])
                    pbid: pl.Scalar[pl.INDEX] = pl.cast(pbid_i32, target_type=pl.INDEX)
                    row: pl.Scalar[pl.INDEX] = pbid * 128
                    cache_tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(cache, [row, 0], [1, 64], [1, 64])
                    out_next: pl.Tensor[[4, 64], pl.FP32] = pl.tile.store(cache_tile, [sb, 0], out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                cache: pl.Tensor[[1024, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
                valid_blocks: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
            ) -> pl.Tensor[[4, 64], pl.FP32]:
                result: pl.Tensor[[4, 64], pl.FP32] = self.cache_read(out, cache, block_table, valid_blocks)
                return result

        After = _run_to_optimize_orch_tensors(Before, window_policy="all")

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "cache_read__windowed" not in printed_main
        assert "pl.tensor.slice(cache__ssa_v0" not in printed_main

    def test_guarded_dynamic_indexed_reader_keeps_full_parent(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def cache_read(
                self,
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
                cache: pl.Tensor[[1024, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
            ) -> pl.Tensor[[4, 64], pl.FP32]:
                for sb, (out_iter,) in pl.range(0, 4, init_values=(out,)):
                    if sb < 0:
                        pbid_i32: pl.Scalar[pl.INT32] = pl.tensor.read(block_table, [sb])
                        pbid: pl.Scalar[pl.INDEX] = pl.cast(pbid_i32, target_type=pl.INDEX)
                        row: pl.Scalar[pl.INDEX] = pbid * 128
                        cache_tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(
                            cache, [row, 0], [1, 64], [1, 64]
                        )
                        out_next: pl.Tensor[[4, 64], pl.FP32] = pl.tile.store(cache_tile, [sb, 0], out_iter)
                    else:
                        zero_tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.full([1, 64], dtype=pl.FP32, value=0.0)
                        out_next: pl.Tensor[[4, 64], pl.FP32] = pl.tile.store(zero_tile, [sb, 0], out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                cache: pl.Tensor[[1024, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
            ) -> pl.Tensor[[4, 64], pl.FP32]:
                result: pl.Tensor[[4, 64], pl.FP32] = self.cache_read(out, cache, block_table)
                return result

        After = _run_to_optimize_orch_tensors(Before)

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "cache_read__windowed" not in printed_main
        assert "pl.tensor.slice(cache__ssa_v0" not in printed_main

    def test_dynamic_indexed_reader_rejects_loop_local_non_dynamic_offset(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def cache_read(
                self,
                out: pl.Out[pl.Tensor[[4, 1], pl.FP32]],
                cache: pl.Tensor[[1024, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
            ) -> pl.Tensor[[4, 1], pl.FP32]:
                for sb, (out_iter,) in pl.range(0, 4, init_values=(out,)):
                    pbid_i32: pl.Scalar[pl.INT32] = pl.tensor.read(block_table, [sb])
                    pbid: pl.Scalar[pl.INDEX] = pl.cast(pbid_i32, target_type=pl.INDEX)
                    row: pl.Scalar[pl.INDEX] = pbid * 128
                    cache_tile: pl.Tile[[1, 1], pl.FP32] = pl.tile.load(cache, [row, sb], [1, 1], [1, 1])
                    out_next: pl.Tensor[[4, 1], pl.FP32] = pl.tile.store(cache_tile, [sb, 0], out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                cache: pl.Tensor[[1024, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
                out: pl.Out[pl.Tensor[[4, 1], pl.FP32]],
            ) -> pl.Tensor[[4, 1], pl.FP32]:
                result: pl.Tensor[[4, 1], pl.FP32] = self.cache_read(out, cache, block_table)
                return result

        After = _run_to_optimize_orch_tensors(Before)

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "cache_read__windowed" not in printed_main
        assert "pl.tensor.slice(cache__ssa_v0" not in printed_main

    def test_windowable_writer_blocked_by_unwindowable_full_out_sibling(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def init_row(
                self,
                out: pl.Out[pl.Tensor[[4, 16], pl.INT32]],
                row: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[4, 16], pl.INT32]:
                invalid: pl.Tile[[1, 16], pl.INT32] = pl.tile.full([1, 16], dtype=pl.INT32, value=-1)
                result: pl.Tensor[[4, 16], pl.INT32] = pl.tile.store(invalid, [row, 0], out)
                return result

            @pl.function(type=pl.FunctionType.InCore)
            def write_prefix(
                self,
                out: pl.Out[pl.Tensor[[4, 16], pl.INT32]],
                row: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[4, 16], pl.INT32]:
                value: pl.Scalar[pl.INT32] = pl.cast(row, target_type=pl.INT32)
                pl.write(out, [row, 0], value)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                out: pl.Out[pl.Tensor[[4, 16], pl.INT32]],
            ) -> pl.Tensor[[4, 16], pl.INT32]:
                for row, (out_iter,) in pl.parallel(4, init_values=(out,)):
                    init_next: pl.Tensor[[4, 16], pl.INT32] = self.init_row(out_iter, row)
                    write_next: pl.Tensor[[4, 16], pl.INT32] = self.write_prefix(init_next, row)
                    out_rv = pl.yield_(write_next)
                return out_rv

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        assert After.get_function("init_row__windowed") is None
        assert After.get_function("write_prefix__windowed") is None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "init_row__windowed" not in printed_main
        assert "write_prefix__windowed" not in printed_main
        assert "pl.tensor.slice(out" not in printed_main

    def test_loop_return_alias_full_writer_blocks_prior_windowable_sibling(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def init_row(
                self,
                out: pl.Out[pl.Tensor[[4, 16], pl.INT32]],
                row: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[4, 16], pl.INT32]:
                invalid: pl.Tile[[1, 16], pl.INT32] = pl.tile.full([1, 16], dtype=pl.INT32, value=-1)
                result: pl.Tensor[[4, 16], pl.INT32] = pl.tile.store(invalid, [row, 0], out)
                return result

            @pl.function(type=pl.FunctionType.InCore)
            def full_overwrite(
                self,
                out: pl.Out[pl.Tensor[[4, 16], pl.INT32]],
            ) -> pl.Tensor[[4, 16], pl.INT32]:
                value: pl.Tile[[4, 16], pl.INT32] = pl.tile.full([4, 16], dtype=pl.INT32, value=1)
                result: pl.Tensor[[4, 16], pl.INT32] = pl.tile.store(value, [0, 0], out)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                out: pl.Out[pl.Tensor[[4, 16], pl.INT32]],
            ) -> pl.Tensor[[4, 16], pl.INT32]:
                init_next: pl.Tensor[[4, 16], pl.INT32] = self.init_row(out, 0)
                for _, (out_iter,) in pl.range(1, init_values=(init_next,)):
                    out_rv = pl.yield_(out_iter)
                result: pl.Tensor[[4, 16], pl.INT32] = self.full_overwrite(out_rv)
                return result

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        assert After.get_function("init_row__windowed") is None
        assert After.get_function("full_overwrite__windowed") is None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "init_row__windowed" not in printed_main
        assert "full_overwrite__windowed" not in printed_main
        assert "pl.tensor.slice(out" not in printed_main

    def test_windowable_writer_blocked_by_callsite_output_sibling(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def init_row(
                self,
                out: pl.Out[pl.Tensor[[4, 16], pl.INT32]],
                row: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[4, 16], pl.INT32]:
                invalid: pl.Tile[[1, 16], pl.INT32] = pl.tile.full([1, 16], dtype=pl.INT32, value=-1)
                result: pl.Tensor[[4, 16], pl.INT32] = pl.tile.store(invalid, [row, 0], out)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                out: pl.Out[pl.Tensor[[4, 16], pl.INT32]],
            ) -> pl.Tensor[[4, 16], pl.INT32]:
                for row, (out_iter,) in pl.parallel(4, init_values=(out,)):
                    init_next: pl.Tensor[[4, 16], pl.INT32] = self.init_row(out_iter, row)
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="write_prefix"):
                        value: pl.Scalar[pl.INT32] = pl.cast(row, target_type=pl.INT32)
                        write_next: pl.Tensor[[4, 16], pl.INT32] = pl.write(init_next, [row, 0], value)
                    out_rv = pl.yield_(write_next)
                return out_rv

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        assert After.get_function("init_row__windowed") is None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "init_row__windowed" not in printed_main
        assert "write_prefix__windowed" not in printed_main
        assert "pl.tensor.slice(out" not in printed_main

    def test_same_region_inout_sibling_allows_prior_windowable_writer(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def init_row(
                self,
                out: pl.Out[pl.Tensor[[4, 16], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[4, 16], pl.FP32]:
                tile: pl.Tile[[1, 4], pl.FP32] = pl.tile.full([1, 4], dtype=pl.FP32, value=1.0)
                result: pl.Tensor[[4, 16], pl.FP32] = pl.tile.store(tile, [row, 0], out)
                return result

            @pl.function(type=pl.FunctionType.InCore)
            def add_residual(
                self,
                resid: pl.InOut[pl.Tensor[[4, 16], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[4, 16], pl.FP32]:
                prev: pl.Tile[[1, 4], pl.FP32] = pl.tile.load(resid, [row, 0], [1, 4])
                bias: pl.Tile[[1, 4], pl.FP32] = pl.tile.full([1, 4], dtype=pl.FP32, value=2.0)
                next_tile: pl.Tile[[1, 4], pl.FP32] = pl.tile.add(prev, bias)
                result: pl.Tensor[[4, 16], pl.FP32] = pl.tile.store(next_tile, [row, 0], resid)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                out: pl.Out[pl.Tensor[[4, 16], pl.FP32]],
            ) -> pl.Tensor[[4, 16], pl.FP32]:
                for row, (out_iter,) in pl.parallel(4, init_values=(out,)):
                    init_next: pl.Tensor[[4, 16], pl.FP32] = self.init_row(out_iter, row)
                    residual_next: pl.Tensor[[4, 16], pl.FP32] = self.add_residual(init_next, row)
                    out_rv = pl.yield_(residual_next)
                return out_rv

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        printed_main = ir.python_print(_get_function(After, "main"))
        printed_residual = ir.python_print(_get_function(After, "add_residual__windowed"))
        assert "init_row__windowed" in printed_main
        assert "add_residual__windowed" in printed_main
        assert "pl.tensor.slice(init_next" in printed_main
        assert "pl.tile.load(resid" in printed_residual
        assert "[0, 0]" in printed_residual
        assert "pl.tile.store(next_tile" in printed_residual

    def test_aggregate_loop_direct_out_return_is_windowed(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def chunk_writer(
                self,
                out: pl.Out[pl.Tensor[[2, 16], pl.FP32]],
                start: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[2, 16], pl.FP32]:
                for col, (out_iter,) in pl.range(start, start + 2, init_values=(out,)):
                    offset: pl.Scalar[pl.INDEX] = col * 4
                    tile: pl.Tile[[2, 4], pl.FP32] = pl.tile.full([2, 4], dtype=pl.FP32, value=1.0)
                    out_next: pl.Tensor[[2, 16], pl.FP32] = pl.tile.store(tile, [0, offset], out_iter)
                    out_rv = pl.yield_(out_next)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                out: pl.Out[pl.Tensor[[2, 16], pl.FP32]],
            ) -> pl.Tensor[[2, 16], pl.FP32]:
                start: pl.Scalar[pl.INDEX] = 1
                result: pl.Tensor[[2, 16], pl.FP32] = self.chunk_writer(out, start)
                return result

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "chunk_writer__windowed" in printed_main
        assert "pl.tensor.slice(out" in printed_main

    def test_aggregate_input_window_loop_rewrites_qk_norm_shape(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def qk_norm_like(
                self,
                q_out: pl.Out[pl.Tensor[[16, 5120], pl.FP32]],
                k_out: pl.Out[pl.Tensor[[16, 1024], pl.FP32]],
                q_proj: pl.Tensor[[16, 5120], pl.FP32],
                k_proj: pl.Tensor[[16, 1024], pl.FP32],
                row: pl.Scalar[pl.INDEX],
            ) -> tuple[pl.Tensor[[16, 5120], pl.FP32], pl.Tensor[[16, 1024], pl.FP32]]:
                for h, (q_iter, k_iter) in pl.range(8, init_values=(q_out, k_out)):
                    q0: pl.Scalar[pl.INDEX] = h * 640
                    k0: pl.Scalar[pl.INDEX] = h * 128
                    q_tile: pl.Tile[[16, 640], pl.FP32] = pl.load(q_proj, [row, q0], [16, 640])
                    k_tile: pl.Tile[[16, 128], pl.FP32] = pl.load(k_proj, [row, k0], [16, 128])
                    q_next: pl.Tensor[[16, 5120], pl.FP32] = pl.store(q_tile, [row, q0], q_iter)
                    k_next: pl.Tensor[[16, 1024], pl.FP32] = pl.store(k_tile, [row, k0], k_iter)
                    q_rv, k_rv = pl.yield_(q_next, k_next)
                return q_rv, k_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                q_proj: pl.Tensor[[16, 5120], pl.FP32],
                k_proj: pl.Tensor[[16, 1024], pl.FP32],
                q_out: pl.Out[pl.Tensor[[16, 5120], pl.FP32]],
                k_out: pl.Out[pl.Tensor[[16, 1024], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 5120], pl.FP32], pl.Tensor[[16, 1024], pl.FP32]]:
                row: pl.Scalar[pl.INDEX] = 0
                return self.qk_norm_like(q_out, k_out, q_proj, k_proj, row)

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        assert After.get_function("qk_norm_like__windowed") is not None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "pl.tensor.slice(q_proj" in printed_main
        assert "pl.tensor.slice(k_proj" in printed_main
        assert "qk_norm_like__windowed" in printed_main
        assert "q_proj__ssa_v0__window" in printed_main
        assert "k_proj__ssa_v0__window" in printed_main

        printed_windowed = ir.python_print(_get_function(After, "qk_norm_like__windowed"))
        assert (
            "q_proj__ssa_v0: pl.Tensor[[16, 5120], pl.FP32, pl.TensorView(stride=[5120, 1]"
            in printed_windowed
        )
        assert (
            "k_proj__ssa_v0: pl.Tensor[[16, 1024], pl.FP32, pl.TensorView(stride=[1024, 1]"
            in printed_windowed
        )
        assert "pl.tile.load(q_proj__ssa_v0, [0, q0__ssa_v0]" in printed_windowed
        assert "pl.tile.load(k_proj__ssa_v0, [0, k0__ssa_v0]" in printed_windowed

    def test_aggregate_input_window_loop_uses_visible_loop_init_parent(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def qk_norm_like(
                self,
                q_out: pl.Out[pl.Tensor[[16, 5120], pl.FP32]],
                k_out: pl.Out[pl.Tensor[[16, 1024], pl.FP32]],
                q_proj: pl.Tensor[[16, 5120], pl.FP32],
                k_proj: pl.Tensor[[16, 1024], pl.FP32],
                row: pl.Scalar[pl.INDEX],
            ) -> tuple[pl.Tensor[[16, 5120], pl.FP32], pl.Tensor[[16, 1024], pl.FP32]]:
                for h, (q_iter, k_iter) in pl.range(8, init_values=(q_out, k_out)):
                    q0: pl.Scalar[pl.INDEX] = h * 640
                    k0: pl.Scalar[pl.INDEX] = h * 128
                    q_tile: pl.Tile[[16, 640], pl.FP32] = pl.load(q_proj, [row, q0], [16, 640])
                    k_tile: pl.Tile[[16, 128], pl.FP32] = pl.load(k_proj, [row, k0], [16, 128])
                    q_next: pl.Tensor[[16, 5120], pl.FP32] = pl.store(q_tile, [row, q0], q_iter)
                    k_next: pl.Tensor[[16, 1024], pl.FP32] = pl.store(k_tile, [row, k0], k_iter)
                    q_norm, k_norm = pl.yield_(q_next, k_next)
                return q_norm, k_norm

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                q_proj: pl.Tensor[[16, 5120], pl.FP32],
                k_proj: pl.Tensor[[16, 1024], pl.FP32],
                q_out: pl.Out[pl.Tensor[[16, 5120], pl.FP32]],
                k_out: pl.Out[pl.Tensor[[16, 1024], pl.FP32]],
            ) -> tuple[pl.Tensor[[16, 5120], pl.FP32], pl.Tensor[[16, 1024], pl.FP32]]:
                for i, (q_iter, k_iter) in pl.range(1, init_values=(q_proj, k_proj)):
                    q_rv, k_rv = pl.yield_(q_iter, k_iter)
                row: pl.Scalar[pl.INDEX] = 0
                return self.qk_norm_like(q_out, k_out, q_rv, k_rv, row)

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "qk_norm_like__windowed" in printed_main
        assert "pl.tensor.slice(q_proj" in printed_main
        assert "pl.tensor.slice(k_proj" in printed_main
        assert "pl.tensor.slice(q_rv" not in printed_main
        assert "pl.tensor.slice(k_rv" not in printed_main

    def test_aggregate_output_window_combines_multiple_stores_per_iteration(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def rope_like(
                self,
                q_out: pl.Out[pl.Tensor[[128, 64], pl.BF16]],
                q_proj: pl.Tensor[[16, 256], pl.FP32],
                row: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 64], pl.BF16]:
                for h, (q_iter,) in pl.range(4, init_values=(q_out,)):
                    q0: pl.Scalar[pl.INDEX] = h * 64
                    out_row: pl.Scalar[pl.INDEX] = row * 8 + h * 2
                    q_tile: pl.Tile[[1, 64], pl.FP32] = pl.load(q_proj, [row, q0], [1, 64])
                    q_bf16: pl.Tile[[1, 64], pl.BF16] = pl.tile.cast(q_tile, target_type=pl.BF16)
                    q_next: pl.Tensor[[128, 64], pl.BF16] = pl.store(q_bf16, [out_row, 0], q_iter)
                    zero: pl.Tile[[1, 64], pl.BF16] = pl.tile.full([1, 64], dtype=pl.BF16, value=0.0)
                    q_next_2: pl.Tensor[[128, 64], pl.BF16] = pl.store(zero, [out_row + 1, 0], q_next)
                    q_rv = pl.yield_(q_next_2)
                return q_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                q_proj: pl.Tensor[[16, 256], pl.FP32],
                q_out: pl.Out[pl.Tensor[[128, 64], pl.BF16]],
            ) -> pl.Tensor[[128, 64], pl.BF16]:
                row: pl.Scalar[pl.INDEX] = 3
                return self.rope_like(q_out, q_proj, row)

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        assert After.get_function("rope_like__windowed") is not None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "rope_like__windowed" in printed_main
        assert "pl.tensor.slice(q_out" in printed_main
        assert "pl.tensor.assemble(q_out" in printed_main

        printed_windowed = ir.python_print(_get_function(After, "rope_like__windowed"))
        assert "pl.Tensor[[8, 64], pl.BF16" in printed_windowed
        assert "pl.TensorView(stride=[64, 1]" in printed_windowed
        assert "q_proj__ssa_v0: pl.Tensor[[1, 256], pl.FP32" in printed_windowed
        assert "pl.tile.load(q_proj__ssa_v0, [0, q0__ssa_v0]" in printed_windowed
        assert (
            "pl.tile.store(q_bf16__ssa_v0, [out_row__ssa_v0 - row__ssa_v0 * 8, 0], q_iter)"
            in printed_windowed
        )
        assert "[out_row__ssa_v0 - row__ssa_v0 * 8 + 1, 0], q_next__ssa_v0" in printed_windowed

    def test_aggregate_output_diagonal_writes_stay_baseline(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def diagonal_like(
                self,
                out: pl.Out[pl.Tensor[[4, 4], pl.FP32]],
            ) -> pl.Tensor[[4, 4], pl.FP32]:
                for i, (out_iter,) in pl.range(4, init_values=(out,)):
                    tile: pl.Tile[[1, 1], pl.FP32] = pl.tile.full([1, 1], dtype=pl.FP32, value=1.0)
                    out_next: pl.Tensor[[4, 4], pl.FP32] = pl.store(tile, [i, i], out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                out: pl.Out[pl.Tensor[[4, 4], pl.FP32]],
            ) -> pl.Tensor[[4, 4], pl.FP32]:
                return self.diagonal_like(out)

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        assert After.get_function("diagonal_like__windowed") is None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "diagonal_like__windowed" not in printed_main

    def test_aggregate_output_overlap_and_hole_stays_baseline(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def overlap_hole_like(
                self,
                out: pl.Out[pl.Tensor[[2, 2], pl.FP32]],
            ) -> pl.Tensor[[2, 2], pl.FP32]:
                for i, (out_iter,) in pl.range(1, init_values=(out,)):
                    row_tile: pl.Tile[[1, 2], pl.FP32] = pl.tile.full([1, 2], dtype=pl.FP32, value=1.0)
                    col_tile: pl.Tile[[2, 1], pl.FP32] = pl.tile.full([2, 1], dtype=pl.FP32, value=2.0)
                    out_next: pl.Tensor[[2, 2], pl.FP32] = pl.store(row_tile, [0, 0], out_iter)
                    out_next_2: pl.Tensor[[2, 2], pl.FP32] = pl.store(col_tile, [0, 0], out_next)
                    out_rv = pl.yield_(out_next_2)
                return out_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                out: pl.Out[pl.Tensor[[2, 2], pl.FP32]],
            ) -> pl.Tensor[[2, 2], pl.FP32]:
                return self.overlap_hole_like(out)

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        assert After.get_function("overlap_hole_like__windowed") is None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "overlap_hole_like__windowed" not in printed_main

    def test_aggregate_output_windows_direct_outputs_when_sibling_has_nested_loop(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def rope_like(
                self,
                all_q: pl.Out[pl.Tensor[[128, 128], pl.BF16]],
                k_cache: pl.Out[pl.Tensor[[1024, 128], pl.BF16]],
                v_cache: pl.Out[pl.Tensor[[1024, 128], pl.BF16]],
                q_proj: pl.Tensor[[1, 5120], pl.FP32],
                ki_chunk: pl.Scalar[pl.INDEX],
                slot_block: pl.Scalar[pl.INDEX],
                slot_offset: pl.Scalar[pl.INDEX],
            ) -> tuple[
                pl.Tensor[[128, 128], pl.BF16],
                pl.Tensor[[1024, 128], pl.BF16],
                pl.Tensor[[1024, 128], pl.BF16],
            ]:
                for ki, (all_q_iter, k_iter, v_iter) in pl.range(
                    ki_chunk, ki_chunk + 8, init_values=(all_q, k_cache, v_cache)
                ):
                    cache_row: pl.Scalar[pl.INDEX] = (slot_block * 8 + ki) * 128 + slot_offset
                    k_lo: pl.Tile[[1, 64], pl.BF16] = pl.tile.full([1, 64], dtype=pl.BF16, value=1.0)
                    k_hi: pl.Tile[[1, 64], pl.BF16] = pl.tile.full([1, 64], dtype=pl.BF16, value=2.0)
                    v_tile: pl.Tile[[1, 128], pl.BF16] = pl.tile.full([1, 128], dtype=pl.BF16, value=3.0)
                    k_next: pl.Tensor[[1024, 128], pl.BF16] = pl.store(k_lo, [cache_row, 0], k_iter)
                    k_next_2: pl.Tensor[[1024, 128], pl.BF16] = pl.store(k_hi, [cache_row, 64], k_next)
                    v_next: pl.Tensor[[1024, 128], pl.BF16] = pl.store(v_tile, [cache_row, 0], v_iter)
                    for qi, (all_q_inner,) in pl.range(5, init_values=(all_q_iter,)):
                        q_col: pl.Scalar[pl.INDEX] = (ki * 5 + qi) * 128
                        q_tile_fp32: pl.Tile[[1, 128], pl.FP32] = pl.tile.load(
                            q_proj, [0, q_col], [1, 128], [1, 128]
                        )
                        q_tile: pl.Tile[[1, 128], pl.BF16] = pl.tile.cast(q_tile_fp32, target_type=pl.BF16)
                        all_q_next: pl.Tensor[[128, 128], pl.BF16] = pl.store(
                            q_tile, [ki * 16 + qi, 0], all_q_inner
                        )
                        all_q_rv = pl.yield_(all_q_next)
                    all_q_out, k_out, v_out = pl.yield_(all_q_rv, k_next_2, v_next)
                return all_q_out, k_out, v_out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                all_q: pl.Out[pl.Tensor[[128, 128], pl.BF16]],
                k_cache: pl.Out[pl.Tensor[[1024, 128], pl.BF16]],
                v_cache: pl.Out[pl.Tensor[[1024, 128], pl.BF16]],
                q_proj: pl.Tensor[[1, 5120], pl.FP32],
            ) -> tuple[
                pl.Tensor[[128, 128], pl.BF16],
                pl.Tensor[[1024, 128], pl.BF16],
                pl.Tensor[[1024, 128], pl.BF16],
            ]:
                return self.rope_like(all_q, k_cache, v_cache, q_proj, 0, 0, 0)

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "rope_like__windowed" in printed_main
        assert "pl.tensor.slice(k_cache" in printed_main
        assert "pl.tensor.slice(v_cache" in printed_main
        assert "pl.tensor.slice(q_proj" in printed_main
        assert "pl.tensor.slice(all_q" in printed_main
        assert "all_q__ssa_v0__window_7" in printed_main

        printed_callee = ir.python_print(_get_function(After, "rope_like__windowed"))
        assert "all_q__ssa_v0: pl.Out[pl.Tensor[[5, 128]" in printed_callee
        assert "all_q__ssa_v0_piece7: pl.Out[pl.Tensor[[5, 128]" in printed_callee

    def test_output_window_disjointness_allows_dynamic_trip_count(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def write_chunk(
                self,
                out: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[16, 256], pl.FP32]:
                tile: pl.Tile[[16, 64], pl.FP32] = pl.tile.full([16, 64], dtype=pl.FP32, value=1.0)
                return pl.store(tile, [0, col], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                out: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
                blocks: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[16, 256], pl.FP32]:
                for b, (out_iter,) in pl.range(blocks, init_values=(out,)):
                    col: pl.Scalar[pl.INDEX] = b * 64
                    out_next: pl.Tensor[[16, 256], pl.FP32] = self.write_chunk(out_iter, col)
                    out_rv = pl.yield_(out_next)
                return out_rv

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "write_chunk__windowed" in printed_main
        assert "pl.tensor.slice(out_iter, [16, 64], [0, col" in printed_main

    def test_output_window_disjointness_allows_nested_dynamic_outer_loop(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def write_chunk(
                self,
                src: pl.Tensor[[16, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[64, 256], pl.FP32]:
                tile: pl.Tile[[16, 64], pl.FP32] = pl.load(src, [0, 0], [16, 64])
                return pl.store(tile, [row, col], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                src: pl.Tensor[[16, 128], pl.FP32],
                offsets: pl.Tensor[[1], pl.INT32],
                blocks: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[64, 256], pl.FP32]:
                out0: pl.Tensor[[64, 256], pl.FP32] = pl.tensor.create(
                    [64, 256], dtype=pl.FP32, layout=pl.TensorLayout.ND
                )
                raw_base: pl.Scalar[pl.INT32] = pl.tensor.read(offsets, [0])
                base: pl.Scalar[pl.INDEX] = pl.cast(raw_base, pl.INDEX)
                for p, (out_outer,) in pl.range(blocks, init_values=(out0,)):
                    row: pl.Scalar[pl.INDEX] = base + p * 16
                    for h, (out_inner,) in pl.range(4, init_values=(out_outer,)):
                        col: pl.Scalar[pl.INDEX] = h * 64
                        out_next: pl.Tensor[[64, 256], pl.FP32] = self.write_chunk(src, out_inner, row, col)
                        out_inner_rv = pl.yield_(out_next)
                    out_outer_rv = pl.yield_(out_inner_rv)
                return out_outer_rv

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "write_chunk__windowed" in printed_main
        assert "pl.tensor.slice(out_inner, [16, 64], [row" in printed_main

    def test_output_window_disjointness_allows_nested_partition_reduction_loop(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_split(
                self,
                lhs: pl.Tensor[[16, 128], pl.FP32],
                weight: pl.Tensor[[128, 128], pl.FP32],
                k_off: pl.Scalar[pl.INDEX],
                n_off: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                lhs_tile: pl.Tile[[16, 64], pl.FP32] = pl.load(lhs, [0, k_off], [16, 64])
                weight_tile: pl.Tile[[64, 64], pl.FP32] = pl.load(weight, [k_off, n_off], [64, 64])
                acc_tile: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(lhs_tile, weight_tile)
                return pl.store(acc_tile, [0, n_off], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                lhs: pl.Tensor[[16, 128], pl.FP32],
                weight: pl.Tensor[[128, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                for n, (out_outer,) in pl.range(2, init_values=(out,)):
                    n_off: pl.Scalar[pl.INDEX] = n * 64
                    for k, (out_inner,) in pl.range(2, init_values=(out_outer,)):
                        k_off: pl.Scalar[pl.INDEX] = k * 64
                        out_next: pl.Tensor[[16, 128], pl.FP32] = self.matmul_split(
                            lhs, weight, k_off, n_off, out_inner
                        )
                        out_inner_rv = pl.yield_(out_next)
                    out_outer_rv = pl.yield_(out_inner_rv)
                return out_outer_rv

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "matmul_split__windowed" in printed_main
        assert "pl.tensor.slice(out_inner, [16, 64], [0, n_off" in printed_main
        assert "pl.tensor.slice(lhs" in printed_main
        assert "pl.tensor.slice(weight" in printed_main

    def test_output_window_disjointness_rejects_overlapping_inner_partition_loop(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def write_overlap(
                self,
                src: pl.Tensor[[16, 64], pl.FP32],
                col: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                tile: pl.Tile[[16, 64], pl.FP32] = pl.load(src, [0, 0], [16, 64])
                return pl.store(tile, [0, col], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                src: pl.Tensor[[16, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                for k, (out_iter,) in pl.range(2, init_values=(out,)):
                    col: pl.Scalar[pl.INDEX] = k * 32
                    out_next: pl.Tensor[[16, 128], pl.FP32] = self.write_overlap(src, col, out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "write_overlap__windowed" not in printed_main

    def test_input_only_windows_aggregate_reads_when_outputs_are_full_local(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_like(
                self,
                lhs: pl.Tensor[[16, 2176], pl.FP32],
                weight: pl.Tensor[[2176, 256], pl.FP32],
                col: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                acc0: pl.Tile[[16, 64], pl.FP32] = pl.tile.full([16, 64], dtype=pl.FP32, value=0.0)
                for k, (acc_iter,) in pl.range(136, init_values=(acc0,)):
                    row: pl.Scalar[pl.INDEX] = k * 16
                    lhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(lhs, [0, row], [16, 16])
                    w_tile: pl.Tile[[16, 64], pl.FP32] = pl.load(weight, [row, col], [16, 64])
                    prod: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(lhs_tile, w_tile)
                    acc_next: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(acc_iter, prod)
                    acc_rv = pl.yield_(acc_next)
                return pl.store(acc_rv, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                lhs: pl.Tensor[[16, 2176], pl.FP32],
                weight: pl.Tensor[[2176, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                return self.matmul_like(lhs, weight, 128, out)

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "matmul_like__windowed" in printed_main
        assert "pl.tensor.slice(weight" in printed_main
        assert "[2176, 64]" in printed_main
        assert "[0, 128]" in printed_main
        assert "pl.tensor.slice(lhs" not in printed_main

    def test_input_only_windows_transposed_aggregate_reads(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_like(
                self,
                lhs: pl.Tensor[[16, 5120], pl.FP32],
                weight: pl.Tensor[[152064, 5120], pl.FP32],
                row: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                acc0: pl.Tile[[16, 64], pl.FP32] = pl.tile.full([16, 64], dtype=pl.FP32, value=0.0)
                for k, (acc_iter,) in pl.range(40, init_values=(acc0,)):
                    col: pl.Scalar[pl.INDEX] = k * 128
                    lhs_tile: pl.Tile[[16, 128], pl.FP32] = pl.load(lhs, [0, col], [16, 128])
                    w_tile: pl.Tile[[128, 64], pl.FP32] = pl.tile.load(
                        weight, [row, col], [64, 128], [64, 128], target_memory=pl.Mem.Mat, transpose=True
                    )
                    prod: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(lhs_tile, w_tile)
                    acc_next: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(acc_iter, prod)
                    acc_rv = pl.yield_(acc_next)
                return pl.store(acc_rv, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                lhs: pl.Tensor[[16, 5120], pl.FP32],
                weight: pl.Tensor[[152064, 5120], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                return self.matmul_like(lhs, weight, 128, out)

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        printed_main = ir.python_print(_get_function(After, "main"))
        printed_windowed = ir.python_print(_get_function(After, "matmul_like__windowed"))
        assert "matmul_like__windowed" in printed_main
        assert "pl.tensor.slice(weight" in printed_main
        assert "[64, 5120]" in printed_main
        assert "[128, 0]" in printed_main
        assert "pl.tensor.slice(lhs" not in printed_main
        assert "transpose=True" in printed_windowed

    def test_aggregate_output_preserves_existing_pure_input_window(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def aggregate_with_header(
                self,
                out: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
                data: pl.Tensor[[16, 256], pl.FP32],
                header: pl.Tensor[[16, 256], pl.FP32],
                row: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[16, 256], pl.FP32]:
                header_tile: pl.Tile[[16, 64], pl.FP32] = pl.load(header, [row, 0], [16, 64])
                for h, (out_iter,) in pl.range(4, init_values=(out,)):
                    col: pl.Scalar[pl.INDEX] = h * 64
                    data_tile: pl.Tile[[16, 64], pl.FP32] = pl.load(data, [row, col], [16, 64])
                    mixed: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(data_tile, header_tile)
                    out_next: pl.Tensor[[16, 256], pl.FP32] = pl.store(mixed, [row, col], out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[16, 256], pl.FP32],
                header: pl.Tensor[[16, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
            ) -> pl.Tensor[[16, 256], pl.FP32]:
                row: pl.Scalar[pl.INDEX] = 0
                return self.aggregate_with_header(out, data, header, row)

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def aggregate_with_header(
                self,
                out: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
                data: pl.Tensor[[16, 256], pl.FP32],
                header: pl.Tensor[[16, 256], pl.FP32],
                row: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[16, 256], pl.FP32]:
                header_tile: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    header, [row, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                for h, (out_iter,) in pl.range(4, init_values=(out,)):
                    col: pl.Scalar[pl.INDEX] = h * 64
                    data_tile: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        data, [row, col], [16, 64], [16, 64], target_memory=pl.Mem.Vec, transpose=False
                    )
                    mixed: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(data_tile, header_tile)
                    out_next: pl.Tensor[[16, 256], pl.FP32] = pl.tile.store(mixed, [row, col], out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

            @pl.function(type=pl.FunctionType.InCore)
            def aggregate_with_header__windowed(
                self,
                out: pl.Out[
                    pl.Tensor[[16, 256], pl.FP32, pl.TensorView(stride=[256, 1], layout=pl.TensorLayout.ND)]
                ],
                data: pl.Tensor[
                    [16, 256], pl.FP32, pl.TensorView(stride=[256, 1], layout=pl.TensorLayout.ND)
                ],
                header: pl.Tensor[
                    [16, 64], pl.FP32, pl.TensorView(stride=[256, 1], layout=pl.TensorLayout.ND)
                ],
                row: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[16, 256], pl.FP32, pl.TensorView(stride=[256, 1], layout=pl.TensorLayout.ND)]:
                header_tile: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    header, [0, 0], [16, 64], [16, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                for h, (out_iter,) in pl.range(4, init_values=(out,)):
                    col: pl.Scalar[pl.INDEX] = h * 64
                    data_tile: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                        data, [0, col], [16, 64], [16, 64], target_memory=pl.Mem.Vec, transpose=False
                    )
                    mixed: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(data_tile, header_tile)
                    out_next: pl.Tensor[
                        [16, 256], pl.FP32, pl.TensorView(stride=[256, 1], layout=pl.TensorLayout.ND)
                    ] = pl.tile.store(mixed, [0, col], out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[16, 256], pl.FP32],
                header: pl.Tensor[[16, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 256], pl.FP32]],
            ) -> pl.Tensor[[16, 256], pl.FP32]:
                header__window: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(header, [16, 64], [0, 0])
                data__window: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(data, [16, 256], [0, 0])
                out__window: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(out, [16, 256], [0, 0])
                result__windowed: pl.Tensor[
                    [16, 256], pl.FP32, pl.TensorView(stride=[256, 1], layout=pl.TensorLayout.ND)
                ] = self.aggregate_with_header__windowed(out__window, data__window, header__window, 0)
                result: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.assemble(out, result__windowed, [0, 0])
                return result

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)
        printed_main = ir.python_print(_get_function(After, "main"))
        printed_windowed = ir.python_print(_get_function(After, "aggregate_with_header__windowed"))
        assert "aggregate_with_header__windowed" in printed_main
        assert "pl.tensor.slice(header" in printed_main
        assert "[16, 64]" in printed_main
        assert "pl.tensor.slice(data" in printed_main
        assert "[16, 256]" in printed_main
        assert "pl.tile.load(header" in printed_windowed
        assert "[0, 0]" in printed_windowed

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
                data: pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)],
                row_offset: pl.Scalar[pl.INDEX],
                bias: pl.Scalar[pl.FP32],
                out: pl.Out[
                    pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)]
                ],
            ) -> pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)]:
                tile: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    data, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
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
                data__window = pl.tensor.slice(data, [64, 64], [64, 0])
                out__window = pl.tensor.slice(out, [64, 64], [64, 0])
                out_next__windowed: pl.Tensor[
                    [64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)
                ] = self.kernel_stripe__windowed(data__window, 64, 1.0, out__window)
                out_next = pl.tensor.assemble(out, out_next__windowed, [64, 0])
                return out_next

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)
        ir.assert_structural_equal(After, Expected)

    def test_output_window_uses_visible_loop_init_parent(self):
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
                for i, (out_iter,) in pl.range(1, init_values=(out,)):
                    out_rv = pl.yield_(out_iter)
                row: pl.Scalar[pl.INDEX] = 64
                result: pl.Tensor[[256, 64], pl.FP32] = self.kernel_stripe(data, row, out_rv)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_stripe(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    data, [row_offset, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                ret = pl.tile.store(tile, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def kernel_stripe__windowed(
                self,
                data: pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[
                    pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)]
                ],
            ) -> pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)]:
                tile: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    data, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                ret: pl.Tensor[
                    [64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)
                ] = pl.tile.store(tile, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                data__window = pl.tensor.slice(data, [64, 64], [64, 0])
                out__window = pl.tensor.slice(out, [64, 64], [64, 0])
                result__windowed: pl.Tensor[
                    [64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)
                ] = self.kernel_stripe__windowed(data__window, 64, out__window)
                result = pl.tensor.assemble(out, result__windowed, [64, 0])
                return result

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)
        ir.assert_structural_equal(After, Expected)

    def test_sibling_writers_to_same_parent_can_window_with_runtime_overlap(self):
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
                first: pl.Tensor[[256, 64], pl.FP32] = self.kernel_stripe(data, 0, out)
                second: pl.Tensor[[256, 64], pl.FP32] = self.kernel_stripe(data, 32, first)
                return second

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        assert After.get_function("kernel_stripe__windowed") is not None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "kernel_stripe__windowed" in printed_main
        assert "pl.tensor.slice(out" in printed_main
        assert "pl.tensor.slice(first" in printed_main

    def test_nested_sibling_writer_to_same_parent_can_window_with_runtime_overlap(self):
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
                first: pl.Tensor[[256, 64], pl.FP32] = self.kernel_stripe(data, 0, out)
                for i, (first_iter,) in pl.range(1, init_values=(first,)):
                    second: pl.Tensor[[256, 64], pl.FP32] = self.kernel_stripe(data, 32, first_iter)
                    second_rv = pl.yield_(second)
                return second_rv

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        assert After.get_function("kernel_stripe__windowed") is not None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "kernel_stripe__windowed" in printed_main
        assert "pl.tensor.slice(out" in printed_main

    def test_if_nested_sibling_writer_to_same_parent_can_window_with_runtime_overlap(self):
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
                flag: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                first: pl.Tensor[[256, 64], pl.FP32] = self.kernel_stripe(data, 0, out)
                if flag == 0:
                    second: pl.Tensor[[256, 64], pl.FP32] = self.kernel_stripe(data, 32, first)
                else:
                    second: pl.Tensor[[256, 64], pl.FP32] = self.kernel_stripe(data, 64, first)
                return second

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        assert After.get_function("kernel_stripe__windowed") is not None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "kernel_stripe__windowed" in printed_main
        assert "pl.tensor.slice(out" in printed_main
        assert "pl.tensor.slice(first" in printed_main

    def test_single_nested_writer_can_still_window(self):
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
                for i, (out_iter,) in pl.range(1, init_values=(out,)):
                    result: pl.Tensor[[256, 64], pl.FP32] = self.kernel_stripe(data, 64, out_iter)
                    result_rv = pl.yield_(result)
                return result_rv

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        assert After.get_function("kernel_stripe__windowed") is not None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "kernel_stripe__windowed" in printed_main
        assert "pl.tensor.slice(out" in printed_main

    def test_tuple_return_sibling_writer_alias_can_window_with_runtime_overlap(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def multi_stripe(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
                scratch: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32]]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                out_next: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile, [row_offset, 0], out)
                scratch_next: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile, [row_offset, 0], scratch)
                return out_next, scratch_next

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
                scratch: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                pair: pl.Tuple[pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32]] = (
                    self.multi_stripe(data, 0, out, scratch)
                )
                first: pl.Tensor[[256, 64], pl.FP32] = pair[0]
                second: pl.Tensor[[256, 64], pl.FP32] = self.kernel_stripe(data, 32, first)
                return second

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        assert After.get_function("multi_stripe__windowed") is not None
        assert After.get_function("kernel_stripe__windowed") is not None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "multi_stripe__windowed" in printed_main
        assert "kernel_stripe__windowed" in printed_main
        assert "pl.tensor.slice(out" in printed_main
        assert "pl.tensor.slice(first" in printed_main

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
                data: pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)],
                row_offset: pl.Scalar[pl.INDEX],
                bias: pl.Scalar[pl.FP32],
                out: pl.Out[
                    pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)]
                ],
            ) -> pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)]:
                tile: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    data, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
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
                        data__window = pl.tensor.slice(data, [64, 64], [row, 0])
                        out_branch__window = pl.tensor.slice(out_branch, [64, 64], [row, 0])
                        out_next__windowed: pl.Tensor[
                            [64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)
                        ] = self.kernel_stripe__windowed(data__window, row, 1.0, out_branch__window)
                        out_next = pl.tensor.assemble(out_branch, out_next__windowed, [row, 0])
                        out_branch_next = pl.yield_(out_next)
                    out_phase_next = pl.yield_(out_branch_next)
                return out_phase_next

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)
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
                data: pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)],
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
                    data, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
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
                data__window = pl.tensor.slice(data, [64, 64], [64, 0])
                k_out__window = pl.tensor.slice(k_out, [64, 64], [64, 0])
                v_out__window = pl.tensor.slice(v_out, [64, 64], [64, 0])
                result__windowed: pl.Tuple[
                    pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)],
                    pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)],
                ] = self.kv_stripe__windowed(data__window, 64, k_out__window, v_out__window)
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

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)
        ir.assert_structural_equal(After, Expected)

    def test_multi_out_same_callsite_parent_stays_baseline(self):
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
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32]]:
                row: pl.Scalar[pl.INDEX] = 64
                result: tuple[pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32]] = self.kv_stripe(
                    data, row, out, out
                )
                return result

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "kv_stripe__windowed" not in printed_main
        assert "pl.tensor.slice(out" not in printed_main
        assert "pl.tensor.assemble(out" not in printed_main

    def test_return_reordered_multi_out_later_parent_read_still_externalizes(self):
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

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        assert After.get_function("kv_stripe__windowed") is not None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "pl.tensor.slice(k_out" in printed_main
        assert "pl.tensor.slice(v_out" in printed_main
        assert "k_next__ssa_v0:" in printed_main
        assert "= result__ssa_v0__assembled_1" in printed_main
        assert "consume_full(k_next__ssa_v0)" in printed_main
        assert "consume_full(k_out)" not in printed_main

    def test_tensor_full_root_later_parent_read_still_externalizes(self):
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

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        assert After.get_function("kernel_stripe__windowed") is not None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "pl.tensor.slice(out" in printed_main
        assert "consume_full(out_next__ssa_v0)" in printed_main
        assert "consume_full(out)" not in printed_main

    def test_loop_returned_output_later_parent_read_still_externalizes(self):
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

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        assert After.get_function("kernel_rows__windowed") is not None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "pl.tensor.slice(out" in printed_main
        assert "consume_full(out_next__ssa_v0)" in printed_main
        assert "consume_full(out)" not in printed_main

    def test_multi_out_final_store_rewrites_only_proven_output(self):
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

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        assert After.get_function("kv_stripe__windowed") is not None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "pl.tensor.slice(k_out" in printed_main
        assert "pl.tensor.assemble(k_out" in printed_main
        assert "pl.tensor.slice(v_out" not in printed_main

        printed_windowed = ir.python_print(_get_function(After, "kv_stripe__windowed"))
        assert "pl.Tensor[[64, 64], pl.FP32" in printed_windowed
        assert "v_out__ssa_v0: pl.Out[pl.Tensor[[256, 64], pl.FP32" in printed_windowed

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

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)
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

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "kv_proj__windowed" in printed_main
        assert "pl.tensor.slice(k_proj_iter" in printed_main
        assert "pl.tensor.slice(v_proj_iter" in printed_main
        assert "pl.tensor.assemble(k_proj_iter" in printed_main
        assert "pl.tensor.assemble(v_proj_iter" in printed_main
        assert "pl.tensor.slice(wk" in printed_main
        assert "pl.tensor.slice(wv" in printed_main

        printed_windowed = ir.python_print(_get_function(After, "kv_proj__windowed"))
        assert "pl.Out[pl.Tensor[[16, 256], pl.FP32" in printed_windowed
        assert printed_windowed.count("pl.Out[pl.Tensor[[16, 256], pl.FP32") >= 2
        assert "pl.Tensor[[128, 256], pl.BF16" in printed_windowed
        assert printed_windowed.count("pl.Tensor[[128, 256], pl.BF16") >= 2

    def test_coalesce_output_pieces_uses_coarse_carrier_and_fine_assemble(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def cache_write(
                self,
                cache: pl.Out[pl.Tensor[[1024, 128], pl.FP32]],
                data: pl.Tensor[[4, 128], pl.FP32],
                slot_offset: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[1024, 128], pl.FP32]:
                for ki, (cache_iter,) in pl.range(0, 4, init_values=(cache,)):
                    src: pl.Tile[[1, 128], pl.FP32] = pl.tile.load(data, [ki, 0], [1, 128], [1, 128])
                    row: pl.Scalar[pl.INDEX] = slot_offset + ki * 128
                    cache_next: pl.Tensor[[1024, 128], pl.FP32] = pl.tile.store(src, [row, 0], cache_iter)
                    cache_rv = pl.yield_(cache_next)
                return cache_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                cache: pl.Out[pl.Tensor[[1024, 128], pl.FP32]],
                data: pl.Tensor[[4, 128], pl.FP32],
            ) -> pl.Tensor[[1024, 128], pl.FP32]:
                slot: pl.Scalar[pl.INDEX] = 7
                result: pl.Tensor[[1024, 128], pl.FP32] = self.cache_write(cache, data, slot)
                return result

        Exact = _run_aggressive_exact_to_optimize_orch_tensors(Before)
        printed_exact_main = ir.python_print(_get_function(Exact, "main"))
        assert "cache__ssa_v0__window_3" in printed_exact_main

        Auto = _run_to_optimize_orch_tensors(
            Before,
            window_policy="auto",
        )
        printed_auto_main = ir.python_print(_get_function(Auto, "main"))
        assert "cache_write__windowed" in printed_auto_main
        assert "cache__ssa_v0__window_base_arg" not in printed_auto_main
        assert "pl.tensor.slice(cache__ssa_v0" not in printed_auto_main
        assert "pl.tensor.assemble(" not in printed_auto_main

        printed_auto_windowed = ir.python_print(_get_function(Auto, "cache_write__windowed"))
        assert "pl.Out[pl.Tensor[[1024, 128], pl.FP32" in printed_auto_windowed
        assert "cache__ssa_v0__window_extent_dyn" not in printed_auto_windowed

        Default = passes.optimize_orch_tensors()(Before)
        printed_default_main = ir.python_print(_get_function(Default, "main"))
        assert "cache_write__windowed" in printed_default_main
        assert "cache__ssa_v0__window_base_arg" not in printed_default_main
        assert "pl.tensor.assemble(" not in printed_default_main

        printed_default_windowed = ir.python_print(_get_function(Default, "cache_write__windowed"))
        assert "pl.Out[pl.Tensor[[1024, 128], pl.FP32" in printed_default_windowed
        assert "cache__ssa_v0__window_extent_dyn" not in printed_default_windowed

        NoMulti = _run_to_optimize_orch_tensors(Before)
        printed_no_multi_main = ir.python_print(_get_function(NoMulti, "main"))
        assert "cache__ssa_v0__window" not in printed_no_multi_main
        assert "pl.tensor.assemble(cache__ssa_v0" not in printed_no_multi_main

        printed_no_multi_windowed = ir.python_print(_get_function(NoMulti, "cache_write__windowed"))
        assert "pl.Out[pl.Tensor[[385, 128], pl.FP32" not in printed_no_multi_windowed

    def test_auto_rejects_multi_output_coalescing(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def cache_write_pair(
                self,
                cache_a: pl.Out[pl.Tensor[[1024, 128], pl.FP32]],
                cache_b: pl.Out[pl.Tensor[[1024, 128], pl.FP32]],
                data: pl.Tensor[[4, 128], pl.FP32],
                slot_offset: pl.Scalar[pl.INDEX],
            ) -> tuple[pl.Tensor[[1024, 128], pl.FP32], pl.Tensor[[1024, 128], pl.FP32]]:
                for ki, (cache_a_iter, cache_b_iter) in pl.range(0, 4, init_values=(cache_a, cache_b)):
                    src: pl.Tile[[1, 128], pl.FP32] = pl.tile.load(data, [ki, 0], [1, 128], [1, 128])
                    row: pl.Scalar[pl.INDEX] = slot_offset + ki * 128
                    cache_a_next: pl.Tensor[[1024, 128], pl.FP32] = pl.tile.store(src, [row, 0], cache_a_iter)
                    cache_b_next: pl.Tensor[[1024, 128], pl.FP32] = pl.tile.store(src, [row, 0], cache_b_iter)
                    cache_a_rv, cache_b_rv = pl.yield_(cache_a_next, cache_b_next)
                return cache_a_rv, cache_b_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                cache_a: pl.Out[pl.Tensor[[1024, 128], pl.FP32]],
                cache_b: pl.Out[pl.Tensor[[1024, 128], pl.FP32]],
                data: pl.Tensor[[4, 128], pl.FP32],
            ) -> tuple[pl.Tensor[[1024, 128], pl.FP32], pl.Tensor[[1024, 128], pl.FP32]]:
                slot: pl.Scalar[pl.INDEX] = 7
                result_a, result_b = self.cache_write_pair(cache_a, cache_b, data, slot)
                return result_a, result_b

        Auto = _run_to_optimize_orch_tensors(
            Before,
            window_policy="auto",
        )
        printed_auto_main = ir.python_print(_get_function(Auto, "main"))
        assert "cache_a__ssa_v0__window" not in printed_auto_main
        assert "cache_b__ssa_v0__window" not in printed_auto_main
        assert "pl.tensor.assemble(cache_a__ssa_v0" not in printed_auto_main
        assert "pl.tensor.assemble(cache_b__ssa_v0" not in printed_auto_main

        printed_auto_windowed = ir.python_print(_get_function(Auto, "cache_write_pair__windowed"))
        assert "pl.Out[pl.Tensor[[385, 128], pl.FP32" not in printed_auto_windowed
        assert printed_auto_windowed.count("pl.Out[pl.Tensor[[1024, 128], pl.FP32") >= 2

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

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

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

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "kv_proj__windowed" in printed_main
        assert "pl.tensor.slice(k_proj" in printed_main
        assert "pl.tensor.slice(v_proj" in printed_main
        assert "pl.tensor.assemble(k_proj" in printed_main
        assert "pl.tensor.assemble(v_proj" in printed_main
        assert "pl.tensor.slice(wk" in printed_main
        assert "pl.tensor.slice(wv" in printed_main

        printed_windowed = ir.python_print(_get_function(After, "kv_proj__windowed"))
        assert printed_windowed.count("pl.Out[pl.Tensor[[16, 256], pl.FP32") >= 2
        assert printed_windowed.count("pl.Tensor[[128, 256], pl.BF16") >= 2

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

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "k_proj__windowed" in printed_main
        assert "pl.tensor.slice(k_iter" in printed_main
        assert "pl.tensor.assemble(k_iter" in printed_main
        assert "pl.tensor.slice(wk" in printed_main

        printed_windowed = ir.python_print(_get_function(After, "k_proj__windowed"))
        assert "pl.Out[pl.Tensor[[16, 256], pl.FP32" in printed_windowed
        assert "pl.Tensor[[128, 256], pl.BF16" in printed_windowed

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

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)
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

    def test_while_return_parent_uses_visible_init_for_output_window(self):
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
                    row_next: pl.Scalar[pl.INDEX] = row_iter + 1
                    row_rv, out_rv = pl.yield_(row_next, out_iter)
                result: pl.Tensor[[256, 64], pl.FP32] = self.kernel_stripe(data, 64, out_rv)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_stripe(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    data, [row_offset, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                ret = pl.tile.store(tile, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def kernel_stripe__windowed(
                self,
                data: pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[
                    pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)]
                ],
            ) -> pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)]:
                tile: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    data, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                ret: pl.Tensor[
                    [64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)
                ] = pl.tile.store(tile, [0, 0], out)
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
                    row_next: pl.Scalar[pl.INDEX] = row_iter + 1
                    row_rv, out_rv = pl.yield_(row_next, out_iter)
                data__window: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.slice(data, [64, 64], [64, 0])
                out__window: pl.Tensor[[64, 64], pl.FP32] = pl.tensor.slice(out, [64, 64], [64, 0])
                result__windowed: pl.Tensor[
                    [64, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)
                ] = self.kernel_stripe__windowed(data__window, 64, out__window)
                result: pl.Tensor[[256, 64], pl.FP32] = pl.tensor.assemble(out, result__windowed, [64, 0])
                return result

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Expected)
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "pl.tensor.slice(out_rv" not in printed_main

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

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)
        ir.assert_structural_equal(After, Expected)

    def test_dynamic_parent_partial_static_output_window_stays_baseline(self):
        user_batch = pl.dynamic("USER_BATCH")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def copy_dyn_batch(
                self,
                data: pl.Tensor[[32, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[user_batch, 128], pl.FP32]],
            ) -> pl.Tensor[[user_batch, 128], pl.FP32]:
                for b0, (out_iter,) in pl.range(0, 32, 16, init_values=(out,)):
                    chunk: pl.Tile[[16, 128], pl.FP32] = pl.load(data, [b0, 0], [16, 128])
                    out_next: pl.Tensor[[user_batch, 128], pl.FP32] = pl.store(chunk, [b0, 0], out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[32, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[user_batch, 128], pl.FP32]],
            ) -> pl.Tensor[[user_batch, 128], pl.FP32]:
                return self.copy_dyn_batch(data, out)

        After = _run_to_optimize_orch_tensors(Before, window_policy="all")

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "pl.tensor.slice(out" not in printed_main
        printed_windowed = ir.python_print(_get_function(After, "copy_dyn_batch__windowed"))
        assert "pl.Out[pl.Tensor[[USER_BATCH, 128], pl.FP32]]" in printed_windowed
        assert "pl.Out[pl.Tensor[[32, 128], pl.FP32" not in printed_windowed

    def test_dynamic_parent_static_nonzero_output_window_stays_baseline(self):
        user_batch = pl.dynamic("USER_BATCH")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def copy_second_tile(
                self,
                data: pl.Tensor[[16, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[user_batch, 128], pl.FP32]],
            ) -> pl.Tensor[[user_batch, 128], pl.FP32]:
                tile: pl.Tile[[16, 128], pl.FP32] = pl.load(data, [0, 0], [16, 128])
                ret = pl.store(tile, [16, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[16, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[user_batch, 128], pl.FP32]],
            ) -> pl.Tensor[[user_batch, 128], pl.FP32]:
                return self.copy_second_tile(data, out)

        After = _run_to_optimize_orch_tensors(Before, window_policy="all")

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "pl.tensor.slice(out" not in printed_main
        assert After.get_function("copy_second_tile__windowed") is None

    def test_dynamic_parent_static_loop_offset_output_window_stays_baseline(self):
        user_batch = pl.dynamic("USER_BATCH")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def copy_tile(
                self,
                data: pl.Tensor[[16, 128], pl.FP32],
                b0: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[user_batch, 128], pl.FP32]],
            ) -> pl.Tensor[[user_batch, 128], pl.FP32]:
                tile: pl.Tile[[16, 128], pl.FP32] = pl.load(data, [0, 0], [16, 128])
                ret = pl.store(tile, [b0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[16, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[user_batch, 128], pl.FP32]],
            ) -> pl.Tensor[[user_batch, 128], pl.FP32]:
                for b0, (out_iter,) in pl.range(0, 32, 16, init_values=(out,)):
                    out_next = self.copy_tile(data, b0, out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

        After = _run_to_optimize_orch_tensors(Before, window_policy="all")

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "pl.tensor.slice(out" not in printed_main
        assert After.get_function("copy_tile__windowed") is None

    def test_dynamic_parent_sparse_nonzero_output_window_coalesces(self):
        cache_rows = pl.dynamic("CACHE_ROWS")

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def cache_write(
                self,
                cache: pl.Out[pl.Tensor[[cache_rows, 128], pl.FP32]],
                data: pl.Tensor[[4, 128], pl.FP32],
                slot_offset: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[cache_rows, 128], pl.FP32]:
                for ki, (cache_iter,) in pl.range(0, 4, init_values=(cache,)):
                    src: pl.Tile[[1, 128], pl.FP32] = pl.tile.load(data, [ki, 0], [1, 128], [1, 128])
                    row: pl.Scalar[pl.INDEX] = slot_offset + ki * 128
                    cache_next: pl.Tensor[[cache_rows, 128], pl.FP32] = pl.tile.store(
                        src, [row, 0], cache_iter
                    )
                    cache_rv = pl.yield_(cache_next)
                return cache_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                cache: pl.Out[pl.Tensor[[cache_rows, 128], pl.FP32]],
                data: pl.Tensor[[4, 128], pl.FP32],
                slot: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[cache_rows, 128], pl.FP32]:
                return self.cache_write(cache, data, slot)

        After = _run_to_optimize_orch_tensors(Before, window_policy="all")

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "cache_write__windowed" in printed_main
        assert "pl.tensor.slice(cache__ssa_v0" in printed_main


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


class TestOutWindowMultiOutSubset:
    """Pattern 5 rewrites each proven Out window independently."""

    def test_one_full_shape_out_does_not_block_windowable_sibling(self):
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

        After = passes.optimize_orch_tensors()(Before)

        assert After.get_function("kv_stripe__windowed") is not None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "kv_stripe__windowed" in printed_main
        assert "pl.tensor.slice(k_out" in printed_main
        assert "pl.tensor.assemble(k_out" in printed_main
        assert "pl.tensor.slice(v_out" not in printed_main

        printed_windowed = ir.python_print(_get_function(After, "kv_stripe__windowed"))
        assert "k_out: pl.Out[pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[64, 1]" in printed_windowed
        assert "v_out: pl.Out[pl.Tensor[[256, 64], pl.FP32" in printed_windowed


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

    def test_submit_windowable_with_later_full_submit_read_still_externalizes(self):
        """Later full-parent Submit reads no longer suppress output windows."""

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
            def consume(
                self,
                src: pl.Tensor[[256, 64], pl.FP32],
                sink: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t: pl.Tile[[64, 64], pl.FP32] = pl.load(src, [0, 0], [64, 64])
                r: pl.Tensor[[64, 64], pl.FP32] = pl.store(t, [0, 0], sink)
                return r

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                with pl.manual_scope():
                    row: pl.Scalar[pl.INDEX] = 64
                    out_next, tid = pl.submit(self.kernel_stripe, data, row, out)
                    sink: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
                    # Later submit reads the FULL `out` (In direction).
                    _consumed, _tid2 = pl.submit(self.consume, out, sink, deps=[tid])
                return out_next

        After = passes.optimize_orch_tensors()(Before)
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "kernel_stripe__windowed" in printed_main
        assert "consume__windowed" not in printed_main
        assert "pl.tensor.slice(out" in printed_main
        assert "pl.Scalar[pl.TASK_ID] = _submit_tmp__windowed[1]" in printed_main
        assert "pl.submit(consume, _submit_tmp__assembled_" in printed_main
        assert "deps=[tid]" in printed_main
        assert "pl.submit(consume, out, sink" not in printed_main

    def test_submit_nested_partition_reduction_projection_is_externalized(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul_split(
                self,
                lhs: pl.Tensor[[16, 128], pl.FP32],
                weight: pl.Tensor[[128, 128], pl.FP32],
                k_off: pl.Scalar[pl.INDEX],
                n_off: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                lhs_tile: pl.Tile[[16, 64], pl.FP32] = pl.load(lhs, [0, k_off], [16, 64])
                weight_tile: pl.Tile[[64, 64], pl.FP32] = pl.load(weight, [k_off, n_off], [64, 64])
                acc_tile: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(lhs_tile, weight_tile)
                return pl.store(acc_tile, [0, n_off], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                lhs: pl.Tensor[[16, 128], pl.FP32],
                weight: pl.Tensor[[128, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            ) -> pl.Tensor[[16, 128], pl.FP32]:
                with pl.manual_scope():
                    out_outer_rv: pl.Tensor[[16, 128], pl.FP32] = out
                    for n, (out_outer,) in pl.range(2, init_values=(out,)):
                        n_off: pl.Scalar[pl.INDEX] = n * 64
                        for k, (out_inner,) in pl.range(2, init_values=(out_outer,)):
                            k_off: pl.Scalar[pl.INDEX] = k * 64
                            out_next, tid = pl.submit(self.matmul_split, lhs, weight, k_off, n_off, out_inner)
                            out_inner_rv = pl.yield_(out_next)
                        out_outer_rv = pl.yield_(out_inner_rv)
                return out_outer_rv

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "matmul_split__windowed" in printed_main
        assert "pl.submit(" in printed_main
        assert "pl.tensor.slice(out_inner" in printed_main
        assert "pl.tensor.slice(lhs" in printed_main
        assert "pl.tensor.slice(weight" in printed_main

    def test_linear_region_overflow_falls_back_to_baseline(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def overflow_store(
                self,
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
                data: pl.Tensor[[1, 64], pl.FP32],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                for i, (out_iter,) in pl.range(0, 2, init_values=(out,)):
                    tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(data, [0, 0], [1, 64], [1, 64])
                    row: pl.Scalar[pl.INDEX] = i * 9223372036854775807 + i
                    out_next: pl.Tensor[[16, 64], pl.FP32] = pl.tile.store(tile, [row, 0], out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
                data: pl.Tensor[[1, 64], pl.FP32],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                result: pl.Tensor[[16, 64], pl.FP32] = self.overflow_store(out, data)
                return result

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        assert After.get_function("overflow_store__windowed") is None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "overflow_store(out__ssa_v0, data__ssa_v0)" in printed_main

    def test_inout_full_read_before_subset_write_stays_baseline(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def update(
                self,
                acc: pl.InOut[pl.Tensor[[128, 64], pl.FP32]],
                data: pl.Tensor[[1, 64], pl.FP32],
            ) -> pl.Tensor[[128, 64], pl.FP32]:
                _full: pl.Tile[[128, 64], pl.FP32] = pl.tile.load(acc, [0, 0], [128, 64], [128, 64])
                tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(data, [0, 0], [1, 64], [1, 64])
                result: pl.Tensor[[128, 64], pl.FP32] = pl.tile.store(tile, [7, 0], acc)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                acc: pl.InOut[pl.Tensor[[128, 64], pl.FP32]],
                data: pl.Tensor[[1, 64], pl.FP32],
            ) -> pl.Tensor[[128, 64], pl.FP32]:
                result: pl.Tensor[[128, 64], pl.FP32] = self.update(acc, data)
                return result

        After = _run_aggressive_exact_to_optimize_orch_tensors(Before)

        assert After.get_function("update__windowed") is None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "update(acc__ssa_v0, data__ssa_v0)" in printed_main


class TestPerKernelAttrs:
    """Phase 2: per-kernel window_outputs / window_inputs attrs."""

    def test_per_kernel_outputs_off(self):
        """Global all plus kernel outputs="off" leaves that kernel unwindowed."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore, attrs={"window_outputs": "off"})
            def writer(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
                data: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.tile.load(data, [0], [64])
                result: pl.Tensor[[64], pl.FP32] = pl.tile.store(t, [0], out)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
                data: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = self.writer(out, data)
                return result

        After = _run_to_optimize_orch_tensors(Before, window_policy="all")
        assert After.get_function("writer__windowed") is None

    def test_per_kernel_inputs_off(self):
        """Global all plus kernel inputs="off" leaves that kernel input unwindowed."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore, attrs={"window_inputs": "off"})
            def reader(
                self,
                out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
                data: pl.Tensor[[128, 64], pl.FP32],
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                t: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(data, [7, 0], [1, 64], [128, 64])
                result: pl.Tensor[[1, 64], pl.FP32] = pl.tile.store(t, [0, 0], out)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
                data: pl.Tensor[[128, 64], pl.FP32],
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                result: pl.Tensor[[1, 64], pl.FP32] = self.reader(out, data)
                return result

        After = _run_to_optimize_orch_tensors(Before, window_policy="all")
        # inputs="off" means no input windowing; output may still be windowed.
        printed_main = ir.python_print(_get_function(After, "main"))
        # Input should NOT be windowed (no data__ssa_v0__window in the call).
        assert "data__ssa_v0__window" not in printed_main

    def test_global_off_overrides_kernel_attrs(self):
        """Global off leaves kernels with attrs unwindowed."""

        @pl.program
        class Before:
            @pl.function(
                type=pl.FunctionType.InCore,
                attrs={"window_outputs": "auto", "window_inputs": "auto"},
            )
            def writer(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
                data: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.tile.load(data, [0], [64])
                result: pl.Tensor[[64], pl.FP32] = pl.tile.store(t, [0], out)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
                data: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = self.writer(out, data)
                return result

        After = _run_to_optimize_orch_tensors(Before, window_policy="off")
        assert After.get_function("writer__windowed") is None

    def test_per_kernel_outputs_coalesce(self):
        """kernel outputs="coalesce" merges multi-piece writes into one bbox."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore, attrs={"window_outputs": "coalesce"})
            def cache_write(
                self,
                cache: pl.Out[pl.Tensor[[1024, 128], pl.FP32]],
                data: pl.Tensor[[4, 128], pl.FP32],
                slot_offset: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[1024, 128], pl.FP32]:
                for ki, (cache_iter,) in pl.range(0, 4, init_values=(cache,)):
                    src: pl.Tile[[1, 128], pl.FP32] = pl.tile.load(data, [ki, 0], [1, 128], [1, 128])
                    row: pl.Scalar[pl.INDEX] = slot_offset + ki * 128
                    cache_next: pl.Tensor[[1024, 128], pl.FP32] = pl.tile.store(src, [row, 0], cache_iter)
                    cache_rv = pl.yield_(cache_next)
                return cache_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                cache: pl.Out[pl.Tensor[[1024, 128], pl.FP32]],
                data: pl.Tensor[[4, 128], pl.FP32],
            ) -> pl.Tensor[[1024, 128], pl.FP32]:
                slot: pl.Scalar[pl.INDEX] = 7
                result: pl.Tensor[[1024, 128], pl.FP32] = self.cache_write(cache, data, slot)
                return result

        After = _run_to_optimize_orch_tensors(Before, window_policy="auto")

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "cache_write__windowed" in printed_main
        assert "cache__ssa_v0__window_base_arg" in printed_main
        assert "cache__ssa_v0__window_extent_arg" in printed_main
        assert "pl.tensor.slice(cache__ssa_v0" in printed_main
        assert "pl.tensor.assemble(" in printed_main
        assert "cache__ssa_v0__window_3" not in printed_main

        printed_windowed = ir.python_print(_get_function(After, "cache_write__windowed"))
        assert "cache__ssa_v0__window_extent_dyn = pl.dynamic" in ir.python_print(After)
        assert "pl.Out[pl.Tensor[[cache__ssa_v0__window_extent_dyn, 128], pl.FP32" in printed_windowed

    def test_policy_all_no_private_dynamic(self):
        """v5: all does not window dynamic readers without a carrier."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def cache_read(
                self,
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
                cache: pl.Tensor[[1024, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
            ) -> pl.Tensor[[4, 64], pl.FP32]:
                for sb, (out_iter,) in pl.range(0, 4, init_values=(out,)):
                    pbid_i32: pl.Scalar[pl.INT32] = pl.tensor.read(block_table, [sb])
                    pbid: pl.Scalar[pl.INDEX] = pl.cast(pbid_i32, target_type=pl.INDEX)
                    row: pl.Scalar[pl.INDEX] = pbid * 128
                    cache_tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(cache, [row, 0], [1, 64], [1, 64])
                    out_next: pl.Tensor[[4, 64], pl.FP32] = pl.tile.store(cache_tile, [sb, 0], out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                cache: pl.Tensor[[1024, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
            ) -> pl.Tensor[[4, 64], pl.FP32]:
                result: pl.Tensor[[4, 64], pl.FP32] = self.cache_read(out, cache, block_table)
                return result

        After = _run_to_optimize_orch_tensors(Before, window_policy="all")
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "cache_read__windowed" not in printed_main

    def test_carrier_fallback_no_private(self):
        """v5: when carrier conditions fail, the reader stays full parent."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def cache_read(
                self,
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
                cache: pl.Tensor[[1024, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
                valid_blocks: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[4, 64], pl.FP32]:
                for sb, (out_iter,) in pl.range(0, valid_blocks, init_values=(out,)):
                    pbid_i32: pl.Scalar[pl.INT32] = pl.tensor.read(block_table, [sb])
                    pbid: pl.Scalar[pl.INDEX] = pl.cast(pbid_i32, target_type=pl.INDEX)
                    row: pl.Scalar[pl.INDEX] = pbid * 128
                    cache_tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(cache, [row, 0], [1, 64], [1, 64])
                    out_next: pl.Tensor[[4, 64], pl.FP32] = pl.tile.store(cache_tile, [sb, 0], out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                cache: pl.Tensor[[1024, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
                valid_blocks: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
            ) -> pl.Tensor[[4, 64], pl.FP32]:
                result: pl.Tensor[[4, 64], pl.FP32] = self.cache_read(out, cache, block_table, valid_blocks)
                return result

        After = _run_to_optimize_orch_tensors(Before, window_policy="all")
        printed_main = ir.python_print(_get_function(After, "main"))
        # Dynamic reader with unknown trip count → no carrier, no private → full parent.
        assert "cache_read__windowed" not in printed_main
        assert "cache__ssa_v0__window" not in printed_main

    def test_coalesce_carrier_output_derived_carrier_dynamic_reader(self):
        """coalesce_carrier keeps dynamic reader candidates when an output carrier exists."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore, attrs={"window_outputs": "coalesce_carrier"})
            def cache_write(
                self,
                cache: pl.Out[pl.Tensor[[1024, 64], pl.FP32]],
                data: pl.Tensor[[2, 64], pl.FP32],
                slot: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[1024, 64], pl.FP32]:
                for ki, (cache_iter,) in pl.range(0, 2, init_values=(cache,)):
                    src: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(data, [ki, 0], [1, 64], [1, 64])
                    row: pl.Scalar[pl.INDEX] = slot + ki * 128
                    cache_next: pl.Tensor[[1024, 64], pl.FP32] = pl.tile.store(src, [row, 0], cache_iter)
                    cache_rv = pl.yield_(cache_next)
                return cache_rv

            @pl.function(type=pl.FunctionType.InCore)
            def cache_read(
                self,
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
                cache: pl.Tensor[[1024, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
            ) -> pl.Tensor[[4, 64], pl.FP32]:
                for sb, (out_iter,) in pl.range(0, 4, init_values=(out,)):
                    pbid_i32: pl.Scalar[pl.INT32] = pl.tensor.read(block_table, [sb])
                    pbid: pl.Scalar[pl.INDEX] = pl.cast(pbid_i32, target_type=pl.INDEX)
                    row: pl.Scalar[pl.INDEX] = pbid * 128
                    cache_tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(cache, [row, 0], [1, 64], [1, 64])
                    out_next: pl.Tensor[[4, 64], pl.FP32] = pl.tile.store(cache_tile, [sb, 0], out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                cache: pl.Out[pl.Tensor[[1024, 64], pl.FP32]],
                data: pl.Tensor[[2, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[1024, 64], pl.FP32], pl.Tensor[[4, 64], pl.FP32]]:
                slot: pl.Scalar[pl.INDEX] = 7
                cache_next: pl.Tensor[[1024, 64], pl.FP32] = self.cache_write(cache, data, slot)
                result: pl.Tensor[[4, 64], pl.FP32] = self.cache_read(out, cache_next, block_table)
                return cache_next, result

        After = _run_to_optimize_orch_tensors(Before, window_policy="auto")

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "cache_write__windowed" in printed_main
        assert "cache_read__windowed" in printed_main
        assert "__carrier_base" in printed_main
        assert "__carrier_extent" in printed_main
        assert "__carrier_scan_" in printed_main

    def test_coalesce_output_no_carrier_contract_dynamic_reader_stays_full_parent(self):
        """coalesce only means output bbox; it does not force a dynamic reader carrier."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore, attrs={"window_outputs": "coalesce"})
            def cache_write(
                self,
                cache: pl.Out[pl.Tensor[[1024, 64], pl.FP32]],
                data: pl.Tensor[[2, 64], pl.FP32],
                slot: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[1024, 64], pl.FP32]:
                for ki, (cache_iter,) in pl.range(0, 2, init_values=(cache,)):
                    src: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(data, [ki, 0], [1, 64], [1, 64])
                    row: pl.Scalar[pl.INDEX] = slot + ki * 128
                    cache_next: pl.Tensor[[1024, 64], pl.FP32] = pl.tile.store(src, [row, 0], cache_iter)
                    cache_rv = pl.yield_(cache_next)
                return cache_rv

            @pl.function(type=pl.FunctionType.InCore)
            def cache_read(
                self,
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
                cache: pl.Tensor[[1024, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
            ) -> pl.Tensor[[4, 64], pl.FP32]:
                for sb, (out_iter,) in pl.range(0, 4, init_values=(out,)):
                    pbid_i32: pl.Scalar[pl.INT32] = pl.tensor.read(block_table, [sb])
                    pbid: pl.Scalar[pl.INDEX] = pl.cast(pbid_i32, target_type=pl.INDEX)
                    row: pl.Scalar[pl.INDEX] = pbid * 128
                    cache_tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(cache, [row, 0], [1, 64], [1, 64])
                    out_next: pl.Tensor[[4, 64], pl.FP32] = pl.tile.store(cache_tile, [sb, 0], out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                cache: pl.Out[pl.Tensor[[1024, 64], pl.FP32]],
                data: pl.Tensor[[2, 64], pl.FP32],
                block_table: pl.Tensor[[4], pl.INT32],
                out: pl.Out[pl.Tensor[[4, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[1024, 64], pl.FP32], pl.Tensor[[4, 64], pl.FP32]]:
                slot: pl.Scalar[pl.INDEX] = 7
                cache_next: pl.Tensor[[1024, 64], pl.FP32] = self.cache_write(cache, data, slot)
                result: pl.Tensor[[4, 64], pl.FP32] = self.cache_read(out, cache_next, block_table)
                return cache_next, result

        After = _run_to_optimize_orch_tensors(Before, window_policy="auto")

        printed_main = ir.python_print(_get_function(After, "main"))
        assert "cache_write__windowed" in printed_main
        assert "cache_read__windowed" not in printed_main
        assert "__carrier_base" not in printed_main
        assert "__carrier_extent" not in printed_main
        assert "__carrier_scan_" not in printed_main

    def test_coalesce_current_barrier_not_model_specific(self):
        """Runtime current barrier is driven by coalesce_carrier attrs, not model-specific names."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore, attrs={"window_outputs": "coalesce_carrier"})
            def produce_segments(
                self,
                src: pl.Tensor[[64, 256], pl.FP32],
                col: pl.Scalar[pl.INDEX],
                scratch_buffer: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
            ) -> pl.Tensor[[64, 256], pl.FP32]:
                tile: pl.Tile[[64, 128], pl.FP32] = pl.tile.load(src, [0, col], [64, 128], [64, 128])
                return pl.tile.store(tile, [0, col], scratch_buffer)

            @pl.function(type=pl.FunctionType.InCore)
            def combine_segments(
                self,
                scratch_buffer: pl.Tensor[[64, 256], pl.FP32],
                col: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
            ) -> tuple[pl.Tensor[[64, 256], pl.FP32], pl.Tensor[[64, 256], pl.FP32]]:
                tile: pl.Tile[[64, 128], pl.FP32] = pl.tile.load(
                    scratch_buffer, [0, col], [64, 128], [64, 128]
                )
                out_next: pl.Tensor[[64, 256], pl.FP32] = pl.tile.store(tile, [0, col], out)
                return out_next, scratch_buffer

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                src: pl.Tensor[[64, 256], pl.FP32],
                scratch_buffer: pl.Tensor[[64, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
            ) -> pl.Tensor[[64, 256], pl.FP32]:
                for i, (scratch_iter,) in pl.range(0, 2, init_values=(scratch_buffer,)):
                    col: pl.Scalar[pl.INDEX] = i * 128
                    scratch_next = self.produce_segments(src, col, scratch_iter)
                    scratch_rv = pl.yield_(scratch_next)

                for j, (out_iter,) in pl.range(0, 2, init_values=(out,)):
                    col: pl.Scalar[pl.INDEX] = j * 128
                    result = self.combine_segments(scratch_rv, col, out_iter)
                    out_next: pl.Tensor[[64, 256], pl.FP32] = result[0]
                    out_rv = pl.yield_(out_next)
                return out_rv

        After = _run_to_optimize_orch_tensors(Before, window_policy="auto")

        printed_main = ir.python_print(_get_function(After, "main"))
        assert After.get_function("__pypto_runtime_current_barrier") is not None, printed_main
        assert "__runtime_current" in printed_main
        assert "produce_segments__windowed" in printed_main
        assert "combine_segments" in printed_main

    def test_coalesce_current_barrier_not_inserted_without_carrier_contract(self):
        """Plain coalesce should not trigger runtime-current barrier insertion."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore, attrs={"window_outputs": "coalesce"})
            def produce_segments(
                self,
                src: pl.Tensor[[64, 256], pl.FP32],
                col: pl.Scalar[pl.INDEX],
                scratch_buffer: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
            ) -> pl.Tensor[[64, 256], pl.FP32]:
                tile: pl.Tile[[64, 128], pl.FP32] = pl.tile.load(src, [0, col], [64, 128], [64, 128])
                return pl.tile.store(tile, [0, col], scratch_buffer)

            @pl.function(type=pl.FunctionType.InCore)
            def combine_segments(
                self,
                scratch_buffer: pl.Tensor[[64, 256], pl.FP32],
                col: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
            ) -> tuple[pl.Tensor[[64, 256], pl.FP32], pl.Tensor[[64, 256], pl.FP32]]:
                tile: pl.Tile[[64, 128], pl.FP32] = pl.tile.load(
                    scratch_buffer, [0, col], [64, 128], [64, 128]
                )
                out_next: pl.Tensor[[64, 256], pl.FP32] = pl.tile.store(tile, [0, col], out)
                return out_next, scratch_buffer

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                src: pl.Tensor[[64, 256], pl.FP32],
                scratch_buffer: pl.Tensor[[64, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
            ) -> pl.Tensor[[64, 256], pl.FP32]:
                for i, (scratch_iter,) in pl.range(0, 2, init_values=(scratch_buffer,)):
                    col: pl.Scalar[pl.INDEX] = i * 128
                    scratch_next = self.produce_segments(src, col, scratch_iter)
                    scratch_rv = pl.yield_(scratch_next)

                for j, (out_iter,) in pl.range(0, 2, init_values=(out,)):
                    col: pl.Scalar[pl.INDEX] = j * 128
                    result = self.combine_segments(scratch_rv, col, out_iter)
                    out_next: pl.Tensor[[64, 256], pl.FP32] = result[0]
                    out_rv = pl.yield_(out_next)
                return out_rv

        After = _run_to_optimize_orch_tensors(Before, window_policy="auto")

        printed_main = ir.python_print(_get_function(After, "main"))
        assert After.get_function("__pypto_runtime_current_barrier") is None, printed_main
        assert "__runtime_current" not in printed_main
        assert "produce_segments__windowed" in printed_main

    def test_coalesce_carrier_no_barrier_when_output_rewrite_falls_back(self):
        """coalesce_carrier attr alone should not trigger runtime-current barrier."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore, attrs={"window_outputs": "coalesce_carrier"})
            def produce_full_tensor(
                self,
                src: pl.Tensor[[64, 256], pl.FP32],
                scratch_buffer: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
            ) -> pl.Tensor[[64, 256], pl.FP32]:
                tile: pl.Tile[[64, 256], pl.FP32] = pl.tile.load(src, [0, 0], [64, 256], [64, 256])
                return pl.tile.store(tile, [0, 0], scratch_buffer)

            @pl.function(type=pl.FunctionType.InCore)
            def combine_segments(
                self,
                scratch_buffer: pl.Tensor[[64, 256], pl.FP32],
                col: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
            ) -> tuple[pl.Tensor[[64, 256], pl.FP32], pl.Tensor[[64, 256], pl.FP32]]:
                tile: pl.Tile[[64, 128], pl.FP32] = pl.tile.load(
                    scratch_buffer, [0, col], [64, 128], [64, 128]
                )
                out_next: pl.Tensor[[64, 256], pl.FP32] = pl.tile.store(tile, [0, col], out)
                return out_next, scratch_buffer

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                src: pl.Tensor[[64, 256], pl.FP32],
                scratch_buffer: pl.Tensor[[64, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
            ) -> pl.Tensor[[64, 256], pl.FP32]:
                for i, (scratch_iter,) in pl.range(0, 2, init_values=(scratch_buffer,)):
                    scratch_next = self.produce_full_tensor(src, scratch_iter)
                    scratch_rv = pl.yield_(scratch_next)

                for j, (out_iter,) in pl.range(0, 2, init_values=(out,)):
                    col: pl.Scalar[pl.INDEX] = j * 128
                    result = self.combine_segments(scratch_rv, col, out_iter)
                    out_next: pl.Tensor[[64, 256], pl.FP32] = result[0]
                    out_rv = pl.yield_(out_next)
                return out_rv

        After = _run_to_optimize_orch_tensors(Before, window_policy="auto")

        printed_main = ir.python_print(_get_function(After, "main"))
        assert After.get_function("__pypto_runtime_current_barrier") is None, printed_main
        assert "__runtime_current" not in printed_main
        assert "produce_full_tensor__windowed" not in printed_main
        assert "produce_full_tensor(src__ssa_v0, scratch_iter)" in printed_main

    def test_invalid_window_outputs_attr_raises(self):
        """kernel attrs={"window_outputs": "unknown"} triggers a CHECK failure."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore, attrs={"window_outputs": "unknown"})
            def writer(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
                data: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.tile.load(data, [0], [64])
                result: pl.Tensor[[64], pl.FP32] = pl.tile.store(t, [0], out)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
                data: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = self.writer(out, data)
                return result

        with pytest.raises(Exception, match="window_outputs"):
            _run_to_optimize_orch_tensors(Before, window_policy="all")

    def test_invalid_window_inputs_attr_raises(self):
        """kernel attrs={"window_inputs": "all"} errors because v5 does not support it."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore, attrs={"window_inputs": "all"})
            def reader(
                self,
                out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
                data: pl.Tensor[[128, 64], pl.FP32],
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                t: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(data, [7, 0], [1, 64], [128, 64])
                result: pl.Tensor[[1, 64], pl.FP32] = pl.tile.store(t, [0, 0], out)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
                data: pl.Tensor[[128, 64], pl.FP32],
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                result: pl.Tensor[[1, 64], pl.FP32] = self.reader(out, data)
                return result

        with pytest.raises(Exception, match="window_inputs"):
            _run_to_optimize_orch_tensors(Before, window_policy="auto")

    def test_invalid_window_policy_raises(self):
        """window_policy="invalid" triggers a CHECK failure."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def writer(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
                data: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.tile.load(data, [0], [64])
                result: pl.Tensor[[64], pl.FP32] = pl.tile.store(t, [0], out)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
                data: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = self.writer(out, data)
                return result

        with pytest.raises(Exception, match="window_policy"):
            _run_to_optimize_orch_tensors(Before, window_policy="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
