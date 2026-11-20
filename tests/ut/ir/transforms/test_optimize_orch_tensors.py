# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for OptimizeOrchTensors pass.

Transform tests use explicit Before/Expected programs where practical. Tests
asserting that the pass declines to rewrite (``..._stays_baseline`` /
``..._falls_back_to_baseline``) compare against ``Before`` normalized by
``_run_prereqs_only`` -- the Default pipeline prefix WITHOUT the pass under
test. Never build a golden by running OptimizeOrchTensors itself: a regression
in the baseline would then change both sides and the test would stay green.
"""

import time

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def _modify_incore_windowize(program, *, enable: bool):
    functions = []
    for func in program.functions.values():
        if not ir.is_incore_type(func.func_type):
            functions.append(func)
            continue
        attrs = dict(func.attrs)
        if enable:
            attrs["windowize"] = True
        else:
            attrs.pop("windowize", None)
        params = list(zip(func.params, func.param_directions, strict=True))
        functions.append(
            ir.Function(
                func.name,
                params,
                func.return_types,
                func.body,
                func.span,
                type=func.func_type,
                level=func.level,
                role=func.role,
                attrs=attrs,
                requires_runtime_binding=func.requires_runtime_binding,
            )
        )
    return ir.Program(functions, program.name, program.span)


def _with_incore_windowize(program):
    return _modify_incore_windowize(program, enable=True)


def _strip_windowize_attrs(program):
    return _modify_incore_windowize(program, enable=False)


def _run_to_optimize_orch_tensors(program):
    """Run the Default pipeline prefix, then OptimizeOrchTensors, with
    ``windowize`` stamped on every InCore function (and stripped again after)."""
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    result = _with_incore_windowize(program)
    for pass_name, pass_obj in zip(pm.pass_names, pm.passes, strict=True):
        if pass_name == "OptimizeOrchTensors":
            result = passes.optimize_orch_tensors()(result)
            return _strip_windowize_attrs(result)
        result = pass_obj(result)
    raise AssertionError("Default pipeline did not run OptimizeOrchTensors")


def _run_prereqs_only(program):
    """Normalize ``program`` with the Default pipeline prefix up to -- but NOT
    including -- OptimizeOrchTensors.

    This brings a program to the pipeline stage the pass under test observes
    (SSA renaming, call flattening, ...) without ever running the pass itself,
    so the result is usable as a golden. Mirrors the ``_run_prereqs_only``
    helper in ``test_fuse_create_assemble_to_slice.py``.
    """
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    result = program
    for pass_name, pass_obj in zip(pm.pass_names, pm.passes, strict=True):
        if pass_name == "OptimizeOrchTensors":
            return result
        result = pass_obj(result)
    raise AssertionError("Default pipeline did not run OptimizeOrchTensors")


def _assert_unchanged_by_pass(before, after):
    """Assert OptimizeOrchTensors left ``before`` structurally unchanged.

    ``after`` is the windowized run of the pass. The golden is ``before``
    normalized by PREREQUISITE passes only -- the pass under test never runs on
    the right-hand side. Building the golden from a second run of the pass (the
    previous behaviour here) let a baseline regression cancel out on both sides
    and keep the test green.
    """
    ir.assert_structural_equal(after, _run_prereqs_only(before))


def _get_function(program, name: str):
    func = program.get_function(name)
    assert func is not None
    return func


def _calls_named(program, op_name: str) -> list[ir.Call]:
    """Every ``op_name`` call in the program, in traversal order."""
    found: list[ir.Call] = []
    target = ir.get_op(op_name).name

    class Collect(ir.IRVisitor):
        def visit_call(self, op):
            if op.op.name == target:
                found.append(op)
            super().visit_call(op)

    Collect().visit_program(program)
    return found


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

    @pytest.mark.parametrize(
        "access",
        [
            pytest.param("alias = stored\n        first = alias", id="alias_chain"),
            pytest.param("pair = (stored, x)\n        first = pair[0]", id="tuple_projection"),
            pytest.param(
                "alias = stored\n        pair = (alias, x)\n        other = pair\n        first = other[0]",
                id="tuple_alias",
            ),
            pytest.param(
                "pair = (stored, x)\n        destination = pair[0]\n"
                "        again = pl.tile.store(tile, [0, 0], destination)\n        first = again",
                id="store_through_projection",
            ),
        ],
    )
    def test_output_stride_propagates_through_aliases(self, access):
        """Pattern 2 propagates the store's refined view without windowization."""
        source = f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[128, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile = pl.tile.load(x, [0, 0], [64, 64], [64, 64])
        stored = pl.tile.store(tile, [0, 0], out)
        {access}
        back = pl.tile.load(first, [0, 0], [64, 64], [64, 64])
        return stored

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[128, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        chunk = pl.tensor.create([64, 64], dtype=pl.FP32)
        result = self.kernel(x, chunk)
        assembled = pl.tensor.assemble(out, result, [32, 32])
        return assembled
"""
        before = pl.parse_program(source)
        expected = pl.parse_program(
            source.replace(
                "out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]",
                "out: pl.Out[pl.Tensor[[64, 64], pl.FP32, "
                "pl.TensorView(stride=[128, 1], layout=pl.TensorLayout.ND)]]",
            )
        )
        after = passes.optimize_orch_tensors()(before)
        ir.assert_structural_equal(after, expected)
        assert after.get_function("kernel__windowed") is None

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

    def test_assemble_carrying_a_fixpipe_epilogue_is_left_alone(self):
        """A `pre_quant` / `pre_relu` assemble declines the rewrite rather than
        losing its epilogue.

        The rewrite replaces the assemble with a store, which moves the write to
        a different destination *class* (tile -> GM tensor) and a different
        destination *dtype*. Both are inputs to whether the epilogue is legal and
        how its scale word encodes, so the kwargs cannot be carried across
        unchanged — and dropping them is worse than declining, because
        `FixpipeEpilogueValid` runs later and by then there would be no epilogue
        left to reject. The ReLU would simply be gone from the kernel.
        """

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
                    acc_next__tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.assemble(
                        acc, chunk__tile, [0, off], pre_quant=0.5, pre_relu=True
                    )
                    result: pl.Tile[[1, 64], pl.FP32] = pl.yield_(acc_next__tile)
                ret0__store: pl.Tensor[[1, 64], pl.FP32] = pl.store(result, [0, 0], ret0__out)
                return ret0__store

            @pl.function
            def main(self, x: pl.Tensor[[1, 32], pl.FP32]) -> pl.Tensor[[1, 64], pl.FP32]:
                ret0__out: pl.Tensor[[1, 64], pl.FP32] = pl.create_tensor([1, 64], dtype=pl.FP32)
                y: pl.Tensor[[1, 64], pl.FP32] = self.main_incore_0(x, ret0__out)
                return y

        After = passes.optimize_orch_tensors()(Before)

        # The whole program is untouched: the assemble keeps its kwargs, and the
        # post-loop store the rewrite would have deleted is still there.
        ir.assert_structural_equal(After, Before)

        assembles = _calls_named(After, "tile.assemble")
        assert len(assembles) == 1, f"the assemble must survive, got {len(assembles)}"
        kwargs = dict(assembles[0].kwargs)
        assert kwargs["pre_quant"] == 0.5
        assert kwargs["pre_relu"] is True
        assert len(_calls_named(After, "tile.store")) == 1, "the post-loop store must survive"

    def test_store_carrying_a_fixpipe_epilogue_is_left_alone(self):
        """The same guard from the other end: the rewrite *deletes* the post-loop
        store, so an epilogue there would go with it."""

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
                ret0__store: pl.Tensor[[1, 64], pl.FP32] = pl.store(result, [0, 0], ret0__out, pre_relu=True)
                return ret0__store

            @pl.function
            def main(self, x: pl.Tensor[[1, 32], pl.FP32]) -> pl.Tensor[[1, 64], pl.FP32]:
                ret0__out: pl.Tensor[[1, 64], pl.FP32] = pl.create_tensor([1, 64], dtype=pl.FP32)
                y: pl.Tensor[[1, 64], pl.FP32] = self.main_incore_0(x, ret0__out)
                return y

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Before)
        stores = _calls_named(After, "tile.store")
        assert len(stores) == 1, "the store must survive rather than be folded into the loop"
        assert dict(stores[0].kwargs)["pre_relu"] is True


class TestAivSliceStrideDataflow:
    """Physical storage survives dataflow edges independently of logical types."""

    @pytest.mark.parametrize(
        "shape, dropped, result_shape, strides, layout",
        [
            ([1, 32], [0], [32], [1], "ND"),
            ([32, 1], [1], [32], [1], "ND"),
            ([2, 1, 32], [1], [2, 32], [32, 1], "ND"),
            ([2, 32, 1], [2], [2, 32], [32, 1], "ND"),
            ([1, 4, 32], [0], [4, 32], [1, 4], "DN"),
        ],
    )
    @pytest.mark.parametrize("submit", [False, True], ids=["call", "submit"])
    def test_dn_slice_dropped_plane_updates_layout(
        self, shape, dropped, result_shape, strides, layout, submit
    ):
        dispatch = "self.kernel(chunk)"
        if submit:
            dispatch = "with pl.manual_scope():\n            tid = pl.submit(self.kernel, chunk)"
        source = f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, x: pl.Tensor[{result_shape}, pl.FP32]):
        return

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, parent: pl.Tensor[{shape}, pl.FP32, pl.TensorView(layout=pl.TensorLayout.DN)]):
        chunk = pl.tensor.slice(parent, {shape}, {[0] * len(shape)}, drop_dims={dropped})
        {dispatch}
"""
        expected = pl.parse_program(
            source.replace(
                f"x: pl.Tensor[{result_shape}, pl.FP32]",
                f"x: pl.Tensor[{result_shape}, pl.FP32, "
                f"pl.TensorView(stride={strides}, layout=pl.TensorLayout.{layout})]",
            )
        )
        ir.assert_structural_equal(passes.optimize_orch_tensors()(pl.parse_program(source)), expected)

    @pytest.mark.parametrize("submit", [False, True], ids=["call", "submit"])
    def test_aiv_infers_dn_layout_from_caller(self, submit):
        dispatch = "self.kernel(chunk)"
        if submit:
            dispatch = "with pl.manual_scope():\n            tid = pl.submit(self.kernel, chunk)"
        source = f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, x: pl.Tensor[[8, 256], pl.FP32]):
        return

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, parent: pl.Tensor[[10, 288], pl.FP32, pl.TensorView(layout=pl.TensorLayout.DN)]):
        chunk = pl.tensor.slice(parent, [8, 256], [1, 7])
        {dispatch}
"""
        expected = pl.parse_program(
            source.replace(
                "x: pl.Tensor[[8, 256], pl.FP32]",
                "x: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(stride=[1, 10], layout=pl.TensorLayout.DN)]",
            )
        )
        ir.assert_structural_equal(passes.optimize_orch_tensors()(pl.parse_program(source)), expected)

    @pytest.mark.parametrize(
        "view, shape, strides, layout",
        [
            ("pl.tensor.transpose(chunk, 0, 1)", [3, 2, 256], [288, 2304, 1], "ND"),
            ("pl.tensor.transpose(chunk, -2, -1)", [2, 256, 3], [2304, 1, 288], "DN"),
            ("pl.tensor.view(chunk, layout=pl.DN)", [2, 256, 3], [2304, 1, 288], "DN"),
        ],
    )
    @pytest.mark.parametrize("submit", [False, True], ids=["call", "submit"])
    def test_metadata_views_preserve_slice_storage(self, view, shape, strides, layout, submit):
        dispatch = "self.kernel(viewed)"
        if submit:
            dispatch = "with pl.manual_scope():\n            tid = pl.submit(self.kernel, viewed)"
        source = f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, x: pl.Tensor[{shape}, pl.FP32]):
        return

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, parent: pl.Tensor[[4, 8, 288], pl.FP32]):
        chunk = pl.tensor.slice(parent, [2, 3, 256], [1, 2, 7])
        viewed = {view}
        {dispatch}
"""
        expected = pl.parse_program(
            source.replace(
                f"x: pl.Tensor[{shape}, pl.FP32]",
                f"x: pl.Tensor[{shape}, pl.FP32, "
                f"pl.TensorView(stride={strides}, layout=pl.TensorLayout.{layout})]",
            )
        )
        ir.assert_structural_equal(passes.optimize_orch_tensors()(pl.parse_program(source)), expected)

    @pytest.mark.parametrize(
        "view, shape",
        [
            ("pl.tensor.transpose(x, 0, 1)", [3, 2, 256]),
            ("pl.tensor.transpose(x, -2, -1)", [2, 256, 3]),
            ("pl.tensor.view(x, layout=pl.DN)", [2, 256, 3]),
            ("pl.tensor.slice(x, [2, 2, 128], [0, 1, 4])", [2, 2, 128]),
            (
                "pl.tensor.set_validshape(pl.tensor.slice(x, [1, 3, 256], [0, 0, 0], drop_dims=[0]), 2, 128)",
                [3, 256],
            ),
        ],
    )
    @pytest.mark.parametrize("submit", [False, True], ids=["call", "submit"])
    def test_aiv_body_metadata_views_use_refined_strides(self, view, shape, submit):
        # The golden deduces every local type from the physical parameter view;
        # checking the whole body catches stale assignments and tile operands.
        dispatch = "self.kernel(chunk)"
        if submit:
            dispatch = "with pl.manual_scope():\n            tid = pl.submit(self.kernel, chunk)"
        source = f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, x: pl.Tensor[[2, 3, 256], pl.FP32]):
        viewed = {view}
        tile = pl.tile.load(viewed, {[0] * len(shape)}, {shape})
        return

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, parent: pl.Tensor[[4, 8, 288], pl.FP32]):
        chunk = pl.tensor.slice(parent, [2, 3, 256], [1, 2, 7])
        {dispatch}
"""
        expected = pl.parse_program(
            source.replace(
                "x: pl.Tensor[[2, 3, 256], pl.FP32]",
                "x: pl.Tensor[[2, 3, 256], pl.FP32, pl.TensorView(stride=[2304, 288, 1])]",
            )
        )
        before = pl.parse_program(source)
        # Isolate parameter substitution, including metadata forms normally
        # rewritten by ConvertTensorToTileOps before this pass.
        with passes.PassContext([], passes.VerificationLevel.NONE):
            after = passes.optimize_orch_tensors()(before)
        ir.assert_structural_equal(after, expected)

    @pytest.mark.parametrize("operation", ["view", "reshape", "reinterpret_view"])
    @pytest.mark.parametrize("pitch", [256, 288])
    def test_aiv_body_shaped_views_check_refined_storage(self, operation, pitch):
        viewed = f"pl.tensor.{operation}(x, [4, 512])"
        if operation == "reinterpret_view":
            viewed = "pl.tensor.reinterpret_view(x, pl.INT32, shape=[4, 512])"
        source = f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, x: pl.Tensor[[8, 256], pl.FP32]):
        viewed = {viewed}
        tile = pl.tile.load(viewed, [0, 0], [4, 512])
        return

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, parent: pl.Tensor[[10, {pitch}], pl.FP32]):
        chunk = pl.tensor.slice(parent, [8, 256], [1, 0])
        self.kernel(chunk)
"""
        before = pl.parse_program(source)
        if pitch != 256:
            with passes.PassContext([], passes.VerificationLevel.NONE):
                with pytest.raises(ValueError, match="packed"):
                    passes.optimize_orch_tensors()(before)
        else:
            expected = pl.parse_program(
                source.replace(
                    "x: pl.Tensor[[8, 256], pl.FP32]",
                    "x: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(stride=[256, 1])]",
                )
            )
            with passes.PassContext([], passes.VerificationLevel.NONE):
                after = passes.optimize_orch_tensors()(before)
            ir.assert_structural_equal(after, expected)

    @pytest.mark.parametrize("caller_layout", ["ND", "DN"])
    @pytest.mark.parametrize("submit", [False, True], ids=["call", "submit"])
    def test_aiv_layout_only_declaration_checks_caller(self, caller_layout, submit):
        dispatch = "self.kernel(chunk)"
        if submit:
            dispatch = "with pl.manual_scope():\n            tid = pl.submit(self.kernel, chunk)"
        source = f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, x: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(layout=pl.DN)]):
        return

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, parent: pl.Tensor[[10, 288], pl.FP32, pl.TensorView(layout=pl.{caller_layout})]):
        chunk = pl.tensor.slice(parent, [8, 256], [1, 7])
        {dispatch}
"""
        before = pl.parse_program(source)
        if caller_layout == "ND":
            with pytest.raises(ValueError, match="explicit layout"):
                passes.optimize_orch_tensors()(before)
        else:
            expected = pl.parse_program(
                source.replace(
                    "pl.TensorView(layout=pl.DN)]):",
                    "pl.TensorView(stride=[1, 10], layout=pl.DN)]):",
                    1,
                )
            )
            ir.assert_structural_equal(passes.optimize_orch_tensors()(before), expected)

    @pytest.mark.parametrize("operation", ["view", "reshape", "reinterpret_view"])
    @pytest.mark.parametrize("pitch", [256, 288])
    def test_shaped_views_require_effectively_packed_source(self, operation, pitch):
        dtype = "INT32" if operation == "reinterpret_view" else "FP32"
        viewed = f"pl.tensor.{operation}(chunk, [4, 512])"
        if operation == "reinterpret_view":
            viewed = "pl.tensor.reinterpret_view(chunk, pl.INT32, shape=[4, 512])"
        source = f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, x: pl.Tensor[[4, 512], pl.{dtype}]):
        return

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, parent: pl.Tensor[[10, {pitch}], pl.FP32]):
        chunk = pl.tensor.slice(parent, [8, 256], [1, 0])
        viewed = {viewed}
        self.kernel(viewed)
"""
        before = pl.parse_program(source)
        if pitch != 256:
            with pytest.raises(ValueError, match="packed"):
                passes.optimize_orch_tensors()(before)
        else:
            expected = pl.parse_program(
                source.replace(
                    f"x: pl.Tensor[[4, 512], pl.{dtype}]",
                    f"x: pl.Tensor[[4, 512], pl.{dtype}, pl.TensorView(stride=[512, 1])]",
                )
            )
            ir.assert_structural_equal(passes.optimize_orch_tensors()(before), expected)

    @pytest.mark.parametrize("consumer", ["unused", "incore", "aiv"])
    @pytest.mark.parametrize("submit", [False, True], ids=["call", "submit"])
    def test_pitched_reshape_constraint_only_checks_aiv_consumers(self, consumer, submit):
        # An unrelated AIV call must not impose the new stride contract on
        # legacy InCore or unused reshape results in the same orchestration.
        consume = ""
        if consumer != "unused":
            consume = f"self.{consumer}(viewed)"
            if submit:
                consume = f"with pl.manual_scope():\n            tid = pl.submit(self.{consumer}, viewed)"
        program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def aiv(self, x: pl.Tensor[[4, 512], pl.FP32]):
        return

    @pl.function(type=pl.FunctionType.InCore)
    def incore(self, x: pl.Tensor[[4, 512], pl.FP32]):
        tile = pl.tile.load(x, [0, 0], [4, 512])
        return

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, parent: pl.Tensor[[10, 288], pl.FP32], dense: pl.Tensor[[4, 512], pl.FP32]):
        chunk = pl.tensor.slice(parent, [8, 256], [1, 0])
        viewed = pl.tensor.reshape(chunk, [4, 512])
        {consume}
        self.aiv(dense)
""")
        if consumer == "aiv":
            with pytest.raises(ValueError, match="tensor.reshape requires packed ND storage"):
                passes.optimize_orch_tensors()(program)
        else:
            ir.assert_structural_equal(passes.optimize_orch_tensors()(program), program)

    @pytest.mark.parametrize("control", ["if", "for", "while"])
    @pytest.mark.parametrize("tuple_carry", [False, True])
    def test_aiv_joins_accept_equivalent_packed_views(self, control, tuple_carry):
        initial = "(x, (x,))" if tuple_carry else "x"
        other = "(dense, (dense,))" if tuple_carry else "dense"
        if control == "if":
            body = f"""if keep:
            result = pl.yield_({initial})
        else:
            result = pl.yield_({other})"""
        elif control == "for":
            body = f"""for i, (carried,) in pl.range(2, init_values=({initial},)):
            result = pl.yield_({other})"""
        else:
            body = f"""for (carried,) in pl.while_(init_values=({initial},)):
            pl.cond(keep)
            result = pl.yield_({other})"""
        target = "result[1][0]" if tuple_carry else "result"
        program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, x: pl.Tensor[[8, 256], pl.FP32], dense: pl.Tensor[[8, 256], pl.FP32],
               keep: pl.Scalar[pl.BOOL]):
        {body}
        tile = pl.tile.load({target}, [0, 0], [8, 256])

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, parent: pl.Tensor[[10, 256], pl.FP32], dense: pl.Tensor[[8, 256], pl.FP32],
             keep: pl.Scalar[pl.BOOL]):
        chunk = pl.tensor.slice(parent, [8, 256], [1, 0])
        self.kernel(chunk, dense, keep)
""")
        after = passes.optimize_orch_tensors()(program)
        load = _calls_named(after, "tile.load")[0]
        tensor = load.args[0].type
        assert isinstance(tensor, ir.TensorType) and tensor.tensor_view is not None
        for dim, expected in zip(tensor.tensor_view.stride, [256, 1], strict=True):
            assert isinstance(dim, ir.ConstInt) and dim.value == expected
        ir.verify_structural_bindings(after)
        ir.assert_structural_equal(after, pl.parse_program(ir.python_print(after)))

    @pytest.mark.parametrize("consume_conflict", [False, True])
    def test_lazy_tuple_constraints_stay_local_to_each_loop(self, consume_conflict):
        operand = "bad[0]" if consume_conflict else "good[0]"
        source = f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, x: pl.Tensor[[8, 256], pl.FP32]):
        return

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, parent: pl.Tensor[[10, 288], pl.FP32], dense: pl.Tensor[[8, 256], pl.FP32]):
        chunk = pl.tensor.slice(parent, [8, 256], [1, 7])
        pair = (chunk, dense)
        for i, (first,) in pl.range(2, init_values=(pair,)):
            bad = pl.yield_((dense, dense))
        for j, (second,) in pl.range(2, init_values=(pair,)):
            good = pl.yield_(second)
        self.kernel({operand})
"""
        before = pl.parse_program(source)
        if consume_conflict:
            with pytest.raises(ValueError, match="consistent physical strides"):
                passes.optimize_orch_tensors()(before)
        else:
            expected = pl.parse_program(
                source.replace(
                    "x: pl.Tensor[[8, 256], pl.FP32]",
                    "x: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(stride=[288, 1])]",
                )
            )
            ir.assert_structural_equal(passes.optimize_orch_tensors()(before), expected)

    def test_unused_wide_tuple_loop_carries_scale_linearly(self):
        def elapsed(size):
            span = ir.Span.unknown()
            tensor = ir.TensorType([8, 256], DataType.FP32)
            args = [ir.Var(f"x{i}", tensor, span) for i in range(size)]
            value = ir.MakeTuple(args, span)
            initial = ir.Var("initial", value.type, span)
            stmts: list[ir.Stmt] = [ir.AssignStmt(initial, value, span)]
            for i in range(size):
                carry = ir.IterArg(f"carry{i}", value.type, initial, span)
                result = ir.Var(f"result{i}", value.type, span)
                stmts.append(
                    ir.ForStmt(
                        ir.Var(f"i{i}", ir.ScalarType(DataType.INDEX), span),
                        ir.ConstInt(0, DataType.INDEX, span),
                        ir.ConstInt(2, DataType.INDEX, span),
                        ir.ConstInt(1, DataType.INDEX, span),
                        [carry],
                        ir.YieldStmt([carry], span),
                        [result],
                        span,
                    )
                )
            func = ir.Function(
                "main", args, [], ir.SeqStmts(stmts, span), span, ir.FunctionType.Orchestration
            )
            before = ir.Program([func], "Wide", span)
            # Time only the pass; printing every carry's wide type measures the
            # instrumentation instead of the analyzer's shared metadata graph.
            with passes.PassContext([], passes.VerificationLevel.NONE):
                start = time.perf_counter()
                after = passes.optimize_orch_tensors()(before)
                duration = time.perf_counter() - start
            ir.assert_structural_equal(after, before)
            return duration

        elapsed(32)
        small = min(elapsed(96) for _ in range(2))
        large = min(elapsed(768) for _ in range(2))
        assert large < small * 24 + 0.1, (small, large)

    @pytest.mark.parametrize("loop_kind", ["for", "while"])
    @pytest.mark.parametrize("tuple_carry", [False, True], ids=["tensor", "nested-tuple"])
    @pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested-loop-and-branch"])
    def test_aiv_loop_carries_reach_simt_specialization(self, loop_kind, tuple_carry, nested):
        initial = "(dst, (dst,))" if tuple_carry else "dst"
        capture = "carried[1][0]" if tuple_carry else "carried"
        yielded = "(updated, (updated,))" if tuple_carry else "updated"
        final = "result[1][0]" if tuple_carry else "result"
        loop = (
            f"for i, (carried,) in pl.range(2, init_values=({initial},)):"
            if loop_kind == "for"
            else f"for carried, i in pl.while_(init_values=({initial}, 0)):\n            pl.cond(i < 2)"
        )
        results = "result" if loop_kind == "for" else "result, final_i"
        values = yielded if loop_kind == "for" else f"{yielded}, i + 1"
        body = (
            "updated = pl.simt.launch(self.update, target, block_dims=(64, 1, 1), result_param_indices=(0,))"
        )
        if nested:
            body = """for j, (inner,) in pl.range(2, init_values=(target,)):
                written = pl.simt.launch(self.update, inner, block_dims=(64, 1, 1), result_param_indices=(0,))
                if i > 0:
                    joined = pl.yield_(written)
                else:
                    joined = pl.yield_(inner)
                updated = pl.yield_(joined)"""
        program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.SIMT)
    def update(self, dst: pl.InOut[pl.Tensor[[8, 256], pl.FP32]]):
        pl.simt.store(dst, (1, 0), pl.const(9.0, pl.FP32))

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, dst: pl.InOut[pl.Tensor[[8, 256], pl.FP32]]):
        {loop}
            target = {capture}
            {body}
            {results} = pl.yield_({values})
        final_target = {final}
        final_updated = pl.simt.launch(
            self.update, final_target, block_dims=(64, 1, 1), result_param_indices=(0,))

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, dst: pl.InOut[pl.Tensor[[10, 288], pl.FP32]]):
        chunk = pl.tensor.slice(dst, [8, 256], [1, 7])
        self.kernel(chunk)
""")
        after = passes.specialize_simt_helpers()(passes.optimize_orch_tensors()(program))
        launches: list[ir.SimtLaunchStmt] = []

        class Collector(ir.IRVisitor):
            def visit_simt_launch_stmt(self, op: ir.SimtLaunchStmt) -> None:
                launches.append(op)

        kernel = after.get_function("kernel")
        assert kernel is not None
        Collector().visit_stmt(kernel.body)
        assert len(launches) == 2
        for launch in launches:
            helper = after.get_function(launch.callee.name)
            assert helper is not None
            for tensor in [launch.args[0], *launch.results, helper.params[0]]:
                assert isinstance(tensor.type, ir.TensorType)
                assert tensor.type.tensor_view is not None
                for dim, expected in zip(tensor.type.tensor_view.stride, [288, 1], strict=True):
                    assert isinstance(dim, ir.ConstInt) and dim.value == expected
        ir.verify_structural_bindings(after)
        ir.assert_structural_equal(after, pl.parse_program(ir.python_print(after)))

    @pytest.mark.parametrize("branch", [False, True], ids=["backedge", "branch-join"])
    def test_aiv_loop_rejects_inconsistent_refined_storage(self, branch):
        body = "result = pl.yield_(dense)"
        if branch:
            body = """if i > 0:
                joined = pl.yield_(carried)
            else:
                joined = pl.yield_(dense)
            result = pl.yield_(joined)"""
        program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, dst: pl.Tensor[[8, 256], pl.FP32], dense: pl.Tensor[[8, 256], pl.FP32]):
        for i, (carried,) in pl.range(2, init_values=(dst,)):
            {body}

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, dst: pl.Tensor[[10, 288], pl.FP32], dense: pl.Tensor[[8, 256], pl.FP32]):
        chunk = pl.tensor.slice(dst, [8, 256], [1, 7])
        self.kernel(chunk, dense)
""")
        with pytest.raises(ValueError, match="after physical stride refinement"):
            passes.optimize_orch_tensors()(program)

    @pytest.mark.parametrize("submit", [False, True], ids=["call", "submit"])
    @pytest.mark.parametrize("multiple", [False, True], ids=["single", "tuple"])
    @pytest.mark.parametrize("writer", ["store", "write"])
    def test_returned_writeback_feeds_next_aiv(self, submit, multiple, writer):
        writes = {
            "store": (
                "tile = pl.tile.load(x, [0, 0], [8, 256])\n        written = pl.tile.store(tile, [0, 0], dst)"
            ),
            "write": (
                "scalar = pl.tensor.read(x, [0, 0])\n        written = pl.tensor.write(dst, [0, 0], scalar)"
            ),
        }
        returns = "pl.Tensor[[8, 256], pl.FP32]"
        returned = "written"
        unpack = "result"
        if multiple:
            returns = f"tuple[{returns}, {returns}]"
            returned = "x, written"
            unpack = "unchanged, result"
        dispatch = f"{unpack} = self.writer(chunk, target)"
        if submit:
            dispatch = (
                "with pl.manual_scope():\n"
                f"            ({unpack}), tid = pl.submit(self.writer, chunk, target)"
            )
        source = f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def writer(self, x: pl.Tensor[[8, 256], pl.FP32],
               dst: pl.InOut[pl.Tensor[[8, 256], pl.FP32]]) -> {returns}:
        {writes[writer]}
        scalar_after = pl.tensor.read(written, [1, 0])
        return {returned}

    @pl.function(type=pl.FunctionType.AIV)
    def reader(self, value: pl.Tensor[[8, 256], pl.FP32]):
        return

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[10, 288], pl.FP32],
             dst: pl.InOut[pl.Tensor[[12, 320], pl.FP32]]):
        chunk = pl.tensor.slice(x, [8, 256], [1, 7])
        target = pl.tensor.slice(dst, [8, 256], [2, 11])
        {dispatch}
        self.reader(result)
"""
        expected = pl.parse_program(
            source.replace(
                "x: pl.Tensor[[8, 256], pl.FP32]",
                "x: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(stride=[288, 1])]",
            )
            .replace(
                "dst: pl.InOut[pl.Tensor[[8, 256], pl.FP32]]",
                "dst: pl.InOut[pl.Tensor[[8, 256], pl.FP32, pl.TensorView(stride=[320, 1])]]",
            )
            .replace(
                "value: pl.Tensor[[8, 256], pl.FP32]",
                "value: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(stride=[320, 1])]",
            )
        )
        ir.assert_structural_equal(passes.optimize_orch_tensors()(pl.parse_program(source)), expected)

    @pytest.mark.parametrize("submit", [False, True], ids=["call", "submit"])
    @pytest.mark.parametrize("writer", ["write", "assemble", "expand_clone"])
    def test_host_writeback_alias_preserves_pitch(self, submit, writer):
        writes = {
            "write": (
                "scalar = pl.tensor.read(chunk, [0, 0, 0])\n"
                "        updated = pl.tensor.write(chunk, [0, 0, 0], scalar)"
            ),
            "assemble": "updated = pl.tensor.assemble(chunk, dense, [0, 0, 0])",
            "expand_clone": "updated = pl.tensor.expand_clone(dense, chunk)",
        }
        dispatch = "self.kernel(updated)"
        if submit:
            dispatch = "with pl.manual_scope():\n            tid = pl.submit(self.kernel, updated)"
        source = f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, value: pl.Tensor[[2, 8, 256], pl.FP32]):
        return

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, parent: pl.Tensor[[2, 10, 288], pl.FP32], dense: pl.Tensor[[2, 8, 256], pl.FP32]):
        chunk = pl.tensor.slice(parent, [2, 8, 256], [0, 1, 7])
        {writes[writer]}
        {dispatch}
"""
        expected = pl.parse_program(
            source.replace(
                "value: pl.Tensor[[2, 8, 256], pl.FP32]",
                "value: pl.Tensor[[2, 8, 256], pl.FP32, pl.TensorView(stride=[2880, 288, 1])]",
            )
        )
        ir.assert_structural_equal(passes.optimize_orch_tensors()(pl.parse_program(source)), expected)

    @pytest.fixture
    def storage_ir(self):
        span = ir.Span.unknown()
        tensor = ir.TensorType([8, 256], DataType.FP32)
        parent = ir.Var("parent", ir.TensorType([10, 288], DataType.FP32), span)
        dense = ir.Var("dense", tensor, span)
        shape = ir.MakeTuple([ir.ConstInt(n, DataType.INDEX, span) for n in (8, 256)], span)
        offset = ir.MakeTuple([ir.ConstInt(n, DataType.INDEX, span) for n in (1, 7)], span)
        sliced = ir.Call(ir.Op("tensor.slice"), [parent, shape, offset], tensor, span)
        chunk = ir.Var("chunk", tensor, span)
        param = ir.Var("value", tensor, span)
        kernel = ir.Function("kernel", [param], [], ir.ReturnStmt([], span), span, type=ir.FunctionType.AIV)
        return span, tensor, parent, dense, shape, offset, sliced, chunk, kernel

    @staticmethod
    def run_program(kernel, params, statements, span, extra_functions=()):
        main = ir.Function(
            "main",
            params,
            [],
            ir.SeqStmts(statements, span) if len(statements) > 1 else statements[0],
            span,
            type=ir.FunctionType.Orchestration,
        )
        return passes.optimize_orch_tensors()(ir.Program([kernel, main, *extra_functions], "Program", span))

    @staticmethod
    def assert_stride(program, stride):
        tensor = program.get_function("kernel").params[0].type
        if stride is None:
            assert tensor.tensor_view is None or not tensor.tensor_view.stride
        else:
            assert [dim.value for dim in tensor.tensor_view.stride] == stride

    @pytest.mark.parametrize("helper", [False, True], ids=["shape_var", "shape_call"])
    def test_tuple_typed_slice_shape(self, storage_ir, helper):
        span, tensor, parent, _, shape, offset, _, chunk, kernel = storage_ir
        shape_var = ir.Var("shape", shape.type, span)
        operand = ir.Call(ir.GlobalVar("shape_helper"), [], shape.type, span) if helper else shape_var
        sliced = ir.Call(ir.Op("tensor.slice"), [parent, operand, offset], tensor, span)
        statements = [
            ir.AssignStmt(shape_var, shape, span),
            ir.AssignStmt(chunk, sliced, span),
            ir.EvalStmt(ir.Call(ir.GlobalVar("kernel"), [chunk], ir.UnknownType(), span), span),
        ]
        helper_function = ir.Function(
            "shape_helper",
            [],
            [shape.type],
            ir.ReturnStmt([shape], span),
            span,
            type=ir.FunctionType.Orchestration,
        )
        # Tuple-valued shape operands are legal core IR; the DSL slice wrapper
        # currently accepts only Python sequences, so retain property checks
        # without the unrelated print/parse instrument for this regression.
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
            after = self.run_program(kernel, [parent], statements, span, [helper_function])
        self.assert_stride(after, [288, 1])

    def test_distributed_parameter_preserves_window(self, storage_ir):
        span, tensor, parent, _, _, _, sliced, chunk, _ = storage_ir
        window = ir.WindowBuffer(
            ir.Var("buffer", ir.PtrType(), span), ir.ConstInt(4096, DataType.INDEX, span)
        )
        distributed = ir.DistributedTensorType(tensor.shape, DataType.FP32, window)
        param = ir.Var("value", distributed, span)
        kernel = ir.Function("kernel", [param], [], ir.ReturnStmt([], span), span, type=ir.FunctionType.AIV)
        distributed_parent = ir.Var(
            "parent", ir.DistributedTensorType(parent.type.shape, DataType.FP32, window), span
        )
        distributed_chunk = ir.Var(chunk.name_hint, distributed, span)
        sliced = ir.Call(sliced.op, [distributed_parent, *sliced.args[1:]], distributed, span)
        statements = [
            ir.AssignStmt(distributed_chunk, sliced, span),
            ir.EvalStmt(ir.Call(ir.GlobalVar("kernel"), [distributed_chunk], ir.UnknownType(), span), span),
        ]
        # This standalone parameter retains an external window reference whose
        # defining communication scope is intentionally outside this pass test.
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
            after = self.run_program(kernel, [distributed_parent], statements, span)
        self.assert_stride(after, [288, 1])
        refined_kernel = after.get_function("kernel")
        assert refined_kernel is not None
        refined = refined_kernel.params[0].type
        assert isinstance(refined, ir.DistributedTensorType)
        assert refined.window_buffer is window

    def test_assignment_reads_old_storage_before_rebinding(self, storage_ir):
        span, tensor, parent, dense, _, _, sliced, chunk, _ = storage_ir
        source = ir.Var("source", tensor, span)
        target = ir.Var("target", tensor, span)
        kernel = ir.Function(
            "kernel",
            [source, (target, ir.ParamDirection.InOut)],
            [tensor],
            ir.ReturnStmt([target], span),
            span,
            type=ir.FunctionType.AIV,
        )
        call = ir.Call(ir.GlobalVar("kernel"), [chunk, dense], tensor, span)
        after = self.run_program(
            kernel,
            [parent, dense],
            [ir.AssignStmt(chunk, sliced, span), ir.AssignStmt(chunk, call, span)],
            span,
        )
        self.assert_stride(after, [288, 1])
        refined_kernel = after.get_function("kernel")
        assert refined_kernel is not None
        target_type = refined_kernel.params[1].type
        assert isinstance(target_type, ir.TensorType)
        assert target_type.tensor_view is None

    @pytest.mark.parametrize("dense_start", [False, True], ids=["sliced_init", "dense_init"])
    @pytest.mark.parametrize("loop_kind", ["for", "while"])
    @pytest.mark.parametrize("consumer", ["body", "result"])
    @pytest.mark.parametrize("backedge", ["same", "writeback", "dense", "branch"])
    def test_loop_stride_consensus(self, storage_ir, loop_kind, consumer, backedge, dense_start):
        span, tensor, parent, dense, _, _, sliced, chunk, kernel = storage_ir
        carried = ir.IterArg("carried", tensor, dense if dense_start else chunk, span)
        result = ir.Var("result", tensor, span)
        one = ir.ConstInt(1, DataType.INDEX, span)
        condition = ir.Var("condition", ir.ScalarType(DataType.BOOL), span)

        def call(value):
            return ir.EvalStmt(ir.Call(ir.GlobalVar("kernel"), [value], ir.UnknownType(), span), span)

        body: list[ir.Stmt] = [call(carried)] if consumer == "body" else []
        other = chunk if dense_start else dense
        yielded = carried if backedge == "same" else other
        if backedge == "branch":
            yielded = ir.Var("merged", tensor, span)
            body.append(
                ir.IfStmt(
                    condition, ir.YieldStmt([carried], span), ir.YieldStmt([other], span), [yielded], span
                )
            )
        if backedge == "writeback":
            offset = ir.MakeTuple([ir.ConstInt(0, DataType.INDEX, span)] * 2, span)
            assembled = ir.Call(ir.Op("tensor.assemble"), [carried, chunk, offset], tensor, span)
            yielded = ir.Var("assembled", tensor, span)
            body.append(ir.AssignStmt(yielded, assembled, span))
        body.append(ir.YieldStmt([yielded], span))
        loop_body = ir.SeqStmts(body, span) if len(body) > 1 else body[0]
        if loop_kind == "for":
            loop = ir.ForStmt(
                ir.Var("i", ir.ScalarType(DataType.INDEX), span),
                ir.ConstInt(0, DataType.INDEX, span),
                ir.ConstInt(2, DataType.INDEX, span),
                one,
                [carried],
                loop_body,
                [result],
                span,
            )
        else:
            loop = ir.WhileStmt(condition, [carried], loop_body, [result], span)
        statements: list[ir.Stmt] = [ir.AssignStmt(chunk, sliced, span), loop]
        if consumer == "result":
            statements.append(call(result))
        if backedge in ("same", "writeback"):
            self.assert_stride(
                self.run_program(kernel, [parent, dense, condition], statements, span),
                None if dense_start else [288, 1],
            )
        else:
            with pytest.raises(
                ValueError, match="consistent physical strides across loop carries and branches"
            ):
                self.run_program(kernel, [parent, dense, condition], statements, span)

    def test_submit_return_alias_with_allocated_out_and_context_suffix(self, storage_ir):
        span, tensor, parent, _, _, _, sliced, chunk, kernel = storage_ir
        source = ir.Var("source", tensor, span)
        allocated = ir.Var("allocated", tensor, span)
        context = ir.Var("context", ir.CommCtxType(), span)
        writer = ir.Function(
            "writer",
            [source, (allocated, ir.ParamDirection.Out), context],
            [tensor, tensor],
            ir.ReturnStmt([source, allocated], span),
            span,
            type=ir.FunctionType.AIV,
        )
        task_id = ir.ScalarType(DataType.TASK_ID)
        result_type = ir.TupleType([tensor, tensor, task_id])
        submit = ir.Submit(ir.GlobalVar("writer"), [chunk, context], [], result_type, span)
        outputs = ir.Var("outputs", result_type, span)
        read = ir.Call(
            ir.GlobalVar("kernel"), [ir.TupleGetItemExpr(outputs, 0, span)], ir.UnknownType(), span
        )
        statements = [
            ir.AssignStmt(chunk, sliced, span),
            ir.AssignStmt(outputs, submit, span),
            ir.EvalStmt(read, span),
        ]
        after = self.run_program(kernel, [parent, context], statements, span, [writer])
        self.assert_stride(after, [288, 1])
        refined_writer = after.get_function("writer")
        assert refined_writer is not None
        allocated_type = refined_writer.params[1].type
        assert isinstance(allocated_type, ir.TensorType)
        assert allocated_type.tensor_view is None
        main = after.get_function("main")
        assert main is not None and isinstance(main.body, ir.SeqStmts)
        assignment = main.body.stmts[1]
        assert isinstance(assignment, ir.AssignStmt)
        launch = assignment.value
        assert isinstance(launch, ir.Submit)
        assert len(launch.args) == 2
        assert launch.args[1] is context
        assert isinstance(launch.type, ir.TupleType)
        ir.assert_structural_equal(launch.type.types[-1], task_id)

    @pytest.mark.parametrize("pitch", [256, 320])
    @pytest.mark.parametrize("allocated_first", [False, True])
    @pytest.mark.parametrize("context_suffix", [False, True])
    def test_submit_allocated_out_stride_consensus(self, pitch, allocated_first, context_suffix):
        span = ir.Span.unknown()
        tensor = ir.TensorType([8, 256], DataType.FP32)
        parent = ir.Var("parent", ir.TensorType([10, pitch], DataType.FP32), span)
        dst = ir.Var("dst", tensor, span)
        context = ir.Var("context", ir.CommCtxType(), span)
        contexts = [context] if context_suffix else []
        kernel = ir.Function(
            "kernel",
            [(dst, ir.ParamDirection.Out), *contexts],
            [tensor],
            ir.ReturnStmt([dst], span),
            span,
            type=ir.FunctionType.AIV,
        )
        shape = ir.MakeTuple([ir.ConstInt(n, DataType.INDEX, span) for n in (8, 256)], span)
        offset = ir.MakeTuple([ir.ConstInt(n, DataType.INDEX, span) for n in (1, 0)], span)
        chunk = ir.Var("chunk", tensor, span)
        sliced = ir.Call(ir.Op("tensor.slice"), [parent, shape, offset], tensor, span)
        result_type = ir.TupleType([tensor, ir.ScalarType(DataType.TASK_ID)])
        supplied = ir.AssignStmt(
            ir.Var("supplied", result_type, span),
            ir.Submit(ir.GlobalVar("kernel"), [chunk, *contexts], [], result_type, span),
            span,
        )
        allocated = ir.AssignStmt(
            ir.Var("allocated", result_type, span),
            ir.Submit(ir.GlobalVar("kernel"), contexts, [], result_type, span),
            span,
        )
        calls = [allocated, supplied] if allocated_first else [supplied, allocated]
        statements = [ir.AssignStmt(chunk, sliced, span), *calls]
        if pitch != 256:
            with pytest.raises(ValueError, match="one consistent static stride"):
                self.run_program(kernel, [parent, *contexts], statements, span)
        else:
            self.assert_stride(self.run_program(kernel, [parent, *contexts], statements, span), [256, 1])

    @pytest.mark.parametrize("pitch", [256, 320])
    def test_submit_allocated_out_checks_explicit_stride(self, pitch):
        source = f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, dst: pl.Out[pl.Tensor[[8, 256], pl.FP32,
               pl.TensorView(stride=[{pitch}, 1])]]) -> pl.Tensor[[8, 256], pl.FP32]:
        return dst

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self):
        result, tid = pl.submit(self.kernel)
"""
        before = pl.parse_program(source)
        if pitch != 256:
            with pytest.raises(ValueError, match="explicit physical strides"):
                passes.optimize_orch_tensors()(before)
        else:
            self.assert_stride(passes.optimize_orch_tensors()(before), [256, 1])

    @pytest.mark.parametrize("layout", ["ND", "DN"])
    def test_submit_allocated_out_checks_layout_only_declaration(self, layout):
        # Runtime outputs are packed ND, regardless of the formal's view.
        source = f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, dst: pl.Out[pl.Tensor[[8, 256], pl.FP32,
               pl.TensorView(layout=pl.{layout})]]) -> pl.Tensor[[8, 256], pl.FP32]:
        return dst

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self):
        result, tid = pl.submit(self.kernel)
"""
        before = pl.parse_program(source)
        if layout == "DN":
            with pytest.raises(ValueError, match="explicit layout"):
                passes.optimize_orch_tensors()(before)
        else:
            ir.assert_structural_equal(passes.optimize_orch_tensors()(before), before)

    @pytest.mark.parametrize("nested", [False, True])
    def test_conditional_parameter_rebinding_joins_following_use(self, storage_ir, nested):
        # A parameter has an incoming binding even without an AssignStmt.
        span, _, parent, dense, _, _, sliced, _, kernel = storage_ir
        condition = ir.Var("condition", ir.ScalarType(DataType.BOOL), span)
        write = ir.AssignStmt(dense, sliced, span)
        if nested:
            write = ir.IfStmt(condition, write, None, [], span)
        statements = [
            ir.IfStmt(condition, write, None, [], span),
            ir.EvalStmt(ir.Call(ir.GlobalVar("kernel"), [dense], ir.UnknownType(), span), span),
        ]
        with pytest.raises(ValueError, match="consistent physical strides"):
            self.run_program(kernel, [parent, dense, condition], statements, span)

    def test_nested_branch_local_bindings_scale_linearly(self):
        def elapsed(depth):
            span = ir.Span.unknown()
            tensor = ir.TensorType([8, 256], DataType.FP32)
            value = ir.Var("value", tensor, span)
            condition = ir.Var("condition", ir.ScalarType(DataType.BOOL), span)
            body = ir.ReturnStmt([], span)
            for level in range(depth):
                # Each local definition must be removed at its own branch exit,
                # rather than repeatedly joined at every enclosing branch.
                locals_ = [
                    ir.AssignStmt(ir.Var(f"local{level}_{i}", tensor, span), value, span) for i in range(32)
                ]
                body = ir.IfStmt(condition, ir.SeqStmts([*locals_, body], span), None, [], span)
            func = ir.Function("main", [value, condition], [], body, span, ir.FunctionType.Orchestration)
            before = ir.Program([func], "Nested", span)
            with passes.PassContext([], passes.VerificationLevel.NONE):
                start = time.perf_counter()
                after = passes.optimize_orch_tensors()(before)
                duration = time.perf_counter() - start
            # Compare one level at a time to avoid the comparator's recursion
            # limit; this check is outside the timed pass invocation.
            after_main = after.get_function("main")
            assert after_main is not None
            actual = after_main.body
            expected = body
            for _ in range(depth):
                assert isinstance(actual, ir.IfStmt)
                assert isinstance(expected, ir.IfStmt)
                ir.assert_structural_equal(actual.condition, expected.condition)
                assert isinstance(actual.then_body, ir.SeqStmts)
                assert isinstance(expected.then_body, ir.SeqStmts)
                for got, want in zip(actual.then_body.stmts[:-1], expected.then_body.stmts[:-1], strict=True):
                    ir.assert_structural_equal(got, want)
                actual = actual.then_body.stmts[-1]
                expected = expected.then_body.stmts[-1]
            ir.assert_structural_equal(actual, expected)
            return duration

        elapsed(8)
        small = min(elapsed(32) for _ in range(3))
        large = min(elapsed(256) for _ in range(3))
        assert large < small * 24 + 0.01, (small, large)

    @pytest.mark.parametrize("tuple_alias", [False, True])
    @pytest.mark.parametrize("write_then", [False, True])
    @pytest.mark.parametrize("nested", [False, True])
    def test_conditional_rebinding_does_not_leak_to_sibling(
        self, storage_ir, tuple_alias, write_then, nested
    ):
        span, tensor, parent, dense, _, _, sliced, _, kernel = storage_ir
        condition = ir.Var("condition", ir.ScalarType(DataType.BOOL), span)
        statements = []
        value = dense
        operand = dense
        replacement = sliced
        if tuple_alias:
            initial = ir.MakeTuple([dense], span)
            value = ir.Var("values", initial.type, span)
            statements.append(ir.AssignStmt(value, initial, span))
            operand = ir.TupleGetItemExpr(value, 0, span)
            replacement = ir.MakeTuple([sliced], span)
        write = ir.AssignStmt(value, replacement, span)
        if nested:
            write = ir.IfStmt(condition, write, None, [], span)
        read = ir.EvalStmt(ir.Call(ir.GlobalVar("kernel"), [operand], ir.UnknownType(), span), span)
        statements.append(
            ir.IfStmt(condition, write if write_then else read, read if write_then else write, [], span)
        )
        self.assert_stride(self.run_program(kernel, [parent, dense, condition], statements, span), None)

    @pytest.mark.parametrize("same_pitch", [False, True])
    @pytest.mark.parametrize("has_else", [False, True])
    def test_conditional_rebinding_joins_following_use(self, storage_ir, same_pitch, has_else):
        span, tensor, parent, dense, _, _, sliced, chunk, kernel = storage_ir
        condition = ir.Var("condition", ir.ScalarType(DataType.BOOL), span)
        initial = sliced if same_pitch else dense
        statements: list[ir.Stmt] = [ir.AssignStmt(chunk, initial, span)]
        then_body = ir.AssignStmt(chunk, sliced, span)
        else_body = ir.AssignStmt(chunk, initial, span) if has_else else None
        statements.append(ir.IfStmt(condition, then_body, else_body, [], span))
        statements.append(ir.EvalStmt(ir.Call(ir.GlobalVar("kernel"), [chunk], ir.UnknownType(), span), span))
        if same_pitch:
            self.assert_stride(
                self.run_program(kernel, [parent, dense, condition], statements, span), [288, 1]
            )
        else:
            with pytest.raises(ValueError, match="consistent physical strides"):
                self.run_program(kernel, [parent, dense, condition], statements, span)

    @pytest.mark.parametrize("depth", [2, 5])
    @pytest.mark.parametrize("alias", [False, True], ids=["projection", "tuple_alias"])
    def test_nested_tuple_storage(self, depth, alias):
        bindings = "inner0 = (chunk,)"
        for level in range(1, depth):
            bindings += f"\n        inner{level} = (inner{level - 1},)"
        operand = f"inner{depth - 1}"
        if alias:
            bindings += f"\n        alias = {operand}[0]"
            operand = "alias"
            depth -= 1
        operand += "[0]" * depth
        source = f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, value: pl.Tensor[[8, 256], pl.FP32]):
        return

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, parent: pl.Tensor[[10, 288], pl.FP32]):
        chunk = pl.tensor.slice(parent, [8, 256], [1, 7])
        {bindings}
        self.kernel({operand})
"""
        expected = pl.parse_program(
            source.replace(
                "value: pl.Tensor[[8, 256], pl.FP32]",
                "value: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(stride=[288, 1])]",
            )
        )
        ir.assert_structural_equal(passes.optimize_orch_tensors()(pl.parse_program(source)), expected)

    @pytest.mark.parametrize("loop_kind", ["for", "while"])
    @pytest.mark.parametrize("conflict", [False, True])
    def test_nested_tuple_loop_storage(self, loop_kind, conflict):
        loop = "for i, (carried,) in pl.range(2, init_values=(outer,)):"
        condition = ""
        if loop_kind == "while":
            loop = "for (carried,) in pl.while_(init_values=(outer,)):"
            condition = "pl.cond(keep)"
        yielded = "((dense,),)" if conflict else "carried"
        source = f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, value: pl.Tensor[[8, 256], pl.FP32]):
        return

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, parent: pl.Tensor[[10, 288], pl.FP32],
             dense: pl.Tensor[[8, 256], pl.FP32], keep: pl.Scalar[pl.BOOL]):
        chunk = pl.tensor.slice(parent, [8, 256], [1, 7])
        outer = ((chunk,),)
        {loop}
            {condition}
            self.kernel(carried[0][0])
            result = pl.yield_({yielded})
"""
        before = pl.parse_program(source)
        if conflict:
            with pytest.raises(ValueError, match="consistent physical strides"):
                passes.optimize_orch_tensors()(before)
        else:
            expected = pl.parse_program(
                source.replace(
                    "value: pl.Tensor[[8, 256], pl.FP32]",
                    "value: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(stride=[288, 1])]",
                )
            )
            ir.assert_structural_equal(passes.optimize_orch_tensors()(before), expected)

    @pytest.mark.parametrize("submit", [False, True], ids=["call", "submit"])
    @pytest.mark.parametrize("conflict", [False, True])
    def test_graph_parameter_stride_consensus(self, submit, conflict):
        first = "first = self.outer(chunk)"
        second = "second = self.outer(dense)" if conflict else "second = self.outer(chunk)"
        if submit:
            first = "first, first_id = pl.submit(self.outer, chunk)"
            second = f"second, second_id = pl.submit(self.outer, {'dense' if conflict else 'chunk'})"
        source = f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, value: pl.Tensor[[8, 256], pl.FP32]) -> pl.Tensor[[8, 256], pl.FP32]:
        return value

    @pl.function(type=pl.FunctionType.Graph)
    def layer(self, x: pl.Tensor[[8, 256], pl.FP32]) -> pl.Tensor[[8, 256], pl.FP32]:
        result = self.kernel(x)
        return result

    @pl.function(type=pl.FunctionType.Graph)
    def outer(self, x: pl.Tensor[[8, 256], pl.FP32]) -> pl.Tensor[[8, 256], pl.FP32]:
        result = self.layer(x)
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, parent: pl.Tensor[[10, 288], pl.FP32], dense: pl.Tensor[[8, 256], pl.FP32]):
        chunk = pl.tensor.slice(parent, [8, 256], [1, 7])
        {first}
        {second}
"""
        before = pl.parse_program(source)
        if conflict:
            with pytest.raises(ValueError, match="consistent physical strides"):
                passes.optimize_orch_tensors()(before)
        else:
            expected = pl.parse_program(
                source.replace(
                    "value: pl.Tensor[[8, 256], pl.FP32]",
                    "value: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(stride=[288, 1])]",
                )
            )
            ir.assert_structural_equal(passes.optimize_orch_tensors()(before), expected)

    @pytest.mark.parametrize("submit", [False, True])
    @pytest.mark.parametrize("nested", [False, True])
    @pytest.mark.parametrize("tuple_result", [False, True])
    def test_graph_created_view_return_preserves_stride(self, submit, nested, tuple_result):
        result_type = "pl.Tensor[[8, 256], pl.FP32]"
        returned = "chunk"
        if tuple_result:
            result_type = f"tuple[{result_type}, {result_type}]"
            returned = "chunk, chunk"
        make = "chunk = pl.tensor.slice(parent, [8, 256], [1, 7])"
        inner = ""
        if nested:
            inner = f"""
    @pl.function(type=pl.FunctionType.Graph)
    def inner(self, parent: pl.Tensor[[10, 288], pl.FP32]) -> pl.Tensor[[8, 256], pl.FP32]:
        {make}
        return chunk
"""
            make = "chunk = self.inner(parent)"
        call = "result = self.window(parent)"
        if submit:
            call = "result, tid = pl.submit(self.window, parent)"
        operand = "result[0]" if tuple_result else "result"
        if submit and tuple_result:
            call = "(result, other), tid = pl.submit(self.window, parent)"
            operand = "result"
        source = f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, value: pl.Tensor[[8, 256], pl.FP32]):
        return
{inner}
    @pl.function(type=pl.FunctionType.Graph)
    def window(self, parent: pl.Tensor[[10, 288], pl.FP32]) -> {result_type}:
        {make}
        return {returned}

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, parent: pl.Tensor[[10, 288], pl.FP32]):
        {call}
        self.kernel({operand})
"""
        expected = pl.parse_program(
            source.replace(
                "value: pl.Tensor[[8, 256], pl.FP32]",
                "value: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(stride=[288, 1])]",
            )
        )
        ir.assert_structural_equal(passes.optimize_orch_tensors()(pl.parse_program(source)), expected)

    @pytest.mark.parametrize("pitch", [288, 320])
    @pytest.mark.parametrize("window_first", [False, True])
    def test_graph_return_stride_consensus_includes_later_calls(self, pitch, window_first):
        first, second = ("parent_window", "dense") if window_first else ("dense", "parent_window")
        source = f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, value: pl.Tensor[[8, 256], pl.FP32]):
        return

    @pl.function(type=pl.FunctionType.Graph)
    def window(self, parent: pl.Tensor[[10, 288], pl.FP32]) -> pl.Tensor[[8, 256], pl.FP32]:
        chunk = pl.tensor.slice(parent, [8, 256], [1, 7])
        return chunk

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, dense: pl.Tensor[[10, 288], pl.FP32], big: pl.Tensor[[12, {pitch}], pl.FP32]):
        parent_window = pl.tensor.slice(big, [10, 288], [1, 0])
        first = self.window({first})
        self.kernel(first)
        second = self.window({second})
"""
        before = pl.parse_program(source)
        if pitch != 288:
            with pytest.raises(ValueError, match="consistent physical strides"):
                passes.optimize_orch_tensors()(before)
        else:
            self.assert_stride(passes.optimize_orch_tensors()(before), [288, 1])

    @pytest.mark.parametrize("single_tuple", [False, True])
    @pytest.mark.parametrize("submit", [False, True])
    @pytest.mark.parametrize("index", [0, 1])
    def test_graph_forwarded_tuple_return_signature(self, storage_ir, single_tuple, submit, index):
        span, tensor, parent, _, _, _, sliced, chunk, kernel = storage_ir
        pair_type = ir.TupleType([tensor, tensor])
        pair = ir.Var("pair", pair_type, span)
        returns = [pair_type] if single_tuple else [tensor, tensor]
        window = ir.Function(
            "window",
            [parent],
            returns,
            ir.SeqStmts(
                [
                    ir.AssignStmt(chunk, sliced, span),
                    ir.AssignStmt(pair, ir.MakeTuple([chunk, chunk], span), span),
                    ir.ReturnStmt([pair], span),
                ],
                span,
            ),
            span,
            type=ir.FunctionType.Graph,
        )
        result_type = ir.TupleType([*returns, ir.ScalarType(DataType.TASK_ID)]) if submit else pair_type
        result = ir.Var("result", result_type, span)
        launch = (
            ir.Submit(ir.GlobalVar("window"), [parent], [], result_type, span)
            if submit
            else ir.Call(ir.GlobalVar("window"), [parent], result_type, span)
        )
        container = ir.TupleGetItemExpr(result, 0, span) if submit and single_tuple else result
        value = ir.TupleGetItemExpr(container, index, span)
        statements = [
            ir.AssignStmt(result, launch, span),
            ir.EvalStmt(ir.Call(ir.GlobalVar("kernel"), [value], ir.UnknownType(), span), span),
        ]
        self.assert_stride(self.run_program(kernel, [parent], statements, span, [window]), [288, 1])

    @pytest.mark.parametrize("two_helpers", [False, True])
    @pytest.mark.parametrize("submit", [False, True])
    def test_later_graph_call_propagates_slice_provenance(self, two_helpers, submit):
        second = "forward2" if two_helpers else "forward1"
        helper = (
            """
    @pl.function(type=pl.FunctionType.Graph)
    def forward2(self, x: pl.Tensor[[8, 256], pl.FP32]):
        self.kernel(x)
"""
            if two_helpers
            else ""
        )
        calls = ["self.forward1(whole)"]
        if two_helpers:
            calls.append("self.forward2(whole)")
        calls.append(f"self.{second}(chunk)")
        if submit:
            calls = [f"tid{i} = pl.submit({call[:-1].replace('(', ', ', 1)})" for i, call in enumerate(calls)]
        body = "\n        ".join(calls)
        source = f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, value: pl.Tensor[[8, 256], pl.FP32]):
        return

    @pl.function(type=pl.FunctionType.Graph)
    def forward1(self, x: pl.Tensor[[8, 256], pl.FP32]):
        self.kernel(x)
{helper}
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, whole: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(stride=[288, 1])],
             parent: pl.Tensor[[10, 288], pl.FP32]):
        chunk = pl.tensor.slice(parent, [8, 256], [1, 7])
        {body}
"""
        expected = pl.parse_program(
            source.replace(
                "value: pl.Tensor[[8, 256], pl.FP32]",
                "value: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(stride=[288, 1])]",
            )
        )
        ir.assert_structural_equal(passes.optimize_orch_tensors()(pl.parse_program(source)), expected)

    @pytest.mark.parametrize("rank", [2, 3])
    @pytest.mark.parametrize("mixed", [False, True])
    def test_allocated_out_dynamic_leading_extent_keeps_static_strides(self, rank, mixed):
        dims = "N, 256" if rank == 2 else "N, 8, 256"
        strides = [256, 1] if rank == 2 else [2048, 256, 1]
        tensor = f"pl.Tensor[[{dims}], pl.FP32, pl.TensorView(stride={strides})]"
        supplied = "supplied, first_tid = pl.submit(self.kernel, whole)" if mixed else ""
        source = f"""
N = pl.dynamic("N")
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, dst: pl.Out[{tensor}]) -> {tensor}:
        return dst

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, whole: {tensor}):
        {supplied}
        allocated, tid = pl.submit(self.kernel)
"""
        before = pl.parse_program(source)
        ir.assert_structural_equal(passes.optimize_orch_tensors()(before), before)

    def test_external_return_attribute_follows_refined_parameter(self, tmp_path):
        cpp = tmp_path / "external.cpp"
        cpp.write_text('extern "C" void kernel_entry(long long* args) {}\n')
        source = """
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV, external_source="external.cpp")
    def kernel(self, dst: pl.Out[pl.Tensor[[8, 256], pl.FP32]]) -> pl.Tensor[[8, 256], pl.FP32]:
        pl.func_attr({"external_return_params": [dst]})
        ...

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, parent: pl.Tensor[[10, 288], pl.FP32]):
        chunk = pl.tensor.slice(parent, [8, 256], [1, 7])
        result = self.kernel(chunk)
"""
        source = source.replace('"external.cpp"', repr(str(cpp)))
        after = passes.optimize_orch_tensors()(pl.parse_program(source))
        expected = pl.parse_program(
            source.replace(
                "dst: pl.Out[pl.Tensor[[8, 256], pl.FP32]]",
                "dst: pl.Out[pl.Tensor[[8, 256], pl.FP32, pl.TensorView(stride=[288, 1])]]",
            )
        )
        ir.assert_structural_equal(after, expected)
        function = after.get_function("kernel")
        assert function is not None
        assert function.attrs["external_return_params"][0] is function.params[0]
        passes.normalize_return_order()(after)

    @pytest.mark.parametrize("pitch", ["dynamic", "different_callers"], ids=["dynamic", "caller-specific"])
    @pytest.mark.parametrize("explicit_view", [False, True], ids=["descriptor", "explicit-view"])
    def test_external_aiv_can_use_runtime_pitch(self, tmp_path, pitch, explicit_view):
        cpp = tmp_path / "external.cpp"
        cpp.write_text('extern "C" void kernel_entry(long long* args) {}\n')
        parent_width = "N" if pitch == "dynamic" else "288"
        second_call = ""
        if pitch == "different_callers":
            second_call = "second = pl.tensor.slice(other, [8, 256], [1, 7])\n        self.dispatch(second)"
        view = ", pl.TensorView(stride=[288, 1])" if explicit_view else ""
        program = pl.parse_program(f"""
N = pl.dynamic("N")
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV, external_source={str(cpp)!r})
    def kernel(self, x: pl.Tensor[[8, 256], pl.FP32{view}]):
        ...

    @pl.function(type=pl.FunctionType.Orchestration)
    def dispatch(self, x: pl.Tensor[[8, 256], pl.FP32]):
        self.kernel(x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, parent: pl.Tensor[[10, {parent_width}], pl.FP32], other: pl.Tensor[[10, 320], pl.FP32]):
        first = pl.tensor.slice(parent, [8, 256], [1, 7])
        self.dispatch(first)
        {second_call}
""")
        if explicit_view:
            with pytest.raises(ValueError, match="requires (consistent|static) physical strides"):
                passes.optimize_orch_tensors()(program)
        else:
            ir.assert_structural_equal(passes.optimize_orch_tensors()(program), program)

    def test_external_aiv_rejects_pitched_reshape(self, tmp_path):
        cpp = tmp_path / "external.cpp"
        cpp.write_text('extern "C" void kernel_entry(long long* args) {}\n')
        program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV, external_source={str(cpp)!r})
    def kernel(self, x: pl.Tensor[[4, 512], pl.FP32]):
        ...

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, parent: pl.Tensor[[10, 288], pl.FP32]):
        chunk = pl.tensor.slice(parent, [8, 256], [1, 7])
        self.kernel(pl.tensor.reshape(chunk, [4, 512]))
""")
        with pytest.raises(ValueError, match="tensor.reshape requires packed ND storage"):
            passes.optimize_orch_tensors()(program)

    def test_refined_tile_store_preserves_attrs(self):
        program = pl.parse_program("""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, dst: pl.InOut[pl.Tensor[[8, 256], pl.FP32]]):
        tile = pl.tile.load(dst, [0, 0], [8, 256])
        result = pl.tile.store(tile, [0, 0], dst)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, parent: pl.Tensor[[10, 288], pl.FP32]):
        chunk = pl.tensor.slice(parent, [8, 256], [1, 7])
        self.kernel(chunk)
""")
        directions = [ir.ArgDirection.Input, ir.ArgDirection.Scalar, ir.ArgDirection.InOut]

        class TagStore(ir.IRMutator):
            def visit_call(self, op):
                if op.op.name == ir.get_op("tile.store").name:
                    return ir.Call(
                        op.op, op.args, op.kwargs, {"arg_directions": directions}, op.type, op.span
                    )
                return super().visit_call(op)

        program = TagStore().visit_program(program)
        after = passes.optimize_orch_tensors()(program)
        store = _calls_named(after, "tile.store")[0]
        assert list(store.arg_directions) == directions
        refined_kernel = after.get_function("kernel")
        assert refined_kernel is not None
        assert store.args[2] is refined_kernel.params[0]


class TestSliceInputStrides:
    """Pattern 4: Attach parent-derived strides to In params for slice patterns."""

    @pytest.mark.parametrize("submit", [False, True], ids=["call", "submit"])
    @pytest.mark.parametrize("direction", ["InOut", "Out"])
    @pytest.mark.parametrize("alias", [False, True], ids=["direct", "tuple_alias"])
    def test_aiv_sliced_read_and_write_arguments(self, submit, direction, alias):
        call = "result = self.kernel(chunk, target)"
        if submit:
            call = "with pl.manual_scope():\n            result, tid = pl.submit(self.kernel, chunk, target)"
        aliases = ""
        if alias:
            aliases = (
                "pair = (chunk, target)\n        other = pair\n"
                "        chunk = other[0]\n        target = other[1]"
            )
        source = f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, x: pl.Tensor[[8, 256], pl.FP32],
               dst: pl.{direction}[pl.Tensor[[8, 256], pl.FP32]]) -> pl.Tensor[[8, 256], pl.FP32]:
        tile = pl.tile.load(x, [0, 0], [8, 256])
        result = pl.tile.store(tile, [0, 0], dst)
        return result

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[10, 288], pl.FP32],
             dst: pl.InOut[pl.Tensor[[12, 320], pl.FP32]]) -> pl.Tensor[[8, 256], pl.FP32]:
        outer = pl.tensor.slice(x, [9, 280], [1, 2])
        chunk = pl.tensor.slice(outer, [8, 256], [0, 5])
        target = pl.tensor.slice(dst, [8, 256], [2, 11])
        {aliases}
        {call}
        return result
"""
        before = pl.parse_program(source)
        expected = pl.parse_program(
            source.replace(
                "x: pl.Tensor[[8, 256], pl.FP32]",
                "x: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(stride=[288, 1])]",
            ).replace(
                f"dst: pl.{direction}[pl.Tensor[[8, 256], pl.FP32]]",
                f"dst: pl.{direction}[pl.Tensor[[8, 256], pl.FP32, pl.TensorView(stride=[320, 1])]]",
            )
        )
        ir.assert_structural_equal(passes.optimize_orch_tensors()(before), expected)

    @pytest.mark.parametrize("second", ["dense", "different_slice"])
    @pytest.mark.parametrize("reverse", [False, True])
    def test_aiv_conflicting_slice_pitches_are_rejected(self, second, reverse):
        first_arg = "chunk"
        second_arg = "dense" if second == "dense" else "other"
        if reverse:
            first_arg, second_arg = second_arg, first_arg
        program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, x: pl.Tensor[[8, 256], pl.FP32]) -> pl.Tensor[[8, 256], pl.FP32]:
        return x

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[10, 288], pl.FP32], y: pl.Tensor[[10, 320], pl.FP32],
             dense: pl.Tensor[[8, 256], pl.FP32]) -> pl.Tensor[[8, 256], pl.FP32]:
        chunk = pl.tensor.slice(x, [8, 256], [1, 7])
        other = pl.tensor.slice(y, [8, 256], [1, 7])
        first = self.kernel({first_arg})
        second = self.kernel({second_arg})
        return second
""")
        with pytest.raises(ValueError, match="consistent static stride"):
            passes.optimize_orch_tensors()(program)

    @pytest.mark.parametrize("layout, strides", [("ND", [96, 32, 1]), ("DN", [96, 1, 3])])
    def test_aiv_slice_dropped_axis_preserves_parent_layout(self, layout, strides):
        source = f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, x: pl.Tensor[[2, 3, 16], pl.FP32, pl.TensorView(layout=pl.TensorLayout.{layout})]):
        return

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, parent: pl.Tensor[[2, 1, 3, 32], pl.FP32, pl.TensorView(layout=pl.TensorLayout.{layout})]):
        chunk = pl.tensor.slice(parent, [2, 1, 3, 16], [0, 0, 0, 7], drop_dims=[1])
        self.kernel(chunk)
        return
"""
        expected = pl.parse_program(
            source.replace(
                f"x: pl.Tensor[[2, 3, 16], pl.FP32, pl.TensorView(layout=pl.TensorLayout.{layout})]",
                f"x: pl.Tensor[[2, 3, 16], pl.FP32, "
                f"pl.TensorView(stride={strides}, layout=pl.TensorLayout.{layout})]",
            )
        )
        ir.assert_structural_equal(passes.optimize_orch_tensors()(pl.parse_program(source)), expected)

    def test_aiv_dynamic_slice_pitch_is_rejected(self):
        N = pl.dynamic("N")

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(self, x: pl.Tensor[[8, 256], pl.FP32]):
                return

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, parent: pl.Tensor[[10, N], pl.FP32]):
                chunk = pl.tensor.slice(parent, [8, 256], [1, 7])
                self.kernel(chunk)

        with pytest.raises(ValueError, match="static physical strides"):
            passes.optimize_orch_tensors()(Program)

    @pytest.mark.parametrize("layout", ["dense", "crop", "mixed", "clamp"])
    @pytest.mark.parametrize("nested", [False, True], ids=["direct", "orchestrated"])
    def test_external_dynamic_pitch_requires_packed_storage_at_every_call(self, tmp_path, layout, nested):
        cpp = tmp_path / "external.cpp"
        cpp.write_text('extern "C" void kernel_entry(long long* args) {}\n')
        callee = "dispatch" if nested else "kernel"
        calls = {
            "dense": f"self.{callee}(slab)",
            "crop": f"self.{callee}(crop)",
            "mixed": f"self.{callee}(slab)\n        self.{callee}(crop)",
            "clamp": f"self.{callee}(clamped)",
        }
        source = f"""
W = pl.dynamic("W")
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV, external_source={str(cpp)!r})
    def kernel(self, x: pl.Tensor[[8, W], pl.FP32]):
        ...

    @pl.function(type=pl.FunctionType.Orchestration)
    def dispatch(self, x: pl.Tensor[[8, W], pl.FP32]):
        self.kernel(x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, data: pl.Tensor[[2, 8, W], pl.FP32], wider: pl.Tensor[[2, 8, W + 8], pl.FP32]):
        slab = pl.tensor.slice(data, [1, 8, W], [1, 0, 0], drop_dims=[0])
        crop = pl.tensor.slice(wider, [1, 8, W], [1, 0, 7], drop_dims=[0])
        clamped = pl.tensor.slice(data, [1, 8, W], [1, 0, 1], drop_dims=[0], clamp=True)
        {calls[layout]}
"""
        program = pl.parse_program(source)
        if layout == "dense":
            ir.assert_structural_equal(passes.optimize_orch_tensors()(program), program)
        else:
            with pytest.raises(
                ValueError, match="consistent physical strides|static stride|static physical stride"
            ):
                passes.optimize_orch_tensors()(program)

    @pytest.mark.parametrize("use_backedge", [False, True], ids=["later_caller", "loop_backedge"])
    def test_external_derived_window_requires_consistent_parent_pitch(self, tmp_path, use_backedge):
        cpp = tmp_path / "external.cpp"
        cpp.write_text('extern "C" void kernel_entry(long long* args) {}\n')
        body = (
            "for _, (carried,) in pl.range(2, init_values=(parent,)):\n"
            "            self.dispatch(carried)\n"
            "            next_parent = pl.tensor.slice(wider, [10, W], [0, 0])\n"
            "            unused = pl.yield_(next_parent)"
            if use_backedge
            else "self.dispatch(parent)\n        self.dispatch(wider)"
        )
        source = f"""
W = pl.dynamic("W")
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV, external_source={str(cpp)!r})
    def kernel(self, x: pl.Tensor[[8, W], pl.FP32]):
        ...

    @pl.function(type=pl.FunctionType.Orchestration)
    def dispatch(self, x: pl.Tensor[[10, W], pl.FP32]):
        slab = pl.tensor.slice(x, [8, W], [1, 0])
        self.kernel(slab)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, parent: pl.Tensor[[10, W], pl.FP32], wider: pl.Tensor[[10, W + 8], pl.FP32]):
        {body}
"""
        with pytest.raises(ValueError, match="consistent physical strides|static physical strides"):
            passes.optimize_orch_tensors()(pl.parse_program(source))

    def test_aiv_blocked_slice_keeps_layout_legalization(self):
        """NZ slices must keep the existing blocked-layout lowering path."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(self, x: pl.Tensor[[32, 32], pl.FP32, pl.NZ]):
                return

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, parent: pl.Tensor[[64, 64], pl.FP32, pl.NZ]):
                chunk = pl.tensor.slice(parent, [32, 32], [0, 0])
                self.kernel(chunk)

        ir.assert_structural_equal(passes.optimize_orch_tensors()(Program), Program)

    def test_aiv_conflicting_explicit_stride_is_rejected(self):
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(
                self,
                x: pl.Tensor[[8, 256], pl.FP32, pl.TensorView(stride=[256, 1], layout=pl.TensorLayout.ND)],
            ):
                return

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, parent: pl.Tensor[[10, 288], pl.FP32]):
                chunk = pl.tensor.slice(parent, [8, 256], [1, 7])
                self.kernel(chunk)

        with pytest.raises(ValueError, match="explicit physical strides"):
            passes.optimize_orch_tensors()(Program)

    @pytest.mark.parametrize(
        "access",
        [
            pytest.param("alias = x\n        first = alias", id="alias_chain"),
            pytest.param("pair = (x, x)\n        first = pair[0]", id="tuple_projection"),
            pytest.param(
                "alias = x\n        pair = (alias, x)\n        other = pair\n        first = other[1]",
                id="tuple_alias",
            ),
        ],
    )
    def test_slice_input_stride_propagates_through_aliases(self, access):
        """Pattern 4 must update every alias without relying on windowization."""
        source = f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[64, 64], pl.FP32],
        out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        {access}
        tile = pl.tile.load(first, [0, 0], [64, 64], [64, 64])
        stored = pl.tile.store(tile, [0, 0], out)
        return stored

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[128, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        chunk = pl.tensor.slice(x, [64, 64], [32, 32])
        result = self.kernel(chunk, out)
        return result
"""
        before = pl.parse_program(source)
        # The golden infers all local value types from an explicitly strided
        # parameter. It never runs the pass under test.
        expected = pl.parse_program(
            source.replace(
                "x: pl.Tensor[[64, 64], pl.FP32]",
                "x: pl.Tensor[[64, 64], pl.FP32, pl.TensorView(stride=[128, 1], layout=pl.TensorLayout.ND)]",
            )
        )
        after = passes.optimize_orch_tensors()(before)
        ir.assert_structural_equal(after, expected)
        assert after.get_function("kernel__windowed") is None

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

    def test_nz_in_param_gets_no_parent_stride(self):
        """An NZ In param fed by a slice keeps an empty stride.

        Row-major parent strides do not describe fractal-blocked NZ bytes.
        BlockNzTensorViews derives the real strides from the blocked shape and
        refuses any explicit one, so a stamped ``[512, 1]`` would surface there
        as a stride error the user never wrote.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def shard_incore_0(
                self,
                x: pl.Tensor[[64, 512], pl.INT8],
                w: pl.Tensor[[256, 512], pl.INT8, pl.NZ],
                ret0__out: pl.Out[pl.Tensor[[64, 256], pl.INT32]],
            ) -> pl.Tensor[[64, 256], pl.INT32]:
                x__tile: pl.Tile[[64, 512], pl.INT8, pl.Mem.Mat] = pl.load(
                    x, [0, 0], [64, 512], target_memory=pl.Mem.Mat
                )
                w__tile: pl.Tile[[256, 512], pl.INT8, pl.Mem.Mat] = pl.load(
                    w, [0, 0], [256, 512], target_memory=pl.Mem.Mat
                )
                acc = pl.matmul(x__tile, pl.tile.transpose_view(w__tile), out_dtype=pl.INT32)
                ret0__store: pl.Tensor[[64, 256], pl.INT32] = pl.store(acc, [0, 0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def shard(
                self,
                x: pl.Tensor[[64, 512], pl.INT8],
                w_all: pl.Tensor[[2, 256, 512], pl.INT8, pl.NZ],
            ) -> pl.Tensor[[64, 256], pl.INT32]:
                w = w_all[1]
                ret0__out: pl.Tensor[[64, 256], pl.INT32] = pl.create_tensor([64, 256], dtype=pl.INT32)
                result: pl.Tensor[[64, 256], pl.INT32] = self.shard_incore_0(x, w, ret0__out)
                return result

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Before)


class TestOutWindowExternalizer:
    """Pattern 5: static out-window externalization."""

    def test_pl_at_windowize_marks_only_outlined_incore_function(self):
        program = pl.parse_program(
            """
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[128, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, windowize=True):
            tile = pl.load(x, [0, 0], [128, 128])
            result = pl.store(tile, [0, 0], out)
        return result
"""
        )

        printed = ir.python_print(program)
        assert "windowize=True" in printed
        reparsed = pl.parse_program(printed)
        assert "windowize" in ir.python_print(reparsed)

        for outline_pass in (
            passes.outline_hierarchy_scopes(),
            passes.outline_incore_scopes(),
            passes.outline_cluster_scopes(),
        ):
            program = outline_pass(program)

        main = _get_function(program, "main")
        outlined = _get_function(program, "main_incore_0")
        assert "windowize" not in main.attrs
        assert outlined.attrs["windowize"] is True

    @pytest.mark.parametrize(
        "access",
        [
            pytest.param("first = out", id="alias"),
            pytest.param("pair = (out, x)\n            first = pair[0]", id="tuple"),
            pytest.param(
                "pair = (out, x)\n            pair2 = pair\n            first = pair2[0]", id="tuple_alias"
            ),
            pytest.param(
                "pair = (out, x)\n"
                "            for i, (p,) in pl.range(2, init_values=(pair,)):\n"
                "                p2 = pl.yield_(p)\n"
                "            first = p2[0]",
                id="tuple_loop_carry",
            ),
            pytest.param(
                "pair = (out, x)\n"
                "            for i, (p,) in pl.range(2, init_values=(pair,)):\n"
                "                inner = p[0]\n"
                "                inner_back = pl.load(inner, [128, 0], [128, 128])\n"
                "                p2 = pl.yield_(p)\n"
                "            first = p2[0]",
                id="tuple_loop_carry_read_in_body",
            ),
        ],
    )
    def test_value_reading_localized_store_result_takes_the_window(self, access):
        """Any non-Call value that reads a narrowed store result -- an alias, a
        tuple, a projection of it, an alias of that tuple, or a loop carry of it
        -- takes the window type, so its assignment keeps
        ``var.type == value.type``, and a load through it is rebased to the
        window like a load of the store result itself."""
        program = pl.parse_program(
            f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[512, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, windowize=True):
            tile = pl.load(x, [128, 0], [128, 128])
            out = pl.store(tile, [128, 0], out)
            {access}
            back = pl.load(first, [128, 0], [128, 128])
        return out
"""
        )

        after = _run_to_optimize_orch_tensors(program)

        printed = ir.python_print(_get_function(after, "main_incore_0__windowed"))
        assert "first__ssa_v0: pl.Tensor[[128, 128]" in printed, printed
        assert "pl.tile.load(first__ssa_v0, [0, 0]," in printed, printed
        # No load through a value derived from the store result keeps its
        # parent-relative offset (``inner`` is read inside the loop body).
        assert "pl.tile.load(inner__ssa_v0, [128, 0]," not in printed, printed
        assert "__FREE_VAR" not in printed, printed

    @pytest.mark.parametrize("localized", [False, True])
    @pytest.mark.parametrize(
        "control_flow",
        [
            pytest.param(
                "if n > 0:\n"
                "                alias = pair\n"
                "                joined = pl.yield_(alias)\n"
                "            else:\n"
                "                alias = pair\n"
                "                joined = pl.yield_(alias)",
                id="if_result",
            ),
            pytest.param(
                "for p, i in pl.while_(init_values=(pair, 0)):\n"
                "                pl.cond(i < n)\n"
                "                alias = p\n"
                "                joined, final_i = pl.yield_(alias, i + 1)",
                id="while_carry",
            ),
            pytest.param(
                "initial = (x, x)\n"
                "            for p, i in pl.while_(init_values=(initial, 0)):\n"
                "                pl.cond(i < n)\n"
                "                alias = pair\n"
                "                joined, final_i = pl.yield_(alias, i + 1)",
                id="while_yield",
            ),
        ],
    )
    def test_localized_control_flow_tuple_results_fall_back(self, control_flow, localized):
        """Unsupported localized results decline the whole candidate; ordinary
        tuples crossing the same control flow still permit windowization.
        """
        value = "stored" if localized else "x"
        program = pl.parse_program(
            f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[512, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        n: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, windowize=True):
            tile = pl.load(x, [128, 0], [128, 128])
            stored = pl.store(tile, [128, 0], out)
            pair = ({value}, x)
            {control_flow}
            first = joined[0]
            back = pl.load(first, [128, 0], [128, 128])
        return stored
"""
        )
        after = _run_to_optimize_orch_tensors(program)
        if localized:
            assert after.get_function("main_incore_0__windowed") is None
            ir.assert_structural_equal(after, _strip_windowize_attrs(_run_prereqs_only(program)))
        else:
            assert after.get_function("main_incore_0__windowed") is not None

    @pytest.mark.parametrize("trip_count", [2, 3])
    def test_tuple_carry_changing_windows_keeps_full_tensor_arguments(self, trip_count):
        """A carry that swaps outputs cannot use one static offset per element."""
        program = pl.parse_program(
            f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[512, 128], pl.FP32],
        out_a: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        out_b: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
    ) -> tuple[pl.Tensor[[512, 128], pl.FP32], pl.Tensor[[512, 128], pl.FP32]]:
        with pl.at(level=pl.Level.CORE_GROUP, windowize=True):
            tile = pl.load(x, [0, 0], [128, 128])
            out_a = pl.store(tile, [128, 0], out_a)
            out_b = pl.store(tile, [256, 0], out_b)
            pair = (out_a, out_b)
            for i, (p,) in pl.range({trip_count}, init_values=(pair,)):
                swapped = (p[1], p[0])
                p2 = pl.yield_(swapped)
            first = p2[0]
            back = pl.load(first, [256, 0], [128, 128])
        return out_a, out_b
"""
        )
        after = _run_to_optimize_orch_tensors(program)
        assert after.get_function("main_incore_0__windowed") is None
        ir.assert_structural_equal(after, _strip_windowize_attrs(_run_prereqs_only(program)))
        kernel = _get_function(after, "main_incore_0")
        printed = ir.python_print(kernel)
        assert "pl.tile.load(first__ssa_v0, [256, 0]," in printed, printed
        assert "__FREE_VAR" not in printed, printed

    @pytest.mark.parametrize("localized", [False, True])
    def test_repeated_nested_tuple_values_preserve_window_offsets(self, localized):
        """Repeated nested values preserve localized and empty provenance.

        Nested tuples are constructed through inferred values, without nested
        type annotations. Run only the target pass: print/parse roundtrips do
        not support those annotations, and unrelated passes need not process
        the DAG to exercise the localizer's resource use.
        """
        depth = 12
        source = "stored" if localized else "x"
        bindings = [f"pair0 = ({source}, {source})"]
        bindings += [f"pair{i} = (pair{i - 1}, pair{i - 1})" for i in range(1, depth)]
        bindings += [
            f"for i, (p,) in pl.range(2, init_values=(pair{depth - 1},)):",
            "    same = (p[0], p[1])",
            "    p2 = pl.yield_(same)",
            "value0 = p2[0]",
        ]
        bindings += [f"value{i} = value{i - 1}[0]" for i in range(1, depth)]
        access = "\n        ".join(bindings)
        program = pl.parse_program(
            f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[512, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        pl.func_attr({{"windowize": True}})
        tile = pl.tile.load(x, [128, 0], [128, 128], [128, 128])
        stored = pl.tile.store(tile, [128, 0], out)
        {access}
        back = pl.tile.load(value{depth - 1}, [128, 0], [128, 128], [128, 128])
        return stored

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[512, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        result = self.kernel(x, out)
        return result
"""
        )
        with passes.PassContext([], passes.VerificationLevel.NONE):
            after = passes.optimize_orch_tensors()(program)
        kernel = _get_function(after, "kernel__windowed")
        loads = _calls_named(ir.Program([kernel], "Windowed", kernel.span), "tile.load")
        assert len(loads) == 2
        offsets = loads[1].args[1]
        assert isinstance(offsets, ir.MakeTuple)
        values = []
        for value in offsets.elements:
            assert isinstance(value, ir.ConstInt)
            values.append(value.value)
        assert values == ([0, 0] if localized else [128, 0])

    def test_default_does_not_windowize_without_kernel_attr(self):
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
                row: pl.Scalar[pl.INDEX] = 64
                out_next: pl.Tensor[[256, 64], pl.FP32] = self.kernel_stripe(data, row, out)
                return out_next

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Before)
        assert After.get_function("kernel_stripe__windowed") is None

    def test_non_incore_type_attr_does_not_windowize(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore, attrs={"windowize": True})
            def eligible_kernel(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Opaque, attrs={"windowize": True})
            def opaque_kernel(
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
                out0: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
                out1: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> tuple[pl.Tensor[[256, 64], pl.FP32], pl.Tensor[[256, 64], pl.FP32]]:
                row: pl.Scalar[pl.INDEX] = 64
                out0_next: pl.Tensor[[256, 64], pl.FP32] = self.eligible_kernel(data, row, out0)
                out1_next: pl.Tensor[[256, 64], pl.FP32] = self.opaque_kernel(data, row, out1)
                return out0_next, out1_next

        After = passes.optimize_orch_tensors()(Before)

        assert After.get_function("eligible_kernel__windowed") is not None
        assert After.get_function("opaque_kernel__windowed") is None

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
        _assert_unchanged_by_pass(Before, After)
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
                    score, [row_offset, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec
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
                    score, [0, 0], [32, 64], [32, 64], target_memory=pl.Mem.Vec
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

        After = _run_to_optimize_orch_tensors(Before)
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

        @pl.program
        class Expected:
            # The pass adds a windowed clone whose params are the [32, 64] strided
            # views of the sliced parents, with the row offset folded to a 0 origin.
            @pl.function(type=pl.FunctionType.InCore)
            def mix_window__windowed(
                self,
                data: pl.Tensor[[32, 64], pl.FP32, pl.TensorView(stride=[128, 1], layout=pl.TensorLayout.ND)],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[
                    pl.Tensor[[32, 64], pl.FP32, pl.TensorView(stride=[128, 1], layout=pl.TensorLayout.ND)]
                ],
            ) -> pl.Tensor[[32, 64], pl.FP32, pl.TensorView(stride=[128, 1], layout=pl.TensorLayout.ND)]:
                tile: pl.Tile[[32, 64], pl.FP32] = pl.tile.load(data, [0, 0], [32, 64], [32, 64])
                result: pl.Tensor[
                    [32, 64], pl.FP32, pl.TensorView(stride=[128, 1], layout=pl.TensorLayout.ND)
                ] = pl.tile.store(tile, [0, 0], out)
                return result

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[64, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                data_win: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.slice(data, [32, 64], [32, 0])
                out_win: pl.Tensor[[32, 64], pl.FP32] = pl.tensor.slice(out, [32, 64], [32, 0])
                windowed: pl.Tensor[
                    [32, 64], pl.FP32, pl.TensorView(stride=[128, 1], layout=pl.TensorLayout.ND)
                ] = self.mix_window__windowed(data_win, 32, out_win)
                return pl.tensor.assemble(out, windowed, [32, 0])

            # The original callee is retained alongside the windowed clone.
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

        After = _run_to_optimize_orch_tensors(Before)
        ir.assert_structural_equal(After, _run_prereqs_only(Expected))

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

        After = _run_to_optimize_orch_tensors(Before)

        _assert_unchanged_by_pass(Before, After)
        assert After.get_function("consume__windowed") is None

    def test_nested_static_loops_without_param_reads_skip_trip_enumeration(self):
        """Statically bounded loops that never read the In param are not unrolled.

        ``ExtractInputAccessSet`` enumerates every trip of a statically bounded
        loop and recurses into the body once per trip, and nested loops
        multiply. ``kMaxEnumeratedInputUses`` does not bound that: it counts
        *recorded uses*, and a nest that never touches the param records
        nothing while still expanding fully. The three 128-trip loops below
        cost ~2M visits (~18s) without the early exit and ~0 with it, so the
        budget has a wide margin in the passing direction.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def consume(
                self,
                src: pl.Tensor[[64, 128], pl.FP32],
                acc: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                seed: pl.Tile[[64, 128], pl.FP32] = pl.load(src, [0, 0], [64, 128])
                out0: pl.Tensor[[64, 128], pl.FP32] = pl.store(seed, [0, 0], acc)
                total0: pl.Scalar[pl.INDEX] = 0
                for _i1, (t1,) in pl.range(128, init_values=(total0,)):
                    for _i2, (t2,) in pl.range(128, init_values=(t1,)):
                        for _i3, (t3,) in pl.range(128, init_values=(t2,)):
                            bumped: pl.Scalar[pl.INDEX] = t3 + 1
                            r3 = pl.yield_(bumped)
                        r2 = pl.yield_(r3)
                    r1 = pl.yield_(r2)
                return out0

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                src: pl.Tensor[[64, 128], pl.FP32],
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                acc: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
                return self.consume(src, acc)

        start = time.perf_counter()
        After = _run_to_optimize_orch_tensors(Before)
        elapsed = time.perf_counter() - start

        assert elapsed < 10.0, f"loop trips were enumerated without a param read: pass took {elapsed:.1f}s"
        _assert_unchanged_by_pass(Before, After)
        assert After.get_function("consume__windowed") is None

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

        After = _run_to_optimize_orch_tensors(Before)
        _assert_unchanged_by_pass(Before, After)
        assert After.get_function("fence__windowed") is None

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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def score_init(
                self,
                score: pl.Out[pl.Tensor[[4, 16], pl.FP32]],
                t0: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[4, 16], pl.FP32]:
                init_tile: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.full(
                    [2, 16], dtype=pl.FP32, value=-1.0
                )
                score_next: pl.Tensor[[4, 16], pl.FP32] = pl.tile.store(init_tile, [t0, 0], score)
                return score_next

            # Both score writers windowize: each writes a statically-shaped
            # sub-region, so the row/col offset moves into the window.
            @pl.function(type=pl.FunctionType.InCore)
            def score_init__windowed(
                self,
                score: pl.Out[
                    pl.Tensor[[2, 16], pl.FP32, pl.TensorView(stride=[16, 1], layout=pl.TensorLayout.ND)]
                ],
                t0: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[2, 16], pl.FP32, pl.TensorView(stride=[16, 1], layout=pl.TensorLayout.ND)]:
                init_tile: pl.Tile[[2, 16], pl.FP32, pl.Mem.Vec] = pl.tile.full(
                    [2, 16], dtype=pl.FP32, value=-1.0
                )
                score_next: pl.Tensor[
                    [2, 16], pl.FP32, pl.TensorView(stride=[16, 1], layout=pl.TensorLayout.ND)
                ] = pl.tile.store(init_tile, [0, 0], score)
                return score_next

            @pl.function(type=pl.FunctionType.InCore)
            def score_writer(
                self,
                score: pl.Out[pl.Tensor[[4, 16], pl.FP32]],
                t0: pl.Scalar[pl.INDEX],
                cache0: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[4, 16], pl.FP32]:
                score_tile: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.full(
                    [2, 4], dtype=pl.FP32, value=1.0
                )
                score_next: pl.Tensor[[4, 16], pl.FP32] = pl.tile.store(score_tile, [t0, cache0], score)
                return score_next

            @pl.function(type=pl.FunctionType.InCore)
            def score_writer__windowed(
                self,
                score: pl.Out[
                    pl.Tensor[[2, 4], pl.FP32, pl.TensorView(stride=[16, 1], layout=pl.TensorLayout.ND)]
                ],
                t0: pl.Scalar[pl.INDEX],
                cache0: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[2, 4], pl.FP32, pl.TensorView(stride=[16, 1], layout=pl.TensorLayout.ND)]:
                score_tile: pl.Tile[[2, 4], pl.FP32, pl.Mem.Vec] = pl.tile.full(
                    [2, 4], dtype=pl.FP32, value=1.0
                )
                score_next: pl.Tensor[
                    [2, 4], pl.FP32, pl.TensorView(stride=[16, 1], layout=pl.TensorLayout.ND)
                ] = pl.tile.store(score_tile, [0, 0], score)
                return score_next

            # topk_like READS score at a dynamic row and is NOT cloned: no
            # topk_like__windowed exists and `score_rv` is passed whole. That
            # asymmetry -- writers windowed, reader full -- is the fact under test.
            @pl.function(type=pl.FunctionType.InCore)
            def topk_like(
                self,
                topk: pl.Out[pl.Tensor[[4, 16], pl.INT32]],
                t0: pl.Scalar[pl.INDEX],
                score: pl.Tensor[[4, 16], pl.FP32],
            ) -> pl.Tensor[[4, 16], pl.INT32]:
                invalid: pl.Tile[[1, 16], pl.INT32, pl.Mem.Vec] = pl.tile.full(
                    [1, 16], dtype=pl.INT32, value=-1
                )
                topk_init: pl.Tensor[[4, 16], pl.INT32] = pl.tile.store(invalid, [t0, 0], topk)
                score_row: pl.Tile[[1, 16], pl.FP32] = pl.tile.load(score, [t0, 0], [1, 16], [1, 16])
                idx_tile: pl.Tile[[1, 16], pl.INT32, pl.Mem.Vec] = pl.tile.cast(
                    score_row, target_type=pl.INT32
                )
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
                    score_win: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.slice(score_iter, [2, 16], [t0, 0])
                    init_windowed: pl.Tensor[
                        [2, 16], pl.FP32, pl.TensorView(stride=[16, 1], layout=pl.TensorLayout.ND)
                    ] = self.score_init__windowed(score_win, t0)
                    score_init_next: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.assemble(
                        score_iter, init_windowed, [t0, 0]
                    )
                    for cb, (score_iter2,) in pl.parallel(4, init_values=(score_init_next,)):
                        cache0: pl.Scalar[pl.INDEX] = cb * 4
                        score2_win: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.slice(
                            score_iter2, [2, 4], [t0, cache0]
                        )
                        writer_windowed: pl.Tensor[
                            [2, 4], pl.FP32, pl.TensorView(stride=[16, 1], layout=pl.TensorLayout.ND)
                        ] = self.score_writer__windowed(score2_win, t0, cache0)
                        score_next: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.assemble(
                            score_iter2, writer_windowed, [t0, cache0]
                        )
                        score_rv = pl.yield_(score_next)
                    # Full `score_rv` handed to the reader -- no slice, no clone.
                    topk_next: pl.Tensor[[4, 16], pl.INT32] = self.topk_like(topk_iter, t0, score_rv)
                    score_out, topk_out = pl.yield_(score_rv, topk_next)
                return score_out, topk_out

        After = _run_to_optimize_orch_tensors(Before)
        ir.assert_structural_equal(After, _run_prereqs_only(Expected))

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

        @pl.program
        class Expected:
            # The static-slot writer IS windowized: slot 7 is a compile-time constant.
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
            def cache_write__windowed(
                self,
                cache: pl.Out[
                    pl.Tensor[[1, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)]
                ],
                data: pl.Tensor[[1, 64], pl.FP32],
                slot: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[1, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)]:
                src: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(data, [0, 0], [1, 64], [1, 64])
                result: pl.Tensor[
                    [1, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)
                ] = pl.tile.store(src, [0, 0], cache)
                return result

            # The reader is NOT windowized and keeps the FULL [1024, 64] parent: its
            # row offset comes from a runtime block_table read, so no static window
            # can cover it. This is the fact under test.
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
                # The trip-count-1 loop of ``Before`` is gone -- UnrollLoops (a
                # prerequisite pass) removes it before OptimizeOrchTensors runs.
                cache_win: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(cache, [1, 64], [7, 0])
                windowed: pl.Tensor[
                    [1, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)
                ] = self.cache_write__windowed(cache_win, data, 7)
                cache_next: pl.Tensor[[1024, 64], pl.FP32] = pl.tensor.assemble(cache, windowed, [7, 0])
                # Full parent passed to the reader -- no slice, no windowed clone.
                result: pl.Tensor[[4, 64], pl.FP32] = self.cache_read(out, cache_next, block_table)
                return cache_next, result

        After = _run_to_optimize_orch_tensors(Before)
        ir.assert_structural_equal(After, _run_prereqs_only(Expected))

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

        @pl.program
        class Expected:
            # Both static-slot writers windowize independently...
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
            def cache_write__windowed(
                self,
                cache: pl.Out[
                    pl.Tensor[[1, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)]
                ],
                data: pl.Tensor[[1, 64], pl.FP32],
                slot: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[1, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)]:
                src: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(data, [0, 0], [1, 64], [1, 64])
                result: pl.Tensor[
                    [1, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)
                ] = pl.tile.store(src, [0, 0], cache)
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
            def unrelated_write__windowed(
                self,
                other: pl.Out[
                    pl.Tensor[[1, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)]
                ],
                data: pl.Tensor[[1, 64], pl.FP32],
            ) -> pl.Tensor[[1, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)]:
                src: pl.Tile[[1, 64], pl.FP32] = pl.tile.load(data, [0, 0], [1, 64], [1, 64])
                result: pl.Tensor[
                    [1, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)
                ] = pl.tile.store(src, [0, 0], other)
                return result

            # ...but the dynamic reader keeps the FULL [1024, 64] parent. The fallback
            # is parent-LOCAL: windowizing `cache` does not spill over onto `other`,
            # and the unrelated writer is windowized regardless. That is the fact
            # under test, and only whole-program equality pins both halves at once.
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
                cache_win: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(cache, [1, 64], [7, 0])
                cache_windowed: pl.Tensor[
                    [1, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)
                ] = self.cache_write__windowed(cache_win, data, 7)
                cache_next: pl.Tensor[[1024, 64], pl.FP32] = pl.tensor.assemble(cache, cache_windowed, [7, 0])
                other_win: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(other, [1, 64], [3, 0])
                other_windowed: pl.Tensor[
                    [1, 64], pl.FP32, pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND)
                ] = self.unrelated_write__windowed(other_win, data)
                other_next: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.assemble(other, other_windowed, [3, 0])
                # Full parent to the dynamic reader -- no slice, no windowed clone.
                result: pl.Tensor[[4, 64], pl.FP32] = self.cache_read(out, cache_next, block_table)
                return cache_next, other_next, result

        After = _run_to_optimize_orch_tensors(Before)
        ir.assert_structural_equal(After, _run_prereqs_only(Expected))

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

        # The guarded dynamic read is unwindowable, so the pass must decline to
        # rewrite anything -- pinned over the whole program, not just `main`.
        _assert_unchanged_by_pass(Before, After)

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

        # The loop-local non-dynamic offset is rejected, so the pass must decline
        # to rewrite anything -- pinned over the whole program, not just `main`.
        _assert_unchanged_by_pass(Before, After)

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

        After = _run_to_optimize_orch_tensors(Before)

        _assert_unchanged_by_pass(Before, After)
        assert After.get_function("init_row__windowed") is None
        assert After.get_function("write_prefix__windowed") is None

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

        After = _run_to_optimize_orch_tensors(Before)

        _assert_unchanged_by_pass(Before, After)
        assert After.get_function("init_row__windowed") is None
        assert After.get_function("full_overwrite__windowed") is None

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

        After = _run_to_optimize_orch_tensors(Before)

        _assert_unchanged_by_pass(Before, After)
        assert After.get_function("init_row__windowed") is None

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

        After = _run_to_optimize_orch_tensors(Before)

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

        @pl.program
        class Expected:
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
                    q_tile: pl.Tile[[16, 640], pl.FP32] = pl.tile.load(
                        q_proj, [row, q0], [16, 640], [16, 640]
                    )
                    k_tile: pl.Tile[[16, 128], pl.FP32] = pl.tile.load(
                        k_proj, [row, k0], [16, 128], [16, 128]
                    )
                    q_next: pl.Tensor[[16, 5120], pl.FP32] = pl.tile.store(q_tile, [row, q0], q_iter)
                    k_next: pl.Tensor[[16, 1024], pl.FP32] = pl.tile.store(k_tile, [row, k0], k_iter)
                    q_norm, k_norm = pl.yield_(q_next, k_next)
                return q_norm, k_norm

            # The windowed clone reads at row 0: the window absorbed the row offset.
            @pl.function(type=pl.FunctionType.InCore)
            def qk_norm_like__windowed(
                self,
                q_out: pl.Out[
                    pl.Tensor[[16, 5120], pl.FP32, pl.TensorView(stride=[5120, 1], layout=pl.TensorLayout.ND)]
                ],
                k_out: pl.Out[
                    pl.Tensor[[16, 1024], pl.FP32, pl.TensorView(stride=[1024, 1], layout=pl.TensorLayout.ND)]
                ],
                q_proj: pl.Tensor[
                    [16, 5120], pl.FP32, pl.TensorView(stride=[5120, 1], layout=pl.TensorLayout.ND)
                ],
                k_proj: pl.Tensor[
                    [16, 1024], pl.FP32, pl.TensorView(stride=[1024, 1], layout=pl.TensorLayout.ND)
                ],
                row: pl.Scalar[pl.INDEX],
            ) -> tuple[
                pl.Tensor[[16, 5120], pl.FP32, pl.TensorView(stride=[5120, 1], layout=pl.TensorLayout.ND)],
                pl.Tensor[[16, 1024], pl.FP32, pl.TensorView(stride=[1024, 1], layout=pl.TensorLayout.ND)],
            ]:
                for h, (q_iter, k_iter) in pl.range(8, init_values=(q_out, k_out)):
                    q0: pl.Scalar[pl.INDEX] = h * 640
                    k0: pl.Scalar[pl.INDEX] = h * 128
                    q_tile: pl.Tile[[16, 640], pl.FP32] = pl.tile.load(q_proj, [0, q0], [16, 640], [16, 640])
                    k_tile: pl.Tile[[16, 128], pl.FP32] = pl.tile.load(k_proj, [0, k0], [16, 128], [16, 128])
                    q_next: pl.Tensor[
                        [16, 5120], pl.FP32, pl.TensorView(stride=[5120, 1], layout=pl.TensorLayout.ND)
                    ] = pl.tile.store(q_tile, [0, q0], q_iter)
                    k_next: pl.Tensor[
                        [16, 1024], pl.FP32, pl.TensorView(stride=[1024, 1], layout=pl.TensorLayout.ND)
                    ] = pl.tile.store(k_tile, [0, k0], k_iter)
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
                # The pass-through trip-count-1 loop of ``Before`` is gone (prereqs),
                # so `q_rv`/`k_rv` collapse to the visible loop-init parents. The
                # input windows are therefore sliced off `q_proj`/`k_proj` -- the
                # parent that is actually in scope -- which is the fact under test.
                q_win: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.slice(q_proj, [16, 5120], [0, 0])
                k_win: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.slice(k_proj, [16, 1024], [0, 0])
                q_out_win: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.slice(q_out, [16, 5120], [0, 0])
                k_out_win: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.slice(k_out, [16, 1024], [0, 0])
                # Multi-return shape: one tuple temp, then per-element
                # projection/assemble pairs interleaved -- mirrors the already
                # converted golden in test_multi_out_final_store_rewrites_both_outputs.
                windowed: pl.Tuple[
                    pl.Tensor[
                        [16, 5120], pl.FP32, pl.TensorView(stride=[5120, 1], layout=pl.TensorLayout.ND)
                    ],
                    pl.Tensor[
                        [16, 1024], pl.FP32, pl.TensorView(stride=[1024, 1], layout=pl.TensorLayout.ND)
                    ],
                ] = self.qk_norm_like__windowed(q_out_win, k_out_win, q_win, k_win, 0)
                q_w: pl.Tensor[
                    [16, 5120], pl.FP32, pl.TensorView(stride=[5120, 1], layout=pl.TensorLayout.ND)
                ] = windowed[0]
                q_res: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(q_out, q_w, [0, 0])
                k_w: pl.Tensor[
                    [16, 1024], pl.FP32, pl.TensorView(stride=[1024, 1], layout=pl.TensorLayout.ND)
                ] = windowed[1]
                k_res: pl.Tensor[[16, 1024], pl.FP32] = pl.tensor.assemble(k_out, k_w, [0, 0])
                res: pl.Tuple[pl.Tensor[[16, 5120], pl.FP32], pl.Tensor[[16, 1024], pl.FP32]] = [
                    q_res,
                    k_res,
                ]
                return res

        After = _run_to_optimize_orch_tensors(Before)
        ir.assert_structural_equal(After, _run_prereqs_only(Expected))

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

        After = _run_to_optimize_orch_tensors(Before)

        _assert_unchanged_by_pass(Before, After)
        assert After.get_function("diagonal_like__windowed") is None

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

        After = _run_to_optimize_orch_tensors(Before)

        _assert_unchanged_by_pass(Before, After)
        assert After.get_function("overlap_hole_like__windowed") is None

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

        After = _run_to_optimize_orch_tensors(Before)

        _assert_unchanged_by_pass(Before, After)
        assert After.get_function("write_overlap__windowed") is None

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
                header_tile: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(header, [row, 0], [16, 64], [16, 64])
                for h, (out_iter,) in pl.range(4, init_values=(out,)):
                    col: pl.Scalar[pl.INDEX] = h * 64
                    data_tile: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(data, [row, col], [16, 64], [16, 64])
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
                header_tile: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(header, [0, 0], [16, 64], [16, 64])
                for h, (out_iter,) in pl.range(4, init_values=(out,)):
                    col: pl.Scalar[pl.INDEX] = h * 64
                    data_tile: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(data, [0, col], [16, 64], [16, 64])
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
                # Local names avoid the pass's own ``X__window`` spelling. Parsing
                # accepts ``__``, but normalizing this golden through
                # ``_run_prereqs_only`` does not: auto-naming raises "IR auto-name
                # base cannot contain reserved delimiter '__'". Structural equality
                # compares IR shape, not names, so shorter spellings match anyway.
                # Slice order follows the pass's emission order (data, header, out).
                data_win: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(data, [16, 256], [0, 0])
                header_win: pl.Tensor[[16, 64], pl.FP32] = pl.tensor.slice(header, [16, 64], [0, 0])
                out_win: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.slice(out, [16, 256], [0, 0])
                windowed: pl.Tensor[
                    [16, 256], pl.FP32, pl.TensorView(stride=[256, 1], layout=pl.TensorLayout.ND)
                ] = self.aggregate_with_header__windowed(out_win, data_win, header_win, 0)
                result: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.assemble(out, windowed, [0, 0])
                return result

        After = _run_to_optimize_orch_tensors(Before)
        ir.assert_structural_equal(After, _run_prereqs_only(Expected))

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
                tile: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(data, [row_offset, 0], [64, 64], [64, 64])
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
                tile: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(data, [0, 0], [64, 64], [64, 64])
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

        After = _run_to_optimize_orch_tensors(Before)
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
                tile: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(data, [row_offset, 0], [64, 64], [64, 64])
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
                tile: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(data, [0, 0], [64, 64], [64, 64])
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

        After = _run_to_optimize_orch_tensors(Before)
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

        After = _run_to_optimize_orch_tensors(Before)

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

        After = _run_to_optimize_orch_tensors(Before)

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

        After = _run_to_optimize_orch_tensors(Before)

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

        After = _run_to_optimize_orch_tensors(Before)

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

        After = _run_to_optimize_orch_tensors(Before)

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
                    data, [row_offset, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                    data, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec
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
                tile: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(data, [row_offset, 0], [64, 64], [64, 64])
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
                tile: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(data, [0, 0], [64, 64], [64, 64])
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

        After = _run_to_optimize_orch_tensors(Before)
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

        After = _run_to_optimize_orch_tensors(Before)

        _assert_unchanged_by_pass(Before, After)
        assert After.get_function("kv_stripe__windowed") is None

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

        After = _run_to_optimize_orch_tensors(Before)

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

        After = _run_to_optimize_orch_tensors(Before)

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

        After = _run_to_optimize_orch_tensors(Before)

        assert After.get_function("kernel_rows__windowed") is not None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "pl.tensor.slice(out" in printed_main
        assert "consume_full(out_next__ssa_v0)" in printed_main
        assert "consume_full(out)" not in printed_main

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
                            attrs={"__compiler_tensor_to_tile_mat_bridge": True},
                        )
                        tile_wk: pl.Tile[[128, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                            wk,
                            [0, kv0],
                            [128, 64],
                            [128, 64],
                            target_memory=pl.Mem.Mat,
                            attrs={"__compiler_tensor_to_tile_mat_bridge": True},
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
                                attrs={"__compiler_tensor_to_tile_mat_bridge": True},
                            )
                            tile_wk_i: pl.Tile[[128, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                                wk,
                                [k0, kv0],
                                [128, 64],
                                [128, 64],
                                target_memory=pl.Mem.Mat,
                                attrs={"__compiler_tensor_to_tile_mat_bridge": True},
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
                            attrs={"__compiler_tensor_to_tile_mat_bridge": True},
                        )
                        tile_wv: pl.Tile[[128, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                            wv,
                            [0, kv0],
                            [128, 64],
                            [128, 64],
                            target_memory=pl.Mem.Mat,
                            attrs={"__compiler_tensor_to_tile_mat_bridge": True},
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
                                attrs={"__compiler_tensor_to_tile_mat_bridge": True},
                            )
                            tile_wv_i: pl.Tile[[128, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                                wv,
                                [k0_2, kv0],
                                [128, 64],
                                [128, 64],
                                target_memory=pl.Mem.Mat,
                                attrs={"__compiler_tensor_to_tile_mat_bridge": True},
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
                k_acc: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(k_out, [0, 0], [16, 64], [16, 64])
                for ob, (k_iter, k_acc_iter) in pl.range(ob_chunk, ob_chunk + 4, init_values=(k_out, k_acc)):
                    kv0: pl.Scalar[pl.INDEX] = ob * 64
                    tile_a: pl.Tile[[16, 128], pl.BF16] = pl.tile.load(
                        normed_tile, [0, 0], [16, 128], [16, 128]
                    )
                    tile_wk: pl.Tile[[128, 64], pl.BF16] = pl.tile.load(wk, [0, kv0], [128, 64], [128, 64])
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
                tile: pl.Tile[[64, 64], pl.FP32] = pl.tile.load(data, [0, 0], [64, 64], [64, 64])
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


class TestOutWindowMultiOutSubset:
    """Pattern 5 rewrites each proven Out window independently."""


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

        After = _run_to_optimize_orch_tensors(Before)
        # A statically provable 64x64 window write at [64, 0] must be
        # externalized when the kernel is explicitly windowized.
        windowed = After.get_function("kernel_stripe__windowed")
        assert windowed is not None
        printed_main = ir.python_print(_get_function(After, "main"))
        assert "pl.tensor.slice(out" in printed_main
        assert "kernel_stripe__windowed" in printed_main

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

        After = _run_to_optimize_orch_tensors(Before)

        _assert_unchanged_by_pass(Before, After)
        assert After.get_function("overflow_store__windowed") is None

    def test_dense_region_volume_overflow_falls_back_to_baseline(self):
        huge_extent = 9223372036854775807

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def overflow_volume(
                self,
                out: pl.Out[pl.Tensor[[8, huge_extent], pl.FP32]],
                data: pl.Tensor[[2, huge_extent], pl.FP32],
            ) -> pl.Tensor[[8, huge_extent], pl.FP32]:
                for i, (out_iter,) in pl.range(0, 2, init_values=(out,)):
                    tile = pl.tile.load(data, [0, 0], [2, huge_extent], [2, huge_extent])
                    out_next = pl.tile.store(tile, [i * 2, 0], out_iter)
                    out_rv = pl.yield_(out_next)
                return out_rv

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                out: pl.Out[pl.Tensor[[8, huge_extent], pl.FP32]],
                data: pl.Tensor[[2, huge_extent], pl.FP32],
            ) -> pl.Tensor[[8, huge_extent], pl.FP32]:
                return self.overflow_volume(out, data)

        After = passes.optimize_orch_tensors()(_with_incore_windowize(Before))

        # The pass runs directly here (no pipeline prefix), so `Before` is already
        # at the stage the pass observes and is itself the golden: an overflowing
        # dense-region volume must fall back to leaving the program untouched.
        ir.assert_structural_equal(_strip_windowize_attrs(After), Before)
        assert After.get_function("overflow_volume__windowed") is None

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

        After = _run_to_optimize_orch_tensors(Before)

        _assert_unchanged_by_pass(Before, After)
        assert After.get_function("update__windowed") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
