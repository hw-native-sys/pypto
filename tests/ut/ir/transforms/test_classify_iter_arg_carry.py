# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the ClassifyIterArgCarry pass.

The pass stamps each Orchestration ``ForStmt`` with a per-iter_arg carry plan:

* ``iter_arg_rebind_<i>`` (bool, one per slot) — ``False`` means the yield value
  aliases the iter_arg (same backing buffer), so codegen routes iter_arg and
  return_var to the init value's emit name. ``True`` means a materialised mutable
  carry variable is needed.
* ``iter_arg_array_size_<i>`` (int, positive extents only) — the ``PTO2TaskId[N]``
  fence-array extent for a ``Scalar[TASK_ID]`` carry inside a ``pl.manual_scope``.

Tests assert on the stamped attrs rather than on a full Expected program: the
pass runs last in the pipeline and rewrites nothing but ``ForStmt.attrs``.
"""

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

N, M = 64, 64


@pytest.fixture(autouse=True)
def _setup_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _classify(program):
    """Run the pass on its declared prerequisites (as the pipeline does)."""
    program = passes.derive_call_directions()(program)
    program = passes.materialize_runtime_scopes()(program)
    return passes.classify_iter_arg_carry()(program)


def _orch_func(program) -> ir.Function:
    for func in program.functions.values():
        if func.func_type == ir.FunctionType.Orchestration:
            return func
    raise AssertionError("no Orchestration function found in program")


def _for_stmts(stmt) -> list[ir.ForStmt]:
    """Collect every ForStmt in a body, outermost first."""
    found: list[ir.ForStmt] = []

    def walk(node) -> None:
        if node is None:
            return
        if isinstance(node, ir.ForStmt):
            found.append(node)
            walk(node.body)
        elif isinstance(node, ir.SeqStmts):
            for child in node.stmts:
                walk(child)
        elif isinstance(node, (ir.RuntimeScopeStmt, ir.WhileStmt)):
            walk(node.body)
        elif isinstance(node, ir.IfStmt):
            walk(node.then_body)
            walk(node.else_body)

    walk(stmt)
    return found


def _carry_attrs(for_stmt: ir.ForStmt) -> dict[str, object]:
    return {
        k: v for k, v in for_stmt.attrs.items() if k.startswith(("iter_arg_rebind_", "iter_arg_array_size_"))
    }


def test_trivial_carry_is_not_a_rebind():
    """The kernel writes ``acc`` in place and returns it, so the yield value is
    an alias of the iter_arg — no materialised carry is needed."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.AIV)
        def accumulate(
            self,
            x: pl.Tensor[[N, M], pl.FP32],
            acc: pl.InOut[pl.Tensor[[N, M], pl.FP32]],
        ) -> pl.Tensor[[N, M], pl.FP32]:
            t: pl.Tile[[N, M], pl.FP32] = pl.load(x, [0, 0], [N, M])
            return pl.store(t, [0, 0], acc)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[N, M], pl.FP32],
            out: pl.Out[pl.Tensor[[N, M], pl.FP32]],
        ) -> pl.Tensor[[N, M], pl.FP32]:
            for _i, (acc,) in pl.range(0, 4, init_values=(out,)):
                acc2: pl.Tensor[[N, M], pl.FP32] = self.accumulate(x, acc)
                (out,) = pl.yield_(acc2)
            return out

    loops = _for_stmts(_orch_func(_classify(Prog)).body)
    assert len(loops) == 1
    assert _carry_attrs(loops[0]) == {"iter_arg_rebind_0": False}


def test_fresh_tensor_yield_is_a_rebind():
    """The body yields a freshly created tensor, so the carry must be a mutable
    variable that the yield assigns back to (issue #1286)."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.AIV)
        def produce(
            self,
            x: pl.Tensor[[N, M], pl.FP32],
            fresh: pl.Out[pl.Tensor[[N, M], pl.FP32]],
        ) -> pl.Tensor[[N, M], pl.FP32]:
            t: pl.Tile[[N, M], pl.FP32] = pl.load(x, [0, 0], [N, M])
            return pl.store(t, [0, 0], fresh)

        @pl.function(type=pl.FunctionType.AIV)
        def consume(
            self,
            acc: pl.Tensor[[N, M], pl.FP32],
            out: pl.Out[pl.Tensor[[N, M], pl.FP32]],
        ) -> pl.Tensor[[N, M], pl.FP32]:
            t: pl.Tile[[N, M], pl.FP32] = pl.load(acc, [0, 0], [N, M])
            return pl.store(t, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[N, M], pl.FP32],
            seed: pl.Tensor[[N, M], pl.FP32],
            out: pl.Out[pl.Tensor[[N, M], pl.FP32]],
        ) -> pl.Tensor[[N, M], pl.FP32]:
            for _i, (acc,) in pl.range(0, 4, init_values=(seed,)):
                fresh: pl.Tensor[[N, M], pl.FP32] = pl.create_tensor([N, M], dtype=pl.FP32)
                fresh2: pl.Tensor[[N, M], pl.FP32] = self.produce(x, fresh)
                (acc_rv,) = pl.yield_(fresh2)
            out = self.consume(acc_rv, out)
            return out

    loops = _for_stmts(_orch_func(_classify(Prog)).body)
    assert len(loops) == 1
    assert _carry_attrs(loops[0]) == {"iter_arg_rebind_0": True}


def test_manual_scope_parallel_task_id_carry_is_sized():
    """A ``Scalar[TASK_ID]`` carry on a const-trip ``pl.parallel`` inside a
    manual scope lowers to a ``PTO2TaskId[N]`` fence array of that trip count."""
    rows, cols, tile = 128, 128, 32

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.AIV)
        def kern(
            self,
            x: pl.Tensor[[rows, cols], pl.FP32],
            out: pl.InOut[pl.Tensor[[rows, cols], pl.FP32]],
            row: pl.Scalar[pl.INDEX],
            col: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[rows, cols], pl.FP32]:
            t: pl.Tile[[tile, tile], pl.FP32] = pl.load(x, [row, col], [tile, tile])
            r: pl.Tile[[tile, tile], pl.FP32] = pl.add(t, t)
            return pl.store(r, [row, col], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[rows, cols], pl.FP32],
            out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
        ) -> pl.Tensor[[rows, cols], pl.FP32]:
            with pl.manual_scope():
                prev_tid = None
                for i in pl.range(4):
                    row: pl.Scalar[pl.INDEX] = i * tile
                    for j in pl.parallel(4):
                        col: pl.Scalar[pl.INDEX] = j * tile
                        out, prev_tid = pl.submit(self.kern, x, out, row, col, deps=[prev_tid])
            return out

    transformed = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)
    loops = _for_stmts(_orch_func(transformed).body)
    assert len(loops) == 2, [loop.kind for loop in loops]
    outer, inner = loops

    # The inner pl.parallel(4) owns the 4-slot fence array; the outer Sequential
    # loop threads the same array through, so it is sized identically.
    inner_attrs = _carry_attrs(inner)
    outer_attrs = _carry_attrs(outer)
    assert inner.kind == ir.ForKind.Parallel
    assert any(v == 4 for k, v in inner_attrs.items() if k.startswith("iter_arg_array_size_"))
    assert any(v == 4 for k, v in outer_attrs.items() if k.startswith("iter_arg_array_size_"))
    # A TaskId carry is never trivial: the runtime hands back a fresh id per iter.
    for attrs in (inner_attrs, outer_attrs):
        sized = [k.rsplit("_", 1)[-1] for k in attrs if k.startswith("iter_arg_array_size_")]
        for idx in sized:
            assert attrs[f"iter_arg_rebind_{idx}"] is True


def test_manual_scope_parallel_dynamic_trip_count_rejected():
    """A dynamic ``pl.parallel`` trip count cannot size the fence array, so the
    pass rejects it with a user-facing error instead of silently mis-lowering."""
    rows, cols, tile = 128, 128, 32

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.AIV)
        def kern(
            self,
            x: pl.Tensor[[rows, cols], pl.FP32],
            out: pl.InOut[pl.Tensor[[rows, cols], pl.FP32]],
            row: pl.Scalar[pl.INDEX],
            col: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[rows, cols], pl.FP32]:
            t: pl.Tile[[tile, tile], pl.FP32] = pl.load(x, [row, col], [tile, tile])
            r: pl.Tile[[tile, tile], pl.FP32] = pl.add(t, t)
            return pl.store(r, [row, col], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[rows, cols], pl.FP32],
            out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            n_branches: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[rows, cols], pl.FP32]:
            with pl.manual_scope():
                prev_tid = None
                for i in pl.range(4):
                    row: pl.Scalar[pl.INDEX] = i * tile
                    for j in pl.parallel(n_branches):
                        col: pl.Scalar[pl.INDEX] = j * tile
                        out, prev_tid = pl.submit(self.kern, x, out, row, col, deps=[prev_tid])
            return out

    with pytest.raises(Exception, match="statically-known trip count"):
        PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Prog)


def test_no_manual_scope_leaves_task_id_carry_unsized():
    """Outside a manual scope there is no fence array: a rebind carry is stamped
    but ``iter_arg_array_size_<i>`` stays absent (extent 0)."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            x: pl.Tensor[[N, M], pl.FP32],
            out: pl.Out[pl.Tensor[[N, M], pl.FP32]],
        ) -> pl.Tensor[[N, M], pl.FP32]:
            t: pl.Tile[[N, M], pl.FP32] = pl.load(x, [0, 0], [N, M])
            return pl.store(t, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[N, M], pl.FP32],
            seed: pl.Tensor[[N, M], pl.FP32],
            out: pl.Out[pl.Tensor[[N, M], pl.FP32]],
        ) -> pl.Tensor[[N, M], pl.FP32]:
            for _i, (acc,) in pl.range(0, 4, init_values=(seed,)):
                fresh: pl.Tensor[[N, M], pl.FP32] = pl.create_tensor([N, M], dtype=pl.FP32)
                fresh2: pl.Tensor[[N, M], pl.FP32] = self.kernel(x, fresh)
                (_acc_rv,) = pl.yield_(fresh2)
            out = self.kernel(x, out)
            return out

    loops = _for_stmts(_orch_func(_classify(Prog)).body)
    assert len(loops) == 1
    attrs = _carry_attrs(loops[0])
    assert attrs == {"iter_arg_rebind_0": True}
    assert "iter_arg_array_size_0" not in attrs


def test_pass_is_idempotent():
    """Re-running the pass replaces the stamped attrs rather than duplicating
    them, so a program that already went through the pipeline is unchanged."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.AIV)
        def accumulate(
            self,
            x: pl.Tensor[[N, M], pl.FP32],
            acc: pl.InOut[pl.Tensor[[N, M], pl.FP32]],
        ) -> pl.Tensor[[N, M], pl.FP32]:
            t: pl.Tile[[N, M], pl.FP32] = pl.load(x, [0, 0], [N, M])
            return pl.store(t, [0, 0], acc)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[N, M], pl.FP32],
            out: pl.Out[pl.Tensor[[N, M], pl.FP32]],
        ) -> pl.Tensor[[N, M], pl.FP32]:
            for _i, (acc,) in pl.range(0, 4, init_values=(out,)):
                acc2: pl.Tensor[[N, M], pl.FP32] = self.accumulate(x, acc)
                (out,) = pl.yield_(acc2)
            return out

    once = _classify(Prog)
    twice = passes.classify_iter_arg_carry()(once)
    ir.assert_structural_equal(twice, once)


def test_pass_metadata():
    p = passes.classify_iter_arg_carry()
    assert p.get_name() == "ClassifyIterArgCarry"
    assert p.get_produced_properties().contains(passes.IRProperty.IterArgCarryClassified)
    assert p.get_required_properties().contains(passes.IRProperty.RuntimeScopesMaterialized)
    assert p.get_required_properties().contains(passes.IRProperty.CallDirectionsResolved)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
