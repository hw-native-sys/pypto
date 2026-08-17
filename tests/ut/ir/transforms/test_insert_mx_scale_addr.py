# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for InsertMxScaleAddr pass."""

import pypto.language as pl
import pytest
from pypto import backend, ir, passes


@pytest.fixture(autouse=True)
def _reset_backend():
    backend.reset_for_testing()
    yield
    backend.reset_for_testing()


class TestInsertMxScaleAddr:
    """InsertMxScaleAddr inserts internal scale-address bindings after InferTileMemorySpace."""

    @staticmethod
    def _run(program):
        return passes.insert_mx_scale_addr()(passes.infer_tile_memory_space()(program))

    @staticmethod
    def _collect_calls(program, op_name: str):
        found = []
        registered_op_name = ir.get_op(op_name).name

        class _Collect(ir.IRVisitor):
            def visit_call(self, op):
                if op.op.name == registered_op_name:
                    found.append(op)
                super().visit_call(op)

        _Collect().visit_program(program)
        return found

    @staticmethod
    def _collect_mx_binding_events(program):
        events = []

        def var_ref(expr: ir.Expr) -> tuple[str, int]:
            assert isinstance(expr, ir.Var)
            return expr.name_hint, expr.unique_id

        class _Collect(ir.IRVisitor):
            def visit_assign_stmt(self, stmt):
                call = stmt.value if isinstance(stmt.value, ir.Call) else None
                if call is not None and call.op.name == ir.get_op("tile.tget_scale_addr").name:
                    events.append(("bind", var_ref(stmt.var), var_ref(call.args[0]), var_ref(call.args[1])))
                elif call is not None and call.op.name == ir.get_op("tile.matmul_mx").name:
                    events.append(("matmul", *(var_ref(arg) for arg in call.args)))
                super().visit_assign_stmt(stmt)

            def visit_if_stmt(self, stmt):
                events.append(("then",))
                self.visit_stmt(stmt.then_body)
                if stmt.else_body is not None:
                    events.append(("else",))
                    self.visit_stmt(stmt.else_body)

        _Collect().visit_program(program)
        return events

    def test_left_and_right_pairs_from_matmul_mx_without_explicit_l0_moves(self):
        """Mat loads + tget + matmul_mx: Infer must pick Left* / Right* pairs (not Left* on rhs)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                a_s: pl.Tensor[[16, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                b_s: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                ta = pl.load(a, [0, 0], [16, 64], target_memory=pl.Mem.Mat)
                tas = pl.load(a_s, [0, 0], [16, 2], target_memory=pl.Mem.Mat)
                tb = pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat)
                tbs = pl.load(b_s, [0, 0], [2, 32], target_memory=pl.Mem.Mat)
                c = pl.matmul_mx(ta, tas, tb, tbs)
                result = pl.store(c, [0, 0], out)
                return result

            @pl.function
            def main(
                self,
                a: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                a_s: pl.Tensor[[16, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                b_s: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                out = pl.create_tensor([16, 32], dtype=pl.FP32)
                return self.kernel(a, a_s, b, b_s, out)

        after = self._run(Before)
        matmuls = self._collect_calls(after, "tile.matmul_mx")
        tgets = self._collect_calls(after, "tile.tget_scale_addr")
        assert len(matmuls) == 1
        assert len(tgets) == 2

        mx = matmuls[0]
        assert [arg.type.memory_space for arg in mx.args] == [
            pl.MemorySpace.Left,
            pl.MemorySpace.LeftScale,
            pl.MemorySpace.Right,
            pl.MemorySpace.RightScale,
        ]

        pairs = {(t.args[0].type.memory_space, t.args[1].type.memory_space) for t in tgets}
        assert pairs == {
            (pl.MemorySpace.LeftScale, pl.MemorySpace.Left),
            (pl.MemorySpace.RightScale, pl.MemorySpace.Right),
        }

        moves = self._collect_calls(after, "tile.move")
        for m in moves:
            src_space = getattr(m.args[0].type, "memory_space", None) if m.args else None
            tgt = m.kwargs.get("target_memory")
            if src_space in (pl.MemorySpace.LeftScale, pl.MemorySpace.RightScale) and tgt in (
                pl.MemorySpace.LeftScale,
                pl.MemorySpace.RightScale,
            ):
                assert src_space == tgt, f"wrong scale-side move: {src_space} -> {tgt}"

    def test_repeated_pass_rebinds_generated_bound_results(self):
        """Bound results still alias mutable buffers and must be rebound on another pass run."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                a_s: pl.Tensor[[16, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                b_s: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                ta = pl.load(a, [0, 0], [16, 64], target_memory=pl.Mem.Mat)
                tas = pl.load(a_s, [0, 0], [16, 2], target_memory=pl.Mem.Mat)
                tb = pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat)
                tbs = pl.load(b_s, [0, 0], [2, 32], target_memory=pl.Mem.Mat)
                result = pl.matmul_mx(ta, tas, tb, tbs)
                stored = pl.store(result, [0, 0], out)
                return stored

            @pl.function
            def main(
                self,
                a: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                a_s: pl.Tensor[[16, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                b_s: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                out = pl.create_tensor([16, 32], dtype=pl.FP32)
                return self.kernel(a, a_s, b, b_s, out)

        once = self._run(Before)
        twice = passes.insert_mx_scale_addr()(once)
        once_tgets = self._collect_calls(once, "tile.tget_scale_addr")
        twice_tgets = self._collect_calls(twice, "tile.tget_scale_addr")
        assert len(once_tgets) == 2
        assert len(twice_tgets) == 4

        once_mx = self._collect_calls(once, "tile.matmul_mx")[0]
        twice_mx = self._collect_calls(twice, "tile.matmul_mx")[0]
        assert once_mx.args[1].name_hint.endswith("_bound")
        assert once_mx.args[3].name_hint.endswith("_bound")
        assert twice_mx.args[1].name_hint.endswith("_bound_bound")
        assert twice_mx.args[3].name_hint.endswith("_bound_bound")
        assert once_mx.args[1].unique_id != twice_mx.args[1].unique_id
        assert once_mx.args[3].unique_id != twice_mx.args[3].unique_id

    def test_shared_scales_across_mx_consumers_get_fresh_bindings(self):
        """Each consumer must rebind even when its scale SSA operands are identical."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                a_s: pl.Tensor[[16, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                b_s: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                ta = pl.load(a, [0, 0], [16, 64], target_memory=pl.Mem.Mat)
                tas = pl.load(a_s, [0, 0], [16, 2], target_memory=pl.Mem.Mat)
                tb = pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat)
                tbs = pl.load(b_s, [0, 0], [2, 32], target_memory=pl.Mem.Mat)
                acc = pl.matmul_mx(ta, tas, tb, tbs)
                result = pl.matmul_mx_acc(acc, ta, tas, tb, tbs)
                stored = pl.store(result, [0, 0], out)
                return stored

            @pl.function
            def main(
                self,
                a: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                a_s: pl.Tensor[[16, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                b_s: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                out = pl.create_tensor([16, 32], dtype=pl.FP32)
                return self.kernel(a, a_s, b, b_s, out)

        after = self._run(Before)
        tgets = self._collect_calls(after, "tile.tget_scale_addr")
        assert len(tgets) == 4

        mx = self._collect_calls(after, "tile.matmul_mx")[0]
        mx_acc = self._collect_calls(after, "tile.matmul_mx_acc")[0]
        assert mx.args[1].name_hint.endswith("_bound")
        assert mx.args[3].name_hint.endswith("_bound")
        assert mx_acc.args[2].name_hint.endswith("_bound")
        assert mx_acc.args[4].name_hint.endswith("_bound")
        assert mx.args[1].unique_id != mx_acc.args[2].unique_id
        assert mx.args[3].unique_id != mx_acc.args[4].unique_id

    def test_bindings_do_not_leak_between_if_branches(self):
        """Each branch must define the bound scale SSAs consumed in that branch."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                a_s: pl.Tensor[[16, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                b_s: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                flag: pl.Scalar[pl.INT32],
                out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                lhs = pl.move(pl.load(a, [0, 0], [16, 64]), target_memory=pl.Mem.Left)
                lhs_scale = pl.move(
                    pl.load(a_s, [0, 0], [16, 2], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.LeftScale,
                )
                rhs = pl.move(pl.load(b, [0, 0], [64, 32]), target_memory=pl.Mem.Right)
                rhs_scale = pl.move(
                    pl.load(b_s, [0, 0], [2, 32], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.RightScale,
                )
                if flag > 0:
                    then_value = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)
                    result = pl.yield_(then_value)
                else:
                    else_value = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)
                    result = pl.yield_(else_value)
                stored = pl.store(result, [0, 0], out)
                return stored

        after = self._run(Before)
        tgets = self._collect_calls(after, "tile.tget_scale_addr")
        matmuls = self._collect_calls(after, "tile.matmul_mx")
        assert len(tgets) == 4
        assert len(matmuls) == 2
        events = self._collect_mx_binding_events(after)
        assert [event[0] for event in events] == [
            "then",
            "bind",
            "bind",
            "matmul",
            "else",
            "bind",
            "bind",
            "matmul",
        ]
        then_lhs, then_rhs, then_mx = events[1:4]
        else_lhs, else_rhs, else_mx = events[5:8]
        assert [ref[0] for ref in then_lhs[1:]] == ["lhs_scale_bound", "lhs_scale", "lhs"]
        assert [ref[0] for ref in then_rhs[1:]] == ["rhs_scale_bound", "rhs_scale", "rhs"]
        assert then_mx[1:] == (then_lhs[3], then_lhs[1], then_rhs[3], then_rhs[1])
        assert else_mx[1:] == (else_lhs[3], else_lhs[1], else_rhs[3], else_rhs[1])
        assert then_lhs[1][1] != else_lhs[1][1]
        assert then_rhs[1][1] != else_rhs[1][1]

    def test_reusing_earlier_data_after_rebind_requires_fresh_binding(self):
        """lhs0 → lhs1 → lhs0 must rebind on the third use (in-place addr is stale)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a0: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                a1: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                a_s: pl.Tensor[[16, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                b_s: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                lhs0 = pl.move(pl.load(a0, [0, 0], [16, 64]), target_memory=pl.Mem.Left)
                lhs1 = pl.move(pl.load(a1, [0, 0], [16, 64]), target_memory=pl.Mem.Left)
                lhs_scale = pl.move(
                    pl.load(a_s, [0, 0], [16, 2], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.LeftScale,
                )
                rhs = pl.move(pl.load(b, [0, 0], [64, 32]), target_memory=pl.Mem.Right)
                rhs_scale = pl.move(
                    pl.load(b_s, [0, 0], [2, 32], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.RightScale,
                )
                _first = pl.matmul_mx(lhs0, lhs_scale, rhs, rhs_scale)
                _second = pl.matmul_mx(lhs1, lhs_scale, rhs, rhs_scale)
                third = pl.matmul_mx(lhs0, lhs_scale, rhs, rhs_scale)
                stored = pl.store(third, [0, 0], out)
                return stored

        after = self._run(Before)
        events = self._collect_mx_binding_events(after)
        assert [event[0] for event in events] == [
            "bind",
            "bind",
            "matmul",
            "bind",
            "bind",
            "matmul",
            "bind",
            "bind",
            "matmul",
        ]
        lhs0_a, rhs0, first_mx, lhs1_binding, rhs1, second_mx, lhs0_b, rhs2, third_mx = events
        assert [ref[0] for ref in lhs0_a[1:]] == ["lhs_scale_bound", "lhs_scale", "lhs0"]
        assert [ref[0] for ref in lhs1_binding[1:]] == ["lhs_scale_bound", "lhs_scale", "lhs1"]
        assert [ref[0] for ref in lhs0_b[1:]] == ["lhs_scale_bound", "lhs_scale", "lhs0"]
        assert lhs0_a[1][1] != lhs0_b[1][1]
        assert lhs0_a[1][1] != lhs1_binding[1][1]
        assert first_mx[1:] == (lhs0_a[3], lhs0_a[1], rhs0[3], rhs0[1])
        assert second_mx[1:] == (lhs1_binding[3], lhs1_binding[1], rhs1[3], rhs1[1])
        assert third_mx[1:] == (lhs0_b[3], lhs0_b[1], rhs2[3], rhs2[1])
        lhs_tgets = [e for e in events if e[0] == "bind" and e[3][0].startswith("lhs")]
        assert len(lhs_tgets) == 3
        rhs_tgets = [e for e in events if e[0] == "bind" and e[3][0] == "rhs"]
        assert len(rhs_tgets) == 3

    def test_bare_if_body_matmul_gets_bindings(self):
        """NormalizedStmtStructure may leave a sole AssignStmt as an if body."""
        span = ir.Span.unknown()

        def tile(name, shape, dtype, space):
            return ir.Var(name, ir.TileType(shape, dtype, memory_space=space), span)

        lhs = tile("lhs", [16, 64], ir.DataType.FP8E4M3FN, ir.MemorySpace.Left)
        lhs_scale = tile("lhs_scale", [16, 2], ir.DataType.FP8E8M0, ir.MemorySpace.LeftScale)
        rhs = tile("rhs", [64, 32], ir.DataType.FP8E4M3FN, ir.MemorySpace.Right)
        rhs_scale = tile("rhs_scale", [2, 32], ir.DataType.FP8E8M0, ir.MemorySpace.RightScale)
        flag = ir.Var("flag", ir.ScalarType(ir.DataType.INT32), span)

        mx_call = ir.op.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, span)
        assert isinstance(mx_call.type, ir.TileType)
        c = ir.Var("c", mx_call.type, span)
        bare_assign = ir.AssignStmt(c, mx_call, span)
        cond = ir.Gt(flag, ir.ConstInt(0, ir.DataType.INT32, span), ir.DataType.BOOL, span)
        # Bare then-body (no enclosing SeqStmts) — the shape NormalizeStmtStructure
        # produces when an if branch contains only one statement. No return_vars:
        # returning `c` from outside the if would be illegal SSA.
        if_stmt = ir.IfStmt(cond, bare_assign, None, [], span)
        func = ir.Function(
            "kernel",
            [lhs, lhs_scale, rhs, rhs_scale, flag],
            [],
            if_stmt,
            span,
            ir.FunctionType.InCore,
        )
        program = ir.Program([func], "bare_if_mx", span)

        after = passes.insert_mx_scale_addr()(program)
        tgets = self._collect_calls(after, "tile.tget_scale_addr")
        matmuls = self._collect_calls(after, "tile.matmul_mx")
        assert len(tgets) == 2
        assert len(matmuls) == 1
        assert matmuls[0].args[1].name_hint.endswith("_bound")
        assert matmuls[0].args[3].name_hint.endswith("_bound")

        # Bindings must live inside the then-body, not leak to the outer SeqStmts.
        events = self._collect_mx_binding_events(after)
        assert [event[0] for event in events] == ["then", "bind", "bind", "matmul"]

    @pytest.mark.parametrize(
        "func_type",
        [ir.FunctionType.AIC, ir.FunctionType.AIV],
        ids=["aic", "aiv"],
    )
    def test_incore_variants_get_bindings(self, func_type):
        """AIC/AIV mixed-kernel bodies must receive tget_scale_addr, not only InCore."""
        span = ir.Span.unknown()

        def tile(name, shape, dtype, space):
            return ir.Var(name, ir.TileType(shape, dtype, memory_space=space), span)

        lhs = tile("lhs", [16, 64], ir.DataType.FP8E4M3FN, ir.MemorySpace.Left)
        lhs_scale = tile("lhs_scale", [16, 2], ir.DataType.FP8E8M0, ir.MemorySpace.LeftScale)
        rhs = tile("rhs", [64, 32], ir.DataType.FP8E4M3FN, ir.MemorySpace.Right)
        rhs_scale = tile("rhs_scale", [2, 32], ir.DataType.FP8E8M0, ir.MemorySpace.RightScale)
        mx_call = ir.op.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, span)
        assert isinstance(mx_call.type, ir.TileType)
        c = ir.Var("c", mx_call.type, span)
        body = ir.AssignStmt(c, mx_call, span)
        func = ir.Function("kernel", [lhs, lhs_scale, rhs, rhs_scale], [], body, span, func_type)
        program = ir.Program([func], "incore_variant_mx", span)

        after = passes.insert_mx_scale_addr()(program)
        tgets = self._collect_calls(after, "tile.tget_scale_addr")
        matmuls = self._collect_calls(after, "tile.matmul_mx")
        assert len(tgets) == 2
        assert len(matmuls) == 1
        assert matmuls[0].args[1].name_hint.endswith("_bound")
        assert matmuls[0].args[3].name_hint.endswith("_bound")

    def test_iter_arg_data_and_scale_tiles_are_accepted(self):
        """pl.range init_values carrying data/scale tiles must not InternalError."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                a_s: pl.Tensor[[16, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                b_s: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                lhs = pl.move(pl.load(a, [0, 0], [16, 64]), target_memory=pl.Mem.Left)
                lhs_scale = pl.move(
                    pl.load(a_s, [0, 0], [16, 2], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.LeftScale,
                )
                rhs = pl.move(pl.load(b, [0, 0], [64, 32]), target_memory=pl.Mem.Right)
                rhs_scale = pl.move(
                    pl.load(b_s, [0, 0], [2, 32], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.RightScale,
                )
                acc0 = pl.tile.create([16, 32], dtype=pl.FP32, target_memory=pl.Mem.Acc)
                for _i, (lhs_i, ls_i, acc) in pl.range(0, 1, 1, init_values=(lhs, lhs_scale, acc0)):
                    acc_next = pl.matmul_mx_acc(acc, lhs_i, ls_i, rhs, rhs_scale)
                    _lhs_o, _ls_o, acc_out = pl.yield_(lhs_i, ls_i, acc_next)
                stored = pl.store(acc_out, [0, 0], out)
                return stored

        after = self._run(Before)
        tgets = self._collect_calls(after, "tile.tget_scale_addr")
        mx_acc = self._collect_calls(after, "tile.matmul_mx_acc")
        assert len(tgets) == 2
        assert len(mx_acc) == 1
        assert mx_acc[0].args[2].name_hint.endswith("_bound")
        assert mx_acc[0].args[4].name_hint.endswith("_bound")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
