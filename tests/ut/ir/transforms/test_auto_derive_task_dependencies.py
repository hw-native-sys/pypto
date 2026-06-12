# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for AutoDeriveTaskDependencies."""

from typing import cast

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.pypto_core import passes as _core_passes


@pytest.fixture(autouse=True)
def pass_verification_context():
    """Use property verification without round-trip checks for compiler attrs."""
    instruments: list[_core_passes.PassInstrument] = [
        _core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)
    ]
    with _core_passes.PassContext(instruments):
        yield


def _user_calls(program: ir.Program, name: str) -> list[ir.Call | ir.Submit]:
    calls: list[ir.Call | ir.Submit] = []

    def collect_stmt(stmt: ir.Stmt):
        if isinstance(stmt, ir.AssignStmt):
            value = stmt.value
            if isinstance(value, ir.Call | ir.Submit):
                op_name = value.op.name
                if not (
                    op_name.startswith("tile.")
                    or op_name.startswith("tensor.")
                    or op_name.startswith("system.")
                    or op_name.startswith("array.")
                ):
                    calls.append(value)
        if isinstance(stmt, ir.SeqStmts):
            for child in stmt.stmts:
                collect_stmt(child)
        elif isinstance(stmt, ir.RuntimeScopeStmt):
            collect_stmt(stmt.body)
        elif isinstance(stmt, ir.IfStmt):
            collect_stmt(stmt.then_body)
            if stmt.else_body is not None:
                collect_stmt(stmt.else_body)
        elif isinstance(stmt, ir.ForStmt | ir.WhileStmt):
            collect_stmt(stmt.body)

    for func in program.functions.values():
        collect_stmt(func.body)
    return [call for call in calls if call.op.name == name]


def _compiler_edges(call: ir.Call | ir.Submit) -> list[ir.Var]:
    return list(call.attrs.get("compiler_manual_dep_edges", []))


def _user_edges(call: ir.Call | ir.Submit) -> list[ir.Var]:
    if isinstance(call, ir.Submit):
        return cast(list[ir.Var], list(call.deps))
    return list(call.attrs.get("manual_dep_edges", []))


def _printed(program: ir.Program) -> str:
    return ir.python_print(program)


def _runtime_scopes(program: ir.Program) -> list[ir.RuntimeScopeStmt]:
    scopes: list[ir.RuntimeScopeStmt] = []

    class Collector(ir.IRVisitor):
        def visit_runtime_scope_stmt(self, op):
            scopes.append(op)
            super().visit_runtime_scope_stmt(op)

    Collector().visit_program(program)
    return scopes


def _run_auto_deps(program: ir.Program, *, analyze_auto_scopes: bool = False) -> ir.Program:
    program = passes.derive_call_directions()(program)
    return passes.auto_derive_task_dependencies(analyze_auto_scopes=analyze_auto_scopes)(program)


class TestAutoDeriveTaskDependencies:
    def test_manual_scope_raw_hazard_is_left_to_user_deps(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    produced, _producer_tid = pl.submit(self.fill, scratch)
                    out, _ = pl.submit(self.consume, produced)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is True
        consume_call = _user_calls(out, "consume")[0]
        assert _compiler_edges(consume_call) == []

    def test_auto_scope_raw_hazard_adds_compiler_edge_when_enabled(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    produced, producer_tid = pl.submit(self.fill, scratch)
                    out, _ = pl.submit(self.consume, produced)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert len(edges) == 1
        assert edges[0].name_hint == "producer_tid"

    def test_auto_scope_read_read_does_not_add_edge_when_enabled(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def read1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def read2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    _a, _tid = pl.submit(self.read1, x)
                    b, _ = pl.submit(self.read2, x)
                return b

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        read2_call = _user_calls(out, "read2")[0]
        assert _compiler_edges(read2_call) == []

    def test_auto_scope_waw_hazard_adds_compiler_edge_when_enabled(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 0, 256)]],
            ) -> pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 0, 256)]:
                return out

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(
                self,
                first: pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 0, 256)],
                second: pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 0, 256)],
            ) -> pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 0, 256)]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    _first, first_tid = pl.submit(self.fill, first)
                    out, _ = pl.submit(self.fill, second)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        second_fill = _user_calls(out, "fill")[1]
        edges = _compiler_edges(second_fill)
        assert len(edges) == 1
        assert edges[0].name_hint == "first_tid"

    def test_auto_scope_war_hazard_adds_compiler_edge_when_enabled(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def read(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    _seen, reader_tid = pl.submit(self.read, scratch)
                    out, _ = pl.submit(self.fill, scratch)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        fill_call = _user_calls(out, "fill")[0]
        edges = _compiler_edges(fill_call)
        assert len(edges) == 1
        assert edges[0].name_hint == "reader_tid"

    def test_default_auto_scope_raw_hazard_skips_compiler_edge(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                produced, _producer_tid = pl.submit(self.fill, scratch)
                out, _ = pl.submit(self.consume, produced)
                return out

        out = _run_auto_deps(Prog)
        assert "compiler_manual_dep_edges" not in _printed(out)

    def test_auto_runtime_scope_raw_hazard_skips_compiler_edge_by_default(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    produced, _producer_tid = pl.submit(self.fill, scratch)
                    out, _ = pl.submit(self.consume, produced)
                return out

        out = _run_auto_deps(Prog)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is False

        assert "compiler_manual_dep_edges" not in _printed(out)

    def test_auto_runtime_scope_raw_hazard_adds_compiler_edge_when_enabled_and_stays_auto(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    produced = self.fill(scratch)
                    out = self.consume(produced)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is False

        printed = _printed(out)
        assert '"compiler_manual_dep_edges": [produced]' in printed

    def test_auto_runtime_scope_dynamic_hazard_falls_back_without_stale_edges(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(
                self,
                scratch: pl.Tensor[[64], pl.FP32],
                src: pl.Tensor[[4, 16], pl.FP32],
                index: pl.Tensor[[4, 8], pl.INT32],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    produced, _producer_tid = pl.submit(self.fill, scratch)
                    produced = self.fill(produced)
                    gathered = pl.tensor.gather(src, -1, index)
                    out, _ = pl.submit(self.consume, gathered)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is False

        assert "compiler_manual_dep_edges" not in _printed(out)

    def test_user_edges_are_preserved_separately_from_compiler_edges(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def unrelated(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(
                self,
                scratch: pl.Tensor[[64], pl.FP32],
                other: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    produced, producer_tid = pl.submit(self.fill, scratch)
                    _unused, user_tid = pl.submit(self.unrelated, other)
                    out, _ = pl.submit(self.consume, produced, deps=[user_tid])
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        consume_call = _user_calls(out, "consume")[0]
        user_edges = _user_edges(consume_call)
        compiler_edges = _compiler_edges(consume_call)
        assert [edge.name_hint for edge in user_edges] == ["user_tid"]
        assert [edge.name_hint for edge in compiler_edges] == ["producer_tid"]

    def test_user_edge_covering_submit_return_keeps_manual_scope(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def producer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def consumer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    produced, producer_tid = pl.submit(self.producer, scratch)
                    out, _ = pl.submit(self.consumer, produced, deps=[producer_tid])
                return out

        out = _run_auto_deps(Prog)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is True

    def test_static_disjoint_slices_do_not_add_compiler_edge(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[32], pl.FP32]],
            ) -> pl.Tensor[[32], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[32], pl.FP32]) -> pl.Tensor[[32], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[32], pl.FP32]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    left = scratch[0:32]
                    right = scratch[32:64]
                    _produced, producer_tid = pl.submit(self.fill, left)
                    out, _ = pl.submit(self.consume, right)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        assert "compiler_manual_dep_edges" not in _printed(out)

    def test_packed_nd_tensor_view_disjoint_slices_do_not_add_compiler_edge(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[
                    pl.Tensor[
                        [32, 32],
                        pl.FP32,
                        pl.TensorView(stride=[32, 1], layout=pl.TensorLayout.ND),
                    ]
                ],
            ) -> pl.Tensor[
                [32, 32],
                pl.FP32,
                pl.TensorView(stride=[32, 1], layout=pl.TensorLayout.ND),
            ]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(
                self,
                x: pl.Tensor[
                    [32, 32],
                    pl.FP32,
                    pl.TensorView(stride=[32, 1], layout=pl.TensorLayout.ND),
                ],
            ) -> pl.Tensor[
                [32, 32],
                pl.FP32,
                pl.TensorView(stride=[32, 1], layout=pl.TensorLayout.ND),
            ]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(
                self,
                scratch: pl.Tensor[
                    [64, 32],
                    pl.FP32,
                    pl.TensorView(stride=[32, 1], layout=pl.TensorLayout.ND),
                ],
            ) -> pl.Tensor[
                [32, 32],
                pl.FP32,
                pl.TensorView(stride=[32, 1], layout=pl.TensorLayout.ND),
            ]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    top = scratch[0:32, 0:32]
                    bottom = scratch[32:64, 0:32]
                    _produced, producer_tid = pl.submit(self.fill, top)
                    out, _ = pl.submit(self.consume, bottom)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        assert "compiler_manual_dep_edges" not in _printed(out)

    def test_strided_nd_tensor_view_disjoint_slices_stay_conservative(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[
                    pl.Tensor[
                        [16, 32],
                        pl.FP32,
                        pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND),
                    ]
                ],
            ) -> pl.Tensor[
                [16, 32],
                pl.FP32,
                pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND),
            ]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(
                self,
                x: pl.Tensor[
                    [16, 32],
                    pl.FP32,
                    pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND),
                ],
            ) -> pl.Tensor[
                [16, 32],
                pl.FP32,
                pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND),
            ]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(
                self,
                scratch: pl.Tensor[
                    [32, 32],
                    pl.FP32,
                    pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND),
                ],
            ) -> pl.Tensor[
                [16, 32],
                pl.FP32,
                pl.TensorView(stride=[64, 1], layout=pl.TensorLayout.ND),
            ]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    top = scratch[0:16, 0:32]
                    bottom = scratch[16:32, 0:32]
                    _produced, producer_tid = pl.submit(self.fill, top)
                    out, _ = pl.submit(self.consume, bottom)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert len(edges) == 1
        assert edges[0].name_hint == "producer_tid"

    def test_dn_tensor_view_disjoint_slices_stay_conservative(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[
                    pl.Tensor[
                        [16, 32],
                        pl.FP32,
                        pl.TensorView(stride=[1, 16], layout=pl.TensorLayout.DN),
                    ]
                ],
            ) -> pl.Tensor[
                [16, 32],
                pl.FP32,
                pl.TensorView(stride=[1, 16], layout=pl.TensorLayout.DN),
            ]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(
                self,
                x: pl.Tensor[
                    [16, 32],
                    pl.FP32,
                    pl.TensorView(stride=[1, 16], layout=pl.TensorLayout.DN),
                ],
            ) -> pl.Tensor[
                [16, 32],
                pl.FP32,
                pl.TensorView(stride=[1, 16], layout=pl.TensorLayout.DN),
            ]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(
                self,
                scratch: pl.Tensor[
                    [32, 32],
                    pl.FP32,
                    pl.TensorView(stride=[1, 32], layout=pl.TensorLayout.DN),
                ],
            ) -> pl.Tensor[
                [16, 32],
                pl.FP32,
                pl.TensorView(stride=[1, 16], layout=pl.TensorLayout.DN),
            ]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    top = scratch[0:16, 0:32]
                    bottom = scratch[16:32, 0:32]
                    _produced, producer_tid = pl.submit(self.fill, top)
                    out, _ = pl.submit(self.consume, bottom)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert len(edges) == 1
        assert edges[0].name_hint == "producer_tid"

    def test_tensor_view_valid_shape_disjoint_slices_stay_conservative(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[
                    pl.Tensor[
                        [16, 32],
                        pl.FP32,
                        pl.TensorView(stride=[32, 1], layout=pl.TensorLayout.ND, valid_shape=[15, 32]),
                    ]
                ],
            ) -> pl.Tensor[
                [16, 32],
                pl.FP32,
                pl.TensorView(stride=[32, 1], layout=pl.TensorLayout.ND, valid_shape=[15, 32]),
            ]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(
                self,
                x: pl.Tensor[
                    [16, 32],
                    pl.FP32,
                    pl.TensorView(stride=[32, 1], layout=pl.TensorLayout.ND, valid_shape=[15, 32]),
                ],
            ) -> pl.Tensor[
                [16, 32],
                pl.FP32,
                pl.TensorView(stride=[32, 1], layout=pl.TensorLayout.ND, valid_shape=[15, 32]),
            ]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(
                self,
                scratch: pl.Tensor[
                    [32, 32],
                    pl.FP32,
                    pl.TensorView(stride=[32, 1], layout=pl.TensorLayout.ND, valid_shape=[31, 32]),
                ],
            ) -> pl.Tensor[
                [16, 32],
                pl.FP32,
                pl.TensorView(stride=[32, 1], layout=pl.TensorLayout.ND, valid_shape=[15, 32]),
            ]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    top = scratch[0:16, 0:32]
                    bottom = scratch[16:32, 0:32]
                    _produced, producer_tid = pl.submit(self.fill, top)
                    out, _ = pl.submit(self.consume, bottom)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert len(edges) == 1
        assert edges[0].name_hint == "producer_tid"

    def test_static_overlapping_slices_add_compiler_edge(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[32], pl.FP32]],
            ) -> pl.Tensor[[32], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[32], pl.FP32]) -> pl.Tensor[[32], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[32], pl.FP32]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    left = scratch[0:32]
                    mid = scratch[16:48]
                    _produced, producer_tid = pl.submit(self.fill, left)
                    out, _ = pl.submit(self.consume, mid)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert len(edges) == 1
        assert edges[0].name_hint == "producer_tid"

    def test_symbolic_slice_offset_stays_conservative(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[32], pl.FP32]],
            ) -> pl.Tensor[[32], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[32], pl.FP32]) -> pl.Tensor[[32], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(
                self,
                scratch: pl.Tensor[[64], pl.FP32],
                offset: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[32], pl.FP32]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    left = scratch[0:32]
                    dynamic = pl.slice(scratch, [32], [offset])
                    _produced, producer_tid = pl.submit(self.fill, left)
                    out, _ = pl.submit(self.consume, dynamic)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert len(edges) == 1
        assert edges[0].name_hint == "producer_tid"

    def test_if_yield_return_var_keeps_storage_lineage(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(
                self,
                scratch: pl.Tensor[[64], pl.FP32],
                cond: pl.Scalar[pl.BOOL],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    produced, producer_tid = pl.submit(self.fill, scratch)
                    if cond:
                        selected = pl.yield_(produced)
                    else:
                        selected = pl.yield_(produced)
                    out, _ = pl.submit(self.consume, selected)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert len(edges) == 1
        assert edges[0].name_hint == "producer_tid"

    def test_if_yield_different_roots_adds_edges_for_both_possible_producers(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(
                self,
                left: pl.Tensor[[64], pl.FP32],
                right: pl.Tensor[[64], pl.FP32],
                cond: pl.Scalar[pl.BOOL],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    produced_left, left_tid = pl.submit(self.fill, left)
                    produced_right, right_tid = pl.submit(self.fill, right)
                    if cond:
                        selected = pl.yield_(produced_left)
                    else:
                        selected = pl.yield_(produced_right)
                    out, _ = pl.submit(self.consume, selected)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert [edge.name_hint for edge in edges] == ["left_tid", "right_tid"]

    def test_loop_yield_different_root_adds_edges_for_init_and_yield_roots(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(
                self,
                left: pl.Tensor[[64], pl.FP32],
                right: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    produced_left, left_tid = pl.submit(self.fill, left)
                    produced_right, right_tid = pl.submit(self.fill, right)
                    for _i, (selected_iter,) in pl.range(0, 4, init_values=(produced_left,)):
                        selected = pl.yield_(produced_right)
                    out, _ = pl.submit(self.consume, selected)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert [edge.name_hint for edge in edges] == ["left_tid", "right_tid"]

    def test_if_yield_mixed_known_and_unresolved_location_falls_back(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[4, 8], pl.FP32]) -> pl.Tensor[[4, 8], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(
                self,
                scratch: pl.Tensor[[4, 8], pl.FP32],
                src: pl.Tensor[[4, 16], pl.FP32],
                index: pl.Tensor[[4, 8], pl.INT32],
                cond: pl.Scalar[pl.BOOL],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    produced, _producer_tid = pl.submit(self.fill, scratch)
                    dynamic = pl.tensor.gather(src, -1, index)
                    if cond:
                        selected = pl.yield_(produced)
                    else:
                        selected = pl.yield_(dynamic)
                    out, _ = pl.submit(self.consume, selected)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is False

    def test_loop_yield_mixed_known_and_unresolved_location_falls_back(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[4, 8], pl.FP32]) -> pl.Tensor[[4, 8], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(
                self,
                scratch: pl.Tensor[[4, 8], pl.FP32],
                src: pl.Tensor[[4, 16], pl.FP32],
                index: pl.Tensor[[4, 8], pl.INT32],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    produced, _producer_tid = pl.submit(self.fill, scratch)
                    dynamic = pl.tensor.gather(src, -1, index)
                    for _i, (selected_iter,) in pl.range(0, 4, init_values=(produced,)):
                        selected = pl.yield_(dynamic)
                    out, _ = pl.submit(self.consume, selected)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is False

    def test_memref_may_alias_adds_compiler_edge(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 0, 256)]],
            ) -> pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 0, 256)]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(
                self,
                x: pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 128, 256)],
            ) -> pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 128, 256)]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(
                self,
                left: pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 0, 256)],
                right: pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 128, 256)],
            ) -> pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 128, 256)]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    _produced, producer_tid = pl.submit(self.fill, left)
                    out, _ = pl.submit(self.consume, right)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert len(edges) == 1
        assert edges[0].name_hint == "producer_tid"

    def test_plain_call_auto_scope_hazard_adds_synthetic_edge_when_enabled(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    produced = self.fill(scratch)
                    out, _ = pl.submit(self.consume, produced)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is False
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert len(edges) == 1
        assert edges[0].name_hint == "produced"

    def test_dynamic_gather_result_falls_back_to_auto_scope(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[4, 8], pl.FP32]) -> pl.Tensor[[4, 8], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(
                self,
                src: pl.Tensor[[4, 16], pl.FP32],
                index: pl.Tensor[[4, 8], pl.INT32],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    gathered = pl.tensor.gather(src, -1, index)
                    out, _ = pl.submit(self.consume, gathered)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is False

    def test_loop_dynamic_fan_in_producer_falls_back_to_auto_scope(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    carried = scratch
                    for _i in pl.range(0, 4):
                        carried, _producer_tid = pl.submit(self.fill, carried)
                    out, _ = pl.submit(self.consume, carried)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is False

    def test_fixed_trip_loop_fan_in_producer_exports_task_id_collection(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                carried = scratch
                last_tid = pl.system.task_invalid()
                for _i, (carried, last_tid) in pl.range(
                    0,
                    4,
                    init_values=(carried, last_tid),  # pyright: ignore[reportArgumentType]
                ):
                    carried, last_tid = pl.submit(self.fill, carried)
                    carried, last_tid = pl.yield_(carried, last_tid)
                out = self.consume(carried)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert len(edges) == 1
        assert edges[0].name_hint.startswith("last_tid")

    def test_dynamic_trip_loop_fan_in_producer_exports_task_id_collection(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                scratch: pl.Tensor[[64], pl.FP32],
                n_steps: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[64], pl.FP32]:
                carried = scratch
                last_tid = pl.system.task_invalid()
                for _i, (carried, last_tid) in pl.range(
                    0,
                    n_steps,
                    init_values=(carried, last_tid),  # pyright: ignore[reportArgumentType]
                ):
                    carried, last_tid = pl.submit(self.fill, carried)
                    carried, last_tid = pl.yield_(carried, last_tid)
                out = self.consume(carried)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert len(edges) == 1
        assert edges[0].name_hint.startswith("last_tid")

    def test_dynamic_trip_tensor_carrier_exports_task_id_collection(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                scratch: pl.Tensor[[64], pl.FP32],
                n_steps: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[64], pl.FP32]:
                carried = scratch
                for _i, (carried,) in pl.range(
                    0,
                    n_steps,
                    init_values=(carried,),
                ):
                    carried = self.fill(carried)
                    carried = pl.yield_(carried)
                out = self.consume(carried)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert len(edges) == 1
        assert edges[0].name_hint.startswith("carried")

    def test_dynamic_trip_tuple_output_tensor_carrier_exports_task_id_collection(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill_pair(
                self,
                a: pl.Out[pl.Tensor[[64], pl.FP32]],
                b: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                return a, b

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                left: pl.Tensor[[64], pl.FP32],
                right: pl.Tensor[[64], pl.FP32],
                n_steps: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[64], pl.FP32]:
                for _i, (left_carried, right_carried) in pl.range(
                    0,
                    n_steps,
                    init_values=(left, right),
                ):
                    left_carried, right_carried = self.fill_pair(left_carried, right_carried)
                    left_carried, right_carried = pl.yield_(left_carried, right_carried)
                out = self.consume(right_carried)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert len(edges) == 1
        assert edges[0].name_hint.startswith("right")

    def test_loop_direct_body_tid_dep_behavior(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    carried = scratch
                    for _i in pl.range(0, 4):
                        carried, producer_tid = pl.submit(self.fill, carried)
                    out, _ = pl.submit(self.consume, carried, deps=[producer_tid])
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is True

        consume_call = _user_calls(out, "consume")[0]
        user_edges = _user_edges(consume_call)
        assert [edge.name_hint for edge in user_edges] == ["producer_tid"]
        assert _compiler_edges(consume_call) == []

    def test_loop_carried_last_tid_dep_behavior(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    carried = scratch
                    last_tid = None
                    for _i, (carried, last_tid) in pl.range(
                        0,
                        4,
                        init_values=(carried, last_tid),  # pyright: ignore[reportArgumentType]
                    ):
                        carried, last_tid = pl.submit(self.fill, carried)
                        carried, last_tid = pl.yield_(carried, last_tid)
                    out, _ = pl.submit(self.consume, carried, deps=[last_tid])
                return out

        out = _run_auto_deps(Prog)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is True

        consume_call = _user_calls(out, "consume")[0]
        user_edges = _user_edges(consume_call)
        assert [edge.name_hint for edge in user_edges] == ["last_tid"]
        assert _compiler_edges(consume_call) == []

    def test_loop_task_id_array_deps_keep_manual_scope(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    tids = pl.array.create(4, pl.TASK_ID)
                    for n in pl.parallel(4):
                        _produced, producer_tid = pl.submit(self.fill, scratch)
                        tids[n] = producer_tid
                    out, _ = pl.submit(self.consume, scratch, deps=[tids[k] for k in range(4)])
                return out

        out = _run_auto_deps(Prog)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is True

        consume_call = _user_calls(out, "consume")[0]
        assert _compiler_edges(consume_call) == []

    def test_loop_task_id_array_slot_temps_keep_manual_scope(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    tids = pl.array.create(4, pl.TASK_ID)
                    for n in pl.parallel(4):
                        _produced, producer_tid = pl.submit(self.fill, scratch)
                        tids[n] = producer_tid
                    dep0 = tids[0]
                    dep1 = tids[1]
                    dep2 = tids[2]
                    dep3 = tids[3]
                    out, _ = pl.submit(self.consume, scratch, deps=[dep0, dep1, dep2, dep3])
                return out

        out = _run_auto_deps(Prog)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is True

        consume_call = _user_calls(out, "consume")[0]
        assert _compiler_edges(consume_call) == []

    def test_partial_loop_task_id_array_deps_keep_manual_scope(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    tids = pl.array.create(4, pl.TASK_ID)
                    for n in pl.parallel(4):
                        _produced, producer_tid = pl.submit(self.fill, scratch)
                        tids[n] = producer_tid
                    out, _ = pl.submit(self.consume, scratch, deps=[tids[0], tids[1]])
                return out

        out = _run_auto_deps(Prog)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is True

        consume_call = _user_calls(out, "consume")[0]
        assert _compiler_edges(consume_call) == []

    def test_partial_user_deps_with_dynamic_auto_hazard_still_falls_back(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def unrelated(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(
                self,
                scratch: pl.Tensor[[64], pl.FP32],
                other: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    carried = scratch
                    for _i in pl.range(0, 4):
                        carried, _producer_tid = pl.submit(self.fill, carried)
                    _unrelated0, unrelated_tid0 = pl.submit(self.unrelated, other)
                    _unrelated1, unrelated_tid1 = pl.submit(self.unrelated, other)
                    out, _ = pl.submit(self.consume, carried, deps=[unrelated_tid0, unrelated_tid1])
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is False

        consume_call = _user_calls(out, "consume")[0]
        assert _compiler_edges(consume_call) == []

    def test_fallback_strips_previous_compiler_edges(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(
                self,
                scratch: pl.Tensor[[64], pl.FP32],
                src: pl.Tensor[[4, 16], pl.FP32],
                index: pl.Tensor[[4, 8], pl.INT32],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    produced, _producer_tid = pl.submit(self.fill, scratch)
                    _first, _ = pl.submit(self.consume, produced)
                    gathered = pl.tensor.gather(src, -1, index)
                    out, _ = pl.submit(self.consume, gathered)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is False
        for call in _user_calls(out, "consume"):
            assert _compiler_edges(call) == []

    def test_default_auto_scope_plain_call_raw_hazard_skips_synthetic_task_id_edge(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                produced = self.fill(scratch)
                out = self.consume(produced)
                return out

        out = _run_auto_deps(Prog)
        consume_call = _user_calls(out, "consume")[0]
        assert _compiler_edges(consume_call) == []

    def test_default_auto_scope_plain_call_raw_hazard_adds_synthetic_task_id_edge_when_enabled(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                produced = self.fill(scratch)
                out = self.consume(produced)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        printed = _printed(out)
        assert '"compiler_manual_dep_edges": [produced]' in printed

    def test_sibling_auto_scopes_raw_hazard_exports_compiler_edge(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[4, 16], pl.FP32]],
            ) -> pl.Tensor[[4, 16], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[4, 16], pl.FP32]) -> pl.Tensor[[4, 16], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.scope():
                    produced = self.fill(scratch)
                with pl.scope():
                    out = self.consume(produced)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert len(edges) == 1
        assert edges[0].name_hint == "produced"

    def test_nested_scope_inner_producer_outer_consumer_exports_edge(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.scope():
                    with pl.scope():
                        produced = self.fill(scratch)
                    out = self.consume(produced)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert len(edges) == 1
        assert edges[0].name_hint == "produced"

    def test_cross_scope_read_read_does_not_add_edge(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def read_a(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def read_b(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.scope():
                    _a = self.read_a(x)
                with pl.scope():
                    b = self.read_b(x)
                return b

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        read_b_call = _user_calls(out, "read_b")[0]
        assert _compiler_edges(read_b_call) == []

    def test_fallback_scope_does_not_export_accesses(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(
                self,
                scratch: pl.Tensor[[4, 16], pl.FP32],
                index: pl.Tensor[[4, 16], pl.INT32],
            ) -> pl.Tensor[[4, 16], pl.FP32]:
                with pl.scope():
                    dynamic = pl.tensor.gather(scratch, -1, index)
                    produced = self.fill(dynamic)
                out = self.consume(produced)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        consume_call = _user_calls(out, "consume")[0]
        assert _compiler_edges(consume_call) == []

    def test_single_trip_loop_producer_exports_scalar_task_id(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                temp = pl.create_tensor([64], dtype=pl.FP32)
                for _i, (carried,) in pl.range(0, 16, 16, init_values=(temp,)):
                    with pl.scope():
                        produced = self.fill(carried)
                    produced = pl.yield_(produced)
                with pl.scope():
                    out = self.consume(produced)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert len(edges) == 1
        assert edges[0].name_hint == "produced"

    def test_whole_body_dynamic_skip_preserves_later_static_cross_scope_edge(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                scratch: pl.Tensor[[64], pl.FP32],
                later: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                loop_buf = pl.create_tensor([64], dtype=pl.FP32)
                for _i, (carried,) in pl.range(0, 4, init_values=(loop_buf,)):
                    loop_produced = self.fill(carried)
                    loop_produced = pl.yield_(loop_produced)
                self.consume(loop_produced)
                later_produced = self.fill(later)
                out = self.consume(later_produced)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        printed = str(out)
        assert "self.consume(loop_produced" in printed
        assert '"compiler_manual_dep_edges": [loop_produced]' in printed

        consume_calls = _user_calls(out, "consume")
        edge_names = []
        for call in consume_calls:
            edges = _compiler_edges(call)
            assert len(edges) == 1
            edge_names.append(edges[0].name_hint)
        assert "later_produced" in edge_names

    def test_large_control_flow_root_set_falls_back_to_auto_scope(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(
                self,
                t0: pl.Tensor[[64], pl.FP32],
                t1: pl.Tensor[[64], pl.FP32],
                t2: pl.Tensor[[64], pl.FP32],
                t3: pl.Tensor[[64], pl.FP32],
                t4: pl.Tensor[[64], pl.FP32],
                c0: pl.Scalar[pl.BOOL],
                c1: pl.Scalar[pl.BOOL],
                c2: pl.Scalar[pl.BOOL],
                c3: pl.Scalar[pl.BOOL],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    p0, _tid0 = pl.submit(self.fill, t0)
                    p1, _tid1 = pl.submit(self.fill, t1)
                    p2, _tid2 = pl.submit(self.fill, t2)
                    p3, _tid3 = pl.submit(self.fill, t3)
                    p4, _tid4 = pl.submit(self.fill, t4)
                    if c0:
                        selected_a = pl.yield_(p0)
                    else:
                        selected_a = pl.yield_(p1)
                    if c1:
                        selected_b = pl.yield_(p2)
                    else:
                        selected_b = pl.yield_(p3)
                    if c2:
                        selected_c = pl.yield_(selected_a)
                    else:
                        selected_c = pl.yield_(selected_b)
                    if c3:
                        selected = pl.yield_(selected_c)
                    else:
                        selected = pl.yield_(p4)
                    out, _ = pl.submit(self.consume, selected)
                return out

        out = _run_auto_deps(Prog, analyze_auto_scopes=True)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is False

        consume_call = _user_calls(out, "consume")[0]
        assert _compiler_edges(consume_call) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
