# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for AutoDeriveTaskDependencies."""

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


class _UserCallCollector(ir.IRVisitor):
    def __init__(self):
        super().__init__()
        self.calls: list[ir.Call] = []

    def visit_call(self, op):
        name = op.op.name
        if not (
            name.startswith("tile.")
            or name.startswith("tensor.")
            or name.startswith("system.")
            or name.startswith("array.")
        ):
            self.calls.append(op)
        super().visit_call(op)


def _user_calls(program: ir.Program, name: str) -> list[ir.Call]:
    collector = _UserCallCollector()
    collector.visit_program(program)
    return [call for call in collector.calls if call.op.name == name]


def _compiler_edges(call: ir.Call) -> list[ir.Var]:
    return list(call.attrs.get("compiler_manual_dep_edges", []))


def _runtime_scopes(program: ir.Program) -> list[ir.RuntimeScopeStmt]:
    scopes: list[ir.RuntimeScopeStmt] = []

    class Collector(ir.IRVisitor):
        def visit_runtime_scope_stmt(self, op):
            scopes.append(op)
            super().visit_runtime_scope_stmt(op)

    Collector().visit_program(program)
    return scopes


def _run_auto_deps(program: ir.Program) -> ir.Program:
    program = passes.derive_call_directions()(program)
    return passes.auto_derive_task_dependencies()(program)


class TestAutoDeriveTaskDependencies:
    def test_manual_scope_raw_hazard_adds_compiler_edge(self):
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
                    produced, producer_tid = pl.submit(self.fill, scratch)
                    out, _ = pl.submit(self.consume, produced)
                return out

        out = _run_auto_deps(Prog)
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert len(edges) == 1
        assert edges[0].name_hint == "producer_tid"

    def test_manual_scope_read_read_does_not_add_edge(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def read1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def read2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    _a, _tid = pl.submit(self.read1, x)
                    b, _ = pl.submit(self.read2, x)
                return b

        out = _run_auto_deps(Prog)
        read2_call = _user_calls(out, "read2")[0]
        assert _compiler_edges(read2_call) == []

    def test_manual_scope_waw_hazard_adds_compiler_edge(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 0, 256)]],
            ) -> pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 0, 256)]:
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                first: pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 0, 256)],
                second: pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 0, 256)],
            ) -> pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 0, 256)]:
                with pl.manual_scope():
                    _first, first_tid = pl.submit(self.fill, first)
                    out, _ = pl.submit(self.fill, second)
                return out

        out = _run_auto_deps(Prog)
        second_fill = _user_calls(out, "fill")[1]
        edges = _compiler_edges(second_fill)
        assert len(edges) == 1
        assert edges[0].name_hint == "first_tid"

    def test_manual_scope_war_hazard_adds_compiler_edge(self):
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

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    _seen, reader_tid = pl.submit(self.read, scratch)
                    out, _ = pl.submit(self.fill, scratch)
                return out

        out = _run_auto_deps(Prog)
        fill_call = _user_calls(out, "fill")[0]
        edges = _compiler_edges(fill_call)
        assert len(edges) == 1
        assert edges[0].name_hint == "reader_tid"

    def test_auto_scope_is_unchanged(self):
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
        consume_call = _user_calls(out, "consume")[0]
        assert "compiler_manual_dep_edges" not in consume_call.attrs

    def test_auto_runtime_scope_raw_hazard_adds_compiler_edge_and_stays_auto(self):
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
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    produced, producer_tid = pl.submit(self.fill, scratch)
                    out, _ = pl.submit(self.consume, produced)
                return out

        out = _run_auto_deps(Prog)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is False

        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert len(edges) == 1
        assert edges[0].name_hint == "producer_tid"

    def test_auto_runtime_scope_unencodable_hazard_falls_back_without_stale_edges(self):
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
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    produced, _producer_tid = pl.submit(self.fill, scratch)
                    produced = self.fill(produced)
                    out, _ = pl.submit(self.consume, produced)
                return out

        out = _run_auto_deps(Prog)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is False

        consume_call = _user_calls(out, "consume")[0]
        assert _compiler_edges(consume_call) == []

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

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                scratch: pl.Tensor[[64], pl.FP32],
                other: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    produced, producer_tid = pl.submit(self.fill, scratch)
                    _unused, user_tid = pl.submit(self.unrelated, other)
                    out, _ = pl.submit(self.consume, produced, deps=[user_tid])
                return out

        out = _run_auto_deps(Prog)
        consume_call = _user_calls(out, "consume")[0]
        user_edges = list(consume_call.attrs.get("manual_dep_edges", []))
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

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[32], pl.FP32]:
                with pl.manual_scope():
                    left = scratch[0:32]
                    right = scratch[32:64]
                    _produced, producer_tid = pl.submit(self.fill, left)
                    out, _ = pl.submit(self.consume, right)
                return out

        out = _run_auto_deps(Prog)
        consume_call = _user_calls(out, "consume")[0]
        assert _compiler_edges(consume_call) == []

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

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[32], pl.FP32]:
                with pl.manual_scope():
                    left = scratch[0:32]
                    mid = scratch[16:48]
                    _produced, producer_tid = pl.submit(self.fill, left)
                    out, _ = pl.submit(self.consume, mid)
                return out

        out = _run_auto_deps(Prog)
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

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                scratch: pl.Tensor[[64], pl.FP32],
                offset: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[32], pl.FP32]:
                with pl.manual_scope():
                    left = scratch[0:32]
                    dynamic = pl.slice(scratch, [32], [offset])
                    _produced, producer_tid = pl.submit(self.fill, left)
                    out, _ = pl.submit(self.consume, dynamic)
                return out

        out = _run_auto_deps(Prog)
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

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                scratch: pl.Tensor[[64], pl.FP32],
                cond: pl.Scalar[pl.BOOL],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    produced, producer_tid = pl.submit(self.fill, scratch)
                    if cond:
                        selected = pl.yield_(produced)
                    else:
                        selected = pl.yield_(produced)
                    out, _ = pl.submit(self.consume, selected)
                return out

        out = _run_auto_deps(Prog)
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

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                left: pl.Tensor[[64], pl.FP32],
                right: pl.Tensor[[64], pl.FP32],
                cond: pl.Scalar[pl.BOOL],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    produced_left, left_tid = pl.submit(self.fill, left)
                    produced_right, right_tid = pl.submit(self.fill, right)
                    if cond:
                        selected = pl.yield_(produced_left)
                    else:
                        selected = pl.yield_(produced_right)
                    out, _ = pl.submit(self.consume, selected)
                return out

        out = _run_auto_deps(Prog)
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

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                left: pl.Tensor[[64], pl.FP32],
                right: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    produced_left, left_tid = pl.submit(self.fill, left)
                    produced_right, right_tid = pl.submit(self.fill, right)
                    for _i, (selected_iter,) in pl.range(0, 4, init_values=(produced_left,)):
                        selected = pl.yield_(produced_right)
                    out, _ = pl.submit(self.consume, selected)
                return out

        out = _run_auto_deps(Prog)
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

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                scratch: pl.Tensor[[4, 8], pl.FP32],
                src: pl.Tensor[[4, 16], pl.FP32],
                index: pl.Tensor[[4, 8], pl.INT32],
                cond: pl.Scalar[pl.BOOL],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                with pl.manual_scope():
                    produced, _producer_tid = pl.submit(self.fill, scratch)
                    dynamic = pl.tensor.gather(src, -1, index)
                    if cond:
                        selected = pl.yield_(produced)
                    else:
                        selected = pl.yield_(dynamic)
                    out, _ = pl.submit(self.consume, selected)
                return out

        out = _run_auto_deps(Prog)
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

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                scratch: pl.Tensor[[4, 8], pl.FP32],
                src: pl.Tensor[[4, 16], pl.FP32],
                index: pl.Tensor[[4, 8], pl.INT32],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                with pl.manual_scope():
                    produced, _producer_tid = pl.submit(self.fill, scratch)
                    dynamic = pl.tensor.gather(src, -1, index)
                    for _i, (selected_iter,) in pl.range(0, 4, init_values=(produced,)):
                        selected = pl.yield_(dynamic)
                    out, _ = pl.submit(self.consume, selected)
                return out

        out = _run_auto_deps(Prog)
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

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                left: pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 0, 256)],
                right: pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 128, 256)],
            ) -> pl.Tensor[[64], pl.FP32, pl.MemRef("shared_ddr", 128, 256)]:
                with pl.manual_scope():
                    _produced, producer_tid = pl.submit(self.fill, left)
                    out, _ = pl.submit(self.consume, right)
                return out

        out = _run_auto_deps(Prog)
        consume_call = _user_calls(out, "consume")[0]
        edges = _compiler_edges(consume_call)
        assert len(edges) == 1
        assert edges[0].name_hint == "producer_tid"

    def test_unencodable_manual_scope_hazard_falls_back_to_auto_scope(self):
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
                    produced = self.fill(scratch)
                    out, _ = pl.submit(self.consume, produced)
                return out

        out = _run_auto_deps(Prog)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is False
        consume_call = _user_calls(out, "consume")[0]
        assert _compiler_edges(consume_call) == []

    def test_dynamic_gather_result_falls_back_to_auto_scope(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[4, 8], pl.FP32]) -> pl.Tensor[[4, 8], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                src: pl.Tensor[[4, 16], pl.FP32],
                index: pl.Tensor[[4, 8], pl.INT32],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                with pl.manual_scope():
                    gathered = pl.tensor.gather(src, -1, index)
                    out, _ = pl.submit(self.consume, gathered)
                return out

        out = _run_auto_deps(Prog)
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

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    carried = scratch
                    for _i in pl.range(0, 4):
                        carried, _producer_tid = pl.submit(self.fill, carried)
                    out, _ = pl.submit(self.consume, carried)
                return out

        out = _run_auto_deps(Prog)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is False

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

        out = _run_auto_deps(Prog)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is True

        consume_call = _user_calls(out, "consume")[0]
        user_edges = list(consume_call.attrs.get("manual_dep_edges", []))
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
        user_edges = list(consume_call.attrs.get("manual_dep_edges", []))
        assert [edge.name_hint for edge in user_edges] == ["last_tid"]
        assert _compiler_edges(consume_call) == []

    def test_partial_user_deps_with_dynamic_hazard_still_falls_back(self):
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

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                scratch: pl.Tensor[[64], pl.FP32],
                other: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    carried = scratch
                    for _i in pl.range(0, 4):
                        carried, _producer_tid = pl.submit(self.fill, carried)
                    _unrelated, unrelated_tid = pl.submit(self.unrelated, other)
                    out, _ = pl.submit(self.consume, carried, deps=[unrelated_tid])
                return out

        out = _run_auto_deps(Prog)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is False

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

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                scratch: pl.Tensor[[64], pl.FP32],
                src: pl.Tensor[[4, 16], pl.FP32],
                index: pl.Tensor[[4, 8], pl.INT32],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                with pl.manual_scope():
                    produced, _producer_tid = pl.submit(self.fill, scratch)
                    _first, _ = pl.submit(self.consume, produced)
                    gathered = pl.tensor.gather(src, -1, index)
                    out, _ = pl.submit(self.consume, gathered)
                return out

        out = _run_auto_deps(Prog)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is False
        for call in _user_calls(out, "consume"):
            assert _compiler_edges(call) == []

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

            @pl.function(type=pl.FunctionType.Orchestration)
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
                with pl.manual_scope():
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

        out = _run_auto_deps(Prog)
        scopes = _runtime_scopes(out)
        assert len(scopes) == 1
        assert scopes[0].manual is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
