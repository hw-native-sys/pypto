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
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    produced, producer_tid = pl.submit(self.fill, scratch)
                    _unused, user_tid = pl.submit(self.unrelated, scratch)
                    out, _ = pl.submit(self.consume, produced, deps=[user_tid])
                return out

        out = _run_auto_deps(Prog)
        consume_call = _user_calls(out, "consume")[0]
        user_edges = list(consume_call.attrs.get("manual_dep_edges", []))
        compiler_edges = _compiler_edges(consume_call)
        assert [edge.name_hint for edge in user_edges] == ["user_tid"]
        assert [edge.name_hint for edge in compiler_edges] == ["producer_tid"]

    def test_disjoint_slices_are_conservatively_treated_as_overlap(self):
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
        edges = _compiler_edges(consume_call)
        assert len(edges) == 1
        assert edges[0].name_hint == "producer_tid"

    def test_prior_hazard_without_producer_task_id_reports_user_error(self):
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

        program = passes.derive_call_directions()(Prog)
        with pytest.raises(ValueError, match="requires a producer TaskId"):
            passes.auto_derive_task_dependencies()(program)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
