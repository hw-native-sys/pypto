# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for FuseCreateAssembleToSlice pass."""

import pypto.language as pl
from pypto import ir, passes
from pypto.pypto_core import ir as ir_core


def _run_prereqs_only(program):
    """Run prerequisite passes without FuseCreateAssembleToSlice."""
    pipeline = passes.PassPipeline()
    pipeline.add_pass(passes.convert_to_ssa())
    pipeline.add_pass(passes.normalize_stmt_structure())
    pipeline.add_pass(passes.flatten_call_expr())
    pipeline.add_pass(passes.outline_hierarchy_scopes())
    pipeline.add_pass(passes.outline_incore_scopes())
    pipeline.add_pass(passes.outline_cluster_scopes())
    ctx = passes.PassContext([], passes.VerificationLevel.NONE)
    with ctx:
        return pipeline.run(program)


def _run_prereqs_and_fuse(program):
    """Run prerequisite passes then FuseCreateAssembleToSlice."""
    pipeline = passes.PassPipeline()
    pipeline.add_pass(passes.convert_to_ssa())
    pipeline.add_pass(passes.normalize_stmt_structure())
    pipeline.add_pass(passes.flatten_call_expr())
    pipeline.add_pass(passes.outline_hierarchy_scopes())
    pipeline.add_pass(passes.outline_incore_scopes())
    pipeline.add_pass(passes.outline_cluster_scopes())
    pipeline.add_pass(passes.fuse_create_assemble_to_slice())
    ctx = passes.PassContext([], passes.VerificationLevel.NONE)
    with ctx:
        return pipeline.run(program)


def _collect_tensor_ops_in_orch(program):
    """Collect sorted tensor op names from Orchestration functions."""

    class OpCollector(ir_core.IRVisitor):
        def __init__(self):
            super().__init__()
            self.ops = []

        def visit_assign_stmt(self, stmt):
            if hasattr(stmt.value, "op") and stmt.value.op.name.startswith("tensor."):
                self.ops.append(stmt.value.op.name)
            super().visit_assign_stmt(stmt)

    all_ops = []
    for func in program.functions.values():
        if func.func_type == ir_core.FunctionType.Orchestration:
            collector = OpCollector()
            collector.visit_stmt(func.body)
            all_ops.extend(collector.ops)
    return sorted(all_ops)


class TestFuseCreateAssembleToSlice:
    """Tests for the FuseCreateAssembleToSlice pass."""

    def test_basic_create_assemble_fused_to_slice(self):
        """tensor.create + single tensor.assemble → tensor.slice, assemble removed."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def fill_row(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                r: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[1, 8], pl.FP32]],
            ) -> pl.Tensor[[1, 8], pl.FP32]:
                row_tile: pl.Tile[[1, 8], pl.FP32] = pl.load(x, [r, 0], [1, 8])
                out_1: pl.Tensor[[1, 8], pl.FP32] = pl.store(row_tile, [0, 0], out)
                return out_1

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                for r in pl.range(4):
                    row: pl.Tensor[[1, 8], pl.FP32] = pl.create_tensor([1, 8], dtype=pl.FP32)
                    row = self.fill_row(x, r, row)
                    out = pl.assemble(out, row, [r, 0])
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def fill_row(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                r: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[1, 8], pl.FP32]],
            ) -> pl.Tensor[[1, 8], pl.FP32]:
                row_tile: pl.Tile[[1, 8], pl.FP32] = pl.load(x, [r, 0], [1, 8])
                out_1: pl.Tensor[[1, 8], pl.FP32] = pl.store(row_tile, [0, 0], out)
                return out_1

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                for r in pl.range(4):
                    row: pl.Tensor[[1, 8], pl.FP32] = pl.slice(out, [1, 8], [r, 0])
                    row = self.fill_row(x, r, row)
                return out

        after = _run_prereqs_and_fuse(Before)
        expected = _run_prereqs_only(Expected)

        after_ops = _collect_tensor_ops_in_orch(after)
        expected_ops = _collect_tensor_ops_in_orch(expected)
        assert after_ops == expected_ops

    def test_duplicate_assemble_not_fused(self):
        """tensor.create assembled more than once → no fusion, IR unchanged."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def fill_row(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                r: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[1, 8], pl.FP32]],
            ) -> pl.Tensor[[1, 8], pl.FP32]:
                row_tile: pl.Tile[[1, 8], pl.FP32] = pl.load(x, [r, 0], [1, 8])
                out_1: pl.Tensor[[1, 8], pl.FP32] = pl.store(row_tile, [0, 0], out)
                return out_1

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                zero: pl.Scalar[pl.INDEX] = 0
                row: pl.Tensor[[1, 8], pl.FP32] = pl.create_tensor([1, 8], dtype=pl.FP32)
                row = self.fill_row(x, zero, row)
                out = pl.assemble(out, row, [0, 0])
                out = pl.assemble(out, row, [1, 0])
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def fill_row(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                r: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[1, 8], pl.FP32]],
            ) -> pl.Tensor[[1, 8], pl.FP32]:
                row_tile: pl.Tile[[1, 8], pl.FP32] = pl.load(x, [r, 0], [1, 8])
                out_1: pl.Tensor[[1, 8], pl.FP32] = pl.store(row_tile, [0, 0], out)
                return out_1

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                zero: pl.Scalar[pl.INDEX] = 0
                row: pl.Tensor[[1, 8], pl.FP32] = pl.create_tensor([1, 8], dtype=pl.FP32)
                row = self.fill_row(x, zero, row)
                out = pl.assemble(out, row, [0, 0])
                out = pl.assemble(out, row, [1, 0])
                return out

        after = _run_prereqs_and_fuse(Before)
        expected = _run_prereqs_only(Expected)
        ir.assert_structural_equal(after, expected)

    def test_slice_source_not_fused(self):
        """tensor.assemble with a tensor.slice source → no fusion, IR unchanged."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                chunk: pl.Tensor[[1, 8], pl.FP32] = pl.slice(x, [1, 8], [0, 0])
                out = pl.assemble(out, chunk, [0, 0])
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                chunk: pl.Tensor[[1, 8], pl.FP32] = pl.slice(x, [1, 8], [0, 0])
                out = pl.assemble(out, chunk, [0, 0])
                return out

        after = _run_prereqs_and_fuse(Before)
        expected = _run_prereqs_only(Expected)
        ir.assert_structural_equal(after, expected)

    def test_no_orchestration_function_noop(self):
        """Pass should be a no-op when there are no Orchestration functions."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out: pl.Out[pl.Tensor[[16], pl.FP32]],
            ) -> pl.Tensor[[16], pl.FP32]:
                t: pl.Tile[[16], pl.FP32] = pl.load(x, [0], [16])
                out_1: pl.Tensor[[16], pl.FP32] = pl.store(t, [0], out)
                return out_1

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16], pl.FP32],
                out: pl.Out[pl.Tensor[[16], pl.FP32]],
            ) -> pl.Tensor[[16], pl.FP32]:
                t: pl.Tile[[16], pl.FP32] = pl.load(x, [0], [16])
                out_1: pl.Tensor[[16], pl.FP32] = pl.store(t, [0], out)
                return out_1

        after = _run_prereqs_and_fuse(Before)
        expected = _run_prereqs_only(Expected)
        ir.assert_structural_equal(after, expected)
