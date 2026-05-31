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
import pytest
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
        ir.assert_structural_equal(after, expected)

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

    def test_multi_iter_arg_partial_fuse(self):
        """Only the assembled iter_arg is stripped; other iter_args survive.

        Reproduces the decode-attention pattern where the outer for loop
        carries multiple iter_args (e.g. attn_out, cache) but only attn_out
        has a create+assemble pattern.  Before the fix, the pass produced
        ``auto attn_out = attn_out;`` in codegen (self-assignment) because
        it replaced assemble with an alias without cleaning up the iter_arg.
        """

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

            @pl.function(type=pl.FunctionType.InCore)
            def update_state(
                self,
                state: pl.Out[pl.Tensor[[4], pl.FP32]],
                r: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[4], pl.FP32]:
                t: pl.Tile[[4], pl.FP32] = pl.load(state, [0], [4])
                state_1: pl.Tensor[[4], pl.FP32] = pl.store(t, [0], state)
                return state_1

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                state: pl.Out[pl.Tensor[[4], pl.FP32]],
                out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
            ) -> tuple[pl.Tensor[[4], pl.FP32], pl.Tensor[[4, 8], pl.FP32]]:
                for r in pl.range(4):
                    state = self.update_state(state, r)
                    row: pl.Tensor[[1, 8], pl.FP32] = pl.create_tensor([1, 8], dtype=pl.FP32)
                    row = self.fill_row(x, r, row)
                    out = pl.assemble(out, row, [r, 0])
                return state, out

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

            @pl.function(type=pl.FunctionType.InCore)
            def update_state(
                self,
                state: pl.Out[pl.Tensor[[4], pl.FP32]],
                r: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[4], pl.FP32]:
                t: pl.Tile[[4], pl.FP32] = pl.load(state, [0], [4])
                state_1: pl.Tensor[[4], pl.FP32] = pl.store(t, [0], state)
                return state_1

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                state: pl.Out[pl.Tensor[[4], pl.FP32]],
                out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
            ) -> tuple[pl.Tensor[[4], pl.FP32], pl.Tensor[[4, 8], pl.FP32]]:
                for r in pl.range(4):
                    state = self.update_state(state, r)
                    row: pl.Tensor[[1, 8], pl.FP32] = pl.slice(out, [1, 8], [r, 0])
                    row = self.fill_row(x, r, row)
                return state, out

        after = _run_prereqs_and_fuse(Before)
        expected = _run_prereqs_only(Expected)
        ir.assert_structural_equal(after, expected)

    def test_3d_target_2d_tile_offset_padded(self):
        """2D create assembled into 3D target → slice shape padded with leading 1.

        Reproduces the prefill projection bug where a [TOK, CHUNK] tile is
        assembled into a [B, S, H] output at offset [b, p, q].  Before the
        fix the fused slice had shape=[TOK,CHUNK] (2D) but offset=[b,p,q]
        (3D), causing a rank mismatch in codegen.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def compute(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[2, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 4], pl.FP32]:
                t: pl.Tile[[2, 4], pl.FP32] = pl.load(x, [0, 0], [2, 4])
                out_1: pl.Tensor[[2, 4], pl.FP32] = pl.store(t, [0, 0], out)
                return out_1

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[2, 4, 8], pl.FP32]],
            ) -> pl.Tensor[[2, 4, 8], pl.FP32]:
                for b in pl.range(2):
                    for c in pl.range(2):
                        col = c * 4
                        chunk: pl.Tensor[[2, 4], pl.FP32] = pl.create_tensor([2, 4], dtype=pl.FP32)
                        chunk = self.compute(x, chunk)
                        out = pl.assemble(out, chunk, [b, 0, col])
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def compute(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[2, 4], pl.FP32]],
            ) -> pl.Tensor[[2, 4], pl.FP32]:
                t: pl.Tile[[2, 4], pl.FP32] = pl.load(x, [0, 0], [2, 4])
                out_1: pl.Tensor[[2, 4], pl.FP32] = pl.store(t, [0, 0], out)
                return out_1

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[2, 4, 8], pl.FP32]],
            ) -> pl.Tensor[[2, 4, 8], pl.FP32]:
                for b in pl.range(2):
                    for c in pl.range(2):
                        col = c * 4
                        chunk: pl.Tensor[[1, 2, 4], pl.FP32] = pl.slice(out, [1, 2, 4], [b, 0, col])
                        # Use a different name because the DSL rejects reassigning
                        # chunk (type [1,2,4]) with compute's return (type [2,4]).
                        # After SSA both programs produce distinct vars anyway, and
                        # name_hint_ is IgnoreField in structural equality.
                        _chunk_ret: pl.Tensor[[2, 4], pl.FP32] = self.compute(x, chunk)
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

    def test_inout_scratch_before_out_not_fused(self):
        """Issue #1564: an InOut scratch ordered before the Out must not be fused.

        ``compute`` takes an InOut scratch (its own ``create_tensor``, a
        different shape/dtype) *before* the Out param that the call actually
        returns. The return value aliases the Out, not the first Out/InOut in
        param order. Before the fix, the call result's buffer root resolved to
        the scratch, so the scratch's ``tensor.create`` was wrongly rewritten to
        ``tensor.slice(out, ...)`` — aliasing the scratch onto the output and
        corrupting it. Only ``row`` (the real returned output) should be fused.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def compute(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                scratch: pl.InOut[pl.Tensor[[2, 8], pl.FP32]],
                out: pl.Out[pl.Tensor[[1, 8], pl.FP32]],
            ) -> pl.Tensor[[1, 8], pl.FP32]:
                s_tile: pl.Tile[[2, 8], pl.FP32] = pl.load(x, [0, 0], [2, 8])
                scratch_1: pl.Tensor[[2, 8], pl.FP32] = pl.store(s_tile, [0, 0], scratch)
                row_tile: pl.Tile[[1, 8], pl.FP32] = pl.load(scratch_1, [0, 0], [1, 8])
                out_1: pl.Tensor[[1, 8], pl.FP32] = pl.store(row_tile, [0, 0], out)
                return out_1

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                for r in pl.range(4):
                    scratch: pl.Tensor[[2, 8], pl.FP32] = pl.create_tensor([2, 8], dtype=pl.FP32)
                    row: pl.Tensor[[1, 8], pl.FP32] = pl.create_tensor([1, 8], dtype=pl.FP32)
                    row = self.compute(x, scratch, row)
                    out = pl.assemble(out, row, [r, 0])
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def compute(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                scratch: pl.InOut[pl.Tensor[[2, 8], pl.FP32]],
                out: pl.Out[pl.Tensor[[1, 8], pl.FP32]],
            ) -> pl.Tensor[[1, 8], pl.FP32]:
                s_tile: pl.Tile[[2, 8], pl.FP32] = pl.load(x, [0, 0], [2, 8])
                scratch_1: pl.Tensor[[2, 8], pl.FP32] = pl.store(s_tile, [0, 0], scratch)
                row_tile: pl.Tile[[1, 8], pl.FP32] = pl.load(scratch_1, [0, 0], [1, 8])
                out_1: pl.Tensor[[1, 8], pl.FP32] = pl.store(row_tile, [0, 0], out)
                return out_1

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                for r in pl.range(4):
                    scratch: pl.Tensor[[2, 8], pl.FP32] = pl.create_tensor([2, 8], dtype=pl.FP32)
                    row: pl.Tensor[[1, 8], pl.FP32] = pl.slice(out, [1, 8], [r, 0])
                    row = self.compute(x, scratch, row)
                return out

        after = _run_prereqs_and_fuse(Before)
        expected = _run_prereqs_only(Expected)
        ir.assert_structural_equal(after, expected)

    def test_ambiguous_return_root_skips_fusion(self):
        """When >1 output-direction param matches the return type, the buffer
        root is ambiguous, so the pass must NOT guess — it skips fusion rather
        than risk aliasing the scratch onto the output (PR #1570 review).

        Here the InOut scratch and the real Out share shape+dtype, so neither can
        be proven to be the return's buffer; the create + assemble stay untouched.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def compute(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                scratch: pl.InOut[pl.Tensor[[1, 8], pl.FP32]],
                out: pl.Out[pl.Tensor[[1, 8], pl.FP32]],
            ) -> pl.Tensor[[1, 8], pl.FP32]:
                s_tile: pl.Tile[[1, 8], pl.FP32] = pl.load(x, [0, 0], [1, 8])
                scratch_1: pl.Tensor[[1, 8], pl.FP32] = pl.store(s_tile, [0, 0], scratch)
                row_tile: pl.Tile[[1, 8], pl.FP32] = pl.load(scratch_1, [0, 0], [1, 8])
                out_1: pl.Tensor[[1, 8], pl.FP32] = pl.store(row_tile, [0, 0], out)
                return out_1

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                for r in pl.range(4):
                    scratch: pl.Tensor[[1, 8], pl.FP32] = pl.create_tensor([1, 8], dtype=pl.FP32)
                    row: pl.Tensor[[1, 8], pl.FP32] = pl.create_tensor([1, 8], dtype=pl.FP32)
                    row = self.compute(x, scratch, row)
                    out = pl.assemble(out, row, [r, 0])
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def compute(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                scratch: pl.InOut[pl.Tensor[[1, 8], pl.FP32]],
                out: pl.Out[pl.Tensor[[1, 8], pl.FP32]],
            ) -> pl.Tensor[[1, 8], pl.FP32]:
                s_tile: pl.Tile[[1, 8], pl.FP32] = pl.load(x, [0, 0], [1, 8])
                scratch_1: pl.Tensor[[1, 8], pl.FP32] = pl.store(s_tile, [0, 0], scratch)
                row_tile: pl.Tile[[1, 8], pl.FP32] = pl.load(scratch_1, [0, 0], [1, 8])
                out_1: pl.Tensor[[1, 8], pl.FP32] = pl.store(row_tile, [0, 0], out)
                return out_1

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                for r in pl.range(4):
                    scratch: pl.Tensor[[1, 8], pl.FP32] = pl.create_tensor([1, 8], dtype=pl.FP32)
                    row: pl.Tensor[[1, 8], pl.FP32] = pl.create_tensor([1, 8], dtype=pl.FP32)
                    row = self.compute(x, scratch, row)
                    out = pl.assemble(out, row, [r, 0])
                return out

        after = _run_prereqs_and_fuse(Before)
        expected = _run_prereqs_only(Expected)
        ir.assert_structural_equal(after, expected)

    def test_atomic_assemble_not_fused(self):
        """tensor.assemble with atomic=Add → not fused; the atomic assemble must survive.

        Fusing an atomic-add assemble into a tensor.slice would silently drop the
        atomic combine mode, degrading split-K accumulation to a plain overwrite.
        """

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
                    out = pl.assemble(out, row, [r, 0], atomic=pl.AtomicType.Add)
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
                    row: pl.Tensor[[1, 8], pl.FP32] = pl.create_tensor([1, 8], dtype=pl.FP32)
                    row = self.fill_row(x, r, row)
                    out = pl.assemble(out, row, [r, 0], atomic=pl.AtomicType.Add)
                return out

        after = _run_prereqs_and_fuse(Before)
        expected = _run_prereqs_only(Expected)
        ir.assert_structural_equal(after, expected)

    def test_while_loop_create_assemble_fused_to_slice(self):
        """create + single assemble inside a ``while`` loop → slice; pass-through while iter arg stripped.

        Mirrors the basic for-loop case but exercises the WhileStmt path
        (``BufferRootCollector::VisitStmt_(WhileStmtPtr)`` threads roots through
        while iter args, and ``StripPassThroughWhileIterArgs`` drops the iter arg
        once the assemble that produced its yielded value is eliminated). The
        loop also carries a real scalar counter ``i`` that must survive.

        ConvertToSSA turns the natural ``while`` into an SSA while whose iter
        args are ``i`` (counter, real loop-carried state) and ``out`` (assembled
        buffer). After fusion the ``out`` create→assemble pattern is replaced by
        a ``tensor.slice(out, ...)`` of the function param ``out``, the assemble
        is dropped, and the now pass-through ``out`` iter arg is stripped, while
        ``i`` survives.
        """

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
                i: pl.Scalar[pl.INDEX] = 0
                while i < 4:
                    row: pl.Tensor[[1, 8], pl.FP32] = pl.create_tensor([1, 8], dtype=pl.FP32)
                    row = self.fill_row(x, i, row)
                    out = pl.assemble(out, row, [i, 0])
                    i = i + 1
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
                i: pl.Scalar[pl.INDEX] = 0
                while i < 4:
                    row: pl.Tensor[[1, 8], pl.FP32] = pl.slice(out, [1, 8], [i, 0])
                    row = self.fill_row(x, i, row)
                    i = i + 1
                return out

        after = _run_prereqs_and_fuse(Before)
        expected = _run_prereqs_only(Expected)
        ir.assert_structural_equal(after, expected)

    def test_tuple_return_callee_root_tracked_and_fused(self):
        """create + tuple-returning callee + single assemble → slice (tuple-root tracking).

        Exercises the tuple-output-root path of ``BufferRootCollector``: the
        callee returns ``Tuple[<stats>, <row>]`` where the 2nd element aliases
        its ``Out`` param (the created ``row`` buffer). The collector records the
        per-element roots in ``tuple_output_roots_`` for the call result, then
        resolves ``row = call_result[1]`` (a ``TupleGetItemExpr``) back to the
        create root. The subsequent single ``pl.assemble(out, row, ...)`` is thus
        recognised as fusible: the create becomes a ``tensor.slice(out, ...)`` and
        the assemble is dropped (pass doc, algorithm phase 1: "tracks tuple roots
        for tuple-returning calls via ``tuple_output_roots_`` and resolves
        ``TupleGetItemExpr`` from those call results").

        ``stats`` is a separately-created scratch (different shape) that is NOT
        assembled, so it stays a plain ``tensor.create``.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def fill_row(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                r: pl.Scalar[pl.INDEX],
                stats: pl.Out[pl.Tensor[[1], pl.FP32]],
                out: pl.Out[pl.Tensor[[1, 8], pl.FP32]],
            ) -> tuple[pl.Tensor[[1], pl.FP32], pl.Tensor[[1, 8], pl.FP32]]:
                s_tile: pl.Tile[[1], pl.FP32] = pl.load(stats, [0], [1])
                stats_1: pl.Tensor[[1], pl.FP32] = pl.store(s_tile, [0], stats)
                row_tile: pl.Tile[[1, 8], pl.FP32] = pl.load(x, [r, 0], [1, 8])
                out_1: pl.Tensor[[1, 8], pl.FP32] = pl.store(row_tile, [0, 0], out)
                return stats_1, out_1

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                for r in pl.range(4):
                    stats: pl.Tensor[[1], pl.FP32] = pl.create_tensor([1], dtype=pl.FP32)
                    row: pl.Tensor[[1, 8], pl.FP32] = pl.create_tensor([1, 8], dtype=pl.FP32)
                    stats, row = self.fill_row(x, r, stats, row)
                    out = pl.assemble(out, row, [r, 0])
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def fill_row(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                r: pl.Scalar[pl.INDEX],
                stats: pl.Out[pl.Tensor[[1], pl.FP32]],
                out: pl.Out[pl.Tensor[[1, 8], pl.FP32]],
            ) -> tuple[pl.Tensor[[1], pl.FP32], pl.Tensor[[1, 8], pl.FP32]]:
                s_tile: pl.Tile[[1], pl.FP32] = pl.load(stats, [0], [1])
                stats_1: pl.Tensor[[1], pl.FP32] = pl.store(s_tile, [0], stats)
                row_tile: pl.Tile[[1, 8], pl.FP32] = pl.load(x, [r, 0], [1, 8])
                out_1: pl.Tensor[[1, 8], pl.FP32] = pl.store(row_tile, [0, 0], out)
                return stats_1, out_1

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                x: pl.Tensor[[4, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                for r in pl.range(4):
                    stats: pl.Tensor[[1], pl.FP32] = pl.create_tensor([1], dtype=pl.FP32)
                    row: pl.Tensor[[1, 8], pl.FP32] = pl.slice(out, [1, 8], [r, 0])
                    stats, row = self.fill_row(x, r, stats, row)
                return out

        after = _run_prereqs_and_fuse(Before)
        expected = _run_prereqs_only(Expected)
        ir.assert_structural_equal(after, expected)

    @pytest.mark.xfail(
        reason="suspected bug: BufferRootCollector uses As<Call> (exact kind), so a "
        "pl.submit (ir.Submit) callee never propagates its Out-param buffer root; the "
        "create+assemble around a submit is left unfused (violates pass-submit-awareness).",
        strict=False,
    )
    def test_submit_create_assemble_fused_to_slice(self):
        """create + pl.submit callee + single assemble inside manual_scope → slice.

        Semantically identical to ``test_basic_create_assemble_fused_to_slice``
        except the InCore call is launched with ``pl.submit`` inside
        ``pl.manual_scope``. ``pl.submit(self.fill_row, x, r, row)`` carries the
        created ``row`` as an Out-direction prefix arg, so its result aliases the
        ``row`` create root exactly as a plain call would.

        Per the pass's documented buffer-root analysis ("propagates roots through
        call output parameters whose direction is Out/InOut") and the project
        ``pass-submit-awareness`` rule (Submit is a sibling call-like kind that
        must be handled wherever Call is), the create should fuse to
        ``tensor.slice(out, ...)`` and the assemble should be dropped — exactly as
        in the plain-call case. The Expected below is the correct fused result.
        """

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
                with pl.manual_scope():
                    for r in pl.range(4):
                        row: pl.Tensor[[1, 8], pl.FP32] = pl.create_tensor([1, 8], dtype=pl.FP32)
                        row, _tid = pl.submit(self.fill_row, x, r, row)
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
                with pl.manual_scope():
                    for r in pl.range(4):
                        row: pl.Tensor[[1, 8], pl.FP32] = pl.slice(out, [1, 8], [r, 0])
                        row, _tid = pl.submit(self.fill_row, x, r, row)
                return out

        after = _run_prereqs_and_fuse(Before)
        expected = _run_prereqs_only(Expected)
        ir.assert_structural_equal(after, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
