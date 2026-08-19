# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Before / After / Expected tests for the CanonicalizeTileSlice pass.

The pass lowers Mat-resident ``tile.slice`` into ``tile.extract``:

* a ``tile.slice`` consumed by ``tile.extract`` is folded away — the extract
  reads the slice's source directly, with the slice offset added into the
  extract index;
* a ``tile.slice`` consumed by a ``tile.matmul`` family operand is replaced by
  a ``tile.extract(target_memory=Left|Right)``;
* a Vec ``tile.slice`` consumed by a ``tile.col_expand_*`` op is replaced by a
  ``tile.extract(target_memory=Vec)`` whenever codegen's lazy ``pto.textract``
  materialization into the slice's own (source-aliasing) buffer would not be an
  identity copy — i.e. the offset is dynamic (issue #1640: the address falls back
  to the bare source base) or the window is not contiguous in the source (issue
  #2010: a column slice of a multi-row tile repacks strided -> dense on top of
  its own live source). An identity-copy slice — const offset AND a contiguous
  window (single row, or full source width) — is left untouched so it keeps
  sharing the source buffer.

The now-dead ``tile.slice`` is dropped. ``ir.assert_structural_equal`` with
auto-mapping compares After against a hand-written Expected, so intermediate
Var names may differ — only types and structure must match.

Coverage:
* offset folding — zero / nonzero-row / nonzero-col / chained slices;
* a slice consumed across a scope boundary (defined outside a pipelined loop,
  extracted inside it);
* a slice with multiple ``tile.extract`` consumers;
* a slice consumed directly by ``tile.matmul`` and ``tile.matmul_acc``;
* Vec slices into ``col_expand_mul`` / ``col_expand_add`` — materialized when
  hazardous (dynamic offset; static-offset column slice of a multi-row tile),
  left untouched when the textract is an identity copy (const ``[0,0]``
  full-shape; single-row ``[5,0]``; full-width multi-row ``[16,0]``);
* Vec operand alignment — materialization for unaligned or unprovable call
  operands in ``AssignStmt`` and ``EvalStmt``, plus plain aliases and loop
  carries; provably aligned static, constant-SSA, dynamic-row, and dynamic
  known-multiple slices stay zero-copy;
* no-op cases — no Mat slice, and safe Vec-resident slices left untouched.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import ir, passes


def _run_pass(program: ir.Program) -> ir.Program:
    return passes.canonicalize_tile_slice()(program)


class TestSliceIntoExtract:
    """A Mat tile.slice consumed by tile.extract is folded into the extract."""

    def test_zero_offset_slice_folded(self):
        """An offset-0 full-shape Mat ``tile.slice`` feeding ``tile.extract``
        is dropped; the extract reads the slice's source directly."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 256], pl.BF16],
                rhs: pl.Tensor[[256, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 256], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [256, 64], target_memory=pl.Mem.Mat
                )
                lhs_slice: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat] = pl.tile.slice(lhs_mat, [16, 256], [0, 0])
                rhs_slice: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.slice(rhs_mat, [256, 64], [0, 0])
                a: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    lhs_slice, 0, 0, shape=[16, 256], target_memory=pl.Mem.Left
                )
                b: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    rhs_slice, 0, 0, shape=[256, 64], target_memory=pl.Mem.Right
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a, b)
                out = pl.store(c, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 256], pl.BF16],
                rhs: pl.Tensor[[256, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 256], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [256, 64], target_memory=pl.Mem.Mat
                )
                a: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    lhs_mat, 0, 0, shape=[16, 256], target_memory=pl.Mem.Left
                )
                b: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    rhs_mat, 0, 0, shape=[256, 64], target_memory=pl.Mem.Right
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a, b)
                out = pl.store(c, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Expected)

    def test_nonzero_row_offset_folded_into_index(self):
        """A Mat ``tile.slice`` at row offset 16 is dropped; the offset is
        folded into the extract's row index."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[32, 256], pl.BF16],
                rhs: pl.Tensor[[256, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[32, 256], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [32, 256], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [256, 64], target_memory=pl.Mem.Mat
                )
                lhs_slice: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat] = pl.tile.slice(
                    lhs_mat, [16, 256], [16, 0]
                )
                a: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    lhs_slice, 0, 0, shape=[16, 256], target_memory=pl.Mem.Left
                )
                b: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    rhs_mat, 0, 0, shape=[256, 64], target_memory=pl.Mem.Right
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a, b)
                out = pl.store(c, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[32, 256], pl.BF16],
                rhs: pl.Tensor[[256, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[32, 256], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [32, 256], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [256, 64], target_memory=pl.Mem.Mat
                )
                a: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    lhs_mat, 16, 0, shape=[16, 256], target_memory=pl.Mem.Left
                )
                b: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    rhs_mat, 0, 0, shape=[256, 64], target_memory=pl.Mem.Right
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a, b)
                out = pl.store(c, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Expected)

    def test_nonzero_col_offset_folded_into_index(self):
        """A Mat ``tile.slice`` at column offset 256 folds into the extract's
        column index."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 512], pl.BF16],
                rhs: pl.Tensor[[256, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 512], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [256, 64], target_memory=pl.Mem.Mat
                )
                lhs_slice: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat] = pl.tile.slice(
                    lhs_mat, [16, 256], [0, 256]
                )
                a: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    lhs_slice, 0, 0, shape=[16, 256], target_memory=pl.Mem.Left
                )
                b: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    rhs_mat, 0, 0, shape=[256, 64], target_memory=pl.Mem.Right
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a, b)
                out = pl.store(c, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 512], pl.BF16],
                rhs: pl.Tensor[[256, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 512], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [256, 64], target_memory=pl.Mem.Mat
                )
                a: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    lhs_mat, 0, 256, shape=[16, 256], target_memory=pl.Mem.Left
                )
                b: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    rhs_mat, 0, 0, shape=[256, 64], target_memory=pl.Mem.Right
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a, b)
                out = pl.store(c, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Expected)

    def test_chained_slices_peeled(self):
        """A slice of a slice is peeled to the root Mat tile; the two offsets
        accumulate into the extract index (8 + 4 = 12)."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[32, 256], pl.BF16],
                rhs: pl.Tensor[[256, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[32, 256], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [32, 256], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [256, 64], target_memory=pl.Mem.Mat
                )
                s1: pl.Tile[[24, 256], pl.BF16, pl.Mem.Mat] = pl.tile.slice(lhs_mat, [24, 256], [8, 0])
                s2: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat] = pl.tile.slice(s1, [16, 256], [4, 0])
                a: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    s2, 0, 0, shape=[16, 256], target_memory=pl.Mem.Left
                )
                b: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    rhs_mat, 0, 0, shape=[256, 64], target_memory=pl.Mem.Right
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a, b)
                out = pl.store(c, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[32, 256], pl.BF16],
                rhs: pl.Tensor[[256, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[32, 256], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [32, 256], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [256, 64], target_memory=pl.Mem.Mat
                )
                a: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    lhs_mat, 12, 0, shape=[16, 256], target_memory=pl.Mem.Left
                )
                b: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    rhs_mat, 0, 0, shape=[256, 64], target_memory=pl.Mem.Right
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a, b)
                out = pl.store(c, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Expected)

    def test_simultaneous_row_and_col_offset_folded(self):
        """A Mat ``tile.slice`` offset at both row 8 and col 128 folds *both*
        offsets into the extract indices (doc lines 31-32 / pass lines 205-206:
        ``extract(slice(src, _, [or, oc]), ir, ic) -> extract(src, ir+or, ic+oc)``).
        With ``ir == ic == 0`` constant-folding leaves the bare offsets 8 / 128."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[32, 512], pl.BF16],
                rhs: pl.Tensor[[256, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[32, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [32, 512], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [256, 64], target_memory=pl.Mem.Mat
                )
                lhs_slice: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat] = pl.tile.slice(
                    lhs_mat, [16, 256], [8, 128]
                )
                a: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    lhs_slice, 0, 0, shape=[16, 256], target_memory=pl.Mem.Left
                )
                b: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    rhs_mat, 0, 0, shape=[256, 64], target_memory=pl.Mem.Right
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a, b)
                out = pl.store(c, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[32, 512], pl.BF16],
                rhs: pl.Tensor[[256, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[32, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [32, 512], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [256, 64], target_memory=pl.Mem.Mat
                )
                a: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    lhs_mat, 8, 128, shape=[16, 256], target_memory=pl.Mem.Left
                )
                b: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    rhs_mat, 0, 0, shape=[256, 64], target_memory=pl.Mem.Right
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a, b)
                out = pl.store(c, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Expected)

    def test_symbolic_extract_index_with_const_offset_folded(self):
        """When the consumer ``tile.extract`` index is symbolic (loop var ``ko``)
        and the Mat slice carries a non-zero *constant* column offset 256, the
        offsets cannot constant-fold: ``MakeCanonicalIndexAdd`` falls through to
        the symbolic ``MakeAdd`` path (pass lines 84-92), so the extract column
        index becomes ``ko + 256`` reading the loaded Mat tile directly."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 1024], pl.BF16],
                rhs: pl.Tensor[[512, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 1024], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 1024], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [512, 64], target_memory=pl.Mem.Mat
                )
                # Mat slice into the right half of lhs_mat (col offset 256).
                lhs_slice: pl.Tile[[16, 512], pl.BF16, pl.Mem.Mat] = pl.tile.slice(
                    lhs_mat, [16, 512], [0, 256]
                )
                c_init: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.create(
                    [16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc
                )
                for ko, (c_iter,) in pl.pipeline(0, 512, 256, init_values=(c_init,), stage=2):
                    a: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                        lhs_slice, 0, ko, shape=[16, 256], target_memory=pl.Mem.Left
                    )
                    b: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        rhs_mat, ko, 0, shape=[256, 64], target_memory=pl.Mem.Right
                    )
                    cc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(c_iter, a, b)
                    c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(cc)
                out = pl.store(c, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 1024], pl.BF16],
                rhs: pl.Tensor[[512, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 1024], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 1024], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [512, 64], target_memory=pl.Mem.Mat
                )
                c_init: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.create(
                    [16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc
                )
                for ko, (c_iter,) in pl.pipeline(0, 512, 256, init_values=(c_init,), stage=2):
                    a: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                        lhs_mat, 0, ko + 256, shape=[16, 256], target_memory=pl.Mem.Left
                    )
                    b: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        rhs_mat, ko, 0, shape=[256, 64], target_memory=pl.Mem.Right
                    )
                    cc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(c_iter, a, b)
                    c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(cc)
                out = pl.store(c, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Expected)

    def test_slice_consumed_inside_pipelined_loop(self):
        """A slice defined in the function body, extracted inside a nested
        pipelined-loop body — exercises the function-wide collector and the
        recursive consumer rewrite across the scope boundary."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 512], pl.BF16],
                rhs: pl.Tensor[[512, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 512], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [512, 64], target_memory=pl.Mem.Mat
                )
                lhs_slice: pl.Tile[[16, 512], pl.BF16, pl.Mem.Mat] = pl.tile.slice(lhs_mat, [16, 512], [0, 0])
                rhs_slice: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat] = pl.tile.slice(rhs_mat, [512, 64], [0, 0])
                c_init: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.create(
                    [16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc
                )
                for ko, (c_iter,) in pl.pipeline(0, 512, 256, init_values=(c_init,), stage=2):
                    a: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                        lhs_slice, 0, ko, shape=[16, 256], target_memory=pl.Mem.Left
                    )
                    b: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        rhs_slice, ko, 0, shape=[256, 64], target_memory=pl.Mem.Right
                    )
                    cc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(c_iter, a, b)
                    c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(cc)
                out = pl.store(c, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 512], pl.BF16],
                rhs: pl.Tensor[[512, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 512], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [512, 64], target_memory=pl.Mem.Mat
                )
                c_init: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.create(
                    [16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc
                )
                for ko, (c_iter,) in pl.pipeline(0, 512, 256, init_values=(c_init,), stage=2):
                    a: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                        lhs_mat, 0, ko, shape=[16, 256], target_memory=pl.Mem.Left
                    )
                    b: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                        rhs_mat, ko, 0, shape=[256, 64], target_memory=pl.Mem.Right
                    )
                    cc: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(c_iter, a, b)
                    c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.yield_(cc)
                out = pl.store(c, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Expected)

    def test_slice_with_multiple_extract_consumers(self):
        """One Mat ``tile.slice`` feeding two ``tile.extract`` ops: both
        extracts are folded and the slice is dropped once dead."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 512], pl.BF16],
                rhs: pl.Tensor[[512, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 512], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [512, 64], target_memory=pl.Mem.Mat
                )
                lhs_slice: pl.Tile[[16, 512], pl.BF16, pl.Mem.Mat] = pl.tile.slice(lhs_mat, [16, 512], [0, 0])
                rhs_slice: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat] = pl.tile.slice(rhs_mat, [512, 64], [0, 0])
                a0: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    lhs_slice, 0, 0, shape=[16, 256], target_memory=pl.Mem.Left
                )
                b0: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    rhs_slice, 0, 0, shape=[256, 64], target_memory=pl.Mem.Right
                )
                a1: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    lhs_slice, 0, 256, shape=[16, 256], target_memory=pl.Mem.Left
                )
                b1: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    rhs_slice, 256, 0, shape=[256, 64], target_memory=pl.Mem.Right
                )
                c0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a0, b0)
                c1: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(c0, a1, b1)
                out = pl.store(c1, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 512], pl.BF16],
                rhs: pl.Tensor[[512, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 512], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 512], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[512, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [512, 64], target_memory=pl.Mem.Mat
                )
                a0: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    lhs_mat, 0, 0, shape=[16, 256], target_memory=pl.Mem.Left
                )
                b0: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    rhs_mat, 0, 0, shape=[256, 64], target_memory=pl.Mem.Right
                )
                a1: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    lhs_mat, 0, 256, shape=[16, 256], target_memory=pl.Mem.Left
                )
                b1: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    rhs_mat, 256, 0, shape=[256, 64], target_memory=pl.Mem.Right
                )
                c0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a0, b0)
                c1: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(c0, a1, b1)
                out = pl.store(c1, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Expected)


class TestSliceIntoMatmul:
    """A Mat tile.slice consumed directly by a matmul operand becomes a
    Mat→Left/Right tile.extract."""

    def test_matmul_operands_become_left_right_extracts(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 256], pl.BF16],
                rhs: pl.Tensor[[256, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 256], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [256, 64], target_memory=pl.Mem.Mat
                )
                lhs_slice: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat] = pl.tile.slice(lhs_mat, [16, 256], [0, 0])
                rhs_slice: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.slice(rhs_mat, [256, 64], [0, 0])
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_slice, rhs_slice)
                out = pl.store(c, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 256], pl.BF16],
                rhs: pl.Tensor[[256, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 256], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [256, 64], target_memory=pl.Mem.Mat
                )
                lhs_left: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    lhs_mat, 0, 0, shape=[16, 256], target_memory=pl.Mem.Left
                )
                rhs_right: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    rhs_mat, 0, 0, shape=[256, 64], target_memory=pl.Mem.Right
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_left, rhs_right)
                out = pl.store(c, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Expected)

    def test_matmul_acc_operands_become_left_right_extracts(self):
        """``tile.matmul_acc`` operands lhs/rhs (indices 1, 2) are rewritten;
        the accumulator operand (index 0) is untouched."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 256], pl.BF16],
                rhs: pl.Tensor[[256, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 256], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [256, 64], target_memory=pl.Mem.Mat
                )
                acc0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.create(
                    [16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc
                )
                lhs_slice: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat] = pl.tile.slice(lhs_mat, [16, 256], [0, 0])
                rhs_slice: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.slice(rhs_mat, [256, 64], [0, 0])
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(acc0, lhs_slice, rhs_slice)
                out = pl.store(c, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 256], pl.BF16],
                rhs: pl.Tensor[[256, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 256], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [256, 64], target_memory=pl.Mem.Mat
                )
                acc0: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.create(
                    [16, 64], dtype=pl.FP32, target_memory=pl.Mem.Acc
                )
                lhs_left: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    lhs_mat, 0, 0, shape=[16, 256], target_memory=pl.Mem.Left
                )
                rhs_right: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    rhs_mat, 0, 0, shape=[256, 64], target_memory=pl.Mem.Right
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_acc(acc0, lhs_left, rhs_right)
                out = pl.store(c, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Expected)

    def test_matmul_bias_operands_become_left_right_extracts(self):
        """``tile.matmul_bias`` lhs/rhs (operand indices 0, 1 — pass lines
        219-220) Mat slices are rewritten to Left/Right extracts; the bias
        operand (index 2) is *not* in the rewrite set, so a plain Mat bias
        tile is carried through untouched."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 256], pl.BF16],
                rhs: pl.Tensor[[256, 64], pl.BF16],
                bias: pl.Tensor[[1, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 256], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [256, 64], target_memory=pl.Mem.Mat
                )
                bias_mat: pl.Tile[[1, 64], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    bias, [0, 0], [1, 64], target_memory=pl.Mem.Mat
                )
                lhs_slice: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat] = pl.tile.slice(lhs_mat, [16, 256], [0, 0])
                rhs_slice: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.slice(rhs_mat, [256, 64], [0, 0])
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_bias(
                    lhs_slice, rhs_slice, bias_mat
                )
                out = pl.store(c, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 256], pl.BF16],
                rhs: pl.Tensor[[256, 64], pl.BF16],
                bias: pl.Tensor[[1, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 256], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [256, 64], target_memory=pl.Mem.Mat
                )
                bias_mat: pl.Tile[[1, 64], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    bias, [0, 0], [1, 64], target_memory=pl.Mem.Mat
                )
                lhs_left: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    lhs_mat, 0, 0, shape=[16, 256], target_memory=pl.Mem.Left
                )
                rhs_right: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    rhs_mat, 0, 0, shape=[256, 64], target_memory=pl.Mem.Right
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul_bias(lhs_left, rhs_right, bias_mat)
                out = pl.store(c, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Expected)


class TestUnalignedVecSlice:
    """Vec slice operands are materialized when their inherited address is not
    provably 32-byte aligned (issue #1789)."""

    def test_unaligned_fp32_column_slice_into_muls_materialized(self):
        """Column 1 of an FP32 tile starts four bytes past the aligned source."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                local: pl.Tile[[16, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [0, 0], [16, 8], target_memory=pl.Mem.Vec
                )
                head: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.slice(local, [16, 1], [0, 1])
                scaled: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.muls(head, 0.5)
                out = pl.store(scaled, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                local: pl.Tile[[16, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [0, 0], [16, 8], target_memory=pl.Mem.Vec
                )
                head_ext: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.extract(
                    local, 0, 1, shape=[16, 1], target_memory=pl.Mem.Vec
                )
                scaled: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.muls(head_ext, 0.5)
                out = pl.store(scaled, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Expected)

    def test_aligned_fp32_column_slice_into_muls_left_untouched(self):
        """Column 8 of an FP32 tile starts at the next 32-byte boundary."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                local: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [0, 0], [16, 16], target_memory=pl.Mem.Vec
                )
                head: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.slice(local, [16, 1], [0, 8])
                scaled: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.muls(head, 0.5)
                out = pl.store(scaled, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Before)

    def test_dynamic_row_with_aligned_stride_left_untouched(self):
        """A dynamic row is safe when every source row begins 32-byte aligned."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[32, 8], pl.FP32],
                row_off: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[1, 1], pl.FP32]],
            ) -> pl.Tensor[[1, 1], pl.FP32]:
                local: pl.Tile[[32, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [0, 0], [32, 8], target_memory=pl.Mem.Vec
                )
                item: pl.Tile[[1, 1], pl.FP32, pl.Mem.Vec] = pl.tile.slice(local, [1, 1], [row_off, 0])
                scaled: pl.Tile[[1, 1], pl.FP32, pl.Mem.Vec] = pl.tile.muls(item, 0.5)
                out = pl.store(scaled, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Before)

    def test_aligned_row_offset_through_constant_ssa_left_untouched(self):
        """A ConstInt stored in an SSA scalar remains a static offset.

        For a ``[32, 1]`` FP32 tile, row 16 begins 64 bytes from the aligned
        base. This mirrors the DeepSeek qk/pv store pattern that previously
        received an unnecessary Vec-to-Vec extract.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[32, 1], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                local: pl.Tile[[32, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [0, 0], [32, 1], target_memory=pl.Mem.Vec
                )
                row_off: pl.Scalar[pl.INDEX] = 16
                tail: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.slice(local, [16, 1], [row_off, 0])
                out = pl.store(tail, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Before)

    def test_unaligned_row_offset_through_constant_ssa_materialized(self):
        """Constant propagation must still materialize a genuinely unaligned
        SSA row offset."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[17, 1], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                local: pl.Tile[[17, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [0, 0], [17, 1], target_memory=pl.Mem.Vec
                )
                row_off: pl.Scalar[pl.INDEX] = 1
                tail: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.slice(local, [16, 1], [row_off, 0])
                scaled: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.muls(tail, 0.5)
                out = pl.store(scaled, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[17, 1], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                local: pl.Tile[[17, 1], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [0, 0], [17, 1], target_memory=pl.Mem.Vec
                )
                row_off: pl.Scalar[pl.INDEX] = 1  # noqa: F841
                tail_ext: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.extract(
                    local, 1, 0, shape=[16, 1], target_memory=pl.Mem.Vec
                )
                scaled: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.muls(tail_ext, 0.5)
                out = pl.store(scaled, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Expected)

    def test_dynamic_column_with_aligned_ssa_multiple_left_untouched(self):
        """A dynamic INT32 column known to be a multiple of 32 is aligned.

        This mirrors ``prefill_csa/rope_cs``. Materializing this store would
        also be illegal on A2/A3 because TEXTRACT does not support INT32.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16, 64], pl.INT32],
                out: pl.Out[pl.Tensor[[16, 64], pl.INT32]],
                block_idx: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[16, 64], pl.INT32]:
                local: pl.Tile[[16, 64], pl.INT32, pl.Mem.Vec] = pl.tile.load(
                    x, [0, 0], [16, 64], target_memory=pl.Mem.Vec
                )
                block_base: pl.Scalar[pl.INDEX] = block_idx * 16
                col_off: pl.Scalar[pl.INDEX] = block_base * 2
                half: pl.Tile[[16, 32], pl.INT32, pl.Mem.Vec] = pl.tile.slice(local, [16, 32], [0, col_off])
                out = pl.store(half, [0, col_off], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Before)

    def test_dynamic_column_into_muls_materialized(self):
        """A dynamic column cannot be proved aligned and is materialized."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16, 8], pl.FP32],
                col_off: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                local: pl.Tile[[16, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [0, 0], [16, 8], target_memory=pl.Mem.Vec
                )
                head: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.slice(local, [16, 1], [0, col_off])
                scaled: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.muls(head, 0.5)
                out = pl.store(scaled, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16, 8], pl.FP32],
                col_off: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                local: pl.Tile[[16, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [0, 0], [16, 8], target_memory=pl.Mem.Vec
                )
                head_ext: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.extract(
                    local, 0, col_off, shape=[16, 1], target_memory=pl.Mem.Vec
                )
                scaled: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.muls(head_ext, 0.5)
                out = pl.store(scaled, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Expected)

    def test_unaligned_slice_in_eval_stmt_materialized(self):
        """An EvalStmt call such as remote_store must not receive an unaligned
        slice subview directly."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16, 8], pl.FP32],
                target: pld.DistributedTensor[[16, 1], pl.FP32],
                peer: pl.Scalar[pl.INT32],
            ):
                local: pl.Tile[[16, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [0, 0], [16, 8], target_memory=pl.Mem.Vec
                )
                head: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.slice(local, [16, 1], [0, 1])
                pld.tile.remote_store(head, target=target, peer=peer, offsets=[0, 0])

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16, 8], pl.FP32],
                target: pld.DistributedTensor[[16, 1], pl.FP32],
                peer: pl.Scalar[pl.INT32],
            ):
                local: pl.Tile[[16, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [0, 0], [16, 8], target_memory=pl.Mem.Vec
                )
                head_ext: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.extract(
                    local, 0, 1, shape=[16, 1], target_memory=pl.Mem.Vec
                )
                pld.tile.remote_store(head_ext, target=target, peer=peer, offsets=[0, 0])

        ir.assert_structural_equal(_run_pass(Before), Expected)

    def test_unaligned_slice_through_plain_alias_materialized(self):
        """A plain SSA alias must not hide an unaligned slice from the pass."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                local: pl.Tile[[16, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [0, 0], [16, 8], target_memory=pl.Mem.Vec
                )
                head: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.slice(local, [16, 1], [0, 1])
                head_alias: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = head
                scaled: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.muls(head_alias, 0.5)
                out = pl.store(scaled, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                local: pl.Tile[[16, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [0, 0], [16, 8], target_memory=pl.Mem.Vec
                )
                head_alias: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.extract(
                    local, 0, 1, shape=[16, 1], target_memory=pl.Mem.Vec
                )
                scaled: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.muls(head_alias, 0.5)
                out = pl.store(scaled, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Expected)

    def test_unaligned_slice_loop_initializer_materialized(self):
        """A slice carried through an IterArg is aligned before it enters the
        loop, so every iteration sees a safe buffer."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                local: pl.Tile[[16, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [0, 0], [16, 8], target_memory=pl.Mem.Vec
                )
                head: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.slice(local, [16, 1], [0, 1])
                for _i, (carried,) in pl.range(1, init_values=(head,)):
                    scaled: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.muls(carried, 0.5)
                    result: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.yield_(scaled)
                out = pl.store(result, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                local: pl.Tile[[16, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [0, 0], [16, 8], target_memory=pl.Mem.Vec
                )
                head_ext: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.extract(
                    local, 0, 1, shape=[16, 1], target_memory=pl.Mem.Vec
                )
                for _i, (carried,) in pl.range(1, init_values=(head_ext,)):
                    scaled: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.muls(carried, 0.5)
                    result: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.yield_(scaled)
                out = pl.store(result, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Expected)

    def test_unaligned_slice_yield_materialized(self):
        """A newly created unaligned slice is materialized before a loop yield
        carries it to the next iteration."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                local: pl.Tile[[16, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [0, 0], [16, 8], target_memory=pl.Mem.Vec
                )
                seed: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.create(
                    [16, 1], dtype=pl.FP32, target_memory=pl.Mem.Vec
                )
                for _i, (carried,) in pl.range(1, init_values=(seed,)):
                    head: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.slice(local, [16, 1], [0, 1])
                    result: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.yield_(head)
                out = pl.store(result, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16, 8], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
            ) -> pl.Tensor[[16, 1], pl.FP32]:
                local: pl.Tile[[16, 8], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [0, 0], [16, 8], target_memory=pl.Mem.Vec
                )
                seed: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.create(
                    [16, 1], dtype=pl.FP32, target_memory=pl.Mem.Vec
                )
                for _i, (carried,) in pl.range(1, init_values=(seed,)):
                    head_ext: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.tile.extract(
                        local, 0, 1, shape=[16, 1], target_memory=pl.Mem.Vec
                    )
                    result: pl.Tile[[16, 1], pl.FP32, pl.Mem.Vec] = pl.yield_(head_ext)
                out = pl.store(result, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Expected)


class TestSliceIntoColExpand:
    """A Vec tile.slice consumed by tile.col_expand_mul / tile.col_expand_add is
    materialized through a fresh tile.extract (issue #1640) so the lazy
    pto.textract no longer writes into the slice's (source-aliasing) result
    buffer."""

    def test_dynamic_offset_vec_slice_into_col_expand_mul_materialized(self):
        """A dynamic-offset Vec ``tile.slice`` feeding ``tile.col_expand_mul`` is
        replaced by a fresh ``tile.extract(target_memory=Vec)`` and dropped."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                scores: pl.Tensor[[16, 256], pl.FP32],
                gamma: pl.Tensor[[1, 256], pl.FP32],
                row_off: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[1, 256], pl.FP32]],
            ) -> pl.Tensor[[1, 256], pl.FP32]:
                local: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    scores, [0, 0], [16, 256], target_memory=pl.Mem.Vec
                )
                gamma_t: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    gamma, [0, 0], [1, 256], target_memory=pl.Mem.Vec
                )
                row: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.slice(local, [1, 256], [row_off, 0])
                scaled: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(row, gamma_t)
                out = pl.store(scaled, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                scores: pl.Tensor[[16, 256], pl.FP32],
                gamma: pl.Tensor[[1, 256], pl.FP32],
                row_off: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[1, 256], pl.FP32]],
            ) -> pl.Tensor[[1, 256], pl.FP32]:
                local: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    scores, [0, 0], [16, 256], target_memory=pl.Mem.Vec
                )
                gamma_t: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    gamma, [0, 0], [1, 256], target_memory=pl.Mem.Vec
                )
                row_ext: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.extract(
                    local, row_off, 0, shape=[1, 256], target_memory=pl.Mem.Vec
                )
                scaled: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(row_ext, gamma_t)
                out = pl.store(scaled, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Expected)

    def test_dynamic_offset_vec_slice_into_col_expand_add_materialized(self):
        """``tile.col_expand_add`` shares the lazy ``pto.textract`` materialization
        with ``col_expand_mul``, so a dynamic-offset Vec slice operand is likewise
        replaced by a fresh ``tile.extract(target_memory=Vec)`` and dropped."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                scores: pl.Tensor[[16, 256], pl.FP32],
                gamma: pl.Tensor[[1, 256], pl.FP32],
                row_off: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[1, 256], pl.FP32]],
            ) -> pl.Tensor[[1, 256], pl.FP32]:
                local: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    scores, [0, 0], [16, 256], target_memory=pl.Mem.Vec
                )
                gamma_t: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    gamma, [0, 0], [1, 256], target_memory=pl.Mem.Vec
                )
                row: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.slice(local, [1, 256], [row_off, 0])
                scaled: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_add(row, gamma_t)
                out = pl.store(scaled, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                scores: pl.Tensor[[16, 256], pl.FP32],
                gamma: pl.Tensor[[1, 256], pl.FP32],
                row_off: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[1, 256], pl.FP32]],
            ) -> pl.Tensor[[1, 256], pl.FP32]:
                local: pl.Tile[[16, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    scores, [0, 0], [16, 256], target_memory=pl.Mem.Vec
                )
                gamma_t: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    gamma, [0, 0], [1, 256], target_memory=pl.Mem.Vec
                )
                row_ext: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.extract(
                    local, row_off, 0, shape=[1, 256], target_memory=pl.Mem.Vec
                )
                scaled: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_add(row_ext, gamma_t)
                out = pl.store(scaled, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Expected)

    def test_static_offset_column_slice_of_multirow_tile_materialized(self):
        """Regression for #2010: a *static*-offset COLUMN slice of a multi-row Vec
        tile feeding ``tile.col_expand_mul`` is a hazard even though the offset is
        const. Its own buffer is dense (row pitch 64) but aliases the source (row
        pitch 128), so the lazy ``pto.textract`` would repack strided -> dense on
        top of its own live source. The window is not contiguous (16 rows, 64 of
        the source's 128 columns), so the pass materializes it through a fresh
        ``tile.extract(target_memory=Vec)`` and drops the slice."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16, 128], pl.FP32],
                gamma: pl.Tensor[[1, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                local: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [0, 0], [16, 128], target_memory=pl.Mem.Vec
                )
                gamma_t: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    gamma, [0, 0], [1, 64], target_memory=pl.Mem.Vec
                )
                hi: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(local, [16, 64], [0, 64])
                scaled: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(hi, gamma_t)
                out = pl.store(scaled, [0, 0], out)
                return out

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16, 128], pl.FP32],
                gamma: pl.Tensor[[1, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                local: pl.Tile[[16, 128], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [0, 0], [16, 128], target_memory=pl.Mem.Vec
                )
                gamma_t: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    gamma, [0, 0], [1, 64], target_memory=pl.Mem.Vec
                )
                hi_ext: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.extract(
                    local, 0, 64, shape=[16, 64], target_memory=pl.Mem.Vec
                )
                scaled: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(hi_ext, gamma_t)
                out = pl.store(scaled, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Expected)

    def test_static_row_slice_full_width_left_untouched(self):
        """A static-offset multi-row slice spanning *every* column of its source is
        contiguous in the source buffer: the slice's dense row pitch equals the
        source's, so the lazy ``pto.textract`` is an identity copy that leaves the
        source intact. The pass leaves it untouched (no duplicate buffer). This is
        the boundary case against #2010's column slice."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                scores: pl.Tensor[[32, 64], pl.FP32],
                gamma: pl.Tensor[[1, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                local: pl.Tile[[32, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    scores, [0, 0], [32, 64], target_memory=pl.Mem.Vec
                )
                gamma_t: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    gamma, [0, 0], [1, 64], target_memory=pl.Mem.Vec
                )
                rows: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(local, [16, 64], [16, 0])
                scaled: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(rows, gamma_t)
                out = pl.store(scaled, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Before)

    def test_const_zero_offset_vec_slice_into_col_expand_left_untouched(self):
        """A const-``[0,0]`` full-shape Vec ``tile.slice`` feeding
        ``tile.col_expand_mul`` is not a hazard: the offset is const (so
        ``AllocateMemoryAddr`` folds it into ``base + 0``) and the window covers the
        whole source (so the dense result pitch equals the source's). The lazy
        ``pto.textract`` is a safe identity copy and the pass leaves the slice
        untouched."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                scores: pl.Tensor[[16, 64], pl.FP32],
                gamma: pl.Tensor[[1, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                local: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    scores, [0, 0], [16, 64], target_memory=pl.Mem.Vec
                )
                gamma_t: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    gamma, [0, 0], [1, 64], target_memory=pl.Mem.Vec
                )
                full: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(local, [16, 64], [0, 0])
                scaled: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(full, gamma_t)
                out = pl.store(scaled, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Before)

    def test_static_nonzero_offset_single_row_slice_left_untouched(self):
        """A *static non-zero* offset **single-row** Vec ``tile.slice`` feeding
        ``tile.col_expand_mul`` is not a hazard either: the const offset folds into
        ``base + off``, and a one-row window is contiguous whatever its width, so
        the lazy ``pto.textract`` materializes the row into its own offset-correct
        address — an identity copy that leaves the source intact. The pass must
        leave this sub-window static slice untouched."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                scores: pl.Tensor[[16, 64], pl.FP32],
                gamma: pl.Tensor[[1, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                local: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    scores, [0, 0], [16, 64], target_memory=pl.Mem.Vec
                )
                gamma_t: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    gamma, [0, 0], [1, 64], target_memory=pl.Mem.Vec
                )
                row: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(local, [1, 64], [5, 0])
                scaled: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(row, gamma_t)
                out = pl.store(scaled, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Before)


class TestNoOp:
    """Cases the pass must leave untouched."""

    def test_program_without_mat_slice_unchanged(self):
        """A matmul kernel with no ``tile.slice`` at all is returned as-is."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 256], pl.BF16],
                rhs: pl.Tensor[[256, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 256], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [256, 64], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lhs_mat, rhs_mat)
                out = pl.store(c, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Before)

    def test_vec_resident_slice_left_untouched(self):
        """A ``tile.slice`` whose result is Vec-resident (not Mat) is not a
        canonicalization target — the pass leaves it intact."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                x_vec: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(
                    x, [0, 0], [16, 64], target_memory=pl.Mem.Vec
                )
                x_slice: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(x_vec, [16, 64], [0, 0])
                out = pl.store(x_slice, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Before)

    def test_non_canonical_4arg_mat_slice_left_untouched(self):
        """A Mat ``tile.slice`` carrying a ``valid_shape`` is a 4-argument IR
        call — not a plain window. ``ParseCanonicalSlice`` rejects it
        (``if (call->args_.size() != 3) return nullopt``), so it is never
        collected and both the slice and its ``tile.extract`` consumer survive
        unchanged."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 256], pl.BF16],
                rhs: pl.Tensor[[256, 64], pl.BF16],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                lhs_mat: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    lhs, [0, 0], [16, 256], target_memory=pl.Mem.Mat
                )
                rhs_mat: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat] = pl.tile.load(
                    rhs, [0, 0], [256, 64], target_memory=pl.Mem.Mat
                )
                # 4-arg slice (carries valid_shape) — not a plain window.
                lhs_slice: pl.Tile[[16, 256], pl.BF16, pl.Mem.Mat] = pl.tile.slice(
                    lhs_mat, [16, 256], [0, 0], valid_shape=[16, 256]
                )
                a: pl.Tile[[16, 256], pl.BF16, pl.Mem.Left] = pl.tile.extract(
                    lhs_slice, 0, 0, shape=[16, 256], target_memory=pl.Mem.Left
                )
                b: pl.Tile[[256, 64], pl.BF16, pl.Mem.Right] = pl.tile.extract(
                    rhs_mat, 0, 0, shape=[256, 64], target_memory=pl.Mem.Right
                )
                c: pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(a, b)
                out = pl.store(c, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Before)

    def test_mat_slice_with_non_extract_non_matmul_consumer_left_untouched(self):
        """A canonical 3-arg Mat ``tile.slice`` consumed by ``tile.move``
        (Mat→Vec) — not by ``tile.extract`` or a matmul — survives the pass
        unchanged. The slice lowers to a valid ``pto.subview`` in codegen."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[32, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                x_mat: pl.Tile[[32, 64], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    x, [0, 0], [32, 64], target_memory=pl.Mem.Mat
                )
                x_slice: pl.Tile[[16, 64], pl.FP32, pl.Mem.Mat] = pl.tile.slice(x_mat, [16, 64], [16, 0])
                x_vec: pl.Tile[[16, 64], pl.FP32, pl.Mem.Vec] = pl.tile.move(
                    x_slice, target_memory=pl.Mem.Vec
                )
                out = pl.store(x_vec, [0, 0], out)
                return out

        ir.assert_structural_equal(_run_pass(Before), Before)


class TestAccAccumulatorSliceContiguity:
    """A matmul accumulator that is a strided window of an Acc (L0C) tile is
    rejected.

    L0C is NZ with block ``(r_b, c_b)`` of an ``[M, N]`` tile at
    ``(c_b * M/16 + r_b) * fractal``, so a window is contiguous only when it
    spans the parent's full row extent or occupies a single 16-column block.
    The MAD writes its destination compactly from a bare pointer and has no
    destination stride, so a strided window silently miscompiles — only the
    first 16 columns of each row tile land correctly.
    """

    def _kernel(self, acc_shape, slice_shape, offset, valid_shape=None):
        """`acc[offset]` of shape `slice_shape` accumulates `a[M, K] @ b[K, N]`,
        with M/N taken from `slice_shape` so the matmul's own shape check passes
        and the contiguity guard is what decides the outcome.

        `valid_shape` selects the 4-arg `tile.slice` form, which is not
        rewrite-eligible. The branch is resolved here in Python rather than in
        the kernel body: a `None` closure variable and an `is` comparison are
        both outside what the DSL parser accepts.
        """
        rows, cols = slice_shape
        k = 64

        if valid_shape is None:

            @pl.program
            class Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def kernel(
                    self,
                    x: pl.Tensor[[rows, k], pl.FP16],
                    w: pl.Tensor[[k, cols], pl.FP16],
                    out: pl.Out[pl.Tensor[acc_shape, pl.FP32]],
                ) -> pl.Tensor[acc_shape, pl.FP32]:
                    x_mat: pl.Tile[[rows, k], pl.FP16, pl.Mem.Mat] = pl.tile.load(
                        x, [0, 0], [rows, k], target_memory=pl.Mem.Mat
                    )
                    a: pl.Tile[[rows, k], pl.FP16, pl.Mem.Left] = pl.tile.move(
                        x_mat, target_memory=pl.Mem.Left
                    )
                    w_mat: pl.Tile[[k, cols], pl.FP16, pl.Mem.Mat] = pl.tile.load(
                        w, [0, 0], [k, cols], target_memory=pl.Mem.Mat
                    )
                    b: pl.Tile[[k, cols], pl.FP16, pl.Mem.Right] = pl.tile.move(
                        w_mat, target_memory=pl.Mem.Right
                    )
                    acc = pl.tile.create(acc_shape, pl.FP32, target_memory=pl.Mem.Acc)
                    acc_win = pl.tile.slice(acc, slice_shape, offset)
                    acc_new = pl.tile.matmul_acc(acc_win, a, b)
                    out = pl.tile.store(acc_new, [0, 0], out)
                    return out

        else:

            @pl.program
            class Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def kernel(
                    self,
                    x: pl.Tensor[[rows, k], pl.FP16],
                    w: pl.Tensor[[k, cols], pl.FP16],
                    out: pl.Out[pl.Tensor[acc_shape, pl.FP32]],
                ) -> pl.Tensor[acc_shape, pl.FP32]:
                    x_mat: pl.Tile[[rows, k], pl.FP16, pl.Mem.Mat] = pl.tile.load(
                        x, [0, 0], [rows, k], target_memory=pl.Mem.Mat
                    )
                    a: pl.Tile[[rows, k], pl.FP16, pl.Mem.Left] = pl.tile.move(
                        x_mat, target_memory=pl.Mem.Left
                    )
                    w_mat: pl.Tile[[k, cols], pl.FP16, pl.Mem.Mat] = pl.tile.load(
                        w, [0, 0], [k, cols], target_memory=pl.Mem.Mat
                    )
                    b: pl.Tile[[k, cols], pl.FP16, pl.Mem.Right] = pl.tile.move(
                        w_mat, target_memory=pl.Mem.Right
                    )
                    acc = pl.tile.create(acc_shape, pl.FP32, target_memory=pl.Mem.Acc)
                    acc_win = pl.tile.slice(acc, slice_shape, offset, valid_shape)
                    acc_new = pl.tile.matmul_acc(acc_win, a, b)
                    out = pl.tile.store(acc_new, [0, 0], out)
                    return out

        return Prog

    def test_row_window_of_multi_block_column_acc_rejected(self):
        """A [16, 32] row window of a [32, 32] Acc tile spans neither the full
        row extent (16 != 32) nor a single block column (32 > 16)."""
        prog = self._kernel([32, 32], [16, 32], [16, 0])
        with pytest.raises(ValueError, match="not contiguous in L0C's block layout"):
            _run_pass(prog)

    def test_full_row_extent_window_accepted(self):
        """A column window spans every row, so compact and parent strides
        coincide and the discarded geometry does not matter."""
        prog = self._kernel([16, 64], [16, 32], [0, 32])
        _run_pass(prog)

    def test_single_block_column_window_accepted(self):
        """A 16-column window occupies one block column, so there is no second
        block column to mis-stride."""
        prog = self._kernel([32, 16], [16, 16], [16, 0])
        _run_pass(prog)

    def test_narrow_window_straddling_two_blocks_rejected(self):
        """Width alone does not make a window safe. A [16, 16] window at column
        offset 8 covers columns 8-23, straddling two 16-column L0C blocks, so
        the MAD's compact write still mis-strides the second one."""
        prog = self._kernel([32, 32], [16, 16], [16, 8])
        with pytest.raises(ValueError, match="not contiguous in L0C's block layout"):
            _run_pass(prog)

    def test_row_window_with_explicit_valid_shape_rejected(self):
        """A slice carrying an explicit valid_shape is not rewrite-eligible, but
        it reaches the MAD with the same broken stride, so the guard must still
        see it."""
        prog = self._kernel([32, 32], [16, 32], [16, 0], valid_shape=[16, 32])
        with pytest.raises(ValueError, match="not contiguous in L0C's block layout"):
            _run_pass(prog)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
