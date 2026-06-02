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
* a *dynamic-offset* Vec ``tile.slice`` consumed by ``tile.col_expand_mul`` /
  ``tile.col_expand_add`` is replaced by a ``tile.extract(target_memory=Vec)``
  (issue #1640 — avoids the lazy ``pto.textract`` materializing into the slice's
  source-aliasing buffer; a static-offset slice is left untouched because
  ``AllocateMemoryAddr`` folds it to ``base + off``, making the textract a safe
  identity copy).

The now-dead ``tile.slice`` is dropped. ``ir.assert_structural_equal`` with
auto-mapping compares After against a hand-written Expected, so intermediate
Var names may differ — only types and structure must match.

Coverage:
* offset folding — zero / nonzero-row / nonzero-col / chained slices;
* a slice consumed across a scope boundary (defined outside a pipelined loop,
  extracted inside it);
* a slice with multiple ``tile.extract`` consumers;
* a slice consumed directly by ``tile.matmul`` and ``tile.matmul_acc``;
* a dynamic-offset Vec slice consumed by ``tile.col_expand_mul`` /
  ``tile.col_expand_add`` (materialized), and static-offset slices — const
  ``[0,0]`` and a sub-window const ``[5,0]`` — into ``col_expand_mul`` (left
  untouched, since a const offset folds to ``base + off``);
* no-op cases — no Mat slice, and a Vec-resident slice (left untouched).
"""

import pypto.language as pl
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

    def test_const_zero_offset_vec_slice_into_col_expand_left_untouched(self):
        """A const-``[0,0]`` Vec ``tile.slice`` feeding ``tile.col_expand_mul`` is
        NOT the #1640 hazard: ``AllocateMemoryAddr`` folds the const offset into
        ``base + 0``, so the lazy ``pto.textract`` is a safe identity copy. The
        pass leaves the slice untouched."""

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

    def test_static_nonzero_offset_vec_slice_into_col_expand_left_untouched(self):
        """A *static non-zero* offset Vec ``tile.slice`` feeding
        ``tile.col_expand_mul`` is also NOT the #1640 hazard: ``AllocateMemoryAddr``
        folds the const offset into ``base + off``, so the lazy ``pto.textract``
        materializes the row into its own offset-correct address — an identity
        copy that leaves the source intact. Only a *dynamic* offset falls back to
        the bare base. The pass must leave this sub-window static slice untouched."""

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
