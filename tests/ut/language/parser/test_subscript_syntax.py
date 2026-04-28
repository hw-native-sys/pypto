# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for subscript syntax on Tensor and Tile types."""

from typing import cast

import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.parser.diagnostics import ParserTypeError
from pypto.language.parser.diagnostics.exceptions import UnsupportedFeatureError


class TestTensorSubscript:
    """Tests for tensor subscript syntax: A[i, j], A[0:16, :]."""

    def test_tensor_read_via_subscript(self):
        """A[i, j] with all integer indices on Tensor -> tensor.read."""

        @pl.function
        def read_elem(
            A: pl.Tensor[[64, 128], pl.FP32],
            i: pl.Scalar[pl.INDEX],
            j: pl.Scalar[pl.INDEX],
        ) -> pl.Scalar[pl.FP32]:
            return A[i, j]

        assert isinstance(read_elem, ir.Function)
        printed = read_elem.as_python()
        assert "tensor.read" in printed

    def test_tensor_slice_via_subscript(self):
        """A[0:16, :] with slices on Tensor -> tensor.slice."""

        @pl.function
        def slice_tensor(
            A: pl.Tensor[[64, 128], pl.FP32],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            return A[0:16, :]

        assert isinstance(slice_tensor, ir.Function)
        printed = slice_tensor.as_python()
        assert "tensor.slice" in printed

    def test_tensor_slice_both_bounds(self):
        """A[0:16, 0:32] with explicit bounds -> tensor.slice with computed shapes."""

        @pl.function
        def slice_both(
            A: pl.Tensor[[64, 128], pl.FP32],
        ) -> pl.Tensor[[16, 32], pl.FP32]:
            return A[0:16, 0:32]

        assert isinstance(slice_both, ir.Function)
        printed = slice_both.as_python()
        assert "tensor.slice" in printed

    def test_tensor_slice_open_end(self):
        """A[32:, :] with open end -> tensor.slice with shape = dim - start."""

        @pl.function
        def slice_open_end(
            A: pl.Tensor[[64, 128], pl.FP32],
        ) -> pl.Tensor[[32, 128], pl.FP32]:
            return A[32:, :]

        assert isinstance(slice_open_end, ir.Function)
        printed = slice_open_end.as_python()
        assert "tensor.slice" in printed

    def test_tensor_mixed_subscript(self):
        """A[0:16, 0] with mixed int and slice -> tensor.slice with shape [16, 1]."""

        @pl.function
        def mixed_sub(
            A: pl.Tensor[[64, 128], pl.FP32],
        ) -> pl.Tensor[[16, 1], pl.FP32]:
            return A[0:16, 0]

        assert isinstance(mixed_sub, ir.Function)
        printed = mixed_sub.as_python()
        assert "tensor.slice" in printed

    def test_tensor_subscript_variable_indices(self):
        """A[i, j] with variable indices on Tensor -> tensor.read."""

        @pl.function
        def read_var(
            A: pl.Tensor[[64, 128], pl.FP32],
            i: pl.Scalar[pl.INDEX],
            j: pl.Scalar[pl.INDEX],
        ) -> pl.Scalar[pl.FP32]:
            return A[i, j]

        assert isinstance(read_var, ir.Function)
        printed = read_var.as_python()
        assert "tensor.read" in printed

    def test_tensor_subscript_step_error(self):
        """A[0:16:2, :] with step should raise error."""
        with pytest.raises(UnsupportedFeatureError, match="step"):

            @pl.function
            def bad_step(
                A: pl.Tensor[[64, 128], pl.FP32],
            ) -> pl.Tensor[[8, 128], pl.FP32]:
                return A[0:16:2, :]

    def test_tensor_subscript_wrong_rank(self):
        """A[0] on a 2D tensor -> error (rank mismatch)."""
        with pytest.raises(ParserTypeError, match="2 indices"):

            @pl.function
            def bad_rank(
                A: pl.Tensor[[64, 128], pl.FP32],
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                return A[0]


class TestTileSubscript:
    """Tests for tile subscript syntax on Tile types."""

    def test_tile_slice_via_subscript(self):
        """A[0:16, :] on Tile -> tile.slice."""

        @pl.function
        def slice_tile(
            x: pl.Tensor[[64, 128], pl.FP32],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            t: pl.Tile[[64, 128], pl.FP32] = pl.load(x, [0, 0], [64, 128])
            sliced: pl.Tile[[16, 128], pl.FP32] = t[0:16, :]
            return pl.store(sliced, [0, 0], x)

        assert isinstance(slice_tile, ir.Function)
        printed = slice_tile.as_python()
        assert "tile.slice" in printed

    def test_tile_slice_dynamic_upper_uses_valid_shape(self):
        """t[:, :valid_cols] keeps static shape and lowers valid_cols into valid_shape."""

        @pl.function
        def slice_tile_dynamic(
            x: pl.Tensor[[64, 128], pl.FP32],
            valid_cols: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            t: pl.Tile[[64, 128], pl.FP32] = pl.load(x, [0, 0], [64, 128])
            sliced: pl.Tile[[64, 128], pl.FP32] = t[:, :valid_cols]
            return pl.store(sliced, [0, 0], x)

        assert isinstance(slice_tile_dynamic, ir.Function)
        assert isinstance(slice_tile_dynamic.body, ir.SeqStmts)

        slice_stmt = slice_tile_dynamic.body.stmts[1]
        assert isinstance(slice_stmt, ir.AssignStmt)
        assert isinstance(slice_stmt.value, ir.Call)
        assert slice_stmt.value.op.name == "tile.slice"
        assert len(slice_stmt.value.args) == 4

        shape_tuple = slice_stmt.value.args[1]
        valid_shape_tuple = slice_stmt.value.args[3]
        assert isinstance(shape_tuple, ir.MakeTuple)
        assert isinstance(valid_shape_tuple, ir.MakeTuple)
        assert [cast(ir.ConstInt, dim).value for dim in shape_tuple.elements] == [64, 128]
        assert cast(ir.ConstInt, valid_shape_tuple.elements[0]).value == 64
        assert isinstance(valid_shape_tuple.elements[1], ir.Min)

    def test_tile_full_slice_preserves_input_valid_shape(self):
        """t[:, :] preserves the source tile's logical valid_shape."""

        @pl.function
        def slice_tile_full(
            x: pl.Tensor[[64, 128], pl.FP32],
            valid_cols: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            t: pl.Tile[[64, 128], pl.FP32] = pl.load(x, [0, 0], [64, 128], valid_shapes=[64, valid_cols])
            sliced: pl.Tile[[64, 128], pl.FP32] = t[:, :]
            return pl.store(sliced, [0, 0], x)

        assert isinstance(slice_tile_full, ir.Function)
        assert isinstance(slice_tile_full.body, ir.SeqStmts)

        slice_stmt = slice_tile_full.body.stmts[1]
        assert isinstance(slice_stmt, ir.AssignStmt)
        assert isinstance(slice_stmt.value, ir.Call)
        assert slice_stmt.value.op.name == "tile.slice"
        assert len(slice_stmt.value.args) == 4

        valid_shape_tuple = slice_stmt.value.args[3]
        assert isinstance(valid_shape_tuple, ir.MakeTuple)
        assert cast(ir.ConstInt, valid_shape_tuple.elements[0]).value == 64
        assert cast(ir.Var, valid_shape_tuple.elements[1]).name_hint == "valid_cols"

    def test_tile_slice_upper_intersects_input_valid_shape(self):
        """t[:, :16] should not widen a source tile that already has narrower valid_shape."""

        @pl.function
        def slice_tile_capped(
            x: pl.Tensor[[64, 128], pl.FP32],
            valid_cols: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            t: pl.Tile[[64, 128], pl.FP32] = pl.load(x, [0, 0], [64, 128], valid_shapes=[64, valid_cols])
            sliced: pl.Tile[[64, 16], pl.FP32] = t[:, :16]
            return pl.store(sliced, [0, 0], x)

        assert isinstance(slice_tile_capped, ir.Function)
        assert isinstance(slice_tile_capped.body, ir.SeqStmts)

        slice_stmt = slice_tile_capped.body.stmts[1]
        assert isinstance(slice_stmt, ir.AssignStmt)
        assert isinstance(slice_stmt.value, ir.Call)
        assert slice_stmt.value.op.name == "tile.slice"
        assert len(slice_stmt.value.args) == 4

        shape_tuple = slice_stmt.value.args[1]
        valid_shape_tuple = slice_stmt.value.args[3]
        assert isinstance(shape_tuple, ir.MakeTuple)
        assert isinstance(valid_shape_tuple, ir.MakeTuple)
        assert [cast(ir.ConstInt, dim).value for dim in shape_tuple.elements] == [64, 16]
        assert cast(ir.ConstInt, valid_shape_tuple.elements[0]).value == 64
        assert isinstance(valid_shape_tuple.elements[1], ir.Min)

    def test_tile_slice_static_upper_clamps_to_source_shape(self):
        """t[:, :256] should clamp its static shape to the source tile extent."""

        @pl.function
        def slice_tile_clamped(
            x: pl.Tensor[[64, 128], pl.FP32],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            t: pl.Tile[[64, 128], pl.FP32] = pl.load(x, [0, 0], [64, 128])
            sliced: pl.Tile[[64, 128], pl.FP32] = t[:, :256]
            return pl.store(sliced, [0, 0], x)

        assert isinstance(slice_tile_clamped, ir.Function)
        assert isinstance(slice_tile_clamped.body, ir.SeqStmts)

        slice_stmt = slice_tile_clamped.body.stmts[1]
        assert isinstance(slice_stmt, ir.AssignStmt)
        assert isinstance(slice_stmt.value, ir.Call)
        assert slice_stmt.value.op.name == "tile.slice"
        assert len(slice_stmt.value.args) == 3

        shape_tuple = slice_stmt.value.args[1]
        assert isinstance(shape_tuple, ir.MakeTuple)
        assert [cast(ir.ConstInt, dim).value for dim in shape_tuple.elements] == [64, 128]

    def test_tile_read_via_subscript(self):
        """A[0, 0] with literal integer indices on Tile -> tile.read."""

        @pl.function
        def read_tile_elem(
            x: pl.Tensor[[64, 128], pl.FP32],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            t: pl.Tile[[64, 128], pl.FP32] = pl.load(x, [0, 0], [64, 128])
            _elem: pl.Scalar[pl.FP32] = t[0, 0]
            return pl.store(t, [0, 0], x)

        assert isinstance(read_tile_elem, ir.Function)
        printed = read_tile_elem.as_python()
        assert "tile.read" in printed

    def test_tile_read_variable_indices(self):
        """A[i, j] with variable INDEX scalars on Tile -> tile.read."""

        @pl.function
        def read_var(
            x: pl.Tensor[[64, 128], pl.FP32],
            i: pl.Scalar[pl.INDEX],
            j: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            t: pl.Tile[[64, 128], pl.FP32] = pl.load(x, [0, 0], [64, 128])
            _elem: pl.Scalar[pl.FP32] = t[i, j]
            return pl.store(t, [0, 0], x)

        assert isinstance(read_var, ir.Function)
        printed = read_var.as_python()
        assert "tile.read" in printed

    def test_tile_read_wrong_rank(self):
        """A[0] on a 2D tile -> error (rank mismatch)."""
        with pytest.raises(ParserTypeError, match="2 indices"):

            @pl.function
            def bad_rank(
                x: pl.Tensor[[64, 128], pl.FP32],
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                t: pl.Tile[[64, 128], pl.FP32] = pl.load(x, [0, 0], [64, 128])
                _elem: pl.Scalar[pl.FP32] = t[0]
                return pl.store(t, [0, 0], x)

    def test_tile_subscript_step_error(self):
        """A[0:16:2, :] with step on tile should raise error."""
        with pytest.raises(UnsupportedFeatureError, match="step"):

            @pl.function
            def bad_step(
                x: pl.Tensor[[64, 128], pl.FP32],
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                t: pl.Tile[[64, 128], pl.FP32] = pl.load(x, [0, 0], [64, 128])
                sliced: pl.Tile[[8, 128], pl.FP32] = t[0:16:2, :]
                return pl.store(sliced, [0, 0], x)

    def test_tile_subscript_dynamic_lower_error(self):
        """A[:, start:] on Tile should reject dynamic lower bounds."""
        with pytest.raises(UnsupportedFeatureError, match="Dynamic lower bounds"):

            @pl.function
            def bad_dynamic_lower(
                x: pl.Tensor[[64, 128], pl.FP32],
                start: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                t: pl.Tile[[64, 128], pl.FP32] = pl.load(x, [0, 0], [64, 128])
                sliced: pl.Tile[[64, 128], pl.FP32] = t[:, start:]
                return pl.store(sliced, [0, 0], x)

    def test_tile_subscript_empty_static_slice_error(self):
        """A[:, 10:5] on Tile should reject empty static slices."""
        with pytest.raises(UnsupportedFeatureError, match="positive static extent"):

            @pl.function
            def bad_empty_static_slice(
                x: pl.Tensor[[64, 128], pl.FP32],
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                t: pl.Tile[[64, 128], pl.FP32] = pl.load(x, [0, 0], [64, 128])
                sliced: pl.Tile[[64, 128], pl.FP32] = t[:, 10:5]
                return pl.store(sliced, [0, 0], x)


class TestTupleSubscript:
    """Verify existing tuple subscript still works."""

    def test_tuple_subscript_still_works(self):
        """For-loop tuple unpacking still works after subscript dispatch changes."""

        @pl.function
        def tuple_access(
            x: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[1], pl.FP32]:
            init: pl.Tensor[[1], pl.FP32] = pl.create_tensor([1], dtype=pl.FP32)
            for i, (acc,) in pl.range(64, init_values=(init,)):
                elem: pl.Tensor[[1], pl.FP32] = pl.slice(x, [1], [i])
                new_acc: pl.Tensor[[1], pl.FP32] = pl.add(acc, elem)
                acc_out: pl.Tensor[[1], pl.FP32] = pl.yield_(new_acc)
            return acc_out

        assert isinstance(tuple_access, ir.Function)


class TestTensorSubscriptWrite:
    """Tests for tensor subscript-write syntax: A[i:i+H, j:j+W] = src."""

    def test_tensor_assemble_via_subscript_write(self):
        """A[i:i+16, j:j+32] = src on Tensor -> tensor.assemble (rebinds A)."""

        @pl.function
        def write_slice(
            out: pl.Tensor[[64, 128], pl.FP32],
            src: pl.Tensor[[16, 32], pl.FP32],
            i: pl.Scalar[pl.INDEX],
            j: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            out[i : i + 16, j : j + 32] = src
            return out

        assert isinstance(write_slice, ir.Function)
        printed = write_slice.as_python()
        assert "tensor.assemble" in printed

    def test_tensor_subscript_write_constant_offsets(self):
        """A[0:16, 0:32] = src lowers to tensor.assemble with literal offsets [0, 0]."""

        @pl.function
        def write_const(
            out: pl.Tensor[[64, 128], pl.FP32],
            src: pl.Tensor[[16, 32], pl.FP32],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            out[0:16, 0:32] = src
            return out

        assert isinstance(write_const, ir.Function)
        assert isinstance(write_const.body, ir.SeqStmts)

        assemble_stmt = write_const.body.stmts[0]
        assert isinstance(assemble_stmt, ir.AssignStmt)
        assert isinstance(assemble_stmt.value, ir.Call)
        assert assemble_stmt.value.op.name == "tensor.assemble"

        offset_tuple = assemble_stmt.value.args[2]
        assert isinstance(offset_tuple, ir.MakeTuple)
        assert [cast(ir.ConstInt, e).value for e in offset_tuple.elements] == [0, 0]

    def test_tensor_subscript_write_open_lower(self):
        """A[:16, :32] = src treats omitted lower bound as 0."""

        @pl.function
        def write_open(
            out: pl.Tensor[[64, 128], pl.FP32],
            src: pl.Tensor[[16, 32], pl.FP32],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            out[:16, :32] = src
            return out

        assert isinstance(write_open.body, ir.SeqStmts)
        assemble_stmt = write_open.body.stmts[0]
        assert isinstance(assemble_stmt, ir.AssignStmt)
        assert isinstance(assemble_stmt.value, ir.Call)
        offset_tuple = assemble_stmt.value.args[2]
        assert isinstance(offset_tuple, ir.MakeTuple)
        assert [cast(ir.ConstInt, e).value for e in offset_tuple.elements] == [0, 0]

    def test_tensor_subscript_write_step_error(self):
        """A[0:16:2, :] = src must reject slice steps."""

        with pytest.raises(UnsupportedFeatureError, match="step"):

            @pl.function
            def bad_step(
                out: pl.Tensor[[64, 128], pl.FP32],
                src: pl.Tensor[[8, 128], pl.FP32],
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                out[0:16:2, :] = src
                return out

    def test_tensor_subscript_write_wrong_rank(self):
        """A[0:16] = src on a 2D tensor must reject (rank mismatch)."""

        with pytest.raises(ParserTypeError, match="2 indices"):

            @pl.function
            def bad_rank(
                out: pl.Tensor[[64, 128], pl.FP32],
                src: pl.Tensor[[16, 128], pl.FP32],
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                out[0:16] = src
                return out

    def test_tensor_subscript_write_element_form_unsupported(self):
        """A[i, j] = scalar must be rejected for now (no element-write op wiring)."""

        with pytest.raises(UnsupportedFeatureError, match="Element-write"):

            @pl.function
            def bad_elem(
                out: pl.Tensor[[64, 128], pl.FP32],
                v: pl.Scalar[pl.FP32],
                i: pl.Scalar[pl.INDEX],
                j: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                out[i, j] = v
                return out

    def test_tensor_subscript_write_strict_ssa_rejected(self):
        """Subscript-write must be rejected under strict_ssa=True."""

        with pytest.raises(UnsupportedFeatureError, match="before SSA conversion"):

            @pl.function(strict_ssa=True)
            def bad_ssa(
                out: pl.Tensor[[64, 128], pl.FP32],
                src: pl.Tensor[[16, 32], pl.FP32],
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                out[0:16, 0:32] = src
                return out

    def test_tensor_subscript_write_shape_mismatch_static(self):
        """Static shape mismatch on a slice axis must be reported with axis + extents."""

        with pytest.raises(ParserTypeError, match="shape mismatch on axis 0"):

            @pl.function
            def bad_shape(
                out: pl.Tensor[[64, 128], pl.FP32],
                src: pl.Tensor[[8, 32], pl.FP32],  # axis 0 should be 16
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                out[0:16, 0:32] = src
                return out

    def test_tensor_subscript_write_full_axis_shape_mismatch(self):
        """`out[:, :] = src` requires src to fill the target shape exactly."""

        with pytest.raises(ParserTypeError, match="shape mismatch on axis 1"):

            @pl.function
            def bad_full(
                out: pl.Tensor[[64, 128], pl.FP32],
                src: pl.Tensor[[64, 32], pl.FP32],  # axis 1 should be 128
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                out[:, :] = src
                return out

    def test_tensor_subscript_write_rank_mismatch(self):
        """A 1D source on a 2D target must be rejected with a rank-mismatch error."""

        with pytest.raises(ParserTypeError, match="must be 2D"):

            @pl.function
            def bad_rank_src(
                out: pl.Tensor[[64, 128], pl.FP32],
                src: pl.Tensor[[32], pl.FP32],
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                out[0:1, 0:32] = src
                return out

    def test_tensor_subscript_write_symbolic_extent_simplifies_match(self):
        """``out[i:i+16, j:j+32] = src`` simplifies (i+16)-i=16 etc. and matches src."""

        @pl.function
        def symbolic_match(
            out: pl.Tensor[[64, 128], pl.FP32],
            src: pl.Tensor[[16, 32], pl.FP32],
            i: pl.Scalar[pl.INDEX],
            j: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            out[i : i + 16, j : j + 32] = src
            return out

        assert isinstance(symbolic_match, ir.Function)
        assert "tensor.assemble" in symbolic_match.as_python()

    def test_tensor_subscript_write_symbolic_extent_simplifies_mismatch(self):
        """``out[i:i+8, ...] = src`` simplifies to 8 — must reject when src has 16."""

        with pytest.raises(ParserTypeError, match="shape mismatch on axis 0"):

            @pl.function
            def bad_symbolic(
                out: pl.Tensor[[64, 128], pl.FP32],
                src: pl.Tensor[[16, 32], pl.FP32],  # axis 0 should be 8
                i: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                out[i : i + 8, 0:32] = src
                return out

    def test_tensor_subscript_write_unfoldable_extent_skipped(self):
        """Genuinely-symbolic extents (``out[:, :k] = src`` with runtime k) are trusted."""

        @pl.function
        def truly_symbolic(
            out: pl.Tensor[[64, 128], pl.FP32],
            src: pl.Tensor[[64, 128], pl.FP32],
            k: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            # Upper bound `k` cannot be statically reduced — the parser cannot
            # prove a mismatch, so it accepts the write.
            out[:, :k] = src
            return out

        assert isinstance(truly_symbolic, ir.Function)
        assert "tensor.assemble" in truly_symbolic.as_python()


class TestTileSubscriptWrite:
    """Tests for tile subscript-write syntax on Tile types."""

    def test_tile_assemble_via_subscript_write(self):
        """t[0:16, 0:32] = src on Tile -> tile.assemble."""

        @pl.function
        def write_tile(
            x: pl.Tensor[[64, 128], pl.FP32],
            src: pl.Tile[[16, 32], pl.FP32],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            t: pl.Tile[[64, 128], pl.FP32] = pl.load(x, [0, 0], [64, 128])
            t[0:16, 0:32] = src
            return pl.store(t, [0, 0], x)

        assert isinstance(write_tile, ir.Function)
        printed = write_tile.as_python()
        assert "tile.assemble" in printed

        assert isinstance(write_tile.body, ir.SeqStmts)
        assemble_stmt = write_tile.body.stmts[1]
        assert isinstance(assemble_stmt, ir.AssignStmt)
        assert isinstance(assemble_stmt.value, ir.Call)
        assert assemble_stmt.value.op.name == "tile.assemble"

        offset_tuple = assemble_stmt.value.args[2]
        assert isinstance(offset_tuple, ir.MakeTuple)
        assert [cast(ir.ConstInt, e).value for e in offset_tuple.elements] == [0, 0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
