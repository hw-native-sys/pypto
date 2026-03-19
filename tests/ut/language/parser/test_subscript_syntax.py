# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for subscript syntax on Tensor and Tile types."""

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
