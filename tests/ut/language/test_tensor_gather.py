# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for pl.gather DSL API (tensor.gather operation).

Verifies that pl.gather() produces correct IR when used inside @pl.function,
including type deduction and error handling.

Test scenarios:
  1. 2D gather with dim=0 (row selection)
  2. 2D gather with dim=1 (MoE top-k pattern)
  3. 3D gather with dim=1 (batch attention pattern)
  4. 3D gather with dim=2 (last-dim selection)
  5. 2D gather with negative dim
  6. 3D gather with negative dim
  7. Orchestration function integration
  8. Round-trip print → parse
  9. Error cases: rank mismatch, non-integer index, dim out of range
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.parser.diagnostics.exceptions import InvalidOperationError


class TestGatherDSL:
    """Tests for pl.gather() via the DSL layer."""

    def test_gather_2d_dim0(self):
        """pl.gather with 2D tensors and dim=0 produces correct IR."""

        @pl.function
        def func(
            inp: pl.Tensor[[8, 4], pl.FP32],
            idx: pl.Tensor[[3, 4], pl.INT32],
        ) -> pl.Tensor[[3, 4], pl.FP32]:
            result: pl.Tensor[[3, 4], pl.FP32] = pl.gather(inp, 0, idx)
            return result

        printed = str(func)
        assert "tensor.gather" in printed

    def test_gather_2d_dim1(self):
        """pl.gather with dim=1 (MoE top-k pattern) produces correct output type."""

        @pl.function
        def func(
            weights: pl.Tensor[[16, 64], pl.FP16],
            topk_ids: pl.Tensor[[16, 4], pl.INT32],
        ) -> pl.Tensor[[16, 4], pl.FP16]:
            selected: pl.Tensor[[16, 4], pl.FP16] = pl.gather(weights, 1, topk_ids)
            return selected

        printed = str(func)
        assert "tensor.gather" in printed

    def test_gather_3d(self):
        """pl.gather with 3D tensors produces correct output shape."""

        @pl.function
        def func(
            inp: pl.Tensor[[4, 8, 16], pl.FP32],
            idx: pl.Tensor[[4, 3, 16], pl.INT64],
        ) -> pl.Tensor[[4, 3, 16], pl.FP32]:
            result: pl.Tensor[[4, 3, 16], pl.FP32] = pl.gather(inp, 1, idx)
            return result

        printed = str(func)
        assert "tensor.gather" in printed

    def test_gather_3d_dim2(self):
        """pl.gather with 3D tensors and dim=2 (last-dim selection)."""

        @pl.function
        def func(
            inp: pl.Tensor[[4, 8, 16], pl.FP32],
            idx: pl.Tensor[[4, 8, 5], pl.INT32],
        ) -> pl.Tensor[[4, 8, 5], pl.FP32]:
            result: pl.Tensor[[4, 8, 5], pl.FP32] = pl.gather(inp, 2, idx)
            return result

        printed = str(func)
        assert "tensor.gather" in printed

    def test_gather_negative_dim(self):
        """pl.gather with negative dim normalizes correctly."""

        @pl.function
        def func(
            inp: pl.Tensor[[4, 8], pl.FP32],
            idx: pl.Tensor[[4, 8], pl.INT32],
        ) -> pl.Tensor[[4, 8], pl.FP32]:
            result: pl.Tensor[[4, 8], pl.FP32] = pl.gather(inp, -1, idx)
            return result

        printed = str(func)
        assert "tensor.gather" in printed

    def test_gather_3d_negative_dim(self):
        """pl.gather with 3D tensors and negative dim (-2 normalizes to dim=1)."""

        @pl.function
        def func(
            inp: pl.Tensor[[4, 8, 16], pl.FP32],
            idx: pl.Tensor[[4, 3, 16], pl.INT32],
        ) -> pl.Tensor[[4, 3, 16], pl.FP32]:
            result: pl.Tensor[[4, 3, 16], pl.FP32] = pl.gather(inp, -2, idx)
            return result

        printed = str(func)
        assert "tensor.gather" in printed

    @pytest.mark.xfail(
        reason="Known limitation: printer outputs dim as kwarg but parser re-interprets "
        "positional args differently, causing 'multiple values for dim'. "
        "Same issue affects scatter_update and other dim-kwarg tensor ops.",
        strict=True,
    )
    def test_gather_roundtrip(self):
        """pl.gather survives print → parse round-trip."""

        @pl.function
        def original(
            inp: pl.Tensor[[8, 4], pl.FP32],
            idx: pl.Tensor[[3, 4], pl.INT32],
        ) -> pl.Tensor[[3, 4], pl.FP32]:
            result: pl.Tensor[[3, 4], pl.FP32] = pl.gather(inp, 0, idx)
            return result

        printed = ir.python_print(original)
        reparsed = pl.parse(printed)
        ir.assert_structural_equal(original, reparsed)

    def test_gather_in_orchestration(self):
        """pl.gather works inside an orchestration function."""

        @pl.program
        class GatherProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_noop(
                self,
                t: pl.Tensor[[3, 4], pl.FP32],
                out: pl.Out[pl.Tensor[[3, 4], pl.FP32]],
            ) -> pl.Tensor[[3, 4], pl.FP32]:
                tile: pl.Tile[[3, 4], pl.FP32] = pl.load(t, [0, 0], [3, 4])
                return pl.store(tile, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                inp: pl.Tensor[[8, 4], pl.FP32],
                idx: pl.Tensor[[3, 4], pl.INT32],
                out: pl.Out[pl.Tensor[[3, 4], pl.FP32]],
            ) -> pl.Tensor[[3, 4], pl.FP32]:
                gathered: pl.Tensor[[3, 4], pl.FP32] = pl.gather(inp, 0, idx)
                out = self.kernel_noop(gathered, out)
                return out

        assert GatherProgram is not None
        printed = str(GatherProgram)
        assert "tensor.gather" in printed


class TestGatherDSLErrors:
    """Tests for pl.gather() error cases via DSL layer."""

    def test_gather_rank_mismatch(self):
        """pl.gather rejects rank mismatch between input and index."""
        with pytest.raises(InvalidOperationError, match="index rank"):

            @pl.function
            def func(
                inp: pl.Tensor[[4, 8, 16], pl.FP32],
                idx: pl.Tensor[[4, 8], pl.INT32],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                return pl.gather(inp, 0, idx)

    def test_gather_non_integer_index(self):
        """pl.gather rejects non-integer index dtype."""
        with pytest.raises(InvalidOperationError, match="index dtype must be integer"):

            @pl.function
            def func(
                inp: pl.Tensor[[4, 8], pl.FP32],
                idx: pl.Tensor[[4, 8], pl.FP32],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                return pl.gather(inp, 0, idx)

    def test_gather_dim_out_of_range(self):
        """pl.gather rejects dim out of valid range."""
        with pytest.raises(InvalidOperationError, match="dim.*out of range"):

            @pl.function
            def func(
                inp: pl.Tensor[[4, 8], pl.FP32],
                idx: pl.Tensor[[4, 8], pl.INT32],
            ) -> pl.Tensor[[4, 8], pl.FP32]:
                return pl.gather(inp, 5, idx)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
