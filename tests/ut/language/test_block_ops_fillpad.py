# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for fillpad block operation.

Tests verify:
1. Top-level access via pl.fillpad()
2. Explicit namespace access via pl.block.fillpad()
3. Both access methods produce structurally equal IR
"""

import pypto.language as pl
import pytest
from pypto import ir


class TestFillpadExport:
    """Verify fillpad is correctly exported to top-level namespace."""

    def test_fillpad_is_exported(self):
        """pl.fillpad should exist and be the same as pl.block.fillpad."""
        assert hasattr(pl, "fillpad")
        assert pl.fillpad is pl.block.fillpad


class TestFillpadOperation:
    """Test fillpad operation produces correct IR."""

    def test_fillpad_unified_vs_explicit(self):
        """pl.fillpad() produces the same IR as pl.block.fillpad()."""

        @pl.function
        def unified(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[64, 64])
            padded: pl.Tile[[64, 64], pl.FP32] = pl.fillpad(tile)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.block.store(
                padded, offsets=[0, 0], output_tensor=out
            )
            return result

        @pl.function
        def explicit(
            t: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile: pl.Tile[[64, 64], pl.FP32] = pl.block.load(t, offsets=[0, 0], shapes=[64, 64])
            padded: pl.Tile[[64, 64], pl.FP32] = pl.block.fillpad(tile)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.block.store(
                padded, offsets=[0, 0], output_tensor=out
            )
            return result

        ir.assert_structural_equal(unified, explicit)

    def test_fillpad_preserves_tile_type(self):
        """fillpad should preserve the tile type (same shape and dtype)."""

        @pl.function
        def fillpad_func(
            t: pl.Tensor[[128, 64], pl.FP16], out: pl.Tensor[[128, 64], pl.FP16]
        ) -> pl.Tensor[[128, 64], pl.FP16]:
            tile: pl.Tile[[128, 64], pl.FP16] = pl.block.load(t, offsets=[0, 0], shapes=[128, 64])
            padded: pl.Tile[[128, 64], pl.FP16] = pl.fillpad(tile)
            result: pl.Tensor[[128, 64], pl.FP16] = pl.block.store(
                padded, offsets=[0, 0], output_tensor=out
            )
            return result

        # Should parse without error - type annotations would fail if fillpad changed the type
        assert fillpad_func is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
