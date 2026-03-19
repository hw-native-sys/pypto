# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for pl.tile.assemble DSL API."""

import pypto.language as pl
import pytest
from pypto import ir


class TestTileAssemble:
    """Tests for tile.assemble via pl.tile.assemble DSL wrapper."""

    def test_assemble_roundtrip_zero_offset(self):
        """tile.assemble with zero offset survives print → parse round-trip."""

        @pl.function
        def original(x: pl.Tensor[[16, 128], pl.FP32]) -> pl.Tensor[[16, 128], pl.FP32]:
            target: pl.Tile[[16, 128], pl.FP32] = pl.load(x, [0, 0], [16, 128])
            source: pl.Tile[[16, 64], pl.FP32] = pl.load(x, [0, 0], [16, 64])
            result: pl.Tile[[16, 128], pl.FP32] = pl.tile.assemble(target, source, [0, 0])
            return pl.store(result, [0, 0], x)

        printed = ir.python_print(original)
        reparsed = pl.parse(printed)
        ir.assert_structural_equal(original, reparsed)

    def test_assemble_roundtrip_nonzero_offset(self):
        """tile.assemble with non-zero offset survives print → parse round-trip."""

        @pl.function
        def original(x: pl.Tensor[[16, 128], pl.FP32]) -> pl.Tensor[[16, 128], pl.FP32]:
            target: pl.Tile[[16, 128], pl.FP32] = pl.load(x, [0, 0], [16, 128])
            source: pl.Tile[[16, 64], pl.FP32] = pl.load(x, [0, 64], [16, 64])
            result: pl.Tile[[16, 128], pl.FP32] = pl.tile.assemble(target, source, [0, 64])
            return pl.store(result, [0, 0], x)

        printed = ir.python_print(original)
        reparsed = pl.parse(printed)
        ir.assert_structural_equal(original, reparsed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
