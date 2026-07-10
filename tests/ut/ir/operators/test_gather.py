# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for tile.gather (index-form pto.tgather) type deduction.

Drives the deducer directly via the IR op (no @pl.program) so the dtype
combinations are exercised without parser closure-variable resolution.

Type contract after the A5 index-form alignment:
    * ``src`` dtype in {FP16, FP32, INT16, INT32}; tile lives in Vec.
    * ``indices`` dtype is INT32 on A2/A3, or INT16/INT32 on A5; tile in Vec.
    * ``tmp`` is a workspace operand required by the IR but NOT read by the
      A5/A2A3 index-form hardware, so any Vec tile dtype is accepted.
    * ``dst`` dtype equals ``src``; shape equals ``indices``.
"""

import pytest
from pypto import DataType, ir
from pypto.ir.op import tile


def _gather(src_dtype: DataType, idx_dtype: DataType, tmp_dtype: DataType):
    span = ir.Span.unknown()
    src = ir.Var("src", ir.TileType([8, 32], src_dtype), span)
    idx = ir.Var("idx", ir.TileType([8, 32], idx_dtype), span)
    tmp = ir.Var("tmp", ir.TileType([8, 32], tmp_dtype), span)
    return tile.gather(src, idx, tmp)


class TestTileGatherIndexTypes:
    """Deducer type-contract tests for the index form."""

    @pytest.mark.parametrize("src_dtype", [DataType.FP16, DataType.FP32, DataType.INT16, DataType.INT32])
    def test_valid_src_dtype(self, src_dtype):
        call = _gather(src_dtype, DataType.INT32, DataType.FP32)
        assert isinstance(call.type, ir.TileType)
        assert call.type.dtype == src_dtype  # dst dtype follows src

    @pytest.mark.parametrize("idx_dtype", [DataType.INT32, DataType.INT16])
    def test_valid_index_dtype(self, idx_dtype):
        # INT16 indices pass deduction (A5 permits them); A2/A3 rejects later at PTOAS.
        call = _gather(DataType.FP32, idx_dtype, DataType.FP32)
        assert isinstance(call.type, ir.TileType)

    @pytest.mark.parametrize("tmp_dtype", [DataType.FP32, DataType.FP16, DataType.INT32, DataType.INT16])
    def test_tmp_dtype_unconstrained(self, tmp_dtype):
        # tmp is not read by the A5/A2A3 index form; any Vec tile dtype is accepted.
        call = _gather(DataType.FP32, DataType.INT32, tmp_dtype)
        assert isinstance(call.type, ir.TileType)

    def test_invalid_src_dtype_raises(self):
        with pytest.raises(Exception, match="src dtype"):
            _gather(DataType.UINT8, DataType.INT32, DataType.FP32)

    def test_invalid_index_dtype_raises(self):
        with pytest.raises(Exception, match="indices dtype"):
            _gather(DataType.FP32, DataType.FP32, DataType.FP32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
