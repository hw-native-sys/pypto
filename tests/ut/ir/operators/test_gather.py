# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for tile.gather (index-form pto.tgather).

Type contract after the A5 index-form alignment:
    * ``src`` dtype in {FP16, FP32, INT16, INT32}; tile lives in Vec.
    * ``indices`` dtype is INT32 on A2/A3, or INT16/INT32 on A5; tile lives in Vec.
    * ``tmp`` is a workspace operand required by the IR but NOT read by the
      A5/A2A3 index-form hardware, so any Vec tile dtype/shape is accepted.
    * ``dst`` dtype equals ``src``; shape equals ``indices``.
"""

import pypto.language as pl
import pytest

_VALID_SRC_DTYPES = [pl.FP16, pl.FP32, pl.INT16, pl.INT32]
_VALID_INDEX_DTYPES = [pl.INT32, pl.INT16]  # INT16 is A5-only; A2/A3 rejects at PTOAS verify


def _build_program(src_dtype=pl.FP32, index_dtype=pl.INT32, tmp_dtype=pl.FP32):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[8, 32], src_dtype],
            idx: pl.Tensor[[8, 32], index_dtype],
            out: pl.Tensor[[8, 32], src_dtype],
        ):
            s: pl.Tile[[8, 32], src_dtype] = pl.load(src, [0, 0], [8, 32])
            i: pl.Tile[[8, 32], index_dtype] = pl.load(idx, [0, 0], [8, 32])
            tmp: pl.Tile[[8, 32], tmp_dtype] = pl.tile.create([8, 32], tmp_dtype)
            d = pl.tile.gather(s, i, tmp)
            pl.store(d, [0, 0], out)

    return Program


class TestTileGatherIndexTypes:
    """Type-contract tests for the index form."""

    @pytest.mark.parametrize("src_dtype", _VALID_SRC_DTYPES)
    def test_valid_src_dtype(self, src_dtype):
        prog = _build_program(src_dtype=src_dtype)
        assert "tile.gather" in str(prog)

    @pytest.mark.parametrize("index_dtype", _VALID_INDEX_DTYPES)
    def test_valid_index_dtype(self, index_dtype):
        # INT16 indices pass deduction (A5 permits them); A2/A3 would reject later.
        prog = _build_program(index_dtype=index_dtype)
        assert "tile.gather" in str(prog)

    @pytest.mark.parametrize("tmp_dtype", [pl.FP32, pl.FP16, pl.INT32, pl.INT16])
    def test_tmp_dtype_unconstrained(self, tmp_dtype):
        # tmp is not read by the A5/A2A3 index form; any Vec tile dtype is accepted.
        prog = _build_program(tmp_dtype=tmp_dtype)
        assert "tile.gather" in str(prog)

    def test_tmp_rank_mismatch_accepted(self):
        # tmp shape is irrelevant to the index form; a different rank no longer raises.
        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                src: pl.Tensor[[8, 32], pl.FP32],
                idx: pl.Tensor[[8, 32], pl.INT32],
                out: pl.Tensor[[8, 32], pl.FP32],
            ):
                s: pl.Tile[[8, 32], pl.FP32] = pl.load(src, [0, 0], [8, 32])
                i: pl.Tile[[8, 32], pl.INT32] = pl.load(idx, [0, 0], [8, 32])
                tmp: pl.Tile[[1, 32], pl.FP32] = pl.tile.create([1, 32], pl.FP32)
                d = pl.tile.gather(s, i, tmp)
                pl.store(d, [0, 0], out)

        assert "tile.gather" in str(Program)

    def test_invalid_src_dtype_raises(self):
        with pytest.raises(Exception, match="src dtype"):
            _build_program(src_dtype=pl.UINT8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
