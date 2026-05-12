# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for tile.gather_compare (compare-form pto.tgather, DPS-via-args).

Type contract (enforced by the op's type deduction):
    * ``src`` dtype in {FP16, FP32, INT16, INT32}; tile lives in Vec.
    * ``dst`` dtype is always INT32 (gathered indices); tile lives in Vec.
    * ``kvalue`` is a scalar whose dtype equals ``src``.
    * ``tmp`` is a UINT8 workspace tile (synthesized by the
      tensor→tile conversion pass; carried through as-is at the tile surface).
"""

import pypto.language as pl
import pytest

_VALID_SRC_DTYPES = [pl.FP16, pl.FP32, pl.INT16, pl.INT32]
_INVALID_SRC_DTYPES = [pl.UINT16, pl.UINT32, pl.UINT8, pl.INT64]


def _build_program(cmp_mode: str | int = "eq", offset=0, src_dtype=pl.FP32):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[32, 64], src_dtype],
            kvalue: pl.Scalar[src_dtype],
            out_dst: pl.Tensor[[32, 8], pl.INT32],
            out_cdst: pl.Tensor[[1, 32], pl.INT32],
        ):
            s: pl.Tile[[32, 64], src_dtype] = pl.load(src, [0, 0], [32, 64])
            tmp: pl.Tile[[32, 64], pl.UINT8] = pl.tile.create([32, 64], pl.UINT8)
            d, c = pl.tile.gather_compare(s, kvalue, tmp, cmp_mode=cmp_mode, offset=offset, out_cols=8)
            pl.store(d, [0, 0], out_dst)
            pl.store(c, [0, 0], out_cdst)

    return Program


class TestTileGatherCompareTypes:
    """Type-contract tests: dtype allowed / disallowed, dst INT32, kvalue matches src."""

    @pytest.mark.parametrize("src_dtype", _VALID_SRC_DTYPES)
    def test_valid_src_dtype(self, src_dtype):
        prog = _build_program(src_dtype=src_dtype)
        text = str(prog)
        assert "tile.gather_compare" in text
        # dst tile must carry INT32 regardless of src dtype.
        assert "pl.INT32" in text

    @pytest.mark.parametrize("src_dtype", _INVALID_SRC_DTYPES)
    def test_invalid_src_dtype_raises(self, src_dtype):
        with pytest.raises(Exception, match="src dtype"):
            _build_program(src_dtype=src_dtype)

    def test_kvalue_dtype_mismatch_raises(self):
        with pytest.raises(Exception, match="kvalue dtype"):

            @pl.program
            class Bad:
                @pl.function(type=pl.FunctionType.InCore)
                def main(
                    self,
                    src: pl.Tensor[[32, 64], pl.FP32],
                    kvalue: pl.Scalar[pl.FP16],  # mismatch: kvalue FP16 vs src FP32
                    out_dst: pl.Tensor[[32, 8], pl.INT32],
                    out_cdst: pl.Tensor[[1, 32], pl.INT32],
                ):
                    s: pl.Tile[[32, 64], pl.FP32] = pl.load(src, [0, 0], [32, 64])
                    tmp: pl.Tile[[32, 64], pl.UINT8] = pl.tile.create([32, 64], pl.UINT8)
                    d, c = pl.tile.gather_compare(s, kvalue, tmp, cmp_mode="eq", out_cols=8)
                    pl.store(d, [0, 0], out_dst)
                    pl.store(c, [0, 0], out_cdst)


class TestTileGatherCompareCmpMode:
    """cmp_mode accepts strings and ints in [0, 5]; otherwise raises."""

    def test_default_eq(self):
        prog = _build_program()
        assert "tile.gather_compare" in str(prog)

    def test_gt_with_offset(self):
        prog = _build_program(cmp_mode="gt", offset=4)
        assert "tile.gather_compare" in str(prog)

    def test_int_cmp_mode(self):
        prog = _build_program(cmp_mode=2)  # lt
        assert "tile.gather_compare" in str(prog)

    def test_invalid_cmp_mode_string(self):
        with pytest.raises(Exception, match="cmp_mode"):
            _build_program(cmp_mode="bogus")

    def test_invalid_cmp_mode_int(self):
        with pytest.raises(Exception, match="cmp_mode"):
            _build_program(cmp_mode=99)


def _build_tensor_compare_program(cmp_mode="eq", offset=0, src_dtype=pl.FP32):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[32, 64], src_dtype],
            kvalue: pl.Scalar[src_dtype],
        ) -> tuple[pl.Tensor[[32, 8], pl.INT32], pl.Tensor[[1, 32], pl.INT32]]:
            d, c = pl.tensor.gather(src, kvalue=kvalue, cmp_mode=cmp_mode, offset=offset, out_cols=8)
            return d, c

    return Program


class TestTensorGatherCompareTypes:
    """Tensor-level unified gather routing to tensor.gather_compare."""

    @pytest.mark.parametrize("src_dtype", _VALID_SRC_DTYPES)
    def test_valid_src_dtype(self, src_dtype):
        prog = _build_tensor_compare_program(src_dtype=src_dtype)
        assert "tensor.gather_compare" in str(prog)

    @pytest.mark.parametrize("src_dtype", _INVALID_SRC_DTYPES)
    def test_invalid_src_dtype_raises(self, src_dtype):
        with pytest.raises(Exception, match="input dtype"):
            _build_tensor_compare_program(src_dtype=src_dtype)

    def test_compare_with_offset(self):
        prog = _build_tensor_compare_program(cmp_mode="gt", offset=4)
        assert "tensor.gather_compare" in str(prog)

    def test_mutually_exclusive_index_and_compare(self):
        with pytest.raises(Exception, match="mutually exclusive"):

            @pl.program
            class Bad:
                @pl.function(type=pl.FunctionType.InCore)
                def main(
                    self,
                    src: pl.Tensor[[32, 64], pl.FP32],
                    idx: pl.Tensor[[32, 8], pl.INT32],
                    kv: pl.Scalar[pl.FP32],
                ) -> pl.Tensor[[32, 8], pl.FP32]:
                    return pl.tensor.gather(src, dim=-1, index=idx, kvalue=kv, cmp_mode="eq", out_cols=8)

    def test_mutually_exclusive_mask_and_compare(self):
        with pytest.raises(Exception, match="mutually exclusive"):

            @pl.program
            class Bad:
                @pl.function(type=pl.FunctionType.InCore)
                def main(
                    self,
                    src: pl.Tensor[[32, 64], pl.FP32],
                    kv: pl.Scalar[pl.FP32],
                ) -> pl.Tensor[[32, 8], pl.INT32]:
                    return pl.tensor.gather(src, mask_pattern=1, kvalue=kv, cmp_mode="eq", out_cols=8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
