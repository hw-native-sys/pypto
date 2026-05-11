# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for tile.gather_compare (compare-form pto.tgather, DPS-via-args)."""

import pypto.language as pl
import pytest


def _build_program(cmp_mode: str | int = "eq", offset=0, src_dtype=pl.FP32):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[32, 64], src_dtype],
            kvalue: pl.Scalar[pl.UINT32],
            out_dst: pl.Tensor[[32, 8], pl.INT32],
            out_cdst: pl.Tensor[[32], pl.INT32],
        ):
            s: pl.Tile[[32, 64], src_dtype] = pl.load(src, [0, 0], [32, 64])
            tmp: pl.Tile[[32, 64], pl.UINT8] = pl.tile.create([32, 64], pl.UINT8)
            d, c = pl.tile.gather_compare(s, kvalue, tmp, cmp_mode=cmp_mode, offset=offset, out_cols=8)
            pl.store(d, [0, 0], out_dst)
            pl.store(c, [0], out_cdst)

    return Program


class TestTileGatherCompare:
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
        with pytest.raises(Exception):
            _build_program(cmp_mode="bogus")

    def test_invalid_cmp_mode_int(self):
        with pytest.raises(Exception):
            _build_program(cmp_mode=99)

    def test_int32_src(self):
        prog = _build_program(src_dtype=pl.INT32)
        assert "tile.gather_compare" in str(prog)


def _build_tensor_compare_program(cmp_mode="eq", offset=0):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[32, 64], pl.FP32],
            kvalue: pl.Scalar[pl.UINT32],
        ) -> tuple[pl.Tensor[[32, 8], pl.INT32], pl.Tensor[[32], pl.INT32]]:
            d, c = pl.tensor.gather(src, kvalue=kvalue, cmp_mode=cmp_mode, offset=offset, out_cols=8)
            return d, c

    return Program


class TestTensorGatherUnified:
    def test_compare_dispatch(self):
        prog = _build_tensor_compare_program()
        assert "tensor.gather_compare" in str(prog)

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
                    kv: pl.Scalar[pl.UINT32],
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
                    kv: pl.Scalar[pl.UINT32],
                ) -> pl.Tensor[[32, 8], pl.INT32]:
                    return pl.tensor.gather(src, mask_pattern=1, kvalue=kv, cmp_mode="eq", out_cols=8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
