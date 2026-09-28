# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicitly boxed tensor.matmul retains valid M/N through the full compiler."""

import pypto.language as pl
import pytest
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.runtime import RunConfig


@pytest.mark.parametrize("a_trans,b_trans", [(False, False), (True, False), (False, True), (True, True)])
def test_explicit_tensor_matmul_valid_shape(tmp_path, ascend_backend, a_trans, b_trans):
    """Allocate a legal 32x32 Acc and write only its 17x24 valid rectangle."""
    a0, a1 = (128, 17) if a_trans else (17, 128)
    b0, b1 = (24, 128) if b_trans else (128, 24)
    a_shape = (128, 32) if a_trans else (32, 128)
    b_shape = (32, 128) if b_trans else (128, 32)

    @pl.jit
    def explicit_matmul(
        a: pl.Tensor[[a0, a1], pl.FP16],
        b: pl.Tensor[[b0, b1], pl.FP16],
        out: pl.Out[pl.Tensor[[17, 24], pl.FP32]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            av = pl.slice(a, a_shape, [0, 0], valid_shape=[a0, a1], clamp=True)
            bv = pl.slice(b, b_shape, [0, 0], valid_shape=[b0, b1], clamp=True)
            c = pl.matmul(av, bv, out_dtype=pl.FP32, a_trans=a_trans, b_trans=b_trans)
            out = pl.assemble(out, c, [0, 0])
        return out

    explicit_matmul.compile(
        config=RunConfig(
            codegen_only=True, save_kernels=True, save_kernels_dir=str(tmp_path), dump_passes=True
        )
    )
    pto_files = list(tmp_path.rglob("*.pto"))
    assert pto_files
    pto = "\n".join(file.read_text() for file in pto_files)
    acc_allocs = [line for line in pto.splitlines() if "pto.alloc_tile" in line and "loc=acc" in line]
    assert acc_allocs
    assert all("rows=32, cols=32" in line for line in acc_allocs), pto
    assert all("valid_row = %c17_index valid_col = %c24_index" in line for line in acc_allocs), pto
    stores = [line for line in pto.splitlines() if "pto.tstore " in line]
    assert len(stores) == 1, pto
    assert "!pto.partition_tensor_view<17x24xf32>" in stores[0], stores[0]
    # JIT runs PTOAS when available. Assert that its C++ output really exists,
    # so a source-only compile cannot masquerade as assembler validation.
    if find_ptoas_binary() is not None:
        assert list((tmp_path / "kernels").rglob("*.cpp"))


@pytest.mark.parametrize("physical_m", [32, 64])
def test_tensor_matmul_peeled_carry_compiles(tmp_path, ascend_backend, physical_m):
    """Static partial products agree with both their SSA carry and Acc compact mode."""

    @pl.jit
    def split_k(
        a: pl.Tensor[[17, 256], pl.FP16],
        b: pl.Tensor[[256, 24], pl.FP16],
        out: pl.Out[pl.Tensor[[physical_m, 32], pl.FP32]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            acc = pl.create_tensor([physical_m, 32], dtype=pl.FP32)
            for k in pl.range(0, 256, 128):
                av = pl.slice(a, [physical_m, 128], [0, k], valid_shape=[17, 128], clamp=True)
                bv = pl.slice(b, [128, 32], [k, 0], valid_shape=[128, 24], clamp=True)
                if k == 0:
                    acc = pl.matmul(av, bv, out_dtype=pl.FP32)
                else:
                    acc = pl.matmul_acc(acc, av, bv)
            out = pl.assemble(out, acc, [0, 0])
        return out

    split_k.compile(config=RunConfig(codegen_only=True, save_kernels=True, save_kernels_dir=str(tmp_path)))
    files = list(tmp_path.rglob("*.pto"))
    assert files
    pto = "\n".join(file.read_text() for file in files)
    assert "pto.tmatmul.acc" in pto
    assert "!pto.partition_tensor_view<17x24xf32>" in pto


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
