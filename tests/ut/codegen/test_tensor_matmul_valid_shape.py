# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicitly boxed tensor.matmul retains valid M/N through the full compiler."""

import re

import pypto.language as pl
import pytest
from pypto import ir
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.pypto_core import passes
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


@pytest.mark.parametrize("physical_m", [32, 64])
def test_tensor_matmul_else_first_carry_compiles(tmp_path, ascend_backend, physical_m):
    """The first product in the else branch still determines the valid carry region."""

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
                if k != 0:
                    acc = pl.matmul_acc(acc, av, bv)
                else:
                    acc = pl.matmul(av, bv, out_dtype=pl.FP32)
            out = pl.assemble(out, acc, [0, 0])
        return out

    split_k.compile(config=RunConfig(codegen_only=True, save_kernels=True, save_kernels_dir=str(tmp_path)))
    files = list(tmp_path.rglob("*.pto"))
    assert files
    pto = "\n".join(file.read_text() for file in files)
    assert "pto.tmatmul.acc" in pto
    assert "!pto.partition_tensor_view<17x24xf32>" in pto


@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
)
@pytest.mark.parametrize("buffer_ir", [False, True], ids=["buffer-disabled", "buffer-enabled"])
@pytest.mark.parametrize("batch_shape", [[], [4]], ids=["matrix", "batch"])
def test_transposed_matmul_l0b_capacity(tmp_path, ascend_backend, planner, buffer_ir, batch_shape):
    """Check L0B capacity and prove the selected IR path for 2D and batch stores."""
    a_shape = [*batch_shape, 64, 128]
    b_shape = [*batch_shape, 512, 128]
    out_shape = [*batch_shape, 64, 512]
    offsets = [0] * len(out_shape)
    selected = []
    lowered_stages = []

    def capture_selection(transform, program):
        if transform.get_name() == "SelectBufferIRFunctions":
            selected.extend(
                function.attrs["buffer_ir_selected"]
                for function in program.functions.values()
                if ir.is_incore_type(function.func_type)
            )
        if transform.get_name() == "LowerTileToBuffer":
            lowered_stages.extend(
                function.ir_stage
                for function in program.functions.values()
                if ir.is_incore_type(function.func_type)
            )

    @pl.jit
    def batch_transposed(
        a: pl.Tensor[a_shape, pl.FP32],
        b: pl.Tensor[b_shape, pl.FP32],
        out: pl.Out[pl.Tensor[out_shape, pl.FP32]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            result = pl.matmul(a, b, b_trans=True, out_dtype=pl.FP32)
            out = pl.assemble(out, result, offsets)
        return out

    instrument = passes.CallbackInstrument(after_pass=capture_selection)
    with passes.PassContext([instrument], memory_planner=planner, enable_buffer_ir=buffer_ir):
        batch_transposed.compile(
            config=RunConfig(codegen_only=True, save_kernels=True, save_kernels_dir=str(tmp_path))
        )
    pto_files = list(tmp_path.rglob("*.pto"))
    assert pto_files
    pto = "\n".join(file.read_text() for file in pto_files)
    # Rank-3 parameters are ineligible for Buffer IR even when it is enabled.
    # The 2D case must actually select Buffer IR to provide lowering coverage.
    assert selected == ([not batch_shape] if buffer_ir else []), selected
    expected_stage = ir.FunctionIRStage.Functional if batch_shape else ir.FunctionIRStage.Buffer
    assert lowered_stages == ([expected_stage] if buffer_ir else []), lowered_stages
    right_shapes = re.findall(r"loc=right, dtype=f32, rows=(\d+), cols=(\d+)", pto)
    assert right_shapes, pto
    # One oversized buffer cannot be repaired by allocator lifetime packing.
    assert all(int(rows) * int(cols) * 4 <= 65536 for rows, cols in right_shapes), right_shapes
    if batch_shape and planner == passes.MemoryPlanner.PTOAS:
        # PTOAS retains automatic dbC and selects the supported M/N path.
        # PYPTO and DSA_RP disable automatic dbC and select full-output K tiling.
        assert "!pto.partition_tensor_view<1x32x128xf32>" in pto, pto
        assert "!pto.partition_tensor_view<1x64x512xf32>" not in pto, pto
    elif batch_shape:
        assert "!pto.partition_tensor_view<1x64x512xf32>" in pto, pto
    if find_ptoas_binary() is not None:
        assert list((tmp_path / "kernels").rglob("*.cpp"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
