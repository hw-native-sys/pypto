# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTO codegen checks for hand-written FP4E2M1X2 GM expand (carrier → nibble ABI)."""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes


@pytest.fixture(autouse=True)
def _reset_backend_after_test():
    yield
    reset_for_testing()


def _emit_incore_mlir(program) -> str:
    reset_for_testing()
    set_backend_type(BackendType.Ascend950)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    parts: list[str] = []
    for func in optimized.functions.values():
        if func.func_type in (pl.FunctionType.Orchestration, pl.FunctionType.Group):
            continue
        single = ir.Program([func], func.name, optimized.span)
        result = codegen.PTOCodegen().generate(single, emit_tile_addr=True)
        parts.append(result if isinstance(result, str) else "".join(result.values()))
    return "\n".join(parts)


def test_fp4e2m1x2_make_tensor_view_expands_to_nibble_units():
    """Param + InCore tensor.view + rank-3 leading strides expand carrier→nibble."""

    @pl.program
    class Rank2:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[2, 256], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[2, 256], pl.FP4E2M1X2]],
        ) -> pl.Tensor[[2, 256], pl.FP4E2M1X2]:
            viewed: pl.Tensor[[2, 256], pl.FP4E2M1X2] = pl.tensor.view(src, [2, 256])
            return pl.store(pl.load(viewed, [0, 0], [2, 256]), [0, 0], out)

    mlir = _emit_incore_mlir(Rank2)
    views = [line for line in mlir.splitlines() if "pto.make_tensor_view" in line and "f4E2M1x2" in line]
    assert views and all("512" in line for line in views), mlir
    # Carrier IR last-axis 256 must not appear unexpanded on make_tensor_view.
    assert all("%c256_index" not in line for line in views), mlir
    # Static ConstInt partition last-axis expand folds *2 (carrier 256 → nibble 512).
    partitions = [line for line in mlir.splitlines() if "partition_view" in line]
    assert partitions and all("%c512_index" in line for line in partitions), mlir

    @pl.program
    class Rank3:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[2, 16, 32], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[2, 16, 32], pl.FP4E2M1X2]],
        ) -> pl.Tensor[[2, 16, 32], pl.FP4E2M1X2]:
            return pl.store(pl.load(src, [0, 0, 0], [2, 16, 32]), [0, 0, 0], out)

    mlir3 = _emit_incore_mlir(Rank3)
    assert "!pto.f4E2M1x2" in mlir3
    views3 = [line for line in mlir3.splitlines() if "pto.make_tensor_view" in line and "f4E2M1x2" in line]
    assert views3, mlir3
    # Carrier [2,16,32] → nibble shape last-axis 32*2=64 (ConstInt fold).
    assert all("shape = [%c2_index, %c16_index, %c64_index]" in line for line in views3), mlir3
    # Leading strides expand via *2: row pitch 32*16 then *2 → 1024; mid stride 32*2 → 64.
    assert "arith.muli %c32_index, %c16_index : index" in mlir3, mlir3
    assert "arith.muli %c32_index, %c2_index : index" in mlir3, mlir3
    assert any(
        "arith.muli" in line and "_s0," in line and "%c2_index" in line for line in mlir3.splitlines()
    ), mlir3
    assert all("%c1_index]" in line for line in views3), mlir3


def test_fp4e2m1x2_rejects_column_vector_last_carrier_dim_one():
    """Carrier last-dim 1 would force DN; packed FP4E2M1X2 bans that path."""

    @pl.program
    class ColVecPacked:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[32, 1], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[32, 1], pl.FP4E2M1X2]],
        ) -> pl.Tensor[[32, 1], pl.FP4E2M1X2]:
            return pl.store(pl.load(src, [0, 0], [32, 1]), [0, 0], out)

    with pytest.raises(ValueError, match=r"last carrier dimension 1|column-vector"):
        _emit_incore_mlir(ColVecPacked)


def test_fp4e2m1x2_rejects_explicit_dn_param_annotation():
    """Non-ND annotations on packed FP4E2M1X2 are rejected at make_tensor_view."""

    @pl.program
    class DnAnnotated:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[
                [16, 32],
                pl.FP4E2M1X2,
                pl.TensorView(stride=[1, 16], layout=pl.TensorLayout.DN),
            ],
            out: pl.Out[pl.Tensor[[16, 32], pl.FP4E2M1X2]],
        ) -> pl.Tensor[[16, 32], pl.FP4E2M1X2]:
            return pl.store(pl.load(src, [0, 0], [16, 32]), [0, 0], out)

    with pytest.raises(ValueError, match=r"FP4E2M1X2 supports ND layout only"):
        _emit_incore_mlir(DnAnnotated)


def test_fp4e2m1x2_slice_cast_and_vec_move():
    """Even last-axis slice + cast emits f4x2; move stays on Vec."""

    @pl.program
    class CastProg:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[16, 32], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[16, 32], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[16, 32], pl.FP8E4M3FN]:
            return pl.store(pl.cast(pl.load(src, [0, 16], [16, 16]), pl.FP8E4M3FN), [0, 0], out)

    mlir = _emit_incore_mlir(CastProg)
    assert "!pto.f4E2M1x2" in mlir and "pto.tcvt" in mlir
    assert "pto.ttrans" not in mlir

    @pl.program
    class MoveProg:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[16, 32], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[16, 32], pl.FP4E2M1X2]],
        ) -> pl.Tensor[[16, 32], pl.FP4E2M1X2]:
            return pl.store(pl.move(pl.load(src, [0, 0], [16, 32]), target_memory=pl.Mem.Vec), [0, 0], out)

    mlir_m = _emit_incore_mlir(MoveProg)
    assert "!pto.f4E2M1x2" in mlir_m
    assert "pto.tmov" in mlir_m or "pto.tload" in mlir_m
