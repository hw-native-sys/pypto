# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTO codegen checks for packed FP4E2M1X2 GM carrier units (PTOAS v0.67+)."""

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


def test_fp4_make_tensor_view_packs_to_carrier_units():
    """Logical param views and partitions are packed exactly once."""

    @pl.program
    class Rank2:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[2, 512], pl.FP4],
            out: pl.Out[pl.Tensor[[2, 512], pl.FP4]],
        ) -> pl.Tensor[[2, 512], pl.FP4]:
            viewed: pl.Tensor[[2, 512], pl.FP4] = pl.tensor.view(src, [2, 512])
            return pl.store(pl.load(viewed, [0, 0], [2, 512]), [0, 0], out)

    mlir = _emit_incore_mlir(Rank2)
    views = [line for line in mlir.splitlines() if "pto.make_tensor_view" in line and "f4E2M1x2" in line]
    assert views and all("shape = [%c2_index, %c256_index]" in line for line in views), mlir
    assert all("strides = [%c256_index, %c1_index]" in line for line in views), mlir
    partitions = [line for line in mlir.splitlines() if "partition_view" in line]
    assert partitions and all("2x256x!pto.f4E2M1x2" in line for line in partitions), mlir

    @pl.program
    class Rank3:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[2, 16, 64], pl.FP4],
            out: pl.Out[pl.Tensor[[2, 16, 64], pl.FP4]],
        ) -> pl.Tensor[[2, 16, 64], pl.FP4]:
            return pl.store(pl.load(src, [0, 0, 0], [2, 16, 64]), [0, 0, 0], out)

    mlir3 = _emit_incore_mlir(Rank3)
    assert "!pto.f4E2M1x2" in mlir3
    views3 = [line for line in mlir3.splitlines() if "pto.make_tensor_view" in line]
    assert views3 and all("shape = [%c2_index, %c16_index, %c32_index]" in line for line in views3), mlir3
    assert "arith.muli %c32_index, %c16_index : index" in mlir3, mlir3
    assert "arith.muli %c32_index, %c2_index : index" not in mlir3, mlir3


def test_fp4_slice_cast_and_vec_move():
    """Even last-axis slice + cast emits f4x2; move stays on Vec."""

    @pl.program
    class CastProg:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[16, 64], pl.FP4],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
            return pl.store(pl.cast(pl.load(src, [0, 32], [16, 32]), pl.FP8E4M3FN), [0, 0], out)

    mlir = _emit_incore_mlir(CastProg)
    assert "!pto.f4E2M1x2" in mlir and "pto.tcvt" in mlir
    assert "pto.ttrans" not in mlir

    @pl.program
    class MoveProg:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[16, 64], pl.FP4],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP4]],
        ) -> pl.Tensor[[16, 64], pl.FP4]:
            return pl.store(pl.move(pl.load(src, [0, 0], [16, 64]), target_memory=pl.Mem.Vec), [0, 0], out)

    mlir_m = _emit_incore_mlir(MoveProg)
    assert "!pto.f4E2M1x2" in mlir_m
    assert "pto.tmov" in mlir_m or "pto.tload" in mlir_m


def test_fp4_cacheinvalid_preserves_carrier_partition():
    """system.cacheinvalid uses carrier widths after PackFp4."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            x: pl.Tensor[[16, 64], pl.FP4],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP4]],
        ) -> pl.Tensor[[16, 64], pl.FP4]:
            updated: pl.Tensor[[16, 64], pl.FP4] = pl.store(pl.load(x, [0, 0], [16, 64]), [0, 0], out)
            pl.system.cacheinvalid(updated, [16, 32], [0, 0])
            return updated

    mlir = _emit_incore_mlir(Prog)
    cmo_lines = [line for line in mlir.splitlines() if "pto.cmo.cacheinvalid" in line]
    assert len(cmo_lines) == 1, mlir
    assert "16x16x!pto.f4E2M1x2" in cmo_lines[0], cmo_lines[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
