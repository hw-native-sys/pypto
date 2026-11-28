# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 primitives for manually scheduled L0 slots."""

import re

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir
from pypto.backend import BackendType, pto_backend
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir.op import tile
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def _generate_mlir(program):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    functions = [f for f in lowered.functions.values() if ir.is_incore_type(f.func_type)]
    return codegen.PTOCodegen().generate(ir.Program(functions, "Kernel", lowered.span))


def test_extract_into_dynamically_selected_l0_slots(tmp_path):
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.AIC)
        def kernel(
            self,
            q: pl.Tensor[[64, 128], pl.BF16],
            kv: pl.Tensor[[128, 64], pl.BF16],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            slot: pl.Scalar[pl.INDEX],
        ):
            l0a: pl.Tile[[64, 512], pl.BF16, pl.MemRef("l0a"), pl.Mem.Left] = pl.tile.create(
                [64, 512], pl.BF16, target_memory=pl.Mem.Left
            )
            l0b: pl.Tile[[512, 64], pl.BF16, pl.MemRef("l0b"), pl.Mem.Right] = pl.tile.create(
                [512, 64], pl.BF16, target_memory=pl.Mem.Right
            )
            p_l1 = pl.tile.load(q, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
            pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 0)
            v_l1 = pl.tile.load(kv, [0, 0], [128, 64], target_memory=pl.Mem.Mat)
            pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 1)
            p = pl.tile.slice(l0a, [64, 128], [0, slot * 128])
            pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 0)
            p = pl.tile.extract_into(p, p_l1, 0, 0)
            # Close both half writes back into the parent SSA value.  The
            # assemble operations must be self-copies of the exact same windows,
            # not extra L0B transfers.
            vlo = pl.tile.slice(l0b, [64, 64], [slot * 128, 0])
            vhi = pl.tile.slice(l0b, [64, 64], [slot * 128 + 64, 0])
            pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 1)
            vlo = pl.tile.extract_into(vlo, v_l1, 0, 0)
            vhi = pl.tile.extract_into(vhi, v_l1, 64, 0)
            l0b = pl.tile.assemble(l0b, vlo, [slot * 128, 0])
            l0b = pl.tile.assemble(l0b, vhi, [slot * 128 + 64, 0])
            pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, 0)
            v = pl.tile.slice(l0b, [128, 64], [slot * 128, 0])
            pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, 0)
            product = pl.tile.matmul(p, v)
            pl.system.set_flag(pl.PipeType.M, pl.PipeType.FIX, 0)
            pl.system.wait_flag(pl.PipeType.M, pl.PipeType.FIX, 0)
            return pl.tile.store(product, [0, 0], out)

    mlir = _generate_mlir(Prog)
    assert mlir.count("pto.textract ") == 3
    assert "pto.tmov " not in mlir
    assert "pto.tinsert " not in mlir
    assert "pto.tmatmul " in mlir
    # Two independent loads target the two existing physical views.
    native = re.findall(r"pto.textract .*? outs\((%[^ :]+)", mlir)
    assert len(set(native)) == 3, mlir
    if find_ptoas_binary() is not None:
        cpp = pto_backend._compile_pto_module(mlir, "kernel", str(tmp_path))
        assert cpp.count("TEXTRACT(") == 3
        assert "TMATMUL(" in cpp
        # Static explicit flags disable PTOAS auto-sync for this function.
        # In the reference schedule both V half transfers are issued together;
        # the compiler must not serialize them with another barrier or event.
        extracts = list(re.finditer(r"TEXTRACT\([^;]+\);", cpp))
        between_v_halves = cpp[extracts[1].end() : extracts[2].start()]
        assert "pipe_barrier" not in between_v_halves
        assert "wait_flag" not in between_v_halves
        assert "set_flag" not in between_v_halves


def test_extract_into_rejects_non_dense_ancestor_through_reshape():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.AIC)
        def kernel(
            self,
            x: pl.Tensor[[32, 32], pl.BF16],
            y: pl.Tensor[[32, 32], pl.BF16],
            out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
        ):
            parent: pl.Tile[[32, 64], pl.BF16, pl.MemRef("l0a"), pl.Mem.Left] = pl.tile.create(
                [32, 64], pl.BF16, target_memory=pl.Mem.Left
            )
            # This row window has gaps between the 16-column boxes of the
            # parent's NZ layout. Reshaping must not hide the original pitch.
            window = pl.tile.slice(parent, [16, 64], [0, 0])
            reshaped = pl.tile.reshape(window, [32, 32])
            src = pl.tile.load(x, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
            dst = pl.tile.extract_into(reshaped, src, 0, 0)
            rhs_l1 = pl.tile.load(y, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
            rhs = pl.tile.extract(rhs_l1, 0, 0, [32, 32], target_memory=pl.Mem.Right)
            return pl.tile.store(pl.tile.matmul(dst, rhs), [0, 0], out)

    with pytest.raises(ValueError, match="destination must be a dense L0 slot"):
        _generate_mlir(Prog)


@pytest.mark.parametrize(
    "initialize",
    [
        "pl.tile.move(src, target_memory=pl.Mem.Left)",
        "pl.tile.extract(src, 0, 0, [32, 32], target_memory=pl.Mem.Left)",
    ],
)
def test_extract_into_accepts_native_a5_left_layout_spellings(initialize, tmp_path):
    # tile.move/creation use canonical NZ. The allocating tile.extract
    # retains its legacy logical row_major/row_major spelling, but PTOAS
    # generates the same ColMajor/RowMajor native A5 Left descriptor.
    program = pl.parse_program(f"""
@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.AIC)
    def kernel(self, x:pl.Tensor[[32,32],pl.BF16], y:pl.Tensor[[32,32],pl.BF16],
               out:pl.Out[pl.Tensor[[32,32],pl.FP32]]):
        src = pl.tile.load(x,[0,0],[32,32],target_memory=pl.Mem.Mat)
        dst = {initialize}
        dst = pl.tile.extract_into(dst,src,0,0)
        rhs_l1 = pl.tile.load(y,[0,0],[32,32],target_memory=pl.Mem.Mat)
        rhs = pl.tile.extract(rhs_l1,0,0,[32,32],target_memory=pl.Mem.Right)
        return pl.tile.store(pl.tile.matmul(dst,rhs),[0,0],out)
""")
    printed = ir.python_print(program)
    assert "pl.tile.extract_into(" in printed
    assert ", dst=" not in printed
    ir.assert_structural_equal(program, pl.parse_program(printed))
    mlir = _generate_mlir(program)
    assert "pto.textract " in mlir
    if find_ptoas_binary() is not None:
        cpp = pto_backend._compile_pto_module(mlir, "kernel", str(tmp_path))
        assert re.search(r"Tile<TileType::Left, bfloat16_t, 32, 32, BLayout::ColMajor", cpp)
        assert "TEXTRACT(" in cpp


@pytest.mark.parametrize(
    ("src_memory", "dst_memory", "dst_dtype", "message"),
    [
        (pl.Mem.Mat, pl.Mem.Mat, pl.BF16, "destination must be in Left or Right"),
        (pl.Mem.Vec, pl.Mem.Left, pl.BF16, "source must be in Mat"),
        (pl.Mem.Mat, pl.Mem.Left, pl.FP16, "dtypes must match"),
    ],
)
def test_extract_into_rejects_invalid_operands(src_memory, dst_memory, dst_dtype, message):
    src = pl.tile.create([32, 32], pl.BF16, target_memory=src_memory)
    dst = pl.tile.create([32, 32], dst_dtype, target_memory=dst_memory)
    with pytest.raises(ValueError, match=message):
        pl.tile.extract_into(dst, src, 0, 0)


def test_extract_into_rejects_unboxed_destination_layout():
    span = ir.Span.unknown()
    src = ir.Var("src", ir.TileType([32, 32], pl.BF16, memory_space=ir.MemorySpace.Mat), span)
    dst = ir.Var(
        "dst",
        ir.TileType(
            [32, 32],
            pl.BF16,
            memory_space=ir.MemorySpace.Left,
            tile_view=ir.TileView(blayout=ir.TileLayout.row_major, slayout=ir.TileLayout.none_box),
        ),
        span,
    )
    with pytest.raises(ValueError, match="native A5 boxed Left or Right TEXTRACT layout"):
        tile.extract_into(dst, src, 0, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
