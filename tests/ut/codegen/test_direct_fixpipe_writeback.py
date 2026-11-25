# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import subprocess

import pypto
import pypto.language as pl
import pytest
from pypto import backend, codegen
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.language.diagnostics import InvalidOperationError


@pytest.fixture(autouse=True)
def a5_backend():
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend950)
    yield
    backend.reset_for_testing()


def lower(program):
    return PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)


def generated_mlir(program):
    lowered = lower(program)
    generated = codegen.PTOCodegen().generate(lowered)
    return generated if isinstance(generated, str) else "\n".join(generated.values())


def compile_native(tmp_path, mlir, *, output_name="writeback.cpp"):
    ptoas = find_ptoas_binary()
    if ptoas is None:
        pytest.skip("PTOAS is unavailable")
    pto = tmp_path / "writeback.pto"
    output = tmp_path / output_name
    pto.write_text(mlir)
    subprocess.run(
        [str(ptoas), "--pto-arch=a5", "--pto-level=level3", str(pto), "-o", str(output)],
        check=True,
        timeout=60,
        capture_output=True,
        text=True,
    )
    return output


def test_direct_pv_writeback(tmp_path):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIC)
        def kernel(
            self,
            p: pl.Tensor[[64, 128], pl.BF16],
            v: pl.Tensor[[128, 64], pl.BF16],
            row: pl.Scalar[pl.INDEX],
            group: pl.Scalar[pl.INDEX],
        ):
            a = pl.tile.load(p, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
            b = pl.tile.load(v, [0, 0], [128, 64], target_memory=pl.Mem.Mat)
            acc = pl.tile.matmul(a, b)
            # The two AIVs each receive 32 rows at local offset 0 or 32, in
            # one 256-column channel. A compact 64-column scratch is not an equivalent destination.
            pool: pl.Tile[[1, 61440], pl.FP32, pl.MemRef("ub"), pl.Mem.Vec] = pl.tile.create(
                [1, 61440], pl.FP32, target_memory=pl.Mem.Vec
            )
            flat = pl.tile.slice(pool, [1, 16384], [0, group * 16384])
            channel = pl.tile.reshape(flat, [64, 256])
            pl.tile.writeback_to_vec(channel, acc, [row, 64], split=1)
            pl.system.set_intra_block(12, pipe=pl.PipeType.FIX)

    mlir = generated_mlir(Program)
    insert = next(line for line in mlir.splitlines() if "pto.tinsert " in line)
    assert "rows=64, cols=64" in insert and "rows=64, cols=256" in insert
    assert "#pto.acc_to_vec_mode<dual_mode_split_m>" in insert
    assert "pto.tpush" not in mlir
    # The placement is explicitly controlled by the caller (row 0 or 32
    # in the reference). Do not insert a scalar offset clamp.
    assert "arith.maxsi" not in mlir
    cpp = compile_native(tmp_path, mlir).read_text()
    assert "TINSERT<" in cpp and "AccToVecMode::DualModeSplitM" in cpp


def qk_program(writeback_phase, producer_phase):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIC)
        def kernel(self, x: pl.Tensor[[64, 64], pl.BF16], y: pl.Tensor[[64, 128], pl.BF16]):
            a = pl.tile.load(x, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
            b = pl.tile.load(y, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
            acc = pl.tile.matmul(a, b, acc_phase=pl.AccPhase.Partial)
            acc = pl.tile.matmul_acc(acc, a, b, acc_phase=producer_phase)
            pool: pl.Tile[[1, 61440], pl.FP32, pl.MemRef("ub"), pl.Mem.Vec] = pl.tile.create(
                [1, 61440], pl.FP32, target_memory=pl.Mem.Vec
            )
            flat = pl.tile.slice(pool, [1, 4096], [0, 32768])
            score = pl.tile.reshape(flat, [32, 128])
            pl.tile.writeback_to_vec(score, acc, [0, 0], split=1, st_phase=writeback_phase)
            pl.system.set_intra_block(12, pipe=pl.PipeType.FIX)

    return Program


def test_direct_final_qk_writeback(tmp_path):
    mlir = generated_mlir(qk_program(pl.STPhase.Final, pl.AccPhase.Final))
    assert "#pto<acc_phase final>" in mlir
    move = next(line for line in mlir.splitlines() if "#pto<st_phase final>" in line)
    assert "pto.tmov " in move
    assert "rows=64, cols=128" in move and "rows=32, cols=128" in move
    assert "#pto.acc_to_vec_mode<dual_mode_split_m>" in move
    assert "pto.tinsert " not in mlir and "pto.tpush" not in mlir
    cpp = compile_native(tmp_path, mlir).read_text()
    assert "TMOV<STPhase::Final" in cpp and "AccToVecMode::DualModeSplitM" in cpp


@pytest.mark.parametrize(
    "store_phase, producer_phase",
    [
        (pl.STPhase.Final, pl.AccPhase.Unspecified),
        (pl.STPhase.Unspecified, pl.AccPhase.Final),
    ],
)
def test_writeback_preserves_unit_flag_check(store_phase, producer_phase):
    with pytest.raises(pypto.Error, match="AccStorePhaseValid"):
        lower(qk_program(store_phase, producer_phase))


def remote_program(marked, read_target="score"):
    # No local tfill/tload: the fixed-address score region is initialized by
    # the peer AIC's FIXPIPE, after a manually managed cross-core event.
    return pl.parse_program(f"""
import pypto.language as pl
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, out:pl.Out[pl.Tensor[[1,64],pl.FP32]]):
        ub:pl.Tile[[1,61440],pl.FP32,pl.MemRef("ub"),pl.Mem.Vec]=pl.tile.create([1,61440],pl.FP32,target_memory=pl.Mem.Vec)
        score:pl.Tile[[1,4096],pl.FP32,pl.Mem.Vec,pl.TileView(valid_shape=[1,4096],stride=[4096,1])]=pl.tile.slice(ub,[1,4096],[0,32768])
        dst:pl.Tile[[1,64],pl.FP32,pl.Mem.Vec,pl.TileView(valid_shape=[1,64],stride=[64,1])]=pl.tile.slice(ub,[1,64],[0,61376])
        neighbor:pl.Tile[[1,64],pl.FP32,pl.Mem.Vec,pl.TileView(valid_shape=[1,64],stride=[64,1])]=pl.tile.slice(ub,[1,64],[0,36864])
        pl.system.wait_intra_block(12,pipe=pl.PipeType.V)
        {"pl.tile.mark_remote_written(score)" if marked else ""}
        with pl.reg_scope():
            x=pl.reg.load({read_target},0,lanes=64)
            dst=pl.reg.store(dst,0,x)
        return pl.tile.store(dst,[0,0],out)
""")


def test_remote_initialized_by_peer(tmp_path):
    mlir = generated_mlir(remote_program(True))
    assert "pto.wait_intra_block" in mlir
    assert "mark_remote_written" not in mlir
    assert "pto.tload " not in mlir
    assert "pto.tfill " not in mlir
    # PTOAS's VF path emits an object rather than C++ source.
    assert compile_native(tmp_path, mlir, output_name="writeback.o").stat().st_size > 0


def test_unmarked_remote_load_still_rejected():
    with pytest.raises(ValueError, match="reg.load reads an uninitialized interval"):
        lower(remote_program(False))


def test_remote_annotation_does_not_initialize_neighbor():
    with pytest.raises(ValueError, match="reg.load reads an uninitialized interval"):
        lower(remote_program(True, "neighbor"))


def test_final_partial_writeback_rejected_at_type_deduction():
    with pytest.raises(
        InvalidOperationError, match="Final writeback requires the complete split-M destination"
    ):
        pl.parse_program("""
import pypto.language as pl
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIC)
    def kernel(self):
        acc = pl.tile.create([64,64],pl.FP32,target_memory=pl.Mem.Acc)
        channel = pl.tile.create([64,256],pl.FP32,target_memory=pl.Mem.Vec)
        pl.tile.writeback_to_vec(channel,acc,[32,64],split=1,st_phase=pl.STPhase.Final)
""")


def test_unsupported_writeback_split_rejected():
    with pytest.raises(InvalidOperationError, match="supports only split=1"):
        pl.parse_program("""
import pypto.language as pl
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIC)
    def kernel(self):
        acc = pl.tile.create([64,64],pl.FP32,target_memory=pl.Mem.Acc)
        channel = pl.tile.create([64,256],pl.FP32,target_memory=pl.Mem.Vec)
        pl.tile.writeback_to_vec(channel,acc,[0,0],split=2)
""")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
