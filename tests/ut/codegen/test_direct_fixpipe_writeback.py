# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import subprocess
from textwrap import indent

import pypto
import pypto.language as pl
import pytest
from pypto import backend, codegen, ir
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.language.diagnostics import InvalidOperationError
from pypto.pypto_core import passes


@pytest.fixture(autouse=True)
def a5_backend():
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend950)
    yield
    backend.reset_for_testing()


def lower(program):
    return PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)


def generated_mlir(program, *, planner=passes.MemoryPlanner.PYPTO):
    with passes.PassContext([], memory_planner=planner):
        lowered = lower(program)
    generated = codegen.PTOCodegen().generate(lowered, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    return generated if isinstance(generated, str) else "\n".join(generated.values())


def compile_native(tmp_path, mlir, *, output_name="writeback.cpp", planner=passes.MemoryPlanner.PYPTO):
    ptoas = find_ptoas_binary()
    if ptoas is None:
        pytest.skip("PTOAS is unavailable")
    pto = tmp_path / "writeback.pto"
    output = tmp_path / output_name
    pto.write_text(mlir)
    level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
    subprocess.run(
        [str(ptoas), "--pto-arch=a5", f"--pto-level={level}", str(pto), "-o", str(output)],
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
            pl.tpush_to_aiv(acc, split=1, dst=channel, offset=[row, 64])
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
            pl.system.tpush_to_aiv(acc, split=1, dst=score, st_phase=writeback_phase)
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


@pytest.mark.parametrize("lane, final", [(0, False), (1, False), (1, True)])
def test_direct_bf16_writeback_to_selected_aiv(tmp_path, lane, final):
    phase = ", st_phase=pl.STPhase.Final" if final else ""
    matmul_phase = ", acc_phase=pl.AccPhase.Partial" if final else ""
    accumulate = "acc = pl.tile.matmul_acc(acc, lhs, rhs, acc_phase=pl.AccPhase.Final)" if final else ""
    program = pl.parse_program(f"""
import pypto.language as pl
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIC)
    def kernel(self, x: pl.Tensor[[96, 64], pl.BF16], y: pl.Tensor[[64, 256], pl.BF16]):
        lhs = pl.tile.load(x, [0, 0], [96, 64], target_memory=pl.Mem.Mat)
        rhs = pl.tile.load(y, [0, 0], [64, 256], target_memory=pl.Mem.Mat)
        acc = pl.tile.matmul(lhs, rhs{matmul_phase})
        {accumulate}
        mailbox = pl.tile.create([96, 256], pl.BF16, target_memory=pl.Mem.Vec)
        pl.tpush_to_aiv(acc, split=0, dst=mailbox, aiv_subblockid={lane}{phase})
""")
    printed = ir.python_print(program)
    assert f"aiv_subblockid={lane}" in printed
    ir.assert_structural_equal(program, pl.parse_program(printed))

    mlir = generated_mlir(program)
    transfers = [line for line in mlir.splitlines() if "accToVecMode" in line]
    assert len(transfers) == 1
    move = transfers[0]
    assert "pto.tmov " in move and f"#pto.acc_to_vec_mode<single_mode_vec{lane}>" in move
    assert "loc=acc, dtype=f32, rows=96, cols=256" in move
    assert "loc=vec, dtype=bf16, rows=96, cols=256" in move
    assert ("#pto<st_phase final>" in move) == final
    assert "pto.tpush" not in mlir and "pto.tinsert " not in mlir
    cpp = compile_native(tmp_path, mlir).read_text()
    assert f"AccToVecMode::SingleModeVec{lane}" in cpp
    assert "bfloat16_t" in cpp
    assert ("TMOV<STPhase::Final" in cpp) == final


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize(
    "pool_shape, slice_shape, offset, drop_dims, contiguous",
    [
        pytest.param([1, 16384], [1, 4096], [0, 4096], [], True, id="flat_pool"),
        pytest.param([4, 8192], [1, 4096], [2, 1024], [], True, id="row_window"),
        pytest.param([4, 8192], [1, 4096], [2, 1024], [0], True, id="rank_reduced_row"),
        pytest.param([4096, 16], [4096, 1], [0, 0], [1], False, id="rank_reduced_column"),
    ],
)
def test_direct_bf16_writeback_checks_reshaped_slice_pitch(
    tmp_path, planner, pool_shape, slice_shape, offset, drop_dims, contiguous
):
    annotation = (
        f': pl.Tile[{pool_shape}, pl.BF16, pl.MemRef("ub"), pl.Mem.Vec]'
        if planner == passes.MemoryPlanner.PYPTO
        else ""
    )
    drop = f", drop_dims={drop_dims}" if drop_dims else ""
    program = pl.parse_program(f"""
import pypto.language as pl
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIC)
    def kernel(self, x: pl.Tensor[[32, 64], pl.BF16], y: pl.Tensor[[64, 128], pl.BF16]):
        lhs = pl.tile.load(x, [0, 0], [32, 64], target_memory=pl.Mem.Mat)
        rhs = pl.tile.load(y, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        acc = pl.tile.matmul(lhs, rhs)
        pool{annotation} = pl.tile.create({pool_shape}, pl.BF16, target_memory=pl.Mem.Vec)
        window = pl.tile.slice(pool, {slice_shape}, {offset}{drop})
        mailbox = pl.tile.reshape(window, [32, 128])
        pl.tpush_to_aiv(acc, split=0, dst=mailbox, aiv_subblockid=0)
""")
    if not contiguous:
        with pytest.raises(ValueError, match="requires a contiguous destination"):
            generated_mlir(program, planner=planner)
        return
    mlir = generated_mlir(program, planner=planner)
    assert "pto.textract " not in mlir
    cpp = compile_native(tmp_path, mlir, planner=planner).read_text()
    assert "AccToVecMode::SingleModeVec0" in cpp


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize("cropped", [False, True], ids=["full_width_rows", "column_crop"])
@pytest.mark.parametrize("binding", ["", "_unused = "], ids=["statement", "assignment"])
@pytest.mark.parametrize(
    "columns, build_mailbox",
    [
        pytest.param(128, "mailbox = pl.tile.slice(parent, [32, 128], [32, 0])", id="slice"),
        pytest.param(128, "mailbox = pl.tile.slice(parent, [32, 128], [row, 0])", id="dynamic_row_slice"),
        pytest.param(
            256,
            "window = pl.tile.slice(parent, [16, 256], [0, 0])\nmailbox = pl.reshape(window, [32, 128])",
            id="reshape",
        ),
        pytest.param(
            128,
            "if choose_first:\n"
            "    mailbox = pl.tile.slice(parent, [32, 128], [0, 0])\n"
            "else:\n"
            "    mailbox = pl.tile.slice(parent, [32, 128], [32, 0])",
            id="if_result",
        ),
    ],
)
def test_direct_bf16_writeback_checks_mailbox_pitch(
    tmp_path, planner, cropped, binding, columns, build_mailbox
):
    parent_cols = columns * (2 if cropped else 1)
    program = pl.parse_program(f"""
import pypto.language as pl
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIC)
    def kernel(self, x: pl.Tensor[[32, 64], pl.BF16], y: pl.Tensor[[64, 128], pl.BF16],
               choose_first: pl.Scalar[pl.BOOL], row: pl.Scalar[pl.INDEX]):
        lhs = pl.tile.load(x, [0, 0], [32, 64], target_memory=pl.Mem.Mat)
        rhs = pl.tile.load(y, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        acc = pl.tile.matmul(lhs, rhs)
        parent = pl.tile.create([64, {parent_cols}], pl.BF16, target_memory=pl.Mem.Vec)
{indent(build_mailbox, "        ")}
        {binding}pl.tpush_to_aiv(acc, split=0, dst=mailbox, aiv_subblockid=0)
""")
    if cropped:
        with pytest.raises(ValueError, match="requires a contiguous destination"):
            generated_mlir(program, planner=planner)
    else:
        mlir = generated_mlir(program, planner=planner)
        assert "pto.textract " not in mlir
        cpp = compile_native(tmp_path, mlir, planner=planner).read_text()
        assert "AccToVecMode::SingleModeVec0" in cpp


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize(
    "parent_cols, build_mailbox",
    [
        pytest.param(128, "mailbox = pl.tile.slice(parent, [32, 128], [32, col])", id="slice"),
        pytest.param(
            128,
            "mailbox = pl.tile.slice(parent, [32, 128], [32, col], valid_shape=[32, 128])",
            id="explicit_valid_shape",
        ),
        pytest.param(
            128,
            "window = pl.tile.slice(parent, [32, 128], [32, col])\nmailbox = window",
            id="alias",
        ),
        pytest.param(
            256,
            "window = pl.tile.slice(parent, [16, 256], [0, col])\nmailbox = pl.reshape(window, [32, 128])",
            id="reshape",
        ),
        pytest.param(
            128,
            "if choose_first:\n"
            "    mailbox = pl.tile.slice(parent, [32, 128], [0, 0])\n"
            "else:\n"
            "    mailbox = pl.tile.slice(parent, [32, 128], [32, col])",
            id="if_result",
        ),
    ],
)
def test_direct_bf16_writeback_rejects_unproven_mailbox_alignment(planner, parent_cols, build_mailbox):
    # The runtime offset may be zero, but its alignment cannot be proved.
    # Extracting the window would make FixPipe write into a different allocation.
    program = pl.parse_program(f"""
import pypto.language as pl
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIC)
    def kernel(self, x: pl.Tensor[[32, 64], pl.BF16], y: pl.Tensor[[64, 128], pl.BF16],
               col: pl.Scalar[pl.INDEX], choose_first: pl.Scalar[pl.BOOL]):
        lhs = pl.tile.load(x, [0, 0], [32, 64], target_memory=pl.Mem.Mat)
        rhs = pl.tile.load(y, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        acc = pl.tile.matmul(lhs, rhs)
        parent = pl.tile.create([64, {parent_cols}], pl.BF16, target_memory=pl.Mem.Vec)
{indent(build_mailbox, "        ")}
        pl.tpush_to_aiv(acc, split=0, dst=mailbox, aiv_subblockid=0)
""")
    with pytest.raises(ValueError, match="tpush_to_aiv.*statically 32-byte aligned"):
        generated_mlir(program, planner=planner)


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
        pl.tile.tpush_to_aiv(acc,split=1,dst=channel,offset=[32,64],st_phase=pl.STPhase.Final)
""")


def test_unsupported_writeback_split_rejected():
    with pytest.raises(InvalidOperationError, match="requires split=1 or split=0"):
        pl.parse_program("""
import pypto.language as pl
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIC)
    def kernel(self):
        acc = pl.tile.create([64,64],pl.FP32,target_memory=pl.Mem.Acc)
        channel = pl.tile.create([64,256],pl.FP32,target_memory=pl.Mem.Vec)
        pl.tile.tpush_to_aiv(acc,split=2,dst=channel)
""")


@pytest.mark.parametrize(
    "dtype, dst_shape, options, diagnostic",
    [
        ("BF16", "[64, 64]", "split=0", "requires aiv_subblockid 0 or 1"),
        ("BF16", "[64, 64]", "split=0, aiv_subblockid=2", "aiv_subblockid must be 0 or 1"),
        ("BF16", "[64, 64]", "split=0, aiv_subblockid=True", "compile-time Python int"),
        ("BF16", "[64, 64]", "split=0, aiv_subblockid=lane", "compile-time Python int"),
        ("FP32", "[64, 64]", "split=0, aiv_subblockid=0", "requires a BF16 destination"),
        ("BF16", "[32, 64]", "split=1", "requires a FP32 destination"),
        ("FP32", "[32, 64]", "split=1, aiv_subblockid=0", "aiv_subblockid requires split=0"),
        ("BF16", "[96, 64]", "split=0, aiv_subblockid=0", "requires a complete destination"),
        ("BF16", "[96, 64]", "split=0, aiv_subblockid=1, offset=[32, 0]", "requires a complete destination"),
    ],
)
def test_invalid_direct_bf16_writeback(dtype, dst_shape, options, diagnostic):
    with pytest.raises(InvalidOperationError, match=diagnostic):
        pl.parse_program(f"""
import pypto.language as pl
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIC)
    def kernel(self, lane: pl.Scalar[pl.INT64]):
        acc = pl.tile.create([64, 64], pl.FP32, target_memory=pl.Mem.Acc)
        mailbox = pl.tile.create({dst_shape}, pl.{dtype}, target_memory=pl.Mem.Vec)
        pl.tile.tpush_to_aiv(acc, dst=mailbox, {options})
""")


def test_public_tpush_roundtrips_direct_mode():
    program = qk_program(pl.STPhase.Final, pl.AccPhase.Final)
    printed = ir.python_print(program)
    # The internal direct operation is not a separately exposed DSL API.
    assert ".tpush_to_aiv(" in printed and "dst=" in printed
    assert "tpush_to_aiv_direct(" not in printed
    ir.assert_structural_equal(program, pl.parse_program(printed))


def test_legacy_fifo_push_keeps_one_operand_and_original_kwargs():
    span = ir.Span.unknown()
    acc = ir.Var("acc", ir.TileType([64, 64], pl.FP32, memory_space=pl.Mem.Acc), span)
    call = ir.op.tile.tpush_to_aiv(acc, split=1, id=3, lane_stride=32)
    assert call.op.name == ir.get_op("tile.tpush_to_aiv").name
    assert len(call.args) == 1
    assert call.kwargs == {"split": 1, "id": 3, "lane_stride": 32}


@pytest.mark.parametrize("kwargs", [{"id": 1}, {"lane_stride": 32}])
def test_direct_push_rejects_fifo_controls(kwargs):
    span = ir.Span.unknown()
    acc = ir.Var("acc", ir.TileType([64, 64], pl.FP32, memory_space=pl.Mem.Acc), span)
    dst = ir.Var("dst", ir.TileType([32, 64], pl.FP32, memory_space=pl.Mem.Vec), span)
    with pytest.raises(ValueError, match="cannot use FIFO id or lane_stride"):
        ir.op.tile.tpush_to_aiv(acc, split=1, dst=dst, **kwargs)


@pytest.mark.parametrize("kwargs", [{"offset": [0, 0]}, {"st_phase": pl.STPhase.Final}])
def test_fifo_push_rejects_direct_controls_without_dst(kwargs):
    span = ir.Span.unknown()
    acc = ir.Var("acc", ir.TileType([64, 64], pl.FP32, memory_space=pl.Mem.Acc), span)
    with pytest.raises(ValueError, match="require an explicit dst"):
        ir.op.tile.tpush_to_aiv(acc, split=1, **kwargs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
