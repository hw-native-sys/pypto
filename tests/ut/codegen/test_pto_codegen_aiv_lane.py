# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit AIV lane: full default lowering and real PTOAS regression coverage."""

import ast
import re
import subprocess
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir, passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import check_ptoas_version, find_ptoas_binary
from pypto.debug.torch_codegen import _handle_tpop_from_aic, _handle_tpush_to_aic
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


@pytest.fixture
def ptoas_binary():
    binary = find_ptoas_binary()
    if binary is None:
        pytest.skip("PTOAS executable is required for real assembler coverage")
    check_ptoas_version(binary)
    return binary


def _assemble(binary, source, arch, tmp_path, *, succeeds=True, level="level2"):
    path = tmp_path / "lane.pto"
    path.write_text(source)
    result = subprocess.run(
        [binary, f"--pto-arch={arch}", f"--pto-level={level}", str(path), "-o", str(tmp_path / "lane.cpp")],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if succeeds:
        assert result.returncode == 0, result.stderr
        return (tmp_path / "lane.cpp").read_text()
    assert result.returncode != 0, "PTOAS unexpectedly accepted the invalid lane contract"
    return result.stderr


@pytest.mark.parametrize("arch", ["a3", "a5"])
@pytest.mark.parametrize("split", [1, 2])
def test_migrated_a3_lane_fixture(ptoas_binary, arch, split, tmp_path):
    """Compile the upstream A3 operand example independently for A3 and A5."""
    source = (Path(__file__).parent / "fixtures/aiv_lane.pto").read_text()
    if split == 2:
        source = source.replace("split = 1", "split = 2").replace(
            "rows=8, cols=16, v_row=8, v_col=16", "rows=16, cols=8, v_row=16, v_col=8"
        )
    cpp = _assemble(ptoas_binary, source, arch, tmp_path)
    assert "get_subblockid()" in cpp
    assert re.search(r"TPUSH<[^;]+, [a-zA-Z_][a-zA-Z_0-9]*\);", cpp)
    assert re.search(r"TPOP<[^;]+, [a-zA-Z_][a-zA-Z_0-9]*\);", cpp)


@pytest.mark.parametrize("arch", ["a3", "a5"])
@pytest.mark.parametrize("invalid", ["nosplit", "dtype", "entry", "kernel"])
def test_ptoas_lane_contract_rejections(ptoas_binary, arch, invalid, tmp_path):
    source = (Path(__file__).parent / "fixtures/aiv_lane.pto").read_text()
    if invalid == "nosplit":
        source = source.replace("split = 1} aiv_subblockid", "split = 0} aiv_subblockid")
        message = "only when 'split' is 1, 2, 3, or 4"
    elif invalid == "dtype":
        source = source.replace(
            "%aiv_subblockid = pto.get_subblock_idx", "%aiv_subblockid = arith.constant 0 : i32"
        )
        message = "i64"
    elif invalid == "entry":
        # Change only the AIV producer operand, leaving the peer/init contract intact.
        start = source.index("    %vec_tile =")
        end = source.index("    %recv_tile =", start)
        mat_push = (
            source[start:end]
            .replace("loc=vec", "loc=mat")
            .replace("rows=8, cols=16, v_row=8", "rows=16, cols=16, v_row=16")
            .replace("blayout=row_major, slayout=none_box", "blayout=col_major, slayout=row_major")
        )
        source = source[:start] + mat_push + source[end:]
        message = "AIV-side vector tile"
    else:
        source = source.replace("#pto.kernel_kind<vector>", "#pto.kernel_kind<cube>")
        message = "vector"
    diagnostics = _assemble(ptoas_binary, source, arch, tmp_path, succeeds=False)
    assert message in diagnostics


def _lane_program(dynamic, explicit=True, push_split=1):
    lane_clause = ", aiv_subblockid=lane" if explicit else ""
    lane = "lane_in" if dynamic else "1"
    lane_binding = f"lane: pl.Scalar[pl.INT64] = {lane}" if explicit else ""
    lane_param = ", lane_in: pl.Scalar[pl.INT64]"
    lane_arg = "lane_in"
    return pl.parse_program(f"""
@pl.program
class LaneProgram:
    @pl.function(type=pl.FunctionType.AIC)
    def cube(self):
        v2c = pl.system.reserve_buffer(name="v2c", size=32768)
        c2v = pl.system.import_peer_buffer(name="c2v", peer_func="vector")
        pl.system.aic_initialize_pipe(
            dir_mask=3, slot_size=2048, slot_num=16, c2v_consumer_buf=c2v, v2c_consumer_buf=v2c
        )
        acc = pl.tile.create([32, 16], dtype=pl.FP32, target_memory=pl.Mem.Acc)
        pl.system.tpush_to_aiv(acc, split=1)
        mat: pl.Tile[[32, 16], pl.FP32, pl.Mem.Mat] = pl.system.tpop_from_aiv(split=1)
        pl.system.tfree_to_aiv(mat)

    @pl.function(type=pl.FunctionType.AIV)
    def vector(self{lane_param}):
        c2v = pl.system.reserve_buffer(name="c2v", size=32768)
        v2c = pl.system.import_peer_buffer(name="v2c", peer_func="cube")
        pl.system.aiv_initialize_pipe(
            dir_mask=3, slot_size=2048, slot_num=16, c2v_consumer_buf=c2v, v2c_consumer_buf=v2c
        )
        {lane_binding}
        vec: pl.Tile[[32, 16], pl.FP32, pl.Mem.Vec] = pl.system.tpop_from_aic(split=1{lane_clause})
        pl.system.tpush_to_aic(vec, split={push_split}{lane_clause})
        pl.system.tfree_to_aic(vec)

    @pl.function(type=pl.FunctionType.Group)
    def group(self{lane_param}):
        self.cube()
        self.vector({lane_arg})

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self{lane_param}):
        self.group({lane_arg})
""")


@pytest.mark.parametrize(("target", "arch"), [(BackendType.Ascend910B, "a3"), (BackendType.Ascend950, "a5")])
@pytest.mark.parametrize("lane_kind", ["static", "dynamic", "omit"])
@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_default_pipeline_keeps_lane_and_compiles(target, arch, lane_kind, planner, ptoas_binary, tmp_path):
    """Default Buffer selection and the A5 layout adapter preserve the scalar operand."""
    backend.reset_for_testing()
    backend.set_backend_type(target)
    with passes.PassContext([], memory_planner=planner):
        before = _lane_program(lane_kind == "dynamic", explicit=lane_kind != "omit")
        expected_count = 0 if lane_kind == "omit" else 2
        assert before.as_python().count("aiv_subblockid=") == expected_count
        lowered = PassManager(OptimizationStrategy.Default).run_passes(before)
        # Cross-core kernels stay on the supported tile path in the default pipeline.
        # Storage-bound calls use the lossless dump spelling on current upstream;
        # inspect operands rather than counting keywords in that internal form.
        lanes = []

        class CollectLanes(ir.IRVisitor):
            def visit_call(self, op):
                if op.op.name == ir.get_op("tile.tpush_to_aic").name and len(op.args) == 2:
                    lanes.append(op.args[1])
                elif op.op.name == ir.get_op("tile.tpop_from_aic").name and len(op.args) == 1:
                    lanes.append(op.args[0])
                super().visit_call(op)

        CollectLanes().visit_program(lowered)
        assert len(lanes) == expected_count
        assert all(isinstance(lane.type, ir.ScalarType) and lane.type.dtype == pl.INT64 for lane in lanes)
        printed = lowered.as_python()
        ir.assert_structural_equal(lowered, pl.parse_program(printed))
        members = [f for f in lowered.functions.values() if ir.is_incore_type(f.func_type)]
        emit_addr = planner == passes.MemoryPlanner.PYPTO
        source = codegen.PTOCodegen().generate(
            ir.Program(members, "lane", lowered.span), emit_tile_addr=emit_addr
        )
        assert source.count(" aiv_subblockid(") == expected_count
        if lane_kind == "static":
            assert re.search(r"arith.constant 1 : i64", source)
        elif lane_kind == "dynamic":
            assert re.search(r"%[\w_]+: i64", source)
        level = "level3" if emit_addr else "level2"
        cpp = _assemble(ptoas_binary, source, arch, tmp_path, level=level)
        assert "TPUSH<" in cpp and "TPOP<" in cpp


@pytest.mark.parametrize("arch", ["a3", "a5"])
def test_l1_to_ub_tmov_stays_rejected(ptoas_binary, arch, tmp_path):
    """A Mat-to-Vec memory-graph edge alone cannot establish a legal TMOV."""
    source = """
module {
  func.func @l1_to_ub() {
    %src = pto.alloc_tile : !pto.tile_buf<mat, 16x16xf32, blayout=col_major, slayout=row_major>
    %dst = pto.alloc_tile : !pto.tile_buf<vec, 16x16xf32>
    pto.tmov ins(%src : !pto.tile_buf<mat, 16x16xf32, blayout=col_major, slayout=row_major>)
             outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
    return
  }
}
"""
    diagnostics = _assemble(ptoas_binary, source, arch, tmp_path, succeeds=False)
    assert "supported tmov address-space pair" in diagnostics


@pytest.mark.parametrize("arch", ["a3", "a5"])
def test_l1_to_ub_pipe_stays_rejected(ptoas_binary, arch, tmp_path):
    """An Acc-to-Vec FIFO cannot be reused as a Mat-to-Vec producer pipe."""
    source = (Path(__file__).parent / "fixtures/aiv_lane.pto").read_text()
    source = source.replace("loc=acc,", "loc=mat,").replace("fractal=1024", "fractal=512")
    diagnostics = _assemble(ptoas_binary, source, arch, tmp_path, succeeds=False)
    assert "tile type must map to a supported producer pipe" in diagnostics


@pytest.mark.parametrize("arch", ["a3", "a5"])
@pytest.mark.parametrize("split", [3, 4])
def test_odd_c2v_pop_with_lane_and_valid_operands(ptoas_binary, arch, split, tmp_path):
    """The i64 lane coexists with both index valid-shape operands."""
    axis_name = "up_down" if split == 3 else "left_right"
    source = (Path(__file__).parent / f"fixtures/aiv_lane_odd_{axis_name}.pto").read_text()
    cpp = _assemble(ptoas_binary, source, arch, tmp_path)
    axis = "UP_DOWN" if split == 3 else "LEFT_RIGHT"
    assert f"TileSplitAxis::TILE_{axis}_ODD" in cpp


@pytest.mark.parametrize("arch", ["a3", "a5"])
def test_globaltensor_lane_stays_rejected(ptoas_binary, arch, tmp_path):
    source = """
module {
  func.func @vector(%gm: !pto.ptr<f32>, %lane: i64)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %slots = pto.make_tensor_view %gm,
      shape = [%c16, %c16], strides = [%c16, %c1] : !pto.tensor_view<16x16xf32>
    pto.aiv_initialize_pipe {dir_mask = 1, slot_size = 1024}
      (gm_slot_tensor = %slots : !pto.tensor_view<16x16xf32>)
    %entry = pto.tpop_from_aic {split = 1} aiv_subblockid(%lane)
      -> !pto.tensor_view<16x16xf32>
    return
  }
}
"""
    diagnostics = _assemble(ptoas_binary, source, arch, tmp_path, succeeds=False)
    assert "does not support 'aiv_subblockid' for !pto.tensor_view" in diagnostics


@pytest.mark.parametrize("target", [BackendType.Ascend910B, BackendType.Ascend950])
@pytest.mark.parametrize("split", [3, 4])
def test_odd_v2c_pypto_restriction_is_preserved(target, split):
    backend.reset_for_testing()
    backend.set_backend_type(target)
    lowered = PassManager(OptimizationStrategy.Default).run_passes(_lane_program(False))

    class SetOddPush(ir.IRMutator):
        def visit_call(self, op):
            if op.op.name == ir.get_op("tile.tpush_to_aic").name:
                kwargs = dict(op.kwargs)
                kwargs["split"] = split
                return ir.Call(op.op, op.args, kwargs, op.attrs, op.type, op.span)
            return super().visit_call(op)

    # Exercise the emitter restriction after standalone split normalization,
    # which otherwise replaces the authored split code with its derived one.
    rejected = SetOddPush().visit_program(lowered)
    members = [f for f in rejected.functions.values() if ir.is_incore_type(f.func_type)]
    with pytest.raises(ValueError, match="no odd split"):
        codegen.PTOCodegen().generate(ir.Program(members, "lane", rejected.span))


@pytest.mark.parametrize("target", [BackendType.Ascend910B, BackendType.Ascend950])
def test_outlining_and_auto_core_placement_keep_lane(target):
    """A captured dynamic lane is retained exclusively in the AIV kernel."""
    backend.reset_for_testing()
    backend.set_backend_type(target)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIC)
        def cube(self):
            v2c = pl.system.reserve_buffer(name="v2c", size=32768)
            c2v = pl.system.import_peer_buffer(name="c2v", peer_func="main_incore_0")
            pl.system.aic_initialize_pipe(
                dir_mask=3, slot_size=2048, slot_num=16, c2v_consumer_buf=c2v, v2c_consumer_buf=v2c
            )
            acc = pl.tile.create([32, 16], dtype=pl.FP32, target_memory=pl.Mem.Acc)
            pl.system.tpush_to_aiv(acc, split=1)
            received: pl.Tile[[32, 16], pl.FP32, pl.Mem.Mat] = pl.system.tpop_from_aiv(split=1)
            pl.system.tfree_to_aiv(received)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[16, 16], pl.FP32], lane: pl.Scalar[pl.INT64]):
            with pl.at(level=pl.Level.CORE_GROUP):
                c2v = pl.system.reserve_buffer(name="c2v", size=32768)
                v2c = pl.system.import_peer_buffer(name="v2c", peer_func="cube")
                pl.system.aiv_initialize_pipe(
                    dir_mask=3, slot_size=2048, slot_num=16, c2v_consumer_buf=c2v, v2c_consumer_buf=v2c
                )
                tile = pl.load(x, [0, 0], [16, 16])
                received: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.system.tpop_from_aic(
                    split=1, aiv_subblockid=lane
                )
                pl.system.tpush_to_aic(tile, split=1, aiv_subblockid=lane)
                pl.system.tfree_to_aic(received)

    outlined = passes.outline_incore_scopes()(Before)
    inferred = passes.infer_tile_memory_space()(passes.convert_to_ssa()(outlined))
    after = passes.expand_mixed_kernel()(inferred)
    ir.assert_structural_equal(after, pl.parse_program(after.as_python()))
    aiv_members = [f for f in after.functions.values() if f.func_type == ir.FunctionType.AIV]
    assert len(aiv_members) == 1
    vector = aiv_members[0]
    assert vector.as_python().count("aiv_subblockid=lane") == 2
    assert any(isinstance(p.type, ir.ScalarType) and p.type.dtype == pl.INT64 for p in vector.params)
    for function in after.functions.values():
        if function.func_type == ir.FunctionType.AIC:
            assert "aiv_subblockid=" not in function.as_python()


@pytest.mark.parametrize("arch", ["a3", "a5"])
def test_l1_gm_store_architecture_boundary(ptoas_binary, arch, tmp_path):
    """A3's Mat store primitive is not proof of an A5 L1-to-GM route."""
    source = """
module {
  func.func @l1_to_gm(%dst: !pto.partition_tensor_view<16x16xf32>) {
    %src = pto.alloc_tile : !pto.tile_buf<mat, 16x16xf32>
    pto.tstore ins(%src : !pto.tile_buf<mat, 16x16xf32>)
               outs(%dst : !pto.partition_tensor_view<16x16xf32>)
    return
  }
}
"""
    output = _assemble(ptoas_binary, source, arch, tmp_path, succeeds=arch == "a3")
    if arch == "a3":
        assert "TSTORE(" in output
    else:
        assert "expects A5 tstore src to use loc=vec or loc=acc" in output


@pytest.mark.parametrize("push", [False, True])
@pytest.mark.parametrize("pipe_id", [0, 3])
def test_reference_emission_preserves_pipe_id_and_lane(push, pipe_id):
    """Emission-only check: the lane must not occupy the existing pipe-id slot."""
    if push:
        source = _handle_tpush_to_aic(["tile", "lane"], {"split": 1, "id": pipe_id})
    else:
        source = _handle_tpop_from_aic(["lane"], {"split": 1, "id": pipe_id})
    expr = ast.parse(source, mode="eval").body
    assert isinstance(expr, ast.Call)
    kwargs = {kw.arg: ast.unparse(kw.value) for kw in expr.keywords}
    assert kwargs["aiv_subblockid"] == "lane"
    if pipe_id:
        assert kwargs["pipe_id"] == str(pipe_id)
    else:
        assert "pipe_id" not in kwargs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
