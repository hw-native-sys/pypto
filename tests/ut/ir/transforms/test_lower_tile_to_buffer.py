# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The public Tile pipeline ends in explicit storage, before native emission."""

from collections import Counter

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen

pytestmark = pytest.mark.usefixtures("ascend_backend")
_PLANNERS = [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]


@pl.program
class StraightLine:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[16, 32], pl.FP32],
        b: pl.Tensor[[16, 32], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
    ) -> pl.Tensor[[16, 32], pl.FP32]:
        lhs: pl.Tile[[16, 32], pl.FP32] = pl.load(a, [0, 0], [16, 32])
        rhs: pl.Tile[[16, 32], pl.FP32] = pl.load(b, [0, 0], [16, 32])
        total: pl.Tile[[16, 32], pl.FP32] = pl.add(lhs, rhs)
        product: pl.Tile[[16, 32], pl.FP32] = pl.mul(total, rhs)
        result: pl.Tensor[[16, 32], pl.FP32] = pl.store(product, [0, 0], output)
        return result


class _BufferCalls(ir.IRVisitor):
    def __init__(self, program):
        super().__init__()
        self.calls: list[ir.Call] = []
        self.allocations: list[ir.AssignStmt] = []
        self.visit_program(program)

    def visit_expr(self, expr):
        assert not isinstance(expr.type, ir.TileType)
        super().visit_expr(expr)

    def visit_call(self, call):
        assert ir.get_op_ir_stage(call.op.name) == ir.OpIRStage.Buffer
        self.calls.append(call)
        super().visit_call(call)

    def visit_assign_stmt(self, stmt):
        if isinstance(stmt.var.type, ir.BufferType):
            assert isinstance(stmt.value, ir.Call)
            assert stmt.value.op.name == ir.get_op("buffer.alloc").name
            self.allocations.append(stmt)
        super().visit_assign_stmt(stmt)


def _lower(planner, enabled=True):
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enabled):
        manager = PassManager.get_strategy(OptimizationStrategy.Default)
        result = manager.run_passes(StraightLine)
    return result, manager.pass_names


@pytest.mark.parametrize("planner", _PLANNERS)
def test_all_planners_end_in_buffer_ir_with_explicit_allocation_operands(planner):
    original = ir.serialize(StraightLine)
    result, names = _lower(planner)
    assert names[-1] == "LowerTileToBuffer"
    assert names.index("VerifyTileStorage") < names.index("LowerTileToBuffer")
    if planner != passes.MemoryPlanner.PTOAS:
        assert names.index("VerifyTileStorage") < names.index("AllocateMemoryAddr")
    kernel = result.get_function("kernel")
    assert kernel is not None and kernel.ir_stage == ir.FunctionIRStage.Buffer
    calls = _BufferCalls(result)
    counts = Counter(call.op.name for call in calls.calls)
    assert counts[ir.get_op("buffer.load").name] == 2
    assert counts[ir.get_op("buffer.store").name] == 1
    assert counts[ir.get_op("buffer.add").name] == 1
    assert counts[ir.get_op("buffer.mul").name] == 1
    assert calls.allocations
    for allocation in calls.allocations:
        assert isinstance(allocation.value, ir.Call)
        assert len(allocation.value.args) == (1 if planner == passes.MemoryPlanner.PTOAS else 2)
        assert isinstance(allocation.value.args[0], ir.MakeTuple)
        assert not allocation.value.args[0].elements
    for call in calls.calls:
        if call.op.name != ir.get_op("buffer.alloc").name:
            assert isinstance(call.type, ir.VoidType)
    # Lowering is an immutable conversion. Its type/stage/SSA facts also survive
    # the binary artifact boundary without preserving a private conversion map.
    assert ir.serialize(StraightLine) == original
    restored = ir.deserialize(ir.serialize(result))
    ir.assert_structural_equal(result, restored, enable_auto_mapping=True)
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        repeated = passes.lower_tile_to_buffer()(result)
    ir.assert_structural_equal(result, repeated, enable_auto_mapping=True)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_lowered_public_program_compiles_natively_without_codegen_storage_recovery(tmp_path, planner):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    result, _ = _lower(planner)
    calls = _BufferCalls(result)
    text = codegen.PTOCodegen().generate(result, emit_tile_addr=False, emit_source_loc=False)
    assert text.count(" = pto.alloc_tile ") == len(calls.allocations)
    assert text.count("pto.tload ins(") == 2
    assert text.count("pto.tstore ins(") == 1
    assert text.count("pto.tadd ins(") == 1
    assert text.count("pto.tmul ins(") == 1
    assert (" addr = " in text) == (planner != passes.MemoryPlanner.PTOAS)
    source = tmp_path / "lowered.pto"
    output = tmp_path / "lowered.cpp"
    source.write_text(text)
    level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
    _run_ptoas(str(source), str(output), [f"--pto-level={level}", "--pto-arch=a2"])
    assert output.is_file()


@pytest.mark.parametrize("planner", _PLANNERS)
def test_default_pipeline_stays_functional_until_the_coordinated_switch(planner):
    result, names = _lower(planner, enabled=False)
    assert "LowerTileToBuffer" not in names
    assert "VerifyTileStorage" not in names
    kernel = result.get_function("kernel")
    assert kernel is not None and kernel.ir_stage == ir.FunctionIRStage.Functional


@pytest.mark.parametrize("planner", _PLANNERS)
def test_lowering_never_falls_back_to_legacy_codegen_for_an_unimplemented_recipe(planner):
    @pl.program
    class Exponential:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            source: pl.Tensor[[16, 32], pl.FP32],
            output: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
        ) -> pl.Tensor[[16, 32], pl.FP32]:
            value: pl.Tile[[16, 32], pl.FP32] = pl.load(source, [0, 0], [16, 32])
            result: pl.Tile[[16, 32], pl.FP32] = pl.exp(value)
            out: pl.Tensor[[16, 32], pl.FP32] = pl.store(result, [0, 0], output)
            return out

    with passes.PassContext([], passes.VerificationLevel.NONE, memory_planner=planner, enable_buffer_ir=True):
        manager = PassManager.get_strategy(OptimizationStrategy.Default)
        with pytest.raises(ValueError, match="no conversion recipe for 'tile.exp'"):
            manager.run_passes(Exponential)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
