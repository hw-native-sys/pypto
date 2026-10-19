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
from buffer_pipeline_utils import buffer_recipe_pipeline
from pypto import DataType, ir, passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.instruments import make_roundtrip_instrument
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen

from .buffer_test_utils import statements

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

    def visit_call(self, op):
        assert ir.get_op_ir_stage(op.op.name) == ir.OpIRStage.Buffer
        self.calls.append(op)
        super().visit_call(op)

    def visit_assign_stmt(self, op):
        if isinstance(op.var.type, ir.BufferType):
            assert isinstance(op.value, ir.Call)
            assert op.value.op.name == ir.get_op("buffer.alloc").name
            self.allocations.append(op)
        super().visit_assign_stmt(op)


def _lower(planner, enabled=True, program=StraightLine):
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enabled):
        manager = buffer_recipe_pipeline()
        result = manager.run(program)
    return result, manager.get_pass_names()


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
        assert allocation.span.filename == allocation.var.span.filename
        assert allocation.span.begin_line == allocation.var.span.begin_line
        assert allocation.value.span.begin_line == allocation.span.begin_line
        assert allocation.span.is_valid()
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
def test_enabled_pipeline_roundtrip_preserves_buffer_representation(planner):
    with passes.PassContext([make_roundtrip_instrument()], memory_planner=planner, enable_buffer_ir=True):
        result = buffer_recipe_pipeline().run(StraightLine)
    kernel = result.get_function("kernel")
    assert kernel is not None and kernel.ir_stage == ir.FunctionIRStage.Buffer
    assert _BufferCalls(result).allocations


def test_buffer_roundtrip_rejects_lost_stage_and_operands(monkeypatch):
    result, _ = _lower(passes.MemoryPlanner.PYPTO)
    monkeypatch.setattr("pypto.pypto_core.ir.deserialize", lambda _data: StraightLine)
    with passes.PassContext([make_roundtrip_instrument()]):
        with pytest.raises(RuntimeError, match="Binary roundtrip failed"):
            passes.lower_tile_to_buffer()(result)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_lowering_never_falls_back_to_legacy_codegen_for_an_unimplemented_recipe(planner):
    @pl.program
    class ReciprocalSquareRoot:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            source: pl.Tensor[[16, 32], pl.FP32],
            output: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
        ) -> pl.Tensor[[16, 32], pl.FP32]:
            value: pl.Tile[[16, 32], pl.FP32] = pl.load(source, [0, 0], [16, 32])
            result: pl.Tile[[16, 32], pl.FP32] = pl.rsqrt(value)
            out: pl.Tensor[[16, 32], pl.FP32] = pl.store(result, [0, 0], output)
            return out

    with passes.PassContext([], passes.VerificationLevel.NONE, memory_planner=planner, enable_buffer_ir=True):
        manager = buffer_recipe_pipeline()
        with pytest.raises(ValueError, match="no conversion recipe for 'tile.rsqrt'"):
            manager.run(ReciprocalSquareRoot)


@pl.program
class MixedBranch:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[16, 32], pl.FP32],
        b: pl.Tensor[[16, 32], pl.FP32],
        output: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
        original: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
        flag: pl.Scalar[pl.BOOL],
    ) -> tuple[pl.Tensor[[32, 64], pl.FP32], pl.Tensor[[16, 32], pl.FP32]]:
        lhs = pl.load(a, [0, 0], [16, 32])
        rhs = pl.load(b, [0, 0], [16, 32])
        if flag:
            product = pl.mul(lhs, rhs)
            selected, row, second, column = pl.yield_(product, 16, lhs, 0)
        else:
            selected, row, second, column = pl.yield_(lhs, 0, rhs, 32)
        total = pl.add(selected, second)
        stored = pl.store(total, [row, column], output)
        saved = pl.store(lhs, [0, 0], original)
        return stored, saved


@pl.program
class NestedBranch:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[16, 32], pl.FP32],
        b: pl.Tensor[[16, 32], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
        outer: pl.Scalar[pl.BOOL],
        inner: pl.Scalar[pl.BOOL],
    ) -> pl.Tensor[[16, 32], pl.FP32]:
        lhs = pl.load(a, [0, 0], [16, 32])
        rhs = pl.load(b, [0, 0], [16, 32])
        if outer:
            if inner:
                selected = pl.yield_(lhs)
            else:
                selected = pl.yield_(rhs)
            then_out = pl.store(selected, [0, 0], output)
            result = pl.yield_(then_out)
        else:
            product = pl.mul(lhs, rhs)
            else_out = pl.store(product, [0, 0], output)
            result = pl.yield_(else_out)
        return result


class _Branches:
    def __init__(self, program):
        regions = list(statements(program))
        self.branches = [region for region in regions if isinstance(region, ir.IfStmt)]
        self.yields = [region for region in regions if isinstance(region, ir.YieldStmt)]


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("program", [MixedBranch, NestedBranch], ids=["mixed", "nested"])
def test_branch_storage_becomes_explicit_writes_and_only_scalars_are_yielded(planner, program):
    before = ir.serialize(program)
    result, _ = _lower(planner, program=program)
    _BufferCalls(result)
    regions = _Branches(result)
    assert len(regions.branches) == (1 if program is MixedBranch else 2)
    for branch in regions.branches:
        assert all(isinstance(value.type, ir.ScalarType) for value in branch.return_vars)
    for statement in regions.yields:
        assert all(isinstance(value.type, ir.ScalarType) for value in statement.value)
    if program is MixedBranch:
        assert len(regions.branches[0].return_vars) == 2
        assert len(regions.yields) == 2
        for statement, expected in zip(regions.yields, [(16, 0), (0, 32)], strict=True):
            actual = []
            for value in statement.value:
                assert isinstance(value, ir.ConstInt)
                actual.append(value.value)
            assert tuple(actual) == expected
    else:
        assert not any(branch.return_vars for branch in regions.branches)
    assert ir.serialize(program) == before
    restored = ir.deserialize(ir.serialize(result))
    ir.assert_structural_equal(restored, result, enable_auto_mapping=True)
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        repeated = passes.lower_tile_to_buffer()(result)
    ir.assert_structural_equal(repeated, result, enable_auto_mapping=True)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("program", [MixedBranch, NestedBranch], ids=["mixed", "nested"])
def test_public_branch_program_compiles_with_one_native_op_per_buffer_write(tmp_path, planner, program):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    result, _ = _lower(planner, program=program)
    calls = _BufferCalls(result)
    counts = Counter(call.op.name for call in calls.calls)
    text = codegen.PTOCodegen().generate(result, emit_source_loc=False)
    assert text.count("scf.if ") == len(_Branches(result).branches)
    assert text.count(" = pto.alloc_tile ") == len(calls.allocations)
    assert text.count("pto.tmov ins(") == counts[ir.get_op("buffer.copy").name]
    assert text.count("pto.tstore ins(") == counts[ir.get_op("buffer.store").name]
    source, output = tmp_path / "branches.pto", tmp_path / "branches.cpp"
    source.write_text(text)
    level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
    _run_ptoas(str(source), str(output), [f"--pto-level={level}", "--pto-arch=a2"])
    assert output.is_file()


@pytest.mark.parametrize("planner", _PLANNERS)
def test_distinct_gm_branch_aliases_require_a_separate_recipe(planner):
    @pl.program
    class SelectGM:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[16, 32], pl.FP32],
            b: pl.Tensor[[16, 32], pl.FP32],
            output: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
            flag: pl.Scalar[pl.BOOL],
        ) -> pl.Tensor[[16, 32], pl.FP32]:
            if flag:
                source = pl.yield_(a)
            else:
                source = pl.yield_(b)
            value = pl.load(source, [0, 0], [16, 32])
            stored = pl.store(value, [0, 0], output)
            return stored

    before = ir.serialize(SelectGM)
    with pytest.raises(ValueError, match="GM branch results must alias the same parameter"):
        _lower(planner, program=SelectGM)
    assert ir.serialize(SelectGM) == before


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("same_alias", [True, False], ids=["same-gm", "different-gm"])
def test_distributed_branch_results_fail_at_lowering_boundary(planner, same_alias):
    """Distributed GM results must not survive as unsupported native region results."""
    span = ir.Span.unknown()
    dtype = ir.DistributedTensorType([16, 32], pl.FP32)
    a, b = ir.Var("a", dtype, span), ir.Var("b", dtype, span)
    flag = ir.Var("flag", ir.ScalarType(pl.BOOL), span)
    result = ir.Var("result", dtype, span)
    branch = ir.IfStmt(
        flag,
        ir.YieldStmt([a], span),
        ir.YieldStmt([a if same_alias else b], span),
        [result],
        span,
    )
    function = ir.Function(
        "kernel",
        [a, b, flag],
        [],
        ir.SeqStmts([branch, ir.ReturnStmt([], span)], span),
        span,
        type=ir.FunctionType.InCore,
    )
    program = ir.Program([function], "DistributedBranch", span)
    before = ir.serialize(program)
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        with pytest.raises(ValueError, match="distributed tensor branch results require a separate"):
            passes.lower_tile_to_buffer()(program)
    assert ir.serialize(program) == before


def _loop_program(while_loop):
    header = (
        "for (left, row, right, column, gm) in pl.while_(init_values=(lhs, 0, rhs, 0, output)):\n"
        "            pl.cond(row < count)"
        if while_loop
        else "for _i, (left, row, right, column, gm) in pl.range(0, count, "
        "init_values=(lhs, 0, rhs, 0, output)):"
    )
    return pl.parse_program(f"""
@pl.program
class LoopTransfer:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[16, 32], pl.FP32],
               b: pl.Tensor[[16, 32], pl.FP32],
               output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
               original: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
               count: pl.Scalar[pl.INDEX], flag: pl.Scalar[pl.BOOL]
               ) -> tuple[pl.Tensor[[64, 64], pl.FP32], pl.Tensor[[16, 32], pl.FP32]]:
        lhs = pl.load(a, [0, 0], [16, 32])
        rhs = pl.load(b, [0, 0], [16, 32])
        {header}
            if flag:
                selected = pl.yield_(left)
            else:
                selected = pl.yield_(right)
            stored = pl.store(selected, [row, column], gm)
            next_row = row + 1
            next_column = column + 2
            final_left, final_row, final_right, final_column, final_gm = pl.yield_(
                right, next_row, left, next_column, stored)
        combined = pl.add(final_left, final_right)
        result = pl.store(combined, [final_row, final_column], final_gm)
        saved = pl.store(lhs, [0, 0], original)
        return result, saved
""")


@pl.program
class NestedLoops:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[16, 32], pl.FP32],
        b: pl.Tensor[[16, 32], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
        count: pl.Scalar[pl.INDEX],
    ) -> pl.Tensor[[16, 32], pl.FP32]:
        lhs = pl.load(a, [0, 0], [16, 32])
        rhs = pl.load(b, [0, 0], [16, 32])
        for _i, (outer,) in pl.range(0, count, init_values=(lhs,)):
            for _j, (inner,) in pl.range(0, count, init_values=(outer,)):
                total = pl.add(inner, rhs)
                inner_result = pl.yield_(total)
            outer_result = pl.yield_(inner_result)
        stored = pl.store(outer_result, [0, 0], output)
        return stored


class _Loops:
    def __init__(self, program):
        self.loops = [
            region for region in statements(program) if isinstance(region, (ir.ForStmt, ir.WhileStmt))
        ]


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("kind", ["for", "while", "nested"])
def test_loops_remove_storage_carries_and_compile_scalar_results(tmp_path, planner, kind):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    program = NestedLoops if kind == "nested" else _loop_program(kind == "while")
    original = ir.serialize(program)
    result, _ = _lower(planner, program=program)
    calls = _BufferCalls(result)
    loops = _Loops(result).loops
    assert len(loops) == (2 if kind == "nested" else 1)
    for loop in loops:
        assert len(loop.iter_args) == len(loop.return_vars) == (0 if kind == "nested" else 2)
        assert all(isinstance(argument.type, ir.ScalarType) for argument in loop.iter_args)
        tail = loop.body.stmts[-1] if isinstance(loop.body, ir.SeqStmts) else loop.body
        assert isinstance(tail, ir.YieldStmt)
        assert len(tail.value) == len(loop.iter_args)
        assert all(isinstance(value.type, ir.ScalarType) for value in tail.value)
    assert ir.serialize(program) == original
    restored = ir.deserialize(ir.serialize(result))
    ir.assert_structural_equal(restored, result, enable_auto_mapping=True)
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        ir.assert_structural_equal(passes.lower_tile_to_buffer()(result), result)
    counts = Counter(call.op.name for call in calls.calls)
    text = codegen.PTOCodegen().generate(result, emit_source_loc=False)
    assert text.count("pto.tmov ins(") == counts[ir.get_op("buffer.copy").name]
    assert text.count("pto.tstore ins(") == counts[ir.get_op("buffer.store").name]
    assert text.count(" = pto.alloc_tile ") == len(calls.allocations)
    headers = [line for line in text.splitlines() if "scf.for " in line or "scf.while " in line]
    assert len(headers) == len(loops)
    assert all("tile_buf" not in header for header in headers)
    assert text.index("pto.alloc_tile") < text.index(headers[0])
    if kind != "nested":
        assert all(isinstance(argument.initValue, ir.ConstInt) for argument in loops[0].iter_args)
        assert "-> (index, index)" in headers[0]
        # Both scalar results are consumed as store offsets after the loop.
        assert counts[ir.get_op("buffer.store").name] == 3
        assert counts[ir.get_op("buffer.copy").name] >= 2
    source, output = tmp_path / "loops.pto", tmp_path / "loops.cpp"
    source.write_text(text)
    level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
    _run_ptoas(str(source), str(output), [f"--pto-level={level}", "--pto-arch=a2"])
    assert output.is_file()


@pytest.mark.parametrize("planner", _PLANNERS)
def test_gm_loop_alias_changes_are_diagnosed(planner):
    @pl.program
    class ChangedGM:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[16, 32], pl.FP32],
            b: pl.Tensor[[16, 32], pl.FP32],
            output: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
            count: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[16, 32], pl.FP32]:
            for _i, (_gm,) in pl.range(0, count, init_values=(a,)):
                chosen = pl.yield_(b)
            value = pl.load(chosen, [0, 0], [16, 32])
            stored = pl.store(value, [0, 0], output)
            return stored

    before = ir.serialize(ChangedGM)
    with pytest.raises(ValueError, match="GM loop results must retain their initial parameter alias"):
        _lower(planner, program=ChangedGM)
    assert ir.serialize(ChangedGM) == before


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("kind", ["for", "while"])
@pytest.mark.parametrize("same_alias", [True, False], ids=["same-gm", "different-gm"])
def test_distributed_loop_carries_fail_at_lowering_boundary(planner, kind, same_alias):
    span = ir.Span.unknown()
    dtype = ir.DistributedTensorType([16, 32], pl.FP32)
    a, b = ir.Var("a", dtype, span), ir.Var("b", dtype, span)
    flag = ir.Var("flag", ir.ScalarType(pl.BOOL), span)
    carried = ir.IterArg("carried", dtype, a, span)
    result = ir.Var("result", dtype, span)
    body = ir.YieldStmt([carried if same_alias else b], span)
    if kind == "while":
        loop = ir.WhileStmt(flag, [carried], body, [result], span)
    else:
        index = ir.Var("i", ir.ScalarType(pl.INDEX), span)
        zero, one = ir.ConstInt(0, pl.INDEX, span), ir.ConstInt(1, pl.INDEX, span)
        loop = ir.ForStmt(index, zero, one, one, [carried], body, [result], span)
    function = ir.Function(
        "kernel",
        [a, b, flag],
        [],
        ir.SeqStmts([loop, ir.ReturnStmt([], span)], span),
        span,
        type=ir.FunctionType.InCore,
    )
    program = ir.Program([function], "DistributedLoop", span)
    before = ir.serialize(program)
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        with pytest.raises(ValueError, match="distributed tensor loop carries require a separate"):
            passes.lower_tile_to_buffer()(program)
    assert ir.serialize(program) == before


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910B, BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize(
    "dtype_name,mlir_type,operation",
    [
        (dtype, mlir, operation)
        for dtype, mlir in [("FP16", "f16"), ("FP32", "f32"), ("INT32", "i32")]
        for operation in ["transfer", "add", "mul"]
    ]
    + [("BF16", "bf16", "transfer")],
)
def test_typed_transfers_and_arithmetic_keep_physical_storage(
    tmp_path, ascend_backend, planner, dtype_name, mlir_type, operation
):
    valid = [13, 24] if operation == "transfer" else [16, 32]
    computation = "result = lhs" if operation == "transfer" else f"result = pl.{operation}(lhs, lhs)"
    program = pl.parse_program(f"""
@pl.program
class TypedTransfers:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, source: pl.Tensor[[32, 64], pl.{dtype_name}],
               output: pl.Out[pl.Tensor[[32, 64], pl.{dtype_name}]]
               ) -> pl.Tensor[[32, 64], pl.{dtype_name}]:
        lhs = pl.load(source, [3, 8], [16, 32], valid_shape={valid})
        {computation}
        output = pl.store(result, [4, 32], output)
        return output
""")
    planned = []

    def before(pass_obj, candidate):
        if pass_obj.get_name() == "LowerTileToBuffer":
            planned.append(candidate)

    instrument = passes.CallbackInstrument(before_pass=before, name="CaptureTypedStorage")
    with passes.PassContext([instrument], memory_planner=planner, enable_buffer_ir=True):
        lowered = buffer_recipe_pipeline().run(program)
    assert len(planned) == 1
    dtype = getattr(DataType, dtype_name)
    tile_types = [
        statement.var.type
        for statement in statements(planned[0])
        if isinstance(statement, ir.AssignStmt) and isinstance(statement.var.type, ir.TileType)
    ]
    assert tile_types
    for tile_type in tile_types:
        assert tile_type.dtype == dtype and tile_type.memref is not None
        assert tile_type.memref.size_ == 16 * 32 * dtype.get_byte()
    restored = ir.deserialize(ir.serialize(lowered))
    assert isinstance(restored, ir.Program)
    ir.assert_structural_equal(lowered, restored, enable_auto_mapping=True)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.BufferIR)
    assert passes.PropertyVerifierRegistry.verify(properties, restored) == []
    calls = _BufferCalls(restored)
    assert calls.allocations
    for allocation in calls.allocations:
        assert isinstance(allocation.var.type, ir.BufferType)
        assert allocation.var.type.dtype == dtype
        assert allocation.var.type.shape == [16, 32]
        assert allocation.var.type.valid_shape == valid
    text = codegen.PTOCodegen().generate(restored, emit_source_loc=False)
    assert text.count(f"!pto.ptr<{mlir_type}>") >= 2
    assert text.count("pto.tload ins(") == text.count("pto.tstore ins(") == 1
    assert "pto.tcast" not in text and "pto.tmov" not in text
    if operation != "transfer":
        assert text.count(f"pto.t{operation} ins(") == 1
    assert (" addr = " in text) == (planner != passes.MemoryPlanner.PTOAS)
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    source, output = tmp_path / "typed.pto", tmp_path / "typed.cpp"
    source.write_text(text)
    arch = "a2" if ascend_backend == BackendType.Ascend910B else "a5"
    level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
    _run_ptoas(str(source), str(output), [f"--pto-arch={arch}", f"--pto-level={level}"])
    assert output.is_file()


@pytest.mark.parametrize(
    "load_hint,store_hint",
    [
        (pl.CachePolicy.NORMAL_FIRST_VICTIM, None),
        (pl.CachePolicy.NORMAL_PERSISTENT, None),
        (None, pl.CachePolicy.NORMAL_FIRST_VICTIM),
        (None, pl.CachePolicy.NORMAL_PERSISTENT),
    ],
)
@pytest.mark.parametrize("force_buffer", [False, True])
def test_buffer_selection_preserves_nondefault_cache_policy(load_hint, store_hint, force_buffer):
    @pl.program
    class Hinted:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, source: pl.Tensor[[16, 32], pl.FP32], output: pl.Out[pl.Tensor[[16, 32], pl.FP32]]
        ) -> pl.Tensor[[16, 32], pl.FP32]:
            value = pl.load(source, [0, 0], [16, 32], cache=load_hint)
            out = pl.store(value, [0, 0], output, cache=store_hint)
            return out

    if force_buffer:
        with pytest.raises(ValueError, match="non-default cache requires a Buffer transfer recipe"):
            _lower(passes.MemoryPlanner.PYPTO, program=Hinted)
    else:
        with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
            result = PassManager(OptimizationStrategy.Default).run_passes(Hinted)
        kernel = result.get_function("kernel")
        assert kernel is not None
        assert kernel.attrs["buffer_ir_selected"] is False
        assert kernel.ir_stage == ir.FunctionIRStage.Functional


@pl.program
class MixedRollout:
    @pl.function(type=pl.FunctionType.InCore)
    def supported(
        self,
        a: pl.Tensor[[16, 32], pl.FP32],
        b: pl.Tensor[[16, 32], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
    ) -> pl.Tensor[[16, 32], pl.FP32]:
        x = pl.load(a, [0, 0], [16, 32])
        y = pl.load(b, [0, 0], [16, 32])
        return pl.store(pl.mul(pl.add(x, y), y), [0, 0], out)

    @pl.function(type=pl.FunctionType.InCore)
    def unsupported(
        self, a: pl.Tensor[[16, 32], pl.FP32], out: pl.Out[pl.Tensor[[16, 1], pl.FP32]]
    ) -> pl.Tensor[[16, 1], pl.FP32]:
        x = pl.load(a, [0, 0], [16, 32])
        tmp = pl.create_tile([16, 32], pl.FP32)
        return pl.store(pl.row_sum(x, tmp_tile=tmp), [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[16, 32], pl.FP32],
        b: pl.Tensor[[16, 32], pl.FP32],
        scratch: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
        out: pl.Out[pl.Tensor[[16, 1], pl.FP32]],
    ) -> pl.Tensor[[16, 1], pl.FP32]:
        with pl.manual_scope():
            scratch, first = pl.submit(self.supported, a, b, scratch)
            out, second = pl.submit(self.unsupported, scratch, out, deps=[first])
        return out


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("enabled", [None, True, False], ids=["default", "on", "off"])
def test_default_rollout_selects_whole_kernels_and_preserves_submit(planner, enabled):
    options = {} if enabled is None else {"enable_buffer_ir": enabled}
    with passes.PassContext([], memory_planner=planner, **options):
        result = PassManager(OptimizationStrategy.Default).run_passes(MixedRollout)
    selected = result.get_function("supported")
    unsupported = result.get_function("unsupported")
    assert selected is not None and unsupported is not None
    assert selected.ir_stage == (
        ir.FunctionIRStage.Functional if enabled is False else ir.FunctionIRStage.Buffer
    )
    assert unsupported.ir_stage == ir.FunctionIRStage.Functional
    submits = [
        stmt.value
        for stmt in statements(result)
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Submit)
    ]
    assert len(submits) == 2
    assert len(submits[1].deps) == 1
    assert submits[0].op.name == "supported"
    assert submits[1].op.name == "unsupported"
    for function in (selected, unsupported):
        single = ir.Program([function], "Single", function.span)
        text = codegen.PTOCodegen().generate(single, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
        assert "pto.tstore" in text
    restored = ir.deserialize(ir.serialize(result))
    ir.assert_structural_equal(result, restored, enable_auto_mapping=True)


def test_default_rollout_without_context():
    result = PassManager(OptimizationStrategy.Default).run_passes(StraightLine)
    function = result.get_function("kernel")
    assert function is not None and function.ir_stage == ir.FunctionIRStage.Buffer


def test_selected_kernel_failure_is_not_retried_as_tile():
    seen = []

    def reject_lowering(pass_obj, program):
        if pass_obj.get_name() == "LowerTileToBuffer":
            seen.append(program.get_function("kernel").attrs["buffer_ir_selected"])
            raise RuntimeError("selected Buffer compilation failed")

    instrument = passes.CallbackInstrument(before_pass=reject_lowering, name="RejectSelectedBuffer")
    with passes.PassContext([instrument]):
        with pytest.raises(RuntimeError, match="selected Buffer compilation failed"):
            PassManager(OptimizationStrategy.Default).run_passes(StraightLine)
    assert seen == [True]


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910B, BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("planner", _PLANNERS)
def test_default_rollout_native_compilation(tmp_path, ascend_backend, planner):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    with passes.PassContext([], memory_planner=planner):
        result = PassManager(OptimizationStrategy.Default).run_passes(StraightLine)
    function = result.get_function("kernel")
    expected = (
        ir.FunctionIRStage.Buffer
        if ascend_backend == BackendType.Ascend910B
        else ir.FunctionIRStage.Functional
    )
    assert function is not None and function.ir_stage == expected
    assert function.attrs["buffer_ir_selected"] == (ascend_backend == BackendType.Ascend910B)
    source, output = tmp_path / "default.pto", tmp_path / "default.cpp"
    source.write_text(
        codegen.PTOCodegen().generate(result, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    )
    arch = "a2" if ascend_backend == BackendType.Ascend910B else "a5"
    level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
    _run_ptoas(str(source), str(output), [f"--pto-level={level}", f"--pto-arch={arch}"])
    assert output.is_file()


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("program", [MixedBranch, NestedBranch])
def test_default_control_flow_selects_buffer_and_off_retains_tile(planner, program):
    for enabled in (False, True):
        with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enabled):
            result = PassManager(OptimizationStrategy.Default).run_passes(program)
        function = result.get_function("kernel")
        assert function is not None
        assert function.ir_stage == (ir.FunctionIRStage.Buffer if enabled else ir.FunctionIRStage.Functional)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize(
    "dtype_name,valid_shape,selected",
    [
        ("FP32", [16, 32], True),
        ("FP16", [16, 32], True),
        ("BF16", [16, 32], True),
        ("INT32", [16, 32], True),
        ("FP32", [13, 24], True),
    ],
)
def test_default_rollout_selects_typed_transfers_and_static_partial_views(
    planner, dtype_name, valid_shape, selected
):
    program = pl.parse_program(f"""
@pl.program
class Transfer:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, source: pl.Tensor[[16, 32], pl.{dtype_name}],
               output: pl.Out[pl.Tensor[[16, 32], pl.{dtype_name}]]
               ) -> pl.Tensor[[16, 32], pl.{dtype_name}]:
        value = pl.load(source, [0, 0], [16, 32], valid_shape={valid_shape})
        return pl.store(value, [0, 0], output)
""")
    with passes.PassContext([], memory_planner=planner):
        result = PassManager(OptimizationStrategy.Default).run_passes(program)
    function = result.get_function("kernel")
    assert function is not None
    assert function.attrs["buffer_ir_selected"] is selected
    assert function.ir_stage == (ir.FunctionIRStage.Buffer if selected else ir.FunctionIRStage.Functional)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("op_name", ["tile.load", "tile.store", "tile.add", "tile.mul"])
def test_call_attributes_outside_lowering_contract_keep_whole_kernel_on_tile(planner, op_name):
    class AddAttribute(ir.IRMutator):
        def visit_call(self, op):
            if op.op.name == ir.get_op(op_name).name:
                return ir.Call(op.op, op.args, op.kwargs, {"unsupported_metadata": True}, op.type, op.span)
            return super().visit_call(op)

    # Stamp physical Tile calls at the selection boundary, after the frontend
    # tensor-to-tile rewrite has established the calls being classified.
    with passes.PassContext([], memory_planner=planner):
        manager = PassManager(OptimizationStrategy.Default)
        boundary = manager.get_pass_names().index("SelectBufferIRFunctions")
        program = StraightLine
        for transform in manager.passes[:boundary]:
            program = transform(program)
    program = AddAttribute().visit_program(program)
    outputs = []
    for enabled in (False, True):
        with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enabled):
            result = program
            for transform in manager.passes[boundary:]:
                result = transform(result)
        function = result.get_function("kernel")
        assert function is not None
        if enabled:
            assert function.attrs["buffer_ir_selected"] is False
        assert function.ir_stage == ir.FunctionIRStage.Functional
        outputs.append(function.body)
    ir.assert_structural_equal(*outputs, enable_auto_mapping=True)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_gm_return_view_mismatch_keeps_whole_kernel_on_tile(tmp_path, planner):
    @pl.program
    class AssembleOutput:
        @pl.function(type=pl.FunctionType.InCore)
        def compute(
            self,
            source: pl.Tensor[[4, 16], pl.FP32],
            output: pl.Out[pl.Tensor[[4, 16], pl.FP32]],
        ) -> pl.Tensor[[4, 16], pl.FP32]:
            tile = pl.load(source, [0, 0], [4, 16])
            return pl.store(tile, [0, 0], output)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            source: pl.Tensor[[4, 16], pl.FP32],
            output: pl.Out[pl.Tensor[[2, 4, 16], pl.FP32]],
        ) -> pl.Tensor[[2, 4, 16], pl.FP32]:
            for index in pl.range(2):
                chunk = pl.create_tensor([4, 16], dtype=pl.FP32)
                chunk = self.compute(source, chunk)
                output = pl.assemble(output, chunk, [index, 0, 0])
            return output

    outputs = []
    for enabled in (False, True):
        with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enabled):
            result = PassManager(OptimizationStrategy.Default).run_passes(AssembleOutput)
        function = result.get_function("compute")
        assert function is not None
        if enabled:
            assert function.attrs["buffer_ir_selected"] is False
        assert function.ir_stage == ir.FunctionIRStage.Functional
        outputs.append(function.body)
    ir.assert_structural_equal(*outputs, enable_auto_mapping=True)
    single = ir.Program([function], "Compute", function.span)
    text = codegen.PTOCodegen().generate(single, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    assert "pto.tload" in text and "pto.tstore" in text
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    source, output = tmp_path / "assemble.pto", tmp_path / "assemble.cpp"
    source.write_text(text)
    level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
    _run_ptoas(str(source), str(output), [f"--pto-level={level}", "--pto-arch=a2"])
    assert output.is_file()


def test_device_helper_callee_is_not_selected_as_a_kernel():
    prepared = StraightLine
    with passes.PassContext([]):
        for transform in PassManager(OptimizationStrategy.Default).passes:
            if transform.get_name() == "SelectBufferIRFunctions":
                break
            prepared = transform(prepared)
        kernel = prepared.get_function("kernel")
        assert kernel is not None
        call = ir.Call(ir.GlobalVar("kernel"), kernel.params, {}, kernel.return_types[0], kernel.span)
        caller = ir.Function(
            "caller",
            list(zip(kernel.params, kernel.param_directions)),
            kernel.return_types,
            ir.ReturnStmt([call], kernel.span),
            kernel.span,
            type=ir.FunctionType.InCore,
        )
        selected = passes.select_buffer_ir_functions()(ir.Program([kernel, caller], "Helpers", kernel.span))
    assert all(function.attrs["buffer_ir_selected"] is False for function in selected.functions.values())


def test_mixed_buffer_verifier_checks_unselected_function_bindings():
    with passes.PassContext([]):
        result = PassManager(OptimizationStrategy.Default).run_passes(MixedRollout)
    tile_function = result.get_function("unsupported")
    assert tile_function is not None
    undefined = ir.Var("undefined", tile_function.params[0].type, tile_function.span)
    malformed = ir.Function(
        tile_function.name,
        list(zip(tile_function.params, tile_function.param_directions)),
        tile_function.return_types,
        ir.ReturnStmt([undefined], tile_function.span),
        tile_function.span,
        type=tile_function.func_type,
        attrs=tile_function.attrs,
    )
    functions = [malformed if f.name == tile_function.name else f for f in result.functions.values()]
    mixed = ir.Program(functions, result.name, result.span)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.BufferIR)
    diagnostics = passes.PropertyVerifierRegistry.verify(properties, mixed)
    assert diagnostics
    assert any("undefined" in diagnostic.message for diagnostic in diagnostics)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
