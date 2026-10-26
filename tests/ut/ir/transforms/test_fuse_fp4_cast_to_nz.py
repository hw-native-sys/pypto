# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Typed producer-site fusion and storage regressions for native FP4 casts."""

import typing
from collections.abc import Mapping

import pypto.language as pl
import pytest
from pypto import DataType, backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.op import tile as tile_ops

SPAN = ir.Span.unknown()
NZ = {
    "target_memory": ir.MemorySpace.Vec,
    "blayout": ir.TileLayout.col_major,
    "slayout": ir.TileLayout.row_major,
}
CAST = {"target_type": DataType.FP8E4M3FN, "mode": 2, "saturation_mode": 0}
_OpKwarg = int | bool | str | float | DataType | ir.MemorySpace | ir.PadValue


class _TileViewFields(typing.TypedDict, total=False):
    valid_shape: list[int]
    stride: list[int]
    start_offset: int
    blayout: ir.TileLayout
    slayout: ir.TileLayout
    fractal: int
    pad: ir.PadValue
    compact: ir.CompactMode


def _const(n):
    return ir.ConstInt(n, DataType.INDEX, SPAN)


def _tile(name="source", dtype=DataType.FP4E2M1X2, shape=(32, 256), view=None, memref=None):
    return ir.Var(name, ir.TileType(list(shape), dtype, memref, view, ir.MemorySpace.Vec), SPAN)


def _pair(source=None, *, move_kwargs=None, cast_kwargs=None, cast_attrs=None, move_attrs=None, tmp=False):
    source = source if source is not None else _tile()
    args = [source, _tile("tmp", DataType.UINT8)] if tmp else [source]
    cast_kwargs = cast_kwargs or CAST
    # Deduce valid types before attaching deliberately unknown negative-test metadata.
    cast = ir.create_op_call(
        "tile.cast",
        args,
        {key: value for key, value in cast_kwargs.items() if key != "unknown_metadata"},
        SPAN,
    )
    if cast_attrs or "unknown_metadata" in cast_kwargs:
        cast = ir.Call(cast.op, list(cast.args), cast_kwargs, cast_attrs or {}, cast.type, SPAN)
    intermediate = ir.Var("nd", cast.type, SPAN)
    move_kwargs = move_kwargs or NZ
    # tile.move accepts TileLayout values at runtime, but create_op_call's
    # shared kwargs stub omits that enum. Preserve the exact metadata here.
    move = ir.create_op_call(
        "tile.move",
        [intermediate],
        typing.cast(
            Mapping[str, _OpKwarg],
            {key: value for key, value in move_kwargs.items() if key != "unknown_metadata"},
        ),
        SPAN,
    )
    if move_attrs or "unknown_metadata" in move_kwargs:
        move = ir.Call(move.op, list(move.args), move_kwargs, move_attrs or {}, move.type, SPAN)
    final = ir.Var("nz", move.type, SPAN)
    return (
        source,
        intermediate,
        final,
        ir.AssignStmt(intermediate, cast, SPAN),
        ir.AssignStmt(final, move, SPAN),
    )


def _program(body, params):
    function = ir.Function(
        "kernel",
        [(p, ir.ParamDirection.In) for p in params],
        [],
        ir.SeqStmts(body, SPAN),
        SPAN,
        ir.FunctionType.AIV,
    )
    return ir.Program([function], "fusion", SPAN)


def _run(program, target: BackendType | None = BackendType.Ascend950, planner=passes.MemoryPlanner.PYPTO):
    backend.reset_for_testing()
    if target is not None:
        backend.set_backend_type(target)
    # Negative tests deliberately construct mismatched types/extra uses. Only
    # the fusion result is under test; positive cases verify properties below.
    with passes.PassContext([], memory_planner=planner):
        return passes.expand_mixed_kernel()(program)


def _body(program):
    return next(iter(program.functions.values())).body


def _verify(program):
    properties = passes.IRPropertySet()
    for prop in (
        passes.IRProperty.SSAForm,
        passes.IRProperty.UseAfterDef,
        passes.IRProperty.AssignTypeSymmetry,
        passes.IRProperty.TypeChecked,
    ):
        properties.insert(prop)
    passes.verify_properties(properties, program, "producer-site NZ fusion")


def _calls(program, name):
    calls = []
    op_name = ir.get_op(name).name

    class Collect(ir.IRVisitor):
        def visit_call(self, op):
            if op.op.name == op_name:
                calls.append(op)
            super().visit_call(op)

    Collect().visit_program(program)
    return calls


def _assert_fresh_nz_addressing(program, source_type):
    (fused,) = _calls(program, "tile.cast_to_nz")
    # Replacing the adapter must preserve the source's own addressing while
    # the newly allocated blocked destination uses its implicit NZ addressing.
    ir.assert_structural_equal(fused.args[0].type, source_type)
    view = fused.type.get_effective_tile_view()
    assert not view.stride
    assert view.start_offset is None


@pytest.fixture(autouse=True)
def _reset_backend():
    yield
    backend.reset_for_testing()


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize("dynamic", [False, True])
def test_fusion_preserves_geometry_metadata_and_producer_position(planner, dynamic):
    dtype, cols = DataType.FP4E2M1X2, 256
    rows = ir.Var("valid_rows", ir.ScalarType(DataType.INDEX), SPAN) if dynamic else _const(23)
    view = ir.TileView(valid_shape=[rows, cols - 16], stride=[cols, 1], start_offset=16, pad=ir.PadValue.zero)
    source, nd, nz, cast, move = _pair(
        _tile(dtype=dtype, shape=(32, cols), view=view), cast_attrs={"compiler_note": "retain"}
    )
    unrelated = ir.Var("unrelated", ir.ScalarType(DataType.INDEX), SPAN)
    middle = ir.AssignStmt(unrelated, _const(7), SPAN)
    before = _program([cast, middle, move, ir.EvalStmt(nz, SPAN)], [source] + ([rows] if dynamic else []))
    after = _run(before, planner=planner)
    statements = list(_body(after).stmts)
    assert len(statements) == 3
    fused = statements[0]
    assert fused.var.same_as(nz) and fused.value.op.name == ir.get_op("tile.cast_to_nz").name
    assert statements[1].same_as(middle)
    assert dict(fused.value.kwargs) == CAST and dict(fused.value.attrs) == {"compiler_note": "retain"}
    ir.assert_structural_equal(fused.value.type, nz.type)
    assert not fused.value.type.get_effective_tile_view().stride
    assert fused.value.type.get_effective_tile_view().start_offset is None
    source_view = fused.value.args[0].type.get_effective_tile_view()
    assert [stride.value for stride in source_view.stride] == [cols, 1]
    assert source_view.start_offset.value == 16
    ir.assert_structural_equal(fused.value.args[0].type, source.type)
    assert "nd:" not in ir.python_print(after)
    _verify(after)


def test_fused_source_read_stays_before_intervening_source_mutation():
    source, _, nz, cast, move = _pair()
    # A functional writeback mutates source storage between cast and move.
    update = ir.create_op_call(
        "tile.assemble", [source, source, ir.MakeTuple([_const(0), _const(0)], SPAN)], {}, SPAN
    )
    mutation = ir.EvalStmt(update, SPAN)
    before = _program([cast, mutation, move, ir.EvalStmt(nz, SPAN)], [source])
    after = _run(before)
    statements = list(_body(after).stmts)
    assert statements[0].value.op.name == ir.get_op("tile.cast_to_nz").name
    assert statements[1].same_as(mutation)
    _verify(after)


@pytest.mark.parametrize("region", ["for", "zero_for", "while", "branch"])
def test_same_region_fuses_without_changing_execution_count(region):
    source, _, nz, cast, move = _pair()
    inner = ir.SeqStmts([cast, move, ir.EvalStmt(nz, SPAN)], SPAN)
    if region in ("for", "zero_for"):
        loop_var = ir.Var("i", ir.ScalarType(DataType.INDEX), SPAN)
        compound = ir.ForStmt(
            loop_var, _const(0), _const(0 if region == "zero_for" else 3), _const(1), [], inner, [], SPAN
        )
    elif region == "while":
        compound = ir.WhileStmt(ir.ConstBool(False, SPAN), [], inner, [], SPAN)
    else:
        compound = ir.IfStmt(ir.ConstBool(True, SPAN), inner, None, [], SPAN)
    after = _run(_program([compound], [source]))
    rewritten = list(_body(after).stmts)[0]
    body = rewritten.then_body if region == "branch" else rewritten.body
    assert len(body.stmts) == 2
    assert body.stmts[0].value.op.name == ir.get_op("tile.cast_to_nz").name
    _verify(after)


@pytest.mark.parametrize("extra", ["return", "yield", "submit", "submit_deps", "initializer"])
def test_all_real_reads_prevent_fusion(extra):
    source, nd, nz, cast, move = _pair()
    if extra == "return":
        use = ir.ReturnStmt([nd], SPAN)
    elif extra == "yield":
        use = ir.YieldStmt([nd], SPAN)
    elif extra in ("submit", "submit_deps"):
        # The deps-only raw case deliberately bypasses task-id typing: this
        # tests the shared syntactic visitor, not Submit's task-id verifier.
        args, deps = ([nd], []) if extra == "submit" else ([], [nd])
        use = ir.EvalStmt(
            ir.Submit(ir.GlobalVar("consumer"), args, deps, ir.ScalarType(DataType.TASK_ID), SPAN), SPAN
        )
    else:
        carry = ir.IterArg("carry", nd.type, nd, SPAN)
        loop_var = ir.Var("i", ir.ScalarType(DataType.INDEX), SPAN)
        body = ir.SeqStmts([ir.EvalStmt(carry, SPAN), ir.YieldStmt([carry], SPAN)], SPAN)
        result = ir.Var("carried", nd.type, SPAN)
        use = ir.ForStmt(loop_var, _const(0), _const(2), _const(1), [carry], body, [result], SPAN)
    before = _program([cast, move, use, ir.EvalStmt(nz, SPAN)], [source])
    ir.assert_structural_equal(_run(before), before)


@pytest.mark.parametrize("kind", ["branch", "loop", "separate_loops"])
def test_cross_region_pairs_remain_native_nd(kind):
    source, _, nz, cast, move = _pair()
    inner = ir.SeqStmts([move, ir.EvalStmt(nz, SPAN)], SPAN)
    if kind == "branch":
        compound = ir.IfStmt(ir.ConstBool(True, SPAN), inner, None, [], SPAN)
    else:
        i = ir.Var("i", ir.ScalarType(DataType.INDEX), SPAN)
        compound = ir.ForStmt(i, _const(0), _const(2), _const(1), [], inner, [], SPAN)
    producer = cast
    if kind == "separate_loops":
        j = ir.Var("j", ir.ScalarType(DataType.INDEX), SPAN)
        producer = ir.ForStmt(j, _const(0), _const(2), _const(1), [], ir.SeqStmts([cast], SPAN), [], SPAN)
    before = _program([producer, compound], [source])
    ir.assert_structural_equal(_run(before), before)


@pytest.mark.parametrize(
    "case",
    [
        "non_nz",
        "dtype",
        "target",
        "tmp",
        "move_attrs",
        "move_kwargs",
        "cast_kwargs",
        "physical",
        "compact",
        "source_layout",
        "memref",
        "result_type",
    ],
)
def test_unfusable_cases_leave_ordinary_cast_unchanged(case):
    options = {}
    source = None
    if case == "non_nz":
        options["move_kwargs"] = {"target_memory": ir.MemorySpace.Vec}
    elif case == "dtype":
        source = _tile(dtype=DataType.BF16, shape=(32, 512))
    elif case == "target":
        options["cast_kwargs"] = {"target_type": DataType.FP8E5M2, "mode": 2}
    elif case == "tmp":
        options["tmp"] = True
    elif case == "move_attrs":
        options["move_attrs"] = {"compiler_note": "decline"}
    elif case == "move_kwargs":
        options["move_kwargs"] = NZ | {"unknown_metadata": 1}
    elif case == "cast_kwargs":
        options["cast_kwargs"] = CAST | {"unknown_metadata": 1}
    elif case == "physical":
        source = _tile(shape=(17, 256))
    elif case == "compact":
        source = _tile(view=ir.TileView(compact=ir.CompactMode.normal))
    elif case == "source_layout":
        source = _tile(view=ir.TileView(blayout=ir.TileLayout.col_major))
    elif case == "memref":
        source = _tile()
    source, nd, nz, cast, move = _pair(source, **options)
    if case in ("memref", "result_type"):
        ty = nz.type
        assert isinstance(ty, ir.TileType)
        view = ty.get_effective_tile_view()
        if case == "result_type":
            view = ir.TileView(
                valid_shape=[20, 512], blayout=ir.TileLayout.col_major, slayout=ir.TileLayout.row_major
            )
        memory = ir.MemRef("allocated", 0, 16384) if case == "memref" else None
        nz = ir.Var("nz", ir.TileType([32, 512], ty.dtype, memory, view, ir.MemorySpace.Vec), SPAN)
        assert isinstance(move.value, ir.Call)
        move = ir.AssignStmt(nz, ir.Call(move.value.op, [nd], dict(move.value.kwargs), nz.type, SPAN), SPAN)
    before = _program([cast, move, ir.EvalStmt(nz, SPAN)], [source])
    ir.assert_structural_equal(_run(before), before)


@pytest.mark.parametrize("target", [None, BackendType.Ascend910B])
def test_capability_gates_fusion(target):
    source, _, nz, cast, move = _pair()
    before = _program([cast, move, ir.EvalStmt(nz, SPAN)], [source])
    ir.assert_structural_equal(_run(before, target), before)


@pytest.mark.parametrize(
    "case",
    [
        "arity",
        "zero_arity",
        "missing_mode",
        "missing_target",
        "source",
        "scalar",
        "target",
        "target_type",
        "mode_type",
        "mode_range",
        "saturation_type",
        "unknown_kwarg",
    ],
)
def test_internal_op_typing_rejects_invalid_contract(case):
    source = _tile()
    kwargs: dict[str, DataType | int | str | ir.Var] = dict(CAST)
    args = [source]
    if case == "arity":
        args.append(source)
    elif case == "zero_arity":
        args = []
    elif case == "missing_target":
        del kwargs["target_type"]
    elif case == "scalar":
        args = [_const(1)]
    elif case == "missing_mode":
        del kwargs["mode"]
    elif case == "source":
        args = [_tile(dtype=DataType.BF16)]
    elif case == "target":
        kwargs["target_type"] = DataType.FP8E5M2
    elif case == "target_type":
        kwargs["target_type"] = "FP8E4M3FN"
    elif case == "mode_type":
        kwargs["mode"] = "round"
    elif case == "mode_range":
        kwargs["mode"] = 7
    elif case == "saturation_type":
        kwargs["saturation_mode"] = "off"
    elif case == "unknown_kwarg":
        kwargs["tmp"] = source
    with pytest.raises((ValueError, TypeError)):
        # The tmp=Var case intentionally violates the shared kwargs stub so
        # the runtime validator, rather than the type checker, rejects it.
        ir.create_op_call("tile.cast_to_nz", args, typing.cast(Mapping[str, _OpKwarg], kwargs), SPAN)


@pytest.mark.parametrize(
    "case",
    [
        "small",
        "unaligned",
        "rows",
        "dynamic",
        "bounds",
        "column_overflow",
        "footprint_overflow",
        "register_overflow",
        "layout",
        "valid",
        "dynamic_valid",
        "memory",
    ],
)
def test_internal_op_typing_rejects_invalid_geometry(case):
    args = [_tile()]
    if case == "small":
        args = [_tile(shape=(32, 16))]
    elif case == "unaligned":
        args = [_tile(shape=(32, 48))]
    elif case == "rows":
        args = [_tile(shape=(17, 256))]
    elif case == "dynamic":
        args = [_tile(shape=(32, ir.Var("n", ir.ScalarType(DataType.INDEX), SPAN)))]
    elif case == "bounds":
        args = [_tile(shape=(65536, 256))]
    elif case == "column_overflow":
        args = [_tile(shape=(16, 1 << 30))]
    elif case == "footprint_overflow":
        args = [_tile(shape=(65520, 32768))]
    elif case == "register_overflow":
        args = [_tile(shape=(16, 16777216))]
    elif case == "memory":
        args = [
            ir.Var(
                "source", ir.TileType([32, 256], DataType.FP4E2M1X2, memory_space=ir.MemorySpace.Mat), SPAN
            )
        ]
    elif case == "layout":
        args = [_tile(view=ir.TileView(compact=ir.CompactMode.normal))]
    elif case == "valid":
        args = [_tile(view=ir.TileView(valid_shape=[32, 257]))]
    elif case == "dynamic_valid":
        args = [_tile(view=ir.TileView(valid_shape=[32, ir.Var("n", ir.ScalarType(DataType.INDEX), SPAN)]))]
    with pytest.raises((ValueError, TypeError)):
        ir.create_op_call("tile.cast_to_nz", args, CAST, SPAN)


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_legalization_is_one_native_cast_with_either_planner(planner):
    source = _tile()
    source, _, _, cast, _ = _pair(source)
    before = _program([cast], [source])
    backend.set_backend_type(BackendType.Ascend950)
    with passes.PassContext([], memory_planner=planner):
        after = passes.legalize_tile_cast()(before)
    ir.assert_structural_equal(after, before)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("loop_carry", [False, True])
def test_packed_widening_has_disjoint_storage_and_exact_24k_capacity(fused, loop_carry):
    backend.set_backend_type(BackendType.Ascend950)
    gm = ir.Var("gm", ir.TensorType([32, 256], DataType.FP4E2M1X2), SPAN)
    indices = ir.MakeTuple([_const(0), _const(0)], SPAN)
    shape = ir.MakeTuple([_const(32), _const(256)], SPAN)
    load = ir.create_op_call(
        "tile.load", [gm, indices, shape, shape], {"target_memory": ir.MemorySpace.Vec}, SPAN
    )
    source = ir.Var("source", load.type, SPAN)
    conversion = ir.create_op_call("tile.cast_to_nz" if fused else "tile.cast", [source], CAST, SPAN)
    result = ir.Var("result", conversion.type, SPAN)
    statements: list[ir.Stmt] = [ir.AssignStmt(source, load, SPAN), ir.AssignStmt(result, conversion, SPAN)]
    if loop_carry:
        # Semantic retargeting onto a carry must retain the same disjointness.
        seed_source = ir.Var("seed_source", load.type, SPAN)
        seed_call = ir.create_op_call("tile.cast_to_nz" if fused else "tile.cast", [seed_source], CAST, SPAN)
        seed = ir.Var("seed", seed_call.type, SPAN)
        carry = ir.IterArg("carry", result.type, seed, SPAN)
        final = ir.Var("final", result.type, SPAN)
        loop = ir.ForStmt(
            ir.Var("i", ir.ScalarType(DataType.INDEX), SPAN),
            _const(0),
            _const(2),
            _const(1),
            [carry],
            ir.SeqStmts(statements + [ir.YieldStmt([result], SPAN)], SPAN),
            [final],
            SPAN,
        )
        statements = [
            ir.AssignStmt(seed_source, load, SPAN),
            ir.AssignStmt(seed, seed_call, SPAN),
            loop,
            ir.EvalStmt(final, SPAN),
        ]
    else:
        statements.append(ir.EvalStmt(result, SPAN))
    program = _program(statements, [gm])
    with passes.PassContext([], enable_buffer_ir=False):
        after = passes.init_mem_ref()(program)
        after = passes.materialize_semantic_aliases()(after)
        after = passes.memory_reuse()(after)
        after = passes.allocate_memory_addr()(after)
    ranges: dict[str, tuple[int, int]] = {}

    class Ranges(ir.IRVisitor):
        def visit_assign_stmt(self, op):
            if op.var.name_hint in ("source", "result"):
                assert isinstance(op.var.type, ir.TileType)
                memory = op.var.type.memref
                assert memory is not None
                assert isinstance(memory.byte_offset_, ir.ConstInt)
                ranges[op.var.name_hint] = (memory.byte_offset_.value, memory.size_)
            super().visit_assign_stmt(op)

    Ranges().visit_program(after)
    source_offset, source_size = ranges["source"]
    result_offset, result_size = ranges["result"]
    assert (source_size, result_size) == (8192, 16384)
    assert source_offset + source_size <= result_offset or result_offset + result_size <= source_offset
    if not loop_carry:
        assert max(source_offset + source_size, result_offset + result_size) == 24576


def test_fused_internal_builder_printer_parser_roundtrip():
    source, _, nz, cast, move = _pair()
    after = _run(_program([cast, move, ir.EvalStmt(nz, SPAN)], [source]))
    text = ir.python_print(after)
    assert "pl.tile.cast_to_nz(" in text
    reparsed = pl.parse_program(text)
    ir.assert_structural_equal(reparsed, after)
    _verify(reparsed)


def test_iterarg_source_reads_preserve_initializer_and_ssa():
    source = _tile()
    carry = ir.IterArg("carry", source.type, source, SPAN)
    _, _, nz, cast, move = _pair(carry)
    # Both the cast and identity yield read the IterArg. Neither visit should
    # revisit the FP4 initializer, and the sole ND consumer remains the move.
    body = ir.SeqStmts([cast, move, ir.EvalStmt(nz, SPAN), ir.YieldStmt([carry], SPAN)], SPAN)
    loop = ir.ForStmt(
        ir.Var("i", ir.ScalarType(DataType.INDEX), SPAN),
        _const(0),
        _const(2),
        _const(1),
        [carry],
        body,
        [ir.Var("final", source.type, SPAN)],
        SPAN,
    )
    after = _run(_program([loop], [source]))
    assert _body(after).stmts[0].body.stmts[0].value.op.name == ir.get_op("tile.cast_to_nz").name
    _verify(after)


@pytest.mark.parametrize(
    "field", ["stride", "start_offset", "blayout", "slayout", "fractal", "pad", "compact", "valid_shape"]
)
def test_full_move_result_view_must_equal_fresh_nz_type(field):
    source, nd, nz, cast, move = _pair()
    changes: _TileViewFields = {
        "stride": [512, 1],
        "start_offset": 1,
        "blayout": ir.TileLayout.row_major,
        "slayout": ir.TileLayout.col_major,
        "fractal": 1024,
        "pad": ir.PadValue.zero,
        "compact": ir.CompactMode.normal,
        "valid_shape": [20, 512],
    }
    fields: _TileViewFields = {
        "valid_shape": [32, 512],
        "blayout": ir.TileLayout.col_major,
        "slayout": ir.TileLayout.row_major,
    }
    # Literal keys keep each selected metadata value paired with its field's
    # constructor type while retaining the same single-field perturbations.
    if field == "stride":
        fields["stride"] = changes["stride"]
    elif field == "start_offset":
        fields["start_offset"] = changes["start_offset"]
    elif field == "blayout":
        fields["blayout"] = changes["blayout"]
    elif field == "slayout":
        fields["slayout"] = changes["slayout"]
    elif field == "fractal":
        fields["fractal"] = changes["fractal"]
    elif field == "pad":
        fields["pad"] = changes["pad"]
    elif field == "compact":
        fields["compact"] = changes["compact"]
    elif field == "valid_shape":
        fields["valid_shape"] = changes["valid_shape"]
    else:
        raise KeyError(field)
    ty = ir.TileType(
        [32, 512], DataType.FP8E4M3FN, tile_view=ir.TileView(**fields), memory_space=ir.MemorySpace.Vec
    )
    nz = ir.Var("nz", ty, SPAN)
    assert isinstance(move.value, ir.Call)
    move = ir.AssignStmt(nz, ir.Call(move.value.op, [nd], dict(move.value.kwargs), ty, SPAN), SPAN)
    before = _program([cast, move, ir.EvalStmt(nz, SPAN)], [source])
    ir.assert_structural_equal(_run(before), before)


def test_move_site_extent_definition_is_not_hoisted_to_cast():
    source, nd, _, cast, move = _pair()
    extent = ir.Var("later_extent", ir.ScalarType(DataType.INDEX), SPAN)
    view = ir.TileView(
        valid_shape=[extent, 512], blayout=ir.TileLayout.col_major, slayout=ir.TileLayout.row_major
    )
    ty = ir.TileType([32, 512], DataType.FP8E4M3FN, tile_view=view, memory_space=ir.MemorySpace.Vec)
    nz = ir.Var("nz", ty, SPAN)
    assert isinstance(move.value, ir.Call)
    move = ir.AssignStmt(nz, ir.Call(move.value.op, [nd], dict(move.value.kwargs), ty, SPAN), SPAN)
    middle = ir.AssignStmt(extent, _const(20), SPAN)
    before = _program([cast, middle, move, ir.EvalStmt(nz, SPAN)], [source])
    ir.assert_structural_equal(_run(before), before)


@pytest.mark.parametrize("allocated", ["nd", "nz", "both_alias"])
def test_preallocated_intermediate_or_aliased_result_declines(allocated):
    source, nd, nz, cast, move = _pair()
    memory = ir.MemRef("allocated", 0, 16384)
    if allocated in ("nd", "both_alias"):
        assert isinstance(nd.type, ir.TileType)
        ty = ir.TileType([32, 512], DataType.FP8E4M3FN, memory, nd.type.tile_view, ir.MemorySpace.Vec)
        nd = ir.Var("nd", ty, SPAN)
        assert isinstance(cast.value, ir.Call)
        cast = ir.AssignStmt(nd, ir.Call(cast.value.op, [source], CAST, ty, SPAN), SPAN)
    if allocated in ("nz", "both_alias"):
        assert isinstance(nz.type, ir.TileType)
        ty = ir.TileType([32, 512], DataType.FP8E4M3FN, memory, nz.type.tile_view, ir.MemorySpace.Vec)
        nz = ir.Var("nz", ty, SPAN)
    assert isinstance(move.value, ir.Call)
    move = ir.AssignStmt(nz, ir.Call(move.value.op, [nd], dict(move.value.kwargs), nz.type, SPAN), SPAN)
    before = _program([cast, move, ir.EvalStmt(nz, SPAN)], [source])
    ir.assert_structural_equal(_run(before), before)


@pytest.mark.parametrize("authored_type", [pl.FunctionType.AIV, pl.FunctionType.InCore])
def test_manual_push_adapter_inserted_by_expand_is_eligible(authored_type):
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIC)
        def cube(self, gm: pl.Tensor[[32, 256], pl.FP4E2M1X2]):
            fifo = pl.reserve_buffer(name="v2c", size=131072, base=pl.AUTO)
            pl.aic_initialize_pipe(pl.const(0, pl.INT32), fifo, dir_mask=2, slot_size=16384, id=1)
            value: pl.Tile[[32, 512], pl.FP8E4M3FN, pl.Mem.Mat] = pl.tpop_from_aiv(split=0, id=1)
            pl.tfree_to_aiv(value, id=1)

        @pl.function(type=pl.FunctionType.AIV)
        def vector(self, gm: pl.Tensor[[32, 256], pl.FP4E2M1X2]):
            peer = pl.import_peer_buffer(name="v2c", peer_func="cube")
            pl.aiv_initialize_pipe(pl.const(0, pl.INT32), peer, dir_mask=2, slot_size=16384, id=1)
            source = pl.load(gm, [0, 0], [32, 256], target_memory=pl.Mem.Vec)
            value = pl.cast(source, pl.FP8E4M3FN)
            pl.tpush_to_aic(value, split=0, id=1)

        @pl.function(type=pl.FunctionType.Group)
        def group(self, gm: pl.Tensor[[32, 256], pl.FP4E2M1X2]):
            self.cube(gm)
            self.vector(gm)

    functions = []
    for func in Before.functions.values():
        if func.name == "vector":
            func = ir.Function(
                func.name,
                list(zip(func.params, func.param_directions)),
                func.return_types,
                func.body,
                func.span,
                authored_type,
                attrs=dict(func.attrs),
                requires_runtime_binding=func.requires_runtime_binding,
                ir_stage=func.ir_stage,
            )
        functions.append(func)
    before = ir.Program(functions, Before.name, Before.span)
    vector_before = before.get_function("vector")
    assert vector_before is not None
    assert vector_before.func_type == authored_type
    (original_cast,) = _calls(before, "tile.cast")
    # Packed widening materializes explicit ND strides. The manual adapter
    # must not copy these into its fresh NZ destination.
    assert [stride.value for stride in original_cast.type.get_effective_tile_view().stride] == [512, 1]
    backend.set_backend_type(BackendType.Ascend950)
    with passes.PassContext([]):
        after = passes.expand_mixed_kernel()(before)
    vector = after.get_function("vector")
    assert vector is not None
    text = ir.python_print(vector)
    assert "tile.cast_to_nz(" in text
    assert "tile.move(" not in text and "tile.cast(" not in text
    _assert_fresh_nz_addressing(after, original_cast.args[0].type)
    (push,) = _calls(after, "tile.tpush_to_aic")
    assert dict(push.kwargs)["id"] == 1 and dict(push.kwargs)["split"] == 0
    _verify(after)


@pytest.mark.parametrize("op_driven", [False, True])
def test_op_driven_gather_adapter_inserted_by_expand_is_eligible(op_driven):
    # The explicit gather carries split semantics into ExpandMixedKernel;
    # provenance alone must not block its compatible Vec ND -> NZ adapter.
    source, nd, _, cast, _ = _pair(_tile(view=ir.TileView(stride=[256, 1], start_offset=16)))
    gather = (
        tile_ops.aic_gather(nd, split=1, span=SPAN)
        if op_driven
        else ir.create_op_call("tile.move", [nd], {"target_memory": ir.MemorySpace.Mat}, SPAN)
    )
    full = ir.Var("full", gather.type, SPAN)
    # Consume the gathered Mat tile on AIC, as in a mixed matmul kernel.
    # A bare EvalStmt(full) is shared and would also survive on AIV.
    left_move = tile_ops.move(full, ir.MemorySpace.Left, span=SPAN)
    left = ir.Var("left", left_move.type, SPAN)
    right = ir.Var(
        "right", ir.TileType([512, 32], DataType.FP8E4M3FN, memory_space=ir.MemorySpace.Right), SPAN
    )
    matmul = tile_ops.matmul(left, right, span=SPAN)
    result = ir.Var("result", matmul.type, SPAN)
    # Keep the cube consumer observable through the mixed kernel's output.
    move_vec = tile_ops.move(result, ir.MemorySpace.Vec, span=SPAN)
    result_vec = ir.Var("result_vec", move_vec.type, SPAN)
    assert isinstance(matmul.type, ir.TileType)
    out = ir.Var("out", ir.TensorType(list(matmul.type.shape), DataType.FP32), SPAN)
    store = tile_ops.store(result_vec, [0, 0], out, span=SPAN)
    out_store = ir.Var("out_store", store.type, SPAN)
    body = ir.SeqStmts(
        [
            cast,
            ir.AssignStmt(full, gather, SPAN),
            ir.AssignStmt(left, left_move, SPAN),
            ir.AssignStmt(result, matmul, SPAN),
            ir.AssignStmt(result_vec, move_vec, SPAN),
            ir.AssignStmt(out_store, store, SPAN),
            ir.ReturnStmt([out_store], SPAN),
        ],
        SPAN,
    )
    func = ir.Function(
        "op_driven",
        [(source, ir.ParamDirection.In), (right, ir.ParamDirection.In), (out, ir.ParamDirection.Out)],
        [out.type],
        body,
        SPAN,
        ir.FunctionType.InCore,
        attrs={"split": pl.SplitMode.UP_DOWN, "split_aiv": True} if op_driven else {},
    )
    program = ir.Program([func], "op_driven", SPAN)
    backend.set_backend_type(BackendType.Ascend950)
    with passes.PassContext([]):
        after = passes.expand_mixed_kernel()(program)
    aic = after.get_function("op_driven_aic")
    aiv = after.get_function("op_driven_aiv")
    assert aic is not None and aiv is not None
    assert "tile.matmul(" in ir.python_print(aic)
    assert "tile.matmul(" not in ir.python_print(aiv)
    (pop,) = _calls(after, "tile.tpop_from_aiv")
    assert [dim.value for dim in pop.type.shape] == [64 if op_driven else 32, 512]
    (cube_matmul,) = _calls(after, "tile.matmul")
    assert [dim.value for dim in cube_matmul.type.shape] == [64 if op_driven else 32, 32]
    text = ir.python_print(after)
    assert "tile.cast_to_nz(" in text
    assert "tile.cast(" not in text
    _assert_fresh_nz_addressing(after, source.type)
    (fused,) = _calls(after, "tile.cast_to_nz")
    source_view = fused.args[0].type.get_effective_tile_view()
    assert [stride.value for stride in source_view.stride] == [256, 1]
    assert source_view.start_offset.value == 16
    _verify(after)


@pytest.mark.parametrize("fused", [False, True])
def test_last_use_packed_source_in_larger_dead_buffer_remains_disjoint(fused):
    backend.set_backend_type(BackendType.Ascend950)
    gm = ir.Var("gm", ir.TensorType([32, 256], DataType.FP4E2M1X2), SPAN)
    large_gm = ir.Var("large_gm", ir.TensorType([32, 512], DataType.FP8E4M3FN), SPAN)

    def load(tensor, shape):
        return ir.create_op_call(
            "tile.load",
            [
                tensor,
                ir.MakeTuple([_const(0), _const(0)], SPAN),
                ir.MakeTuple([_const(x) for x in shape], SPAN),
                ir.MakeTuple([_const(x) for x in shape], SPAN),
            ],
            {"target_memory": ir.MemorySpace.Vec},
            SPAN,
        )

    large_load0 = load(large_gm, [32, 512])
    large0 = ir.Var("dead_large0", large_load0.type, SPAN)
    large_load1 = load(large_gm, [32, 512])
    large1 = ir.Var("dead_large1", large_load1.type, SPAN)
    source_load = load(gm, [32, 256])
    source = ir.Var("source", source_load.type, SPAN)
    conversion = ir.create_op_call("tile.cast_to_nz" if fused else "tile.cast", [source], CAST, SPAN)
    result = ir.Var("result", conversion.type, SPAN)
    program = _program(
        [
            ir.AssignStmt(large0, large_load0, SPAN),
            ir.AssignStmt(large1, large_load1, SPAN),
            ir.EvalStmt(large0, SPAN),
            ir.EvalStmt(large1, SPAN),
            ir.AssignStmt(source, source_load, SPAN),
            ir.AssignStmt(result, conversion, SPAN),
            ir.EvalStmt(result, SPAN),
        ],
        [gm, large_gm],
    )
    with passes.PassContext([], enable_buffer_ir=False):
        after = passes.init_mem_ref()(program)
        after = passes.materialize_semantic_aliases()(after)
        after = passes.memory_reuse()(after)
        after = passes.allocate_memory_addr()(after)
    memories: dict[str, ir.MemRef] = {}

    class Collect(ir.IRVisitor):
        def visit_assign_stmt(self, op):
            if op.var.name_hint in ("source", "result", "dead_large0", "dead_large1"):
                assert isinstance(op.var.type, ir.TileType)
                memory = op.var.type.memref
                assert memory is not None
                memories[op.var.name_hint] = memory
            super().visit_assign_stmt(op)

    Collect().visit_program(after)
    src, dst, dead0, dead1 = (memories[x] for x in ("source", "result", "dead_large0", "dead_large1"))
    assert dead0.size_ == dead1.size_ == 32 * 512
    assert not dead0.base_.same_as(dead1.base_)
    assert src.base_.same_as(dead0.base_) or src.base_.same_as(dead1.base_), (
        "The regression requires actual reuse of a larger dead buffer"
    )
    assert isinstance(src.byte_offset_, ir.ConstInt)
    assert isinstance(dst.byte_offset_, ir.ConstInt)
    assert (
        src.byte_offset_.value + src.size_ <= dst.byte_offset_.value
        or dst.byte_offset_.value + dst.size_ <= src.byte_offset_.value
    )


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_e1m2_never_uses_the_e2m1_direct_path(planner):
    """An equal-width packed format is not an E2M1 conversion contract."""
    source = _tile(dtype=DataType.FP4E1M2X2)
    with pytest.raises(ValueError, match="requires FP4E2M1X2 source"):
        ir.create_op_call("tile.cast_to_nz", [source], CAST, SPAN)
    source, _, _, cast, move = _pair(source)
    before = _program([cast, move], [source])
    after = _run(before, planner=planner)
    assert not _calls(after, "tile.cast_to_nz")
    backend.set_backend_type(BackendType.Ascend950)
    with passes.PassContext([], memory_planner=planner):
        with pytest.raises(ValueError, match="no native cast path"):
            passes.legalize_tile_cast()(before)


def test_odd_carrier_valid_columns_expand_only_at_cast_boundary():
    source = _tile(view=ir.TileView(valid_shape=[23, 129]))
    cast = ir.create_op_call("tile.cast_to_nz", [source], CAST, SPAN)
    assert isinstance(source.type, ir.TileType)
    assert isinstance(cast.type, ir.TileType)
    for actual, expected in (
        (source.type.get_effective_tile_view().valid_shape, [23, 129]),
        (cast.type.get_effective_tile_view().valid_shape, [23, 258]),
        (cast.type.shape, [32, 512]),
    ):
        assert len(actual) == len(expected)
        for dimension, value in zip(actual, expected, strict=True):
            assert isinstance(dimension, ir.ConstInt)
            assert dimension.value == value


def test_a2a3_still_rejects_packed_fp4_native_conversion():
    """Adding the A5 edge must not introduce an A2/A3 legalization route."""
    source, _, _, cast, _ = _pair()
    before = _program([cast, ir.ReturnStmt([], SPAN)], [source])
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    with pytest.raises(ValueError, match="no native cast path"):
        passes.legalize_tile_cast()(before)


def _padded_push_program(shape=(128, 128), valid=None, extra_read=False):
    source, _, nz, cast, move = _pair(
        _tile(shape=shape, view=ir.TileView(valid_shape=valid) if valid else None)
    )
    push = tile_ops.tpush_to_aic(nz, split=1, span=SPAN)
    statements = [cast, move, ir.EvalStmt(push, SPAN)]
    if extra_read:
        statements.append(ir.EvalStmt(nz, SPAN))
    return _program(statements, [source])


@pytest.mark.parametrize("planner", list(passes.MemoryPlanner))
@pytest.mark.parametrize("enabled", [False, True])
def test_row_plus_one_selects_the_full_aiv_producer_and_same_tpush_type(planner, enabled):
    before = _padded_push_program()
    backend.set_backend_type(BackendType.Ascend950)
    with passes.PassContext(
        [], memory_planner=planner, enable_fp4_nz_row_plus_one=enabled, enable_buffer_ir=False
    ):
        after = passes.expand_mixed_kernel()(before)
    (cast,) = _calls(after, "tile.cast_to_nz")
    (push,) = _calls(after, "tile.tpush_to_aic")
    mode = ir.CompactMode.row_plus_one if enabled else ir.CompactMode.null
    assert cast.type.get_effective_tile_view().compact == mode
    assert push.args[0].type.get_effective_tile_view().compact == mode
    assert [dim.value for dim in cast.type.shape] == [128, 256]
    ir.assert_structural_equal(cast.type, push.args[0].type)
    _verify(after)


@pytest.mark.parametrize(
    "shape,valid,extra_read",
    [
        ((64, 128), None, False),
        ((128, 256), None, False),
        ((128, 128), [127, 128], False),
        ((128, 128), [128, 127], False),
        ((128, 128), None, True),
    ],
)
def test_row_plus_one_unsupported_geometry_or_consumer_keeps_ordinary_nz(shape, valid, extra_read):
    before = _padded_push_program(shape, valid, extra_read)
    backend.set_backend_type(BackendType.Ascend950)
    with passes.PassContext([], enable_fp4_nz_row_plus_one=True):
        after = passes.expand_mixed_kernel()(before)
    (cast,) = _calls(after, "tile.cast_to_nz")
    assert cast.type.get_effective_tile_view().compact == ir.CompactMode.null
    _verify(after)


def test_row_plus_one_storage_reserves_all_panels_and_roundtrips():
    backend.set_backend_type(BackendType.Ascend950)
    gm = ir.Var("gm", ir.TensorType([128, 128], DataType.FP4E2M1X2), SPAN)
    indices = ir.MakeTuple([_const(0), _const(0)], SPAN)
    shape = ir.MakeTuple([_const(128), _const(128)], SPAN)
    load = ir.create_op_call(
        "tile.load", [gm, indices, shape, shape], {"target_memory": ir.MemorySpace.Vec}, SPAN
    )
    source = ir.Var("source", load.type, SPAN)
    cast = ir.create_op_call("tile.cast_to_nz", [source], {**CAST, "compact": 2}, SPAN)
    result = ir.Var("result", cast.type, SPAN)
    before = _program(
        [ir.AssignStmt(source, load, SPAN), ir.AssignStmt(result, cast, SPAN), ir.EvalStmt(result, SPAN)],
        [gm],
    )
    # Both type serialization and printed IR must retain the padded descriptor;
    # a Null consumer would read the wrong panel stride.
    ir.assert_structural_equal(ir.deserialize(ir.serialize(before)), before)
    ir.assert_structural_equal(pl.parse_program(ir.python_print(before)), before)
    with passes.PassContext([], enable_buffer_ir=False):
        after = passes.init_mem_ref()(before)
        after = passes.materialize_semantic_aliases()(after)
        after = passes.memory_reuse()(after)
        after = passes.allocate_memory_addr()(after)
    destinations = []

    class CollectDestination(ir.IRVisitor):
        def visit_assign_stmt(self, op):
            if isinstance(op.value, ir.Call) and op.value.op == ir.get_op("tile.cast_to_nz"):
                # Allocation belongs to the bound SSA variable, not the
                # operation's independently inferred result type.
                assert op.var.type.tile_view.compact == ir.CompactMode.row_plus_one
                destinations.append(op.var.type.memref)
            super().visit_assign_stmt(op)

    CollectDestination().visit_program(after)
    (converted,) = _calls(after, "tile.cast_to_nz")
    src = converted.args[0].type.memref
    (dst,) = destinations
    assert src is not None and dst is not None
    assert (src.size_, dst.size_) == (16384, 33024)
    assert (
        src.byte_offset_.value + src.size_ <= dst.byte_offset_.value
        or dst.byte_offset_.value + dst.size_ <= src.byte_offset_.value
    )


@pytest.mark.parametrize("valid", [[127, 128], [128, 127]])
def test_explicit_row_plus_one_rejects_unproven_tail_contract(valid):
    source = _tile(shape=(128, 128), view=ir.TileView(valid_shape=valid))
    with pytest.raises(ValueError, match="currently requires full 128x256 output"):
        ir.create_op_call("tile.cast_to_nz", [source], {**CAST, "compact": 2}, SPAN)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
