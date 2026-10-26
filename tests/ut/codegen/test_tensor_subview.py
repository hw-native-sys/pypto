# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Full-pipeline GM subview regression and conservative safety gates."""

import subprocess

import pypto.language as pl
import pytest
import torch
from pypto import DataType, backend, codegen, ir, passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.debug import torch_codegen
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


@pl.jit
def copy_selected_rows(cache: pl.Tensor, output: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        selected = pl.tensor.subview(cache, [2, 512], [1, 0], steps=[3, 1])
        pair = pl.load(selected, [0, 0], [2, 512])
        pl.store(pair, [0, 0], output)
    return output


def make_program(
    offset: int | ir.Expr = 1,
    step: int | ir.Expr = 3,
    *,
    narrowed=False,
    sliced=False,
    extra_params=(),
    valid_rows=1,
    subview_rows: int | ir.Expr = 2,
):
    span = ir.Span.unknown()
    cache = ir.Var("cache", ir.TensorType([8, 512], DataType.BF16), span)
    output = ir.Var("output", ir.TensorType([2, 512], DataType.BF16), span)
    call = ir.op.tensor.subview(cache, [subview_rows, 512], [offset, 0], steps=[step, 1])
    selected = ir.Var("selected", call.type, span)
    statements: list[ir.Stmt] = [ir.AssignStmt(selected, call, span)]
    if narrowed:
        call = ir.op.tensor.set_validshape(selected, valid_rows, 512)
        selected = ir.Var("narrowed", call.type, span)
        statements.append(ir.AssignStmt(selected, call, span))
    if sliced:
        call = ir.op.tensor.slice(selected, [2, 512], [0, 0])
    else:
        call = ir.op.tile.load(selected, [0, 0], [2, 512], target_memory=ir.MemorySpace.Vec)
    pair = ir.Var("pair", call.type, span)
    statements.append(ir.AssignStmt(pair, call, span))
    store = (
        ir.op.tile.store(pair, [0, 0], output) if not sliced else ir.op.tensor.assemble(output, pair, [0, 0])
    )
    result = ir.Var("result", store.type, span)
    statements.extend([ir.AssignStmt(result, store, span), ir.ReturnStmt([result], span)])
    func = ir.Function(
        "copy_selected",
        [cache, output, *extra_params],
        [output.type],
        ir.SeqStmts(statements, span),
        span,
        type=ir.FunctionType.InCore,
    )
    return ir.Program([func], "subview", span)


def statements(func: ir.Function) -> list[ir.Stmt]:
    assert isinstance(func.body, ir.SeqStmts)
    return list(func.body.stmts)


def lower(program):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    return PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)


@pytest.mark.parametrize("sliced", [False, True])
def test_subview_pipeline_emits_only_result_tile(tmp_path, sliced):
    optimized = lower(make_program(sliced=sliced))
    borrowed: set[str] = set()

    class CheckBorrowedStorage(ir.IRVisitor):
        def visit_var(self, op: ir.Var) -> None:
            tensor = op.type
            if isinstance(tensor, ir.TensorType):
                view = tensor.tensor_view
                if view and view.addressing_kind == ir.TensorAddressingKind.AffineSubview:
                    assert tensor.memref is None
                    borrowed.add(op.name_hint)
            super().visit_var(op)

    CheckBorrowedStorage().visit_program(optimized)
    assert borrowed
    text = str(optimized)
    assert "tensor.subview" in text
    assert "tensor.slice" not in text
    mlir = codegen.PTOCodegen().generate(optimized)
    assert mlir.count("pto.addptr") == 1
    assert mlir.count("pto.tload") == 1
    assert mlir.count("pto.tstore") == 1
    assert "1536" in mlir
    assert "pto.tmov" not in mlir
    ptoas = find_ptoas_binary()
    if ptoas is not None:
        source = tmp_path / "subview.pto"
        source.write_text(mlir)
        for arch in ("a2", "a3"):
            output = tmp_path / f"subview-{arch}.cpp"
            run = subprocess.run(
                [
                    ptoas,
                    str(source),
                    f"--pto-arch={arch}",
                    "--pto-level=level3",
                    "--enable-insert-sync",
                    "-o",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            assert run.returncode == 0, run.stdout + run.stderr + mlir
            cpp = output.read_text()
            assert cpp.count("TLOAD(") == 1
            assert cpp.count("TSTORE(") == 1
            declarations = [line for line in cpp.splitlines() if line.lstrip().startswith("Tile<")]
            assert len(declarations) == 1
            assert "Tile<TileType::Vec, bfloat16_t, 2, 512," in declarations[0]


def test_subview_cpu_reference_selects_rows_one_and_four_without_snapshot():
    program = make_program()
    func = next(iter(program.functions.values()))
    namespace = {}
    exec(torch_codegen(func), namespace)
    rows = torch.arange(8)[:, None]
    cols = torch.arange(512)[None, :]
    cache = (16 * rows + cols % 16).to(torch.bfloat16)
    output = torch.empty((2, 512), dtype=torch.bfloat16)
    actual = namespace["copy_selected"](cache, output)
    torch.testing.assert_close(actual, cache[[1, 4]], rtol=0, atol=0)
    borrowed = namespace["_tensor_subview"](cache, [2, 512], [1, 0], [3, 1])
    assert borrowed.stride() == (1536, 1)
    cache[1, 0] = 127
    assert borrowed[0, 0] == 127


def test_subview_unknown_bounds_are_rejected_by_direct_conversion():
    span = ir.Span.unknown()
    unknown = ir.Var("unknown", ir.ScalarType(DataType.INDEX), span)
    program = make_program(offset=unknown, extra_params=(unknown,))
    with pytest.raises(ValueError, match="subview.*cannot prove"):
        passes.convert_tensor_to_tile_ops()(program)


def test_subview_validshape_remains_a_borrowed_tensor():
    optimized = lower(make_program(narrowed=True))
    assert "tensor.set_validshape" in str(optimized)
    assert "tile.set_validshape" not in str(optimized)
    assert "pto.tload" in codegen.PTOCodegen().generate(optimized)


def test_subview_validshape_may_preserve_full_parent_validity():
    optimized = lower(make_program(narrowed=True, valid_rows=2))
    assert "pto.tload" in codegen.PTOCodegen().generate(optimized)


def test_subview_dsl_has_only_tensor_namespace_entry():
    assert callable(pl.tensor.subview)
    assert not hasattr(pl, "subview")


def test_subview_bounded_device_parameters_survive_full_pipeline(tmp_path):
    span = ir.Span.unknown()
    lo = ir.Var("lo", ir.ScalarType(DataType.INT32), span)
    delta = ir.Var("delta", ir.ScalarType(DataType.INT32), span)
    program = make_program(offset=lo, step=delta)
    func = next(iter(program.functions.values()))

    def constant(value):
        return ir.ConstInt(value, DataType.INT32, span)

    def between(var, first, last):
        return ir.And(
            ir.Ge(var, constant(first), DataType.BOOL, span),
            ir.Le(var, constant(last), DataType.BOOL, span),
            DataType.BOOL,
            span,
        )

    condition = ir.And(between(lo, 1, 2), between(delta, 1, 3), DataType.BOOL, span)
    # Geometry facts dominate construction on device. They are compile-time
    # proof facts, with no assumed runtime precondition on the function args.
    stmts = statements(func)
    branch = ir.IfStmt(condition, ir.SeqStmts(stmts[:-1], span), None, [], span)
    body = ir.SeqStmts([branch, ir.ReturnStmt([func.params[1]], span)], span)
    guarded = ir.Function(
        func.name, [*func.params, lo, delta], func.return_types, body, span, type=ir.FunctionType.InCore
    )
    optimized = lower(ir.Program([guarded], "dynamic_subview", span))
    mlir = codegen.PTOCodegen().generate(optimized)
    assert mlir.count("pto.addptr") == 1
    assert "arith.muli" in mlir
    assert "pto.tload" in mlir
    ptoas = find_ptoas_binary()
    if ptoas is not None:
        source = tmp_path / "dynamic-subview.pto"
        source.write_text(mlir)
        for arch in ("a2", "a3"):
            output = tmp_path / f"dynamic-subview-{arch}.cpp"
            run = subprocess.run(
                [
                    ptoas,
                    str(source),
                    f"--pto-arch={arch}",
                    "--pto-level=level3",
                    "--enable-insert-sync",
                    "-o",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            assert run.returncode == 0, run.stdout + run.stderr + mlir
            cpp = output.read_text()
            assert cpp.count("TLOAD(") == 1
            assert cpp.count("TSTORE(") == 1


@pytest.mark.parametrize("guarded", [True, False])
def test_subview_dynamic_shape_with_static_load(tmp_path, guarded):
    span = ir.Span.unknown()
    rows = ir.Var("rows", ir.ScalarType(DataType.INDEX), span)
    program = make_program(step=1, subview_rows=rows, extra_params=(rows,))
    func = next(iter(program.functions.values()))
    if not guarded:
        with pytest.raises(ValueError, match="subview.*cannot prove"):
            lower(program)
        return
    condition = ir.And(
        ir.Ge(rows, ir.ConstInt(2, DataType.INDEX, span), DataType.BOOL, span),
        ir.Le(rows, ir.ConstInt(4, DataType.INDEX, span), DataType.BOOL, span),
        DataType.BOOL,
        span,
    )
    body = ir.SeqStmts(
        [
            ir.IfStmt(condition, ir.SeqStmts(statements(func)[:-1], span), None, [], span),
            ir.ReturnStmt([func.params[1]], span),
        ],
        span,
    )
    function = ir.Function(func.name, func.params, func.return_types, body, span, type=ir.FunctionType.InCore)
    optimized = lower(ir.Program([function], "dynamic_shape", span))
    mlir = codegen.PTOCodegen().generate(optimized)
    assert mlir.count("pto.tload") == 1
    assert "rows=2, cols=512" in mlir
    ptoas = find_ptoas_binary()
    if ptoas is not None:
        source = tmp_path / "dynamic-shape.pto"
        source.write_text(mlir)
        for arch in ("a2", "a3"):
            run = subprocess.run(
                [
                    ptoas,
                    str(source),
                    f"--pto-arch={arch}",
                    "--pto-level=level3",
                    "--enable-insert-sync",
                    "-o",
                    str(tmp_path / f"dynamic-shape-{arch}.cpp"),
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            assert run.returncode == 0, run.stdout + run.stderr + mlir


def test_subview_unknown_bounds_are_rejected_by_direct_codegen():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    span = ir.Span.unknown()
    unknown = ir.Var("unknown", ir.ScalarType(DataType.INDEX), span)
    with pytest.raises(ValueError, match="subview.*cannot prove"):
        codegen.PTOCodegen().generate(make_program(offset=unknown, extra_params=(unknown,)))


def test_subview_rejects_compiler_owned_root():
    program = make_program()
    func = next(iter(program.functions.values()))
    span = ir.Span.unknown()
    create = ir.op.tensor.create([8, 512], DataType.BF16)
    body = ir.SeqStmts([ir.AssignStmt(func.params[0], create, span), *statements(func)], span)
    bad = ir.Function(func.name, [func.params[1]], func.return_types, body, span, type=ir.FunctionType.InCore)
    with pytest.raises(ValueError, match="subview.*caller-owned"):
        passes.convert_tensor_to_tile_ops()(ir.Program([bad], "owned_root", span))


def test_subview_rejects_overflowing_address_arithmetic_before_conversion():
    span = ir.Span.unknown()
    source = ir.Var("cache", ir.TensorType([2**62, 512], DataType.BF16), span)
    call = ir.op.tensor.subview(source, [2, 512], [0, 0], steps=[2**61, 1])
    selected = ir.Var("selected", call.type, span)
    body = ir.SeqStmts([ir.AssignStmt(selected, call, span), ir.ReturnStmt([], span)], span)
    func = ir.Function("overflow", [source], [], body, span, type=ir.FunctionType.InCore)
    with pytest.raises(ValueError, match="subview.*overflow-free"):
        passes.convert_tensor_to_tile_ops()(ir.Program([func], "overflow", span))


def test_subview_assignment_cannot_discard_addressing_kind():
    span = ir.Span.unknown()
    func = next(iter(make_program().functions.values()))
    original = statements(func)[0]
    assert isinstance(original, ir.AssignStmt)
    canonical = ir.Var("invalid_alias", ir.TensorType([2, 512], DataType.BF16), span)
    body = ir.SeqStmts(
        [original, ir.AssignStmt(canonical, original.var, span), ir.ReturnStmt([func.params[1]], span)],
        span,
    )
    bad = ir.Function(func.name, func.params, func.return_types, body, span, type=ir.FunctionType.InCore)
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with pytest.raises(ValueError, match="subview.*discard affine"):
        codegen.PTOCodegen().generate(ir.Program([bad], "lost_addressing", span))


@pytest.mark.parametrize("control_flow", ["if", "while", "for"])
def test_subview_cannot_use_overflowed_control_flow_bounds(control_flow):
    span = ir.Span.unknown()
    row = ir.Var("row", ir.ScalarType(DataType.INDEX), span)
    # Mathematical cancellation gives 1. INT32 multiplication wraps to zero,
    # so the actual bound is -4294967295 and admits negative GM origins.
    integer = ir.ConstInt(65536, DataType.INT32, span)
    overflow = ir.Mul(integer, integer, DataType.INT32, span)
    bound = ir.Add(
        ir.Sub(
            ir.Cast(overflow, DataType.INDEX, span),
            ir.ConstInt(2**32, DataType.INDEX, span),
            DataType.INDEX,
            span,
        ),
        ir.ConstInt(1, DataType.INDEX, span),
        DataType.INDEX,
        span,
    )
    func = next(iter(make_program(offset=row).functions.values()))
    body_stmts = statements(func)[:-1]
    condition = ir.And(
        ir.Ge(row, bound, DataType.BOOL, span),
        ir.Le(row, ir.ConstInt(2, DataType.INDEX, span), DataType.BOOL, span),
        DataType.BOOL,
        span,
    )
    if control_flow == "if":
        region = ir.IfStmt(condition, ir.SeqStmts(body_stmts, span), None, [], span)
    elif control_flow == "while":
        region = ir.WhileStmt(
            condition, [], ir.SeqStmts([*body_stmts, ir.YieldStmt([], span)], span), [], span
        )
    else:
        region = ir.ForStmt(
            row,
            bound,
            ir.ConstInt(2, DataType.INDEX, span),
            ir.ConstInt(1, DataType.INDEX, span),
            [],
            ir.SeqStmts([*body_stmts, ir.YieldStmt([], span)], span),
            [],
            span,
        )
    body = ir.SeqStmts([region, ir.ReturnStmt([func.params[1]], span)], span)
    params = func.params if control_flow == "for" else [*func.params, row]
    bad = ir.Function(func.name, params, func.return_types, body, span, type=ir.FunctionType.InCore)
    with pytest.raises(ValueError, match="subview.*(cannot prove|overflow-free)"):
        passes.convert_tensor_to_tile_ops()(ir.Program([bad], "overflowed_facts", span))


def test_subview_rejects_narrow_overflow_hidden_by_widening_and_cancellation():
    span = ir.Span.unknown()
    integer = ir.ConstInt(65536, DataType.INT32, span)
    product = ir.Mul(integer, integer, DataType.INT32, span)
    widened = ir.Var("widened", ir.ScalarType(DataType.INDEX), span)
    offset = ir.Add(
        ir.Sub(widened, ir.ConstInt(2**32, DataType.INDEX, span), DataType.INDEX, span),
        ir.ConstInt(1, DataType.INDEX, span),
        DataType.INDEX,
        span,
    )
    func = next(iter(make_program(offset=offset).functions.values()))
    body = ir.SeqStmts(
        [ir.AssignStmt(widened, ir.Cast(product, DataType.INDEX, span), span), *statements(func)], span
    )
    bad = ir.Function(func.name, func.params, func.return_types, body, span, type=ir.FunctionType.InCore)
    with pytest.raises(ValueError, match="subview.*overflow-free"):
        passes.convert_tensor_to_tile_ops()(ir.Program([bad], "narrow_overflow", span))


def test_subview_nested_origins_and_view_alias_preserve_current_pointer():
    span = ir.Span.unknown()
    func = next(iter(make_program().functions.values()))
    original = statements(func)[0]
    assert isinstance(original, ir.AssignStmt)
    nested = ir.op.tensor.subview(original.var, [1, 496], [1, 16])
    child = ir.Var("child", nested.type, span)
    view = ir.op.tensor.view(child, layout=ir.TensorLayout.ND)
    alias = ir.Var("alias", view.type, span)
    load = ir.op.tile.load(alias, [0, 16], [1, 480], target_memory=ir.MemorySpace.Vec)
    tile = ir.Var("tile", load.type, span)
    store = ir.op.tile.store(tile, [0, 0], func.params[1])
    result = ir.Var("result", store.type, span)
    body = ir.SeqStmts(
        [
            original,
            ir.AssignStmt(child, nested, span),
            ir.AssignStmt(alias, view, span),
            ir.AssignStmt(tile, load, span),
            ir.AssignStmt(result, store, span),
            ir.ReturnStmt([result], span),
        ],
        span,
    )
    nested_func = ir.Function(
        func.name, func.params, func.return_types, body, span, type=ir.FunctionType.InCore
    )
    program = ir.Program([nested_func], "nested", span)
    namespace = {}
    exec(torch_codegen(nested_func), namespace)
    cache = torch.arange(8 * 512, dtype=torch.float32).reshape(8, 512).to(torch.bfloat16)
    output = torch.zeros((2, 512), dtype=torch.bfloat16)
    actual = namespace["copy_selected"](cache, output)
    torch.testing.assert_close(actual[0, :480], cache[4, 32:], rtol=0, atol=0)
    optimized = lower(program)
    mlir = codegen.PTOCodegen().generate(optimized)
    assert mlir.count("pto.addptr") == 2
    assert mlir.count("pto.tload") == 1


@pytest.mark.parametrize(
    "dtype,parent_shape,shape,steps,reason",
    [
        (DataType.FP32, [8, 512], [2, 512], [3, 1], "rank-2 ND BF16"),
        (DataType.BF16, [8, 512], [2, 256], [3, 2], "unit inner stride"),
        (DataType.BF16, [8, 512], [2, 511], [3, 1], "row-byte alignment"),
        (DataType.BF16, [10000, 16], [5000, 16], [1, 1], "row-count limit"),
    ],
)
def test_subview_backend_rejects_unsupported_native_loads(dtype, parent_shape, shape, steps, reason):
    span = ir.Span.unknown()
    source = ir.Var("cache", ir.TensorType(parent_shape, dtype), span)
    subview = ir.op.tensor.subview(source, shape, [0, 0], steps=steps)
    selected = ir.Var("selected", subview.type, span)
    load = ir.op.tile.load(selected, [0, 0], shape, target_memory=ir.MemorySpace.Vec)
    tile = ir.Var("tile", load.type, span)
    body = ir.SeqStmts(
        [ir.AssignStmt(selected, subview, span), ir.AssignStmt(tile, load, span), ir.ReturnStmt([], span)],
        span,
    )
    func = ir.Function("unsupported", [source], [], body, span, type=ir.FunctionType.InCore)
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with pytest.raises(ValueError, match=f"subview.*{reason}"):
        codegen.PTOCodegen().generate(ir.Program([func], "unsupported", span))


def test_subview_return_is_rejected():
    program = make_program()
    func = next(iter(program.functions.values()))
    stmts = statements(func)
    assert isinstance(stmts[0], ir.AssignStmt)
    selected = stmts[0].var
    span = ir.Span.unknown()
    body = ir.SeqStmts([stmts[0], ir.ReturnStmt([selected], span)], span)
    bad = ir.Function(func.name, func.params, [selected.type], body, span, type=ir.FunctionType.InCore)
    with pytest.raises(ValueError, match="subview.*escape"):
        passes.convert_tensor_to_tile_ops()(ir.Program([bad], "bad_escape", span))


def test_subview_writer_is_rejected_before_lowering():
    program = make_program()
    func = next(iter(program.functions.values()))
    stmts = statements(func)
    assert isinstance(stmts[0], ir.AssignStmt)
    selected = stmts[0].var
    span = ir.Span.unknown()
    write = ir.op.tensor.write(selected, [0, 0], ir.ConstFloat(0, DataType.BF16, span))
    result = ir.Var("written", write.type, span)
    body = ir.SeqStmts(
        [stmts[0], ir.AssignStmt(result, write, span), ir.ReturnStmt([func.params[1]], span)], span
    )
    bad = ir.Function(func.name, func.params, func.return_types, body, span, type=ir.FunctionType.InCore)
    with pytest.raises(ValueError, match="subview.*read-only"):
        passes.convert_tensor_to_tile_ops()(ir.Program([bad], "bad_write", span))


def test_subview_transpose_requires_explicit_materialization():
    span = ir.Span.unknown()
    func = next(iter(make_program().functions.values()))
    definition = statements(func)[0]
    assert isinstance(definition, ir.AssignStmt)
    call = ir.op.tensor.transpose(definition.var, 0, 1)
    result = ir.Var("transposed", call.type, span)
    body = ir.SeqStmts(
        [definition, ir.AssignStmt(result, call, span), ir.ReturnStmt([func.params[1]], span)], span
    )
    bad = ir.Function(func.name, func.params, func.return_types, body, span, type=ir.FunctionType.InCore)
    with pytest.raises(ValueError, match="subview.*explicit load"):
        passes.convert_tensor_to_tile_ops()(ir.Program([bad], "transpose", span))


@pytest.mark.parametrize("rows", [2, 4])
def test_subview_view_symbolic_shape_must_preserve_element_count(rows):
    span = ir.Span.unknown()
    size = ir.Var("size", ir.ScalarType(DataType.INDEX), span)
    func = next(iter(make_program(step=1).functions.values()))
    definition = statements(func)[0]
    assert isinstance(definition, ir.AssignStmt)
    call = ir.op.tensor.view(definition.var, [size, 512])
    alias = ir.Var("alias", call.type, span)
    condition = ir.Eq(size, ir.ConstInt(rows, DataType.INDEX, span), DataType.BOOL, span)
    # No load is needed to expose the contract violation: view itself must not
    # enlarge a borrowed window merely because its shape is symbolic at build.
    body = ir.SeqStmts(
        [
            ir.IfStmt(
                condition,
                ir.SeqStmts([definition, ir.AssignStmt(alias, call, span)], span),
                None,
                [],
                span,
            ),
            ir.ReturnStmt([func.params[1]], span),
        ],
        span,
    )
    guarded = ir.Function(
        func.name, [*func.params, size], func.return_types, body, span, type=ir.FunctionType.InCore
    )
    program = ir.Program([guarded], "symbolic_view", span)
    if rows == 2:
        passes.convert_tensor_to_tile_ops()(program)
    else:
        with pytest.raises(ValueError, match="subview.*(cannot prove|element count)"):
            passes.convert_tensor_to_tile_ops()(program)


def test_subview_print_parse_roundtrip_keeps_keyword_steps_and_addressing():
    program = make_program()
    printed = ir.python_print(program)
    assert "steps=" in printed
    replayed = pl.parse_program(printed)
    ir.assert_structural_equal(program, replayed)


def test_subview_scalar_phi_geometry_is_proven_from_both_branches():
    span = ir.Span.unknown()
    merged = ir.Var("lo", ir.ScalarType(DataType.INDEX), span)
    flag = ir.Var("flag", ir.ScalarType(DataType.BOOL), span)
    program = make_program(offset=merged)
    func = next(iter(program.functions.values()))
    choice = ir.IfStmt(
        flag,
        ir.YieldStmt([ir.ConstInt(1, DataType.INDEX, span)], span),
        ir.YieldStmt([ir.ConstInt(2, DataType.INDEX, span)], span),
        [merged],
        span,
    )
    body = ir.SeqStmts([choice, *statements(func)], span)
    guarded = ir.Function(
        func.name, [*func.params, flag], func.return_types, body, span, type=ir.FunctionType.InCore
    )
    optimized = lower(ir.Program([guarded], "scalar_phi", span))
    assert "pto.addptr" in codegen.PTOCodegen().generate(optimized)


def test_subview_loop_local_device_offset_is_supported():
    span = ir.Span.unknown()
    row = ir.Var("row", ir.ScalarType(DataType.INDEX), span)
    program = make_program(offset=row)
    func = next(iter(program.functions.values()))
    loop_body = ir.SeqStmts([*statements(func)[:-1], ir.YieldStmt([], span)], span)
    loop = ir.ForStmt(
        row,
        ir.ConstInt(0, DataType.INDEX, span),
        ir.ConstInt(4, DataType.INDEX, span),
        ir.ConstInt(1, DataType.INDEX, span),
        [],
        loop_body,
        [],
        span,
    )
    body = ir.SeqStmts([loop, ir.ReturnStmt([func.params[1]], span)], span)
    guarded = ir.Function(func.name, func.params, func.return_types, body, span, type=ir.FunctionType.InCore)
    optimized = lower(ir.Program([guarded], "loop_local", span))
    mlir = codegen.PTOCodegen().generate(optimized)
    assert "scf.for" in mlir
    assert mlir.count("pto.addptr") == 1


def test_subview_worked_jit_example_specializes_and_outlines():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    cache = torch.zeros((8, 512), dtype=torch.bfloat16)
    output = torch.zeros((2, 512), dtype=torch.bfloat16)
    optimized = copy_selected_rows.lower(cache, output)
    kernels = [func for func in optimized.functions.values() if ir.is_incore_type(func.func_type)]
    assert kernels
    mlir = codegen.PTOCodegen().generate(ir.Program(kernels, "worked_example", optimized.span))
    assert mlir.count("pto.tload") == 1
    assert mlir.count("pto.addptr") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
