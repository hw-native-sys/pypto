# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GM subviews use ordinary Tensor descriptors and shared MemRef storage."""

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
    dtype: DataType = DataType.BF16,
):
    span = ir.Span.unknown()
    cache = ir.Var("cache", ir.TensorType([8, 512], dtype), span)
    output = ir.Var("output", ir.TensorType([2, 512], dtype), span)
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


def lower(program, backend_type=BackendType.Ascend910):
    backend.reset_for_testing()
    backend.set_backend_type(backend_type)
    return PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)


@pytest.mark.parametrize("sliced", [False, True])
def test_subview_pipeline_emits_only_result_tile(tmp_path, sliced):
    optimized = lower(make_program(sliced=sliced))
    selected_view: set[str] = set()
    view_memrefs = []
    root_memrefs = []

    class CheckSubviewStorage(ir.IRVisitor):
        def visit_var(self, op: ir.Var) -> None:
            tensor = op.type
            if isinstance(tensor, ir.TensorType):
                if op.name_hint.startswith("cache") and tensor.memref is not None:
                    root_memrefs.append(tensor.memref)
                view = tensor.tensor_view
                if view and op.name_hint.startswith(("selected", "narrowed")):
                    assert tensor.memref is not None
                    view_memrefs.append(tensor.memref)
                    selected_view.add(op.name_hint)
            super().visit_var(op)

    CheckSubviewStorage().visit_program(optimized)
    assert selected_view
    assert root_memrefs
    assert all(memref.base_ is root_memrefs[0].base_ for memref in view_memrefs)
    assert all(memref.byte_offset_ == 1024 for memref in view_memrefs)
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


def test_subview_direct_codegen_derives_origin_without_memrefs():
    program = make_program()
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)

    mlir = codegen.PTOCodegen().generate(program)

    assert mlir.count("pto.addptr") == 1
    assert "512" in mlir


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
    selected_view = namespace["_tensor_subview"](cache, [2, 512], [1, 0], [3, 1])
    assert selected_view.stride() == (1536, 1)
    cache[1, 0] = 127
    assert selected_view[0, 0] == 127


def test_subview_unknown_bounds_are_allowed_by_direct_conversion():
    span = ir.Span.unknown()
    unknown = ir.Var("unknown", ir.ScalarType(DataType.INDEX), span)
    program = make_program(offset=unknown, extra_params=(unknown,))
    converted = passes.convert_tensor_to_tile_ops()(program)
    assert "tensor.subview" in str(converted)


def test_subview_validshape_remains_a_gm_tensor():
    optimized = lower(make_program(narrowed=True))
    assert "tensor.set_validshape" in str(optimized)
    assert "tile.set_validshape" not in str(optimized)
    assert "pto.tload" in codegen.PTOCodegen().generate(optimized)


def test_subview_validshape_may_preserve_full_parent_validity():
    optimized = lower(make_program(narrowed=True, valid_rows=2))
    assert "pto.tload" in codegen.PTOCodegen().generate(optimized)


@pytest.mark.parametrize("implicit_view", [False, True])
def test_canonical_validshape_alias_can_be_a_subview_parent(implicit_view):
    span = ir.Span.unknown()
    cache = ir.Var("cache", ir.TensorType([8, 512], DataType.BF16), span)
    output = ir.Var("output", ir.TensorType([2, 512], DataType.BF16), span)
    if implicit_view:
        cache = ir.Var(
            "cache",
            ir.TensorType(
                [8, 512], DataType.BF16, tensor_view=ir.TensorView([], ir.TensorLayout.ND, [6, 512])
            ),
            span,
        )
    valid_call = ir.op.tensor.set_validshape(cache, 5, 512)
    valid = ir.Var("valid", valid_call.type, span)
    subview_call = ir.op.tensor.subview(valid, [2, 512], [1, 0], steps=[3, 1])
    selected = ir.Var("selected", subview_call.type, span)
    load = ir.op.tile.load(selected, [0, 0], [2, 512], target_memory=ir.MemorySpace.Vec)
    tile = ir.Var("tile", load.type, span)
    store = ir.op.tile.store(tile, [0, 0], output)
    stored = ir.Var("stored", store.type, span)
    function = ir.Function(
        "canonical_validshape_subview",
        [cache, output],
        [output.type],
        ir.SeqStmts(
            [
                ir.AssignStmt(valid, valid_call, span),
                ir.AssignStmt(selected, subview_call, span),
                ir.AssignStmt(tile, load, span),
                ir.AssignStmt(stored, store, span),
                ir.ReturnStmt([stored], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.InCore,
    )

    optimized = lower(ir.Program([function], "canonical_validshape_subview", span))
    text = str(optimized)
    assert "tensor.set_validshape" in text
    assert "tile.set_validshape" not in text
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
    optimized = lower(ir.Program([function], "dynamic_shape", span) if guarded else program)
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


def test_subview_unknown_bounds_are_allowed_by_direct_codegen():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    span = ir.Span.unknown()
    unknown = ir.Var("unknown", ir.ScalarType(DataType.INDEX), span)
    mlir = codegen.PTOCodegen().generate(make_program(offset=unknown, extra_params=(unknown,)))
    assert mlir.count("pto.addptr") == 1


def test_subview_requires_gm_after_tensor_to_tile_like_view():
    program = make_program()
    func = next(iter(program.functions.values()))
    span = ir.Span.unknown()
    create = ir.op.tensor.create([8, 512], DataType.BF16)
    body = ir.SeqStmts([ir.AssignStmt(func.params[0], create, span), *statements(func)], span)
    bad = ir.Function(func.name, [func.params[1]], func.return_types, body, span, type=ir.FunctionType.InCore)
    # tensor.create becomes a Tile; both view APIs require a GM Tensor input.
    with pytest.raises(ValueError, match="requires a GM Tensor input"):
        passes.convert_tensor_to_tile_ops()(ir.Program([bad], "owned_root", span))


def test_subview_assignment_preserves_memref_alias():
    span = ir.Span.unknown()
    func = next(iter(make_program().functions.values()))
    original = statements(func)[0]
    assert isinstance(original, ir.AssignStmt)
    alias = ir.Var("selected_alias", original.var.type, span)
    load_assign = statements(func)[1]
    assert isinstance(load_assign, ir.AssignStmt)
    load = load_assign.value
    assert isinstance(load, ir.Call)
    aliased_load = ir.Call(load.op, [alias, *load.args[1:]], load.kwargs, load.type, span)
    body = ir.SeqStmts(
        [
            original,
            ir.AssignStmt(alias, original.var, span),
            ir.AssignStmt(load_assign.var, aliased_load, span),
            *statements(func)[2:],
        ],
        span,
    )
    aliased = ir.Function(func.name, func.params, func.return_types, body, span, type=ir.FunctionType.InCore)
    optimized = lower(ir.Program([aliased], "aliased_subview", span))
    selected_memrefs = []

    class CollectSelectedMemRefs(ir.IRVisitor):
        def visit_var(self, op: ir.Var) -> None:
            if op.name_hint.startswith("selected") and isinstance(op.type, ir.TensorType):
                if op.type.memref is not None:
                    selected_memrefs.append(op.type.memref)
            super().visit_var(op)

    CollectSelectedMemRefs().visit_program(optimized)
    assert len(selected_memrefs) >= 2
    assert all(memref.base_ is selected_memrefs[0].base_ for memref in selected_memrefs)
    assert all(memref.byte_offset_ == 1024 for memref in selected_memrefs)


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


@pytest.mark.parametrize("backend_type", [BackendType.Ascend910, BackendType.Ascend950])
@pytest.mark.parametrize(
    "dtype,parent_shape,shape,steps",
    [
        (DataType.FP32, [8, 512], [2, 512], [3, 1]),
        (DataType.BF16, [8, 512], [2, 256], [3, 1]),
        (DataType.BF16, [8, 512], [2, 511], [3, 1]),
        (DataType.BF16, [10000, 16], [5000, 16], [1, 1]),
        (DataType.BF16, [8, (1 << 32) + 16], [2, 16], [1, 1]),
        (DataType.FP32, [4, 8, 16], [2, 4, 16], [1, 1, 1]),
    ],
)
def test_subview_descriptor_has_no_native_load_profile(backend_type, dtype, parent_shape, shape, steps):
    """Descriptor validity does not depend on a future consumer's DMA encoding."""
    span = ir.Span.unknown()
    source = ir.Var("cache", ir.TensorType(parent_shape, dtype), span)
    subview = ir.op.tensor.subview(source, shape, [0] * len(shape), steps=steps)
    selected = ir.Var("selected", subview.type, span)
    body = ir.SeqStmts([ir.AssignStmt(selected, subview, span), ir.ReturnStmt([], span)], span)
    func = ir.Function("describe", [source], [], body, span, type=ir.FunctionType.InCore)
    backend.reset_for_testing()
    backend.set_backend_type(backend_type)
    mlir = codegen.PTOCodegen().generate(ir.Program([func], "descriptor", span))
    assert "pto.make_tensor_view" in mlir
    assert "pto.tload" not in mlir
    assert "pto.alloc_tile" not in mlir


def test_subview_strided_columns_are_rejected_before_load():
    span = ir.Span.unknown()
    source = ir.Var("cache", ir.TensorType([8, 512], DataType.BF16), span)
    with pytest.raises(ValueError, match="subview.*contiguous inner storage axis"):
        ir.op.tensor.subview(source, [2, 256], [0, 0], steps=[3, 2])


@pytest.mark.parametrize("backend_type,arch", [(BackendType.Ascend910, "a3"), (BackendType.Ascend950, "a5")])
def test_subview_fp32_load_uses_existing_backend_path(tmp_path, backend_type, arch):
    optimized = lower(make_program(dtype=DataType.FP32), backend_type)
    mlir = codegen.PTOCodegen().generate(optimized)
    assert mlir.count("pto.tload") == 1
    assert mlir.count("pto.tstore") == 1
    ptoas = find_ptoas_binary()
    if ptoas is not None:
        source = tmp_path / "fp32-subview.pto"
        source.write_text(mlir)
        result = subprocess.run(
            [
                ptoas,
                str(source),
                f"--pto-arch={arch}",
                "--pto-level=level3",
                "-o",
                str(tmp_path / "fp32-subview.cpp"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("direct_codegen", [False, True])
def test_subview_scalar_read_uses_ordinary_tensor_path(direct_codegen):
    span = ir.Span.unknown()
    func = next(iter(make_program().functions.values()))
    definition = statements(func)[0]
    assert isinstance(definition, ir.AssignStmt)
    read = ir.op.tensor.read(definition.var, [1, 0])
    body = ir.SeqStmts(
        [definition, ir.AssignStmt(ir.Var("scalar", read.type, span), read, span), ir.ReturnStmt([], span)],
        span,
    )
    function = ir.Function("scalar_read", func.params, [], body, span, type=ir.FunctionType.InCore)
    program = ir.Program([function], "scalar_read", span)
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    if not direct_codegen:
        program = passes.convert_tensor_to_tile_ops()(program)
    mlir = codegen.PTOCodegen().generate(program)
    assert "pto.load" in mlir
    assert "pto.addptr" in mlir
    read_line = next(line for line in mlir.splitlines() if " = pto.load " in line)
    assert "[%c1536_index]" in read_line


@pytest.mark.parametrize("backend_type,arch", [(BackendType.Ascend910, "a3"), (BackendType.Ascend950, "a5")])
@pytest.mark.parametrize("valid", [None, [1, 8, 16]])
def test_subview_nd_load_survives_tile_flattening(tmp_path, backend_type, arch, valid):
    span = ir.Span.unknown()
    source = ir.Var("source", ir.TensorType([4, 8, 16], DataType.FP32), span)
    subview = ir.op.tensor.subview(source, [2, 8, 16], [0, 0, 0], steps=[2, 1, 1])
    selected = ir.Var("selected", subview.type, span)
    load = ir.op.tile.load(
        selected, [0, 0, 0], [2, 8, 16], valid_shape=valid, target_memory=ir.MemorySpace.Vec
    )
    body = ir.SeqStmts(
        [
            ir.AssignStmt(selected, subview, span),
            ir.AssignStmt(ir.Var("tile", load.type, span), load, span),
            ir.ReturnStmt([], span),
        ],
        span,
    )
    function = ir.Function("nd_load", [source], [], body, span, type=ir.FunctionType.InCore)
    optimized = lower(ir.Program([function], "nd_load", span), backend_type)
    loads: list[ir.Call] = []

    class FindLoad(ir.IRVisitor):
        def visit_call(self, op: ir.Call) -> None:
            if op.op.name == ir.get_op("tile.load").name:
                loads.append(op)
            super().visit_call(op)

    FindLoad().visit_program(optimized)
    assert len(loads) == 1
    tile_type = loads[0].type
    assert isinstance(tile_type, ir.TileType)
    for dim, expected in zip(tile_type.shape, [16, 16], strict=True):
        assert isinstance(dim, ir.ConstInt) and dim.value == expected
    expected_valid = [16 if valid is None else 8, 16]
    for dim, expected in zip(tile_type.get_effective_tile_view().valid_shape, expected_valid, strict=True):
        assert isinstance(dim, ir.ConstInt) and dim.value == expected
    mlir = codegen.PTOCodegen().generate(optimized)
    assert mlir.count("pto.tload") == 1
    ptoas = find_ptoas_binary()
    if ptoas is not None:
        path = tmp_path / "nd-subview.pto"
        path.write_text(mlir)
        result = subprocess.run(
            [ptoas, str(path), f"--pto-arch={arch}", "--pto-level=level3", "-o", str(tmp_path / "nd.cpp")],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr + mlir


@pytest.mark.parametrize("dtype", [DataType.FP4E2M1X2, DataType.FP4E1M2X2])
@pytest.mark.parametrize("valid", [None, [16, 64]])
def test_subview_fp4_load_checks_source_window_before_packing(tmp_path, dtype, valid):
    """ND carrier rows/columns differ from the Mat destination packing axes."""
    span = ir.Span.unknown()
    source = ir.Var("source", ir.TensorType([64, 64], dtype), span)
    subview = ir.op.tensor.subview(source, [32, 64], [0, 0])
    selected = ir.Var("selected", subview.type, span)
    load = ir.op.tile.load(selected, [0, 0], [32, 64], valid_shape=valid, target_memory=ir.MemorySpace.Mat)
    body = ir.SeqStmts(
        [
            ir.AssignStmt(selected, subview, span),
            ir.AssignStmt(ir.Var("tile", load.type, span), load, span),
            ir.ReturnStmt([], span),
        ],
        span,
    )
    function = ir.Function("fp4_load", [source], [], body, span, type=ir.FunctionType.InCore)
    optimized = lower(ir.Program([function], "fp4_load", span), BackendType.Ascend950)
    mlir = codegen.PTOCodegen().generate(optimized)
    assert mlir.count("pto.tload") == 1
    assert "rows=16, cols=128" in mlir
    ptoas = find_ptoas_binary()
    if ptoas is not None:
        path = tmp_path / "fp4-subview.pto"
        path.write_text(mlir)
        result = subprocess.run(
            [ptoas, str(path), "--pto-arch=a5", "--pto-level=level3", "-o", str(tmp_path / "fp4.cpp")],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr + mlir


def test_subview_return_uses_ordinary_output_conversion():
    program = make_program()
    func = next(iter(program.functions.values()))
    stmts = statements(func)
    assert isinstance(stmts[0], ir.AssignStmt)
    selected = stmts[0].var
    span = ir.Span.unknown()
    body = ir.SeqStmts([stmts[0], ir.ReturnStmt([selected], span)], span)
    bad = ir.Function(func.name, func.params, [selected.type], body, span, type=ir.FunctionType.InCore)
    converted = passes.convert_tensor_to_tile_ops()(ir.Program([bad], "return_view", span))
    assert isinstance(converted, ir.Program)


def test_subview_writer_uses_ordinary_tensor_path():
    program = make_program()
    func = next(iter(program.functions.values()))
    stmts = statements(func)
    assert isinstance(stmts[0], ir.AssignStmt)
    selected = stmts[0].var
    span = ir.Span.unknown()
    write = ir.op.tensor.write(selected, [1, 0], ir.ConstFloat(0, DataType.BF16, span))
    result = ir.Var("written", write.type, span)
    body = ir.SeqStmts(
        [stmts[0], ir.AssignStmt(result, write, span), ir.ReturnStmt([func.params[1]], span)], span
    )
    bad = ir.Function(func.name, func.params, func.return_types, body, span, type=ir.FunctionType.InCore)
    converted = passes.convert_tensor_to_tile_ops()(ir.Program([bad], "write_view", span))
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    mlir = codegen.PTOCodegen().generate(converted)
    assert "pto.store" in mlir
    assert "pto.addptr" in mlir
    write_line = next(line for line in mlir.splitlines() if "pto.store " in line)
    assert "[%c1536_index]" in write_line


def test_subview_transpose_uses_existing_conversion():
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
    converted = passes.convert_tensor_to_tile_ops()(ir.Program([bad], "transpose", span))
    assert "tile.transpose" in str(converted)


def test_subview_does_not_strengthen_unrelated_canonical_view_contract():
    span = ir.Span.unknown()
    size = ir.Var("size", ir.ScalarType(DataType.INDEX), span)
    func = next(iter(make_program().functions.values()))
    view_call = ir.op.tensor.view(func.params[0], [size, 512])
    unused = ir.Var("unrelated_view", view_call.type, span)
    body = ir.SeqStmts([ir.AssignStmt(unused, view_call, span), *statements(func)], span)
    function = ir.Function(
        func.name, [*func.params, size], func.return_types, body, span, type=ir.FunctionType.InCore
    )
    converted = passes.convert_tensor_to_tile_ops()(ir.Program([function], "unrelated_view", span))
    assert isinstance(converted, ir.Program)


def test_subview_print_parse_roundtrip_keeps_keyword_steps():
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


def test_subview_dynamic_offsets_use_ordinary_tensor_path():
    span = ir.Span.unknown()
    offset = ir.Var("offset", ir.ScalarType(DataType.INDEX), span)
    optimized = lower(make_program(offset=offset, extra_params=(offset,)))
    mlir = codegen.PTOCodegen().generate(optimized)
    assert mlir.count("pto.addptr") == 1
    assert mlir.count("pto.tload") == 1


def test_gm_validshape_does_not_require_subview_provenance():
    span = ir.Span.unknown()
    source = ir.Var("source", ir.TensorType([8, 512], DataType.BF16), span)
    valid = ir.op.tensor.set_validshape(source, 6, 512)
    alias = ir.Var("alias", valid.type, span)
    load = ir.op.tile.load(alias, [0, 0], [2, 512], target_memory=ir.MemorySpace.Vec)
    function = ir.Function(
        "gm_metadata",
        [source],
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(alias, valid, span),
                ir.AssignStmt(ir.Var("tile", load.type, span), load, span),
                ir.ReturnStmt([], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.InCore,
    )
    converted = passes.convert_tensor_to_tile_ops()(ir.Program([function], "gm_metadata", span))
    text = str(converted)
    assert "tensor.set_validshape" in text
    assert "tile.set_validshape" not in text
    assert "tile.load" in text


@pytest.mark.parametrize("forged_stride", [False, True])
def test_explicit_subview_call_rejects_static_geometry_mismatch(forged_stride):
    span = ir.Span.unknown()
    source = ir.Var("source", ir.TensorType([8, 512], DataType.BF16), span)
    call = ir.op.tensor.subview(source, [2, 512], [1, 0], steps=[3, 1])
    result_type = ir.TensorType(
        [2, 512] if forged_stride else [3, 512],
        DataType.BF16,
        tensor_view=ir.TensorView([2048, 1] if forged_stride else [1536, 1], ir.TensorLayout.ND),
    )
    forged = ir.Call(call.op, call.args, result_type, span)
    function = ir.Function(
        "bad_geometry",
        [source],
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(ir.Var("selected", result_type, span), forged, span),
                ir.ReturnStmt([], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.InCore,
    )
    with pytest.raises(ValueError, match="subview result metadata does not match"):
        passes.convert_tensor_to_tile_ops()(ir.Program([function], "bad_geometry", span))


def test_subview_can_flow_through_ordinary_tensor_yield():
    span = ir.Span.unknown()
    func = next(iter(make_program().functions.values()))
    definition = statements(func)[0]
    assert isinstance(definition, ir.AssignStmt)
    merged = ir.Var("merged", definition.var.type, span)
    flag = ir.Var("flag", ir.ScalarType(DataType.BOOL), span)
    choice = ir.IfStmt(
        flag,
        ir.YieldStmt([definition.var], span),
        ir.YieldStmt([definition.var], span),
        [merged],
        span,
    )
    load = ir.op.tile.load(merged, [0, 0], [2, 512], target_memory=ir.MemorySpace.Vec)
    function = ir.Function(
        "yield_view",
        [*func.params, flag],
        [],
        ir.SeqStmts(
            [
                definition,
                choice,
                ir.AssignStmt(ir.Var("tile", load.type, span), load, span),
                ir.ReturnStmt([], span),
            ],
            span,
        ),
        span,
        type=ir.FunctionType.InCore,
    )
    converted = passes.convert_tensor_to_tile_ops()(ir.Program([function], "yield_view", span))
    assert "tensor.subview" in str(converted)
    assert "tile.load" in str(converted)


def test_subview_validshape_before_slice_keeps_gm_metadata():
    converted = passes.convert_tensor_to_tile_ops()(make_program(narrowed=True, sliced=True))
    text = str(converted)
    assert "tensor.set_validshape" in text
    assert "tile.set_validshape" not in text
    assert text.count("tile.load(") == 1
