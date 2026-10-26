# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTOAS destinations keep each variable's type when storage is shared (#3017)."""

import re
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import ir
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.pypto_core import codegen


def _generate(
    program: ir.Program,
    *,
    backend_type=BackendType.Ascend910,
    emit_tile_addr: bool = False,
) -> str:
    reset_for_testing()
    set_backend_type(backend_type)
    return codegen.PTOCodegen().generate(program, emit_tile_addr=emit_tile_addr)


def _line(mlir: str, text: str) -> str:
    matches = [line for line in mlir.splitlines() if text in line]
    assert len(matches) == 1, mlir
    return matches[0]


def _operands(line: str, direction: str) -> tuple[list[str], list[str]]:
    match = re.search(rf"\b{direction}\(([^)]*)\)", line)
    assert match, line
    names, annotations = match.group(1).split(" : ", maxsplit=1)
    return re.findall(r"%[A-Za-z0-9_]+", names), re.findall(r"!pto\.tile_buf<[^>]+>", annotations)


def _definitions(mlir: str) -> dict[str, str]:
    definitions = {}
    for line in mlir.splitlines():
        match = re.match(r"\s*(%[A-Za-z0-9_]+) = pto\.(?:alloc_tile|treshape)\b", line)
        if match:
            definitions[match.group(1)] = re.findall(r"!pto\.tile_buf<[^>]+>", line)[-1]
    return definitions


def _assert_operand_declarations(line: str, definitions: dict[str, str]) -> None:
    for direction in ("ins", "outs"):
        operands, annotations = _operands(line, direction)
        assert len(operands) == len(annotations), line
        for operand, annotation in zip(operands, annotations):
            assert definitions[operand] == annotation, line


def _elementwise_program(
    operation: str,
    root_shape: tuple[int, int],
    result_shape: tuple[int, int],
    *,
    dynamic_valid: bool = False,
    with_consumer: bool = False,
    consume_root: bool = False,
    source_shares_root: bool = False,
) -> ir.Program:
    """Isolate destination selection from control flow and layout repair.

    Both row/column directions exercise the type resolver. Real PTOAS tests
    below use row-major arithmetic operands, as the normal layout pass requires.
    """
    span = ir.Span.unknown()
    zero = ir.ConstInt(0, pl.INDEX, span)
    valid_rows = ir.Var("valid_rows", ir.ScalarType(pl.INDEX), span)
    layout = ir.TileLayout.col_major if result_shape[1] == 1 else ir.TileLayout.row_major
    view = ir.TileView(valid_shape=[valid_rows, result_shape[1]] if dynamic_valid else None, blayout=layout)

    def tile_type(name: str, shape: tuple[int, int], tile_view=None) -> ir.TileType:
        storage = ir.MemRef(ir.Var(name, ir.PtrType(), span), zero, shape[0] * shape[1] * 4, span)
        return ir.TileType(shape, pl.FP32, storage, tile_view, ir.MemorySpace.Vec)

    root_type = tile_type("root_base", root_shape)
    source_shape = (32, 32) if operation == "col_max" else result_shape
    source_type = (
        ir.TileType(source_shape, pl.FP32, root_type.memref, view, ir.MemorySpace.Vec)
        if source_shares_root
        else tile_type("source_base", source_shape, None if operation == "col_max" else view)
    )
    destination_type = ir.TileType(result_shape, pl.FP32, root_type.memref, view, ir.MemorySpace.Vec)
    root = ir.Var("root", root_type, span)
    source = ir.Var("source", source_type, span)
    destination = ir.Var("destination", destination_type, span)
    output = ir.Var("output", ir.TensorType(result_shape, pl.FP32), span)
    stored = ir.Var("stored", output.type, span)
    offsets = ir.MakeTuple([zero, zero], span)
    args = [source, source] if operation == "add" else [source]
    statements: list[ir.Stmt] = [
        ir.AssignStmt(root, ir.Call(ir.Op("tile.create"), [], root_type, span), span),
        ir.AssignStmt(source, ir.Call(ir.Op("tile.create"), [], source_type, span), span),
        ir.AssignStmt(destination, ir.Call(ir.Op(f"tile.{operation}"), args, destination_type, span), span),
    ]
    result = destination
    if with_consumer:
        consumer_type = tile_type("consumer_base", root_shape if consume_root else result_shape)
        consumer = ir.Var("consumer", consumer_type, span)
        statements.append(
            ir.AssignStmt(
                consumer,
                ir.Call(ir.Op("tile.exp"), [root if consume_root else destination], consumer_type, span),
                span,
            )
        )
        if not consume_root:
            result = consumer
    statements.extend(
        [
            ir.AssignStmt(
                stored, ir.Call(ir.Op("tile.store"), [result, offsets, output], output.type, span), span
            ),
            ir.ReturnStmt([stored], span),
        ]
    )
    params = [(output, ir.ParamDirection.Out)]
    if dynamic_valid:
        params.append((valid_rows, ir.ParamDirection.In))
    function = ir.Function(
        "kernel", params, [output.type], ir.SeqStmts(statements, span), span, ir.FunctionType.InCore
    )
    return ir.Program([function], "SharedStorageDestination", span)


@pytest.mark.parametrize("operation", ["add", "exp"])
@pytest.mark.parametrize(
    ("root_shape", "result_shape"),
    [((1, 32), (32, 1)), ((32, 1), (1, 32)), ((4, 8), (8, 4))],
    ids=["row-to-column", "column-to-row", "matrix-to-matrix"],
)
def test_elementwise_output_uses_its_own_shape(operation, root_shape, result_shape):
    mlir = _generate(_elementwise_program(operation, root_shape, result_shape))
    producer = _line(mlir, f"pto.t{operation} ins(")
    _assert_operand_declarations(producer, _definitions(mlir))
    destinations, types = _operands(producer, "outs")
    rows, cols = result_shape
    assert f"rows={rows}, cols={cols}," in types[0], producer
    assert f"{destinations[0]} = pto.treshape %root " in mlir, mlir
    assert _operands(_line(mlir, "pto.tstore"), "ins")[0] == destinations, mlir
    assert mlir.count("= pto.alloc_tile") == 2, mlir


def test_reduction_output_uses_its_own_shape():
    mlir = _generate(_elementwise_program("col_max", (32, 1), (1, 32)))
    producer = _line(mlir, "pto.tcolmax ins(")
    _assert_operand_declarations(producer, _definitions(mlir))
    destinations, types = _operands(producer, "outs")
    assert "rows=1, cols=32," in types[0], producer
    assert f"{destinations[0]} = pto.treshape %root " in mlir, mlir
    assert _operands(_line(mlir, "pto.tstore"), "ins")[0] == destinations, mlir
    assert mlir.count("= pto.alloc_tile") == 2, mlir


def test_later_operation_reads_the_correct_output_view():
    mlir = _generate(_elementwise_program("add", (1, 32), (32, 1), with_consumer=True))
    producer = _line(mlir, "pto.tadd ins(")
    consumer = _line(mlir, "pto.texp ins(")
    definitions = _definitions(mlir)
    _assert_operand_declarations(producer, definitions)
    _assert_operand_declarations(consumer, definitions)
    destinations, types = _operands(producer, "outs")
    assert "rows=32, cols=1," in types[0], producer
    assert _operands(consumer, "ins") == (destinations, types), mlir
    assert mlir.count("= pto.alloc_tile") == 3, mlir


def test_destination_view_does_not_change_the_original_variable_binding():
    program = _elementwise_program("add", (1, 32), (32, 1), with_consumer=True, consume_root=True)
    mlir = _generate(program)
    producer = _line(mlir, "pto.tadd ins(")
    consumer = _line(mlir, "pto.texp ins(")
    definitions = _definitions(mlir)
    _assert_operand_declarations(producer, definitions)
    _assert_operand_declarations(consumer, definitions)
    destinations, types = _operands(producer, "outs")
    assert "rows=32, cols=1," in types[0], producer
    source_names, source_types = _operands(consumer, "ins")
    assert source_names == ["%root"], consumer
    assert "rows=1, cols=32," in source_types[0], consumer
    assert _operands(_line(mlir, "pto.tstore"), "ins")[0] == destinations, mlir
    assert mlir.count("= pto.alloc_tile") == 3, mlir


def test_same_storage_move_keeps_its_own_shape_without_copying():
    program = _elementwise_program("move", (1, 32), (32, 1), source_shares_root=True)
    mlir = _generate(program)
    assert "pto.tmov" not in mlir, mlir
    assert mlir.count("= pto.alloc_tile") == 1, mlir
    store = _line(mlir, "pto.tstore")
    sources, types = _operands(store, "ins")
    assert len(sources) == len(types) == 1, store
    assert "rows=32, cols=1," in types[0], store
    assert _definitions(mlir)[sources[0]] == types[0], store
    assert f"{sources[0]} = pto.treshape %root " in mlir, mlir


@pytest.mark.parametrize("operation", ["add", "exp"])
def test_matching_output_type_does_not_create_a_view(operation):
    mlir = _generate(_elementwise_program(operation, (1, 32), (1, 32)))
    _assert_operand_declarations(_line(mlir, f"pto.t{operation} ins("), _definitions(mlir))
    assert "pto.treshape" not in mlir, mlir
    assert mlir.count("= pto.alloc_tile") == 2, mlir


def _packed_fp4_destination_program(root_dtype, result_dtype) -> ir.Program:
    """Load into a shared byte window whose existing descriptor may differ."""
    span = ir.Span.unknown()
    zero = ir.ConstInt(0, pl.INDEX, span)
    address = ir.ConstInt(0, pl.INT64, span)
    shape = (16, 32)
    storage = ir.MemRef(ir.MemorySpace.Vec, address, 512, 0)
    root_type = ir.TileType(shape, root_dtype, storage, None, ir.MemorySpace.Vec)
    result_type = ir.TileType(shape, result_dtype, storage, None, ir.MemorySpace.Vec)
    root = ir.Var("root", root_type, span)
    destination = ir.Var("destination", result_type, span)
    source = ir.Var("source", ir.TensorType(shape, result_dtype), span)
    output = ir.Var("output", source.type, span)
    stored = ir.Var("stored", output.type, span)
    offsets = ir.MakeTuple([zero, zero], span)
    shapes = ir.MakeTuple([ir.ConstInt(dim, pl.INDEX, span) for dim in shape], span)
    body = ir.SeqStmts(
        [
            ir.AssignStmt(root, ir.Call(ir.Op("tile.create"), [], root_type, span), span),
            ir.AssignStmt(
                destination,
                ir.Call(ir.Op("tile.load"), [source, offsets, shapes], result_type, span),
                span,
            ),
            ir.AssignStmt(
                stored,
                ir.Call(ir.Op("tile.store"), [destination, offsets, output], output.type, span),
                span,
            ),
            ir.ReturnStmt([stored], span),
        ],
        span,
    )
    function = ir.Function(
        "kernel",
        [(source, ir.ParamDirection.In), (output, ir.ParamDirection.Out)],
        [output.type],
        body,
        span,
        ir.FunctionType.InCore,
    )
    return ir.Program([function], "PackedFp4SharedDestination", span)


@pytest.mark.parametrize(
    ("root_dtype", "result_dtype"),
    [(pl.UINT8, pl.FP4E2M1X2), (pl.FP4E2M1X2, pl.UINT8)],
    ids=["byte-to-packed", "packed-to-byte"],
)
def test_shared_destination_rejects_packed_fp4_treshape(root_dtype, result_dtype):
    # Equal carrier sizes do not make packed-FP4 treshape legal in PTOAS.
    program = _packed_fp4_destination_program(root_dtype, result_dtype)
    with pytest.raises(ValueError, match="FP4E2M1X2 tile aliases require PYPTO or DSA_RP addressed memory"):
        _generate(program, backend_type=BackendType.Ascend950)


def test_matching_packed_fp4_destination_does_not_create_a_view():
    program = _packed_fp4_destination_program(pl.FP4E2M1X2, pl.FP4E2M1X2)
    mlir = _generate(program, backend_type=BackendType.Ascend950)
    assert "pto.treshape" not in mlir, mlir
    assert mlir.count("= pto.alloc_tile") == 1, mlir
    targets, annotations = _operands(_line(mlir, "pto.tload"), "outs")
    assert targets == ["%root"], mlir
    assert "dtype=!pto.f4E2M1x2" in annotations[0], mlir
    assert _operands(_line(mlir, "pto.tstore"), "ins") == (targets, annotations), mlir


@pytest.mark.parametrize(
    ("root_dtype", "result_dtype"),
    [(pl.UINT8, pl.FP4E2M1X2), (pl.FP4E2M1X2, pl.UINT8)],
    ids=["byte-to-packed", "packed-to-byte"],
)
def test_addressed_packed_fp4_destination_keeps_its_own_allocation(root_dtype, result_dtype):
    # PYPTO and DSA_RP provide an address, so each dtype gets an alloc at that
    # same address instead of a treshape of the other dtype's descriptor.
    program = _packed_fp4_destination_program(root_dtype, result_dtype)
    mlir = _generate(program, backend_type=BackendType.Ascend950, emit_tile_addr=True)
    assert "pto.treshape" not in mlir, mlir
    allocations = [line for line in mlir.splitlines() if "= pto.alloc_tile" in line]
    assert len(allocations) == 2, mlir
    assert all("addr = %c0_i64" in line for line in allocations), mlir
    targets, annotations = _operands(_line(mlir, "pto.tload"), "outs")
    assert targets == ["%destination"], mlir
    assert _definitions(mlir)[targets[0]] == annotations[0], mlir
    assert _operands(_line(mlir, "pto.tstore"), "ins") == (targets, annotations), mlir


@pytest.mark.parametrize("operation", ["add", "exp"])
def test_shared_destination_initializes_dynamic_valid_shape(operation):
    program = _elementwise_program(operation, (1, 32), (32, 1), dynamic_valid=True)
    mlir = _generate(program)
    producer = _line(mlir, f"pto.t{operation} ins(")
    definitions = _definitions(mlir)
    _assert_operand_declarations(producer, definitions)
    destinations, annotations = _operands(producer, "outs")
    destination = destinations[0]
    view = _line(mlir, f"{destination} = pto.treshape %root ")
    assert view.endswith(f"-> {annotations[0]}"), mlir
    assert "rows=32, cols=1, v_row=?, v_col=?" in annotations[0], mlir
    initialized = _line(mlir, f"pto.set_validshape {destination},")
    source = _line(mlir, "%source = pto.alloc_tile")
    valid_rows = re.search(r"valid_row = (%[A-Za-z0-9_]+)", source)
    assert valid_rows is not None, source
    assert f"{valid_rows[1]}, %c1_index : {annotations[0]}" in initialized, mlir
    assert mlir.index(view) < mlir.index(initialized) < mlir.index(producer), mlir
    assert _operands(_line(mlir, "pto.tstore"), "ins") == (destinations, annotations), mlir
    assert "pto.set_validshape %root," not in mlir, mlir
    assert mlir.count("= pto.alloc_tile") == 2, mlir


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("operation", ["add", "exp", "col_max"])
def test_shared_destinations_assemble_with_ptoas(operation, tmp_path: Path):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    # Arithmetic and col_max require row-major operands/results on A2/A3. The
    # shared root is column-major, so each output still needs a real typed view.
    mlir = _generate(_elementwise_program(operation, (32, 1), (1, 32)))
    source, output = tmp_path / "shared_destination.pto", tmp_path / "shared_destination.cpp"
    source.write_text(mlir)
    _run_ptoas(str(source), str(output), ["--pto-arch=a2", "--pto-level=level2"])
    assert output.is_file()


def test_each_tuple_output_uses_its_own_shared_storage_view():
    """PrepareTupleOutputs must resolve both destinations, not only the first."""
    span = ir.Span.unknown()
    zero = ir.ConstInt(0, pl.INDEX, span)

    def tile(name: str, shape: tuple[int, int], dtype) -> ir.Var:
        storage = ir.MemRef(
            ir.Var(f"{name}_base", ir.PtrType(), span), zero, shape[0] * shape[1] * dtype.get_byte(), span
        )
        return ir.Var(name, ir.TileType(shape, dtype, storage, None, ir.MemorySpace.Vec), span)

    indices_root = tile("indices_root", (8, 32), pl.INT32)
    counts_root = tile("counts_root", (32, 1), pl.INT32)
    source = tile("source", (32, 64), pl.FP32)
    workspace = tile("workspace", (32, 64), pl.UINT8)
    assert isinstance(indices_root.type, ir.TileType)
    assert isinstance(counts_root.type, ir.TileType)
    indices_type = ir.TileType(
        (32, 8),
        pl.INT32,
        indices_root.type.memref,
        ir.TileView(blayout=ir.TileLayout.row_major),
        ir.MemorySpace.Vec,
    )
    counts_type = ir.TileType(
        (1, 32),
        pl.INT32,
        counts_root.type.memref,
        ir.TileView(blayout=ir.TileLayout.row_major),
        ir.MemorySpace.Vec,
    )
    indices, counts = ir.Var("indices", indices_type, span), ir.Var("counts", counts_type, span)
    pair_type = ir.TupleType([indices_type, counts_type])
    pair = ir.Var("pair", pair_type, span)
    indices_out = ir.Var("indices_out", ir.TensorType((32, 8), pl.INT32), span)
    counts_out = ir.Var("counts_out", ir.TensorType((1, 32), pl.INT32), span)
    indices_stored = ir.Var("indices_stored", indices_out.type, span)
    counts_stored = ir.Var("counts_stored", counts_out.type, span)
    offsets = ir.MakeTuple([zero, zero], span)
    statements: list[ir.Stmt] = [
        ir.AssignStmt(value, ir.Call(ir.Op("tile.create"), [], value.type, span), span)
        for value in (indices_root, counts_root, source, workspace)
    ]
    statements.extend(
        [
            ir.AssignStmt(
                pair,
                ir.Call(
                    ir.Op("tile.gather_compare"),
                    [source, ir.ConstFloat(1.0, pl.FP32, span), workspace],
                    {"cmp_mode": 0, "offset": 0},
                    pair_type,
                    span,
                ),
                span,
            ),
            ir.AssignStmt(indices, ir.TupleGetItemExpr(pair, 0, span), span),
            ir.AssignStmt(counts, ir.TupleGetItemExpr(pair, 1, span), span),
            ir.AssignStmt(
                indices_stored,
                ir.Call(ir.Op("tile.store"), [indices, offsets, indices_out], indices_out.type, span),
                span,
            ),
            ir.AssignStmt(
                counts_stored,
                ir.Call(ir.Op("tile.store"), [counts, offsets, counts_out], counts_out.type, span),
                span,
            ),
            ir.ReturnStmt([indices_stored, counts_stored], span),
        ]
    )
    function = ir.Function(
        "kernel",
        [(indices_out, ir.ParamDirection.Out), (counts_out, ir.ParamDirection.Out)],
        [indices_out.type, counts_out.type],
        ir.SeqStmts(statements, span),
        span,
        ir.FunctionType.InCore,
    )
    mlir = _generate(ir.Program([function], "SharedStorageTupleOutputs", span))
    producer = _line(mlir, "pto.tgather ins(")
    destinations, types = _operands(producer, "outs")
    definitions = _definitions(mlir)
    assert len(destinations) == len(types) == 2, producer
    for destination, annotation in zip(destinations, types):
        assert definitions[destination] == annotation, producer
    assert "rows=32, cols=8," in types[0], producer
    assert "rows=1, cols=32," in types[1], producer
    assert f"{destinations[0]} = pto.treshape %indices_root " in mlir, mlir
    assert f"{destinations[1]} = pto.treshape %counts_root " in mlir, mlir
    stores = [line for line in mlir.splitlines() if "pto.tstore" in line]
    assert [_operands(store, "ins")[0][0] for store in stores] == destinations, mlir
    assert mlir.count("= pto.alloc_tile") == 4, mlir


@pytest.mark.parametrize(
    "dynamic_valid,body_local",
    [(False, False), (True, False), (True, True)],
    ids=["static-valid", "dynamic-valid", "dynamic-body-local"],
)
def test_tile_write_result_keeps_the_actual_input_binding(dynamic_valid, body_local):
    """A scalar update returns its input phi, even when its MemRef names an old root."""
    span = ir.Span.unknown()
    zero = ir.ConstInt(0, pl.INDEX, span)
    storage = ir.MemRef(ir.Var("root_base", ir.PtrType(), span), zero, 128, span)
    valid_rows = ir.Var("valid_rows", ir.ScalarType(pl.INDEX), span)
    valid_input = ir.Var("valid_input", ir.ScalarType(pl.INDEX), span)
    root_type = ir.TileType((1, 32), pl.FP32, storage, None, ir.MemorySpace.Vec)
    column_type = ir.TileType(
        (32, 1),
        pl.FP32,
        storage,
        ir.TileView(valid_shape=[valid_rows, 1] if dynamic_valid else None, blayout=ir.TileLayout.col_major),
        ir.MemorySpace.Vec,
    )
    root = ir.Var("root", root_type, span)
    merged = ir.Var("merged", column_type, span)
    updated = ir.Var("updated", column_type, span)
    flag = ir.Var("flag", ir.ScalarType(pl.BOOL), span)
    output = ir.Var("output", ir.TensorType((32, 1), pl.FP32), span)
    stored = ir.Var("stored", output.type, span)
    offsets = ir.MakeTuple([zero, zero], span)
    valid_statements = (
        [
            ir.AssignStmt(
                valid_rows, ir.Min(valid_input, ir.ConstInt(32, pl.INDEX, span), pl.INDEX, span), span
            )
        ]
        if body_local
        else []
    )

    def branch(name: str) -> ir.SeqStmts:
        tile = ir.Var(name, column_type, span)
        return ir.SeqStmts(
            [
                ir.AssignStmt(tile, ir.Call(ir.Op("tile.create"), [], column_type, span), span),
                ir.YieldStmt([tile], span),
            ],
            span,
        )

    body = ir.SeqStmts(
        [
            ir.AssignStmt(root, ir.Call(ir.Op("tile.create"), [], root_type, span), span),
            *valid_statements,
            ir.IfStmt(flag, branch("left"), branch("right"), [merged], span),
            ir.AssignStmt(
                updated,
                ir.Call(
                    ir.Op("tile.write"),
                    [merged, offsets, ir.ConstFloat(7.0, pl.FP32, span)],
                    column_type,
                    span,
                ),
                span,
            ),
            ir.AssignStmt(
                stored, ir.Call(ir.Op("tile.store"), [updated, offsets, output], output.type, span), span
            ),
            ir.ReturnStmt([stored], span),
        ],
        span,
    )
    params = [(flag, ir.ParamDirection.In), (output, ir.ParamDirection.Out)]
    if dynamic_valid:
        params.append((valid_input if body_local else valid_rows, ir.ParamDirection.In))
    function = ir.Function("kernel", params, [output.type], body, span, ir.FunctionType.InCore)
    mlir = _generate(ir.Program([function], "SharedStorageTileWrite", span))
    write = _line(mlir, "pto.tsetval ins(")
    store = _line(mlir, "pto.tstore")
    destination, annotations = _operands(write, "outs")
    assert _operands(store, "ins") == (destination, annotations), mlir
    assert destination != ["%root"], mlir
    view = _line(mlir, f"{destination[0]} = pto.treshape %root ")
    assert view.endswith(f"-> {annotations[0]}"), mlir
    assert _definitions(mlir)[destination[0]] == annotations[0], mlir
    assert "rows=32, cols=1," in annotations[0], write
    assert mlir.index(view) < mlir.index("scf.if") < mlir.index(write), mlir
    if dynamic_valid:
        initialized = _line(mlir, f"pto.set_validshape {destination[0]},")
        if body_local:
            definition = _line(mlir, "arith.minsi")
            valid_ssa = re.search(r"(%[A-Za-z0-9_]+) = arith.minsi", definition)
            assert mlir.index(definition) < mlir.index(view), mlir
        else:
            valid_ssa = re.search(r"(%[A-Za-z0-9_]+): index", _line(mlir, "func.func @kernel"))
        assert valid_ssa is not None, mlir
        assert f"{valid_ssa[1]}, %c1_index : {annotations[0]}" in initialized, mlir
        assert mlir.index(view) < mlir.index(initialized) < mlir.index("scf.if"), mlir
    else:
        assert "v_row=32, v_col=1," in annotations[0], mlir
    assert "pto.set_validshape %root," not in mlir, mlir
    assert "pto.tmov" not in mlir, mlir
    assert mlir.count("= pto.alloc_tile") == 1, mlir


def test_branch_move_copies_to_its_independent_typed_phi():
    """An independent result keeps the real moves and their own operand types."""
    span = ir.Span.unknown()
    zero = ir.ConstInt(0, pl.INDEX, span)
    storage = ir.MemRef(ir.Var("root_base", ir.PtrType(), span), zero, 128, span)
    root_type = ir.TileType((1, 32), pl.FP32, storage, None, ir.MemorySpace.Vec)
    column_type = ir.TileType(
        (32, 1), pl.FP32, storage, ir.TileView(blayout=ir.TileLayout.col_major), ir.MemorySpace.Vec
    )
    destination_type = ir.TileType(
        (32, 1),
        pl.FP32,
        ir.MemRef(ir.Var("destination_base", ir.PtrType(), span), zero, 128, span),
        ir.TileView(blayout=ir.TileLayout.col_major),
        ir.MemorySpace.Vec,
    )
    root = ir.Var("root", root_type, span)
    source = ir.Var("source", column_type, span)
    merged = ir.Var("merged", destination_type, span)
    flag = ir.Var("flag", ir.ScalarType(pl.BOOL), span)
    output = ir.Var("output", ir.TensorType((32, 1), pl.FP32), span)
    stored = ir.Var("stored", output.type, span)

    def branch(name: str, line: int) -> ir.SeqStmts:
        move_span = ir.Span("shared_storage_branch.py", line, 1, line, 20)
        moved = ir.Var(name, destination_type, move_span)
        return ir.SeqStmts(
            [
                ir.AssignStmt(
                    moved, ir.Call(ir.Op("tile.move"), [source], destination_type, move_span), move_span
                ),
                ir.YieldStmt([moved], span),
            ],
            span,
        )

    body = ir.SeqStmts(
        [
            ir.AssignStmt(root, ir.Call(ir.Op("tile.create"), [], root_type, span), span),
            ir.AssignStmt(source, ir.Call(ir.Op("tile.create"), [], column_type, span), span),
            ir.IfStmt(flag, branch("left", 10), branch("right", 20), [merged], span),
            ir.AssignStmt(
                stored,
                ir.Call(
                    ir.Op("tile.store"),
                    [merged, ir.MakeTuple([zero, zero], span), output],
                    output.type,
                    span,
                ),
                span,
            ),
            ir.ReturnStmt([stored], span),
        ],
        span,
    )
    function = ir.Function(
        "kernel",
        [(flag, ir.ParamDirection.In), (output, ir.ParamDirection.Out)],
        [output.type],
        body,
        span,
        ir.FunctionType.InCore,
    )
    mlir = _generate(ir.Program([function], "SharedStorageBranchMove", span))
    definitions = _definitions(mlir)
    phi = _operands(_line(mlir, "pto.tstore"), "ins")[0][0]
    assert phi != "%root", mlir
    phi_alloc = _line(mlir, f"{phi} = pto.alloc_tile")
    assert mlir.index(phi_alloc) < mlir.index("scf.if"), mlir
    assert "rows=32, cols=1," in definitions[phi], mlir
    branches = mlir.split("scf.if", maxsplit=1)[1].split("} else {", maxsplit=1)
    assert len(branches) == 2, mlir
    for branch_mlir, line in zip(branches, (10, 20)):
        move = _line(branch_mlir, "pto.tmov ins(")
        _assert_operand_declarations(move, definitions)
        assert _operands(move, "outs")[0] == [phi], move
        assert _operands(move, "ins")[0] != [phi], move
        # The actual move must survive destination rebinding. A synthesized
        # end-of-branch copy must not mask its elimination as an IR-only no-op.
        assert f'loc("shared_storage_branch.py":{line}:1)' in move, move
    assert mlir.count("= pto.alloc_tile") == 2, mlir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
