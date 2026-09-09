# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Direct native emission from explicit Buffer IR, without Tile/MemRef reconstruction."""

import re
from typing import Any

import pytest
from pypto import DataType, ir, passes
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.pypto_core import codegen

pytestmark = pytest.mark.usefixtures("ascend_backend")
SPAN = ir.Span.unknown()


def _int(value, dtype=DataType.INDEX):
    return ir.ConstInt(value, dtype, SPAN)


def _scalar(name, dtype=DataType.INDEX):
    return ir.Var(name, ir.ScalarType(dtype), SPAN)


def _type(shape=(16, 32), valid=(), **kwargs):
    return ir.BufferType(list(shape), DataType.FP32, ir.MemorySpace.Vec, list(valid), **kwargs)


def _call(name, args, result=None, kwargs=None):
    return ir.Call(ir.Op(name), args, kwargs or {}, None, result or ir.VoidType(), SPAN)


def _alloc(name, descriptor=None, valid=(), address=None):
    descriptor = descriptor or _type()
    var = ir.Var(name, descriptor, SPAN)
    args = [ir.MakeTuple(list(valid), SPAN)]
    if address is not None:
        args.append(address)
    return var, ir.AssignStmt(var, _call("buffer.alloc", args, descriptor), SPAN)


def _eval(name, *args):
    return ir.EvalStmt(_call(name, list(args)), SPAN)


def _program(stmts, params=()):
    function = ir.Function(
        "kernel", list(params), [], ir.SeqStmts(stmts, SPAN), SPAN, type=ir.FunctionType.InCore
    )
    return ir.Program([function], "Buffers", SPAN)


def _emit(program, flag=True):
    return codegen.PTOCodegen().generate(program, emit_tile_addr=flag, emit_source_loc=False)


def _allocations(text):
    return re.findall(r"(%\w+) = pto\.alloc_tile([^\n]*)", text)


def _compile_native(tmp_path, text, addressed):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    source_path = tmp_path / "buffers.pto"
    output_path = tmp_path / "buffers.cpp"
    source_path.write_text(text)
    _run_ptoas(
        str(source_path),
        str(output_path),
        [f"--pto-level={'level3' if addressed else 'level2'}", "--pto-arch=a2"],
    )
    assert output_path.is_file()


def _window(*values):
    return ir.MakeTuple([_int(value) if isinstance(value, int) else value for value in values], SPAN)


def _gm_program(addressed=False, source_name="input"):
    tensor_type = ir.TensorType([32, 64], DataType.FP32)
    source = ir.Var(source_name, tensor_type, SPAN)
    other = ir.Var("other", tensor_type, SPAN)
    output = ir.Var("output", tensor_type, SPAN)
    condition = _scalar("condition", DataType.BOOL)
    lhs, lhs_alloc = _alloc("lhs", address=_int(0) if addressed else None)
    rhs, rhs_alloc = _alloc("rhs", address=_int(2048) if addressed else None)
    dst, dst_alloc = _alloc("dst", address=_int(4096) if addressed else None)
    branch = ir.IfStmt(
        condition,
        _eval("buffer.add", dst, rhs, dst),
        _eval("buffer.copy", lhs, dst),
        [],
        SPAN,
    )
    body = ir.SeqStmts(
        [
            lhs_alloc,
            rhs_alloc,
            dst_alloc,
            _eval("buffer.load", source, _window(8, 16), _window(16, 32), lhs),
            _eval("buffer.load", other, _window(0, 0), _window(16, 32), rhs),
            _eval("buffer.mul", lhs, rhs, dst),
            branch,
            _eval("buffer.store", dst, _window(4, 8), _window(16, 32), output),
            ir.ReturnStmt([output], SPAN),
        ],
        SPAN,
    )
    # Deliberately interleave scalar and Tensor params. Native PTOParam places
    # input, output, other pointers first and the condition last.
    kernel = ir.Function(
        "kernel",
        [condition, source, (output, ir.ParamDirection.Out), other],
        [tensor_type],
        body,
        SPAN,
        type=ir.FunctionType.AIV,
    )
    return ir.Program([kernel], "BufferGM", SPAN)


@pytest.mark.parametrize("addressed", [False, True])
def test_native_gm_program_round_trip_preserves_abi_and_explicit_destinations(tmp_path, addressed):
    program = _gm_program(addressed)
    restored = ir.deserialize(ir.serialize(program))
    assert isinstance(restored, ir.Program)
    ir.assert_structural_equal(program, restored, enable_auto_mapping=True)
    original_kernel = next(iter(program.functions.values()))
    kernel = next(iter(restored.functions.values()))
    assert kernel.param_directions == original_kernel.param_directions
    assert len(kernel.return_types) == 1
    text = _emit(restored, flag=not addressed)
    signature = next(line for line in text.splitlines() if "func.func @kernel" in line)
    assert "(%arg0: !pto.ptr<f32>, %arg1: !pto.ptr<f32>, %arg2: !pto.ptr<f32>, %arg3: i1)" in signature
    assert "->" not in signature
    assert "scf.if %arg3 {" in text
    assert "scf.yield" not in text
    allocations = _allocations(text)
    assert len(allocations) == 3
    lhs, rhs, dst = (allocation[0] for allocation in allocations)
    assert text.count(" = pto.make_tensor_view ") == 3
    assert text.count(" = pto.partition_view ") == 3
    assert text.count("pto.tload ins(") == 2
    assert text.count("pto.tstore ins(") == 1
    assert f"pto.tmul ins({lhs}, {rhs} : " in text
    assert f"pto.tadd ins({dst}, {rhs} : " in text
    assert f"pto.tmov ins({lhs} : " in text
    assert f"pto.tstore ins({dst} : " in text
    assert text.count(f") outs({dst} : ") == 3
    assert not re.search(r"= pto\.t(load|store|mul|add|mov)", text)
    # ABI emission must not mutate the caller's IR signature or output aliases.
    ir.assert_structural_equal(program, restored, enable_auto_mapping=True)
    _compile_native(tmp_path, text, addressed)


def test_native_gm_parameter_names_do_not_trigger_legacy_pipe_handling(tmp_path):
    text = _emit(_gm_program(source_name="__gm_pipe_buffer"))
    assert re.search(r"%\w+ = pto.make_tensor_view %arg0,", text)
    _compile_native(tmp_path, text, addressed=False)


def test_native_gm_runtime_windows_and_valid_casts_remain_inside_loop(tmp_path):
    tensor_type = ir.TensorType([256, 64], DataType.FP32)
    source = ir.Var("source", tensor_type, SPAN)
    output = ir.Var("output", tensor_type, SPAN)
    rows = _scalar("rows", DataType.UINT8)
    index = _scalar("i")
    column = _scalar("column")
    buffer, allocation = _alloc("buffer", _type(shape=(256, 32), valid=(-1, -1)), [rows, _int(32)])
    loop = ir.ForStmt(
        index,
        _int(0),
        _int(2),
        _int(1),
        [],
        ir.SeqStmts(
            [
                ir.AssignStmt(column, ir.Mul(index, _int(16), DataType.INDEX, SPAN), SPAN),
                allocation,
                _eval("buffer.set_validshape", buffer, _window(rows, 32)),
                _eval("buffer.load", source, _window(0, column), _window(rows, 32), buffer),
                _eval("buffer.store", buffer, _window(0, column), _window(rows, 32), output),
            ],
            SPAN,
        ),
        [],
        SPAN,
    )
    kernel = ir.Function(
        "kernel",
        [rows, source, (output, ir.ParamDirection.Out)],
        [tensor_type],
        ir.SeqStmts([loop, ir.ReturnStmt([output], SPAN)], SPAN),
        SPAN,
        type=ir.FunctionType.AIV,
    )
    text = _emit(ir.Program([kernel], "RuntimeGMWindows", SPAN))
    loop_position = text.index("scf.for")
    assert text.index("pto.alloc_tile") > loop_position
    partitions = re.findall(r"pto.partition_view [^\n]+", text)
    assert len(partitions) == 2
    assert text.index(partitions[0]) > loop_position
    for partition in partitions:
        extent = re.search(r"sizes = \[(%\w+),", partition)
        assert extent is not None
        cast = re.search(rf"{re.escape(extent[1])} = arith.index_cast (%\w+) : i64 to index", text)
        assert cast is not None
        widen = re.search(rf"{re.escape(cast[1])} = arith.extui (%\w+) : i8 to i64", text)
        assert widen is not None
        assert f"{widen[1]} = builtin.unrealized_conversion_cast %arg2 : ui8 to i8" in text
        assert text.index(cast[0]) > loop_position
    assert len(_allocations(text)) == 1
    # This checks native syntax and unsigned dataflow, not numerical execution.
    _compile_native(tmp_path, text, addressed=False)


@pytest.mark.parametrize(
    "name,direction", [("buffer.load", ir.ParamDirection.Out), ("buffer.store", ir.ParamDirection.In)]
)
def test_gm_transfers_reject_incompatible_parameter_directions(name, direction):
    tensor = ir.Var("tensor", ir.TensorType([16, 32], DataType.FP32), SPAN)
    buffer, allocation = _alloc("buffer")
    source, destination = (tensor, buffer) if name == "buffer.load" else (buffer, tensor)
    program = _program(
        [allocation, _eval(name, source, _window(0, 0), _window(16, 32), destination)],
        [(tensor, direction)],
    )
    with pytest.raises(ValueError, match="parameter direction"):
        _emit(program)


@pytest.mark.parametrize(
    "tensor_type,message",
    [
        (ir.TensorType([16, 32], DataType.FP16), "rank-2 FP32"),
        (ir.TensorType([16, 1], DataType.FP32), "columns > 1"),
        (
            ir.TensorType([16, 32], DataType.FP32, None, ir.TensorView([1, 16], ir.TensorLayout.DN)),
            "ND layout",
        ),
        (
            ir.TensorType([16, 32], DataType.FP32, None, ir.TensorView([64, 1], ir.TensorLayout.ND)),
            "packed row-major",
        ),
        (ir.TensorType([_scalar("rows"), _int(32)], DataType.FP32), "static physical shapes"),
    ],
)
def test_unsupported_gm_parameter_recipes_fail_before_emission(tensor_type, message):
    tensor = ir.Var("tensor", tensor_type, SPAN)
    _, allocation = _alloc("buffer")
    with pytest.raises(ValueError, match=message):
        _emit(_program([allocation], [tensor]))


def test_gm_tensor_carries_cannot_enter_legacy_control_flow_emission():
    tensor = ir.Var("tensor", ir.TensorType([16, 32], DataType.FP32), SPAN)
    carry = ir.IterArg("carry", tensor.type, tensor, SPAN)
    result = ir.Var("result", tensor.type, SPAN)
    _, allocation = _alloc("buffer")
    loop = ir.ForStmt(
        _scalar("i"), _int(0), _int(2), _int(1), [carry], ir.YieldStmt([carry], SPAN), [result], SPAN
    )
    with pytest.raises(ValueError, match="only scalar region results and carries"):
        _emit(_program([allocation, loop], [tensor]))


def test_gm_return_must_be_a_normalized_parameter_alias():
    tensor = ir.Var("tensor", ir.TensorType([16, 32], DataType.FP32), SPAN)
    _, allocation = _alloc("buffer")
    function = ir.Function(
        "kernel", [tensor], [tensor.type], ir.SeqStmts([allocation], SPAN), SPAN, type=ir.FunctionType.InCore
    )
    with pytest.raises(ValueError, match="final normalized GM tensor return"):
        _emit(ir.Program([function], "MissingReturn", SPAN))


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("addressed", [False, True])
def test_allocation_address_is_determined_only_by_the_ir(flag, addressed):
    address = _int(0, DataType.INT64) if addressed else None
    _, allocation = _alloc("storage", address=address)
    text = _emit(_program([allocation]), flag)
    allocations = _allocations(text)
    assert len(allocations) == 1
    assert (" addr = " in allocations[0][1]) == addressed
    assert "rows=16, cols=32" in allocations[0][1]
    assert "v_row=16, v_col=32" in allocations[0][1]
    assert "valid_row = " not in allocations[0][1]
    assert "valid_col = " not in allocations[0][1]
    if addressed:
        address_ssa = re.search(r"addr = (%\w+)", allocations[0][1])
        assert address_ssa is not None
        assert f"{address_ssa[1]} = arith.constant 0 : i64" in text


def test_equal_addresses_do_not_merge_distinct_buffer_ssa_definitions():
    first, first_alloc = _alloc("first", address=_int(0))
    second, second_alloc = _alloc("second", address=_int(0))
    text = _emit(_program([first_alloc, second_alloc, _eval("buffer.copy", first, second)]))
    allocations = _allocations(text)
    assert len(allocations) == 2
    assert allocations[0][0] != allocations[1][0]
    assert f"pto.tmov ins({allocations[0][0]} : " in text
    assert f") outs({allocations[1][0]} : " in text


def test_mul_and_copy_use_explicit_destinations_without_result_ssa_or_hidden_allocations():
    source, source_alloc = _alloc("source")
    destination, destination_alloc = _alloc("destination")
    text = _emit(
        _program(
            [
                source_alloc,
                destination_alloc,
                _eval("buffer.mul", source, source, destination),
                _eval("buffer.copy", destination, source),
            ]
        )
    )
    allocations = _allocations(text)
    assert len(allocations) == 2
    source_ssa, destination_ssa = (a[0] for a in allocations)
    assert f"pto.tmul ins({source_ssa}, {source_ssa} : " in text
    assert f") outs({destination_ssa} : " in text
    assert f"pto.tmov ins({destination_ssa} : " in text
    assert f") outs({source_ssa} : " in text
    assert not re.search(r"= pto\.t(mul|mov)", text)


def _dynamic_loop_program(addressed, metadata=True):
    count = _scalar("count")
    index = _scalar("i")
    extent = _scalar("extent")
    address = _scalar("address", DataType.INT32)
    extent_assign = ir.AssignStmt(extent, ir.Add(index, _int(1), DataType.INDEX, SPAN), SPAN)
    address_assign = ir.AssignStmt(
        address, ir.Add(_int(2048, DataType.INT32), _int(4096, DataType.INT32), DataType.INT32, SPAN), SPAN
    )
    descriptor = _type(valid=(-1, -1) if metadata else (-1, 32))
    runtime_valid = [extent, _int(32)] if metadata else [extent]
    source, source_alloc = _alloc("source", descriptor, runtime_valid, address if addressed else None)
    destination, destination_alloc = _alloc(
        "destination", descriptor, runtime_valid, _int(8192) if addressed else None
    )
    stmts: list[ir.Stmt] = [extent_assign, address_assign, source_alloc, destination_alloc]
    stmts.append(_eval("buffer.mul", source, source, destination))
    if metadata:
        stmts.append(_eval("buffer.set_validshape", destination, ir.MakeTuple([extent, _int(32)], SPAN)))
    stmts.append(_eval("buffer.copy", destination, source))
    body = ir.SeqStmts(stmts, SPAN)
    loop = ir.ForStmt(index, _int(0), count, _int(1), [], body, [], SPAN)
    return _program([loop], [count])


def test_dynamic_allocation_and_metadata_operands_stay_in_the_loop_scope():
    text = _emit(_dynamic_loop_program(True), False)
    lines = text.splitlines()
    loop_line = next(i for i, line in enumerate(lines) if "scf.for" in line)
    allocation_lines = [i for i, line in enumerate(lines) if "pto.alloc_tile" in line]
    assert len(allocation_lines) == 2
    assert all(i > loop_line for i in allocation_lines)
    first = lines[allocation_lines[0]]
    valid_ssa = re.search(r"valid_row = (%\w+)", first)
    address_ssa = re.search(r"addr = (%\w+)", first)
    assert valid_ssa is not None and address_ssa is not None
    for operand in [valid_ssa[1], address_ssa[1]]:
        definition = next(i for i, line in enumerate(lines) if f"{operand} = " in line)
        assert loop_line < definition < allocation_lines[0]
    assert "arith.extsi" in text
    assert "v_row=?, v_col=?" in first
    assert text.count("pto.set_validshape") == 1


def test_mixed_valid_descriptor_emits_only_the_dynamic_axis_operand(tmp_path):
    text = _emit(_dynamic_loop_program(False, metadata=False))
    for _, allocation in _allocations(text):
        assert "v_row=?, v_col=32" in allocation
        assert "valid_row = " in allocation
        assert "valid_col = " not in allocation
    _compile_native(tmp_path, text, addressed=False)


def test_scalar_if_results_remain_ssa_while_buffer_writes_use_the_existing_handles():
    condition = _scalar("condition", DataType.BOOL)
    selected = _scalar("selected")
    source, source_alloc = _alloc("source")
    destination, destination_alloc = _alloc("destination")
    branch = ir.IfStmt(
        condition,
        ir.SeqStmts([_eval("buffer.mul", source, source, destination), ir.YieldStmt([_int(1)], SPAN)], SPAN),
        ir.SeqStmts([_eval("buffer.copy", source, destination), ir.YieldStmt([_int(2)], SPAN)], SPAN),
        [selected],
        SPAN,
    )
    text = _emit(_program([source_alloc, destination_alloc, branch], [condition]))
    assert re.search(r"%\w+ = scf.if %arg0 -> \(index\)", text)
    assert text.count("scf.yield") == 2
    assert len(_allocations(text)) == 2
    assert len(re.findall(r"outs\(%destination\s*:", text)) == 2


def test_rank_one_descriptor_maps_to_a_single_native_row():
    _, allocation = _alloc("vector", _type(shape=(32,), valid=(17,)))
    text = _emit(_program([allocation]))
    assert "rows=1, cols=32, v_row=1, v_col=17" in text
    assert "valid_row = " not in text
    assert "valid_col = " not in text


def test_codegen_validates_original_buffer_calls_before_emission():
    destination, allocation = _alloc("destination")
    malformed = ir.EvalStmt(_call("buffer.copy", [destination, destination], kwargs={"unexpected": 1}), SPAN)
    with pytest.raises(ValueError, match="Invalid Buffer IR.*Invalid buffer call"):
        _emit(_program([allocation, malformed]))


def test_codegen_rejects_undefined_buffer_operands():
    undefined = ir.Var("undefined", _type(), SPAN)
    destination, allocation = _alloc("destination")
    with pytest.raises(ValueError, match="Invalid Buffer IR"):
        _emit(_program([allocation, _eval("buffer.copy", undefined, destination)]))


@pytest.mark.parametrize("owner", ["ForStmt", "RuntimeScopeStmt", "ClusterScopeStmt", "GraphScopeStmt"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_buffer_referenced_only_in_region_attributes_cannot_bypass_validation(owner, wrapped):
    hidden = ir.Var("hidden", _type(), SPAN)
    reference = ir.Add(hidden, _int(1), DataType.INDEX, SPAN) if wrapped else hidden
    carrier = ir.ForStmt(
        _scalar("i"),
        _int(0),
        _int(2),
        _int(1),
        [],
        ir.EvalStmt(_int(0), SPAN),
        [],
        SPAN,
        attrs={"hidden_buffer": reference},
    )
    if owner == "ForStmt":
        region = carrier
    else:
        # Scope attrs are compiler metadata with no public mutation API. Their
        # readers consume the shared span/body/attrs fields and ignore the
        # extra loop fields, so change just the serialized node-kind string.
        payload = bytes(ir.serialize(carrier))
        marker = b"\xa4type\xa7ForStmt"
        kind = owner.encode()
        assert len(kind) < 32 and payload.count(marker) == 1
        region = ir.deserialize(payload.replace(marker, b"\xa4type" + bytes([0xA0 + len(kind)]) + kind))
        assert isinstance(region, ir.ScopeStmt)
    # A scalar wrapper may report its illegal operand before the attribute
    # diagnostic. Both forms must reach BufferIR instead of legacy emission.
    with pytest.raises(ValueError, match="Invalid Buffer IR.*cannot carry buffer handles"):
        _emit(_program([region]))


def test_buffer_parameter_requires_a_supported_device_abi():
    incoming = ir.Var("incoming", _type(), SPAN)
    with pytest.raises(ValueError, match="buffer-parameter ABI"):
        _emit(_program([_eval("buffer.copy", incoming, incoming)], [incoming]))


@pytest.mark.parametrize("owner", ["call", "submit"])
def test_buffer_referenced_only_in_kwargs_cannot_bypass_validation(owner):
    hidden = ir.Var("hidden", _type(), SPAN)
    # Deliberately exceed the public keyword schema to check malformed IR.
    kwargs: dict[str, Any] = {"hidden_buffer": hidden}
    if owner == "call":
        expression = _call("tile.get_block_idx", [], ir.ScalarType(DataType.INDEX), kwargs)
    else:
        expression = ir.Submit(
            ir.GlobalVar("callee"),
            [],
            [],
            kwargs,
            None,
            ir.TupleType([ir.ScalarType(DataType.TASK_ID)]),
            SPAN,
        )
    with pytest.raises(ValueError, match="Invalid Buffer IR"):
        _emit(_program([ir.EvalStmt(expression, SPAN)]))


@pytest.mark.parametrize("field", ["shape", "valid_shape", "stride"])
def test_buffer_referenced_only_in_type_metadata_cannot_bypass_validation(field):
    hidden = ir.Var("hidden", _type(), SPAN)
    view = ir.TensorView(
        stride=[hidden] if field == "stride" else [],
        layout=ir.TensorLayout.ND,
        valid_shape=[hidden] if field == "valid_shape" else [],
    )
    tensor = ir.Var(
        "tensor", ir.TensorType([hidden] if field == "shape" else [_int(16)], DataType.FP32, None, view), SPAN
    )
    with pytest.raises(ValueError, match="Invalid Buffer IR.*Type metadata"):
        _emit(_program([], [tensor]))


@pytest.mark.parametrize("kind", ["add", "compare", "neg", "cast_width", "cast_index"])
def test_unsupported_unsigned_scalar_recipes_fail_before_emission(kind):
    operand = _scalar("operand", DataType.UINT32)
    if kind == "add":
        expression = ir.Add(operand, _int(1, DataType.UINT32), DataType.UINT32, SPAN)
    elif kind == "compare":
        expression = ir.Lt(operand, _int(1, DataType.UINT32), DataType.BOOL, SPAN)
    elif kind == "neg":
        expression = ir.Neg(operand, DataType.UINT32, SPAN)
    else:
        expression = ir.Cast(operand, DataType.UINT64 if kind == "cast_width" else DataType.INDEX, SPAN)
    _, allocation = _alloc("storage")
    with pytest.raises(ValueError, match="Unsigned scalar (arithmetic|cast) is not supported"):
        _emit(_program([allocation, ir.EvalStmt(expression, SPAN)], [operand]))


@pytest.mark.parametrize("field", ["base", "offset"])
def test_buffer_referenced_only_in_gm_memref_cannot_bypass_validation(field):
    hidden = ir.Var("hidden", _type(), SPAN)
    base = hidden if field == "base" else ir.Var("gm", ir.PtrType(), SPAN)
    memref = ir.MemRef(base, hidden if field == "offset" else _int(0), 2048, SPAN)
    tensor = ir.Var("tensor", ir.TensorType([_int(16)], DataType.FP32, memref), SPAN)
    with pytest.raises(ValueError, match="Invalid Buffer IR.*Type metadata"):
        _emit(_program([], [tensor]))


@pytest.mark.parametrize("dtype", [DataType.INT32, DataType.INT64])
def test_non_index_for_induction_fails_before_emission(dtype):
    index = _scalar("i", dtype)
    _, allocation = _alloc("storage", _type(valid=(-1, 32)), [index])
    loop = ir.ForStmt(index, _int(0), _int(16), _int(1), [], allocation, [], SPAN)
    with pytest.raises(ValueError, match="requires an INDEX for-loop induction variable"):
        _emit(_program([loop]))


def test_unsigned_for_bound_fails_before_emission():
    stop = _scalar("stop", DataType.UINT8)
    _, allocation = _alloc("storage")
    loop = ir.ForStmt(_scalar("i"), _int(0), stop, _int(1), [], allocation, [], SPAN)
    with pytest.raises(ValueError, match="requires INDEX or signed integer for-loop bounds"):
        _emit(_program([loop], [stop]))


@pytest.mark.parametrize(
    ("descriptor", "message"),
    [
        (_type(shape=(2, 16, 32)), "rank-1 or rank-2"),
        (_type(blayout=ir.TileLayout.col_major), "dense row-major Vec"),
        (ir.BufferType([16, 32], DataType.FP4, ir.MemorySpace.Vec), "dense row-major Vec"),
    ],
)
def test_unsupported_physical_descriptors_fail_explicitly(descriptor, message):
    _, allocation = _alloc("destination", descriptor)
    with pytest.raises(ValueError, match=message):
        _emit(_program([allocation]))


@pytest.mark.parametrize("addressed", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_native_ptoas_accepts_direct_buffer_emission(tmp_path, addressed, dynamic):
    if dynamic:
        program = _dynamic_loop_program(addressed)
    else:
        source, source_alloc = _alloc("source", address=_int(0) if addressed else None)
        destination, destination_alloc = _alloc("destination", address=_int(2048) if addressed else None)
        program = _program(
            [source_alloc, destination_alloc, _eval("buffer.mul", source, source, destination)]
        )
    _compile_native(tmp_path, _emit(program, not addressed), addressed)


@pytest.mark.parametrize("addressed", [False, True])
def test_complete_buffer_program_round_trip_verifies_and_compiles(tmp_path, addressed):
    program = _dynamic_loop_program(addressed)
    restored = ir.deserialize(ir.serialize(program))
    assert isinstance(restored, ir.Program)
    ir.assert_structural_equal(program, restored, enable_auto_mapping=True)

    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.BufferIR)
    diagnostics = passes.PropertyVerifierRegistry.verify(properties, restored)
    assert diagnostics == [], passes.PropertyVerifierRegistry.generate_report(diagnostics)

    text = _emit(restored, not addressed)
    # Compare complete emission so serialization must preserve the loop's scalar
    # address/valid operands and their use sites, not just buffer descriptors.
    assert text == _emit(program, not addressed)
    allocations = _allocations(text)
    assert len(allocations) == 2
    source, destination = (allocation[0] for allocation in allocations)
    assert source != destination
    assert all((" addr = " in allocation[1]) == addressed for allocation in allocations)
    assert text.index("scf.for") < text.index("pto.alloc_tile")

    multiply = [line for line in text.splitlines() if "pto.tmul " in line]
    copy = [line for line in text.splitlines() if "pto.tmov " in line]
    metadata = [line for line in text.splitlines() if "pto.set_validshape " in line]
    assert len(multiply) == len(copy) == len(metadata) == 1
    assert f"ins({source}, {source} : " in multiply[0]
    assert f") outs({destination} : " in multiply[0]
    assert f"ins({destination} : " in copy[0]
    assert f") outs({source} : " in copy[0]
    assert f"pto.set_validshape {destination}, " in metadata[0]
    _compile_native(tmp_path, text, addressed)


@pytest.mark.parametrize("loop_kind", ["for", "while"])
def test_native_scalar_loop_carries_preserve_explicit_buffer_writes(tmp_path, loop_kind):
    count = _scalar("count")
    carried = ir.IterArg("carried", ir.ScalarType(DataType.INDEX), _int(0), SPAN)
    next_value = _scalar("next_value")
    result = _scalar("result")
    source, source_alloc = _alloc("source", _type(valid=(-1, -1)), [_int(16), _int(32)])
    destination, destination_alloc = _alloc("destination", _type(valid=(-1, -1)), [_int(16), _int(32)])
    body = ir.SeqStmts(
        [
            _eval("buffer.mul", source, source, destination),
            ir.AssignStmt(next_value, ir.Add(carried, _int(1), DataType.INDEX, SPAN), SPAN),
            _eval("buffer.copy", destination, source),
            ir.YieldStmt([next_value], SPAN),
        ],
        SPAN,
    )
    if loop_kind == "for":
        loop = ir.ForStmt(_scalar("i"), _int(0), count, _int(1), [carried], body, [result], SPAN)
    else:
        loop = ir.WhileStmt(ir.Lt(carried, count, DataType.BOOL, SPAN), [carried], body, [result], SPAN)
    # The result is consumed after the loop, so losing or misbinding its scalar
    # SSA result cannot hide behind a dead result. At runtime count is in [0,16].
    update = _eval("buffer.set_validshape", destination, ir.MakeTuple([result, _int(32)], SPAN))
    text = _emit(_program([source_alloc, destination_alloc, loop, update], [count]))
    allocations = _allocations(text)
    assert len(allocations) == 2
    header = next(line for line in text.splitlines() if f"= scf.{loop_kind} " in line)
    assert "-> (index)" in header
    assert "tile_buf" not in header
    result_ssa = re.search(r"(%\w+) = scf\.", header)
    assert result_ssa is not None
    assert f"pto.set_validshape {allocations[1][0]}, {result_ssa[1]}," in text
    assert text.index("pto.alloc_tile") < text.index(header)
    assert text.count("pto.tmul") == text.count("pto.tmov") == 1
    assert f"pto.tmul ins({allocations[0][0]}, {allocations[0][0]} : " in text
    assert f"pto.tmov ins({allocations[1][0]} : " in text
    _compile_native(tmp_path, text, addressed=False)


def test_native_unsigned_address_and_valid_operands_use_signless_integer_casts(tmp_path):
    rows = _scalar("rows")
    address = _scalar("address", DataType.UINT32)
    valid = _scalar("valid", DataType.UINT32)
    # Clamp before converting to UINT32: every runtime valid extent is in the
    # physical descriptor's bounds, and the emitter still sees a dynamic ui32.
    bounded = ir.Max(ir.Min(rows, _int(16), DataType.INDEX, SPAN), _int(0), DataType.INDEX, SPAN)
    valid_assign = ir.AssignStmt(valid, ir.Cast(bounded, DataType.UINT32, SPAN), SPAN)
    buffer, allocation = _alloc("storage", _type(valid=(-1, -1)), [valid, _int(32)], address)
    program = _program(
        [
            valid_assign,
            allocation,
            _eval("buffer.set_validshape", buffer, ir.MakeTuple([valid, _int(32)], SPAN)),
            _eval("buffer.mul", buffer, buffer, buffer),
        ],
        [rows, address],
    )
    text = _emit(program, False)
    assert "ui32 to i32" in text
    assert re.search(r"arith\.extui %\w+ : i32 to i64", text)
    assert re.search(r"arith\.index_cast %\w+ : i64 to index", text)
    assert not re.search(r"arith\.(extui|index_cast) %\w+ : ui32", text)
    _compile_native(tmp_path, text, addressed=True)


def test_native_uint8_valid_extents_preserve_the_high_bit(tmp_path):
    valid = _scalar("valid", DataType.UINT8)
    # Every UINT8 value, including 200, fits the physical column extent. The
    # generated conversion must keep values in [128, 255] nonnegative.
    buffer, allocation = _alloc("storage", _type(shape=(16, 256), valid=(-1, -1)), [_int(16), valid])
    update = _eval("buffer.set_validshape", buffer, ir.MakeTuple([_int(16), valid], SPAN))
    text = _emit(_program([allocation, update], [valid]))

    allocated_valid = re.search(r"valid_col = (%\w+)", text)
    updated_valid = re.search(r"pto\.set_validshape %\w+, %\w+, (%\w+)", text)
    assert allocated_valid is not None and updated_valid is not None
    for extent in (allocated_valid[1], updated_valid[1]):
        index_cast = re.search(rf"{re.escape(extent)} = arith\.index_cast (%\w+) : i64 to index", text)
        assert index_cast is not None
        extension = re.search(rf"{re.escape(index_cast[1])} = arith\.extui (%\w+) : i8 to i64", text)
        assert extension is not None
        assert f"{extension[1]} = builtin.unrealized_conversion_cast %arg0 : ui8 to i8" in text
    assert not re.search(r"arith\.index_cast %\w+ : i8 to index", text)
    _compile_native(tmp_path, text, addressed=False)


@pytest.mark.parametrize("valid", [(16, 32), (-1, 32), (16, -1)])
def test_metadata_mutation_requires_native_dynamic_valid_fields(valid):
    buffer, allocation = _alloc(
        "storage", _type(valid=valid), [_int(16 if i == 0 else 32) for i, v in enumerate(valid) if v == -1]
    )
    update = _eval("buffer.set_validshape", buffer, ir.MakeTuple([_int(16), _int(32)], SPAN))
    with pytest.raises(ValueError, match="both valid dimensions dynamic"):
        _emit(_program([allocation, update]))


def test_rank_one_dynamic_valid_operand_uses_native_column_axis(tmp_path):
    valid = _scalar("valid")
    _, allocation = _alloc("vector", _type(shape=(32,), valid=(-1,)), [valid])
    text = _emit(_program([allocation], [valid]))
    assert "rows=1, cols=32, v_row=1, v_col=?" in text
    assert "valid_row = " not in text
    assert "valid_col = %arg0" in text
    _compile_native(tmp_path, text, addressed=False)
