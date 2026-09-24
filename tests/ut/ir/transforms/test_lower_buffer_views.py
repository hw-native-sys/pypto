# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Planned allocation capacity and typed aliases survive the Buffer boundary."""

from pathlib import Path

import pypto
import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen

from .buffer_test_utils import statements

_SPAN = ir.Span.unknown()
_PLANNERS = (passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS)
pytestmark = pytest.mark.usefixtures("ascend_backend")


def _int(value: int) -> ir.ConstInt:
    return ir.ConstInt(value, DataType.INDEX, _SPAN)


def _tuple(*values: int) -> ir.MakeTuple:
    return ir.MakeTuple([_int(value) for value in values], _SPAN)


def _planned_program(
    addressed: bool,
    *,
    offset: int = 64,
    capacity: int = 4096,
    anchor: bool = True,
    allocation: bool = True,
    window_shape: tuple[int, int] = (16, 32),
) -> ir.Program:
    """Model post-placement IR, including an interior member visited first."""
    origin = 4096 if addressed else 0
    base = ir.Var("storage", ir.PtrType(), _SPAN)
    output = ir.Var("output", ir.TensorType([16, 64], DataType.FP32), _SPAN)
    body: list[ir.Stmt] = []
    if allocation:
        call = ir.Call(
            ir.get_op("tile.alloc"), [_int(ir.MemorySpace.Vec.value), _int(capacity)], base.type, _SPAN
        )
        body.append(ir.AssignStmt(base, call, _SPAN))

    def tile(name: str, shape: tuple[int, int], address: int, size: int) -> ir.Var:
        memory = ir.MemRef(base, address, size, _SPAN, is_pinned=False)
        type_ = ir.TileType([_int(d) for d in shape], DataType.FP32, memory, None, ir.MemorySpace.Vec)
        return ir.Var(name, type_, _SPAN)

    def assign(var: ir.Var, name: str, args: list[ir.Expr], kwargs=None) -> None:
        body.append(ir.AssignStmt(var, ir.Call(ir.get_op(name), args, kwargs or {}, var.type, _SPAN), _SPAN))

    window = tile("window", window_shape, origin + offset, 2048)
    assign(window, "tile.create", [_tuple(*window_shape)], {"dtype": DataType.FP32})
    if anchor:
        full = tile("full_capacity", (32, 32), origin, capacity)
        assign(full, "tile.create", [_tuple(32, 32)], {"dtype": DataType.FP32})
    filled = tile("filled", window_shape, origin + offset, 2048)
    assign(
        filled,
        "tile.full",
        [_tuple(*window_shape), ir.ConstFloat(1.25, DataType.FP32, _SPAN)],
        {"dtype": DataType.FP32},
    )
    for row in (0, 8):
        # Distinct TileType and Var objects describe the same physical view.
        viewed = tile("reshaped", (8, 64), origin + offset, 2048)
        assign(viewed, "tile.reshape", [filled, _tuple(8, 64)])
        stored = ir.Var("stored", output.type, _SPAN)
        assign(stored, "tile.store", [viewed, _tuple(row, 0), output])
    body.append(ir.ReturnStmt([output], _SPAN))
    function = ir.Function(
        "kernel",
        [(output, ir.ParamDirection.Out)],
        [output.type],
        ir.SeqStmts(body, _SPAN),
        _SPAN,
        type=ir.FunctionType.InCore,
    )
    return ir.Program([function], "PlannedViews", _SPAN)


def _lower(program: ir.Program, planner: passes.MemoryPlanner) -> ir.Program:
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        return passes.lower_tile_to_buffer()(program)


def _calls(program: ir.Program, name: str) -> list[ir.Call]:
    result = []
    for statement in statements(program):
        expr = (
            statement.value
            if isinstance(statement, ir.AssignStmt)
            else (statement.expr if isinstance(statement, ir.EvalStmt) else None)
        )
        if isinstance(expr, ir.Call) and expr.op.name == ir.get_op(name).name:
            result.append(expr)
    return result


def _verify_round_trip(program: ir.Program) -> ir.Program:
    restored = ir.deserialize(ir.serialize(program))
    assert isinstance(restored, ir.Program)
    ir.assert_structural_equal(program, restored, enable_auto_mapping=True)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.BufferIR)
    assert passes.PropertyVerifierRegistry.verify(properties, restored) == []
    return restored


@pytest.mark.parametrize("planner", _PLANNERS)
def test_planned_capacity_address_and_repeated_view_identity_are_preserved(planner):
    addressed = planner != passes.MemoryPlanner.PTOAS
    original = _planned_program(addressed)
    snapshot = ir.serialize(original)
    result = _verify_round_trip(_lower(original, planner))
    assert ir.serialize(original) == snapshot
    (allocation,) = _calls(result, "buffer.alloc")
    assert isinstance(allocation.type, ir.BufferType)
    assert allocation.type.dtype == DataType.UINT8
    assert list(allocation.type.shape) == [128, 32]  # planned 4096 bytes, not the smaller descriptor
    assert len(allocation.args) == (2 if addressed else 1)
    if addressed:
        address = allocation.args[1]
        assert isinstance(address, ir.ConstInt) and address.value == 4096
    (subview,) = _calls(result, "buffer.subview")
    offsets = subview.args[1]
    assert isinstance(offsets, ir.MakeTuple)
    assert all(isinstance(value, ir.ConstInt) for value in offsets.elements)
    ir.assert_structural_equal(offsets, _tuple(2, 0))  # relative offset 64, not effective address 4160
    assert len(_calls(result, "buffer.reshape")) == 3
    stores = _calls(result, "buffer.store")
    assert len(stores) == 2 and stores[0].args[0].same_as(stores[1].args[0])
    assert not _calls(result, "buffer.copy")
    ir.assert_structural_equal(_lower(result, planner), result, enable_auto_mapping=True)


@pytest.mark.parametrize(
    "options,message",
    [
        ({"anchor": False}, "full-capacity MemRef anchor"),
        ({"allocation": False}, "planned tile.alloc capacity"),
        ({"capacity": 1024}, "descriptor window exceeds"),
        ({"capacity": 4097}, "capacity aligned to 32 bytes"),
        ({"offset": 16}, "offsets and windows aligned to 32 bytes"),
        ({"offset": 4096}, "descriptor window exceeds"),
        ({"window_shape": (16, 4)}, "physical rows aligned to 32 bytes"),
    ],
)
def test_unproved_or_non_native_planned_windows_fail_before_codegen(options, message):
    with pytest.raises(ValueError, match=message):
        _lower(_planned_program(True, **options), passes.MemoryPlanner.PYPTO)


def _reshaped_load(dtype_name: str) -> ir.Program:
    return pl.parse_program(f"""
@pl.program
class ReshapedLoad:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, source: pl.Tensor[[16, 32], pl.{dtype_name}],
               output: pl.Out[pl.Tensor[[8, 64], pl.{dtype_name}]]
               ) -> pl.Tensor[[8, 64], pl.{dtype_name}]:
        loaded = pl.load(source, [0, 0], [16, 32])
        reshaped = pl.reshape(loaded, [8, 64])
        result = pl.store(reshaped, [0, 0], output)
        return result
""")


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910B, BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize(
    "planned,dtype_name",
    [(False, dtype) for dtype in ("FP16", "BF16", "FP32", "INT32")] + [(True, "FP32")],
)
def test_automatic_views_compile_without_new_allocations_or_data_copies(
    tmp_path: Path, ascend_backend: BackendType, planner: passes.MemoryPlanner, planned: bool, dtype_name: str
) -> None:
    addressed = planner != passes.MemoryPlanner.PTOAS
    if planned:
        result = _lower(_planned_program(addressed), planner)
    else:
        with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
            manager = PassManager.get_strategy(OptimizationStrategy.Default)
            result = manager.run_passes(_reshaped_load(dtype_name))
    restored = _verify_round_trip(result)
    assert len(_calls(restored, "buffer.alloc")) == 1
    text = codegen.PTOCodegen().generate(restored, emit_tile_addr=False, emit_source_loc=False)
    assert text.count("pto.alloc_tile") == 1
    assert text.count("pto.subview") == (1 if planned else 0)
    assert text.count("pto.treshape") == (3 if planned else 2)
    assert "pto.tmov" not in text and "pto.textract" not in text
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    source, output = tmp_path / "lowered_views.pto", tmp_path / "lowered_views.cpp"
    source.write_text(text)
    arch = "a2" if ascend_backend == BackendType.Ascend910B else "a5"
    _run_ptoas(
        str(source), str(output), [f"--pto-arch={arch}", f"--pto-level={'level3' if addressed else 'level2'}"]
    )
    assert "TRESHAPE(" in output.read_text()


_WINDOW_PROGRAMS = """
@pl.program
class StaticWindow:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]
               ) -> pl.Tensor[[32, 32], pl.FP32]:
        whole = pl.load(x, [0, 0], [64, 64])
        window = pl.tile.slice(whole, [16, 32], [16, 32])
        doubled = pl.add(window, window)
        canvas = pl.tile.full([32, 32], dtype=pl.FP32, value=0.0)
        placed = pl.tile.assemble(canvas, doubled, [16, 0])
        result = pl.store(placed, [0, 0], out)
        return result

@pl.program
class RuntimeWindow:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32], row: pl.Scalar[pl.INDEX],
               out: pl.Out[pl.Tensor[[16, 64], pl.FP32]]) -> pl.Tensor[[16, 64], pl.FP32]:
        whole = pl.load(x, [0, 0], [64, 64])
        window = pl.tile.slice(whole, [16, 64], [row, 0])
        doubled = pl.add(window, window)
        result = pl.store(doubled, [0, 0], out)
        return result

@pl.program
class RuntimeValid:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[16, 64], pl.FP32], cols: pl.Scalar[pl.INDEX],
               out: pl.Out[pl.Tensor[[16, 64], pl.FP32]]) -> pl.Tensor[[16, 64], pl.FP32]:
        loaded: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(
            x, [0, 0], [16, 64], [16, cols], target_memory=pl.MemorySpace.Vec)
        doubled = pl.add(loaded, loaded)
        result = pl.store(doubled, [0, 0], out)
        return result

@pl.program
class NarrowedValid:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[16, 64], pl.FP32], cols: pl.Scalar[pl.INDEX],
               out: pl.Out[pl.Tensor[[16, 64], pl.FP32]]) -> pl.Tensor[[16, 64], pl.FP32]:
        loaded = pl.load(x, [0, 0], [16, 64])
        narrowed = pl.tile.set_validshape(loaded, 16, cols)
        doubled = pl.add(narrowed, narrowed)
        result = pl.store(doubled, [0, 0], out)
        return result
"""


def _window_program(name: str) -> ir.Program:
    source = _WINDOW_PROGRAMS.split("@pl.program")
    (text,) = [block for block in source if f"class {name}:" in block]
    return pl.parse_program("@pl.program" + text)


def _buffer_type(call: ir.Call) -> ir.BufferType:
    assert isinstance(call.type, ir.BufferType)
    return call.type


def _lower_default(program: ir.Program, planner: passes.MemoryPlanner) -> ir.Program:
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=True):
        return _verify_round_trip(PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program))


def _compile(tmp_path: Path, program: ir.Program, backend: BackendType, addressed: bool) -> str:
    text = codegen.PTOCodegen().generate(program, emit_tile_addr=False, emit_source_loc=False)
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    source, output = tmp_path / "window.pto", tmp_path / "window.cpp"
    source.write_text(text)
    arch = "a2" if backend == BackendType.Ascend910B else "a5"
    _run_ptoas(
        str(source), str(output), [f"--pto-arch={arch}", f"--pto-level={'level3' if addressed else 'level2'}"]
    )
    return text


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910B, BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("planner", _PLANNERS)
def test_slice_and_assemble_become_use_site_windows(
    tmp_path: Path, ascend_backend: BackendType, planner: passes.MemoryPlanner
) -> None:
    program = _window_program("StaticWindow")
    addressed = planner != passes.MemoryPlanner.PTOAS
    if planner == passes.MemoryPlanner.DSA_RP:
        # DSA_RP places the sum over part of the live window's source; the
        # verifier cannot prove that partial overlap safe and fails closed.
        with pytest.raises(pypto.Error, match="partially overlapping"):
            _lower_default(program, planner)
        return
    lowered = _lower_default(program, planner)
    windows = _calls(lowered, "buffer.subview")
    # Slice window and assemble window, each with an explicit valid clause.
    typed = [call for call in windows if _buffer_type(call).dtype == DataType.FP32]
    assert len(typed) == 2 and all(len(call.args) == 3 for call in typed)
    assert len(_calls(lowered, "buffer.copy")) == 1
    text = _compile(tmp_path, lowered, ascend_backend, addressed)
    assert "valid [" in text and "pto.tmov ins(" in text


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910B, BackendType.Ascend950], indirect=True)
def test_runtime_slice_offset_stays_an_operand_of_its_window(
    tmp_path: Path, ascend_backend: BackendType
) -> None:
    lowered = _lower_default(_window_program("RuntimeWindow"), passes.MemoryPlanner.PTOAS)
    (window,) = [
        call for call in _calls(lowered, "buffer.subview") if _buffer_type(call).dtype == DataType.FP32
    ]
    offsets = window.args[1]
    assert isinstance(offsets, ir.MakeTuple) and not isinstance(offsets.elements[0], ir.ConstInt)
    _compile(tmp_path, lowered, ascend_backend, addressed=False)


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
def test_runtime_window_overlapping_its_planned_destination_fails_closed(
    planner: passes.MemoryPlanner,
) -> None:
    # The planner reuses the parent's storage for the sum; a runtime window
    # offset cannot prove that range disjoint from the source window.
    with pytest.raises(pypto.Error, match="partially overlapping"):
        _lower_default(_window_program("RuntimeWindow"), planner)


@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910B, BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("planner", _PLANNERS)
# NarrowedValid shares one handle: the load states its full extents, then
# set_validshape narrows them, then the sum states its own.
@pytest.mark.parametrize("name,updates", [("RuntimeValid", 2), ("NarrowedValid", 3)])
def test_runtime_valid_extents_update_the_handle_metadata(
    tmp_path: Path, ascend_backend: BackendType, planner: passes.MemoryPlanner, name: str, updates: int
) -> None:
    lowered = _lower_default(_window_program(name), planner)
    assert len(_calls(lowered, "buffer.set_validshape")) == updates
    dynamic = [call for call in _calls(lowered, "buffer.alloc") if -1 in list(_buffer_type(call).valid_shape)]
    extents = [call.args[0] for call in dynamic]
    assert extents and all(isinstance(value, ir.MakeTuple) and len(value.elements) == 2 for value in extents)
    text = _compile(tmp_path, lowered, ascend_backend, planner != passes.MemoryPlanner.PTOAS)
    assert "v_row=?, v_col=?" in text and text.count("pto.set_validshape") == updates


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
