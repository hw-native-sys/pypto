# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Mat value selection must preserve incoming payloads without Mat-to-Mat copies."""

import re
from textwrap import indent

import pypto.language as pl
import pytest
import torch
from pypto import DataType, ir, passes
from pypto.backend._ptoas_locate import PTOAS_MIN_VERSION, check_ptoas_version, find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.debug import torch_codegen
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen

pytestmark = pytest.mark.usefixtures("ascend_backend")
_PLANNERS = [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]


def _program(body: str, selected: tuple[str, ...] = ("selected",)) -> ir.Program:
    # The original first halves remain independently observable after selection.
    # Local producers read different halves, so a redundant-load optimization
    # cannot turn an internally created value into either external allocation.
    names = ("a_mat", "b_mat", *selected)
    tail = []
    for index, name in enumerate(names):
        tail.extend(
            [
                f"left_{index} = pl.tile.move({name}, target_memory=pl.Mem.Left)",
                f"product_{index} = pl.tile.matmul(left_{index}, right)",
                f"output_{index} = pl.tile.store(product_{index}, [{16 * index}, 0], "
                f"{'out' if index == 0 else f'output_{index - 1}'})",
            ]
        )
    tail.append(f"return output_{len(names) - 1}")
    tail_text = "\n".join(tail)
    return pl.parse_program(f"""
@pl.program
class MatReferencePipeline:
    @pl.function(type=pl.FunctionType.AIC)
    def kernel(self, a: pl.Tensor[[32, 32], pl.FP16], b: pl.Tensor[[32, 32], pl.FP16],
               rhs: pl.Tensor[[32, 16], pl.FP16], flag: pl.Scalar[pl.BOOL],
               other: pl.Scalar[pl.BOOL], count: pl.Scalar[pl.INDEX],
               out: pl.Out[pl.Tensor[[{16 * len(names)}, 16], pl.FP32]]
               ) -> pl.Tensor[[{16 * len(names)}, 16], pl.FP32]:
        a_mat = pl.tile.load(a, [0, 0], [16, 32], target_memory=pl.Mem.Mat)
        b_mat = pl.tile.load(b, [0, 0], [16, 32], target_memory=pl.Mem.Mat)
        rhs_mat = pl.tile.load(rhs, [0, 0], [32, 16], target_memory=pl.Mem.Mat)
        right = pl.tile.move(rhs_mat, target_memory=pl.Mem.Right)
{indent(body.strip(), "        ")}
{indent(tail_text, "        ")}
""")


def _creation_program(kind: str) -> ir.Program:
    then_value = "pl.tile.load(a, [16, 0], [16, 32], target_memory=pl.Mem.Mat)" if kind[0] == "I" else "a_mat"
    else_value = "pl.tile.load(b, [16, 0], [16, 32], target_memory=pl.Mem.Mat)" if kind[1] == "I" else "b_mat"
    return _program(f"""
if flag:
    incoming_then = {then_value}
    selected = pl.yield_(incoming_then)
else:
    incoming_else = {else_value}
    selected = pl.yield_(incoming_else)
""")


def _regions(program):
    for function in program.functions.values():
        pending = [function.body]
        while pending:
            statement = pending.pop()
            if isinstance(statement, ir.SeqStmts):
                pending.extend(reversed(statement.stmts))
            elif isinstance(statement, ir.IfStmt):
                yield statement
                if statement.else_body is not None:
                    pending.append(statement.else_body)
                pending.append(statement.then_body)
            elif isinstance(statement, (ir.ForStmt, ir.WhileStmt)):
                yield statement
                pending.append(statement.body)


def _is_reference(value):
    descriptor = value.type
    return isinstance(descriptor, ir.BufferType) or (
        isinstance(descriptor, ir.TileType) and descriptor.memref is None
    )


def _lower(program, planner, enable_buffer_ir):
    captured = {}

    def after(pass_object, candidate):
        if pass_object.get_name() == "MaterializeSemanticAliases":
            captured["legalized"] = candidate

    instrument = passes.CallbackInstrument(after_pass=after)
    with passes.PassContext([instrument], memory_planner=planner, enable_buffer_ir=enable_buffer_ir):
        manager = PassManager.get_strategy(OptimizationStrategy.Default)
        lowered = manager.run_passes(program)
    assert "legalized" in captured
    kernel = lowered.get_function("kernel")
    assert kernel is not None
    expected_stage = ir.FunctionIRStage.Buffer if enable_buffer_ir else ir.FunctionIRStage.Functional
    assert kernel.ir_stage == expected_stage
    restored = ir.deserialize(ir.serialize(lowered))
    assert isinstance(restored, ir.Program)
    ir.assert_structural_equal(lowered, restored, enable_auto_mapping=True)
    native = codegen.PTOCodegen().generate(
        restored,
        emit_tile_addr=planner != passes.MemoryPlanner.PTOAS,
        emit_source_loc=False,
    )
    assert not any("pto.tmov" in line and line.count("loc=mat") == 2 for line in native.splitlines()), native
    return lowered, captured["legalized"], native


def _check_payloads(program, selected_halves, *, flag=False, other=False, count=0):
    a = torch.ones((32, 32), dtype=torch.float16)
    a[16:] = 2
    b = torch.full((32, 32), 3, dtype=torch.float16)
    b[16:] = 4
    rhs = torch.arange(1, 17, dtype=torch.float16).repeat(32, 1)
    sources = {"a0": a[:16], "a1": a[16:], "b0": b[:16], "b1": b[16:]}
    expected = torch.cat([sources[name].float() @ rhs.float() for name in ("a0", "b0", *selected_halves)])
    output = torch.full_like(expected, -777)
    namespace = {}
    exec(torch_codegen(program), namespace)  # noqa: S102
    namespace["kernel"](a, b, rhs, flag, other, count, output)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("enable_buffer_ir", [False, True])
@pytest.mark.parametrize("kind", ["II", "IE", "EI", "EE"])
def test_mat_yields_preserve_values_for_every_creation_location(planner, enable_buffer_ir, kind):
    lowered, legalized, native = _lower(_creation_program(kind), planner, enable_buffer_ir)
    references = [
        value for region in _regions(legalized) for value in region.return_vars if _is_reference(value)
    ]
    # II can write one destination directly. Any external source here still has
    # an independent old read, so sharing it would corrupt that output block.
    assert bool(references) == (kind != "II")
    if kind != "II":
        assert re.search(r"scf\.if[^\n]*->[^\n]*!pto\.tile_buf<loc=mat", native), native
    for flag in (False, True):
        arm = 0 if flag else 1
        source = "a" if flag else "b"
        half = 1 if kind[arm] == "I" else 0
        _check_payloads(lowered, (f"{source}{half}",), flag=flag)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("enable_buffer_ir", [False, True])
def test_branch_local_alias_of_external_mat_remains_an_external_value(planner, enable_buffer_ir):
    before = _program("""
if flag:
    branch_local_name = pl.tile.reshape(a_mat, [16, 32])
    selected = pl.yield_(branch_local_name)
else:
    selected = pl.yield_(b_mat)
""")
    lowered, legalized, _ = _lower(before, planner, enable_buffer_ir)
    assert any(_is_reference(value) for region in _regions(legalized) for value in region.return_vars)
    for flag in (False, True):
        _check_payloads(lowered, ("a0" if flag else "b0",), flag=flag)


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
@pytest.mark.parametrize("enable_buffer_ir", [False, True])
def test_internal_pinned_mat_producers_yield_references_without_retargeting(planner, enable_buffer_ir):
    # Both values are produced inside their arm, but the declarations prohibit
    # moving either allocation into a common target. PTOAS intentionally rejects
    # single-slot pinned declarations before this stage, so it is not included.
    before = _program("""
if flag:
    first: pl.Tile[[16, 32], pl.FP16, pl.MemRef("then_slot"), pl.Mem.Mat] = pl.tile.load(
        a, [16, 0], [16, 32], target_memory=pl.Mem.Mat)
    selected = pl.yield_(first)
else:
    second: pl.Tile[[16, 32], pl.FP16, pl.MemRef("else_slot"), pl.Mem.Mat] = pl.tile.load(
        b, [16, 0], [16, 32], target_memory=pl.Mem.Mat)
    selected = pl.yield_(second)
""")
    lowered, legalized, native = _lower(before, planner, enable_buffer_ir)
    assert any(_is_reference(value) for region in _regions(legalized) for value in region.return_vars)
    assert re.search(r"scf\.if[^\n]*->[^\n]*!pto\.tile_buf<loc=mat", native), native
    for flag in (False, True):
        _check_payloads(lowered, ("a1" if flag else "b1",), flag=flag)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("enable_buffer_ir", [False, True])
def test_nested_multi_result_mat_references_preserve_each_selected_value(planner, enable_buffer_ir):
    before = _program(
        """
if flag:
    if other:
        nested = pl.yield_(a_mat)
    else:
        nested = pl.yield_(b_mat)
    first, second = pl.yield_(nested, b_mat)
else:
    first, second = pl.yield_(b_mat, a_mat)
""",
        ("first", "second"),
    )
    lowered, _, _ = _lower(before, planner, enable_buffer_ir)
    for flag in (False, True):
        for other in (False, True):
            expected = (("a0" if other else "b0"), "b0") if flag else ("b0", "a0")
            _check_payloads(lowered, expected, flag=flag, other=other)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("enable_buffer_ir", [False, True])
def test_internal_duplicate_mat_slots_preserve_distinct_else_values(planner, enable_buffer_ir):
    before = _program(
        """
if flag:
    x = pl.tile.load(a, [16, 0], [16, 32], target_memory=pl.Mem.Mat)
    first, second = pl.yield_(x, x)
else:
    y = pl.tile.load(a, [16, 0], [16, 32], target_memory=pl.Mem.Mat)
    z = pl.tile.load(b, [16, 0], [16, 32], target_memory=pl.Mem.Mat)
    first, second = pl.yield_(y, z)
""",
        ("first", "second"),
    )
    lowered, legalized, _ = _lower(before, planner, enable_buffer_ir)
    assert any(_is_reference(value) for region in _regions(legalized) for value in region.return_vars)
    for flag in (False, True):
        _check_payloads(lowered, ("a1", "a1" if flag else "b1"), flag=flag)


def _loop_reference_program(kind):
    header = (
        "for (carried, spare, index) in pl.while_(init_values=(a_mat, b_mat, 0)):\n    pl.cond(index < count)"
        if kind == "while"
        else "for index, (carried, spare) in pl.range(count, init_values=(a_mat, b_mat)):"
    )
    boundary = (
        "first, second, final_index = pl.yield_(spare, carried, index + 1)"
        if kind == "while"
        else "first, second = pl.yield_(spare, carried)"
    )
    return _program(f"{header}\n    {boundary}", ("first", "second"))


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("enable_buffer_ir", [False, True])
@pytest.mark.parametrize("kind", ["for", "while"])
def test_readonly_mat_handle_carries_keep_zero_trip_and_parallel_backedges(planner, enable_buffer_ir, kind):
    before = _loop_reference_program(kind)
    lowered, _, native = _lower(before, planner, enable_buffer_ir)
    loops = [region for region in _regions(lowered) if isinstance(region, (ir.ForStmt, ir.WhileStmt))]
    assert len(loops) == 1
    assert sum(_is_reference(argument) for argument in loops[0].iter_args) == 2
    assert "!pto.tile_buf<loc=mat" in native
    # The pair is yielded simultaneously. A copy-based swap would either
    # request unsupported Mat transfers or collapse both outputs to one value.
    for count in (0, 1, 2, 3):
        expected = ("b0", "a0") if count % 2 else ("a0", "b0")
        _check_payloads(lowered, expected, count=count)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("enable_buffer_ir", [False, True])
@pytest.mark.parametrize("kind", ["if", "for", "while"])
def test_external_mat_reference_pipeline_compiles_natively(tmp_path, planner, enable_buffer_ir, kind):
    compiler = find_ptoas_binary()
    if compiler is None:
        pytest.skip("PTOAS is not available")
    version = check_ptoas_version(compiler)
    # A development checkout can support handle phis before the release pinned
    # by PyPTO. This check deliberately validates the supported release binary.
    release = re.escape(PTOAS_MIN_VERSION.removeprefix("v"))
    assert re.search(rf"\bptoas(?:\s+version)?\s+v?{release}(?:\s|$)", version), version
    before = _creation_program("EE") if kind == "if" else _loop_reference_program(kind)
    _, _, native = _lower(before, planner, enable_buffer_ir)
    source = tmp_path / "selected_mat.pto"
    output = tmp_path / "selected_mat.cpp"
    source.write_text(native)
    level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
    _run_ptoas(str(source), str(output), [f"--pto-level={level}", "--pto-arch=a3"])
    assert output.is_file()


def _declared_valid_reference_program(
    dynamic, addressed, *, nested=False, source_dynamic=None, matching_metadata=False
):
    """Already normalized references retain a result descriptor unlike either input."""
    span = ir.Span.unknown()

    def integer(value):
        return ir.ConstInt(value, DataType.INDEX, span)

    def dimensions(*values):
        return ir.MakeTuple([integer(value) if isinstance(value, int) else value for value in values], span)

    def call(name, arguments, result_type, kwargs=None):
        return ir.Call(ir.get_op(name), arguments, kwargs or {}, None, result_type, span)

    tensor_type = ir.TensorType([16, 32], DataType.FP32)
    output_type = ir.TensorType([32, 32], DataType.FP32)
    first_input = ir.Var("first_input", tensor_type, span)
    second_input = ir.Var("second_input", tensor_type, span)
    output = ir.Var("output", output_type, span)
    flag = ir.Var("flag", ir.ScalarType(DataType.BOOL), span)
    source_rows = ir.Var("source_rows", ir.ScalarType(DataType.INDEX), span)
    required_rows = ir.Var("required_rows", ir.ScalarType(DataType.INDEX), span)
    source_dynamic = dynamic if source_dynamic is None else source_dynamic
    incoming_rows = source_rows if source_dynamic else integer(12)
    selected_rows = incoming_rows if matching_metadata else required_rows if dynamic else integer(8)

    allocations = []

    def tile(name, rows, address):
        base = ir.Var(f"{name}_base", ir.PtrType(), span)
        memory = ir.MemRef(base, address if addressed else 0, 2048, span)
        tile_type = ir.TileType(
            [16, 32],
            DataType.FP32,
            memory,
            ir.TileView(valid_shape=[rows, integer(32)]),
            ir.MemorySpace.Vec,
        )
        allocation = call("tile.alloc", [integer(ir.MemorySpace.Vec.value), integer(2048)], ir.PtrType())
        allocations.append(ir.AssignStmt(base, allocation, span))
        return ir.Var(name, tile_type, span)

    first = tile("first", integer(16), 0)
    second = tile("second", incoming_rows, 2048)
    selected = ir.Var(
        "selected",
        ir.TileType(
            [16, 32],
            DataType.FP32,
            tile_view=ir.TileView(valid_shape=[selected_rows, integer(32)]),
            memory_space=ir.MemorySpace.Vec,
        ),
        span,
    )
    regions = []
    first_choice = first
    if nested:
        incoming = ir.Var(
            "incoming_selected",
            ir.TileType(
                [16, 32],
                DataType.FP32,
                tile_view=ir.TileView(valid_shape=[incoming_rows, integer(32)]),
                memory_space=ir.MemorySpace.Vec,
            ),
            span,
        )
        regions.append(
            ir.IfStmt(flag, ir.YieldStmt([first], span), ir.YieldStmt([second], span), [incoming], span)
        )
        # This source reference and the result have equal dynamic native types,
        # while their logical valid extents name different scalar parameters.
        first_choice = incoming
    regions.append(
        ir.IfStmt(flag, ir.YieldStmt([first_choice], span), ir.YieldStmt([second], span), [selected], span)
    )
    saved = ir.Var("saved", output_type, span)
    stored = ir.Var("stored", output_type, span)
    statements: list[ir.Stmt] = [
        *allocations,
        ir.AssignStmt(
            first,
            call(
                "tile.load",
                [first_input, dimensions(0, 0), dimensions(16, 32), dimensions(16, 32)],
                first.type,
            ),
            span,
        ),
        ir.AssignStmt(
            second,
            call(
                "tile.load",
                [second_input, dimensions(0, 0), dimensions(16, 32), dimensions(incoming_rows, 32)],
                second.type,
            ),
            span,
        ),
        *regions,
        ir.AssignStmt(saved, call("tile.store", [selected, dimensions(0, 0), output], output_type), span),
        # This read must still see the full first input after a narrow selected
        # result was consumed. Changing the incoming handle's metadata loses
        # the last eight rows of this independent output.
        ir.AssignStmt(stored, call("tile.store", [first, dimensions(16, 0), saved], output_type), span),
        ir.ReturnStmt([output], span),
    ]
    function = ir.Function(
        "kernel",
        [first_input, second_input, flag, source_rows, required_rows, (output, ir.ParamDirection.Out)],
        [output_type],
        ir.SeqStmts(statements, span),
        span,
        type=ir.FunctionType.AIV,
    )
    return ir.Program([function], "DeclaredReferenceValidity", span)


def _declared_valid_reference_artifact(dynamic, planner, enable_buffer_ir, **options):
    addressed = planner != passes.MemoryPlanner.PTOAS
    original = _declared_valid_reference_program(dynamic, addressed, **options)
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enable_buffer_ir):
        program = passes.lower_tile_to_buffer()(original) if enable_buffer_ir else original
        native = codegen.PTOCodegen().generate(program, emit_tile_addr=addressed, emit_source_loc=False)
    return program, native


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("enable_buffer_ir", [False, True])
@pytest.mark.parametrize(
    "dynamic,source_dynamic,nested",
    [(False, False, False), (True, True, False), (False, False, True), (False, True, True)],
    ids=["static-valid", "dynamic-owner", "static-selected", "dynamic-selected-to-static"],
)
def test_reference_result_owns_valid_extents_without_changing_the_full_source(
    planner, enable_buffer_ir, dynamic, source_dynamic, nested
):
    program, native = _declared_valid_reference_artifact(
        dynamic, planner, enable_buffer_ir, nested=nested, source_dynamic=source_dynamic
    )
    loads = [line for line in native.splitlines() if "pto.tload" in line]
    stores = [line for line in native.splitlines() if "pto.tstore" in line]
    assert len(loads) == len(stores) == 2, native
    first_load = re.search(r"outs\((%[A-Za-z0-9_]+)", loads[0])
    old_source_store = re.search(r"ins\((%[A-Za-z0-9_]+)", stores[1])
    assert first_load is not None and old_source_store is not None, native
    first = first_load.group(1)
    assert old_source_store.group(1) == first, native
    assert "pto.tmov" not in native, native
    constants = {
        name: int(value)
        for name, value in re.findall(r"(%[A-Za-z0-9_]+) = arith.constant (\d+) : index", native)
    }
    signature = next(line for line in native.splitlines() if "func.func @kernel" in line)
    required_rows = re.findall(r"(%[A-Za-z0-9_]+): index", signature)[-1] if dynamic else None
    edge_views = [line for line in native.splitlines() if "pto.subview" in line]
    updates = [line for line in native.splitlines() if "pto.set_validshape" in line]
    requested_views = 0
    if enable_buffer_ir:
        for view in edge_views:
            validity = re.search(r"valid \[(%[A-Za-z0-9_]+), (%[A-Za-z0-9_]+)\]", view)
            assert validity is not None, view
            rows, cols = validity.groups()
            requested_views += (rows == required_rows) if dynamic else (constants.get(rows) == 8)
            assert constants[cols] == 32
    else:
        for update in updates:
            validity = re.search(
                r"pto.set_validshape (%[A-Za-z0-9_]+), (%[A-Za-z0-9_]+), (%[A-Za-z0-9_]+)", update
            )
            assert validity is not None, update
            handle, rows, cols = validity.groups()
            assert handle != first, "A selected result must not narrow its old full source"
            requested_views += rows == required_rows if dynamic else False
            assert constants[cols] == 32
        assert "pto.tile_valid_rows" not in native and "pto.tile_valid_cols" not in native, native
    if not dynamic:
        requested_views += sum(
            "pto.treshape" in line and re.search(r"->[^\n]*v_row=8, v_col=32", line) is not None
            for line in native.splitlines()
        )
    assert requested_views == 2, native
    if enable_buffer_ir:
        namespace = {}
        exec(torch_codegen(program), namespace)  # noqa: S102
        first_input = torch.arange(512, dtype=torch.float32).reshape(16, 32)
        second_input = first_input + 1000
        for flag in (False, True):
            output = torch.full((32, 32), -777.0)
            namespace["kernel"](first_input, second_input, flag, 12, 8, output)
            expected = torch.full_like(output, -777)
            expected[:8] = (first_input if flag else second_input)[:8]
            expected[16:] = first_input
            torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize("enable_buffer_ir", [False, True])
@pytest.mark.parametrize(
    "dynamic,source_dynamic,nested",
    [(False, False, False), (True, True, False), (False, False, True), (False, True, True)],
)
def test_declared_reference_validity_compiles_natively(
    tmp_path, planner, enable_buffer_ir, dynamic, source_dynamic, nested
):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    _, native = _declared_valid_reference_artifact(
        dynamic, planner, enable_buffer_ir, nested=nested, source_dynamic=source_dynamic
    )
    source = tmp_path / "reference_valid.pto"
    output = tmp_path / "reference_valid.cpp"
    source.write_text(native)
    level = "level2" if planner == passes.MemoryPlanner.PTOAS else "level3"
    _run_ptoas(str(source), str(output), [f"--pto-level={level}", "--pto-arch=a3"])
    assert output.is_file()


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("enable_buffer_ir", [False, True])
def test_selected_reference_runtime_metadata_change_reports_the_pinned_backend_bound(
    planner, enable_buffer_ir
):
    # A locally allocated handle can get a fresh runtime-valid metadata view.
    # PTOAS v0.67 cannot perform that operation on a selected SCF handle; the
    # compiler must diagnose it before emitting invalid native operations.
    with pytest.raises(
        ValueError, match="cannot change runtime valid extents on a selected storage reference"
    ):
        _declared_valid_reference_artifact(True, planner, enable_buffer_ir, nested=True)


@pytest.mark.parametrize("enable_buffer_ir", [False, True])
def test_selected_reference_can_forward_unchanged_runtime_metadata(tmp_path, enable_buffer_ir):
    _, native = _declared_valid_reference_artifact(
        True, passes.MemoryPlanner.PTOAS, enable_buffer_ir, nested=True, matching_metadata=True
    )
    assert native.count(" = scf.if ") == 2, native
    if find_ptoas_binary() is not None:
        source = tmp_path / "unchanged_reference_valid.pto"
        output = tmp_path / "unchanged_reference_valid.cpp"
        source.write_text(native)
        _run_ptoas(str(source), str(output), ["--pto-level=level2", "--pto-arch=a3"])
        assert output.is_file()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
