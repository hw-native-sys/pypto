# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Native If Tile handles retain their sources without payload copies or pools."""

import re
import subprocess
from textwrap import indent

import pypto.language as pl
import pytest
from pypto import codegen, ir, passes
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _get_ptoas_flags
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

from .storage_test_utils import _assert_descriptor_carries, _assert_dynamic_if_results, _Nodes, _program

_PLANNERS = [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]


def _lower(program, planner, instruments=(), *, enable_buffer_ir=False):
    with passes.PassContext(list(instruments), memory_planner=planner, enable_buffer_ir=enable_buffer_ir):
        lowered = PassManager(OptimizationStrategy.Default).run_passes(program)
        text = codegen.PTOCodegen().generate(
            lowered, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS, emit_source_loc=False
        )
    return lowered, text


PinnedInputs = _program(
    """
    a: pl.Tile[[4, 8], pl.FP32, pl.MemRef("a_storage"), pl.Mem.Vec] = pl.load(
        x, [0, 0], [4, 8], target_memory=pl.Mem.Vec
    )
    b: pl.Tile[[4, 8], pl.FP32, pl.MemRef("b_storage"), pl.Mem.Vec] = pl.load(
        x, [4, 0], [4, 8], target_memory=pl.Mem.Vec
    )
    if flag > 0:
        selected = pl.yield_(a)
    else:
        selected = pl.yield_(b)
    return pl.store(selected, [0, 0], out)
"""
)


MixedResults = _program(
    """
    a = pl.load(x, [0, 0], [4, 8])
    b = pl.load(x, [4, 0], [4, 8])
    if flag > 0:
        local_a = pl.tile.adds(a, 1.0)
        first, number, second = pl.yield_(local_a, flag + pl.const(1, pl.INT32), b)
    else:
        local_b = pl.tile.muls(b, 2.0)
        first, number, second = pl.yield_(local_b, flag - pl.const(1, pl.INT32), a)
    combined = pl.tile.adds(first, pl.cast(number, pl.FP32))
    pl.store(combined, [0, 0], out)
    return pl.store(second, [4, 0], out)
""",
    output_shape="[8, 8]",
)


DynamicValid = _program(
    """
    a = pl.tile.load(x, [0, 0], [4, 8], [rows_a, 8], target_memory=pl.Mem.Vec)
    b = pl.tile.load(x, [4, 0], [4, 8], [rows_a, 8], target_memory=pl.Mem.Vec)
    if flag > 0:
        selected = pl.yield_(a)
    else:
        selected = pl.yield_(b)
    return pl.store(selected, [0, 0], out)
""",
    rows=True,
)


SelectedView = _program(
    """
    parent = pl.load(x, [0, 0], [8, 8])
    a = pl.tile.slice(parent, [4, 8], [0, 0])
    b = pl.tile.slice(parent, [4, 8], [4, 0])
    if flag > 0:
        selected = pl.yield_(a)
    else:
        selected = pl.yield_(b)
    alias = selected
    reshaped = pl.tile.reshape(alias, [2, 16])
    return pl.store(reshaped, [0, 0], out)
""",
    output_shape="[2, 16]",
)


@pytest.mark.parametrize("program", [PinnedInputs, MixedResults, DynamicValid, SelectedView])
@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("enable_buffer_ir", [False, True])
def test_native_if_compiles_without_copies(
    program, planner, enable_buffer_ir, tmp_path, ascend_backend, pass_verification_instruments
):
    """Check real PTOAS acceptance, including views and runtime validity."""
    binary = find_ptoas_binary()
    if binary is None:
        pytest.skip("PTOAS is unavailable")
    # PTOAS has no author-pinned single-allocation contract. Exercise declared
    # storage with the planners that implement that existing contract.
    if program is PinnedInputs and planner == passes.MemoryPlanner.PTOAS:
        pytest.skip("Single-slot declared allocations require PYPTO or DSA_RP")
    lowered, text = _lower(program, planner, pass_verification_instruments, enable_buffer_ir=enable_buffer_ir)
    _assert_dynamic_if_results(lowered)
    assert re.search(r"scf\.if .* -> \(.*!pto\.tile_buf", text), text
    assert "pto.tmov" not in text
    assert "slots=2" not in lowered.as_python()
    _compile_native_loop(text, planner, tmp_path)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("placement", ["inside_for", "inside_while", "beside_for"])
@pytest.mark.parametrize("planner", _PLANNERS)
def test_native_if_with_loops_bypasses_global_buffer_ir(
    placement, planner, tmp_path, ascend_backend, pass_verification_instruments
):
    """Tile selections remain native even when the same function also has a loop."""
    binary = find_ptoas_binary()
    if binary is None:
        pytest.skip("PTOAS is unavailable")
    selection = """if flag > 0:
    selected = pl.yield_(a)
else:
    selected = pl.yield_(b)
pl.store(selected, [0, 0], out)"""
    if placement == "inside_for":
        body = f"for i in pl.range(count):\n{indent(selection, '    ')}"
    elif placement == "inside_while":
        body = f"i = pl.const(0, pl.INDEX)\nwhile i < count:\n{indent(selection, '    ')}\n    i = i + 1"
    else:
        body = f"for i in pl.range(count):\n    pl.store(a, [0, 0], out)\n{selection}"
    program = _program(
        f"""
        a = pl.load(x, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
        b = pl.load(x, [4, 0], [4, 8], target_memory=pl.Mem.Vec)
{indent(body, "        ")}
        return out
"""
    )
    lowered, text = _lower(program, planner, pass_verification_instruments, enable_buffer_ir=True)
    function = lowered.get_function("kernel")
    assert function is not None and function.ir_stage == ir.FunctionIRStage.Functional, lowered.as_python()
    assert re.search(r"scf\.if .* -> \(.*!pto\.tile_buf", text), text
    assert ("scf.while" if placement == "inside_while" else "scf.for") in text, text
    _compile_native_loop(text, planner, tmp_path)


@pytest.mark.ptoas_compile
def test_shared_accumulator_declaration_dominates_if_arms(tmp_path, ascend_backend):
    """Coalesced producers need one declaration, with validity set in the loop."""

    @pl.program
    class SharedAccumulator:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            b: pl.Tensor[[16, 16], pl.FP16],
            flag: pl.Scalar[pl.INT32],
            valid_rows: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            rhs = pl.load(b, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
            for i in pl.range(2):
                rows = pl.min(valid_rows + i, 16)
                lhs = pl.load(a, [0, 0], [16, 16], valid_shape=[rows, 16], target_memory=pl.MemorySpace.Mat)
                if flag != 0:
                    seed = pl.matmul(lhs, rhs)
                    result = pl.matmul_acc(seed, lhs, rhs)
                else:
                    result = pl.matmul(lhs, rhs)
                out = pl.store(result, [0, 0], out)
            return out

    binary = find_ptoas_binary()
    if binary is None:
        pytest.skip("PTOAS is unavailable")
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS, enable_buffer_ir=False):
        lowered = PassManager(OptimizationStrategy.Default).run_passes(SharedAccumulator)
    text = codegen.PTOCodegen().generate(lowered, emit_tile_addr=False, emit_source_loc=False)
    allocations = [line for line in text.splitlines() if "pto.alloc_tile" in line and "loc=acc" in line]
    assert len(allocations) == 1, text
    assert text.index(allocations[0]) < text.index("scf.for"), text
    # Shared producers still declare one buffer, while the If result uniformly
    # remains a dynamic descriptor even when its arms yield that same handle.
    _assert_dynamic_if_results(lowered)
    assert re.search(r"scf.if .* -> .*loc=acc", text), text
    _compile_native_loop(text, passes.MemoryPlanner.PTOAS, tmp_path)


def _compile_native_loop(text, planner, tmp_path):
    binary = find_ptoas_binary()
    if binary is None:
        # Ordinary unit-test environments need no compiler. The dedicated CI
        # job uses --require-ptoas to enforce compilation with pinned PTOAS.
        pytest.skip("PTOAS is unavailable")
    source = tmp_path / "native_loop.pto"
    source.write_text(text)
    compiled = subprocess.run(
        [binary, str(source), "-o", str(tmp_path / "native_loop.cpp"), *_get_ptoas_flags(planner)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("planner", _PLANNERS)
def test_dynamic_acc_initializer_keeps_one_handle_during_inplace_loop(planner, ascend_backend, tmp_path):
    """A stable runtime handle needs neither an entry Acc copy nor a fixed MemRef."""
    program = pl.parse_program("""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[32, 32], pl.FP16], flag: pl.Scalar[pl.INT32],
               out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]) -> pl.Tensor[[32, 32], pl.FP32]:
        tile = pl.load(x, [0, 0], [32, 32], target_memory=pl.Mem.Mat)
        lhs = pl.tile.move(tile, target_memory=pl.Mem.Left)
        rhs = pl.tile.move(tile, target_memory=pl.Mem.Right)
        a = pl.tile.matmul(lhs, rhs)
        b = pl.tile.matmul(lhs, rhs)
        if flag > 0:
            selected = pl.yield_(a)
        else:
            selected = pl.yield_(b)
        for i, (carry,) in pl.range(2, init_values=(selected,)):
            updated = pl.tile.matmul_acc(carry, lhs, rhs)
            result = pl.yield_(updated)
        return pl.store(result, [0, 0], out)
""")
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=False):
        lowered = PassManager(OptimizationStrategy.Default).run_passes(program)
        _assert_dynamic_if_results(lowered)
        text = codegen.PTOCodegen().generate(
            lowered, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS, emit_source_loc=False
        )
    assert "= scf.if" in text and "scf.for" in text, text
    assert not any("pto.tmov" in line and line.count("loc=acc") == 2 for line in text.splitlines()), text
    _compile_native_loop(text, planner, tmp_path)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("loop", ["for", "while"])
@pytest.mark.parametrize("boundary", ["initializer", "backedge"])
@pytest.mark.parametrize("planner", _PLANNERS)
def test_updating_loop_isolates_selected_or_observed_initializer(
    loop, boundary, planner, tmp_path, ascend_backend, pass_verification_instruments
):
    """Both loops forward selected descriptors without copying their initializers."""
    select = """if flag > 0:
    selected = a
else:
    selected = b"""
    update = (
        "selected = pl.add(selected, delta)"
        if boundary == "initializer"
        else """if flag > 0:
    selected = selected
else:
    selected = b"""
    )
    before = select if boundary == "initializer" else "selected = a"
    body = update if loop == "for" else update + "\ni = i + 1"
    header = "for i in pl.range(count):" if loop == "for" else "while i < count:"

    def indent(text, spaces):
        return "\n".join(" " * spaces + line for line in text.splitlines())

    program = _program(
        f"""
        a = pl.load(x, [0, 0], [4, 8])
        b = pl.load(x, [4, 0], [4, 8])
        delta = pl.load(x, [8, 0], [4, 8])
{indent(before, 8)}
        i = pl.const(0, pl.INDEX)
        {header}
{indent(body, 12)}
        pl.store(selected, [0, 0], out)
        pl.store(a, [4, 0], out)
        return pl.store(b, [8, 0], out)
""",
        input_shape="[12, 8]",
        output_shape="[12, 8]",
    )
    lowered, text = _lower(program, planner, pass_verification_instruments)
    assert "tile.move(" not in lowered.as_python(), "descriptor carries need no entry/writeback copy"
    keyword = "scf.for" if loop == "for" else "scf.while"
    assert "!pto.tile_buf" in next(line for line in text.splitlines() if keyword in line)
    assert ("scf.for" if loop == "for" else "scf.while") in text, text
    _compile_native_loop(text, planner, tmp_path)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("fresh_then", [False, True])
@pytest.mark.parametrize("loop", ["for", "while"])
def test_conditional_carry_swap_uses_descriptors(
    planner, fresh_then, loop, ascend_backend, tmp_path, pass_verification_instruments
):
    """Fresh merge outputs can hide the opposite arm's carry-copy cycle."""
    fresh = """fresh_left = pl.tile.mrgsort(left, block_len=64)
fresh_right = pl.tile.mrgsort(right, block_len=64)
next_left, next_right = pl.yield_(fresh_left, fresh_right)"""
    swap = "next_left, next_right = pl.yield_(right, left)"

    def indent(text):
        return "\n".join(" " * 16 + line for line in text.splitlines())

    header = (
        "for i, (left, right) in pl.range(count, init_values=(a, b)):"
        if loop == "for"
        else "for i, left, right in pl.while_(init_values=(0, a, b)):\n            pl.cond(i < count)"
    )
    backedge = (
        "result_left, result_right = pl.yield_(next_left, next_right)"
        if loop == "for"
        else "final_i, result_left, result_right = pl.yield_(i + 1, next_left, next_right)"
    )
    program = _program(
        f"""
        a = pl.load(x, [0, 0], [1, 4096])
        b = pl.load(x, [1, 0], [1, 4096])
        {header}
            if flag > 0:
{indent(fresh if fresh_then else swap)}
            else:
{indent(swap if fresh_then else fresh)}
            {backedge}
        pl.store(result_left, [0, 0], out)
        return pl.store(result_right, [1, 0], out)
""",
        input_shape="[2, 4096]",
        output_shape="[2, 4096]",
    )
    # Both slots can retain the other producer's old payload across a swap.
    # Stage both backedges before updating their retained buffers.
    lowered, text = _lower(program, planner, pass_verification_instruments)
    _assert_descriptor_carries(lowered, 2)
    assert text.count("pto.tmov") == 4
    _compile_native_loop(text, planner, tmp_path)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("producer", ["mrgsort", "column_stats"])
@pytest.mark.parametrize("fresh_then", [False, True])
def test_independent_conditional_carry_copies_remain_supported(
    planner, nested, producer, fresh_then, ascend_backend
):
    """Fresh-or-own choices snapshot unchanged arms before updating any carry."""
    if producer == "mrgsort":
        input_shape, output_shape, tile_shape = "[2, 4096]", "[2, 4096]", "[1, 4096]"
        second_offset = "[1, 0]"
        fresh = """fresh_left = pl.tile.mrgsort(left, block_len=64)
fresh_right = pl.tile.mrgsort(right, block_len=64)"""
    else:
        input_shape, output_shape, tile_shape = "[32, 4]", "[32, 2]", "[32, 1]"
        second_offset = "[0, 1]"
        # Like CSA's online softmax, both statistics become column vectors
        # after arithmetic on row vectors. The maximum must preserve old left
        # until alpha is computed, and both outputs need independent writeback.
        fresh = """delta_left = pl.load(x, [0, 2], [32, 1])
delta_right = pl.load(x, [0, 3], [32, 1])
fresh_left = pl.maximum(left, delta_left)
alpha = pl.exp(pl.sub(left, fresh_left))
beta = pl.exp(pl.sub(delta_left, fresh_left))
fresh_right = pl.add(pl.mul(alpha, right), pl.mul(beta, delta_right))"""

    def indent(text, spaces):
        return "\n".join(" " * spaces + line for line in text.splitlines())

    fresh += "\nnext_left, next_right = pl.yield_(fresh_left, fresh_right)"
    if nested:
        fresh = (
            "if count > 1:\n"
            + indent(fresh.replace("next_left, next_right", "inner_left, inner_right"), 4)
            + "\nelse:\n    inner_left, inner_right = pl.yield_(left, right)"
            + "\nnext_left, next_right = pl.yield_(inner_left, inner_right)"
        )
    unchanged = "next_left, next_right = pl.yield_(left, right)"
    program = _program(
        f"""
        a = pl.load(x, [0, 0], {tile_shape})
        b = pl.load(x, {second_offset}, {tile_shape})
        for i, (left, right) in pl.range(count, init_values=(a, b)):
            if flag > 0:
{indent(fresh if fresh_then else unchanged, 16)}
            else:
{indent(unchanged if fresh_then else fresh, 16)}
            result_left, result_right = pl.yield_(next_left, next_right)
        pl.store(result_left, [0, 0], out)
        return pl.store(result_right, {second_offset}, out)
""",
        input_shape=input_shape,
        output_shape=output_shape,
    )
    lowered, text = _lower(program, planner)
    _assert_descriptor_carries(lowered, 2)
    # mrgsort snapshots its inputs; column statistics retain old left until
    # alpha has consumed it, using a staged transfer at the backedge.
    assert text.count("pto.tmov") == 2, text
    assert text.count("scf.if") >= 1 + int(nested), text


def _assert_saved_inputs_have_independent_updates(program, initial_names):
    originals, updates = [], []
    for assign in _Nodes(program).assignments:
        tile = assign.var.type
        name = assign.var.name_hint.split("__")[0]
        if isinstance(tile, ir.TileType) and tile.memref is not None:
            if name in initial_names:
                originals.append(tile.memref)
            elif name == "fresh":
                updates.append(tile.memref)
    assert len(originals) == len(initial_names) and updates
    assert all(not ir.MemRef.may_alias(old, new) for old in originals for new in updates)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("fresh_then", [False, True])
def test_single_conditional_carry_copy_snapshots_unchanged_branch(planner, fresh_then, ascend_backend):
    """Even one selected writeback must avoid copying the unchanged carry onto itself."""

    fresh = """fresh = pl.tile.mrgsort(carried, block_len=64)
selected = pl.yield_(fresh)"""
    unchanged = "selected = pl.yield_(carried)"

    def indent(text):
        return "\n".join(" " * 16 + line for line in text.splitlines())

    program = _program(
        f"""
        initial = pl.load(x, [0, 0], [1, 4096])
        for i, (carried,) in pl.range(count, init_values=(initial,)):
            if flag > 0:
{indent(fresh if fresh_then else unchanged)}
            else:
{indent(unchanged if fresh_then else fresh)}
            result = pl.yield_(selected)
        return pl.store(result, [0, 0], out)
""",
        input_shape="[1, 4096]",
        output_shape="[1, 4096]",
    )

    lowered, text = _lower(program, planner)
    _assert_descriptor_carries(lowered, 1)
    assert text.count("pto.tmov") == 1, text


def test_zero_trip_selected_same_hint_does_not_insert_self_copy(ascend_backend):
    """Standalone storage passes preserve the dead body without adding an illegal TMOV."""

    Before = _program(
        """
        initial = pl.load(x, [0, 0], [1, 4096], target_memory=pl.Mem.Vec)
        for _i, (carried,) in pl.range(0, init_values=(initial,)):
            if flag > 0:
                selected = pl.yield_(carried)
            else:
                fresh = pl.tile.mrgsort(carried, block_len=64)
                selected = pl.yield_(fresh)
            result = pl.yield_(selected)
        return pl.store(result, [0, 0], out)
    """,
        input_shape="[1, 4096]",
        output_shape="[1, 4096]",
    )

    # The full pipeline simplifies zero-trip loops away before storage repair.
    # Invoke these passes directly so elimination cannot hide a dead self-copy.
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO, enable_buffer_ir=False):
        lowered = passes.memory_reuse()(passes.materialize_semantic_aliases()(passes.init_mem_ref()(Before)))

    nodes = _Nodes(lowered)
    copies = [
        op
        for op in nodes.assignments
        if isinstance(op.value, ir.Call) and op.value.op.name == ir.get_op("tile.move").name
    ]
    assert len(nodes.loops) == len(nodes.branches) == 1, "the standalone dead loop body must remain"
    # The retained non-inplace producer needs an operand snapshot, even though
    # this loop never executes. It must not become a same-address carry copy.
    assert len(copies) == 1
    assert copies[0].var.name_hint.startswith("recurrent_input_")
    assert copies[0].var.type.memref is not None
    assert copies[0].value.args[0].type.memref is None
    _assert_descriptor_carries(lowered, 1)
    assert not any("_carry_spill_" in op.var.name_hint for op in nodes.assignments)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("update_then", [False, True])
def test_nested_conditional_assemble_carries_need_no_snapshot(planner, update_then, ascend_backend):
    """Nested choices among required writebacks keep one descriptor per carry."""
    unchanged = "next_packed, next_scale, next_lanes = pl.yield_(packed, scale, lanes)"
    updates = """if i % 2 == 0:
    packed_update = pl.tile.assemble(packed, packed_piece, [i % 8, 0])
    scale_update = pl.tile.assemble(scale, scale_piece, [0, 0])
    lanes_update = pl.tile.assemble(lanes, lane_piece, [0, 0])
    inner_packed, inner_scale, inner_lanes = pl.yield_(packed_update, scale_update, lanes_update)
else:
    inner_packed, inner_scale, inner_lanes = pl.yield_(packed, scale, lanes)
next_packed, next_scale, next_lanes = pl.yield_(inner_packed, inner_scale, inner_lanes)"""

    def indent(text):
        return "\n".join(" " * 16 + line for line in text.splitlines())

    program = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[8, 64], pl.INT8], flag: pl.Scalar[pl.INT32],
               count: pl.Scalar[pl.INDEX],
               scale_out: pl.Out[pl.Tensor[[1, 32], pl.INT8]],
               lanes_out: pl.Out[pl.Tensor[[1, 32], pl.INT32]],
               out: pl.Out[pl.Tensor[[8, 64], pl.INT8]]) -> pl.Tensor[[8, 64], pl.INT8]:
        dead = pl.load(x, [0, 0], [8, 64])
        pl.store(dead, [0, 0], out)
        packed_initial = pl.tile.full([8, 64], dtype=pl.INT8, value=0)
        scale_initial = pl.tile.full([1, 32], dtype=pl.INT8, value=0)
        lanes_initial = pl.tile.full([1, 32], dtype=pl.INT32, value=0)
        packed_piece = pl.load(x, [0, 0], [1, 64])
        scale_piece = pl.load(x, [1, 0], [1, 32])
        lane_piece = pl.tile.full([1, 32], dtype=pl.INT32, value=1)
        for i, (packed, scale, lanes) in pl.range(
            count, init_values=(packed_initial, scale_initial, lanes_initial)
        ):
            if flag > 0:
{indent(updates if update_then else unchanged)}
            else:
{indent(unchanged if update_then else updates)}
            result_packed, result_scale, result_lanes = pl.yield_(next_packed, next_scale, next_lanes)
        pl.store(result_scale, [0, 0], scale_out)
        pl.store(result_lanes, [0, 0], lanes_out)
        return pl.store(result_packed, [0, 0], out)
""")
    lowered, text = _lower(program, planner)
    # Integer assemble uses a data TMOV into each destination window. There
    # must be no additional loop-carry moves or snapshots around those writes.
    assert "tile.move(" not in lowered.as_python()
    assert "_carry_spill_" not in lowered.as_python()
    assert text.count("pto.tmov") == 3, text
    assert text.count("scf.if") >= 2, text
    assert "scf.for" in text, text


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("saved_then_a", [False, True])
@pytest.mark.parametrize("alias", ["rename", "reshape"])
def test_selected_observer_of_loop_initializer_cannot_be_clobbered(
    planner, saved_then_a, alias, tmp_path, ascend_backend
):
    """An old selection can observe the initializer through either runtime arm."""
    saved_shape = "[2, 2048]" if alias == "reshape" else "[1, 4096]"
    alias_statement = (
        "saved_alias = pl.tile.reshape(saved, [2, 2048])" if alias == "reshape" else "saved_alias = saved"
    )
    program = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[2, 4096], pl.FP32], saved_flag: pl.Scalar[pl.INT32],
               update_flag: pl.Scalar[pl.INT32], count: pl.Scalar[pl.INDEX],
               saved_out: pl.Out[pl.Tensor[{saved_shape}, pl.FP32]],
               out: pl.Out[pl.Tensor[[1, 4096], pl.FP32]]) -> pl.Tensor[[1, 4096], pl.FP32]:
        a = pl.load(x, [0, 0], [1, 4096])
        b = pl.load(x, [1, 0], [1, 4096])
        if saved_flag > 0:
            saved = pl.yield_({"a" if saved_then_a else "b"})
        else:
            saved = pl.yield_({"b" if saved_then_a else "a"})
        {alias_statement}
        for i, (carried,) in pl.range(count, init_values=(a,)):
            if update_flag > 0:
                fresh = pl.tile.mrgsort(carried, block_len=64)
                selected = pl.yield_(fresh)
            else:
                selected = pl.yield_(carried)
            result = pl.yield_(selected)
        pl.store(saved_alias, [0, 0], saved_out)
        return pl.store(result, [0, 0], out)
""")
    lowered, text = _lower(program, planner)
    _assert_descriptor_carries(lowered, 1)
    _assert_saved_inputs_have_independent_updates(lowered, {"a", "b"})
    assert re.search(r"scf\.if .* -> \(.*!pto\.tile_buf", text), text
    assert "scf.for" in text, text
    _compile_native_loop(text, planner, tmp_path)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("saved_then_initial", [False, True])
@pytest.mark.parametrize("alias", ["rename", "reshape"])
def test_saved_acc_selection_preserves_descriptor_sources(
    planner, saved_then_initial, alias, ascend_backend, tmp_path
):
    """A saved Acc descriptor can still observe either input to the carry."""
    alias_statement = (
        "saved_alias = pl.tile.reshape(saved, [16, 64])" if alias == "reshape" else "saved_alias = saved"
    )
    program = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, lhs: pl.Tensor[[32, 64], pl.BF16], rhs: pl.Tensor[[64, 64], pl.BF16],
               saved_flag: pl.Scalar[pl.INT32], update_flag: pl.Scalar[pl.INT32],
               count: pl.Scalar[pl.INDEX],
               saved_out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
               out: pl.Out[pl.Tensor[[16, 64], pl.FP32]]) -> pl.Tensor[[16, 64], pl.FP32]:
        lhs_mat = pl.tile.load(lhs, [0, 0], [16, 64], target_memory=pl.Mem.Mat)
        other_lhs_mat = pl.tile.load(lhs, [16, 0], [16, 64], target_memory=pl.Mem.Mat)
        rhs_mat = pl.tile.load(rhs, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
        left = pl.tile.extract(lhs_mat, 0, 0, [16, 64], target_memory=pl.Mem.Left)
        other_left = pl.tile.extract(other_lhs_mat, 0, 0, [16, 64], target_memory=pl.Mem.Left)
        right = pl.tile.extract(rhs_mat, 0, 0, [64, 64], target_memory=pl.Mem.Right)
        initial = pl.tile.matmul(left, right)
        other = pl.tile.matmul(other_left, right)
        if saved_flag > 0:
            saved = pl.yield_({"initial" if saved_then_initial else "other"})
        else:
            saved = pl.yield_({"other" if saved_then_initial else "initial"})
        {alias_statement}
        for i, (carried,) in pl.range(count, init_values=(initial,)):
            if update_flag > 0:
                fresh = pl.tile.matmul(other_left, right)
                selected = pl.yield_(fresh)
            else:
                accumulated = pl.tile.matmul_acc(carried, left, right)
                selected = pl.yield_(accumulated)
            result = pl.yield_(selected)
        pl.store(saved_alias, [0, 0], saved_out)
        return pl.store(result, [0, 0], out)
""")
    lowered, text = _lower(program, planner)
    _assert_descriptor_carries(lowered, 1)
    _assert_saved_inputs_have_independent_updates(lowered, {"initial", "other"})
    assert not any("pto.tmov" in line and line.count("loc=acc") == 2 for line in text.splitlines())
    _compile_native_loop(text, planner, tmp_path)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("selected_initial", [False, True])
def test_native_while_carries_tile_descriptors_without_payload_copies(
    planner, selected_initial, ascend_backend, tmp_path
):
    """Descriptor swaps preserve old inputs and carry the actual selected address."""
    initial = (
        """if flag > 0:
            seed = pl.yield_(a)
        else:
            seed = pl.yield_(b)"""
        if selected_initial
        else "seed = a"
    )
    program = _program(
        f"""
        a = pl.load(x, [0, 0], [4, 8])
        b = pl.load(x, [4, 0], [4, 8])
        {initial}
        for i, left, right in pl.while_(init_values=(0, seed, b)):
            pl.cond(i < count)
            final_i, final_left, final_right = pl.yield_(i + 1, right, left)
        pl.store(final_left, [0, 0], out)
        pl.store(final_right, [4, 0], out)
        return pl.store(a, [8, 0], out)
""",
        output_shape="[12, 8]",
    )
    lowered, text = _lower(program, planner)
    header = next(line for line in text.splitlines() if "scf.while" in line)
    assert header.count("!pto.tile_buf") == 4, text
    assert "tile.move(" not in lowered.as_python()
    assert "pto.tmov" not in text
    _compile_native_loop(text, planner, tmp_path)


@pytest.mark.parametrize("planner", _PLANNERS)
def test_while_backedge_storage_survives_temporaries_before_its_definition(planner, ascend_backend):
    """A lexical last-use interval misses the value retained from the preceding iteration."""
    program = _program(
        """
        initial = pl.load(x, [0, 0], [4, 8])
        for i, current in pl.while_(init_values=(0, initial)):
            pl.cond(i < count)
            scratch = pl.tile.full([4, 8], pl.FP32, 1.0)
            pl.store(scratch, [0, 0], out)
            updated = pl.tile.recip(current)
            final_i, result = pl.yield_(i + 1, updated)
        return pl.store(result, [4, 0], out)
""",
        input_shape="[4, 8]",
        output_shape="[8, 8]",
    )
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=False):
        lowered = PassManager(OptimizationStrategy.Default).run_passes(program)
    tiles = {
        op.var.name_hint.split("__")[0]: op.var.type.memref
        for op in _Nodes(lowered).assignments
        if isinstance(op.var.type, ir.TileType)
    }
    scratch, updated = tiles["scratch"], tiles["updated"]
    assert scratch is not None and updated is not None
    assert not ir.MemRef.may_alias(scratch, updated), lowered.as_python()


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("loop", ["for", "while"])
@pytest.mark.parametrize("alias", ["direct", "view", "selection"])
@pytest.mark.parametrize("read_after", [False, True])
def test_fresh_backedge_preserves_live_payload_versions(
    planner, loop, alias, read_after, ascend_backend, pass_verification_instruments, tmp_path
):
    """A later read needs the old payload even when the producer allows in-place input."""
    old = {
        "direct": "old = current",
        "view": "old = pl.tile.reshape(current, [4, 8])",
        "selection": "if flag > 1:\n    old = pl.yield_(current)\nelse:\n    old = pl.yield_(b)",
    }[alias]
    read = "combined = pl.add(old, fresh)" if read_after else "combined = pl.add(old, old)"
    fresh = "fresh = pl.tile.adds(current, 1.0)"
    operations = (
        [fresh, read, "pl.store(combined, [0, 0], out)"]
        if read_after
        else [read, "pl.store(combined, [0, 0], out)", fresh]
    )
    header = (
        "for i, (current,) in pl.range(count, init_values=(seed,)):"
        if loop == "for"
        else ("for i, current in pl.while_(init_values=(0, seed)):\n    pl.cond(i < count)")
    )
    backedge = "result = pl.yield_(fresh)" if loop == "for" else "final_i, result = pl.yield_(i + 1, fresh)"
    body = header + "\n" + indent("\n".join([old, *operations, backedge]), "    ")
    program = _program(f"""
        a = pl.load(x, [0, 0], [4, 8])
        b = pl.load(x, [4, 0], [4, 8])
        if flag > 0:
            seed = pl.yield_(a)
        else:
            seed = pl.yield_(b)
{indent(body, "        ")}
        return out
""")
    lowered, text = _lower(program, planner, pass_verification_instruments)
    _assert_descriptor_carries(lowered, 1)
    retained = [a.var for a in _Nodes(lowered).assignments if a.var.name_hint.startswith("recurrent_value_")]
    assert len(retained) == int(read_after)
    assert text.count("pto.tmov") == 2 * int(read_after)
    if read_after:
        produced = next(a.var for a in _Nodes(lowered).assignments if a.var.name_hint.startswith("fresh__"))
        produced_type, retained_type = produced.type, retained[0].type
        assert isinstance(produced_type, ir.TileType) and isinstance(retained_type, ir.TileType)
        assert produced_type.memref is not None and retained_type.memref is not None
        assert not ir.MemRef.may_alias(produced_type.memref, retained_type.memref)
    _compile_native_loop(text, planner, tmp_path)


@pytest.mark.parametrize("loop", ["for", "while"])
def test_loop_if_branch_reuse_preserves_previous_iteration_payload(
    loop, ascend_backend, pass_verification_instruments
):
    """Lexically exclusive producers can coexist with each other's preceding iteration."""
    header = (
        "for i, (current,) in pl.range(count, init_values=(seed,)):"
        if loop == "for"
        else "for i, current in pl.while_(init_values=(0, seed)):\n    pl.cond(i < count)"
    )
    backedge = (
        "result = pl.yield_(selected)" if loop == "for" else "final_i, result = pl.yield_(i + 1, selected)"
    )
    body = (
        """if i % 2 == 0:
    a = pl.load(x, [4, 0], [4, 8], target_memory=pl.Mem.Vec)
    selected = pl.yield_(a)
else:
    c = pl.load(x, [8, 0], [4, 8], target_memory=pl.Mem.Vec)
    selected = pl.yield_(c)
pl.store(current, [0, 0], out)
"""
        + backedge
    )
    loop_body = header + "\n" + indent(body, "    ")
    program = _program(
        f"""
        seed = pl.load(x, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
{indent(loop_body, "        ")}
        return pl.store(result, [4, 0], out)
""",
        input_shape="[12, 8]",
        output_shape="[8, 8]",
    )
    lowered, _ = _lower(program, passes.MemoryPlanner.PYPTO, pass_verification_instruments)
    _assert_dynamic_if_results(lowered)
    _assert_descriptor_carries(lowered, 1)
    nodes = _Nodes(lowered)
    assert len(nodes.loops) == len(nodes.branches) == 1
    loop_stmt, branch = nodes.loops[0], nodes.branches[0]
    definitions = {statement.var.unique_id: statement for statement in nodes.assignments}
    tiles = {
        statement.var.name_hint.split("__", 1)[0]: statement.var
        for statement in nodes.assignments
        if isinstance(statement.var.type, ir.TileType)
    }

    def window(value):
        tile = value.type
        assert isinstance(tile, ir.TileType) and tile.memref is not None
        offset = tile.memref.byte_offset_
        assert isinstance(offset, ir.ConstInt)
        return offset.value, offset.value + tile.memref.size_

    def disjoint(left, right):
        a, b = window(left), window(right)
        return a[1] <= b[0] or b[1] <= a[0]

    # Iteration zero produces A before observing seed; the next iteration
    # produces C before observing A. A/C are not dynamically exclusive unless
    # the preceding payload has already moved to independent retained storage.
    assert disjoint(tiles["seed"], tiles["a"]), lowered.as_python()
    index, current = next(
        (index, argument)
        for index, argument in enumerate(loop_stmt.iter_args)
        if isinstance(argument.type, ir.TileType)
    )
    statements = loop_stmt.body.stmts if isinstance(loop_stmt.body, ir.SeqStmts) else [loop_stmt.body]
    positions = {
        statement.var.unique_id: position
        for position, statement in enumerate(statements)
        if isinstance(statement, ir.AssignStmt)
    }
    branch_position = next(i for i, statement in enumerate(statements) if isinstance(statement, ir.IfStmt))
    calls = [
        (i, statement.value if isinstance(statement, ir.AssignStmt) else statement.expr)
        for i, statement in enumerate(statements)
        if isinstance(statement, (ir.AssignStmt, ir.EvalStmt))
    ]
    old_read_position, old_read = next(
        (i, call)
        for i, call in calls
        if isinstance(call, ir.Call) and call.op.name == ir.get_op("tile.store").name
    )
    assert old_read_position > branch_position

    def moves_from(value: ir.Expr, source: ir.Var | ir.IterArg):
        moves = []
        while True:
            assert isinstance(value, (ir.Var, ir.IterArg))
            if value.unique_id == source.unique_id:
                break
            definition = definitions[value.unique_id]
            assert isinstance(definition.value, ir.Call)
            assert definition.value.op.name == ir.get_op("tile.move").name
            moves.append(definition)
            value = definition.value.args[0]
        return moves

    observed = old_read.args[0]
    old_snapshots = moves_from(observed, current)
    if old_snapshots:
        assert all(positions[move.var.unique_id] < branch_position for move in old_snapshots)
        assert all(disjoint(observed, tiles[name]) for name in ("a", "c")), lowered.as_python()
    if not disjoint(tiles["a"], tiles["c"]) and not old_snapshots:
        # Accept legal producer reuse only when the descriptor carried into
        # the next iteration points at a protected payload, not at A/C's bytes.
        yield_stmt = ir.get_last_yield_stmt(loop_stmt.body)
        assert yield_stmt is not None
        retained = yield_stmt.value[index]
        transfers = moves_from(retained, branch.return_vars[0])
        assert transfers, "overlapping producers need a retained version of the old carry"
        assert all(positions[move.var.unique_id] > old_read_position for move in transfers)
        assert all(disjoint(retained, tiles[name]) for name in ("a", "c")), lowered.as_python()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
