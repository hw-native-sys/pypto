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

import pypto.language as pl
import pytest
from pypto import InternalError, codegen, ir, passes
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _get_ptoas_flags
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

_PLANNERS = [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]


@pl.program
class PinnedInputs:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[8, 8], pl.FP32],
        flag: pl.Scalar[pl.INT32],
        out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
    ) -> pl.Tensor[[4, 8], pl.FP32]:
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


@pl.program
class MixedResults:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[8, 8], pl.FP32],
        flag: pl.Scalar[pl.INT32],
        out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
    ) -> pl.Tensor[[8, 8], pl.FP32]:
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


@pl.program
class DynamicValid:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[8, 8], pl.FP32],
        flag: pl.Scalar[pl.INT32],
        rows_a: pl.Scalar[pl.INDEX],
        out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
    ) -> pl.Tensor[[4, 8], pl.FP32]:
        a = pl.tile.load(x, [0, 0], [4, 8], [rows_a, 8], target_memory=pl.Mem.Vec)
        b = pl.tile.load(x, [4, 0], [4, 8], [rows_a, 8], target_memory=pl.Mem.Vec)
        if flag > 0:
            selected = pl.yield_(a)
        else:
            selected = pl.yield_(b)
        return pl.store(selected, [0, 0], out)


@pl.program
class SelectedView:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[8, 8], pl.FP32],
        flag: pl.Scalar[pl.INT32],
        out: pl.Out[pl.Tensor[[2, 16], pl.FP32]],
    ) -> pl.Tensor[[2, 16], pl.FP32]:
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
    with passes.PassContext(
        pass_verification_instruments, memory_planner=planner, enable_buffer_ir=enable_buffer_ir
    ):
        lowered = PassManager(OptimizationStrategy.Default).run_passes(program)
        text = codegen.PTOCodegen().generate(
            lowered, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS, emit_source_loc=False
        )
    assert re.search(r"scf\.if .* -> \(.*!pto\.tile_buf", text), text
    assert "pto.tmov" not in text
    assert "slots=2" not in lowered.as_python()
    source = tmp_path / "kernel.pto"
    source.write_text(text)
    compiled = subprocess.run(
        [binary, str(source), "-o", str(tmp_path / "kernel.cpp"), *_get_ptoas_flags(planner)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr


@pl.program
class CandidateLifetimes:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[12, 8], pl.FP32],
        flag: pl.Scalar[pl.INT32],
        out: pl.Out[pl.Tensor[[16, 8], pl.FP32]],
    ) -> pl.Tensor[[16, 8], pl.FP32]:
        a = pl.load(x, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
        b = pl.load(x, [4, 0], [4, 8], target_memory=pl.Mem.Vec)
        if flag > 0:
            selected = pl.yield_(a)
        else:
            selected = pl.yield_(b)
        alias = selected
        view = pl.tile.reshape(alias, [4, 8])
        scratch = pl.load(x, [8, 0], [4, 8], target_memory=pl.Mem.Vec)
        pl.store(scratch, [0, 0], out)
        pl.store(view, [4, 0], out)
        # Both candidates die at the store through the selected view. Later
        # allocations should reuse them without reserving a special pool.
        later_a = pl.load(x, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
        later_b = pl.load(x, [4, 0], [4, 8], target_memory=pl.Mem.Vec)
        pl.store(later_a, [8, 0], out)
        return pl.store(later_b, [12, 0], out)


def _allocation_windows(program: ir.Program) -> dict[str, tuple[int, int]]:
    windows: dict[str, tuple[int, int]] = {}

    class Collector(ir.IRVisitor):
        def visit_assign_stmt(self, op: ir.AssignStmt) -> None:
            tile = op.var.type
            if isinstance(tile, ir.TileType) and tile.memref is not None:
                offset = tile.memref.byte_offset_
                assert isinstance(offset, ir.ConstInt)
                windows[op.var.name_hint] = (offset.value, offset.value + tile.memref.size_)
            super().visit_assign_stmt(op)

    for function in program.functions.values():
        Collector().visit_stmt(function.body)
    return windows


@pytest.mark.parametrize("planner", _PLANNERS[:2])
def test_candidates_live_through_selected_alias_and_are_reused(
    planner, ascend_backend, pass_verification_instruments
):
    """A clobber between selection and use cannot overwrite either possible input."""
    with passes.PassContext(pass_verification_instruments, memory_planner=planner, enable_buffer_ir=False):
        program = passes.materialize_semantic_aliases()(passes.init_mem_ref()(CandidateLifetimes))
        if planner == passes.MemoryPlanner.PYPTO:
            program = passes.memory_reuse()(program)
        placed = passes.allocate_memory_addr()(program)

    windows = _allocation_windows(placed)

    def overlaps(left: str, right: str) -> bool:
        a, b = windows[left], windows[right]
        return a[0] < b[1] and b[0] < a[1]

    assert not overlaps("a", "b")
    assert not overlaps("scratch", "a")
    assert not overlaps("scratch", "b")
    assert any(overlaps(later, old) for later in ("later_a", "later_b") for old in ("a", "b"))


@pytest.mark.parametrize("planner", _PLANNERS[:2])
def test_nested_selection_does_not_make_colive_inputs_exclusive(
    planner, ascend_backend, pass_verification_instruments
):
    """Two candidates born in one outer arm still coexist inside that arm."""

    @pl.program
    class Nested:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[12, 8], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
        ) -> pl.Tensor[[4, 8], pl.FP32]:
            if flag > 0:
                a = pl.load(x, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
                b = pl.load(x, [4, 0], [4, 8], target_memory=pl.Mem.Vec)
                if flag > 1:
                    inner = pl.yield_(a)
                else:
                    inner = pl.yield_(b)
                # A and B have both been initialized before this read. The
                # enclosing branch must never label them mutually exclusive.
                combined = pl.tile.add(a, b)
                pl.store(combined, [0, 0], out)
                selected = pl.yield_(inner)
            else:
                c = pl.load(x, [8, 0], [4, 8], target_memory=pl.Mem.Vec)
                selected = pl.yield_(c)
            return pl.store(selected, [0, 0], out)

    with passes.PassContext(pass_verification_instruments, memory_planner=planner, enable_buffer_ir=False):
        program = passes.materialize_semantic_aliases()(passes.init_mem_ref()(Nested))
        if planner == passes.MemoryPlanner.PYPTO:
            program = passes.memory_reuse()(program)
        placed = passes.allocate_memory_addr()(program)
    windows = _allocation_windows(placed)
    a, b = windows["a"], windows["b"]
    assert a[1] <= b[0] or b[1] <= a[0]


@pytest.mark.parametrize("planner", _PLANNERS[:2])
def test_non_inplace_output_is_disjoint_from_every_selected_root(
    planner, ascend_backend, pass_verification_instruments
):
    """A dying phi input still forbids output reuse at the read/write boundary."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[32, 16], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            a = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
            b = pl.load(x, [16, 0], [16, 16], target_memory=pl.Mem.Vec)
            if flag > 0:
                selected = pl.yield_(a)
            else:
                selected = pl.yield_(b)
            transposed = pl.tile.transpose(selected, 0, 1)
            return pl.store(transposed, [0, 0], out)

    with passes.PassContext(pass_verification_instruments, memory_planner=planner, enable_buffer_ir=False):
        program = passes.flatten_tile_nd_to_2d()(Before)
        program = passes.materialize_semantic_aliases()(passes.init_mem_ref()(program))
        if planner == passes.MemoryPlanner.PYPTO:
            program = passes.memory_reuse()(program)
        placed = passes.allocate_memory_addr()(program)

    windows = _allocation_windows(placed)
    destination = windows["transposed"]
    # Both candidates' last use is the transpose itself, so ordinary lifetime
    # overlap alone permits same-point reuse. The op's non-inplace contract
    # must expand the phi's input roots and exclude both, including else's B.
    for candidate in ("a", "b"):
        source = windows[candidate]
        assert destination[1] <= source[0] or source[1] <= destination[0], (candidate, windows)


@pytest.mark.parametrize("planner", _PLANNERS[:2])
def test_selected_written_workspace_cannot_inherit_an_earlier_read_buffer(
    planner, ascend_backend, pass_verification_instruments
):
    """Both runtime scratch choices retain the intrinsic's hidden-write hazard."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[32, 16], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            dead = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
            src = pl.load(x, [16, 0], [16, 16], target_memory=pl.Mem.Vec)
            mask = pl.tile.cmps(dead, 0.0, cmp_type=4)
            scratch_a = pl.tile.create([16, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            scratch_b = pl.tile.create([16, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            if flag > 0:
                selected = pl.yield_(scratch_a)
            else:
                selected = pl.yield_(scratch_b)
            result = pl.tile.sels(mask, src, selected, -1.0)
            return pl.store(result, [0, 0], out)

    with passes.PassContext(pass_verification_instruments, memory_planner=planner, enable_buffer_ir=False):
        program = passes.materialize_semantic_aliases()(passes.init_mem_ref()(Before))
        if planner == passes.MemoryPlanner.PYPTO:
            program = passes.memory_reuse()(program)
        placed = passes.allocate_memory_addr()(program)

    windows = _allocation_windows(placed)
    previous = windows["dead"]
    # CMPS retires dead logically before either scratch is created. TSELS writes
    # its scalar into the chosen scratch on an untracked pipe, so neither choice
    # may inherit bytes the preceding vector reader can still be using.
    for candidate in ("scratch_a", "scratch_b"):
        workspace = windows[candidate]
        assert workspace[1] <= previous[0] or previous[1] <= workspace[0], (candidate, windows)


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
    # Matching handles require no descriptor selection, and retain the local
    # provenance PTOAS needs for subsequent mutable-validity operations.
    assert not re.search(r"scf.if .* -> .*loc=acc", text), text
    source = tmp_path / "shared_accumulator.pto"
    source.write_text(text)
    compiled = subprocess.run(
        [
            binary,
            str(source),
            "-o",
            str(tmp_path / "shared_accumulator.cpp"),
            *_get_ptoas_flags(passes.MemoryPlanner.PTOAS),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr


@pytest.mark.parametrize("loop", ["for", "while"])
@pytest.mark.parametrize("boundary", ["initializer", "backedge"])
@pytest.mark.parametrize("planner", _PLANNERS)
def test_updating_loop_rejects_unresolved_tile_selection(loop, boundary, planner, ascend_backend):
    """Both loop kinds must diagnose storage choices their fixed carry cannot hold."""
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

    program = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[12, 8], pl.FP32], flag: pl.Scalar[pl.INT32],
               count: pl.Scalar[pl.INDEX],
               out: pl.Out[pl.Tensor[[12, 8], pl.FP32]]) -> pl.Tensor[[12, 8], pl.FP32]:
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
""")
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=False):
        with pytest.raises(ValueError, match="Tile selected by an if cannot cross"):
            lowered = PassManager(OptimizationStrategy.Default).run_passes(program)
            codegen.PTOCodegen().generate(lowered, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("fresh_then", [False, True])
def test_conditional_carry_swap_is_diagnosed_before_writeback_moves(planner, fresh_then, ascend_backend):
    """Fresh merge outputs can hide the opposite arm's carry-copy cycle."""
    fresh = """fresh_left = pl.tile.mrgsort(left, block_len=64)
fresh_right = pl.tile.mrgsort(right, block_len=64)
next_left, next_right = pl.yield_(fresh_left, fresh_right)"""
    swap = "next_left, next_right = pl.yield_(right, left)"

    def indent(text):
        return "\n".join(" " * 16 + line for line in text.splitlines())

    program = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[2, 4096], pl.FP32], flag: pl.Scalar[pl.INT32],
               count: pl.Scalar[pl.INDEX],
               out: pl.Out[pl.Tensor[[2, 4096], pl.FP32]]) -> pl.Tensor[[2, 4096], pl.FP32]:
        a = pl.load(x, [0, 0], [1, 4096])
        b = pl.load(x, [1, 0], [1, 4096])
        for i, (left, right) in pl.range(count, init_values=(a, b)):
            if flag > 0:
{indent(fresh if fresh_then else swap)}
            else:
{indent(swap if fresh_then else fresh)}
            result_left, result_right = pl.yield_(next_left, next_right)
        pl.store(result_left, [0, 0], out)
        return pl.store(result_right, [1, 0], out)
""")
    # mrgsort is not inplace-safe, so the fresh arm forces separate storage.
    # Its representative addresses used to conceal the sibling arm's swap when
    # two generated writebacks were ordered, then to hide it from codegen too.
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=False):
        with pytest.raises(ValueError, match="conditional Tile selection cannot feed multiple loop-carry"):
            lowered = PassManager(OptimizationStrategy.Default).run_passes(program)
            codegen.PTOCodegen().generate(lowered, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)


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
    program = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[{input_shape}, pl.FP32], flag: pl.Scalar[pl.INT32],
               count: pl.Scalar[pl.INDEX],
               out: pl.Out[pl.Tensor[{output_shape}, pl.FP32]]) -> pl.Tensor[{output_shape}, pl.FP32]:
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
""")
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=False):
        lowered = PassManager(OptimizationStrategy.Default).run_passes(program)
        text = codegen.PTOCodegen().generate(lowered, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    _assert_selected_carry_sources_are_snapshotted(
        lowered, expected_copies=2, addressed=planner != passes.MemoryPlanner.PTOAS
    )
    assert text.count("pto.tmov") == 4, text
    assert text.count("scf.if") >= 1 + int(nested), text


def _assert_selected_carry_sources_are_snapshotted(program, expected_copies, addressed):
    """Selected handles only write fresh scratch; carry writes read that scratch."""
    checked = 0

    class Checker(ir.IRVisitor):
        allocations = set()

        def visit_assign_stmt(self, op):
            if isinstance(op.value, ir.Call) and op.value.op.name == ir.get_op("tile.alloc").name:
                self.allocations.add(op.var.unique_id)
            super().visit_assign_stmt(op)

        def visit_for_stmt(self, op):
            nonlocal checked
            carries: list[ir.MemRef] = []
            for arg in op.iter_args:
                initial_type = arg.initValue.type
                assert isinstance(initial_type, ir.TileType)
                carry = initial_type.memref
                assert carry is not None
                carries.append(carry)
            body = op.body.stmts if isinstance(op.body, ir.SeqStmts) else [op.body]
            moves = [
                stmt
                for stmt in body
                if isinstance(stmt, ir.AssignStmt)
                and isinstance(stmt.value, ir.Call)
                and stmt.value.op.name == ir.get_op("tile.move").name
            ]
            if not moves:
                return super().visit_for_stmt(op)
            snapshots = moves[:expected_copies]
            writebacks = moves[expected_copies:]
            assert len(snapshots) == len(writebacks) == expected_copies
            scratch_values = {stmt.var.unique_id for stmt in snapshots}
            for snapshot in snapshots:
                snapshot_type = snapshot.var.type
                assert isinstance(snapshot_type, ir.TileType)
                scratch = snapshot_type.memref
                assert scratch is not None
                assert "_carry_spill_" in scratch.base_.name_hint
                assert scratch.base_.unique_id in self.allocations, "scratch must be declared before the loop"
                assert all(not ir.MemRef.may_alias(scratch, carry) for carry in carries)
                if addressed:
                    scratch_offset = scratch.byte_offset_
                    assert isinstance(scratch_offset, ir.ConstInt)
                    start = scratch_offset.value
                    for carry in carries:
                        carry_offset = carry.byte_offset_
                        assert isinstance(carry_offset, ir.ConstInt)
                        carry_start = carry_offset.value
                        assert start + scratch.size_ <= carry_start or carry_start + carry.size_ <= start
            for writeback in writebacks:
                call = writeback.value
                assert isinstance(call, ir.Call)
                source = call.args[0]
                assert isinstance(source, ir.Var)
                source_type = source.type
                destination_type = writeback.var.type
                assert isinstance(source_type, ir.TileType)
                assert isinstance(destination_type, ir.TileType)
                source_memref = source_type.memref
                destination = destination_type.memref
                assert source_memref is not None and destination is not None
                assert source.unique_id in scratch_values, "a selected handle could self-copy onto its carry"
                assert any(ir.MemRef.may_alias(destination, carry) for carry in carries)
                assert not ir.MemRef.may_alias(source_memref, destination)
            checked += 1
            super().visit_for_stmt(op)

    Checker().visit_program(program)
    assert checked == 1


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("fresh_then", [False, True])
def test_single_conditional_carry_copy_snapshots_unchanged_branch(planner, fresh_then, ascend_backend):
    """Even one selected writeback must avoid copying the unchanged carry onto itself."""

    fresh = """fresh = pl.tile.mrgsort(carried, block_len=64)
selected = pl.yield_(fresh)"""
    unchanged = "selected = pl.yield_(carried)"

    def indent(text):
        return "\n".join(" " * 16 + line for line in text.splitlines())

    program = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[1, 4096], pl.FP32], flag: pl.Scalar[pl.INT32],
               count: pl.Scalar[pl.INDEX],
               out: pl.Out[pl.Tensor[[1, 4096], pl.FP32]]) -> pl.Tensor[[1, 4096], pl.FP32]:
        initial = pl.load(x, [0, 0], [1, 4096])
        for i, (carried,) in pl.range(count, init_values=(initial,)):
            if flag > 0:
{indent(fresh if fresh_then else unchanged)}
            else:
{indent(unchanged if fresh_then else fresh)}
            result = pl.yield_(selected)
        return pl.store(result, [0, 0], out)
""")

    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=False):
        lowered = PassManager(OptimizationStrategy.Default).run_passes(program)
        text = codegen.PTOCodegen().generate(lowered, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    _assert_selected_carry_sources_are_snapshotted(
        lowered, expected_copies=1, addressed=planner != passes.MemoryPlanner.PTOAS
    )
    assert text.count("pto.tmov") == 2, text


def test_zero_trip_selected_same_hint_does_not_insert_self_copy(ascend_backend):
    """Standalone storage passes preserve the dead body without adding an illegal TMOV."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[1, 4096], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[1, 4096], pl.FP32]],
        ) -> pl.Tensor[[1, 4096], pl.FP32]:
            initial = pl.load(x, [0, 0], [1, 4096], target_memory=pl.Mem.Vec)
            for _i, (carried,) in pl.range(0, init_values=(initial,)):
                if flag > 0:
                    selected = pl.yield_(carried)
                else:
                    fresh = pl.tile.mrgsort(carried, block_len=64)
                    selected = pl.yield_(fresh)
                result = pl.yield_(selected)
            return pl.store(result, [0, 0], out)

    # The full pipeline simplifies zero-trip loops away before storage repair.
    # Invoke these passes directly so elimination cannot hide a dead self-copy.
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO, enable_buffer_ir=False):
        lowered = passes.memory_reuse()(passes.materialize_semantic_aliases()(passes.init_mem_ref()(Before)))

    loops, branches, calls, spills = [], [], [], []

    class Collector(ir.IRVisitor):
        def visit_for_stmt(self, op):
            loops.append(op)
            super().visit_for_stmt(op)

        def visit_if_stmt(self, op):
            branches.append(op)
            super().visit_if_stmt(op)

        def visit_call(self, op):
            calls.append(op.op.name)
            super().visit_call(op)

        def visit_assign_stmt(self, op):
            if "_carry_spill_" in op.var.name_hint:
                spills.append(op)
            super().visit_assign_stmt(op)

    Collector().visit_program(lowered)
    assert len(loops) == len(branches) == 1, "the dead loop body must remain in this standalone regression"
    assert ir.get_op("tile.move").name not in calls
    assert not spills


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
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=False):
        lowered = PassManager(OptimizationStrategy.Default).run_passes(program)
        text = codegen.PTOCodegen().generate(lowered, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    # Integer assemble uses a data TMOV into each destination window. There
    # must be no additional loop-carry moves or snapshots around those writes.
    assert "tile.move(" not in lowered.as_python()
    assert "_carry_spill_" not in lowered.as_python()
    assert text.count("pto.tmov") == 3, text
    assert text.count("scf.if") >= 2, text
    assert "scf.for" in text, text


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("saved_then_a", [False, True])
@pytest.mark.parametrize("alias", ["rename", "reshape"])
def test_selected_observer_of_loop_initializer_cannot_be_clobbered(
    planner, saved_then_a, alias, ascend_backend
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
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=False):
        with pytest.raises(ValueError, match="Tile selected by an if cannot cross"):
            lowered = PassManager(OptimizationStrategy.Default).run_passes(program)
            codegen.PTOCodegen().generate(lowered, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)


@pytest.mark.parametrize("planner", _PLANNERS)
@pytest.mark.parametrize("saved_then_initial", [False, True])
@pytest.mark.parametrize("alias", ["rename", "reshape"])
def test_saved_acc_selection_blocks_destructive_carry_coalescing(
    planner, saved_then_initial, alias, ascend_backend
):
    """A saved Acc descriptor can still observe the carry's non-representative input."""
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
    if planner == passes.MemoryPlanner.PYPTO:
        # PYPTO postpones fixed-carry reconciliation until MemoryReuse. Check
        # the coalescer's decision before that required Acc transfer is rejected.
        with passes.PassContext([], memory_planner=planner, enable_buffer_ir=False):
            materialized = passes.materialize_semantic_aliases()(passes.init_mem_ref()(program))
        sources: dict[str, ir.MemRef] = {}

        class CollectSources(ir.IRVisitor):
            def visit_assign_stmt(self, op):
                if op.var.name_hint in {"initial", "fresh"}:
                    tile_type = op.var.type
                    assert isinstance(tile_type, ir.TileType)
                    assert tile_type.memref is not None
                    sources[op.var.name_hint] = tile_type.memref
                super().visit_assign_stmt(op)

        CollectSources().visit_program(materialized)
        assert set(sources) == {"initial", "fresh"}
        assert not ir.MemRef.may_alias(sources["initial"], sources["fresh"]), materialized.as_python()

    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=False):
        with pytest.raises(InternalError, match="cannot reconcile divergent L0C accumulator buffers"):
            lowered = PassManager(OptimizationStrategy.Default).run_passes(program)
            codegen.PTOCodegen().generate(lowered, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
