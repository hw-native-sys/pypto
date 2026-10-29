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
from pypto import codegen, ir, passes
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
