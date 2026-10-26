# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression: if/else-yield tile phi under memory_planner=PTOAS (issue #1956).

MaterializeSemanticAliases settles each result's storage before PTOAS skips
opportunistic MemoryReuse. Codegen must preserve that storage through differently
typed views and control-flow scopes: branch writes and post-if reads must reach
the same physical allocation, and each SSA handle must retain its own type and
dominate every use.
"""

# pyright: reportUndefinedVariable=false

import re
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import ir as _ir
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes


@pl.program
class IfPhiProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[64, 64], pl.FP32],
        flag: pl.Scalar[pl.INT32],
        out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        a: pl.Tile[[64, 64], pl.FP32] = pl.load(x, [0, 0], [64, 64])
        if flag == 0:
            r: pl.Tile[[64, 64], pl.FP32] = pl.add(a, 1.0)
        else:
            r: pl.Tile[[64, 64], pl.FP32] = pl.mul(a, 2.0)
        return pl.store(r, [0, 0], out)


@pl.program
class IfPhiPassThroughProgram:
    """One branch yields a fresh producer; the other yields the outer tile `a`
    unchanged (a pass-through). A post-if observation preserves the old value of
    `a`, so the phi needs independent storage and the pass-through needs a copy."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[64, 64], pl.FP32],
        flag: pl.Scalar[pl.INT32],
        out: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
    ) -> pl.Tensor[[128, 64], pl.FP32]:
        a: pl.Tile[[64, 64], pl.FP32] = pl.load(x, [0, 0], [64, 64])
        if flag == 0:
            r: pl.Tile[[64, 64], pl.FP32] = pl.add(a, 1.0)
        else:
            r: pl.Tile[[64, 64], pl.FP32] = a
        saved = pl.store(r, [0, 0], out)
        return pl.store(a, [64, 0], saved)


@pl.program
class DynamicValidMatmulIfPhiProgram:
    """A matmul if-phi whose logical M extent is computed in the function body.

    PTO tile-phi handles must dominate both branches, so their ``alloc_tile`` is
    emitted at the function head.  The body-local ``valid_rows`` SSA value does
    not dominate that declaration; codegen must allocate the physical box at the
    head and restore the logical valid shape immediately before the ``scf.if``.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        lhs: pl.Tensor[[16, 128], pl.BF16],
        rhs: pl.Tensor[[128, 32], pl.BF16],
        rows: pl.Scalar[pl.INDEX],
        flag: pl.Scalar[pl.INT32],
        out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
    ) -> pl.Tensor[[16, 32], pl.FP32]:
        valid_rows = pl.min(rows, 16)
        lhs_mat = pl.load(
            lhs,
            [0, 0],
            [16, 128],
            valid_shape=[valid_rows, 128],
            target_memory=pl.MemorySpace.Mat,
        )
        rhs_mat = pl.load(rhs, [0, 0], [128, 32], target_memory=pl.MemorySpace.Mat)
        lhs_left = pl.move(lhs_mat, target_memory=pl.MemorySpace.Left)
        rhs_right = pl.move(rhs_mat, target_memory=pl.MemorySpace.Right)
        if flag == 0:
            result = pl.matmul(lhs_left, rhs_right)
        else:
            result = pl.matmul(lhs_left, rhs_right)
        return pl.store(result, [0, 0], out)


def _ptoas_pto(program) -> str:
    reset_for_testing()
    set_backend_type(BackendType.Ascend910)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    return codegen.PTOCodegen().generate(_ir.Program([func], "kernel", optimized.span), emit_tile_addr=False)


def _planner_pto(program, planner: passes.MemoryPlanner) -> str:
    reset_for_testing()
    set_backend_type(BackendType.Ascend910)
    with passes.PassContext([], memory_planner=planner):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    return codegen.PTOCodegen().generate(
        _ir.Program([func], "kernel", optimized.span),
        emit_tile_addr=planner == passes.MemoryPlanner.PYPTO,
    )


def _sole_line(mlir: str, needle: str) -> str:
    lines = [ln for ln in mlir.splitlines() if needle in ln]
    assert len(lines) == 1, f"expected exactly one {needle!r} line, got {lines}:\n{mlir}"
    return lines[0]


def _first_group(pattern: str, text: str) -> str:
    m = re.search(pattern, text)
    assert m is not None, f"pattern {pattern!r} not found in:\n{text}"
    return m.group(1)


def test_if_phi_branches_write_one_head_declared_handle():
    mlir = _ptoas_pto(IfPhiProgram)
    # The tile the store reads is the phi buffer.
    store = next(ln for ln in mlir.splitlines() if "pto.tstore" in ln)
    phi = _first_group(r"pto\.tstore ins\((%[A-Za-z0-9_]+)", store)

    # It is declared exactly once (at the function head, dominating both branches).
    decls = [ln for ln in mlir.splitlines() if f"{phi} = pto.alloc_tile" in ln]
    assert len(decls) == 1, f"phi handle {phi} must be declared exactly once:\n{mlir}"

    # Both branch producers (tadds / tmuls) write that same phi handle directly —
    # a branch-local producer is re-pointed, so no copy is needed.
    producer_outs = [
        _first_group(r"outs\((%[A-Za-z0-9_]+)", ln)
        for ln in mlir.splitlines()
        if ("pto.tadds" in ln or "pto.tmuls" in ln) and "outs(" in ln
    ]
    assert len(producer_outs) == 2, f"expected two branch producers, got {producer_outs}:\n{mlir}"
    for out in producer_outs:
        assert out == phi, f"branch producer writes {out}, not the phi handle {phi}:\n{mlir}"
    assert "pto.tmov" not in mlir, f"branch-local phi must be copy-free (no tmov):\n{mlir}"


def test_if_phi_pass_through_branch_copies_into_phi_handle():
    """An independently observed old input forces a real pass-through fallback copy."""
    mlir = _ptoas_pto(IfPhiPassThroughProgram)
    stores = [ln for ln in mlir.splitlines() if "pto.tstore" in ln]
    assert len(stores) == 2, mlir
    phi = _first_group(r"pto\.tstore ins\((%[A-Za-z0-9_]+)", stores[0])
    old = _first_group(r"outs\((%[A-Za-z0-9_]+)", _sole_line(mlir, "pto.tload"))
    assert _first_group(r"pto\.tstore ins\((%[A-Za-z0-9_]+)", stores[1]) == old, mlir
    assert old != phi, f"the branch producer must not overwrite the independently observed input:\n{mlir}"

    # The producing branch writes the phi handle directly.
    prod = next(ln for ln in mlir.splitlines() if "pto.tadds" in ln)
    assert _first_group(r"outs\((%[A-Za-z0-9_]+)", prod) == phi, f"producer must write phi {phi}:\n{mlir}"

    # The pass-through branch (yields the outer tile unchanged) copies it into the
    # phi handle via tmov, so the post-if store never reads an uninitialised buffer.
    movs = [ln for ln in mlir.splitlines() if "pto.tmov" in ln and f"outs({phi}" in ln]
    assert len(movs) == 1, f"pass-through branch must tmov into the phi handle {phi}:\n{mlir}"
    assert _first_group(r"pto\.tmov ins\((%[A-Za-z0-9_]+)", movs[0]) == old, mlir


def test_if_phi_pass_through_reuses_an_unobserved_incoming_handle():
    """The same branch pattern needs no fallback copy when its old input is dead after the merge."""

    @pl.program
    class SafePassThrough:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[64, 64], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a = pl.load(x, [0, 0], [64, 64])
            if flag == 0:
                r = pl.add(a, 1.0)
            else:
                r = a
            return pl.store(r, [0, 0], out)

    mlir = _ptoas_pto(SafePassThrough)
    incoming = _first_group(r"outs\((%[A-Za-z0-9_]+)", _sole_line(mlir, "pto.tload"))
    stored = _first_group(r"pto\.tstore ins\((%[A-Za-z0-9_]+)", _sole_line(mlir, "pto.tstore"))
    updated = _first_group(r"outs\((%[A-Za-z0-9_]+)", _sole_line(mlir, "pto.tadds"))
    assert incoming == stored == updated, mlir
    assert "pto.tmov" not in mlir, mlir


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_dynamic_valid_matmul_if_phi_restores_shape_at_lexical_site(planner):
    """A body-local valid extent must not escape into the head alloc declaration."""
    mlir = _planner_pto(DynamicValidMatmulIfPhiProgram, planner)
    lines = mlir.splitlines()

    store = next(ln for ln in lines if "pto.tstore" in ln)
    phi = _first_group(r"pto\.tstore ins\((%[A-Za-z0-9_]+)", store)
    alloc_index, alloc = next((i, ln) for i, ln in enumerate(lines) if f"{phi} = pto.alloc_tile" in ln)
    set_index, set_shape = next(
        (i, ln) for i, ln in enumerate(lines) if ln.strip().startswith(f"pto.set_validshape {phi},")
    )
    if_index = next(i for i, ln in enumerate(lines) if "scf.if" in ln)

    # The head declaration uses the full 16x32 physical box, not the body-local
    # valid_rows SSA value.  The latter is restored after its defining scalar op
    # and before either branch writes the phi handle.
    assert "valid_row = %c16_index" in alloc, f"phi head alloc must use physical M=16:\n{mlir}"
    logical_row = _first_group(rf"pto\.set_validshape {re.escape(phi)}, (%[A-Za-z0-9_]+),", set_shape)
    defining_indices = [i for i, ln in enumerate(lines) if ln.lstrip().startswith(f"{logical_row} =")]
    assert len(defining_indices) == 1, f"expected one definition of {logical_row}:\n{mlir}"
    assert alloc_index < defining_indices[0] < set_index < if_index, (
        "the dynamic valid row must be defined before lexical shape restoration, "
        f"while the physical phi alloc remains head-declared:\n{mlir}"
    )


@pl.program
class LoopCarriedIfPhiProgram:
    """A `pl.range` accumulator whose per-iteration update is an if/else. The loop
    init, the iter_arg, the if-phi and the loop result all denote one buffer, so
    under PTOAS they must all collapse onto a single `tile_buf` handle."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[64, 64], pl.FP32],
        out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        acc: pl.Tile[[64, 64], pl.FP32] = pl.load(x, [0, 0], [64, 64])
        for i in pl.range(4):  # noqa: B007 - loop index drives the branch
            if i == 0:
                acc = pl.add(acc, 1.0)
            else:
                acc = pl.mul(acc, 2.0)
        return pl.store(acc, [0, 0], out)


def test_loop_carried_if_phi_shares_the_accumulator_handle():
    """The if-phi must reuse the accumulator's handle, not declare a second buffer.

    Under PTOAS no `addr` is baked, so two addr-less `pto.alloc_tile` are two
    independent buffers to ptoas PlanMemory. Head-declaring a fresh handle for the
    phi therefore stranded the accumulator: the branch producers wrote the phi
    buffer while the loop-carried read and the post-loop store used the accumulator
    handle, which no branch ever wrote — a silent miscompile (issue: the loop-carried
    counterpart of the `scf.if` phi fix #1956/#1985).
    """
    mlir = _ptoas_pto(LoopCarriedIfPhiProgram)

    store = next(ln for ln in mlir.splitlines() if "pto.tstore" in ln)
    acc = _first_group(r"pto\.tstore ins\((%[A-Za-z0-9_]+)", store)

    # Exactly one tile_buf is declared for the accumulator, and nothing else.
    decls = [ln for ln in mlir.splitlines() if "= pto.alloc_tile" in ln]
    assert len(decls) == 1, f"expected a single alloc_tile, got {len(decls)}:\n{mlir}"
    assert f"{acc} = pto.alloc_tile" in decls[0], f"the sole alloc_tile must be {acc}:\n{mlir}"

    # Both branch producers write that handle, and both read it back (the carry).
    producers = [ln for ln in mlir.splitlines() if "pto.tadds" in ln or "pto.tmuls" in ln]
    assert len(producers) == 2, f"expected two branch producers, got {producers}:\n{mlir}"
    for prod in producers:
        assert _first_group(r"outs\((%[A-Za-z0-9_]+)", prod) == acc, f"producer must write {acc}:\n{prod}"
        assert _first_group(r"ins\((%[A-Za-z0-9_]+)", prod) == acc, f"producer must read {acc}:\n{prod}"

    assert "pto.tmov" not in mlir, f"a shared handle needs no copy:\n{mlir}"


@pytest.mark.parametrize("enable_buffer_ir", [False, True])
def test_loop_carried_if_phi_keeps_baked_addr_aliasing_under_pypto(enable_buffer_ir):
    """The carry shares storage in both Tile and Buffer IR."""
    reset_for_testing()
    set_backend_type(BackendType.Ascend910)
    with passes.PassContext([], enable_buffer_ir=enable_buffer_ir):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(LoopCarriedIfPhiProgram)
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    mlir = codegen.PTOCodegen().generate(_ir.Program([func], "kernel", optimized.span))

    decls = [ln for ln in mlir.splitlines() if "= pto.alloc_tile" in ln]
    if enable_buffer_ir:
        assert len(decls) == 1, f"Buffer IR uses one shared carry handle:\n{mlir}"
    else:
        assert len(decls) > 1, f"Tile IR declares one alloc_tile per var:\n{mlir}"
    addrs = {_first_group(r"addr = (%[A-Za-z0-9_]+)", ln) for ln in decls}
    assert len(addrs) == 1, f"the aliased tiles must share one baked addr, got {addrs}:\n{mlir}"


# ── [N, 1] col-vector carry: view yields must be copied, and the carry written ──

COLVEC_ROWS, COLVEC_STEPS = 16, 4


@pl.program
class ColVectorCarryProgram:
    """A `[N, 1]` col-vector loop carry updated through an if/else — the shape of an
    online-softmax `mi` / `li` carry.

    pypto runs `[N, 1]` elementwise on the `[1, N]` row-major view, so each branch
    yields a *reshape view* of its producer, not the producer itself.
    """

    @pl.function
    def kernel(
        self,
        x: pl.Tensor[[COLVEC_ROWS, 1], pl.FP32],
        y: pl.Tensor[[COLVEC_ROWS, 1], pl.FP32],
        out: pl.Out[pl.Tensor[[COLVEC_ROWS, 1], pl.FP32]],
    ) -> pl.Tensor[[COLVEC_ROWS, 1], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="colvec_carry"):
            li = x[:, :]
            for i in pl.range(COLVEC_STEPS):  # noqa: B007 - index drives the branch
                b = y[:, :]
                if i == 0:
                    li = pl.mul(li, b)
                else:
                    li = pl.add(pl.mul(li, b), b)
            out[:, :] = li
        return out


def _incore_ptoas_pto(program) -> str:
    """Emit PTO for a `pl.at`-outlined kernel under the PTOAS planner."""
    reset_for_testing()
    set_backend_type(BackendType.Ascend910)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    incore = [f for f in optimized.functions.values() if f.func_type != pl.FunctionType.Orchestration]
    assert len(incore) == 1, f"expected one in-core function, got {[f.name for f in incore]}"
    single = _ir.Program([incore[0]], incore[0].name, optimized.span)
    return codegen.PTOCodegen().generate(single, emit_tile_addr=False)


def _outs_of(line: str) -> str:
    return _first_group(r"outs\((%[A-Za-z0-9_]+)", line)


def _ins_of(line: str) -> str:
    return _first_group(r"ins\((%[A-Za-z0-9_]+)", line)


def _tile_handle_definitions(mlir):
    """Record exact SSA types and the allocation shared by each reshape view."""
    definitions = {}
    for index, line in enumerate(mlir.splitlines()):
        match = re.match(r"\s*(%[A-Za-z0-9_]+) = pto\.(alloc_tile|treshape)\b", line)
        if match is None:
            continue
        name, kind = match.groups()
        assert name not in definitions, f"duplicate tile handle definition: {line}"
        types = _tile_buf_types(line)
        root = name
        if kind == "treshape":
            source = _first_group(r"pto\.treshape (%[A-Za-z0-9_]+)", line)
            assert source in definitions, f"view source must dominate its definition: {line}"
            assert types[0] == definitions[source][1], line
            root = definitions[source][2]
        definitions[name] = (index, types[-1], root)
    return definitions


def _if_arm_lines(mlir):
    lines = mlir.splitlines()
    start = lines.index(_sole_line(mlir, "scf.if "))
    arms = [[], []]
    depth, arm = 1, 0
    for index in range(start + 1, len(lines)):
        line = lines[index]
        stripped = line.strip()
        if stripped == "} else {" and depth == 1:
            arm = 1
            continue
        if stripped == "}":
            depth -= 1
            if depth == 0:
                return start, index, arms
        elif stripped.endswith("{"):
            depth += 1
        arms[arm].append((index, line))
    raise AssertionError(f"unterminated if region:\n{mlir}")


def _assert_branch_tile_values(mlir, destination, initial, expected):
    """Follow arithmetic and copies through views, requiring arm-local dominance."""
    definitions = _tile_handle_definitions(mlir)
    start, _, arms = _if_arm_lines(mlir)
    target = definitions[destination][2]
    assert definitions[target][0] < start, "the destination allocation must dominate both arms"
    initial_roots = {definitions[handle][2]: value for handle, value in initial.items()}
    for arm, expected_value in zip(arms, expected, strict=True):
        indices = {index for index, _ in arm}
        state = dict(initial_roots)
        for index, line in arm:
            operation = re.search(r"\bpto\.(tmov|tadd|tmul|tadds)\b", line)
            if operation is None:
                continue
            operands = line.split("ins(", 1)[1].split(" :", 1)[0]
            inputs = [handle for handle in re.findall(r"%[A-Za-z0-9_]+", operands) if handle in definitions]
            output = _outs_of(line)
            for handle in [*inputs, output]:
                definition = definitions[handle][0]
                assert definition < index and (definition < start or definition in indices), line
            values = tuple(state[definitions[handle][2]] for handle in inputs)
            state[definitions[output][2]] = values[0] if operation[1] == "tmov" else (operation[1], values)
        assert state[target] == expected_value, f"branch did not write its selected value:\n{mlir}"


def test_colvec_carry_branch_views_are_copied_into_the_phi_handle():
    """Each arm's view must reach the shared carry, directly or through an edge copy."""
    mlir = _incore_ptoas_pto(ColVectorCarryProgram)

    store = _sole_line(mlir, "pto.tstore")
    carry = _ins_of(store)

    loads = [line for line in mlir.splitlines() if "pto.tload" in line]
    assert len(loads) == 2, mlir
    assert _outs_of(loads[0]) == carry, mlir
    other = _outs_of(loads[1])
    product = ("tmul", ("carry", "other"))
    _assert_branch_tile_values(
        mlir, carry, {carry: "carry", other: "other"}, (product, ("tadd", (product, "other")))
    )
    _assert_tmov_operand_types(mlir)


def test_colvec_carry_tmov_annotates_each_operand_with_its_own_type():
    """A `pto.treshape` view carries static valid dims; an `alloc_tile` handle does
    not. Annotating both tmov operands with one type makes the def/use types
    disagree and ptoas rejects the module."""
    # Independent physical roots guarantee a real transfer even when branch
    # planning can otherwise eliminate an intermediate phi copy.
    program = _move_to_shared_storage_program((32, 1), (32, 1), source_shape=(1, 32))
    mlir = _shared_storage_move_pto(program)
    _assert_tmov_operand_types(mlir)
    move = _sole_line(mlir, "pto.tmov")
    definitions = _tile_handle_definitions(mlir)
    source, target = _ins_of(move), _outs_of(move)
    assert definitions[source][2] != definitions[target][2], mlir
    source_type, target_type = _tile_buf_types(move)
    assert "v_row=32, v_col=1" in source_type and "v_row=?, v_col=?" in target_type, move
    assert _ins_of(_sole_line(mlir, "pto.tstore")) == target, mlir


def test_colvec_carry_is_written_back_under_pypto_planner():
    """Default planner: MemoryReuse's YieldFixupMutator inserts the same write-back.

    Aliasing is by baked address, not by SSA name — each var keeps its own
    `alloc_tile` — so the write-back's destination is a *different* handle sitting
    at the carry's address.
    """
    reset_for_testing()
    set_backend_type(BackendType.Ascend910)
    optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(ColVectorCarryProgram)
    incore = [f for f in optimized.functions.values() if f.func_type != pl.FunctionType.Orchestration]
    mlir = codegen.PTOCodegen().generate(_ir.Program([incore[0]], incore[0].name, optimized.span))

    assert "pto.treshape" not in mlir, f"PYPTO expresses views as re-declared allocs:\n{mlir}"

    def addr_of(handle: str) -> str:
        return _first_group(r"addr = (%[A-Za-z0-9_]+)", _sole_line(mlir, f"{handle} = pto.alloc_tile"))

    carry_addr = addr_of(_ins_of(_sole_line(mlir, "pto.tstore")))
    written = [ln for ln in mlir.splitlines() if "pto.tmov" in ln and addr_of(_outs_of(ln)) == carry_addr]
    assert written, f"the carry's buffer must be written before the store:\n{mlir}"


@pl.program
class BranchReshapeProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[32, 1], pl.FP32],
        flag: pl.Scalar[pl.INT32],
        out: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
    ) -> pl.Tensor[[1, 16], pl.FP32]:
        base = pl.load(x, [0, 0], [32, 1])
        if flag != 0:
            row = pl.reshape(base, [1, 32])
            updated = pl.add(row, 1.0)
            column = pl.reshape(updated, [32, 1])
            result = column
        else:
            result = base
        after = pl.reshape(result, [1, 32])
        head = pl.tile.extract(after, 0, 0, [1, 16], target_memory=pl.Mem.Vec)
        return pl.store(head, [0, 0], out)


@pl.program
class BranchTransposeViewProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[32, 1], pl.FP32],
        flag: pl.Scalar[pl.INT32],
        out: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
    ) -> pl.Tensor[[1, 16], pl.FP32]:
        base = pl.load(x, [0, 0], [32, 1])
        if flag != 0:
            row = pl.reshape(base, [1, 32])
            updated = pl.add(row, 1.0)
            column = pl.reshape(updated, [32, 1])
            result = column
        else:
            result = base
        after = pl.tile.transpose_view(result)
        head = pl.tile.extract(after, 0, 0, [1, 16], target_memory=pl.Mem.Vec)
        return pl.store(head, [0, 0], out)


@pytest.mark.parametrize("program", [BranchReshapeProgram, BranchTransposeViewProgram])
def test_post_if_view_reads_merged_phi_handle(program):
    """A same-typed old target cannot replace a view of the actual if result (#3015)."""
    mlir = _ptoas_pto(program)
    lines = mlir.splitlines()
    extract = _sole_line(mlir, "pto.textract ins(")
    source = _ins_of(extract)
    view = _sole_line(mlir, f"{source} =")
    assert "= pto.treshape " in view, mlir
    phi = _first_group(r"= pto\.treshape (%[A-Za-z0-9_]+)", view)

    # The dead incoming value can own the destination. Its pass-through arm
    # retains the load; the other arm must update those same physical bytes.
    definitions = _tile_handle_definitions(mlir)
    root = definitions[phi][2]
    incoming = _outs_of(_sole_line(mlir, "pto.tload"))
    assert root == definitions[incoming][2], mlir
    _, end, _ = _if_arm_lines(mlir)
    assert definitions[phi][0] < lines.index(_sole_line(mlir, "scf.if ")), mlir
    assert lines.index(view) > end, "the post-if view must be rebuilt in the continuation"
    _assert_branch_tile_values(mlir, phi, {incoming: "incoming"}, (("tadds", ("incoming",)), "incoming"))


@pl.program
class RowInitializedColumnCarryProgram:
    """A column carry whose original allocation was defined as a row (#3017)."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        flag: pl.Scalar[pl.INT32],
        out: pl.Out[pl.Tensor[[32, 1], pl.FP32]],
    ) -> pl.Tensor[[32, 1], pl.FP32]:
        row_base = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)
        base = pl.tile.reshape(row_base, [32, 1])
        for i, (carry,) in pl.range(2, init_values=(base,)):  # noqa: B007 - explicit carry binding
            if flag != 0:
                row = pl.reshape(carry, [1, 32])
                updated = pl.add(row, 1.0)
                column = pl.reshape(updated, [32, 1])
                value = column
            else:
                value = carry
            result = pl.yield_(value)
        output = pl.store(result, [0, 0], out)
        return output


def _tile_buf_types(line: str) -> list[str]:
    return re.findall(r"!pto\.tile_buf<[^>]+>", line)


def _assert_tmov_operand_types(mlir: str) -> None:
    """TMOV must preserve each SSA definition and copy matching tile geometry."""
    definitions = {}
    for line in mlir.splitlines():
        match = re.match(r"\s*(%[A-Za-z0-9_]+) = pto\.(?:alloc_tile|treshape)\b", line)
        if match:
            definitions[match.group(1)] = _tile_buf_types(line)[-1]
    moves = [line for line in mlir.splitlines() if "pto.tmov" in line]
    assert moves, mlir
    for move in moves:
        source, destination = _tile_buf_types(move)
        assert source == definitions[_ins_of(move)], move
        assert destination == definitions[_outs_of(move)], move
        for field in ("rows", "cols", "blayout", "slayout"):
            pattern = rf"\b{field}=([^,>]+)"
            assert _first_group(pattern, source) == _first_group(pattern, destination), move


def test_row_initialized_column_carry_writes_a_view_of_its_original_storage():
    """The loop write-back must reinterpret the row allocation before TMOV."""
    mlir = _ptoas_pto(RowInitializedColumnCarryProgram)
    _assert_tmov_operand_types(mlir)
    root = _outs_of(_sole_line(mlir, "pto.texpands"))
    root_alloc = _sole_line(mlir, f"{root} = pto.alloc_tile")
    assert "rows=1, cols=32" in root_alloc and "v_row=?, v_col=?" in root_alloc, root_alloc
    # A column descriptor is a view of row storage, never another physical
    # allocation that could strand the loop's original carry.
    allocations = [line for line in mlir.splitlines() if "= pto.alloc_tile" in line]
    assert allocations and all("rows=1, cols=32" in line for line in allocations), mlir

    # Both the post-loop store and the loop write-back address the original bytes.
    store_handle = _ins_of(_sole_line(mlir, "pto.tstore"))
    assert f"{store_handle} = pto.treshape {root} " in mlir, mlir
    definitions = _tile_handle_definitions(mlir)
    assert definitions[store_handle][2] == root, mlir
    _assert_branch_tile_values(mlir, store_handle, {root: "carry"}, (("tadds", ("carry",)), "carry"))
    _, _, arms = _if_arm_lines(mlir)
    into_root = [
        (index, line)
        for arm in arms
        for index, line in arm
        if "pto.tmov" in line and definitions[_outs_of(line)][2] == root
    ]
    assert into_root, "the typed branch view must address the original carry storage"
    for index, move in into_root:
        destination = _outs_of(move)
        view_index, view_type, view_root = definitions[destination]
        assert view_root == root and view_index < index, move
        assert "rows=32, cols=1" in view_type and "blayout=col_major" in view_type, move


def test_row_initialized_column_carry_keeps_baked_address_codegen():
    mlir = _planner_pto(RowInitializedColumnCarryProgram, passes.MemoryPlanner.PYPTO)
    assert "pto.treshape" not in mlir, mlir
    assert "= pto.alloc_tile addr = " in mlir, mlir
    _assert_tmov_operand_types(mlir)


@pytest.mark.ptoas_compile
def test_row_initialized_column_carry_assembles_with_ptoas(tmp_path: Path):
    """PTOAS must accept the real loop write-back, beyond matching annotations."""
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    mlir = _ptoas_pto(RowInitializedColumnCarryProgram)
    source, output = tmp_path / "column_carry.pto", tmp_path / "column_carry.cpp"
    source.write_text(mlir)
    _run_ptoas(str(source), str(output), ["--pto-arch=a2", "--pto-level=level2"])
    assert "TMOV(" in output.read_text()


def _move_to_shared_storage_program(
    root_shape: tuple[int, int],
    logical_shape: tuple[int, int],
    *,
    valid_shape: str = "full",
    source_shape: tuple[int, int] | None = None,
) -> _ir.Program:
    """Build the lowered shared-MemRef state without high-level layout repair."""
    span = _ir.Span.unknown()
    zero = _ir.ConstInt(0, pl.INDEX, span)
    root_memref = _ir.MemRef(_ir.Var("root_base", _ir.PtrType(), span), zero, 128, span)
    source_memref = _ir.MemRef(_ir.Var("source_base", _ir.PtrType(), span), zero, 128, span)
    valid_rows = _ir.Var("valid_rows", _ir.ScalarType(pl.INDEX), span)
    layout = _ir.TileLayout.col_major if logical_shape[1] == 1 else _ir.TileLayout.row_major
    valid = [valid_rows, 1] if valid_shape == "dynamic" else [0, 0] if valid_shape == "zero" else None
    view = _ir.TileView(valid_shape=valid, blayout=layout)
    root_type = _ir.TileType(root_shape, pl.FP32, root_memref, None, _ir.MemorySpace.Vec)
    source_type = _ir.TileType(logical_shape, pl.FP32, source_memref, view, _ir.MemorySpace.Vec)
    destination_type = _ir.TileType(logical_shape, pl.FP32, root_memref, view, _ir.MemorySpace.Vec)
    root = _ir.Var("root", root_type, span)
    source = _ir.Var("source", source_type, span)
    destination = _ir.Var("destination", destination_type, span)
    output = _ir.Var("output", _ir.TensorType(logical_shape, pl.FP32), span)
    stored = _ir.Var("stored", output.type, span)
    offsets = _ir.MakeTuple([zero, zero], span)
    source_statements = [_ir.AssignStmt(source, _ir.Call(_ir.Op("tile.create"), [], source_type, span), span)]
    if source_shape is not None:
        source_storage_type = _ir.TileType(source_shape, pl.FP32, source_memref, None, _ir.MemorySpace.Vec)
        source_storage = _ir.Var("source_storage", source_storage_type, span)
        shape = _ir.MakeTuple([_ir.ConstInt(size, pl.INDEX, span) for size in logical_shape], span)
        source_statements = [
            _ir.AssignStmt(
                source_storage, _ir.Call(_ir.Op("tile.create"), [], source_storage_type, span), span
            ),
            _ir.AssignStmt(
                source, _ir.Call(_ir.Op("tile.reshape"), [source_storage, shape], source_type, span), span
            ),
        ]
    body = _ir.SeqStmts(
        [
            _ir.AssignStmt(root, _ir.Call(_ir.Op("tile.create"), [], root_type, span), span),
            *source_statements,
            _ir.AssignStmt(
                destination,
                _ir.Call(_ir.Op("tile.move"), [source], destination_type, span),
                span,
            ),
            _ir.AssignStmt(
                stored,
                _ir.Call(_ir.Op("tile.store"), [destination, offsets, output], output.type, span),
                span,
            ),
            _ir.ReturnStmt([stored], span),
        ],
        span,
    )
    params = [(output, _ir.ParamDirection.Out)]
    if valid_shape == "dynamic":
        params.append((valid_rows, _ir.ParamDirection.In))
    function = _ir.Function("kernel", params, [output.type], body, span, _ir.FunctionType.InCore)
    return _ir.Program([function], "SharedStorageMove", span)


def _shared_storage_move_pto(program: _ir.Program) -> str:
    reset_for_testing()
    set_backend_type(BackendType.Ascend910)
    return codegen.PTOCodegen().generate(program, emit_tile_addr=False)


@pytest.mark.parametrize(
    ("root_shape", "logical_shape"),
    [((1, 32), (32, 1)), ((32, 1), (1, 32)), ((1, 32), (1, 32))],
    ids=["row-storage-column-result", "column-storage-row-result", "matching-row-types"],
)
def test_shared_storage_move_uses_the_logical_destination_type(root_shape, logical_shape):
    mlir = _shared_storage_move_pto(_move_to_shared_storage_program(root_shape, logical_shape))
    _assert_tmov_operand_types(mlir)
    assert mlir.count("= pto.alloc_tile") == 2, mlir
    move = _sole_line(mlir, "pto.tmov")
    assert _ins_of(_sole_line(mlir, "pto.tstore")) == _outs_of(move), mlir
    if root_shape == logical_shape:
        assert "pto.treshape" not in mlir, mlir
    else:
        view = _sole_line(mlir, "pto.treshape")
        assert f"{_outs_of(move)} = pto.treshape %root " in view, mlir
        rows, cols = logical_shape
        assert f"rows={rows}, cols={cols}, v_row={rows}, v_col={cols}" in view, view


def test_shared_storage_move_initializes_a_dynamic_valid_destination_view():
    program = _move_to_shared_storage_program((1, 32), (32, 1), valid_shape="dynamic")
    mlir = _shared_storage_move_pto(program)
    _assert_tmov_operand_types(mlir)
    definitions = _tile_handle_definitions(mlir)
    move = _sole_line(mlir, "pto.tmov")
    source, destination = _ins_of(move), _outs_of(move)
    assert definitions[source][2] != definitions[destination][2], mlir
    assert definitions[destination][2] == "%root", mlir
    assert "rows=32, cols=1, v_row=?, v_col=?" in definitions[destination][1], mlir
    initialized = _sole_line(mlir, f"pto.set_validshape {destination},")
    valid_rows = re.search(r"(%[A-Za-z0-9_]+): index", _sole_line(mlir, "func.func @kernel"))
    assert valid_rows is not None, mlir
    assert f"{valid_rows[1]}, %c1_index : {definitions[destination][1]}" in initialized, mlir
    assert definitions[destination][0] < mlir.splitlines().index(initialized), mlir
    assert mlir.index(initialized) < mlir.index(move), mlir
    assert _ins_of(_sole_line(mlir, "pto.tstore")) == destination, mlir
    assert "pto.set_validshape %root," not in mlir, mlir
    assert mlir.count("= pto.alloc_tile") == 2, mlir


def test_shared_storage_move_keeps_the_zero_valid_sentinel_dynamic():
    program = _move_to_shared_storage_program((1, 32), (32, 1), valid_shape="zero")
    mlir = _shared_storage_move_pto(program)
    _assert_tmov_operand_types(mlir)
    assert mlir.count("= pto.alloc_tile") == 2, mlir
    view = _sole_line(mlir, "pto.treshape")
    assert "rows=32, cols=1, v_row=?, v_col=?" in _tile_buf_types(view)[-1], view
    assert "v_row=0" not in view and "v_col=0" not in view, view
    assert f"{_outs_of(_sole_line(mlir, 'pto.tmov'))} = pto.treshape %root " in view, mlir


def _if_shared_metadata_program(*, dynamic_result: bool = False) -> _ir.Program:
    """Keep one lowered allocation while its if result owns different live metadata."""
    span = _ir.Span.unknown()
    zero = _ir.ConstInt(0, pl.INDEX, span)
    rows = _ir.Var("root_rows", _ir.ScalarType(pl.INDEX), span)
    selected_rows = _ir.Var("selected_rows", _ir.ScalarType(pl.INDEX), span)
    condition = _ir.Var("condition", _ir.ScalarType(pl.BOOL), span)
    memref = _ir.MemRef(_ir.Var("root_base", _ir.PtrType(), span), zero, 512, span)
    root_view = _ir.TileView(
        valid_shape=[rows, 32], pad=_ir.PadValue.zero if dynamic_result else _ir.PadValue.max
    )
    result_view = _ir.TileView(
        valid_shape=[selected_rows, 32] if dynamic_result else [4, 32],
        pad=_ir.PadValue.zero,
    )
    root_type = _ir.TileType([4, 32], pl.FP32, memref, root_view, _ir.MemorySpace.Vec)
    result_type = _ir.TileType([4, 32], pl.FP32, memref, result_view, _ir.MemorySpace.Vec)
    root = _ir.Var("root", root_type, span)
    selected = _ir.Var("selected", result_type, span)
    incoming_selected = _ir.Var("incoming_selected", result_type, span) if dynamic_result else selected
    output = _ir.Var("output", _ir.TensorType([4, 32], pl.FP32), span)
    stored = _ir.Var("stored", output.type, span)
    observed = _ir.Var("observed", output.type, span)
    offsets = _ir.MakeTuple([zero, zero], span)

    def branch(name: str) -> _ir.SeqStmts:
        value = _ir.Var(name, result_type, span)
        # create accepts the already-lowered dynamic descriptor; fillpad makes
        # the static full-valid result while retaining root's partial metadata.
        call = (
            _ir.Call(_ir.Op("tile.create"), [], result_type, span)
            if dynamic_result
            else _ir.Call(
                _ir.Op("tile.fillpad_inplace"),
                [root],
                {"pad_value": _ir.PadValue.zero},
                result_type,
                span,
            )
        )
        return _ir.SeqStmts([_ir.AssignStmt(value, call, span), _ir.YieldStmt([value], span)], span)

    branches = [_ir.IfStmt(condition, branch("left"), branch("right"), [incoming_selected], span)]
    if dynamic_result:
        # The source already has its own live dynamic metadata before this if.
        # Another result view must share its bytes without emitting a self-copy.
        branches.append(
            _ir.IfStmt(
                condition,
                _ir.YieldStmt([incoming_selected], span),
                _ir.YieldStmt([incoming_selected], span),
                [selected],
                span,
            )
        )
    body = _ir.SeqStmts(
        [
            _ir.AssignStmt(root, _ir.Call(_ir.Op("tile.create"), [], root_type, span), span),
            *branches,
            _ir.AssignStmt(
                stored, _ir.Call(_ir.Op("tile.store"), [selected, offsets, output], output.type, span), span
            ),
            _ir.AssignStmt(
                observed, _ir.Call(_ir.Op("tile.store"), [root, offsets, stored], output.type, span), span
            ),
            _ir.ReturnStmt([observed], span),
        ],
        span,
    )
    params = [
        (output, _ir.ParamDirection.Out),
        (rows, _ir.ParamDirection.In),
        (condition, _ir.ParamDirection.In),
    ]
    if dynamic_result:
        params.append((selected_rows, _ir.ParamDirection.In))
    function = _ir.Function("kernel", params, [output.type], body, span, _ir.FunctionType.InCore)
    return _ir.Program([function], "IfMetadataViews", span)


@pytest.mark.parametrize(
    "dynamic_result", [False, True], ids=["fillpad-full-valid", "distinct-dynamic-valid"]
)
def test_if_canonical_storage_retains_independent_result_metadata(dynamic_result):
    mlir = _shared_storage_move_pto(_if_shared_metadata_program(dynamic_result=dynamic_result))
    lines = mlir.splitlines()
    signature = _sole_line(mlir, "func.func @kernel(")
    index_parameters = re.findall(r"(%[A-Za-z0-9_]+): index", signature)
    assert len(index_parameters) == (2 if dynamic_result else 1), signature
    root_rows = index_parameters[0]
    definitions = _tile_handle_definitions(mlir)
    allocation = _first_group(r"(%[A-Za-z0-9_]+) = pto\.alloc_tile", _sole_line(mlir, "= pto.alloc_tile"))
    stores = [line for line in lines if "pto.tstore" in line]
    assert len(stores) == 2, mlir
    result, old_alias = _ins_of(stores[0]), _ins_of(stores[1])
    assert old_alias == allocation and result != allocation, mlir
    assert definitions[result][2] == allocation, mlir
    root_pad = 1 if dynamic_result else 2
    assert f"pad={root_pad}>" in definitions[allocation][1], mlir
    assert "pad=1>" in definitions[result][1], mlir
    assert _tile_buf_types(stores[0]) == [definitions[result][1]], mlir
    assert _tile_buf_types(stores[1]) == [definitions[allocation][1]], mlir
    if_index = max(i for i, line in enumerate(lines) if "scf.if" in line)
    initialized = [
        (i, line) for i, line in enumerate(lines) if line.strip().startswith(f"pto.set_validshape {result},")
    ]
    assert len(initialized) == 1, mlir
    assert definitions[result][0] < initialized[0][0] < if_index, mlir
    expected_rows = index_parameters[1] if dynamic_result else "%c4_index"
    assert f"pto.set_validshape {result}, {expected_rows}, %c32_index" in initialized[0][1], mlir
    # Restoring r must never update the canonical allocation or an older alias.
    assert not any(line.strip().startswith(f"pto.set_validshape {allocation},") for line in lines), mlir
    assert "pto.tmov" not in mlir, "metadata-only aliases must not cause an overlapping self-copy"
    if dynamic_result:
        assert definitions[result][1] == definitions[allocation][1], mlir
    alloc_line = _sole_line(mlir, "= pto.alloc_tile")
    assert f"valid_row = {root_rows}" in alloc_line, mlir
    if not dynamic_result:
        fillpads = [line for line in lines if "pto.tfillpad" in line]
        assert len(fillpads) == 2, mlir
        for fillpad in fillpads:
            assert _ins_of(fillpad) == allocation, mlir
            assert definitions[_outs_of(fillpad)][2] == allocation, mlir
            assert "pad=1>" in _tile_buf_types(fillpad)[1], mlir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
