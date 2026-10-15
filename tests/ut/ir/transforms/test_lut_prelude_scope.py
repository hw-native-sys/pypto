# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""t25: gather LUT prelude scope-placement regression tests.

The LegalizeTileCast gather route builds a 256-entry u16 LUT on the first
FP4->FP8E4M3FN cast. The construction (pto.tci + bit-op chain) must land
at the InCore scope body top, NOT inside any scf.for — a non-peeled
structure (cast directly inside pl.pipeline without chunk0 peeling) would
rebuild the ~58-op table on every iteration.
"""

import re

import pypto.language as pl
from pypto import backend, ir
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes


def _emit_incore_mlir(program) -> str:
    reset_for_testing()
    set_backend_type(BackendType.Ascend950)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    parts: list[str] = []
    for func in optimized.functions.values():
        if func.func_type in (pl.FunctionType.Orchestration, pl.FunctionType.Group):
            continue
        single = ir.Program([func], func.name, optimized.span)
        result = codegen.PTOCodegen().generate(single, emit_tile_addr=True)
        parts.append(result if isinstance(result, str) else "".join(result.values()))
    return "\n".join(parts)


def _tci_before_first_for(mlir: str) -> bool:
    """True if every pto.tci appears before the first scf.for.

    Records ALL pto.tci positions (PR#2987 review, Major 4): the old helper
    only appended while ``for_pos is None``, so every recorded position was
    by construction before the loop and the check could never fail.
    """
    tci_positions = []
    lines = mlir.splitlines()
    for_pos = None
    for i, line in enumerate(lines):
        if "pto.tci" in line:
            tci_positions.append(i)
        if "scf.for" in line and for_pos is None:
            for_pos = i
    if for_pos is None:
        return True  # no loops at all
    return all(p < for_pos for p in tci_positions)


def _luts_defined_before_use(mlir: str) -> bool:
    """Textual def-before-use check: every tgather's LUT operand must have a
    definition line EARLIER in the emitted text.

    This is a linear-text scan, not region/dominance awareness: it cannot
    catch a use that references a definition inside an already-closed scf
    region printed earlier (t25's revalidation reproduced that false
    negative). The same test's ``len(tables) >= 2`` assertion is what
    actually guards the PR#2987 Major-1 region leak — with the fix, the
    post-loop cast rebuilds its own table instead of referencing the
    in-region definition. This helper still catches the plain ordering
    failure where a gather appears before its table's construction text.
    """
    lut_name = r"[A-Za-z0-9_]*fp4_lut__table[A-Za-z0-9_]*"
    defined: set[str] = set()
    for line in mlir.splitlines():
        m = re.match(rf"\s*%({lut_name}) = ", line)
        if m:
            defined.add(m.group(1))
        for v in re.findall(rf"pto\.tgather ins\(%({lut_name})", line):
            if v not in defined:
                return False
    return True


@pl.program
class PeeledCast:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self, b: pl.Tensor[[16, 32], pl.FP4E2M1X2], out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]]
    ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
        f8 = pl.cast(pl.load(b, [0, 0], [16, 32]), pl.FP8E4M3FN)
        return pl.store(f8, [0, 0], out)


@pl.program
class NonPeeledCast:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self, b: pl.Tensor[[16, 32], pl.FP4E2M1X2], out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]]
    ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
        # Non-peeled: the FIRST cast is inside the pipeline loop (no chunk0).
        # The t25 InCoreScope prelude hoist must move tile.ci out of the loop.
        for kb in pl.pipeline(0, 4, stage=1):
            k0 = kb * 16
            f8 = pl.cast(pl.load(b, [0, k0 // 2], [16, 16]), pl.FP8E4M3FN)
            out_t: pl.Tensor[[16, 64], pl.FP8E4M3FN] = pl.store(f8, [0, 0], out)
        return out_t


def test_lut_ci_not_inside_loop_peeled():
    mlir = _emit_incore_mlir(PeeledCast)
    assert "pto.tci" in mlir, f"expected LUT ramp pto.tci, got:\n{mlir[:500]}"
    assert _tci_before_first_for(mlir), (
        "pto.tci appears inside an scf.for — LUT would be rebuilt per iteration"
    )


def test_lut_ci_not_inside_loop_non_peeled():
    mlir = _emit_incore_mlir(NonPeeledCast)
    assert "pto.tci" in mlir, f"expected LUT ramp pto.tci, got:\n{mlir[:500]}"
    assert "scf.for" in mlir, "expected the pipeline loop in the non-peeled kernel"
    assert _tci_before_first_for(mlir), (
        "pto.tci appears inside an scf.for — the t25 InCoreScope prelude hoist "
        "should have moved the LUT construction to the scope body top"
    )


@pl.program
class LoopThenTrailingCast:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        b1: pl.Tensor[[16, 32], pl.FP4E2M1X2],
        b2: pl.Tensor[[16, 32], pl.FP4E2M1X2],
        out1: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
        out2: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
    ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
        for kb in pl.pipeline(0, 4, stage=1):
            k0 = kb * 8
            f8a = pl.cast(pl.load(b1, [0, k0 // 2], [16, 8]), pl.FP8E4M3FN)
            out_t: pl.Tensor[[16, 64], pl.FP8E4M3FN] = pl.store(f8a, [0, 0], out1)
        # Cast AFTER the loop region: must not reuse the LUT Var built inside
        # the loop body (region definitions do not dominate code after the
        # region; PTOAS rejects the reference as an undeclared SSA name).
        f8b = pl.cast(pl.load(b2, [0, 0], [16, 32]), pl.FP8E4M3FN)
        out_t = pl.store(f8b, [0, 0], out2)
        return out_t


def test_cast_after_loop_rebuilds_its_own_lut():
    """PR#2987 review Major 1 regression: a function-level cast following a
    pipeline-loop cast must rebuild its own LUT table instead of referencing
    the one defined inside the loop region."""
    mlir = _emit_incore_mlir(LoopThenTrailingCast)
    tables = set(re.findall(r"%(\S*fp4_lut__table\S*) = ", mlir))
    assert len(tables) >= 2, (
        f"expected the post-loop cast to rebuild its own LUT (>=2 table "
        f"definitions), got {sorted(tables)}:\n{mlir[:800]}"
    )
    assert _luts_defined_before_use(mlir), (
        "a tgather references a LUT table defined later or inside a closed region — dominance violation"
    )


def _lut_table_defs_and_uses(mlir_or_ir_text: str):
    names = sorted(set(re.findall(r"fp4_lut__table_\w+", mlir_or_ir_text)))
    defs = {v: len(re.findall(rf"\b{v}: pl\.Tile", mlir_or_ir_text)) for v in names}
    uses = {v: len(re.findall(rf"\b{v}\b", mlir_or_ir_text)) - defs[v] for v in names}
    return defs, uses


def test_if_else_branches_each_build_their_own_lut():
    """Codex P1 (PR#2987 round 2): FP4->FP8 casts in BOTH branches of a
    function-level if must each get a dominating table — the else branch must
    not reuse the then branch's (non-dominating) table var."""

    @pl.program
    class IfElseCasts:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            s: pl.Tensor[[1, 1], pl.INT32],
            b1: pl.Tensor[[16, 32], pl.FP4E2M1X2],
            b2: pl.Tensor[[16, 32], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
            cond = pl.read(s, [0, 0])
            t1 = pl.load(b1, [0, 0], [16, 32])
            t2 = pl.load(b2, [0, 0], [16, 32])
            if cond > 0:
                f1 = pl.cast(t1, pl.FP8E4M3FN)
                out_t: pl.Tensor[[16, 64], pl.FP8E4M3FN] = pl.store(f1, [0, 0], out)
            else:
                f2 = pl.cast(t2, pl.FP8E4M3FN)
                out_t = pl.store(f2, [0, 0], out)
            return out_t

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    try:
        after = passes.legalize_tile_cast()(IfElseCasts)
    finally:
        backend.reset_for_testing()
    text = ir.python_print(after)
    defs, uses = _lut_table_defs_and_uses(text)
    assert sum(defs.values()) == 2, f"expected one table per branch, got {defs}"
    assert all(d == 1 and u >= 1 for d, u in zip(defs.values(), uses.values())), (defs, uses)


def test_loop_branch_then_cast_keeps_pending_prelude_defs():
    """Codex P1 (PR#2987 round 2): loop { if { first cast } ... cast }: the
    second build (after the branch guard restores the cache) must not wipe the
    still-pending first table's definitions from the prelude."""

    @pl.program
    class LoopIfCast:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            b1: pl.Tensor[[16, 32], pl.FP4E2M1X2],
            b2: pl.Tensor[[16, 32], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
            out_t: pl.Tensor[[16, 64], pl.FP8E4M3FN] = pl.store(
                pl.cast(pl.load(b1, [0, 0], [16, 32]), pl.FP8E4M3FN), [0, 0], out
            )
            for kb in pl.pipeline(0, 4, stage=1):
                if kb == 0:
                    c1 = pl.cast(pl.load(b1, [0, 0], [16, 32]), pl.FP8E4M3FN)
                    out_t = pl.store(c1, [0, 0], out)
                c2 = pl.cast(pl.load(b2, [0, 0], [16, 32]), pl.FP8E4M3FN)
                out_t = pl.store(c2, [0, 0], out)
            return out_t

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    try:
        after = passes.legalize_tile_cast()(LoopIfCast)
    finally:
        backend.reset_for_testing()
    text = ir.python_print(after)
    defs, uses = _lut_table_defs_and_uses(text)
    assert sum(defs.values()) >= 2, f"expected both tables' definitions kept, got {defs}"
    assert all(d >= 1 for d in defs.values()), f"a table definition was wiped: {defs}"
    assert all(u >= 1 for u in uses.values()), (defs, uses)


def test_outer_loop_cast_inner_loop_keeps_prelude_ownership():
    """Codex P2 (PR#2987 round 4): an outer loop whose body casts FP4->FP8
    BEFORE an inner (cast-free) loop — the pending LUT construction belongs
    to the OUTER loop; hoisting it above the inner loop would place it after
    the referencing gather (non-dominating SSA)."""

    @pl.program
    class OuterCastInnerLoop:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            b: pl.Tensor[[16, 32], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
            out_t: pl.Tensor[[16, 64], pl.FP8E4M3FN] = pl.store(
                pl.cast(pl.load(b, [0, 0], [16, 32]), pl.FP8E4M3FN), [0, 0], out
            )
            for k in pl.pipeline(0, 4, stage=1):
                f8 = pl.cast(pl.load(b, [0, 0], [16, 32]), pl.FP8E4M3FN)
                out_t = pl.store(f8, [0, 0], out)
                for j in pl.range(0, 2):
                    out_t = pl.store(f8, [0, 0], out)
            return out_t

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    try:
        after = passes.legalize_tile_cast()(OuterCastInnerLoop)
    finally:
        backend.reset_for_testing()
    lines = ir.python_print(after).splitlines()
    assert not any("__FREE_VAR" in line for line in lines)
    gather_pos = next((i for i, line in enumerate(lines) if "tile.gather" in line), -1)
    table_def = next(
        (i for i, line in enumerate(lines) if "fp4_lut__table" in line and ": pl.Tile" in line), -1
    )
    assert gather_pos >= 0 and table_def >= 0
    assert table_def < gather_pos, (
        "the LUT construction must precede its referencing gather "
        f"(table@{table_def} vs gather@{gather_pos}) — the inner loop hoisted a "
        "prelude owned by the outer body"
    )


def test_nested_loop_first_cast_in_inner_loop_hoists_above_outer_loop():
    """Codex P2 (PR#2987 round 6): the FIRST FP4->FP8 cast sits inside NESTED
    function-level loops. The constant LUT construction must ride up above the
    OUTER loop (built once) — flushing it above the inner loop would rebuild
    the ~58-op table on every outer iteration."""

    @pl.program
    class NestedInnerCast:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            b: pl.Tensor[[16, 32], pl.FP4E2M1X2],
            x: pl.Tensor[[16, 64], pl.FP8E4M3FN],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
            out_t: pl.Tensor[[16, 64], pl.FP8E4M3FN] = pl.store(pl.load(x, [0, 0], [16, 64]), [0, 0], out)
            for k in pl.pipeline(0, 4, stage=1):
                for j in pl.range(0, 2):
                    f8 = pl.cast(pl.load(b, [0, 0], [16, 32]), pl.FP8E4M3FN)
                    out_t = pl.store(f8, [0, 0], out)
            return out_t

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    try:
        after = passes.legalize_tile_cast()(NestedInnerCast)
    finally:
        backend.reset_for_testing()
    lines = ir.python_print(after).splitlines()
    assert not any("__FREE_VAR" in line for line in lines)
    table_defs = [
        i for i, line in enumerate(lines) if re.match(r"^\s*\w*fp4_lut__table\w*\s*:\s*pl\.Tile", line)
    ]
    assert len(table_defs) == 1, f"the LUT must be built exactly once, got defs at {table_defs}"
    outer_loop = next((i for i, line in enumerate(lines) if "pipeline" in line or "for k" in line), -1)
    inner_loop = next((i for i, line in enumerate(lines) if "range" in line or "for j" in line), -1)
    assert outer_loop >= 0 and inner_loop > outer_loop, (
        f"could not locate the nested loops (outer@{outer_loop}, inner@{inner_loop})"
    )
    assert table_defs[0] < outer_loop, (
        f"LUT construction (line {table_defs[0]}) must sit above the OUTER loop "
        f"(line {outer_loop}); above the inner loop (line {inner_loop}) would rebuild "
        "the table on every outer iteration"
    )
