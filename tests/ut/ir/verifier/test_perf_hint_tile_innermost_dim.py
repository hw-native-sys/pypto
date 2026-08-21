# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the TileInnermostDimGranularity perf-hint check (issue #1180, PH001)."""

from __future__ import annotations

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


@pytest.fixture(autouse=True)
def reset_backend_around_test():
    """Each test owns its backend selection; reset before and after."""
    backend.reset_for_testing()
    yield
    backend.reset_for_testing()


def _run_perf_hint_check(program: ir.Program) -> list[passes.Diagnostic]:
    """Run only the TileInnermostDimGranularity check and return its diagnostics.

    The verifier early-returns without an active PassContext, so a context must
    exist for the check to fire at all — it comes from the repo conftest's
    autouse ``pass_verification_context``. The backend is resolved live, so
    callers only need to select one (``_activate_a5`` / ``_activate_a3``) first.
    This helper runs no pass pipeline, so the context's verification level never
    applies — do not wrap calls in a verification-disabling context.
    """
    checks = passes.DiagnosticCheckSet()
    checks.insert(passes.DiagnosticCheck.TileInnermostDimGranularity)
    return passes.DiagnosticCheckRegistry.run_checks(checks, passes.DiagnosticPhase.POST_PIPELINE, program)


def _activate_a5() -> None:
    backend.set_backend_type(BackendType.Ascend950)


def _activate_a3() -> None:
    backend.set_backend_type(BackendType.Ascend910B)


# ---------------------------------------------------------------------------
# IR fixtures — tile.load / tile.store programs of various innermost sizes
# ---------------------------------------------------------------------------


def _make_load_program(innermost: int, dtype) -> ir.Program:
    """Build an InCore program with a tile.load whose innermost dim is `innermost`."""
    rows = 16

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[rows, innermost], dtype],
            out: pl.Out[pl.Tensor[[rows, innermost], dtype]],
        ) -> pl.Tensor[[rows, innermost], dtype]:
            t: pl.Tile[[rows, innermost], dtype] = pl.load(
                x,
                [0, 0],
                [rows, innermost],
                target_memory=pl.Mem.Vec,
            )
            out_1: pl.Tensor[[rows, innermost], dtype] = pl.store(t, [0, 0], out)
            return out_1

    return Prog


def _make_store_program(innermost: int, dtype) -> ir.Program:
    """Build an InCore program with a tile.store whose source tile innermost is `innermost`."""
    return _make_load_program(innermost, dtype)  # same shape covers both ops


# ---------------------------------------------------------------------------
# Below-threshold detection
# ---------------------------------------------------------------------------


def test_below_threshold_a5_emits():
    """A5 backend, FP32 [16, 16] → 64B innermost → fires PH001."""
    _activate_a5()
    program = _make_load_program(16, pl.FP32)
    diags = _run_perf_hint_check(program)
    perf_hints = [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint]
    # Both the tile.load and the tile.store carry a 64B innermost-dim tile,
    # so the check fires on both. We assert at least one with the correct code.
    assert len(perf_hints) >= 1
    assert all(d.hint_code == "PH001" for d in perf_hints)
    assert all(d.rule_name == "TileInnermostDimGranularity" for d in perf_hints)
    msg = perf_hints[0].message
    assert "64B" in msg
    assert ">= 128B" in msg
    assert "a5" in msg
    assert "L2 cache line = 512B" in msg


def test_above_threshold_a5_silent():
    """A5 backend, FP32 [16, 128] → 512B innermost → silent."""
    _activate_a5()
    program = _make_load_program(128, pl.FP32)
    diags = _run_perf_hint_check(program)
    assert [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint] == []


def test_at_threshold_a5_silent():
    """A5 backend, FP32 [16, 32] → exactly 128B innermost → silent (>= recommended)."""
    _activate_a5()
    program = _make_load_program(32, pl.FP32)
    diags = _run_perf_hint_check(program)
    assert [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint] == []


def test_below_threshold_a3_emits():
    """A3 backend (512B threshold), FP32 [16, 32] → 128B innermost → fires."""
    _activate_a3()
    program = _make_load_program(32, pl.FP32)
    diags = _run_perf_hint_check(program)
    perf_hints = [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint]
    assert len(perf_hints) >= 1
    msg = perf_hints[0].message
    assert "128B" in msg
    assert ">= 512B" in msg
    assert "a2a3" in msg


def test_above_threshold_a3_silent():
    """A3 backend, FP32 [16, 128] → 512B innermost → silent (matches threshold)."""
    _activate_a3()
    program = _make_load_program(128, pl.FP32)
    diags = _run_perf_hint_check(program)
    assert [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint] == []


# ---------------------------------------------------------------------------
# Dtype affects byte size
# ---------------------------------------------------------------------------


def test_dtype_int8_silent_at_128_elements_a5():
    """A5: INT8 with innermost=128 → 128B → silent (boundary)."""
    _activate_a5()
    program = _make_load_program(128, pl.INT8)
    diags = _run_perf_hint_check(program)
    assert [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint] == []


def test_dtype_int8_below_threshold_a5_emits():
    """A5: INT8 with innermost=64 → 64B → fires."""
    _activate_a5()
    program = _make_load_program(64, pl.INT8)
    diags = _run_perf_hint_check(program)
    perf_hints = [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint]
    assert len(perf_hints) >= 1
    assert "64B" in perf_hints[0].message


def test_dtype_fp16_threshold_a5_silent():
    """A5: FP16 with innermost=64 → 128B → silent."""
    _activate_a5()
    program = _make_load_program(64, pl.FP16)
    diags = _run_perf_hint_check(program)
    assert [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint] == []


# ---------------------------------------------------------------------------
# Op coverage and noise floor
# ---------------------------------------------------------------------------


def test_tile_store_also_checked():
    """tile.store with a small innermost source tile is also flagged.

    A small innermost size triggers the check on both ops in the program;
    we assert at least one diagnostic mentions tile.store.
    """
    _activate_a5()
    program = _make_load_program(16, pl.FP32)  # tile.load + tile.store both 64B
    diags = _run_perf_hint_check(program)
    perf_hints = [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint]
    rules = {d.rule_name for d in perf_hints}
    assert rules == {"TileInnermostDimGranularity"}
    messages = [d.message for d in perf_hints]
    assert any("tile.load" in m for m in messages)
    assert any("tile.store" in m for m in messages)


# ---------------------------------------------------------------------------
# Disabling
# ---------------------------------------------------------------------------


def test_disabled_perf_hint_silent():
    """Adding the check to disabled_diagnostics suppresses it via PassPipeline."""
    _activate_a5()
    program = _make_load_program(16, pl.FP32)
    disabled = passes.DiagnosticCheckSet()
    disabled.insert(passes.DiagnosticCheck.UnusedControlFlowResult)
    disabled.insert(passes.DiagnosticCheck.TileInnermostDimGranularity)
    with passes.PassContext([], disabled_diagnostics=disabled):
        all_checks = passes.DiagnosticCheckRegistry.get_all_checks()
        effective = all_checks.difference(disabled)
        diags = passes.DiagnosticCheckRegistry.run_checks(
            effective, passes.DiagnosticPhase.POST_PIPELINE, program
        )
    assert [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint] == []


# ---------------------------------------------------------------------------
# Memory-space coverage (issue #1305 ask 1, corrected by issue #2309)
# ---------------------------------------------------------------------------


def _make_cube_matmul_program(k: int, dtype) -> ir.Program:
    """Build an InCore matmul kernel whose tiles live in cube-private L0/L1.

    A is loaded into Mat (L1) with a small inner ``k`` (below threshold), B into
    Mat, both moved to Left/Right (L0A/L0B), multiplied into Acc (L0C), then
    stored. ``n`` is chosen so the B load and the final store meet the threshold
    and stay silent on their own, leaving the A load as the single hit.

    The A load lands in Mat, but it *reads GM*: ``tile.load``'s source is a
    ``TensorType``, which is always off-chip. So it crosses L2 at its 32B
    innermost granularity like any other GM read, and PH001 must flag it. Only
    the two ``pl.move`` calls below are genuinely cube-private, and PH001 does
    not inspect ``tile.move`` at all.
    """
    m, n = 16, 32

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def matmul(
            self,
            a: pl.Tensor[[m, k], dtype],
            b: pl.Tensor[[k, n], dtype],
            c: pl.Out[pl.Tensor[[m, n], dtype]],
        ) -> pl.Tensor[[m, n], dtype]:
            tile_a_l1 = pl.load(a, offsets=[0, 0], shapes=[m, k], target_memory=pl.Mem.Mat)
            tile_b_l1 = pl.load(b, offsets=[0, 0], shapes=[k, n], target_memory=pl.Mem.Mat)
            tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.Mem.Left)
            tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.Mem.Right)
            tile_c_l0c = pl.matmul(tile_a_l0a, tile_b_l0b)
            out_c = pl.store(tile_c_l0c, offsets=[0, 0], output_tensor=c)
            return out_c

    return Prog


def test_gm_to_mat_load_is_flagged_a5():
    """A5: a GM->Mat load is a GM transfer, so the L2 threshold applies.

    Regression guard for issue #2309. An earlier revision skipped every tile in a
    cube-private space, which silenced exactly this shape: the GM->Mat weight
    load of a ``b_trans`` matmul, whose window is the caller's [N, K] slice
    transposed. Since ``tile.load`` / ``tile.store`` are the only ops inspected
    and their non-tile side is always an off-chip ``TensorType``, that skip could
    only ever suppress true positives.
    """
    _activate_a5()
    program = _make_cube_matmul_program(8, pl.FP32)  # A-Mat innermost = 32B (< 128B)
    diags = _run_perf_hint_check(program)
    perf_hints = [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint]
    assert len(perf_hints) == 1
    msg = perf_hints[0].message
    assert "tile.load" in msg
    assert "target_memory=Mat" in msg
    # Volume clause: 16 rows of 8 fp32 elements = 512B in 32B rows.
    assert "moves 512B as 16 x 32B rows" in msg


def test_volume_is_self_consistent_for_subbyte_dtype_a5():
    """Sub-byte volume is rows x row bytes, not the packed whole-tile size.

    Rounding the whole tile to bytes in one step under-reports a bit-packed
    transfer by the packing factor and contradicts the row figures printed
    beside it: a [16, 1] bool tile moves 16 separately-addressed 1B rows, so
    16B, not the 2B a single packed rounding would claim.
    """
    _activate_a5()
    program = _make_load_program(1, pl.BOOL)  # 16 rows x 1 bool = 1B rows
    perf_hints = [
        d for d in _run_perf_hint_check(program) if d.severity == passes.DiagnosticSeverity.PerfHint
    ]
    assert len(perf_hints) >= 1
    for diag in perf_hints:
        assert "moves 16B as 16 x 1B rows" in diag.message


def test_volume_follows_valid_shape_not_physical_shape_a5():
    """A padded tile transfers only its valid region, and the hint says so.

    ``tile.load`` / ``tile.store`` size their GM partition from ``valid_shape``,
    so reporting the physical allocation would overstate traffic on every tail
    tile -- here by 4x -- and would split the dedup key away from an otherwise
    identical transfer.
    """
    _activate_a5()
    rows, inner, valid_rows = 16, 16, 4

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[rows, inner], pl.FP32],
            out: pl.Out[pl.Tensor[[rows, inner], pl.FP32]],
        ) -> pl.Tensor[[rows, inner], pl.FP32]:
            t: pl.Tile[[rows, inner], pl.FP32] = pl.load(
                x, [0, 0], [rows, inner], valid_shape=[valid_rows, inner], target_memory=pl.Mem.Vec
            )
            return pl.store(t, [0, 0], out)

    perf_hints = [d for d in _run_perf_hint_check(Prog) if d.severity == passes.DiagnosticSeverity.PerfHint]
    assert len(perf_hints) >= 1
    for diag in perf_hints:
        # 4 valid rows of 16 fp32, not the physical 16 rows / 1024B.
        assert "moves 256B as 4 x 64B rows" in diag.message


def test_chip_internal_move_not_flagged_a5():
    """Mat->L0 ``tile.move`` is genuinely cube-private and is never inspected.

    This is the half of the old memory-space rule that survives: PH001 visits
    only ``tile.load`` / ``tile.store``, so the two ``pl.move`` calls staging
    Mat->Left/Right in the fixture produce no diagnostic regardless of their
    innermost dim.
    """
    _activate_a5()
    program = _make_cube_matmul_program(8, pl.FP32)
    diags = _run_perf_hint_check(program)
    messages = [d.message for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint]
    assert not any("tile.move" in m for m in messages)


# ---------------------------------------------------------------------------
# Report clarity: (shape, dtype, target_memory) tuple (issue #1305 ask 5)
# ---------------------------------------------------------------------------


def test_message_includes_dtype_shape_memory_tuple_a5():
    """The hint echoes the (dtype[innermost], target_memory) tuple it evaluated."""
    _activate_a5()
    program = _make_load_program(16, pl.FP32)  # Vec load, 64B innermost
    diags = _run_perf_hint_check(program)
    perf_hints = [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint]
    assert len(perf_hints) >= 1
    msg = perf_hints[0].message
    # innermost = 16 elements of fp32; the fixture requests target_memory=Vec explicitly.
    assert "fp32[16]" in msg
    assert "target_memory=Vec" in msg


# ---------------------------------------------------------------------------
# Span propagation
# ---------------------------------------------------------------------------


def test_span_propagates_to_tile_op():
    """Diagnostic span resolves to a valid source location, not Span::unknown."""
    _activate_a5()
    program = _make_load_program(16, pl.FP32)
    diags = _run_perf_hint_check(program)
    perf_hints = [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint]
    assert len(perf_hints) >= 1
    # At least one diagnostic must have a real source location: the @pl.program
    # parser attaches spans to every Call expression.
    spans_with_loc = [d.span for d in perf_hints if d.span.is_valid()]
    assert len(spans_with_loc) >= 1


# ---------------------------------------------------------------------------
# Deduplication on transfer facts (issue #1305 ask 4)
# ---------------------------------------------------------------------------


def _site_of(diag: passes.Diagnostic) -> tuple[str, int, int, str]:
    """The (file, line, col, op) site a diagnostic is keyed on for dedup.

    The op name is the leading token of the message (``tile.load`` /
    ``tile.store``), before the ``B`` byte figure / count suffix that vary per
    transfer.
    """
    return (diag.span.filename, diag.span.begin_line, diag.span.begin_column, diag.message.split(" ", 1)[0])


def test_dedup_collapses_repeated_site_a3():
    """Post-pipeline hits are keyed one-per-source-site (issue #1305 ask 4).

    Runs the default pipeline over a loop kernel and asserts that both loads
    survive as their own hit and that no two surviving hits share a
    ``(file, line, col, op)`` site. This is the post-pipeline shape the unit
    fixtures above (single hand-built ops) cannot exercise, so the full pipeline
    is run here.

    Both assertions are needed. ``pl.range`` is not unrolled, so the entry load
    and the per-iteration load are one transfer each at two distinct source
    lines. A pass that coarsened synthesized ops onto the enclosing function's
    ``def`` span would merge them into a single bogus "2 occurrences at this
    source location" hit for two unrelated loads — which the uniqueness check
    alone would still accept, since one load site plus one store site is unique.
    """
    _activate_a3()
    rows, inner = 64, 64  # fp32 inner = 256B < 512B (a3) -> fires; rows give the loop distinct offsets

    @pl.program
    class LoopLoadProg:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[rows, inner], pl.FP32],
            out: pl.Out[pl.Tensor[[16, inner], pl.FP32]],
        ) -> pl.Tensor[[16, inner], pl.FP32]:
            acc: pl.Tile[[16, inner], pl.FP32] = pl.load(x, [0, 0], [16, inner], target_memory=pl.Mem.Vec)
            for i in pl.range(4):
                t: pl.Tile[[16, inner], pl.FP32] = pl.load(
                    x,
                    [i * 16, 0],
                    [16, inner],
                    target_memory=pl.Mem.Vec,
                )
                acc = pl.add(acc, t)
            return pl.store(acc, [0, 0], out)

    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    post = pm.run_passes(LoopLoadProg)
    diags = _run_perf_hint_check(post)
    perf_hints = [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint]
    assert len(perf_hints) >= 1
    assert all(d.hint_code == "PH001" for d in perf_hints)

    sites = [_site_of(d) for d in perf_hints]

    # Both loads must survive as their own hit. Site uniqueness alone cannot see
    # the coarsening regression: if the two loads shared the `def` span they
    # would dedup into ONE tile.load hit, and a set of one load site plus one
    # store site is still trivially unique. Pinning the count is what fails.
    load_sites = {site for site in sites if site[3] == "tile.load"}
    assert len(load_sites) == 2, f"expected the entry and per-iteration load as distinct sites: {sites}"

    # Dedup invariant: every surviving hit is at a distinct (file, line, col, op)
    # site — identical transfers were collapsed, not emitted repeatedly.
    assert len(sites) == len(set(sites)), f"hits not deduplicated per site: {sites}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
