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


@pytest.fixture(autouse=True)
def reset_backend_around_test():
    """Each test owns its backend selection; reset before and after."""
    backend.reset_for_testing()
    yield
    backend.reset_for_testing()


def _run_perf_hint_check(program: ir.Program) -> list[passes.Diagnostic]:
    """Run only the TileInnermostDimGranularity check and return its diagnostics.

    The verifier reads the active backend from PassContext, so callers must
    set up backend + context before invoking this helper.
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
            t: pl.Tile[[rows, innermost], dtype] = pl.load(x, [0, 0], [rows, innermost])
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
    with passes.PassContext([], verification_level=passes.VerificationLevel.NONE):
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
    with passes.PassContext([], verification_level=passes.VerificationLevel.NONE):
        diags = _run_perf_hint_check(program)
    assert [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint] == []


def test_at_threshold_a5_silent():
    """A5 backend, FP32 [16, 32] → exactly 128B innermost → silent (>= recommended)."""
    _activate_a5()
    program = _make_load_program(32, pl.FP32)
    with passes.PassContext([], verification_level=passes.VerificationLevel.NONE):
        diags = _run_perf_hint_check(program)
    assert [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint] == []


def test_below_threshold_a3_emits():
    """A3 backend (512B threshold), FP32 [16, 32] → 128B innermost → fires."""
    _activate_a3()
    program = _make_load_program(32, pl.FP32)
    with passes.PassContext([], verification_level=passes.VerificationLevel.NONE):
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
    with passes.PassContext([], verification_level=passes.VerificationLevel.NONE):
        diags = _run_perf_hint_check(program)
    assert [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint] == []


# ---------------------------------------------------------------------------
# Dtype affects byte size
# ---------------------------------------------------------------------------


def test_dtype_int8_silent_at_128_elements_a5():
    """A5: INT8 with innermost=128 → 128B → silent (boundary)."""
    _activate_a5()
    program = _make_load_program(128, pl.INT8)
    with passes.PassContext([], verification_level=passes.VerificationLevel.NONE):
        diags = _run_perf_hint_check(program)
    assert [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint] == []


def test_dtype_int8_below_threshold_a5_emits():
    """A5: INT8 with innermost=64 → 64B → fires."""
    _activate_a5()
    program = _make_load_program(64, pl.INT8)
    with passes.PassContext([], verification_level=passes.VerificationLevel.NONE):
        diags = _run_perf_hint_check(program)
    perf_hints = [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint]
    assert len(perf_hints) >= 1
    assert "64B" in perf_hints[0].message


def test_dtype_fp16_threshold_a5_silent():
    """A5: FP16 with innermost=64 → 128B → silent."""
    _activate_a5()
    program = _make_load_program(64, pl.FP16)
    with passes.PassContext([], verification_level=passes.VerificationLevel.NONE):
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
    with passes.PassContext([], verification_level=passes.VerificationLevel.NONE):
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
    with passes.PassContext(
        [], verification_level=passes.VerificationLevel.NONE, disabled_diagnostics=disabled
    ):
        all_checks = passes.DiagnosticCheckRegistry.get_all_checks()
        effective = all_checks.difference(disabled)
        diags = passes.DiagnosticCheckRegistry.run_checks(
            effective, passes.DiagnosticPhase.POST_PIPELINE, program
        )
    assert [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint] == []


# ---------------------------------------------------------------------------
# Memory-space awareness (issue #1305 ask 1)
# ---------------------------------------------------------------------------


def _make_cube_matmul_program(k: int, dtype) -> ir.Program:
    """Build an InCore matmul kernel whose tiles live in cube-private L0/L1.

    A is loaded into Mat (L1) with a small inner ``k`` (below threshold), B into
    Mat, both moved to Left/Right (L0A/L0B), multiplied into Acc (L0C), then
    stored. The A-Mat load's innermost dim is below threshold but it never
    traverses L2, so PH001 must not fire on it. ``n`` is chosen so the final
    store's innermost dim meets the threshold and stays silent on its own.
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


def test_cube_memory_space_not_flagged_a5():
    """A5: cube-private (Mat/Left/Right/Acc) transfers are never flagged.

    Mat/Left/Right/Acc are cube-private L0/L1 buffers that never traverse L2, so
    the L2-cache-line threshold does not apply and PH001 must stay silent even
    though the A-side tiles have an 8-element (32B) innermost dim, well below the
    128B A5 recommendation.
    """
    _activate_a5()
    program = _make_cube_matmul_program(8, pl.FP32)  # A-Mat innermost = 32B (< 128B)
    with passes.PassContext([], verification_level=passes.VerificationLevel.NONE):
        diags = _run_perf_hint_check(program)
    perf_hints = [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint]
    assert perf_hints == []


# ---------------------------------------------------------------------------
# Report clarity: (shape, dtype, target_memory) tuple (issue #1305 ask 5)
# ---------------------------------------------------------------------------


def test_message_includes_dtype_shape_memory_tuple_a5():
    """The hint echoes the (dtype[innermost], target_memory) tuple it evaluated."""
    _activate_a5()
    program = _make_load_program(16, pl.FP32)  # Vec load, 64B innermost
    with passes.PassContext([], verification_level=passes.VerificationLevel.NONE):
        diags = _run_perf_hint_check(program)
    perf_hints = [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint]
    assert len(perf_hints) >= 1
    msg = perf_hints[0].message
    # innermost = 16 elements of fp32, default target_memory = Vec.
    assert "fp32[16]" in msg
    assert "target_memory=Vec" in msg


# ---------------------------------------------------------------------------
# Span propagation
# ---------------------------------------------------------------------------


def test_span_propagates_to_tile_op():
    """Diagnostic span resolves to a valid source location, not Span::unknown."""
    _activate_a5()
    program = _make_load_program(16, pl.FP32)
    with passes.PassContext([], verification_level=passes.VerificationLevel.NONE):
        diags = _run_perf_hint_check(program)
    perf_hints = [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint]
    assert len(perf_hints) >= 1
    # At least one diagnostic must have a real source location: the @pl.program
    # parser attaches spans to every Call expression.
    spans_with_loc = [d.span for d in perf_hints if d.span.is_valid()]
    assert len(spans_with_loc) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
