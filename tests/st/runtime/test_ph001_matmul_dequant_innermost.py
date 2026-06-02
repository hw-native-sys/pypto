# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-board system test for issue #1305: PH001 ``TileInnermostDimGranularity``
on a real cube-matmul -> vec-dequant kernel.

The issue reports that, on a moderately complex kernel with a cube ``pl.at``
followed by a vec ``pl.at``, PH001 floods the channel with noise: it flags the
cube-private L0/L1 fragments (which never traverse L2) with a misleading "L2
cache line" message, repeats the same site once per loop-unroll / fragment
instance, and prints a bare byte count that can't be reconciled against the IR.

This is the canonical repro from the issue: a ``pl.matmul`` (the cube ``pl.at``)
whose result is spilled to GM and reloaded by a ``pl.cast`` dequant (the vec
``pl.at``). After the default a2a3 pipeline runs, the post-pipeline IR contains:

* cube-private ``tile.load`` of the matmul operands into ``Mem.Mat`` with a
  128 B innermost dim (``BF16[64]``), plus the ``Mem.Left`` / ``Mem.Right`` /
  ``Mem.Acc`` matmul fragments — all below the 512 B a2a3 recommendation, and
  all of which the *old* verifier flagged with the L2-cache-line message; and
* the genuine GM<->Vec spill/reload transfers that *do* traverse L2.

Two complementary checks guard the fix:

1. ``test_matmul_dequant_onboard`` runs the kernel on real hardware and
   validates the numerical result against the PyTorch golden. The PH001 change
   is diagnostic-only, so codegen and on-device correctness must be unaffected.
   Contrary to the PR's original claim, CI *does* run device ST — this is that
   coverage.
2. ``test_ph001_skips_cube_private_transfers`` compiles the same kernel through
   the full default a2a3 pipeline and asserts the post-pipeline PH001 stream now
   matches the issue's asks: no cube-private hit (ask 1), every hit carries the
   ``(dtype[innermost], target_memory)`` tuple (ask 5), and hits are
   deduplicated per source site (ask 4).

**Why these inputs are exact.** ``a[m, :] = m + 1`` (row-constant, distinct per
row) and ``b`` is all-ones, so ``acc[m, :] = K * (m + 1)`` accumulated in FP32,
then dequantized by ``0.5`` to ``32 * (m + 1)`` (a multiple of 32, <= 1024 for
M=32, exactly representable in BF16). The cast is therefore bit-exact and the
golden matches with zero tolerance slack.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto import backend, passes
from pypto.backend import BackendType
from pypto.ir import python_print
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

# Shapes are chosen so every GM<->Vec transfer's innermost dim sits below the
# 512 B a2a3 L2 cache line (N=64: FP32 -> 256 B, BF16 -> 128 B), which is what
# makes PH001 fire at all, while the cube operand loads (BF16[64] -> 128 B) are
# what the old verifier wrongly flagged.
M, K, N = 32, 64, 64
DEQUANT_SCALE = 0.5


@pl.program
class MatmulDequantProgram:
    """Cube matmul -> GM spill -> vec dequant reload, fused in one CORE_GROUP scope."""

    @pl.function(type=pl.FunctionType.Opaque)
    def matmul_dequant(
        self,
        a: pl.Tensor[[M, K], pl.BF16],
        b: pl.Tensor[[K, N], pl.BF16],
        out: pl.Out[pl.Tensor[[M, N], pl.BF16]],
    ) -> pl.Tensor[[M, N], pl.BF16]:
        acc = pl.create_tensor([M, N], dtype=pl.FP32)
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.LEFT_RIGHT)]):
            acc[:, :] = pl.matmul(a, b, out_dtype=pl.FP32)  # cube pl.at -> L0/L1 fragments
            out[:, :] = pl.cast(  # vec pl.at -> GM reload of acc, then dequant + cast
                pl.mul(acc[:, :], DEQUANT_SCALE), target_type=pl.BF16, mode="rint"
            )
        return out


def _make_a() -> torch.Tensor:
    """``[M, K]`` row-constant BF16: ``a[m, j] = m + 1`` (distinct per row, exact)."""
    rows = torch.arange(M, dtype=torch.float32).reshape(M, 1) + 1.0
    return rows.expand(M, K).contiguous().to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Check 1: on-board numerical correctness (real device ST)
# ---------------------------------------------------------------------------


class MatmulDequantTestCase(PTOTestCase):
    """Issue #1305 repro kernel: cube matmul -> vec dequant, run end-to-end."""

    def get_name(self) -> str:
        return "ph001_matmul_dequant_1305"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, K], DataType.BF16, init_value=_make_a),
            TensorSpec("b", [K, N], DataType.BF16, init_value=lambda: torch.ones(K, N, dtype=torch.bfloat16)),
            TensorSpec("out", [M, N], DataType.BF16, is_output=True),
        ]

    def get_program(self) -> Any:
        return MatmulDequantProgram

    def compute_expected(self, tensors, params=None):
        acc = tensors["a"].float() @ tensors["b"].float()
        tensors["out"][:] = (acc * DEQUANT_SCALE).to(tensors["out"].dtype)


class TestMatmulDequantOnboard:
    """Cube matmul -> vec dequant fused in one CORE_GROUP scope, run on device."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_matmul_dequant_onboard(self, test_runner, platform):
        result = test_runner.run(MatmulDequantTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


# ---------------------------------------------------------------------------
# Check 2: PH001 behaves on the full-pipeline IR of the real kernel
# ---------------------------------------------------------------------------

# Cube-private memory spaces (L0/L1) that never traverse L2; PH001's
# L2-cache-line threshold must not flag transfers landing in any of these.
_CUBE_MEMORY_TAGS = ("target_memory=Mat", "target_memory=Left", "target_memory=Right", "target_memory=Acc")


def _compile_and_collect_ph001(backend_type: BackendType) -> tuple[str, list[passes.Diagnostic]]:
    """Run the default pipeline for *backend_type* and return (post-pipeline IR
    text, PH001 perf-hint diagnostics) for ``MatmulDequantProgram``."""
    backend.reset_for_testing()
    backend.set_backend_type(backend_type)
    try:
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        with passes.PassContext([], verification_level=passes.VerificationLevel.NONE):
            post = pm.run_passes(MatmulDequantProgram)
            checks = passes.DiagnosticCheckSet()
            checks.insert(passes.DiagnosticCheck.TileInnermostDimGranularity)
            diags = passes.DiagnosticCheckRegistry.run_checks(
                checks, passes.DiagnosticPhase.POST_PIPELINE, post
            )
    finally:
        backend.reset_for_testing()
    ph = [d for d in diags if d.severity == passes.DiagnosticSeverity.PerfHint]
    return python_print(post), ph


class TestPH001OnFullPipelineIR:
    """Issue #1305 fix asserted on the real post-pipeline IR (no device needed)."""

    def test_ph001_skips_cube_private_transfers(self):
        """The repro fires PH001, but only on GM<->Vec transfers; cube-private
        L0/L1 fragments are skipped, every hit carries the evaluated tuple, and
        hits are deduplicated per source site."""
        ir_text, ph = _compile_and_collect_ph001(BackendType.Ascend910B)

        # The kernel must actually exercise the cube path, otherwise "no cube
        # hit" would be vacuously true. The matmul operands are loaded into
        # Mem.Mat with a 128 B innermost dim (BF16[64], below the 512 B a2a3
        # recommendation) — exactly the transfer the old verifier mis-flagged.
        assert "target_memory=pl.Mem.Mat" in ir_text, (
            "expected cube-private Mem.Mat loads in the post-pipeline IR; "
            "the kernel no longer exercises the cube path"
        )

        # It is a genuine repro: the surviving GM<->Vec transfers do fire PH001.
        assert len(ph) >= 1, "expected at least one PH001 hit on the GM<->Vec spill/reload"
        assert all(d.hint_code == "PH001" for d in ph)
        assert all(d.rule_name == "TileInnermostDimGranularity" for d in ph)

        for d in ph:
            msg = d.message
            # Ask 1: never flag cube-private (L0/L1) transfers — they don't
            # touch L2, so the L2-cache-line threshold is meaningless for them.
            assert not any(tag in msg for tag in _CUBE_MEMORY_TAGS), (
                f"PH001 must not flag cube-private transfers, got: {msg}"
            )
            assert "target_memory=Vec" in msg, f"every surviving hit is a GM<->Vec transfer, got: {msg}"
            # Ask 5: echo the evaluated (dtype[innermost], target_memory) tuple
            # so the reported byte figure can be reconciled against the IR.
            assert "tile " in msg and "target_memory=" in msg, f"hit must carry the evaluated tuple: {msg}"

        # Ask 4: dedup by source site + transfer facts. The message encodes
        # (op, innermost bytes, dtype[innermost], target_memory), so identical
        # (file, line, col, message) tuples would be undeduplicated repeats.
        keys = [(d.span.filename, d.span.begin_line, d.span.begin_column, d.message) for d in ph]
        assert len(keys) == len(set(keys)), f"PH001 hits must be deduplicated per site, got {keys}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
