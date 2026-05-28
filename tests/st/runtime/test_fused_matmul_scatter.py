# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end test for issue #1564: cube matmul -> per-row scatter to distinct
GM rows fused in one ``CORE_GROUP`` scope.

A single ``pl.matmul`` writes its FP32 result into a scope-local
``create_tensor`` (``kv_final``); a per-row loop then casts each row and
scatters it to a **distinct, strided** GM row of ``cache`` (row ``b * CACHE``).
On Ascend910B this AIV scatter loop is lowered through SplitVectorKernel's
no-split dual-AIV dispatch path. Before the fix, lane 1 was replayed as an empty
no-op (``valid_shape=[0, 0]`` + dropped store), so the rows it owned came back
all-zero. With iteration-space splitting, each of the two AIV sub-blocks runs a
disjoint half of the loop and writes real data, so every scattered row is
populated.

**Why these inputs are discriminating.** ``x[b, :] = b + 1`` (row-constant,
distinct per row) and ``w`` is all-ones, so ``kv_final[b, :] = 64 * (b + 1)`` —
distinct per row and exactly representable in BF16 (≤ 4 significant bits), so the
cast is bit-exact and the golden matches without tolerance slack. The unwritten
rows of ``cache`` stay zero. If lane 1's rows (``b = 8..15``) regress to zero,
they mismatch the nonzero golden immediately.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

# Shapes match the issue repro exactly.
B = 16
H = 64
CACHE = 256


def _make_x() -> torch.Tensor:
    """``[B, H]`` row-constant BF16: ``x[b, j] = b + 1`` (distinct per row, exact)."""
    rows = torch.arange(B, dtype=torch.float32).reshape(B, 1) + 1.0
    return rows.expand(B, H).contiguous().to(torch.bfloat16)


def _make_kv_final() -> torch.Tensor:
    """``[B, H]`` row-constant FP32: ``kv_final[b, j] = b + 1`` (distinct per row)."""
    rows = torch.arange(B, dtype=torch.float32).reshape(B, 1) + 1.0
    return rows.expand(B, H).contiguous()


@pl.program
class FusedMatmulScatterProgram:
    """matmul -> store kv_final -> per-row cast + scatter, all in one CORE_GROUP scope."""

    @pl.function(type=pl.FunctionType.Opaque)
    def fused(
        self,
        x: pl.Tensor[[B, H], pl.BF16],
        w: pl.Tensor[[H, H], pl.BF16],
        cache: pl.Out[pl.Tensor[[B * CACHE, H], pl.BF16]],
    ) -> pl.Tensor[[B * CACHE, H], pl.BF16]:
        kv_final = pl.create_tensor([B, H], dtype=pl.FP32)
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.LEFT_RIGHT)]):
            kv_final[:, :] = pl.matmul(x, w, out_dtype=pl.FP32)  # cube matmul, whole-tile store
            for b in pl.range(B):  # per-row scatter to distinct cache rows
                cache[b * CACHE : b * CACHE + 1, :] = pl.cast(
                    kv_final[b : b + 1, :], target_type=pl.BF16, mode="rint"
                )
        return cache


@pl.program
class FusedMatmulScatterNoSplitProgram:
    """Exact #1564 repro: identical to ``FusedMatmulScatterProgram`` but the scope
    requests ``pl.split(pl.SplitMode.NONE)``. This is the **failing** variant.

    On Ascend910B this mixed (cube + vector) kernel cannot be split, so
    ``ExpandMixedKernel`` tags the AIV function ``dual_aiv_dispatch=True`` and
    ``SplitVectorKernel`` takes the no-split dual-AIV path: it wraps the body in
    ``if subblock_idx == 0: <original> else: <replay>`` where the replay keeps the
    AIC<->AIV handshake but **drops every tile.store** and forces tile producers
    to ``valid_shape=[0, 0]``. That model assumes lane 0 alone can do all the real
    work — true for a wholesale-consumed cube result, but false here: the per-row
    scatter writes B distinct GM rows and the cube result is partitioned across the
    two AIV sub-blocks. The rows owned by sub-block 1 are never scattered and come
    back all-zero, so the test fails. ``FusedMatmulScatterProgram``
    (``SplitMode.LEFT_RIGHT``) is the passing baseline for the same compute."""

    @pl.function(type=pl.FunctionType.Opaque)
    def fused(
        self,
        x: pl.Tensor[[B, H], pl.BF16],
        w: pl.Tensor[[H, H], pl.BF16],
        cache: pl.Out[pl.Tensor[[B * CACHE, H], pl.BF16]],
    ) -> pl.Tensor[[B * CACHE, H], pl.BF16]:
        kv_final = pl.create_tensor([B, H], dtype=pl.FP32)
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.NONE)]):
            kv_final[:, :] = pl.matmul(x, w, out_dtype=pl.FP32)  # cube matmul, whole-tile store
            for b in pl.range(B):  # per-row scatter to distinct cache rows
                cache[b * CACHE : b * CACHE + 1, :] = pl.cast(
                    kv_final[b : b + 1, :], target_type=pl.BF16, mode="rint"
                )
        return cache


@pl.program
class ScatterOnlyProgram:
    """Cube-free isolation of #1564: kv_final is a direct input (no matmul), so
    the only work is the AIV per-row scatter under the no-split dual-AIV path.
    A PASS here while FusedMatmulScatterProgram FAILS points at the cube->vec
    kv_final handoff (RAW sync); a FAIL here points at the AIV scatter itself."""

    @pl.function(type=pl.FunctionType.Opaque)
    def fused(
        self,
        kv_final: pl.Tensor[[B, H], pl.FP32],
        cache: pl.Out[pl.Tensor[[B * CACHE, H], pl.BF16]],
    ) -> pl.Tensor[[B * CACHE, H], pl.BF16]:
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.NONE)]):
            for b in pl.range(B):  # per-row scatter to distinct cache rows
                cache[b * CACHE : b * CACHE + 1, :] = pl.cast(
                    kv_final[b : b + 1, :], target_type=pl.BF16, mode="rint"
                )
        return cache


class FusedMatmulScatterTestCase(PTOTestCase):
    """Issue #1564 repro: matmul -> per-row scatter to distinct GM rows."""

    def get_name(self) -> str:
        return "fused_matmul_scatter_1564"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [B, H], DataType.BF16, init_value=_make_x),
            TensorSpec("w", [H, H], DataType.BF16, init_value=lambda: torch.ones(H, H, dtype=torch.bfloat16)),
            TensorSpec("cache", [B * CACHE, H], DataType.BF16, is_output=True),
        ]

    def get_program(self) -> Any:
        return FusedMatmulScatterProgram

    def compute_expected(self, tensors, params=None):
        out = (tensors["x"].float() @ tensors["w"].float()).to(tensors["cache"].dtype)
        cache = tensors["cache"]
        cache.zero_()  # unwritten rows stay zero (matches device output init)
        for b in range(B):
            cache[b * CACHE] = out[b]


class FusedMatmulScatterNoSplitTestCase(FusedMatmulScatterTestCase):
    """Issue #1564 exact repro under ``SplitMode.NONE`` — the failing variant.

    Same tensors and golden as ``FusedMatmulScatterTestCase``; only the program
    (and thus the requested split mode) differs."""

    def get_name(self) -> str:
        return "fused_matmul_scatter_1564_nosplit"

    def get_program(self) -> Any:
        return FusedMatmulScatterNoSplitProgram


class ScatterOnlyTestCase(PTOTestCase):
    """Cube-free diagnostic: kv_final is a direct input, only the AIV scatter runs."""

    def get_name(self) -> str:
        return "scatter_only_1564_diagnostic"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("kv_final", [B, H], DataType.FP32, init_value=_make_kv_final),
            TensorSpec("cache", [B * CACHE, H], DataType.BF16, is_output=True),
        ]

    def get_program(self) -> Any:
        return ScatterOnlyProgram

    def compute_expected(self, tensors, params=None):
        kv = tensors["kv_final"].to(tensors["cache"].dtype)
        cache = tensors["cache"]
        cache.zero_()  # unwritten rows stay zero (matches device output init)
        for b in range(B):
            cache[b * CACHE] = kv[b]


class TestFusedMatmulScatterCoreGroup:
    """matmul -> per-row scatter to distinct GM rows fused in one CORE_GROUP scope."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_fused_matmul_scatter(self, test_runner, platform):
        result = test_runner.run(FusedMatmulScatterTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


class TestFusedMatmulScatterNoSplit:
    """Issue #1564 exact repro: matmul -> per-row scatter under SplitMode.NONE."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_fused_matmul_scatter_nosplit(self, test_runner, platform):
        result = test_runner.run(FusedMatmulScatterNoSplitTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


class TestScatterOnlyDiagnostic:
    """Cube-free isolation: per-row scatter of a direct input (no matmul)."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_scatter_only(self, test_runner, platform):
        result = test_runner.run(ScatterOnlyTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
