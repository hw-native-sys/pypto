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
On Ascend910B this lowers to an AIC cube kernel (the matmul, storing ``kv_final``
to GM) and an AIV vector kernel (the scatter, reading ``kv_final`` back from GM).

Two independent codegen bugs corrupted the output (both fixed in this change):

1. **Scratch aliased onto the output.** ``FuseCreateAssembleToSlice`` resolved
   the windowed group call's return to the first ``Out``/``InOut`` param, which
   was the ``InOut`` scratch ``kv_final``, and fused its ``tensor.create`` into a
   ``tensor.slice`` of ``cache``. The FP32 matmul result then landed on top of
   the BF16 output. Fixed by matching the return root by shape+dtype.
2. **Missing cube->vector fence.** ``ExpandMixedKernel`` only fenced a GM
   store/load pair in the same body, so the top-level cube store feeding the
   in-loop vector load got no tpush/tpop. The AIV raced ahead and read
   ``kv_final`` before the matmul store landed, so the first scatter iterations
   (``b = 0, 1`` -> rows 0 and 256) came back zero. Fixed by hoisting the fence
   tpop before the consumer loop.

**Why these inputs are discriminating.** ``x[b, :] = b + 1`` (row-constant,
distinct per row) and ``w`` is all-ones, so ``kv_final[b, :] = 64 * (b + 1)`` —
distinct per row and exactly representable in BF16 (≤ 4 significant bits), so the
cast is bit-exact and the golden matches without tolerance slack. The unwritten
rows of ``cache`` stay zero. Any row that regresses (a raced cube->vec read, or
an FP32 scratch byte scribbled over the output) mismatches the golden at once.
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


def _make_block_table() -> torch.Tensor:
    """``[B]`` INT32 block table: ``block_table[b] = b`` (block id for row ``b``).

    The kernel reads this block id at runtime and computes the scatter row as
    ``block_id * CACHE`` (the paged ``block_id * block_size`` idiom), so the
    store lowers to a *dynamic-address* ``pto.tstore``. The ``* CACHE`` keeps the
    offset ``index``-typed, so codegen is clean — matching the issue (compiles
    clean, deadlocks at runtime). The golden output is identical to the
    static-address variants; only the addressing mode (and codegen path) differ."""
    return torch.arange(B, dtype=torch.int32).contiguous()


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
    requests ``pl.split(pl.SplitMode.NONE)``. This was the originally-reported
    failing variant.

    The failure was never split-mode-specific: ``NONE`` and ``LEFT_RIGHT`` lower
    to the same orchestration and the same cube->vector GM handoff, so both hit
    the two bugs described in the module docstring — the FP32 scratch ``kv_final``
    aliased onto ``cache`` (``FuseCreateAssembleToSlice``) and the unfenced
    cube->vector read race (``ExpandMixedKernel``). Both variants are kept as
    regression coverage; both must pass."""

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


@pl.program
class FusedMatmulDynScatterProgram:
    """Issue #1592: like ``FusedMatmulScatterProgram`` but the scatter row index
    is **read from a GM block table** (``cache_row = pl.read(block_table, [b])``)
    instead of the affine ``b * CACHE``.

    This makes the per-row store a **dynamic-address** ``pto.tstore`` on the vec
    side of the cube+vec mix scope. Per the #1592 isolation matrix this is the
    variant that compiles clean but deadlocks at runtime (``507018``); the
    static-address variants above run. ``SplitMode.UP_DOWN`` is the mode named in
    the issue report; it exercises the ``ProcessFunction`` split path."""

    @pl.function(type=pl.FunctionType.Opaque)
    def fused(
        self,
        x: pl.Tensor[[B, H], pl.BF16],
        w: pl.Tensor[[H, H], pl.BF16],
        block_table: pl.Tensor[[B], pl.INT32],
        cache: pl.Out[pl.Tensor[[B * CACHE, H], pl.BF16]],
    ) -> pl.Tensor[[B * CACHE, H], pl.BF16]:
        kv_final = pl.create_tensor([B, H], dtype=pl.FP32)
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
            kv_final[:, :] = pl.matmul(x, w, out_dtype=pl.FP32)  # cube matmul, whole-tile store
            for b in pl.range(B):  # per-row scatter to a data-dependent cache row
                block_id: pl.Scalar[pl.INT32] = pl.read(block_table, [b])
                cache_row = block_id * CACHE  # block id -> row; keeps offset index-typed
                cache[cache_row : cache_row + 1, :] = pl.cast(
                    kv_final[b : b + 1, :], target_type=pl.BF16, mode="rint"
                )
        return cache


@pl.program
class FusedMatmulDynScatterNoSplitProgram:
    """Issue #1592 under ``SplitMode.NONE`` — the no-split dual-AIV dispatch path
    (``ProcessNoSplitDualAivFunction``). Same dynamic-address scatter and golden
    as ``FusedMatmulDynScatterProgram``; only the requested split mode differs."""

    @pl.function(type=pl.FunctionType.Opaque)
    def fused(
        self,
        x: pl.Tensor[[B, H], pl.BF16],
        w: pl.Tensor[[H, H], pl.BF16],
        block_table: pl.Tensor[[B], pl.INT32],
        cache: pl.Out[pl.Tensor[[B * CACHE, H], pl.BF16]],
    ) -> pl.Tensor[[B * CACHE, H], pl.BF16]:
        kv_final = pl.create_tensor([B, H], dtype=pl.FP32)
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.NONE)]):
            kv_final[:, :] = pl.matmul(x, w, out_dtype=pl.FP32)  # cube matmul, whole-tile store
            for b in pl.range(B):  # per-row scatter to a data-dependent cache row
                block_id: pl.Scalar[pl.INT32] = pl.read(block_table, [b])
                cache_row = block_id * CACHE  # block id -> row; keeps offset index-typed
                cache[cache_row : cache_row + 1, :] = pl.cast(
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


class FusedMatmulDynScatterTestCase(PTOTestCase):
    """Issue #1592: matmul -> per-row scatter to a **block-table-indirected** GM
    row (dynamic address) under ``SplitMode.UP_DOWN``."""

    def get_name(self) -> str:
        return "fused_matmul_dyn_scatter_1592"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [B, H], DataType.BF16, init_value=_make_x),
            TensorSpec("w", [H, H], DataType.BF16, init_value=lambda: torch.ones(H, H, dtype=torch.bfloat16)),
            TensorSpec("block_table", [B], DataType.INT32, init_value=_make_block_table),
            TensorSpec("cache", [B * CACHE, H], DataType.BF16, is_output=True),
        ]

    def get_program(self) -> Any:
        return FusedMatmulDynScatterProgram

    def compute_expected(self, tensors, params=None):
        out = (tensors["x"].float() @ tensors["w"].float()).to(tensors["cache"].dtype)
        block_table = tensors["block_table"]
        cache = tensors["cache"]
        cache.zero_()  # unwritten rows stay zero (matches device output init)
        for b in range(B):
            cache[int(block_table[b]) * CACHE] = out[b]


class FusedMatmulDynScatterNoSplitTestCase(FusedMatmulDynScatterTestCase):
    """Issue #1592 under ``SplitMode.NONE`` (no-split dual-AIV path). Same tensors
    and golden as ``FusedMatmulDynScatterTestCase``; only the program differs."""

    def get_name(self) -> str:
        return "fused_matmul_dyn_scatter_1592_nosplit"

    def get_program(self) -> Any:
        return FusedMatmulDynScatterNoSplitProgram


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


class TestFusedMatmulDynScatter:
    """Issue #1592: matmul -> per-row scatter to a block-table-indirected
    (dynamic-address) GM row, fused in one CORE_GROUP scope."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_fused_matmul_dyn_scatter(self, test_runner, platform):
        result = test_runner.run(FusedMatmulDynScatterTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


class TestFusedMatmulDynScatterNoSplit:
    """Issue #1592 under SplitMode.NONE (no-split dual-AIV dispatch path)."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_fused_matmul_dyn_scatter_nosplit(self, test_runner, platform):
        result = test_runner.run(FusedMatmulDynScatterNoSplitTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
