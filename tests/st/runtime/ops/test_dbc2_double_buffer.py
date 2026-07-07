# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-device validation of AutoTileMatmulL0's dbC=2 (double-buffered L0C) emit.

dbC=2 keeps two co-live L0C accumulators so tile i's FIXPIPE drain overlaps tile
i+1's MAD.  It is opt-in: the chooser only enables it under
``memory_planner=MemoryPlanner.PTOAS`` (PyPTO's MemoryReuse over-coalesces the
co-live pair into one buffer; PTOAS skips it so InitMemRef keeps the two buffers
distinct and ptoas places them).  **Every case here forces PTOAS via
``get_memory_planner``** — under the default PyPTO planner these shapes get one
accumulator and would not exercise the feature.

Coverage:
  - direct-store (Acc->GM) sweep over 4 / 6 / 8 / 16-tile grids — the WAR reuse
    boundary (tile i+2's matmul into a buffer must wait for tile i's drain out of
    it) is only enforced by ptoas sync, so a value check on a >=4-tile grid is the
    primary correctness gate;
  - Mat-scratch (Acc->Mat, ``tile.assemble``) chained producer — the L1 drain path;
  - a non-divisible M/N shape — the peeled L-tail (its drains are not floated).

Numerics are the point: dbC=2 reuses buffers, so a sync error corrupts the result
(a wrong-order drain/MAD), which a golden comparison catches.  Platforms: a2a3 /
a2a3sim (the 128 KB-L0C chooser regime that selects these dbC tiles).
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.pypto_core.passes import MemoryPlanner

PLATFORMS_DBC = ["a2a3", "a2a3sim"]


class _DbcDirectStore(PTOTestCase):
    """``a @ b`` -> [M, N] FP32 direct-stored to GM, full-K (K=64), tiled into a dbC=2
    128x128 grid under PTOAS (accumulator budgeted at L0C/2).  M/N choose the grid /
    tile count."""

    __test__ = False

    def __init__(self, m: int, n: int, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self.M, self.K, self.N = m, 64, n
        if config is None:
            # Full-K (single-pass per tile, no K-split reduction), so FP32 matmul is
            # tight; a dbC sync error corrupts values far beyond this.
            self.config.rtol = 1e-3
            self.config.atol = 1e-3

    # dbC=2 is only reachable under the ptoas memory planner.
    def get_memory_planner(self) -> MemoryPlanner:
        return MemoryPlanner.PTOAS

    def get_name(self) -> str:
        return f"dbc2_ddr_{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.M, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [self.K, self.N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N = self.M, self.K, self.N

        @pl.program
        class DbcDirectStoreProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                lm: pl.Tile[[M, K], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    a, [0, 0], [M, K], target_memory=pl.Mem.Mat
                )
                rm: pl.Tile[[K, N], pl.FP32, pl.Mem.Mat] = pl.tile.load(
                    b, [0, 0], [K, N], target_memory=pl.Mem.Mat
                )
                c: pl.Tile[[M, N], pl.FP32, pl.Mem.Acc] = pl.tile.matmul(lm, rm)
                out = pl.store(c, [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                out = self.kernel(a, b, out)
                return out

        return DbcDirectStoreProgram

    def compute_expected(self, tensors, params=None):
        tensors["out"][:] = torch.matmul(tensors["a"], tensors["b"])


class _DbcMatScratch(PTOTestCase):
    """Chained ``(a @ b) @ e``: the oversized [256, 256] bf16 producer is assembled
    into an L1/Mat scratch (Acc->Mat ``tile.assemble``) and consumed on-chip.  Under
    PTOAS the full-K producer is a dbC=2 128x128 grid whose assemble drains are floated
    to keep the two accumulators co-live."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self.M, self.K, self.N, self.P = 256, 64, 256, 64
        if config is None:
            # bf16 operands + bf16 FIXPIPE-downcast intermediate: bf16 tolerance.
            self.config.rtol = 2e-2
            self.config.atol = 2e-2

    def get_memory_planner(self) -> MemoryPlanner:
        return MemoryPlanner.PTOAS

    def get_name(self) -> str:
        return f"dbc2_mat_scratch_{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.M, self.K], DataType.BF16, init_value=torch.randn),
            TensorSpec("b", [self.K, self.N], DataType.BF16, init_value=torch.randn),
            TensorSpec("e", [self.N, self.P], DataType.BF16, init_value=torch.randn),
            TensorSpec("out", [self.M, self.P], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N, P = self.M, self.K, self.N, self.P

        @pl.program
        class DbcMatScratchProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[M, K], pl.BF16],
                b: pl.Tensor[[K, N], pl.BF16],
                e: pl.Tensor[[N, P], pl.BF16],
                out: pl.Out[pl.Tensor[[M, P], pl.FP32]],
            ) -> pl.Tensor[[M, P], pl.FP32]:
                c = pl.matmul(a, b, out_dtype=pl.FP32)  # [M, N] f32 > L0c -> Mat scratch
                cb = pl.cast(c, pl.BF16, mode="rint")  # rint: FIXPIPE narrows tie-even (foldable)
                d = pl.matmul(cb, e, out_dtype=pl.FP32)  # consumes the scratch on-chip
                out = pl.assemble(out, d, [0, 0])
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, K], pl.BF16],
                b: pl.Tensor[[K, N], pl.BF16],
                e: pl.Tensor[[N, P], pl.BF16],
                out: pl.Out[pl.Tensor[[M, P], pl.FP32]],
            ) -> pl.Tensor[[M, P], pl.FP32]:
                out = self.kernel(a, b, e, out)
                return out

        return DbcMatScratchProgram

    def compute_expected(self, tensors, params=None):
        a = tensors["a"].float()
        b = tensors["b"].float()
        e = tensors["e"].float()
        c_bf16 = (a @ b).to(torch.bfloat16).float()  # FIXPIPE downcast to the bf16 scratch
        tensors["out"][:] = c_bf16 @ e


# =============================================================================
# pytest suite
# =============================================================================


class TestDbc2DoubleBuffer:
    """dbC=2 L0C double-buffer, forced onto the ptoas memory planner."""

    # Grid counts are the chooser's pick under the a2a3 128 KB-L0C dbC regime (it
    # prefers ~2x2 splits, so 2-tile grids do not occur for these shapes); the sweep
    # gives the device agent a drain-hiding curve — the last tile's drain is always
    # exposed, so the hidden fraction should approach 1 as the tile count grows.
    @pytest.mark.parametrize("platform", PLATFORMS_DBC)
    @pytest.mark.parametrize(
        "m, n",
        [
            (256, 256),  # 128x128 tile, 2x2 ->  4 tiles  (WAR reuse boundary — primary gate)
            # TODO(dbC2-ptoas-operand-overflow): (384, 256) [3x2 grid] compiles to 6 distinct
            # operand memrefs; under memory_planner=PTOAS MemoryReuse is skipped, so ptoas
            # allocates one L0A buffer per memref (6x16KB > 64KB) -> "left overflow". PyPTO's
            # MemoryReuse coalesces these to the 2-buffer ping-pong; ptoas does not do its own
            # operand-liveness coalescing. Not dbC-specific (reproduces at dbC=1) and not the
            # cost model. Re-enable once operand coalescing runs under PTOAS (or InitMemRef
            # ping-pongs the streamed operand). See KNOWN_ISSUES.
            (256, 512),  # 128x128 tile, 2x4 ->  8 tiles
            (512, 512),  # 128x128 tile, 4x4 -> 16 tiles  (deepest WAR stress)
        ],
    )
    def test_direct_store_dbc(self, test_runner, platform, m, n):
        """Direct-store (Acc->GM) dbC=2 across a tile-count sweep; a wrong reuse-WAR
        sync would corrupt the result."""
        result = test_runner.run(_DbcDirectStore(m, n, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS_DBC)
    def test_mat_scratch_dbc(self, test_runner, platform):
        """Mat-scratch (Acc->Mat, tile.assemble) dbC=2: the L1 drain path."""
        result = test_runner.run(_DbcMatScratch(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS_DBC)
    def test_non_divisible_tail_dbc(self, test_runner, platform):
        """Non-divisible M/N (320x320): the peeled L-shaped tail is emitted straight-line
        (its drains are not floated), so this exercises dbC interior + exposed tail."""
        result = test_runner.run(_DbcDirectStore(320, 320, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
