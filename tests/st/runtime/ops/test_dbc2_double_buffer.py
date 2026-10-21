# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Golden checks on historical AutoTile shapes with planner-specific dbC policy.

The legacy opt-in is exercised under every planner. These workloads retain
ordinary tiling and operand pipelines. PYPTO and DSA-RP select single-C;
PTOAS retains its previous automatic dbC selection.
Explicit multi-buffer emission is covered by codegen and lowering unit tests.
The historical filename is retained for existing targeted test invocations."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto import backend as _backend
from pypto import ir, passes
from pypto.backend import BackendType
from pypto.pypto_core import passes as _core_passes
from pypto.pypto_core.passes import MemoryPlanner

PLATFORMS_DBC = ["a2a3", "a2a3sim"]


def _choose_a2a3_dbc(m: int, k: int, n: int, *, planner: MemoryPlanner, bytes_a: int, bytes_b: int):
    """Mirror the planner-specific 910B choice for a historical dbC problem."""
    cfg = _core_passes.l0_tile_chooser.L0TileConfig()
    cfg.M, cfg.K, cfg.N = m, k, n
    cfg.l0a_bytes = cfg.l0b_bytes = 64 * 1024
    cfg.l0c_bytes = 128 * 1024
    cfg.bytes_a, cfg.bytes_b, cfg.bytes_c = bytes_a, bytes_b, 4
    cfg.allow_a_stationary = True
    cfg.allow_b_stationary = True
    cfg.allow_double_buffer_c = planner == MemoryPlanner.PTOAS
    cfg.allow_k_boundary = True
    return _core_passes.l0_tile_chooser.choose_l0_tile(cfg)


def _choose_a2a3_fp32_dbc(m: int, n: int, *, planner: MemoryPlanner):
    return _choose_a2a3_dbc(m, 64, n, planner=planner, bytes_a=4, bytes_b=4)


def _printed_after_auto_tile(test_case: PTOTestCase, planner: MemoryPlanner) -> str:
    """Lower a device case through AutoTile to check its planner policy."""
    _backend.reset_for_testing()
    _backend.set_backend_type(BackendType.Ascend910)
    program = test_case.get_program()
    with passes.PassContext(
        [],
        memory_planner=planner,
        enable_pypto_l0c_double_buffer=test_case.get_enable_pypto_l0c_double_buffer() or False,
    ):
        for make_pass in (
            passes.convert_to_ssa,
            passes.convert_tensor_to_tile_ops,
            passes.lower_composite_ops,
            passes.flatten_tile_nd_to_2d,
            passes.auto_tile_matmul_l0,
        ):
            program = make_pass()(program)
    return ir.python_print(program)


class _DbcDirectStore(PTOTestCase):
    """Direct-store matmul on shapes formerly used for automatic dbC."""

    __test__ = False

    def __init__(
        self,
        m: int,
        n: int,
        *,
        planner: MemoryPlanner = MemoryPlanner.PTOAS,
        platform: str | None = None,
        config=None,
    ):
        super().__init__(
            config,
            platform=platform,
            memory_planner=planner,
            enable_pypto_l0c_double_buffer=planner == MemoryPlanner.PYPTO,
        )
        self.M, self.K, self.N = m, 64, n
        self._planner = planner
        if config is None:
            # Full-K (single-pass per tile, no K-split reduction), so FP32 matmul is
            # tight; a dbC sync error corrupts values far beyond this.
            self.config.rtol = 1e-3
            self.config.atol = 1e-3

    def get_name(self) -> str:
        tag = {
            MemoryPlanner.PYPTO: "pypto",
            MemoryPlanner.DSA_RP: "dsa_rp",
            MemoryPlanner.PTOAS: "ptoas",
        }[self._planner]
        return f"dbc2_ddr_{tag}_{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.M, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [self.K, self.N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N = self.M, self.K, self.N

        @pl.jit.incore
        def kernel(
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

        @pl.jit
        def orchestrator(
            a: pl.Tensor[[M, K], pl.FP32],
            b: pl.Tensor[[K, N], pl.FP32],
            out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        ) -> pl.Tensor[[M, N], pl.FP32]:
            out = kernel(a, b, out)
            return out

        return orchestrator.specialize()

    def compute_expected(self, tensors, params=None):
        tensors["out"][:] = torch.matmul(tensors["a"], tensors["b"])


class _DbcMatScratch(PTOTestCase):
    """Chained ``(a @ b) @ e`` with an on-chip L1/Mat intermediate.

    The oversized producer is assembled with ``tile.assemble`` and consumed
    on-chip. AutoTile retains dbC selection for PTOAS only.
    """

    __test__ = False

    def __init__(
        self,
        *,
        planner: MemoryPlanner = MemoryPlanner.PTOAS,
        platform: str | None = None,
        config=None,
    ):
        super().__init__(
            config,
            platform=platform,
            memory_planner=planner,
            enable_pypto_l0c_double_buffer=planner == MemoryPlanner.PYPTO,
        )
        self.M, self.K, self.N, self.P = 256, 64, 256, 64
        self._planner = planner
        if config is None:
            # bf16 operands + bf16 FIXPIPE-downcast intermediate: bf16 tolerance.
            self.config.rtol = 2e-2
            self.config.atol = 2e-2

    def get_name(self) -> str:
        tag = {
            MemoryPlanner.PYPTO: "pypto",
            MemoryPlanner.DSA_RP: "dsa_rp",
            MemoryPlanner.PTOAS: "ptoas",
        }[self._planner]
        return f"dbc2_mat_scratch_{tag}_{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        M, K, N, P = self.M, self.K, self.N, self.P

        # Keep the chained BF16 result O(1) so the fixed absolute tolerance covers
        # cancellation elements, and use local generators because the harness does
        # not seed torch before materializing inputs.
        def seeded(rows, cols, seed, scale=1.0):
            return torch.randn(rows, cols, generator=torch.Generator().manual_seed(seed)) * scale

        return [
            TensorSpec("a", [M, K], DataType.BF16, init_value=lambda: seeded(M, K, 1)),
            TensorSpec("b", [K, N], DataType.BF16, init_value=lambda: seeded(K, N, 2, 1 / K**0.5)),
            TensorSpec("e", [N, P], DataType.BF16, init_value=lambda: seeded(N, P, 3, 1 / N**0.5)),
            TensorSpec("out", [M, P], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N, P = self.M, self.K, self.N, self.P

        @pl.jit.incore
        def kernel(
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

        @pl.jit
        def orchestrator(
            a: pl.Tensor[[M, K], pl.BF16],
            b: pl.Tensor[[K, N], pl.BF16],
            e: pl.Tensor[[N, P], pl.BF16],
            out: pl.Out[pl.Tensor[[M, P], pl.FP32]],
        ) -> pl.Tensor[[M, P], pl.FP32]:
            out = kernel(a, b, e, out)
            return out

        return orchestrator.specialize()

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
    """Automatic dbC stays disabled under PYPTO/DSA-RP and available under PTOAS."""

    # Historical shapes retain golden coverage. Check the selected buffering
    # policy without forcing the old dbC geometry onto the in-tree planners.
    @pytest.mark.parametrize("platform", PLATFORMS_DBC)
    @pytest.mark.parametrize(
        "planner",
        [
            pytest.param(MemoryPlanner.PYPTO, id="pypto"),
            pytest.param(MemoryPlanner.DSA_RP, id="dsa_rp"),
            pytest.param(MemoryPlanner.PTOAS, id="ptoas"),
        ],
    )
    @pytest.mark.parametrize(
        "m,n,tile_m,tile_n,tile_count,stationarity",
        [
            (160, 160, 80, 128, 4, _core_passes.l0_tile_chooser.Stationarity.OutputStationary),
            (144, 144, 48, 128, 6, _core_passes.l0_tile_chooser.Stationarity.OutputStationary),
            (256, 256, 32, 256, 8, _core_passes.l0_tile_chooser.Stationarity.BStationary),
            (448, 448, 64, 256, 14, _core_passes.l0_tile_chooser.Stationarity.OutputStationary),
        ],
    )
    def test_direct_store_dbc(
        self, test_runner, platform, planner, m, n, tile_m, tile_n, tile_count, stationarity
    ):
        """Retain direct-store numerical coverage with planner-specific selection."""
        choice = _choose_a2a3_fp32_dbc(m, n, planner=planner)
        count = ((m + choice.m - 1) // choice.m) * ((n + choice.n - 1) // choice.n)
        assert count >= 1
        assert choice.double_buffer_c == (planner == MemoryPlanner.PTOAS)
        case = _DbcDirectStore(m, n, planner=planner, platform=platform)
        printed = _printed_after_auto_tile(case, planner)
        assert ("pipeline_double_buffer_c" in printed) == choice.double_buffer_c
        assert "slots=2" not in printed
        result = test_runner.run(case)
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS_DBC)
    @pytest.mark.parametrize(
        "planner",
        [
            pytest.param(MemoryPlanner.PYPTO, id="pypto"),
            pytest.param(MemoryPlanner.DSA_RP, id="dsa_rp"),
        ],
    )
    @pytest.mark.parametrize(
        "m,n,tile_m,tile_n,holds_a",
        [
            (16, 256, 16, 128, True),
            (256, 16, 128, 16, False),
        ],
    )
    def test_one_dimensional_dbc(self, test_runner, platform, planner, m, n, tile_m, tile_n, holds_a):
        """Singleton axes do not automatically enable accumulator ping-pong."""
        choice = _choose_a2a3_fp32_dbc(m, n, planner=planner)
        assert not choice.double_buffer_c

        case = _DbcDirectStore(m, n, planner=planner, platform=platform)
        printed = _printed_after_auto_tile(case, planner)
        assert ("pipeline_double_buffer_c" in printed) == choice.double_buffer_c
        result = test_runner.run(case)
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS_DBC)
    def test_ptoas_384x256_operand_allocation(self, test_runner, platform):
        """Retain numerical coverage of the formerly oversized PTOAS operand."""
        planner = MemoryPlanner.PTOAS
        choice = _choose_a2a3_fp32_dbc(384, 256, planner=planner)
        count = ((384 + choice.m - 1) // choice.m) * ((256 + choice.n - 1) // choice.n)
        assert count >= 1
        assert choice.double_buffer_c
        result = test_runner.run(_DbcDirectStore(384, 256, planner=MemoryPlanner.PTOAS, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS_DBC)
    @pytest.mark.parametrize(
        "planner",
        [
            pytest.param(MemoryPlanner.PYPTO, id="pypto"),
            pytest.param(MemoryPlanner.DSA_RP, id="dsa_rp"),
            pytest.param(MemoryPlanner.PTOAS, id="ptoas"),
        ],
    )
    def test_mat_scratch_dbc(self, test_runner, platform, planner):
        """Retain the Mat-scratch chain under each planner's accumulator policy."""
        choice = _choose_a2a3_dbc(256, 64, 256, planner=planner, bytes_a=2, bytes_b=2)
        assert choice.double_buffer_c == (planner == MemoryPlanner.PTOAS)
        case = _DbcMatScratch(planner=planner, platform=platform)
        printed = _printed_after_auto_tile(case, planner)
        assert ("pipeline_double_buffer_c" in printed) == choice.double_buffer_c
        result = test_runner.run(case)
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS_DBC)
    @pytest.mark.parametrize(
        "planner",
        [
            pytest.param(MemoryPlanner.PYPTO, id="pypto"),
            pytest.param(MemoryPlanner.DSA_RP, id="dsa_rp"),
            pytest.param(MemoryPlanner.PTOAS, id="ptoas"),
        ],
    )
    def test_non_divisible_tail_dbc(self, test_runner, platform, planner):
        """Retain padded M/N boundary coverage with planner-specific dbC selection."""
        choice = _choose_a2a3_fp32_dbc(320, 320, planner=planner)
        assert choice.double_buffer_c == (planner == MemoryPlanner.PTOAS)
        case = _DbcDirectStore(320, 320, planner=planner, platform=platform)
        printed = _printed_after_auto_tile(case, planner)
        assert "pipeline_membership" not in printed
        assert "slots=2" not in printed
        result = test_runner.run(case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
