# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
System tests for ND tensor.matmul / tensor.matmul_acc lowering.

These tests exercise the high-level ``pl.matmul`` / ``pl.matmul_acc`` DSL
entry points with rank>2 tensor inputs, validating that the compiler:

* dispatches ``tensor.matmul`` / ``tensor.matmul_acc`` to the batched tile
  ops (``tile.batch_matmul`` / ``tile.batch_matmul_acc``) when any operand
  has rank > 2 (see ``op_conversion_registry.cpp``);
* propagates ``b_trans=True`` to the producer ``pl.load`` via the
  ``InputSpaceReq`` mechanism (no explicit ``pl.transpose`` is emitted);
* unrolls the batched tile op via ``FlattenTileNdTo2D`` into per-batch 2D
  tile ops and produces correct numerical results.

This is the end-to-end DeepSeek-V4 MoE W2 matmul shape (``2D @ 3D``) the
``deepseek_v4_decode_moe_w2_matmul`` kernel relies on.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness import st
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.pypto_core.passes import MemoryPlanner


class TestTensorMatmul2dBy3d(PTOTestCase):
    """``pl.matmul(2D, 3D, b_trans=True)`` — single-batch broadcast.

    Validates the rank-based dispatch in ConvertTensorToTileOps and the
    batch-1 fast path in FlattenTileNdTo2D's batch_matmul lowering.
    """

    __test__ = False

    def __init__(
        self,
        m: int = 16,
        k: int = 64,
        n: int = 64,
        *,
        platform: str | None = None,
        config=None,
    ):
        super().__init__(config, platform=platform)
        # fp32 K-split matmul: the device reduces K in a different order than the
        # single-pass torch golden -> ~K*eps_fp32 drift (~1.5e-5 at K=128), above the
        # 1e-5 default for near-zero (cancellation) elements. Observed CI flake
        # (3d_btrans_4x64x128x512, a2a3): 1/131072 elems, diff ~1.685e-5; 1e-4 clears
        # it with margin. Same class as AutoL0.
        if config is None:
            self.config.rtol = 1e-4
            self.config.atol = 1e-4
        self.M = m
        self.K = k
        self.N = n

    def get_name(self) -> str:
        return f"tensor_matmul_2d_by_3d_{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.M, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [1, self.N, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [1, self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N = self.M, self.K, self.N

        @pl.jit.incore
        def tensor_matmul_2d_by_3d(
            a: pl.Tensor[[M, K], pl.FP32],
            b: pl.Tensor[[1, N, K], pl.FP32],
            c: pl.Out[pl.Tensor[[1, M, N], pl.FP32]],
        ) -> pl.Tensor[[1, M, N], pl.FP32]:
            # ND tensor-level matmul: lowers to tile.batch_matmul because b is rank-3.
            matmul_out: pl.Tensor[[1, M, N], pl.FP32] = pl.matmul(a, b, b_trans=True, out_dtype=pl.FP32)
            # Explicitly write back into the Out tensor `c` so the runtime sees the
            # result on the bound output buffer (returning a fresh tensor would
            # otherwise leak a separate result slot and leave `c` uninitialised).
            c = pl.assemble(c, matmul_out, offset=[0, 0, 0])
            return c

        @pl.jit
        def orchestrator(
            a: pl.Tensor[[M, K], pl.FP32],
            b: pl.Tensor[[1, N, K], pl.FP32],
            out_c: pl.Out[pl.Tensor[[1, M, N], pl.FP32]],
        ) -> pl.Tensor[[1, M, N], pl.FP32]:
            out_c = tensor_matmul_2d_by_3d(a, b, out_c)
            return out_c

        # PTOTestCase consumes pre-pass IR; its runner performs full compilation.
        return orchestrator.specialize()

    def compute_expected(self, tensors, params=None):
        a = tensors["a"].to(torch.float32)
        b = tensors["b"].to(torch.float32)
        tensors["c"][:] = torch.matmul(a.unsqueeze(0), b.transpose(-2, -1))


class TestTensorMatmulAcc2dBy3d(PTOTestCase):
    """``pl.matmul`` + ``pl.matmul_acc`` chained on (2D, 3D) inputs.

    K is split into two halves: the first chunk feeds ``pl.matmul`` and the
    second chunk feeds ``pl.matmul_acc``. This is the DeepSeek-V4 MoE W2
    pattern from ``deepseek_v4_decode_moe_w2_matmul`` (3D expert weights
    consumed without manual 2D slicing).
    """

    __test__ = False

    def __init__(
        self,
        m: int = 16,
        k: int = 64,
        n: int = 64,
        *,
        platform: str | None = None,
        config=None,
    ):
        super().__init__(config, platform=platform)
        # fp32 K-split matmul: the device reduces K in a different order than the
        # single-pass torch golden -> ~K*eps_fp32 drift (~1.5e-5 at K=128), above the
        # 1e-5 default for near-zero (cancellation) elements. Observed CI flake
        # (3d_btrans_4x64x128x512, a2a3): 1/131072 elems, diff ~1.685e-5; 1e-4 clears
        # it with margin. Same class as AutoL0.
        if config is None:
            self.config.rtol = 1e-4
            self.config.atol = 1e-4
        self.M = m
        self.K = k
        self.N = n
        assert self.K % 2 == 0, "K must split evenly for two-chunk accumulation"

    def get_name(self) -> str:
        return f"tensor_matmul_acc_2d_by_3d_{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.M, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [1, self.N, self.K], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [1, self.M, self.N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        M, K, N = self.M, self.K, self.N
        K_HALF = K // 2

        @pl.jit.incore
        def tensor_matmul_acc_2d_by_3d(
            a: pl.Tensor[[M, K], pl.FP32],
            b: pl.Tensor[[1, N, K], pl.FP32],
            c: pl.Out[pl.Tensor[[1, M, N], pl.FP32]],
        ) -> pl.Tensor[[1, M, N], pl.FP32]:
            a0: pl.Tensor[[M, K_HALF], pl.FP32] = pl.slice(a, [M, K_HALF], [0, 0])
            a1: pl.Tensor[[M, K_HALF], pl.FP32] = pl.slice(a, [M, K_HALF], [0, K_HALF])
            b0: pl.Tensor[[1, N, K_HALF], pl.FP32] = pl.slice(b, [1, N, K_HALF], [0, 0, 0])
            b1: pl.Tensor[[1, N, K_HALF], pl.FP32] = pl.slice(b, [1, N, K_HALF], [0, 0, K_HALF])
            acc: pl.Tensor[[1, M, N], pl.FP32] = pl.matmul(a0, b0, b_trans=True, out_dtype=pl.FP32)
            acc_final: pl.Tensor[[1, M, N], pl.FP32] = pl.matmul_acc(acc, a1, b1, b_trans=True)
            # Write back into the Out tensor `c` (see TestTensorMatmul2dBy3d for why).
            c = pl.assemble(c, acc_final, offset=[0, 0, 0])
            return c

        @pl.jit
        def orchestrator(
            a: pl.Tensor[[M, K], pl.FP32],
            b: pl.Tensor[[1, N, K], pl.FP32],
            out_c: pl.Out[pl.Tensor[[1, M, N], pl.FP32]],
        ) -> pl.Tensor[[1, M, N], pl.FP32]:
            out_c = tensor_matmul_acc_2d_by_3d(a, b, out_c)
            return out_c

        return orchestrator.specialize()

    def compute_expected(self, tensors, params=None):
        a = tensors["a"].to(torch.float32)
        b = tensors["b"].to(torch.float32)
        tensors["c"][:] = torch.matmul(a.unsqueeze(0), b.transpose(-2, -1))


class TestTensorBatchMatmulMixedAddBTrans(PTOTestCase):
    """Mixed-kernel probe: a Vec compute result feeds a ``b_trans=True`` ND matmul.

    ``bt = b0 + b1`` is a vector (Vec/UB) op result over 3D operands;
    ``c[i] = a[i] @ bt[i]^T``. The transposed operand of the resulting
    ``tile.batch_matmul`` originates from a compute op, NOT a load — so the
    legacy transpose-at-load path has no transposed load to ride on. This
    probes the CURRENT (pre-migration) ND behavior for a compute-sourced
    transposed batch_matmul operand: silently dropped transpose, wrong
    numerics, or a compile-time failure.
    """

    __test__ = False

    def __init__(
        self,
        b: int = 2,
        m: int = 16,
        k: int = 64,
        n: int = 64,
        *,
        platform: str | None = None,
        config=None,
    ):
        super().__init__(config, platform=platform)
        # fp32 K-split matmul: the device reduces K in a different order than the
        # single-pass torch golden -> ~K*eps_fp32 drift (~1.5e-5 at K=128), above the
        # 1e-5 default for near-zero (cancellation) elements. Observed CI flake
        # (3d_btrans_4x64x128x512, a2a3): 1/131072 elems, diff ~1.685e-5; 1e-4 clears
        # it with margin. Same class as AutoL0.
        if config is None:
            self.config.rtol = 1e-4
            self.config.atol = 1e-4
        self.B = b
        self.M = m
        self.K = k
        self.N = n

    def get_name(self) -> str:
        return f"tensor_batch_matmul_mixed_add_btrans_{self.B}x{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        B, M, K, N = self.B, self.M, self.K, self.N
        return [
            TensorSpec("a", [B, M, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b0", [B, N, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b1", [B, N, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [B, M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        B, M, K, N = self.B, self.M, self.K, self.N

        @pl.jit.incore
        def mixed_add_btrans_nd(
            a: pl.Tensor[[B, M, K], pl.FP32],
            b0: pl.Tensor[[B, N, K], pl.FP32],
            b1: pl.Tensor[[B, N, K], pl.FP32],
            c: pl.Out[pl.Tensor[[B, M, N], pl.FP32]],
        ) -> pl.Tensor[[B, M, N], pl.FP32]:
            bt = pl.add(b0, b1)  # Vec compute -> [B, N, K]
            # rank-3 operands -> tile.batch_matmul; bt is the b_trans operand.
            cm = pl.matmul(a, bt, b_trans=True, out_dtype=pl.FP32)  # batch a @ bt^T -> [B, M, N]
            out_c = pl.assemble(c, cm, offset=[0, 0, 0])
            return out_c

        @pl.jit
        def orchestrator(
            a: pl.Tensor[[B, M, K], pl.FP32],
            b0: pl.Tensor[[B, N, K], pl.FP32],
            b1: pl.Tensor[[B, N, K], pl.FP32],
            out_c: pl.Out[pl.Tensor[[B, M, N], pl.FP32]],
        ) -> pl.Tensor[[B, M, N], pl.FP32]:
            out_c = mixed_add_btrans_nd(a, b0, b1, out_c)
            return out_c

        return orchestrator.specialize()

    def compute_expected(self, tensors, params=None):
        a = tensors["a"].to(torch.float32)
        bt = tensors["b0"].to(torch.float32) + tensors["b1"].to(torch.float32)
        tensors["c"][:] = torch.matmul(a, bt.transpose(-2, -1))


class TestTensorBatchMatmul3dBy3dBTrans(PTOTestCase):
    """``pl.matmul(3D, 3D, b_trans=True)`` — true multi-batch GM b_trans.

    Both operands are per-batch rank-3 (``a`` = ``[B, M, K]``, ``b`` =
    ``[B, N, K]``); ``c[i] = a[i] @ b[i]^T``. This exercises the unified
    transpose_view path: ``b`` is loaded whole to Mat, reinterpreted by a
    single ``tile.transpose_view`` into ``[K, B*N]`` (batch concatenated on
    the column axis), then column-sliced per batch at offset ``i*N``. Run
    with non-32-aligned N/K to stress per-batch slice fractal alignment.
    """

    __test__ = False

    def __init__(
        self,
        b: int = 2,
        m: int = 16,
        k: int = 80,
        n: int = 48,
        *,
        platform: str | None = None,
        config=None,
        memory_planner: MemoryPlanner | None = None,
    ):
        super().__init__(config, platform=platform, memory_planner=memory_planner)
        # fp32 K-split matmul: the device reduces K in a different order than the
        # single-pass torch golden -> ~K*eps_fp32 drift (~1.5e-5 at K=128), above the
        # 1e-5 default for near-zero (cancellation) elements. Observed CI flake
        # (3d_btrans_4x64x128x512, a2a3): 1/131072 elems, diff ~1.685e-5; 1e-4 clears
        # it with margin. Same class as AutoL0.
        if config is None:
            self.config.rtol = 1e-4
            self.config.atol = 1e-4
        self.B = b
        self.M = m
        self.K = k
        self.N = n

    def get_name(self) -> str:
        planner = self.get_memory_planner()
        suffix = f"_{planner.name.lower()}" if planner is not None else ""
        return f"tensor_batch_matmul_3d_by_3d_btrans_{self.B}x{self.M}x{self.K}x{self.N}{suffix}"

    def define_tensors(self) -> list[TensorSpec]:
        B, M, K, N = self.B, self.M, self.K, self.N
        return [
            TensorSpec("a", [B, M, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [B, N, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [B, M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        B, M, K, N = self.B, self.M, self.K, self.N

        @pl.jit.incore
        def batch_matmul_3d_btrans(
            a: pl.Tensor[[B, M, K], pl.FP32],
            b: pl.Tensor[[B, N, K], pl.FP32],
            c: pl.Out[pl.Tensor[[B, M, N], pl.FP32]],
        ) -> pl.Tensor[[B, M, N], pl.FP32]:
            cm = pl.matmul(a, b, b_trans=True, out_dtype=pl.FP32)  # batch a @ b^T -> [B, M, N]
            out_c = pl.assemble(c, cm, offset=[0, 0, 0])
            return out_c

        @pl.jit
        def orchestrator(
            a: pl.Tensor[[B, M, K], pl.FP32],
            b: pl.Tensor[[B, N, K], pl.FP32],
            out_c: pl.Out[pl.Tensor[[B, M, N], pl.FP32]],
        ) -> pl.Tensor[[B, M, N], pl.FP32]:
            out_c = batch_matmul_3d_btrans(a, b, out_c)
            return out_c

        return orchestrator.specialize()

    def compute_expected(self, tensors, params=None):
        a = tensors["a"].to(torch.float32)
        b = tensors["b"].to(torch.float32)
        tensors["c"][:] = torch.matmul(a, b.transpose(-2, -1))


class TestTensorBatchMatmulSlicedBTrans(PTOTestCase):
    """``pl.matmul(3D, slice(3D), b_trans=True)`` — dsv4 proj_a grouped-slice pattern.

    The b_trans operand is a SLICE of a larger per-group weight ``wfull[G, N, K]``
    (sliced to ``[B, N, K]`` at group offset ``G0``). The slice offset folds into
    the natural whole-load's 2D-collapsed row offset, then the unified
    ``tile.transpose_view`` + per-batch column-slice path runs (#1776). Exercised
    end-to-end (the convert UT only checks IR shape) with non-aligned N/K.
    """

    __test__ = False

    def __init__(
        self,
        g: int = 4,
        g0: int = 1,
        b: int = 2,
        m: int = 16,
        k: int = 80,
        n: int = 48,
        *,
        platform: str | None = None,
        config=None,
    ):
        super().__init__(config, platform=platform)
        # fp32 K-split matmul: the device reduces K in a different order than the
        # single-pass torch golden -> ~K*eps_fp32 drift (~1.5e-5 at K=128), above the
        # 1e-5 default for near-zero (cancellation) elements. Observed CI flake
        # (3d_btrans_4x64x128x512, a2a3): 1/131072 elems, diff ~1.685e-5; 1e-4 clears
        # it with margin. Same class as AutoL0.
        if config is None:
            self.config.rtol = 1e-4
            self.config.atol = 1e-4
        self.G, self.G0, self.B, self.M, self.K, self.N = g, g0, b, m, k, n

    def get_name(self) -> str:
        return f"tensor_batch_matmul_sliced_btrans_{self.G}g{self.G0}_{self.B}x{self.M}x{self.K}x{self.N}"

    def define_tensors(self) -> list[TensorSpec]:
        G, B, M, K, N = self.G, self.B, self.M, self.K, self.N
        return [
            TensorSpec("a", [B, M, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("wfull", [G, N, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("c", [B, M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        G, G0, B, M, K, N = self.G, self.G0, self.B, self.M, self.K, self.N

        @pl.jit.incore
        def sliced_btrans(
            a: pl.Tensor[[B, M, K], pl.FP32],
            wfull: pl.Tensor[[G, N, K], pl.FP32],
            c: pl.Out[pl.Tensor[[B, M, N], pl.FP32]],
        ) -> pl.Tensor[[B, M, N], pl.FP32]:
            w = pl.slice(wfull, [B, N, K], [G0, 0, 0])  # grouped slice -> [B, N, K]
            cm = pl.matmul(a, w, b_trans=True, out_dtype=pl.FP32)
            out_c = pl.assemble(c, cm, offset=[0, 0, 0])
            return out_c

        @pl.jit
        def orchestrator(
            a: pl.Tensor[[B, M, K], pl.FP32],
            wfull: pl.Tensor[[G, N, K], pl.FP32],
            out_c: pl.Out[pl.Tensor[[B, M, N], pl.FP32]],
        ) -> pl.Tensor[[B, M, N], pl.FP32]:
            out_c = sliced_btrans(a, wfull, out_c)
            return out_c

        return orchestrator.specialize()

    def compute_expected(self, tensors, params=None):
        a = tensors["a"].to(torch.float32)
        w = tensors["wfull"].to(torch.float32)[self.G0 : self.G0 + self.B]
        tensors["c"][:] = torch.matmul(a, w.transpose(-2, -1))


_ND_SHAPES = [(16, 64, 64), (16, 128, 64)]


class TestTensorBatchMatmulOperations:
    """Test suite for ND ``pl.matmul`` / ``pl.matmul_acc`` lowering."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("b,m,k,n", [(2, 16, 80, 48)])
    def test_tensor_batch_matmul_mixed_add_btrans(self, test_runner, platform, b, m, k, n):
        """Mixed-kernel: a Vec compute (add) result feeds a b_trans=True ND matmul.

        The transposed operand originates from a Vec compute (not a load), so it is
        moved WHOLE to Mat and reinterpreted via a zero-copy ``tile.transpose_view``,
        then per-batch column-sliced. A small (non-aligned) shape is used so the
        whole-tile V2C move fits the fixed 8-deep cross-core ring (``8 * slot_size
        <= L1``); larger shapes need a per-batch V2C move (separate follow-up).
        """
        result = test_runner.run(TestTensorBatchMatmulMixedAddBTrans(b=b, m=m, k=k, n=n, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("b,m,k,n", [(2, 16, 80, 48), (3, 16, 64, 64)])
    def test_tensor_batch_matmul_3d_by_3d_btrans(self, test_runner, platform, b, m, k, n):
        """Multi-batch ``pl.matmul(3D, 3D, b_trans=True)``: whole transpose_view + per-batch column slice."""
        result = test_runner.run(TestTensorBatchMatmul3dBy3dBTrans(b=b, m=m, k=k, n=n, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("g,g0,b,m,k,n", [(4, 1, 2, 16, 80, 48)])
    def test_tensor_batch_matmul_sliced_btrans(self, test_runner, platform, g, g0, b, m, k, n):
        """b_trans operand is a grouped SLICE of a 3D weight (dsv4 proj_a) — end-to-end."""
        result = test_runner.run(
            TestTensorBatchMatmulSlicedBTrans(g=g, g0=g0, b=b, m=m, k=k, n=n, platform=platform)
        )
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("b,m,k,n", [(4, 64, 128, 512)])
    @pytest.mark.parametrize(
        "planner",
        [
            pytest.param(MemoryPlanner.PYPTO, id="pypto"),
            pytest.param(MemoryPlanner.DSA_RP, id="dsa-rp"),
            pytest.param(MemoryPlanner.PTOAS, id="ptoas"),
        ],
    )
    def test_tensor_batch_matmul_3d_btrans_not_fit(self, test_runner, platform, b, m, k, n, planner):
        """Issue #2872: per-batch loads and L0B blocking must work for every planner.

        Whole lhs+rhs exceed L1, so the capacity gate selects per-batch loads.
        AutoTile must then split the 256 KiB RHS into legal 64 KiB L0B panels.
        """
        result = test_runner.run(
            TestTensorBatchMatmul3dBy3dBTrans(b=b, m=m, k=k, n=n, platform=platform, memory_planner=planner)
        )
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("m,k,n", _ND_SHAPES)
    def test_tensor_matmul_2d_by_3d(self, test_runner, platform, m, k, n):
        """``pl.matmul(2D, 3D, b_trans=True)`` end-to-end."""
        result = test_runner.run(TestTensorMatmul2dBy3d(m=m, k=k, n=n, platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("m,k,n", _ND_SHAPES)
    def test_tensor_matmul_acc_2d_by_3d(self, test_runner, platform, m, k, n):
        """Chained ``pl.matmul`` + ``pl.matmul_acc`` on (2D, 3D) inputs."""
        result = test_runner.run(TestTensorMatmulAcc2dBy3d(m=m, k=k, n=n, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


def _offset_batch_matmul_case(batch_shape: list[int], m: int, k: int, n: int, planner: MemoryPlanner):
    """Check ND placement and untouched output guards with exact integer products."""
    a_shape = batch_shape + [m, k]
    b_shape = batch_shape + [n, k]
    output_shape = [b + 2 for b in batch_shape] + [m + 32, n + 64]
    offsets = [1] * len(batch_shape) + [16, 32]

    @pl.jit.incore
    def body(
        a: pl.Tensor[a_shape, pl.FP32],
        b: pl.Tensor[b_shape, pl.FP32],
        c: pl.InOut[pl.Tensor[output_shape, pl.FP32]],
    ) -> pl.Tensor[output_shape, pl.FP32]:
        product = pl.matmul(a, b, b_trans=True, out_dtype=pl.FP32)
        return pl.assemble(c, product, offset=offsets)

    @pl.jit
    def kernel(
        a: pl.Tensor[a_shape, pl.FP32],
        b: pl.Tensor[b_shape, pl.FP32],
        c: pl.InOut[pl.Tensor[output_shape, pl.FP32]],
    ) -> pl.Tensor[output_shape, pl.FP32]:
        return body(a, b, c)

    generator = torch.Generator().manual_seed(2872)
    a = torch.randint(-3, 4, a_shape, generator=generator).float()
    b = torch.randint(-3, 4, b_shape, generator=generator).float()
    c = torch.full(output_shape, -12345.0)
    window = tuple(slice(1, size + 1) for size in batch_shape) + (slice(16, m + 16), slice(32, n + 32))

    def golden(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        expected = tensors["c"].clone()
        expected[window] = torch.matmul(tensors["a"], tensors["b"].transpose(-2, -1))
        return expected

    label = "x".join(str(size) for size in batch_shape + [m, k, n])
    return st.case(
        kernel,
        a,
        b,
        c,
        name=f"batch_matmul_nd_offsets_{label}_{planner.name.lower()}",
        golden=golden,
        memory_planner=planner,
        rtol=0.0,
        atol=0.0,
    )


@st.cases(
    *(
        _offset_batch_matmul_case(batch_shape, m, k, n, planner)
        for batch_shape, m, k, n in (([4], 64, 128, 512), ([2, 2], 80, 128, 544), ([2], 256, 256, 256))
        for planner in (MemoryPlanner.PYPTO, MemoryPlanner.DSA_RP, MemoryPlanner.PTOAS)
    )
)
def test_batch_matmul_nd_offsets(case_run):
    """M/N tiling must preserve all batch axes, matrix offsets and guard regions."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
