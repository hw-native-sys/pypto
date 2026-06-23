# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for tile-level Cube ops: matmul_bias (active) + gemv family (skipped).

matmul_bias: C[M,N] = A[M,K] @ B[K,N] + bias[1,N]. Operands load to Mat (L1);
the layout passes (AutoTileMatmulL0 / CanonicalizeTileSlice) insert the L0
Left/Right extracts. Coverage: several M/K/N shapes (incl. non-square and a
K=128 case that forces AutoTileMatmulL0 to K-split), BF16 inputs with an FP32
accumulator, narrowed valid_shape on the output rows (M), output cols (N) and
the contraction (K), and a non-zero output row offset. Cube accumulation
reorders the K reduction vs torch, so a relaxed FP32 tolerance is used.

gemv / gemv_acc / gemv_bias are @pytest.mark.skip: gemv/gemv_bias hit pto-isa
TExtract static_assert(dstRow % 16 == 0) (1-row vector); gemv_acc hits the
acc->acc pto.tmov gap. Tracked in KNOWN_ISSUES.

Scope is a2a3 only (``@pytest.mark.platforms("a2a3")``); a5 coverage is a
separate PR.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

K = 64
N = 64
M = 16
VALID_N = 32
VALID_M = 8

_RTOL = 1e-3
_ATOL = 1e-3

_PL_DT = {DataType.FP32: pl.FP32, DataType.BF16: pl.BF16}


def _cfg() -> RunConfig:
    return RunConfig(rtol=_RTOL, atol=_ATOL)


# ===========================================================================
# matmul_bias (ACTIVE)
# ===========================================================================


class MatmulBiasTestCase(PTOTestCase):
    """C[M,N] = A[M,K] @ B[K,N] + bias[1,N], parametrized over shape/narrow/dtype/offset.

    narrow: None | 'M' (rows) | 'N' (cols) | 'K' (contraction). ab_dtype is the
    A/B element type; bias and output are always FP32 (the accumulator type).
    """

    __test__ = False

    def __init__(
        self, *, m=M, k=K, n=N, narrow=None, ab_dtype=DataType.FP32, out_m=None, off_row=0, config=None
    ):
        super().__init__(config)
        self._m, self._k, self._n = m, k, n
        self._narrow, self._ab = narrow, ab_dtype
        self._out_m, self._off_row = out_m or m, off_row

    def get_name(self) -> str:
        nrw = f"_n{self._narrow}" if self._narrow else ""
        o = f"_off{self._off_row}" if self._off_row else ""
        return f"tile_matmul_bias_{self._m}x{self._k}x{self._n}_{self._ab.value}{nrw}{o}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self._m, self._k], self._ab, init_value=torch.randn),
            TensorSpec("b", [self._k, self._n], self._ab, init_value=torch.randn),
            TensorSpec("bias", [1, self._n], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [self._out_m, self._n], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        m, k, n, om = self._m, self._k, self._n, self._out_m
        off = [self._off_row, 0]
        ab = _PL_DT[self._ab]
        vm = [VALID_M, k] if self._narrow == "M" else [m, k]
        vk_a = [m, VALID_N] if self._narrow == "K" else [m, k]
        vk_b = [VALID_N, n] if self._narrow == "K" else [k, n]
        vn_b = [k, VALID_N] if self._narrow == "N" else [k, n]
        vn_bias = [1, VALID_N] if self._narrow == "N" else [1, n]
        a_v = vk_a if self._narrow == "K" else vm
        b_v = vk_b if self._narrow == "K" else vn_b

        @pl.program
        class MatmulBiasProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[m, k], ab],
                b: pl.Tensor[[k, n], ab],
                bias: pl.Tensor[[1, n], pl.FP32],
                out: pl.Out[pl.Tensor[[om, n], pl.FP32]],
            ) -> pl.Tensor[[om, n], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [m, k], valid_shapes=a_v, target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [k, n], valid_shapes=b_v, target_memory=pl.MemorySpace.Mat)
                tile_bias = pl.load(
                    bias, [0, 0], [1, n], valid_shapes=vn_bias, target_memory=pl.MemorySpace.Mat
                )
                out = pl.store(pl.tile.matmul_bias(tile_a, tile_b, tile_bias), off, out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[m, k], ab],
                b: pl.Tensor[[k, n], ab],
                bias: pl.Tensor[[1, n], pl.FP32],
                out: pl.Out[pl.Tensor[[om, n], pl.FP32]],
            ) -> pl.Tensor[[om, n], pl.FP32]:
                out = self.kernel(a, b, bias, out)
                return out

        return MatmulBiasProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"].to(torch.float32)
        b = tensors["b"].to(torch.float32)
        bias = tensors["bias"]
        out = torch.zeros_like(tensors["out"])
        if self._narrow == "K":
            full = torch.matmul(a[:, :VALID_N], b[:VALID_N, :]) + bias
        else:
            full = torch.matmul(a, b) + bias
        if self._narrow == "M":
            res = torch.zeros(self._m, self._n)
            res[:VALID_M, :] = full[:VALID_M, :]
        elif self._narrow == "N":
            res = torch.zeros(self._m, self._n)
            res[:, :VALID_N] = full[:, :VALID_N]
        else:
            res = full
        out[self._off_row : self._off_row + self._m, :] = res
        tensors["out"][:] = out


_MKN = [(16, 64, 64), (64, 64, 64), (128, 64, 128), (64, 128, 64)]


class TestMatmulBias:
    """Cube matmul_bias on a2a3 across M/K/N, dtype, narrow valid_shape, offset."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("m,k,n", _MKN, ids=[f"{m}x{k}x{n}" for m, k, n in _MKN])
    def test_tile_matmul_bias(self, test_runner, m, k, n):
        result = test_runner.run(MatmulBiasTestCase(m=m, k=k, n=n, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_matmul_bias_ksplit(self, test_runner):
        """K=128 forces AutoTileMatmulL0 K-split on top of the bias add."""
        result = test_runner.run(MatmulBiasTestCase(m=64, k=128, n=128, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_matmul_bias_bf16(self, test_runner):
        result = test_runner.run(
            MatmulBiasTestCase(m=16, k=128, n=256, ab_dtype=DataType.BF16, config=_cfg())
        )
        assert result.passed, f"Test failed: {result.error}"

    # narrow-N (narrowing B/bias output cols) is omitted: the cube does not zero
    # the [:, VALID_N:] output region the way row/contraction narrowing does
    # (verified wrong on a2a3) — KNOWN_ISSUES. narrow-M and narrow-K work.
    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("narrow", ["M", "K"])
    def test_tile_matmul_bias_narrow(self, test_runner, narrow):
        result = test_runner.run(MatmulBiasTestCase(narrow=narrow, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_matmul_bias_offset(self, test_runner):
        result = test_runner.run(MatmulBiasTestCase(out_m=2 * M, off_row=M, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"


# ===========================================================================
# gemv / gemv_acc / gemv_bias (SKIPPED — a2a3 cube/PTOAS gaps, see KNOWN_ISSUES)
# ===========================================================================

_GEMV_SKIP = pytest.mark.skip(
    reason="a2a3 cube codegen gap (gemv 1-row dstRow%16 / acc-tmov); see KNOWN_ISSUES"
)


class GemvTestCase(PTOTestCase):
    __test__ = False

    def __init__(self, *, narrow: bool = False, config=None):
        super().__init__(config)
        self._narrow = narrow

    def get_name(self) -> str:
        return "tile_gemv_narrow" if self._narrow else "tile_gemv"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [1, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [K, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [1, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        b_vshape = [K, VALID_N] if self._narrow else [K, N]

        @pl.program
        class GemvProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[1, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
            ) -> pl.Tensor[[1, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [1, K], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [K, N], valid_shapes=b_vshape, target_memory=pl.MemorySpace.Mat)
                out = pl.store(pl.tile.gemv(tile_a, tile_b), [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[1, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
            ) -> pl.Tensor[[1, N], pl.FP32]:
                out = self.kernel(a, b, out)
                return out

        return GemvProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        full = torch.matmul(tensors["a"], tensors["b"])
        if self._narrow:
            out = torch.zeros_like(tensors["out"])
            out[:, :VALID_N] = full[:, :VALID_N]
            tensors["out"][:] = out
        else:
            tensors["out"][:] = full


class GemvAccTestCase(PTOTestCase):
    __test__ = False

    def __init__(self, *, narrow: bool = False, config=None):
        super().__init__(config)
        self._narrow = narrow

    def get_name(self) -> str:
        return "tile_gemv_acc_narrow" if self._narrow else "tile_gemv_acc"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("acc", [1, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("a", [1, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [K, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [1, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        acc_vshape = [1, VALID_N] if self._narrow else [1, N]
        b_vshape = [K, VALID_N] if self._narrow else [K, N]

        @pl.program
        class GemvAccProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                acc: pl.Tensor[[1, N], pl.FP32],
                a: pl.Tensor[[1, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
            ) -> pl.Tensor[[1, N], pl.FP32]:
                tile_acc = pl.load(
                    acc, [0, 0], [1, N], valid_shapes=acc_vshape, target_memory=pl.MemorySpace.Mat
                )
                tile_a = pl.load(a, [0, 0], [1, K], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [K, N], valid_shapes=b_vshape, target_memory=pl.MemorySpace.Mat)
                out = pl.store(pl.tile.gemv_acc(tile_acc, tile_a, tile_b), [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                acc: pl.Tensor[[1, N], pl.FP32],
                a: pl.Tensor[[1, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
            ) -> pl.Tensor[[1, N], pl.FP32]:
                out = self.kernel(acc, a, b, out)
                return out

        return GemvAccProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        full = tensors["acc"] + torch.matmul(tensors["a"], tensors["b"])
        if self._narrow:
            out = torch.zeros_like(tensors["out"])
            out[:, :VALID_N] = full[:, :VALID_N]
            tensors["out"][:] = out
        else:
            tensors["out"][:] = full


class GemvBiasTestCase(PTOTestCase):
    __test__ = False

    def __init__(self, *, narrow: bool = False, config=None):
        super().__init__(config)
        self._narrow = narrow

    def get_name(self) -> str:
        return "tile_gemv_bias_narrow" if self._narrow else "tile_gemv_bias"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [1, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [K, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("bias", [1, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [1, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        b_vshape = [K, VALID_N] if self._narrow else [K, N]
        bias_vshape = [1, VALID_N] if self._narrow else [1, N]

        @pl.program
        class GemvBiasProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[1, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                bias: pl.Tensor[[1, N], pl.FP32],
                out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
            ) -> pl.Tensor[[1, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [1, K], target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [K, N], valid_shapes=b_vshape, target_memory=pl.MemorySpace.Mat)
                tile_bias = pl.load(
                    bias, [0, 0], [1, N], valid_shapes=bias_vshape, target_memory=pl.MemorySpace.Mat
                )
                out = pl.store(pl.tile.gemv_bias(tile_a, tile_b, tile_bias), [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[1, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                bias: pl.Tensor[[1, N], pl.FP32],
                out: pl.Out[pl.Tensor[[1, N], pl.FP32]],
            ) -> pl.Tensor[[1, N], pl.FP32]:
                out = self.kernel(a, b, bias, out)
                return out

        return GemvBiasProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        full = torch.matmul(tensors["a"], tensors["b"]) + tensors["bias"]
        if self._narrow:
            out = torch.zeros_like(tensors["out"])
            out[:, :VALID_N] = full[:, :VALID_N]
            tensors["out"][:] = out
        else:
            tensors["out"][:] = full


class TestGemvSkipped:
    """gemv / gemv_acc / gemv_bias — skipped pending a2a3 cube 1-row / acc-tmov fixes."""

    @_GEMV_SKIP
    @pytest.mark.platforms("a2a3")
    def test_tile_gemv(self, test_runner):
        result = test_runner.run(GemvTestCase(config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @_GEMV_SKIP
    @pytest.mark.platforms("a2a3")
    def test_tile_gemv_narrow(self, test_runner):
        result = test_runner.run(GemvTestCase(narrow=True, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @_GEMV_SKIP
    @pytest.mark.platforms("a2a3")
    def test_tile_gemv_acc(self, test_runner):
        result = test_runner.run(GemvAccTestCase(config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @_GEMV_SKIP
    @pytest.mark.platforms("a2a3")
    def test_tile_gemv_bias(self, test_runner):
        result = test_runner.run(GemvBiasTestCase(config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
