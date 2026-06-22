# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for tile-level Cube ops: gemv, gemv_acc, gemv_bias, matmul_bias.

Operands are loaded into Mat (L1); the layout passes (AutoTileMatmulL0 /
CanonicalizeTileSlice) insert the Left/Right L0 extracts, mirroring the proven
Mat-resident pattern in test_matmul.py::TestMatmulAutoL0.

Each op is exercised aligned and with a narrowed valid_shape on the OUTPUT
dimension (so the golden is robust to the contraction-dim valid semantics):
  * gemv / gemv_acc / gemv_bias — M is fixed at 1, so the output COLUMN extent
    is narrowed to VALID_N via the operand's N valid_shape; out[:, VALID_N:] = 0.
  * matmul_bias — the output ROW extent is narrowed to VALID_M via A's row
    valid_shape; out[VALID_M:, :] = 0.

Cube accumulation reorders the K reduction relative to torch's BLAS reference,
so a relaxed FP32 tolerance is used.

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


def _cfg() -> RunConfig:
    return RunConfig(rtol=_RTOL, atol=_ATOL)


# ---------------------------------------------------------------------------
# gemv: C[1, N] = A[1, K] @ B[K, N]
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# gemv_acc: C[1, N] = acc[1, N] + A[1, K] @ B[K, N]
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# gemv_bias: C[1, N] = A[1, K] @ B[K, N] + bias[1, N]
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# matmul_bias: C[M, N] = A[M, K] @ B[K, N] + bias[1, N]
# ---------------------------------------------------------------------------


class MatmulBiasTestCase(PTOTestCase):
    __test__ = False

    def __init__(self, *, narrow: bool = False, config=None):
        super().__init__(config)
        self._narrow = narrow

    def get_name(self) -> str:
        return "tile_matmul_bias_narrow" if self._narrow else "tile_matmul_bias"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, K], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [K, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("bias", [1, N], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        a_vshape = [VALID_M, K] if self._narrow else [M, K]

        @pl.program
        class MatmulBiasProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                bias: pl.Tensor[[1, N], pl.FP32],
                out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [M, K], valid_shapes=a_vshape, target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [K, N], target_memory=pl.MemorySpace.Mat)
                tile_bias = pl.load(bias, [0, 0], [1, N], target_memory=pl.MemorySpace.Mat)
                out = pl.store(pl.tile.matmul_bias(tile_a, tile_b, tile_bias), [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, K], pl.FP32],
                b: pl.Tensor[[K, N], pl.FP32],
                bias: pl.Tensor[[1, N], pl.FP32],
                out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                out = self.kernel(a, b, bias, out)
                return out

        return MatmulBiasProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        full = torch.matmul(tensors["a"], tensors["b"]) + tensors["bias"]
        if self._narrow:
            out = torch.zeros_like(tensors["out"])
            out[:VALID_M, :] = full[:VALID_M, :]
            tensors["out"][:] = out
        else:
            tensors["out"][:] = full


class TestGemv:
    """Cube gemv / gemv_acc / gemv_bias / matmul_bias on a2a3."""

    @pytest.mark.platforms("a2a3")
    def test_tile_gemv(self, test_runner):
        result = test_runner.run(GemvTestCase(config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_gemv_narrow(self, test_runner):
        result = test_runner.run(GemvTestCase(narrow=True, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_gemv_acc(self, test_runner):
        result = test_runner.run(GemvAccTestCase(config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_gemv_acc_narrow(self, test_runner):
        result = test_runner.run(GemvAccTestCase(narrow=True, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_gemv_bias(self, test_runner):
        result = test_runner.run(GemvBiasTestCase(config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_gemv_bias_narrow(self, test_runner):
        result = test_runner.run(GemvBiasTestCase(narrow=True, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_matmul_bias(self, test_runner):
        result = test_runner.run(MatmulBiasTestCase(config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_matmul_bias_narrow(self, test_runner):
        result = test_runner.run(MatmulBiasTestCase(narrow=True, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
