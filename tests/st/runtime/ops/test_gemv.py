# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for the tile-level Cube ops matmul_bias and the gemv family.

matmul_bias: C[M,N] = A[M,K] @ B[K,N] + bias[1,N]. Operands load to Mat (L1);
the layout passes (AutoTileMatmulL0 / CanonicalizeTileSlice) insert the L0
Left/Right extracts. Coverage: several M/K/N shapes (incl. non-square and a
K=128 case that forces AutoTileMatmulL0 to K-split), BF16 inputs with an FP32
accumulator, narrowed valid_shape on the output rows (M) and the contraction
(K), and a non-zero output row offset. Cube accumulation reorders the K
reduction vs torch, so a relaxed FP32 tolerance is used.

gemv / gemv_bias / gemv_acc: the M==1 specialization of the cube matmul family
(C[1,N] = A[1,K] @ B[K,N], optionally + bias[1,N] or += into an accumulator).
Operands load to Mat (L1) and move to Left/Right (pto.tmov); the 1-row lhs hits
the pto-isa TMovToLeft Rows==1 vector path. Coverage per op: several K/N shapes,
a narrowed contraction (valid_shape != shape on K), and BF16 inputs. gemv_acc
accumulates over two K chunks (fresh gemv then gemv_acc).

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

_PL_DT = {DataType.FP32: pl.FP32, DataType.BF16: pl.BF16, DataType.FP16: pl.FP16}


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
# gemv / gemv_bias (M == 1 matmul family)
# ===========================================================================

# The cube gemv lhs is extracted to Left via the pto-isa 1-row vector path
# (TExtractToAVector), which requires the lhs column count (the contraction K) to
# be a whole number of 512-byte cube blocks: K % (512 / sizeof(dtype)) == 0, i.e.
# K % 128 == 0 for fp32 and K % 256 == 0 for bf16. Shapes below honour that; the
# narrowed-K case keeps an aligned physical K and narrows only the valid extent.
VALID_K = 64  # narrowed contraction valid extent (< physical K) for valid != shape


class GemvTestCase(PTOTestCase):
    """C[1,N] = A[1,K] @ B[K,N], optionally + bias[1,N].

    bias=True adds a [1,N] bias (gemv_bias). narrow='K' narrows the contraction
    valid_shape (valid_shape != shape on the lhs cols / rhs rows). ab_dtype is
    the A/B element type; bias and output are always FP32 (the accumulator type).
    """

    __test__ = False

    def __init__(self, *, k=K, n=N, bias=False, narrow=None, ab_dtype=DataType.FP32, config=None):
        super().__init__(config)
        self._k, self._n = k, n
        self._bias, self._narrow, self._ab = bias, narrow, ab_dtype

    def get_name(self) -> str:
        op = "gemv_bias" if self._bias else "gemv"
        nrw = f"_n{self._narrow}" if self._narrow else ""
        return f"tile_{op}_1x{self._k}x{self._n}_{self._ab.value}{nrw}"

    def define_tensors(self) -> list[TensorSpec]:
        specs = [
            TensorSpec("a", [1, self._k], self._ab, init_value=torch.randn),
            TensorSpec("b", [self._k, self._n], self._ab, init_value=torch.randn),
            TensorSpec("out", [1, self._n], DataType.FP32, is_output=True),
        ]
        if self._bias:
            specs.append(TensorSpec("bias", [1, self._n], DataType.FP32, init_value=torch.randn))
        return specs

    def get_program(self) -> Any:
        k, n = self._k, self._n
        ab = _PL_DT[self._ab]
        use_bias = self._bias
        a_v = [1, VALID_K] if self._narrow == "K" else [1, k]
        b_v = [VALID_K, n] if self._narrow == "K" else [k, n]

        @pl.program
        class GemvProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[1, k], ab],
                b: pl.Tensor[[k, n], ab],
                bias: pl.Tensor[[1, n], pl.FP32],
                out: pl.Out[pl.Tensor[[1, n], pl.FP32]],
            ) -> pl.Tensor[[1, n], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [1, k], valid_shapes=a_v, target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [k, n], valid_shapes=b_v, target_memory=pl.MemorySpace.Mat)
                if use_bias:
                    tile_bias = pl.load(
                        bias, [0, 0], [1, n], valid_shapes=[1, n], target_memory=pl.MemorySpace.Mat
                    )
                    res = pl.tile.gemv_bias(tile_a, tile_b, tile_bias)
                else:
                    res = pl.tile.gemv(tile_a, tile_b)
                out = pl.store(res, [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[1, k], ab],
                b: pl.Tensor[[k, n], ab],
                bias: pl.Tensor[[1, n], pl.FP32],
                out: pl.Out[pl.Tensor[[1, n], pl.FP32]],
            ) -> pl.Tensor[[1, n], pl.FP32]:
                out = self.kernel(a, b, bias, out)
                return out

        return GemvProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"].to(torch.float32)
        b = tensors["b"].to(torch.float32)
        if self._narrow == "K":
            res = torch.matmul(a[:, :VALID_K], b[:VALID_K, :])
        else:
            res = torch.matmul(a, b)
        if self._bias:
            res = res + tensors["bias"]
        tensors["out"][:] = res


# K aligned to 128 (fp32 cube-block); gemv keeps the whole rhs [K,N] resident in
# L0B (no K-split), so K*N*sizeof must stay within the 64 KiB Right-buffer limit.
_KN = [(128, 64), (256, 64), (128, 128)]


class TestGemv:
    """Cube gemv (C[1,N] = A[1,K] @ B[K,N]) on a2a3 across K/N, narrow K, dtype."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("k,n", _KN, ids=[f"1x{k}x{n}" for k, n in _KN])
    def test_tile_gemv(self, test_runner, k, n):
        result = test_runner.run(GemvTestCase(k=k, n=n, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_gemv_narrow_k(self, test_runner):
        """valid_shape != shape: physical K=128 (aligned), valid K=VALID_K narrowed."""
        result = test_runner.run(GemvTestCase(k=128, narrow="K", config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_gemv_bf16(self, test_runner):
        result = test_runner.run(GemvTestCase(k=256, n=64, ab_dtype=DataType.BF16, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_gemv_fp16(self, test_runner):
        # fp16 is the only dtype pto-isa exercises for gemv mode (a3 RunTGEMV
        # cases are all f16xf16->f32); K aligned to 512/2 = 256.
        result = test_runner.run(GemvTestCase(k=256, n=64, ab_dtype=DataType.FP16, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"


class TestGemvBias:
    """Cube gemv_bias (C[1,N] = A[1,K] @ B[K,N] + bias[1,N]) on a2a3."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("k,n", _KN, ids=[f"1x{k}x{n}" for k, n in _KN])
    def test_tile_gemv_bias(self, test_runner, k, n):
        result = test_runner.run(GemvTestCase(k=k, n=n, bias=True, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_gemv_bias_narrow_k(self, test_runner):
        """valid_shape != shape: physical K=128 (aligned), valid K narrowed, with bias."""
        result = test_runner.run(GemvTestCase(k=128, bias=True, narrow="K", config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"


# ===========================================================================
# gemv_acc (C[1,N] += A[1,K] @ B[K,N], accumulated over K chunks)
# ===========================================================================


class GemvAccTestCase(PTOTestCase):
    """C[1,N] = sum over K chunks of A[1,k]@B[k,N] via gemv + gemv_acc.

    K is split into NUM_CHUNKS chunks: the first chunk is a fresh ``pl.gemv``,
    the rest accumulate with ``pl.gemv_acc``. narrow='K' narrows the per-chunk
    contraction valid_shape (valid_shape != shape).
    """

    __test__ = False
    NUM_CHUNKS = 2  # two K chunks: a fresh gemv followed by one gemv_acc

    # Each chunk's physical K feeds the cube gemv lhs, so it must stay aligned
    # to the fp32 cube block (128); the narrow case narrows only the valid K.
    def __init__(self, *, k_chunk=128, n=N, narrow=None, config=None):
        super().__init__(config)
        self._k_chunk, self._n, self._narrow = k_chunk, n, narrow
        self._k = k_chunk * self.NUM_CHUNKS

    def get_name(self) -> str:
        nrw = f"_n{self._narrow}" if self._narrow else ""
        return f"tile_gemv_acc_1x{self._k}x{self._n}_chunks{self.NUM_CHUNKS}{nrw}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [1, self._k], DataType.FP32, init_value=torch.randn),
            TensorSpec("b", [self._k, self._n], DataType.FP32, init_value=torch.randn),
            TensorSpec("out", [1, self._n], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        kc, n, ktot = self._k_chunk, self._n, self._k
        vkc = VALID_K if self._narrow == "K" else kc
        a_v = [1, vkc]
        b_v = [vkc, n]

        @pl.program
        class GemvAccProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[1, ktot], pl.FP32],
                b: pl.Tensor[[ktot, n], pl.FP32],
                out: pl.Out[pl.Tensor[[1, n], pl.FP32]],
            ) -> pl.Tensor[[1, n], pl.FP32]:
                # chunk 0: fresh gemv
                a0 = pl.load(a, [0, 0], [1, kc], valid_shapes=a_v, target_memory=pl.MemorySpace.Mat)
                b0 = pl.load(b, [0, 0], [kc, n], valid_shapes=b_v, target_memory=pl.MemorySpace.Mat)
                acc = pl.tile.gemv(a0, b0)
                # chunk 1: accumulate into acc
                a1 = pl.load(a, [0, kc], [1, kc], valid_shapes=a_v, target_memory=pl.MemorySpace.Mat)
                b1 = pl.load(b, [kc, 0], [kc, n], valid_shapes=b_v, target_memory=pl.MemorySpace.Mat)
                acc = pl.tile.gemv_acc(acc, a1, b1)
                out = pl.store(acc, [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[1, ktot], pl.FP32],
                b: pl.Tensor[[ktot, n], pl.FP32],
                out: pl.Out[pl.Tensor[[1, n], pl.FP32]],
            ) -> pl.Tensor[[1, n], pl.FP32]:
                out = self.kernel(a, b, out)
                return out

        return GemvAccProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"].to(torch.float32)
        b = tensors["b"].to(torch.float32)
        kc, num = self._k_chunk, self.NUM_CHUNKS
        res = torch.zeros(1, self._n)
        for kb in range(num):
            k0 = kb * kc
            vkc = VALID_K if self._narrow == "K" else kc
            res = res + torch.matmul(a[:, k0 : k0 + vkc], b[k0 : k0 + vkc, :])
        tensors["out"][:] = res


class TestGemvAcc:
    """Cube gemv_acc (C[1,N] += A[1,k]@B[k,N] over K chunks) on a2a3."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("n", [64, 128], ids=["n64", "n128"])
    def test_tile_gemv_acc(self, test_runner, n):
        result = test_runner.run(GemvAccTestCase(n=n, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_gemv_acc_narrow_k(self, test_runner):
        """valid_shape != shape: narrowed per-chunk contraction."""
        result = test_runner.run(GemvAccTestCase(narrow="K", config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
