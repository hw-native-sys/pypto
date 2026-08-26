# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for the tile-level Cube ops matmul_bias and the GEMV family.

matmul_bias: C[M,N] = A[M,K] @ B[K,N] + bias[1,N]. Operands load to Mat (L1);
the layout passes (AutoTileMatmulL0 / CanonicalizeTileSlice) insert the L0
Left/Right extracts. Coverage: several M/K/N shapes (incl. non-square and a
K=128 case that forces AutoTileMatmulL0 to K-split), BF16 inputs with an FP32
accumulator, narrowed valid_shape on the output rows (M) and the contraction
(K), and a non-zero output row offset. Cube accumulation reorders the K
reduction vs torch, so a relaxed FP32 tolerance is used.

gemv / gemv_bias / gemv_acc are the M==1 specialization of the Cube matmul
family: C[1,N] = A[1,K] @ B[K,N], optionally + bias[1,N] or accumulated into
an existing result. Operands load to Mat (L1) and move to Left/Right; the
single-row lhs uses PTO-ISA's Mat-to-Left vector path. Coverage includes
several K/N shapes, every A2/A3 datatype triple (INT8, FP16, BF16, FP32),
contraction/output/combined valid-shape tails, and exact base/bias/acc
instruction forms. AccPhase coverage includes standalone Final base/bias
operations and a Partial base followed by a Final accumulate. A row tail is
not applicable because the GEMV verifier requires the logical row extent to
be exactly one.

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
VALID_N = 30
VALID_M = 8

_RTOL = 1e-3
_ATOL = 1e-3

_PL_DT = {
    DataType.FP32: pl.FP32,
    DataType.BF16: pl.BF16,
    DataType.FP16: pl.FP16,
    DataType.INT8: pl.INT8,
}


def _gemv_input(shape: list[int], dtype: DataType) -> torch.Tensor:
    if dtype in (DataType.INT8, DataType.INT32):
        return torch.randint(-4, 5, shape, dtype=dtype.torch_dtype)
    return torch.randn(shape, dtype=dtype.torch_dtype)


def _gemv_output_dtype(dtype: DataType) -> DataType:
    return DataType.INT32 if dtype == DataType.INT8 else DataType.FP32


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
            TensorSpec(
                "out",
                [self._out_m, self._n],
                DataType.FP32,
                init_value=torch.zeros,
                is_output=True,
            ),
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
                out: pl.InOut[pl.Tensor[[om, n], pl.FP32]],
            ) -> pl.Tensor[[om, n], pl.FP32]:
                tile_a = pl.load(a, [0, 0], [m, k], valid_shape=a_v, target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(b, [0, 0], [k, n], valid_shape=b_v, target_memory=pl.MemorySpace.Mat)
                tile_bias = pl.load(
                    bias, [0, 0], [1, n], valid_shape=vn_bias, target_memory=pl.MemorySpace.Mat
                )
                out = pl.store(pl.tile.matmul_bias(tile_a, tile_b, tile_bias), off, out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[m, k], ab],
                b: pl.Tensor[[k, n], ab],
                bias: pl.Tensor[[1, n], pl.FP32],
                out: pl.InOut[pl.Tensor[[om, n], pl.FP32]],
            ) -> pl.Tensor[[om, n], pl.FP32]:
                out = self.kernel(a, b, bias, out)
                return out

        return MatmulBiasProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"].to(torch.float32)
        b = tensors["b"].to(torch.float32)
        bias = tensors["bias"]
        out = tensors["out"].clone()
        if self._narrow == "K":
            full = torch.matmul(a[:, :VALID_N], b[:VALID_N, :]) + bias
        else:
            full = torch.matmul(a, b) + bias
        if self._narrow == "M":
            out[self._off_row : self._off_row + VALID_M, :] = full[:VALID_M, :]
        elif self._narrow == "N":
            out[self._off_row : self._off_row + self._m, :VALID_N] = full[:, :VALID_N]
        else:
            out[self._off_row : self._off_row + self._m, :] = full
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

# The Cube GEMV lhs is extracted to Left through PTO-ISA's single-row vector
# path. The physical K must occupy a whole 512-byte Cube block: K % 128 == 0
# for FP32, K % 256 == 0 for BF16/FP16, and K % 512 == 0 for INT8. A narrowed
# contraction retains an aligned physical K and changes only its valid extent.
VALID_K = 64


class GemvTestCase(PTOTestCase):
    """C[1,N] = A[1,K] @ B[K,N], optionally + bias[1,N]."""

    __test__ = False

    def __init__(
        self,
        *,
        k=K,
        n=N,
        bias=False,
        narrow=None,
        ab_dtype=DataType.FP32,
        acc_phase="unspecified",
        config=None,
    ):
        super().__init__(config)
        self._k, self._n = k, n
        self._bias, self._narrow, self._ab = bias, narrow, ab_dtype
        self._acc_phase = acc_phase

    def get_name(self) -> str:
        op = "gemv_bias" if self._bias else "gemv"
        narrowed = f"_n{self._narrow}" if self._narrow else ""
        phase = f"_{self._acc_phase}" if self._acc_phase != "unspecified" else ""
        return f"tile_{op}_1x{self._k}x{self._n}_{self._ab.value}{narrowed}{phase}"

    def define_tensors(self) -> list[TensorSpec]:
        out_dtype = _gemv_output_dtype(self._ab)
        valid_n = VALID_N if self._narrow in ("N", "KN") else self._n
        # N tails are compact in GM; padding belongs only to the aligned Mat/Right/Bias tiles.
        specs = [
            TensorSpec("a", [1, self._k], self._ab, init_value=lambda: _gemv_input([1, self._k], self._ab)),
            TensorSpec(
                "b",
                [self._k, valid_n],
                self._ab,
                init_value=lambda: _gemv_input([self._k, valid_n], self._ab),
            ),
        ]
        if self._bias:
            specs.append(
                TensorSpec(
                    "bias",
                    [1, valid_n],
                    out_dtype,
                    init_value=lambda: _gemv_input([1, valid_n], out_dtype),
                )
            )
        specs.append(TensorSpec("out", [1, valid_n], out_dtype, is_output=True))
        return specs

    def get_program(self) -> Any:
        k, n = self._k, self._n
        ab = _PL_DT[self._ab]
        out_dt = pl.INT32 if self._ab == DataType.INT8 else pl.FP32
        valid_k = VALID_K if self._narrow in ("K", "KN") else k
        valid_n = VALID_N if self._narrow in ("N", "KN") else n
        acc_phase = self._acc_phase
        a_valid = [1, valid_k]
        b_valid = [valid_k, valid_n]

        if not self._bias:

            @pl.program
            class GemvProgram:
                @pl.function(type=pl.FunctionType.InCore)
                def kernel(
                    self,
                    a: pl.Tensor[[1, k], ab],
                    b: pl.Tensor[[k, valid_n], ab],
                    out: pl.Out[pl.Tensor[[1, valid_n], out_dt]],
                ) -> pl.Tensor[[1, valid_n], out_dt]:
                    tile_a = pl.load(
                        a,
                        [0, 0],
                        [1, k],
                        valid_shape=a_valid,
                        target_memory=pl.MemorySpace.Mat,
                    )
                    tile_b = pl.load(
                        b,
                        [0, 0],
                        [k, n],
                        valid_shape=b_valid,
                        target_memory=pl.MemorySpace.Mat,
                        clamp=True,
                    )
                    out = pl.store(pl.tile.gemv(tile_a, tile_b, acc_phase=acc_phase), [0, 0], out)
                    return out

                @pl.function(type=pl.FunctionType.Orchestration)
                def orchestrator(
                    self,
                    a: pl.Tensor[[1, k], ab],
                    b: pl.Tensor[[k, valid_n], ab],
                    out: pl.Out[pl.Tensor[[1, valid_n], out_dt]],
                ) -> pl.Tensor[[1, valid_n], out_dt]:
                    out = self.kernel(a, b, out)
                    return out

            return GemvProgram

        @pl.program
        class GemvBiasProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[1, k], ab],
                b: pl.Tensor[[k, valid_n], ab],
                bias: pl.Tensor[[1, valid_n], out_dt],
                out: pl.Out[pl.Tensor[[1, valid_n], out_dt]],
            ) -> pl.Tensor[[1, valid_n], out_dt]:
                tile_a = pl.load(a, [0, 0], [1, k], valid_shape=a_valid, target_memory=pl.MemorySpace.Mat)
                tile_b = pl.load(
                    b,
                    [0, 0],
                    [k, n],
                    valid_shape=b_valid,
                    target_memory=pl.MemorySpace.Mat,
                    clamp=True,
                )
                tile_bias = pl.load(
                    bias,
                    [0, 0],
                    [1, n],
                    valid_shape=[1, valid_n],
                    target_memory=pl.MemorySpace.Mat,
                    clamp=True,
                )
                result = pl.tile.gemv_bias(tile_a, tile_b, tile_bias, acc_phase=acc_phase)
                out = pl.store(result, [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[1, k], ab],
                b: pl.Tensor[[k, valid_n], ab],
                bias: pl.Tensor[[1, valid_n], out_dt],
                out: pl.Out[pl.Tensor[[1, valid_n], out_dt]],
            ) -> pl.Tensor[[1, valid_n], out_dt]:
                out = self.kernel(a, b, bias, out)
                return out

        return GemvBiasProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        out_dtype = _gemv_output_dtype(self._ab).torch_dtype
        if self._ab == DataType.INT8:
            a = tensors["a"].to(torch.int32)
            b = tensors["b"].to(torch.int32)
        else:
            a = tensors["a"].to(torch.float32)
            b = tensors["b"].to(torch.float32)
        valid_k = VALID_K if self._narrow in ("K", "KN") else self._k
        valid_n = VALID_N if self._narrow in ("N", "KN") else self._n
        result = torch.matmul(a[:, :valid_k], b[:valid_k, :valid_n])
        if self._bias:
            result = result + tensors["bias"][:, :valid_n]
        tensors["out"][:] = result.to(out_dtype)


# GEMV keeps the whole rhs resident in the 64-KiB Right buffer rather than
# K-splitting it, so these K/N pairs stay within that capacity.
_KN = [(128, 64), (256, 64), (128, 128)]


class TestGemv:
    """Cube gemv on A2/A3 across K/N, valid-region tails, and supported input dtypes."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("k,n", _KN, ids=[f"1x{k}x{n}" for k, n in _KN])
    def test_tile_gemv(self, test_runner, k, n):
        result = test_runner.run(GemvTestCase(k=k, n=n, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_gemv_narrow_k(self, test_runner):
        result = test_runner.run(GemvTestCase(k=128, narrow="K", config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_gemv_bf16(self, test_runner):
        result = test_runner.run(GemvTestCase(k=256, n=64, ab_dtype=DataType.BF16, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_gemv_fp16(self, test_runner):
        result = test_runner.run(GemvTestCase(k=256, n=64, ab_dtype=DataType.FP16, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_gemv_int8(self, test_runner):
        result = test_runner.run(GemvTestCase(k=512, n=64, ab_dtype=DataType.INT8, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.skip(
        reason="Issue #2428: GEMV with acc_phase='final' intermittently hangs on device "
        "(SCHEDULER_TIMEOUT sub_class=S1:running-stalled), which poisons the shared chip-run "
        "lane and fails every remaining artifact in the ST batch. Temporarily skipped to "
        "unblock CI; re-enable once the hang is root-caused."
    )
    @pytest.mark.platforms("a2a3")
    def test_tile_gemv_final_phase(self, test_runner):
        result = test_runner.run(GemvTestCase(k=128, n=64, acc_phase="final", config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("narrow", ["N", "KN"])
    def test_tile_gemv_output_tail(self, test_runner, narrow):
        result = test_runner.run(GemvTestCase(k=128, n=32, narrow=narrow, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"


class TestGemvBias:
    """Cube gemv_bias on A2/A3 across K/N, valid-region tails, and supported dtypes."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("k,n", _KN, ids=[f"1x{k}x{n}" for k, n in _KN])
    def test_tile_gemv_bias(self, test_runner, k, n):
        result = test_runner.run(GemvTestCase(k=k, n=n, bias=True, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_gemv_bias_narrow_k(self, test_runner):
        result = test_runner.run(GemvTestCase(k=128, bias=True, narrow="K", config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("ab_dtype", [DataType.BF16, DataType.FP16, DataType.INT8])
    def test_tile_gemv_bias_dtype(self, test_runner, ab_dtype):
        k = 512 if ab_dtype == DataType.INT8 else 256
        result = test_runner.run(GemvTestCase(k=k, n=64, bias=True, ab_dtype=ab_dtype, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.skip(
        reason="Issue #2428: GEMV with acc_phase='final' intermittently hangs on device "
        "(SCHEDULER_TIMEOUT sub_class=S1:running-stalled), which poisons the shared chip-run "
        "lane and fails every remaining artifact in the ST batch. Temporarily skipped to "
        "unblock CI; re-enable once the hang is root-caused."
    )
    @pytest.mark.platforms("a2a3")
    def test_tile_gemv_bias_final_phase(self, test_runner):
        result = test_runner.run(GemvTestCase(k=128, n=64, bias=True, acc_phase="final", config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("narrow", ["N", "KN"])
    def test_tile_gemv_bias_output_tail(self, test_runner, narrow):
        result = test_runner.run(GemvTestCase(k=128, n=32, bias=True, narrow=narrow, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"


# ===========================================================================
# gemv_acc (C[1,N] += A[1,K] @ B[K,N])
# ===========================================================================


class GemvAccTestCase(PTOTestCase):
    """Accumulate two K chunks through one fresh gemv and one gemv_acc."""

    __test__ = False
    NUM_CHUNKS = 2

    def __init__(
        self,
        *,
        k_chunk=128,
        n=N,
        narrow=None,
        ab_dtype=DataType.FP32,
        phased=False,
        config=None,
    ):
        super().__init__(config)
        self._k_chunk, self._n, self._narrow = k_chunk, n, narrow
        self._ab = ab_dtype
        self._phased = phased
        self._k = k_chunk * self.NUM_CHUNKS

    def get_name(self) -> str:
        narrowed = f"_n{self._narrow}" if self._narrow else ""
        phase = "_partial_final" if self._phased else ""
        return (
            f"tile_gemv_acc_1x{self._k}x{self._n}_{self._ab.value}_chunks{self.NUM_CHUNKS}{narrowed}{phase}"
        )

    def define_tensors(self) -> list[TensorSpec]:
        out_dtype = _gemv_output_dtype(self._ab)
        valid_n = VALID_N if self._narrow in ("N", "KN") else self._n
        # Keep the logical GM row stride while each on-chip rhs tile stays physically aligned.
        return [
            TensorSpec("a", [1, self._k], self._ab, init_value=lambda: _gemv_input([1, self._k], self._ab)),
            TensorSpec(
                "b",
                [self._k, valid_n],
                self._ab,
                init_value=lambda: _gemv_input([self._k, valid_n], self._ab),
            ),
            TensorSpec("out", [1, valid_n], out_dtype, is_output=True),
        ]

    def get_program(self) -> Any:
        k_chunk, n, k_total = self._k_chunk, self._n, self._k
        ab = _PL_DT[self._ab]
        out_dt = pl.INT32 if self._ab == DataType.INT8 else pl.FP32
        valid_k = VALID_K if self._narrow in ("K", "KN") else k_chunk
        valid_n = VALID_N if self._narrow in ("N", "KN") else n
        first_phase = "partial" if self._phased else "unspecified"
        last_phase = "final" if self._phased else "unspecified"
        a_valid = [1, valid_k]
        b_valid = [valid_k, valid_n]

        @pl.program
        class GemvAccProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[1, k_total], ab],
                b: pl.Tensor[[k_total, valid_n], ab],
                out: pl.Out[pl.Tensor[[1, valid_n], out_dt]],
            ) -> pl.Tensor[[1, valid_n], out_dt]:
                a0 = pl.load(a, [0, 0], [1, k_chunk], valid_shape=a_valid, target_memory=pl.MemorySpace.Mat)
                b0 = pl.load(
                    b,
                    [0, 0],
                    [k_chunk, n],
                    valid_shape=b_valid,
                    target_memory=pl.MemorySpace.Mat,
                    clamp=True,
                )
                acc = pl.tile.gemv(a0, b0, acc_phase=first_phase)

                a1 = pl.load(
                    a,
                    [0, k_chunk],
                    [1, k_chunk],
                    valid_shape=a_valid,
                    target_memory=pl.MemorySpace.Mat,
                )
                b1 = pl.load(
                    b,
                    [k_chunk, 0],
                    [k_chunk, n],
                    valid_shape=b_valid,
                    target_memory=pl.MemorySpace.Mat,
                    clamp=True,
                )
                acc = pl.tile.gemv_acc(acc, a1, b1, acc_phase=last_phase)
                out = pl.store(acc, [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[1, k_total], ab],
                b: pl.Tensor[[k_total, valid_n], ab],
                out: pl.Out[pl.Tensor[[1, valid_n], out_dt]],
            ) -> pl.Tensor[[1, valid_n], out_dt]:
                out = self.kernel(a, b, out)
                return out

        return GemvAccProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        out_dtype = _gemv_output_dtype(self._ab).torch_dtype
        if self._ab == DataType.INT8:
            a = tensors["a"].to(torch.int32)
            b = tensors["b"].to(torch.int32)
        else:
            a = tensors["a"].to(torch.float32)
            b = tensors["b"].to(torch.float32)
        valid_n = VALID_N if self._narrow in ("N", "KN") else self._n
        valid_k = VALID_K if self._narrow in ("K", "KN") else self._k_chunk
        result = torch.zeros(1, valid_n, dtype=out_dtype)
        for chunk in range(self.NUM_CHUNKS):
            start = chunk * self._k_chunk
            result = result + torch.matmul(
                a[:, start : start + valid_k],
                b[start : start + valid_k, :valid_n],
            ).to(out_dtype)
        tensors["out"][:] = result.to(out_dtype)


class GemvAccInitCondTestCase(PTOTestCase):
    """Split-K GEMV driven by ``init_cond`` instead of a peeled first step.

    ``GemvAccTestCase`` above spells the same reduction the old way: a
    straight-line ``tile.gemv`` for the first K chunk, then ``tile.gemv_acc`` for
    the rest. Here every chunk is one predicated ``tile.gemv_acc`` inside the
    loop, so the accumulator stays single-def.

    The property is numeric, not structural. The accumulator is minted by
    ``tile.create`` and never zeroed, so if ``init_cond`` failed to select the
    overwriting form on ``k0 == 0`` the result would carry whatever L0C held and
    the comparison against ``a @ b`` would fail.

    It is created at the *padded* physical shape -- a ``[1, N]`` GEMV result
    occupies 16 rows -- and then narrowed to its valid ``[1, N]`` rectangle. The
    peeled ``tile.gemv`` this replaces produced that type implicitly.
    """

    __test__ = False

    def __init__(self, *, k_chunk=128, chunks=2, n=N, config=None):
        super().__init__(config)
        self._k_chunk, self._chunks, self._n = k_chunk, chunks, n
        self._k = k_chunk * chunks

    def get_name(self) -> str:
        return f"tile_gemv_acc_init_cond_1x{self._k}x{self._n}_kt{self._k_chunk}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "a",
                [1, self._k],
                DataType.FP32,
                init_value=lambda: _gemv_input([1, self._k], DataType.FP32),
            ),
            TensorSpec(
                "b",
                [self._k, self._n],
                DataType.FP32,
                init_value=lambda: _gemv_input([self._k, self._n], DataType.FP32),
            ),
            TensorSpec("out", [1, self._n], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        k_chunk, n, k_total = self._k_chunk, self._n, self._k

        @pl.program
        class GemvAccInitCondProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[1, k_total], pl.FP32],
                b: pl.Tensor[[k_total, n], pl.FP32],
                out: pl.Out[pl.Tensor[[1, n], pl.FP32]],
            ) -> pl.Tensor[[1, n], pl.FP32]:
                acc_raw = pl.tile.create([16, n], pl.FP32, target_memory=pl.MemorySpace.Acc)
                acc = pl.tile.set_validshape(acc_raw, 1, n)
                for k0 in pl.range(0, k_total, k_chunk):
                    a_l1 = pl.load(a, [0, k0], [1, k_chunk], target_memory=pl.MemorySpace.Mat)
                    b_l1 = pl.load(b, [k0, 0], [k_chunk, n], target_memory=pl.MemorySpace.Mat)
                    acc = pl.tile.gemv_acc(acc, a_l1, b_l1, init_cond=(k0 == 0))
                out = pl.store(acc, [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[1, k_total], pl.FP32],
                b: pl.Tensor[[k_total, n], pl.FP32],
                out: pl.Out[pl.Tensor[[1, n], pl.FP32]],
            ) -> pl.Tensor[[1, n], pl.FP32]:
                out = self.kernel(a, b, out)
                return out

        return GemvAccInitCondProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = tensors["a"].to(torch.float32)
        b = tensors["b"].to(torch.float32)
        tensors["out"][:] = torch.matmul(a, b).to(torch.float32)


class TestGemvAcc:
    """Cube gemv_acc on A2/A3 across N, valid-region tails, and supported dtypes."""

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("n", [64, 128], ids=["n64", "n128"])
    def test_tile_gemv_acc(self, test_runner, n):
        result = test_runner.run(GemvAccTestCase(n=n, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    def test_tile_gemv_acc_narrow_k(self, test_runner):
        result = test_runner.run(GemvAccTestCase(narrow="K", config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("ab_dtype", [DataType.BF16, DataType.FP16, DataType.INT8])
    def test_tile_gemv_acc_dtype(self, test_runner, ab_dtype):
        k_chunk = 512 if ab_dtype == DataType.INT8 else 256
        result = test_runner.run(GemvAccTestCase(k_chunk=k_chunk, n=64, ab_dtype=ab_dtype, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.skip(
        reason="Issue #2428: GEMV with acc_phase='final' intermittently hangs on device "
        "(SCHEDULER_TIMEOUT sub_class=S1:running-stalled), which poisons the shared chip-run "
        "lane and fails every remaining artifact in the ST batch. Temporarily skipped to "
        "unblock CI; re-enable once the hang is root-caused."
    )
    @pytest.mark.platforms("a2a3")
    def test_tile_gemv_acc_partial_final_phases(self, test_runner):
        result = test_runner.run(GemvAccTestCase(n=64, phased=True, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("narrow", ["N", "KN"])
    def test_tile_gemv_acc_output_tail(self, test_runner, narrow):
        result = test_runner.run(GemvAccTestCase(n=32, narrow=narrow, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.platforms("a2a3")
    @pytest.mark.parametrize("n", [64, 128], ids=["n64", "n128"])
    def test_tile_gemv_acc_init_cond(self, test_runner, n):
        """Split-K where ``init_cond=(k0 == 0)`` replaces the peeled first step."""
        result = test_runner.run(GemvAccInitCondTestCase(n=n, config=_cfg()))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
