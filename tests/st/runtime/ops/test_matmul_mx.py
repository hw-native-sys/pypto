# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 runtime tests for host-prequantized MX matmul.

Native packed-FP4 matmul is intentionally outside the supported surface. The
FP4×FP8 case explicitly casts its FP4 lhs to FP8E4M3FN before validating both
``matmul_mx`` and ``matmul_mx_acc`` against an FP32 torch golden. A homogeneous
MXFP8 case remains as the instruction baseline.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

pytestmark = pytest.mark.skip(
    reason=(
        "PTOAS temporarily pinned to v0.57 (revert #2523 level3 TMP for pypto#2558); "
        "MX pack/layout requires PTOAS v0.60 — re-enable after tile.ci root-cause fix"
    ),
)

_REQUIRED_TORCH_DTYPES = ("float4_e2m1fn_x2", "float8_e4m3fn", "float8_e8m0fnu")
if not all(hasattr(torch, name) for name in _REQUIRED_TORCH_DTYPES):
    pytest.skip("torch MXFP4/MXFP8/E8M0 dtypes required", allow_module_level=True)

M, K, N = 64, 128, 128
MX_GROUP_SIZE = 32
SCALE_BLOCK_SIZE = 16
SCALE_C0_SIZE = 2

_MX_DATA_DTYPES = (DataType.FP4, DataType.FP8E4M3FN)


def _pack_a_scale(scale_codes: torch.Tensor) -> torch.Tensor:
    """Pack logical A scales into the MX_A_ZZ physical layout."""
    m, k_groups = scale_codes.shape
    assert m % SCALE_BLOCK_SIZE == 0
    assert k_groups % SCALE_C0_SIZE == 0
    return (
        scale_codes.reshape(
            m // SCALE_BLOCK_SIZE,
            SCALE_BLOCK_SIZE,
            k_groups // SCALE_C0_SIZE,
            SCALE_C0_SIZE,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(m, k_groups)
    )


def _unpack_a_scale(packed_codes: torch.Tensor) -> torch.Tensor:
    """Restore MX_A_ZZ physical scale bytes to logical [M, K/32]."""
    m, k_groups = packed_codes.shape
    return (
        packed_codes.reshape(
            m // SCALE_BLOCK_SIZE,
            k_groups // SCALE_C0_SIZE,
            SCALE_BLOCK_SIZE,
            SCALE_C0_SIZE,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(m, k_groups)
    )


def _pack_b_scale(scale_codes: torch.Tensor) -> torch.Tensor:
    """Pack logical B scales into the MX_B_NN physical layout."""
    k_groups, n = scale_codes.shape
    assert k_groups % SCALE_C0_SIZE == 0
    assert n % SCALE_BLOCK_SIZE == 0
    return (
        scale_codes.reshape(
            k_groups // SCALE_C0_SIZE,
            SCALE_C0_SIZE,
            n // SCALE_BLOCK_SIZE,
            SCALE_BLOCK_SIZE,
        )
        .permute(2, 0, 3, 1)
        .contiguous()
        .reshape(k_groups, n)
    )


def _unpack_b_scale(packed_codes: torch.Tensor) -> torch.Tensor:
    """Restore MX_B_NN physical scale bytes to logical [K/32, N]."""
    k_groups, n = packed_codes.shape
    return (
        packed_codes.reshape(
            n // SCALE_BLOCK_SIZE,
            k_groups // SCALE_C0_SIZE,
            SCALE_BLOCK_SIZE,
            SCALE_C0_SIZE,
        )
        .permute(1, 3, 0, 2)
        .contiguous()
        .reshape(k_groups, n)
    )


def _decode_fp4_data(data: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Decode a physical x2 FP4 tensor into its logical float64 matrix."""
    packed = data.contiguous().view(torch.uint8).reshape(rows, cols // 2)
    codes = torch.empty((rows, cols), dtype=torch.long)
    codes[:, 0::2] = (packed & 0x0F).to(torch.long)
    codes[:, 1::2] = ((packed >> 4) & 0x0F).to(torch.long)
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float64,
    )
    return values[codes]


def _matmul_mx_golden(
    a: torch.Tensor,
    a_scale_codes: torch.Tensor,
    b: torch.Tensor,
    b_scale_codes: torch.Tensor,
) -> torch.Tensor:
    """Compute MX matmul from decoded values and logical per-group E8M0 scales."""
    k_group = torch.arange(K) // MX_GROUP_SIZE
    a_scale = torch.pow(2.0, a_scale_codes.to(torch.float64) - 127)
    b_scale = torch.pow(2.0, b_scale_codes.to(torch.float64) - 127)
    a_scaled = a.to(torch.float64) * a_scale[:, k_group]
    b_scaled = b.to(torch.float64) * b_scale[k_group, :]
    return torch.matmul(a_scaled, b_scaled).to(torch.float32)


class MatmulMxTestCase(PTOTestCase):
    """One host-prequantized MX dtype combination with base+acc outputs."""

    __test__ = False

    @staticmethod
    def _make_fp4_data(shape: tuple[int, int], generator: torch.Generator) -> torch.Tensor:
        """Create signed finite FP4 data in torch's physical x2 representation."""
        _, cols = shape
        assert cols % 2 == 0
        codes = torch.randint(0, 16, shape, generator=generator).to(torch.uint8)
        codes.reshape(-1)[:16] = torch.arange(16, dtype=torch.uint8)
        assert torch.any(codes < 8)
        assert torch.any(codes >= 8)
        packed = ((codes[:, 1::2] & 0x0F) << 4) | (codes[:, 0::2] & 0x0F)
        return packed.contiguous().view(torch.float4_e2m1fn_x2)

    @staticmethod
    def _make_fp8_data(shape: tuple[int, int], generator: torch.Generator) -> torch.Tensor:
        """Create finite E4M3 data."""
        return torch.randint(-2, 3, shape, generator=generator).to(torch.float8_e4m3fn)

    @staticmethod
    def _make_case_inputs(lhs_fp4: bool) -> tuple[torch.Tensor, ...]:
        """Build deterministic host-prequantized inputs for one dtype combination."""
        generator = torch.Generator().manual_seed(23 + lhs_fp4 * 7)
        make_data = MatmulMxTestCase._make_fp4_data if lhs_fp4 else MatmulMxTestCase._make_fp8_data
        a = make_data((M, K), generator)
        b = MatmulMxTestCase._make_fp8_data((K, N), generator)

        a_scale_codes = torch.randint(126, 130, (M, K // MX_GROUP_SIZE), generator=generator).to(torch.uint8)
        b_scale_codes = torch.randint(126, 130, (K // MX_GROUP_SIZE, N), generator=generator).to(torch.uint8)
        assert torch.unique(a_scale_codes).numel() > 1
        assert torch.unique(b_scale_codes).numel() > 1
        a_scale = _pack_a_scale(a_scale_codes).view(torch.float8_e8m0fnu)
        b_scale = _pack_b_scale(b_scale_codes).view(torch.float8_e8m0fnu)
        return a, a_scale, b, b_scale

    @staticmethod
    def _build_mxfp8_program():
        """Build the homogeneous MXFP8×MXFP8 base+acc baseline."""

        @pl.program
        class MatmulMxHomogeneousProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[M, K], pl.FP8E4M3FN],
                a_scale: pl.Tensor[[M, K // 32], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[K, N], pl.FP8E4M3FN],
                b_scale: pl.Tensor[[K // 32, N], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> tuple[pl.Tensor[[M, N], pl.FP32], pl.Tensor[[M, N], pl.FP32]]:
                lhs = pl.move(
                    pl.load(a, [0, 0], [M, K], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.Left,
                )
                lhs_scale = pl.move(
                    pl.load(a_scale, [0, 0], [M, K // 32], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.LeftScale,
                )
                rhs = pl.move(
                    pl.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.Right,
                )
                rhs_scale = pl.move(
                    pl.load(b_scale, [0, 0], [K // 32, N], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.RightScale,
                )
                base = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)
                out = pl.store(base, [0, 0], out)
                accumulated = pl.matmul_mx_acc(base, lhs, lhs_scale, rhs, rhs_scale)
                out_acc = pl.store(accumulated, [0, 0], out_acc)
                return out, out_acc

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, K], pl.FP8E4M3FN],
                a_scale: pl.Tensor[[M, K // 32], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[K, N], pl.FP8E4M3FN],
                b_scale: pl.Tensor[[K // 32, N], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> tuple[pl.Tensor[[M, N], pl.FP32], pl.Tensor[[M, N], pl.FP32]]:
                out, out_acc = self.kernel(a, a_scale, b, b_scale, out, out_acc)
                return out, out_acc

        return MatmulMxHomogeneousProgram

    @staticmethod
    def _build_fp4_fp8_program():
        """Build MXFP4×MXFP8 base+acc coverage with an explicit lhs cast."""

        @pl.program
        class MatmulMxFp4Fp8Program:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[M, K], pl.FP4],
                a_scale: pl.Tensor[[M, K // 32], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[K, N], pl.FP8E4M3FN],
                b_scale: pl.Tensor[[K // 32, N], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> tuple[pl.Tensor[[M, N], pl.FP32], pl.Tensor[[M, N], pl.FP32]]:
                lhs_fp8 = pl.cast(pl.load(a, [0, 0], [M, K]), pl.FP8E4M3FN)
                lhs_mat = pl.move(lhs_fp8, target_memory=pl.Mem.Mat)
                lhs = pl.move(lhs_mat, target_memory=pl.Mem.Left)
                lhs_scale = pl.move(
                    pl.load(a_scale, [0, 0], [M, K // 32], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.LeftScale,
                )
                rhs = pl.move(
                    pl.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.Right,
                )
                rhs_scale = pl.move(
                    pl.load(b_scale, [0, 0], [K // 32, N], target_memory=pl.Mem.Mat),
                    target_memory=pl.Mem.RightScale,
                )
                base = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)
                out = pl.store(base, [0, 0], out)
                accumulated = pl.matmul_mx_acc(base, lhs, lhs_scale, rhs, rhs_scale)
                out_acc = pl.store(accumulated, [0, 0], out_acc)
                return out, out_acc

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[M, K], pl.FP4],
                a_scale: pl.Tensor[[M, K // 32], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[K, N], pl.FP8E4M3FN],
                b_scale: pl.Tensor[[K // 32, N], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
                out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> tuple[pl.Tensor[[M, N], pl.FP32], pl.Tensor[[M, N], pl.FP32]]:
                out, out_acc = self.kernel(a, a_scale, b, b_scale, out, out_acc)
                return out, out_acc

        return MatmulMxFp4Fp8Program

    def __init__(self, lhs_dtype: DataType, rhs_dtype: DataType):
        if lhs_dtype not in _MX_DATA_DTYPES or rhs_dtype != DataType.FP8E4M3FN:
            raise ValueError(
                "Supported MX matmul pairs are FP8E4M3FN×FP8E4M3FN and "
                f"FP4×FP8E4M3FN with an explicit lhs cast; got {lhs_dtype}, {rhs_dtype}"
            )
        super().__init__(RunConfig(rtol=0.0, atol=0.0), platform="a5")
        self._lhs_dtype = lhs_dtype
        self._rhs_dtype = rhs_dtype
        self._lhs_fp4 = lhs_dtype == DataType.FP4

    def get_name(self) -> str:
        return f"matmul_mx_{self._lhs_dtype.value}_x_{self._rhs_dtype.value}_base_acc"

    def define_tensors(self) -> list[TensorSpec]:
        a, a_scale, b, b_scale = self._make_case_inputs(self._lhs_fp4)
        return [
            TensorSpec("a", list(a.shape), self._lhs_dtype, init_value=a),
            TensorSpec("a_scale", list(a_scale.shape), DataType.FP8E8M0, init_value=a_scale),
            TensorSpec("b", list(b.shape), self._rhs_dtype, init_value=b),
            TensorSpec("b_scale", list(b_scale.shape), DataType.FP8E8M0, init_value=b_scale),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
            TensorSpec("out_acc", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        if self._lhs_fp4:
            return self._build_fp4_fp8_program()
        return self._build_mxfp8_program()

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = _decode_fp4_data(tensors["a"], M, K) if self._lhs_fp4 else tensors["a"].to(torch.float64)
        b = tensors["b"].to(torch.float64)
        a_scale_codes = _unpack_a_scale(tensors["a_scale"].view(torch.uint8))
        b_scale_codes = _unpack_b_scale(tensors["b_scale"].view(torch.uint8))
        base = _matmul_mx_golden(a, a_scale_codes, b, b_scale_codes)
        tensors["out"][:] = base
        tensors["out_acc"][:] = 2 * base


@pytest.mark.platforms("a5")
class TestMatmulMx:
    """Numerical execution coverage for the supported A5 MX dtype pairs."""

    @pytest.mark.parametrize(
        ("lhs_dtype", "rhs_dtype"),
        [
            pytest.param(DataType.FP8E4M3FN, DataType.FP8E4M3FN, id="mxfp8-x-mxfp8"),
            pytest.param(DataType.FP4, DataType.FP8E4M3FN, id="mxfp4-x-mxfp8"),
        ],
    )
    def test_matmul_mx_base_and_acc(self, test_runner, lhs_dtype, rhs_dtype):
        case = MatmulMxTestCase(lhs_dtype, rhs_dtype)
        result = test_runner.run(case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
