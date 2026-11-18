# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 runtime tests for MXFP8 and native packed-FP4 matmul and accumulation.

The mixed FP4×FP8 case explicitly casts FP4 to FP8. Native FP4 cases retain
both packed formats through GM loading, L0 placement, and MX instructions.
"""

from functools import cache
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness import st
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

_REQUIRED_TORCH_DTYPES = ("float4_e2m1fn_x2", "float8_e4m3fn", "float8_e8m0fnu")
if not all(hasattr(torch, name) for name in _REQUIRED_TORCH_DTYPES):
    pytest.skip("torch MXFP4/MXFP8/E8M0 dtypes required", allow_module_level=True)

M, K, N = 64, 128, 128
MX_GROUP_SIZE = 32
SCALE_BLOCK_SIZE = 16
SCALE_C0_SIZE = 2

_MX_FP8_DTYPES = (DataType.FP8E4M3FN, DataType.FP8E5M2)


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
    k_group = torch.arange(a.shape[1]) // MX_GROUP_SIZE
    a_scale = torch.pow(2.0, a_scale_codes.to(torch.float64) - 127)
    b_scale = torch.pow(2.0, b_scale_codes.to(torch.float64) - 127)
    a_scaled = a.to(torch.float64) * a_scale[:, k_group]
    b_scaled = b.to(torch.float64) * b_scale[k_group, :]
    return torch.matmul(a_scaled, b_scaled).to(torch.float32)


def _mx_matmul_entry(lhs_dtype: DataType, rhs_dtype: DataType):
    """Build the jitted entry for one FP8 pair.

    The operand dtypes appear in the kernel's annotations, so one entry is
    specialized per pair rather than branching inside a single kernel.
    """
    lhs_pl = pl.FP8E5M2 if lhs_dtype == DataType.FP8E5M2 else pl.FP8E4M3FN
    rhs_pl = pl.FP8E5M2 if rhs_dtype == DataType.FP8E5M2 else pl.FP8E4M3FN

    @pl.jit.incore
    def mx_matmul_kernel(
        a: pl.Tensor[[M, K], lhs_pl],
        a_scale: pl.Tensor[[M, K // 32], pl.FP8E8M0, pl.MX_A_ZZ],
        b: pl.Tensor[[K, N], rhs_pl],
        b_scale: pl.Tensor[[K // 32, N], pl.FP8E8M0, pl.MX_B_NN],
        bias: pl.Tensor[[1, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        out_bias: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> tuple[
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
    ]:
        # Whole scale views avoid PTOAS #1763 (packed partition offsets and parent strides).
        lhs = pl.load(a, [0, 0], [M, K])
        lhs_scale = pl.load(a_scale, [0, 0], [M, K // 32])
        rhs = pl.load(b, [0, 0], [K, N])
        rhs_scale = pl.load(b_scale, [0, 0], [K // 32, N])
        bias_tile = pl.load(bias, [0, 0], [1, N])
        base = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)
        out = pl.store(base, [0, 0], out)
        out_acc = pl.store(pl.matmul_mx_acc(base, lhs, lhs_scale, rhs, rhs_scale), [0, 0], out_acc)
        out_bias = pl.store(pl.matmul_mx_bias(lhs, lhs_scale, rhs, rhs_scale, bias_tile), [0, 0], out_bias)
        return out, out_acc, out_bias

    @pl.jit
    def mx_matmul(
        a: pl.Tensor[[M, K], lhs_pl],
        a_scale: pl.Tensor[[M, K // 32], pl.FP8E8M0, pl.MX_A_ZZ],
        b: pl.Tensor[[K, N], rhs_pl],
        b_scale: pl.Tensor[[K // 32, N], pl.FP8E8M0, pl.MX_B_NN],
        bias: pl.Tensor[[1, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        out_bias: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> tuple[
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
    ]:
        return mx_matmul_kernel(a, a_scale, b, b_scale, bias, out, out_acc, out_bias)

    return mx_matmul


@pl.jit.incore
def mxfp4_fp8_matmul_kernel(
    a: pl.Tensor[[M, K // 2], pl.FP4E2M1X2],
    a_scale: pl.Tensor[[M, K // 32], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[K, N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[K // 32, N], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
) -> tuple[pl.Tensor[[M, N], pl.FP32], pl.Tensor[[M, N], pl.FP32]]:
    # Torch stores two FP4 values per carrier; the cast restores logical K.
    lhs = pl.cast(pl.load(a, [0, 0], [M, K // 2]), pl.FP8E4M3FN)
    lhs_scale = pl.load(a_scale, [0, 0], [M, K // 32])
    rhs = pl.load(b, [0, 0], [K, N])
    rhs_scale = pl.load(b_scale, [0, 0], [K // 32, N])
    base = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)
    out = pl.store(base, [0, 0], out)
    accumulated = pl.matmul_mx_acc(base, lhs, lhs_scale, rhs, rhs_scale)
    out_acc = pl.store(accumulated, [0, 0], out_acc)
    return out, out_acc


@pl.jit
def mxfp4_fp8_matmul(
    a: pl.Tensor[[M, K // 2], pl.FP4E2M1X2],
    a_scale: pl.Tensor[[M, K // 32], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[K, N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[K // 32, N], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
) -> tuple[pl.Tensor[[M, N], pl.FP32], pl.Tensor[[M, N], pl.FP32]]:
    return mxfp4_fp8_matmul_kernel(a, a_scale, b, b_scale, out, out_acc)


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
    def _make_fp8_data(shape: tuple[int, int], generator: torch.Generator, dtype: DataType) -> torch.Tensor:
        """Use exactly representable values that distinguish the two FP8 formats."""
        values = torch.tensor([-2.0, -1.5, 0.5, 1.0, 2.0, 3.0])
        return values[torch.randint(len(values), shape, generator=generator)].to(dtype.torch_dtype)

    def _make_case_inputs(self) -> tuple[torch.Tensor, ...]:
        """Build deterministic host-prequantized inputs for one dtype combination."""
        generator = torch.Generator().manual_seed(23 + self._lhs_fp4 * 7)
        if self._lhs_fp4:
            a = self._make_fp4_data((M, K), generator)
        else:
            a = self._make_fp8_data((M, K), generator, self._lhs_dtype)
        b = self._make_fp8_data((K, N), generator, self._rhs_dtype)

        a_scale_codes = torch.randint(126, 130, (M, K // MX_GROUP_SIZE), generator=generator).to(torch.uint8)
        b_scale_codes = torch.randint(126, 130, (K // MX_GROUP_SIZE, N), generator=generator).to(torch.uint8)
        assert torch.unique(a_scale_codes).numel() > 1
        assert torch.unique(b_scale_codes).numel() > 1
        a_scale = _pack_a_scale(a_scale_codes).view(torch.float8_e8m0fnu)
        b_scale = _pack_b_scale(b_scale_codes).view(torch.float8_e8m0fnu)
        return a, a_scale, b, b_scale

    def __init__(self, lhs_dtype: DataType, rhs_dtype: DataType):
        super().__init__(RunConfig(rtol=0.0, atol=0.0), platform="a5")
        self._lhs_dtype = lhs_dtype
        self._rhs_dtype = rhs_dtype
        self._lhs_fp4 = lhs_dtype == DataType.FP4E2M1X2

    def get_name(self) -> str:
        return f"matmul_mx_{self._lhs_dtype.value}_x_{self._rhs_dtype.value}"

    def define_tensors(self) -> list[TensorSpec]:
        a, a_scale, b, b_scale = self._make_case_inputs()
        inputs = [
            TensorSpec("a", list(a.shape), self._lhs_dtype, init_value=a),
            TensorSpec("a_scale", list(a_scale.shape), DataType.FP8E8M0, init_value=a_scale),
            TensorSpec("b", list(b.shape), self._rhs_dtype, init_value=b),
            TensorSpec("b_scale", list(b_scale.shape), DataType.FP8E8M0, init_value=b_scale),
        ]
        outputs = [
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
            TensorSpec("out_acc", [M, N], DataType.FP32, is_output=True),
        ]
        if not self._lhs_fp4:
            bias = (torch.arange(N, dtype=torch.float32) % 7 - 3).reshape(1, N)
            inputs.append(TensorSpec("bias", [1, N], DataType.FP32, init_value=bias))
            outputs.append(TensorSpec("out_bias", [M, N], DataType.FP32, is_output=True))
        return inputs + outputs

    def get_program(self) -> Any:
        if self._lhs_fp4:
            return mxfp4_fp8_matmul.specialize()
        return _mx_matmul_entry(self._lhs_dtype, self._rhs_dtype).specialize()

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a = _decode_fp4_data(tensors["a"], M, K) if self._lhs_fp4 else tensors["a"].to(torch.float64)
        b = tensors["b"].to(torch.float64)
        a_scale_codes = _unpack_a_scale(tensors["a_scale"].view(torch.uint8))
        b_scale_codes = _unpack_b_scale(tensors["b_scale"].view(torch.uint8))
        base = _matmul_mx_golden(a, a_scale_codes, b, b_scale_codes)
        tensors["out"][:] = base
        tensors["out_acc"][:] = 2 * base
        if not self._lhs_fp4:
            tensors["out_bias"][:] = base + tensors["bias"]


@pytest.mark.platforms("a5")
class TestMatmulMx:
    """Numerical execution coverage for the supported A5 MX dtype pairs."""

    @pytest.mark.parametrize(
        ("lhs_dtype", "rhs_dtype"),
        [
            *[
                pytest.param(lhs, rhs, id=f"{lhs.value}-x-{rhs.value}")
                for lhs in _MX_FP8_DTYPES
                for rhs in _MX_FP8_DTYPES
            ],
            pytest.param(DataType.FP4E2M1X2, DataType.FP8E4M3FN, id="mxfp4-x-mxfp8"),
        ],
    )
    def test_matmul_mx_variants(self, test_runner, lhs_dtype, rhs_dtype):
        if DataType.FP8E5M2 in (lhs_dtype, rhs_dtype) and not hasattr(torch, "float8_e5m2"):
            pytest.skip("torch.float8_e5m2 is unavailable")
        case = MatmulMxTestCase(lhs_dtype, rhs_dtype)
        result = test_runner.run(case)
        assert result.passed, f"Test failed: {result.error}"


def _native_fp4_entry(lhs_dtype, rhs_dtype):
    @pl.jit(auto_scope=False)
    def entry(
        a: pl.Tensor[[M, K // 2], pl.UINT8],
        a_scale: pl.Tensor[[M, K // 32], pl.FP8E8M0, pl.MX_A_ZZ],
        b: pl.Tensor[[N, K // 2], pl.UINT8],
        b_scale: pl.Tensor[[K // 32, N], pl.FP8E8M0, pl.MX_B_NN],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ):
        a_fp4 = pl.reinterpret_view(a, lhs_dtype)
        b_fp4 = pl.reinterpret_view(b, rhs_dtype)
        with pl.spmd(1):
            row = pl.tile.get_block_idx() * M
            lhs_mat = pl.tile.load(a_fp4, [row, 0], [M, K // 2], target_memory=pl.MemorySpace.Mat)
            rhs_nk = pl.tile.load(b_fp4, [0, 0], [N, K // 2], target_memory=pl.MemorySpace.Mat)
            # Transpose the logical matrix without transposing packed bytes.
            rhs_mat = pl.tile.transpose_view(rhs_nk)
            lhs = pl.tile.move(lhs_mat, pl.MemorySpace.Left)
            rhs = pl.tile.move(rhs_mat, pl.MemorySpace.Right)
            sa = pl.tile.load(a_scale, [row, 0], [M, K // 32], target_memory=pl.MemorySpace.Mat)
            sb = pl.tile.load(b_scale, [0, 0], [K // 32, N], target_memory=pl.MemorySpace.Mat)
            sa_l0 = pl.tile.move(sa, pl.MemorySpace.LeftScale)
            sb_l0 = pl.tile.move(sb, pl.MemorySpace.RightScale)
            base = pl.matmul_mx(lhs, sa_l0, rhs, sb_l0)
            pl.tile.store(base, [row, 0], out)
            acc = pl.matmul_mx_acc(base, lhs, sa_l0, rhs, sb_l0)
            pl.tile.store(acc, [row, 0], out_acc)
        return out, out_acc

    return entry


def _native_fp4_specs():
    gen = torch.Generator().manual_seed(71)
    a = MatmulMxTestCase._make_fp4_data((M, K), gen).view(torch.uint8)
    b = MatmulMxTestCase._make_fp4_data((N, K), gen).view(torch.uint8)
    sa = _pack_a_scale(torch.randint(123, 130, (M, K // 32), generator=gen).to(torch.uint8))
    sb = _pack_b_scale(torch.randint(123, 130, (K // 32, N), generator=gen).to(torch.uint8))
    return [
        TensorSpec("a", [M, K // 2], DataType.UINT8, init_value=a),
        TensorSpec("a_scale", [M, K // 32], DataType.FP8E8M0, init_value=sa.view(torch.float8_e8m0fnu)),
        TensorSpec("b", [N, K // 2], DataType.UINT8, init_value=b),
        TensorSpec("b_scale", [K // 32, N], DataType.FP8E8M0, init_value=sb.view(torch.float8_e8m0fnu)),
        TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        TensorSpec("out_acc", [M, N], DataType.FP32, is_output=True),
    ]


def _native_fp4_golden(lhs_dtype, rhs_dtype):
    def decode(data, dtype):
        raw = data.view(torch.uint8)
        codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2).long()
        levels = (
            [0, 0.5, 1, 1.5, 2, 3, 4, 6]
            if dtype == pl.fp4e2m1x2
            else [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]
        )
        values = torch.tensor(levels + [-v for v in levels], dtype=torch.float64)
        return values[codes]

    def golden(tensors):
        a = decode(tensors["a"], lhs_dtype)
        b = decode(tensors["b"], rhs_dtype).T
        sa = _unpack_a_scale(tensors["a_scale"].view(torch.uint8))
        sb = _unpack_b_scale(tensors["b_scale"].view(torch.uint8))
        base = _matmul_mx_golden(a, sa, b, sb)
        return {"out": base, "out_acc": 2 * base}

    return golden


@pytest.mark.platforms("a5")
@st.cases(
    *[
        st.case(
            _native_fp4_entry(lhs, rhs),
            tensors=_native_fp4_specs(),
            name=f"native_mx_{lhs}_{rhs}",
            golden=_native_fp4_golden(lhs, rhs),
            rtol=0.0,
            atol=0.0,
        )
        for lhs in (pl.fp4e2m1x2, pl.fp4e1m2x2)
        for rhs in (pl.fp4e2m1x2, pl.fp4e1m2x2)
    ]
)
def test_native_fp4_matmul_mx(case_run):
    case_run.assert_passed()


def _gemv_mx_entry(dtype):
    @pl.jit(auto_scope=False)
    def entry(
        a: pl.Tensor[[1, 512], pl.UINT8],
        a_scale: pl.Tensor[[1, 2], pl.FP8E8M0, pl.MX_A_ND],
        b: pl.Tensor[[64, 512], pl.UINT8],
        b_scale: pl.Tensor[[2, 64], pl.FP8E8M0, pl.MX_B_NN],
        bias: pl.Tensor[[1, 64], pl.FP32],
        out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
        out_acc: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
        out_bias: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
    ):
        a_fp4 = pl.reinterpret_view(a, dtype)
        b_fp4 = pl.reinterpret_view(b, dtype)
        with pl.spmd(1):
            row = pl.tile.get_block_idx()
            # The 512-byte row satisfies the L0A alignment. Its 32 valid bytes
            # represent K=64 and consume two groups. N=64 exercises later Acc fractals.
            lhs = pl.tile.move(
                pl.tile.load(a_fp4, [row, 0], [1, 512], valid_shape=[1, 32], target_memory=pl.Mem.Mat),
                pl.Mem.Left,
            )
            rhs_nk = pl.tile.load(b_fp4, [0, 0], [64, 512], valid_shape=[64, 32], target_memory=pl.Mem.Mat)
            rhs = pl.tile.move(pl.tile.transpose_view(rhs_nk), pl.Mem.Right)
            # Use complete, zero-offset scale views to avoid PTOAS issue #1763.
            sa = pl.tile.move(
                pl.tile.load(a_scale, [0, 0], [1, 2], target_memory=pl.Mem.Mat), pl.Mem.LeftScale
            )
            sb = pl.tile.move(
                pl.tile.load(b_scale, [0, 0], [2, 64], target_memory=pl.Mem.Mat), pl.Mem.RightScale
            )
            bias_tile = pl.tile.move(
                pl.tile.load(bias, [0, 0], [1, 64], target_memory=pl.Mem.Mat), pl.Mem.Bias
            )
            base = pl.tile.gemv_mx(lhs, sa, rhs, sb)
            accumulated = pl.tile.gemv_mx_acc(base, lhs, sa, rhs, sb)
            pl.tile.store(accumulated, [row, 0], out_acc)
            biased = pl.tile.gemv_mx_bias(lhs, sa, rhs, sb, bias_tile)
            pl.tile.store(biased, [row, 0], out_bias)
            plain = pl.tile.gemv_mx(lhs, sa, rhs, sb)
            pl.tile.store(plain, [row, 0], out)
        return out, out_acc, out_bias

    return entry


def _gemv_mx_specs():
    # Large Torch operations can start OpenMP threads at collection time.
    # Defer input creation until the case runs, after pytest-forked forks.
    @cache
    def inputs():
        gen = torch.Generator().manual_seed(79)
        a = MatmulMxTestCase._make_fp4_data((1, 1024), gen).view(torch.uint8)
        b = MatmulMxTestCase._make_fp4_data((64, 1024), gen).view(torch.uint8)
        sa = torch.randint(126, 130, (1, 2), generator=gen).to(torch.uint8)
        sb = _pack_b_scale(torch.randint(126, 130, (2, 64), generator=gen).to(torch.uint8))
        bias = torch.randint(-4, 5, (1, 64), generator=gen).float()
        return a, sa.view(torch.float8_e8m0fnu), b, sb.view(torch.float8_e8m0fnu), bias

    return [
        TensorSpec("a", [1, 512], DataType.UINT8, init_value=lambda: inputs()[0]),
        TensorSpec("a_scale", [1, 2], DataType.FP8E8M0, init_value=lambda: inputs()[1]),
        TensorSpec("b", [64, 512], DataType.UINT8, init_value=lambda: inputs()[2]),
        TensorSpec("b_scale", [2, 64], DataType.FP8E8M0, init_value=lambda: inputs()[3]),
        TensorSpec("bias", [1, 64], DataType.FP32, init_value=lambda: inputs()[4]),
        TensorSpec("out", [1, 64], DataType.FP32, is_output=True),
        TensorSpec("out_acc", [1, 64], DataType.FP32, is_output=True),
        TensorSpec("out_bias", [1, 64], DataType.FP32, is_output=True),
    ]


def _gemv_mx_golden(dtype):
    def golden(tensors):
        levels = (
            [0, 0.5, 1, 1.5, 2, 3, 4, 6]
            if dtype == pl.FP4E2M1X2
            else [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]
        )
        values = torch.tensor(levels + [-v for v in levels], dtype=torch.float64)

        def decode(raw):
            codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2).long()
            return values[codes]

        base = _matmul_mx_golden(
            decode(tensors["a"])[:, :64],
            tensors["a_scale"].view(torch.uint8),
            decode(tensors["b"])[:, :64].T,
            _unpack_b_scale(tensors["b_scale"].view(torch.uint8)),
        )
        return {"out": base, "out_acc": 2 * base, "out_bias": base + tensors["bias"]}

    return golden


@pytest.mark.platforms("a5")
@st.cases(
    *[
        st.case(
            _gemv_mx_entry(dtype),
            tensors=_gemv_mx_specs(),
            name=f"gemv_mx_{dtype}_base_acc_bias",
            golden=_gemv_mx_golden(dtype),
            rtol=0.0,
            atol=0.0,
        )
        for dtype in (pl.FP4E2M1X2, pl.FP4E1M2X2)
    ]
)
def test_native_fp4_gemv_mx(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
