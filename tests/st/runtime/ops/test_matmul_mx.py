# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""User-facing A5 runtime tests for MX quantization and matrix multiplication."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

M, K, N = 16, 64, 32
# Multi-box shapes: A-scale [32, 4] = 2x2 [16, 2] boxes; B-scale [4, 64] = 2x4
# [2, 16] boxes. Packing is no longer a byte-identity for these tensors.
MB_M, MB_K, MB_N = 32, 128, 64
BIG_M, BIG_K, BIG_N = MB_M, MB_K, MB_N
MX_GROUP_SIZE = 32
SCALE_BLOCK_SIZE = 16
SCALE_C0_SIZE = 2
BIG_G = BIG_M * BIG_K // MX_GROUP_SIZE
BIG_BG = (BIG_K // MX_GROUP_SIZE) * BIG_N


def _pack_a_scale(scale_codes: torch.Tensor) -> torch.Tensor:
    """Pack logical A scales into the MX_A_ZZ physical layout.

    Logical ND ``[M, K/32]`` is rewritten as stacked ``[16, 2]`` boxes via
    ``reshape(M/16, 16, K/64, 2).permute(0, 2, 1, 3)``. Single-box tensors are
    a no-op; multi-box tensors must be reordered before an MX_A_ZZ TLOAD.
    """
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


def _pack_b_scale(scale_codes: torch.Tensor) -> torch.Tensor:
    """Pack logical B scales into the MX_B_NN physical layout.

    Logical ND ``[K/32, N]`` becomes stacked ``[2, 16]`` boxes via
    ``reshape(K/64, 2, N/16, 16).permute(2, 0, 3, 1)``.
    """
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


def _unpack_a_scale(packed: torch.Tensor) -> torch.Tensor:
    """Restore logical ``[M, K/32]`` scales from MX_A_ZZ box order."""
    m, k_groups = packed.shape
    return (
        packed.reshape(
            m // SCALE_BLOCK_SIZE,
            k_groups // SCALE_C0_SIZE,
            SCALE_BLOCK_SIZE,
            SCALE_C0_SIZE,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(m, k_groups)
    )


def _unpack_b_scale(packed: torch.Tensor) -> torch.Tensor:
    """Restore logical ``[K/32, N]`` scales from MX_B_NN box order."""
    k_groups, n = packed.shape
    return (
        packed.reshape(
            n // SCALE_BLOCK_SIZE,
            k_groups // SCALE_C0_SIZE,
            SCALE_BLOCK_SIZE,
            SCALE_C0_SIZE,
        )
        .permute(1, 3, 0, 2)
        .contiguous()
        .reshape(k_groups, n)
    )


def _matmul_mx_golden(
    a: torch.Tensor,
    a_scale_codes: torch.Tensor,
    b: torch.Tensor,
    b_scale_codes: torch.Tensor,
) -> torch.Tensor:
    """Compute MXFP8 matmul with logical per-32-element E8M0 scales."""
    k = a.shape[1]
    k_group = torch.arange(k) // MX_GROUP_SIZE
    a_scale = torch.pow(2.0, a_scale_codes.to(torch.float64) - 127)
    b_scale = torch.pow(2.0, b_scale_codes.to(torch.float64) - 127)
    a_scaled = a.to(torch.float64) * a_scale[:, k_group]
    b_scaled = b.to(torch.float64) * b_scale[k_group, :]
    return torch.matmul(a_scaled, b_scaled).to(torch.float32)


def _exact_quantizable_matrix(rows: int, cols: int, *, transpose_pattern: bool = False) -> torch.Tensor:
    """Create FP32 blocks that quant_mx represents exactly with power-of-two scales."""
    pattern = (torch.arange(MX_GROUP_SIZE, dtype=torch.float32) % 9) - 4
    pattern[0] = 448
    pattern[1] = -448
    groups_per_row = cols // MX_GROUP_SIZE
    exponents = (torch.arange(rows * groups_per_row) % 4 - 1).reshape(rows, groups_per_row)
    if transpose_pattern:
        exponents = exponents.flip(0)
    scales = torch.pow(2.0, exponents).to(torch.float32)
    return (pattern.reshape(1, 1, MX_GROUP_SIZE) * scales.unsqueeze(-1)).reshape(rows, cols).contiguous()


def _host_prequant_b_nk(b_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Host MX-quant of B laid out as ``[N, K]``, returning ``[K, N]`` data + logical scales.

    Groups along K (last dim), matching on-chip ``quant_mx``. Returns FP8E4M3FN
    ``b_q`` with shape ``[K, N]`` and uint8 scale codes with shape ``[K/32, N]``.
    """
    n, k = b_t.shape
    b_blocks = b_t.reshape(n, k // MX_GROUP_SIZE, MX_GROUP_SIZE)
    b_absmax = b_blocks.abs().amax(dim=2)
    b_exp = torch.ceil(torch.log2(torch.clamp(b_absmax, min=1e-30))).to(torch.int32)
    b_scale_codes_nk = (b_exp + 127).clamp(0, 254).to(torch.uint8)
    b_scale_f = torch.pow(2.0, b_exp.to(torch.float32))
    b_q = (b_blocks / b_scale_f.unsqueeze(-1)).reshape(n, k).T.contiguous().to(torch.float8_e4m3fn)
    return b_q, b_scale_codes_nk.T.contiguous()


@pl.jit
def matmul_mx_onboard(
    a: pl.Tensor[[M, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[M, K // 32], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[K, N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[K // 32, N], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    """Run base and accumulating MX matmul with shared GM operands."""
    with pl.at(level=pl.Level.CORE_GROUP):
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


@pl.jit.incore
def _quantized_matmul_mx_quant_a(
    a: pl.Tensor[[M, K], pl.FP32],
    a_q_gm: pl.Out[pl.Tensor[[M, K], pl.FP8E4M3FN]],
    a_s_gm: pl.Out[pl.Tensor[[1, M * K // MX_GROUP_SIZE], pl.FP8E8M0]],
):
    """Quantize A through the public packed-A interface; write value/scale to GM."""
    a_src = pl.load(a, [0, 0], [M, K])
    a_quant, a_scale = pl.quant_mx(a_src, layout=pl.MX_A_ZZ)
    a_q_gm = pl.store(a_quant, [0, 0], a_q_gm)
    a_s_gm = pl.store(a_scale, [0, 0], a_s_gm)
    return a_q_gm, a_s_gm


@pl.jit.incore
def _quantized_matmul_mx_load_mm(
    a_q_gm: pl.Tensor[[M, K], pl.FP8E4M3FN],
    a_s_gm: pl.Tensor[[1, M * K // MX_GROUP_SIZE], pl.FP8E8M0],
    b: pl.Tensor[[K, N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[K // MX_GROUP_SIZE, N], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    """Cube-side load of dynamic A scales plus host-packed B, then matmul_mx."""
    a_s_mx = pl.tensor.view(a_s_gm, [M, K // MX_GROUP_SIZE], layout=pl.MX_A_ZZ)
    lhs = pl.move(
        pl.load(a_q_gm, [0, 0], [M, K], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.Left,
    )
    lhs_scale = pl.move(
        pl.load(a_s_mx, [0, 0], [M, K // MX_GROUP_SIZE], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.LeftScale,
    )
    rhs = pl.move(
        pl.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.Right,
    )
    rhs_scale = pl.move(
        pl.load(b_scale, [0, 0], [K // MX_GROUP_SIZE, N], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.RightScale,
    )
    out = pl.store(pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale), [0, 0], out)
    return out


@pl.jit
def quantized_matmul_mx_onboard(
    a: pl.Tensor[[M, K], pl.FP32],
    b: pl.Tensor[[K, N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[K // MX_GROUP_SIZE, N], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    """On-chip MX-quant of A, then matmul_mx with host-packed B scales.

    A is quantized on AIV through ``pl.quant_mx(layout=MX_A_ZZ)``, stored to GM,
    and reloaded on AIC via an ``MX_A_ZZ`` view. B stays host-prequantized with
    an ``MX_B_NN`` scale tensor.
    """
    a_q_gm = pl.create_tensor([M, K], dtype=pl.FP8E4M3FN)
    a_s_gm = pl.create_tensor([1, M * K // MX_GROUP_SIZE], dtype=pl.FP8E8M0)
    a_q_gm, a_s_gm = _quantized_matmul_mx_quant_a(a, a_q_gm, a_s_gm)
    out = _quantized_matmul_mx_load_mm(a_q_gm, a_s_gm, b, b_scale, out)
    return out


@pl.jit.incore
def _quantized_matmul_mx_layout_quant_ab(
    a: pl.Tensor[[BIG_M, BIG_K], pl.FP32],
    b_nk: pl.Tensor[[BIG_N, BIG_K], pl.FP32],
    a_q_gm: pl.Out[pl.Tensor[[BIG_M, BIG_K], pl.FP8E4M3FN]],
    a_s_gm: pl.Out[pl.Tensor[[1, BIG_G], pl.FP8E8M0]],
    b_q_gm: pl.Out[pl.Tensor[[BIG_K, BIG_N], pl.FP8E4M3FN]],
    b_s_gm: pl.Out[pl.Tensor[[1, BIG_BG], pl.FP8E8M0]],
):
    """Quantize multi-box A and B through the two public packed layouts."""
    a_q, a_s = pl.quant_mx(pl.load(a, [0, 0], [BIG_M, BIG_K]), layout=pl.MX_A_ZZ)
    b_q, b_s = pl.quant_mx(pl.load(b_nk, [0, 0], [BIG_N, BIG_K]), layout=pl.MX_B_NN)
    a_q_gm = pl.store(a_q, [0, 0], a_q_gm)
    a_s_gm = pl.store(a_s, [0, 0], a_s_gm)
    b_q_gm = pl.store(b_q, [0, 0], b_q_gm)
    b_s_gm = pl.store(b_s, [0, 0], b_s_gm)
    return a_q_gm, a_s_gm, b_q_gm, b_s_gm


@pl.jit.incore
def _quantized_matmul_mx_layout_mm(
    a_q_gm: pl.Tensor[[BIG_M, BIG_K], pl.FP8E4M3FN],
    a_s_gm: pl.Tensor[[1, BIG_G], pl.FP8E8M0],
    b_q_gm: pl.Tensor[[BIG_K, BIG_N], pl.FP8E4M3FN],
    b_s_gm: pl.Tensor[[1, BIG_BG], pl.FP8E8M0],
    out: pl.Out[pl.Tensor[[BIG_M, BIG_N], pl.FP32]],
    out_acc: pl.Out[pl.Tensor[[BIG_M, BIG_N], pl.FP32]],
):
    """Cube ``matmul_mx`` + ``matmul_mx_acc`` over continuous ZZ/NN scales."""
    a_s_mx = pl.tensor.view(a_s_gm, [BIG_M, BIG_K // MX_GROUP_SIZE], layout=pl.MX_A_ZZ)
    b_s_mx = pl.tensor.view(b_s_gm, [BIG_K // MX_GROUP_SIZE, BIG_N], layout=pl.MX_B_NN)
    lhs = pl.move(
        pl.load(a_q_gm, [0, 0], [BIG_M, BIG_K], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.Left,
    )
    lhs_scale = pl.move(
        pl.load(a_s_mx, [0, 0], [BIG_M, BIG_K // MX_GROUP_SIZE], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.LeftScale,
    )
    rhs = pl.move(
        pl.load(b_q_gm, [0, 0], [BIG_K, BIG_N], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.Right,
    )
    rhs_scale = pl.move(
        pl.load(b_s_mx, [0, 0], [BIG_K // MX_GROUP_SIZE, BIG_N], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.RightScale,
    )
    base = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)
    out = pl.store(base, [0, 0], out)
    accumulated = pl.matmul_mx_acc(base, lhs, lhs_scale, rhs, rhs_scale)
    out_acc = pl.store(accumulated, [0, 0], out_acc)
    return out, out_acc


@pl.jit
def quantized_matmul_mx_layout_ab_onboard(
    a: pl.Tensor[[BIG_M, BIG_K], pl.FP32],
    b_nk: pl.Tensor[[BIG_N, BIG_K], pl.FP32],
    out: pl.Out[pl.Tensor[[BIG_M, BIG_N], pl.FP32]],
    out_acc: pl.Out[pl.Tensor[[BIG_M, BIG_N], pl.FP32]],
):
    """Public multi-box A/B quantization followed by MX matmul and accumulation."""
    a_q_gm = pl.create_tensor([BIG_M, BIG_K], dtype=pl.FP8E4M3FN)
    a_s_gm = pl.create_tensor([1, BIG_G], dtype=pl.FP8E8M0)
    b_q_gm = pl.create_tensor([BIG_K, BIG_N], dtype=pl.FP8E4M3FN)
    b_s_gm = pl.create_tensor([1, BIG_BG], dtype=pl.FP8E8M0)
    a_q_gm, a_s_gm, b_q_gm, b_s_gm = _quantized_matmul_mx_layout_quant_ab(
        a, b_nk, a_q_gm, a_s_gm, b_q_gm, b_s_gm
    )
    out, out_acc = _quantized_matmul_mx_layout_mm(a_q_gm, a_s_gm, b_q_gm, b_s_gm, out, out_acc)
    return out, out_acc


@pl.jit
def matmul_mx_multibox_onboard(
    a: pl.Tensor[[MB_M, MB_K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[MB_M, MB_K // 32], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[MB_K, MB_N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[MB_K // 32, MB_N], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[MB_M, MB_N], pl.FP32]],
):
    """MX matmul over multi-box packed A/B scales (host ZZ/NN reorder)."""
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs = pl.move(
            pl.load(a, [0, 0], [MB_M, MB_K], target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.Left,
        )
        lhs_scale = pl.move(
            pl.load(a_scale, [0, 0], [MB_M, MB_K // 32], target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.LeftScale,
        )
        rhs = pl.move(
            pl.load(b, [0, 0], [MB_K, MB_N], target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.Right,
        )
        rhs_scale = pl.move(
            pl.load(b_scale, [0, 0], [MB_K // 32, MB_N], target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.RightScale,
        )
        result = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)
        out = pl.store(result, [0, 0], out)
    return out


class _JitMxTestCase(PTOTestCase):
    """Adapt a statically annotated JIT sample to the common ST harness."""

    __test__ = False
    entry: Any

    def get_program(self) -> Any:
        specialization, _config = self.entry._resolve_specialization((), {}, allow_signature_mode=True)
        return self.entry._compile_to_program(
            specialization.tensor_meta,
            specialization.scalar_values,
            specialization.scalar_dtypes,
            specialization.per_func_dyn,
            pl,
        )


class TestMatmulMx(_JitMxTestCase):
    """Base and accumulating MX matmul with packed scales."""

    __test__ = False
    entry = matmul_mx_onboard

    def __init__(self, *, platform: str | None = None):
        super().__init__(RunConfig(rtol=0, atol=0), platform=platform)

    def get_name(self) -> str:
        return "matmul_mx_16x64x32"

    def define_tensors(self) -> list[TensorSpec]:
        generator = torch.Generator().manual_seed(19)
        a = torch.randint(-2, 3, (M, K), generator=generator).to(torch.float8_e4m3fn)
        b = torch.randint(-2, 3, (K, N), generator=generator).to(torch.float8_e4m3fn)
        a_codes = torch.randint(126, 130, (M, K // MX_GROUP_SIZE), generator=generator).to(torch.uint8)
        b_codes = torch.randint(126, 130, (K // MX_GROUP_SIZE, N), generator=generator).to(torch.uint8)
        return [
            TensorSpec("a", [M, K], DataType.FP8E4M3FN, init_value=a),
            TensorSpec(
                "a_scale",
                [M, K // MX_GROUP_SIZE],
                DataType.FP8E8M0,
                init_value=_pack_a_scale(a_codes).view(torch.float8_e8m0fnu),
            ),
            TensorSpec("b", [K, N], DataType.FP8E4M3FN, init_value=b),
            TensorSpec(
                "b_scale",
                [K // MX_GROUP_SIZE, N],
                DataType.FP8E8M0,
                init_value=_pack_b_scale(b_codes).view(torch.float8_e8m0fnu),
            ),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
            TensorSpec("out_acc", [M, N], DataType.FP32, is_output=True),
        ]

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a_codes = _unpack_a_scale(tensors["a_scale"].view(torch.uint8))
        b_codes = _unpack_b_scale(tensors["b_scale"].view(torch.uint8))
        expected = _matmul_mx_golden(tensors["a"], a_codes, tensors["b"], b_codes)
        tensors["out"][:] = expected
        tensors["out_acc"][:] = 2 * expected


class TestMatmulMxMultibox(TestMatmulMx):
    """MX matmul whose A/B scales span multiple physical boxes."""

    __test__ = False
    entry = matmul_mx_multibox_onboard

    def get_name(self) -> str:
        return "matmul_mx_multibox_32x128x64"

    def define_tensors(self) -> list[TensorSpec]:
        generator = torch.Generator().manual_seed(23)
        a = torch.randint(-2, 3, (MB_M, MB_K), generator=generator).to(torch.float8_e4m3fn)
        b = torch.randint(-2, 3, (MB_K, MB_N), generator=generator).to(torch.float8_e4m3fn)
        a_codes = torch.randint(126, 130, (MB_M, MB_K // MX_GROUP_SIZE), generator=generator).to(torch.uint8)
        b_codes = torch.randint(126, 130, (MB_K // MX_GROUP_SIZE, MB_N), generator=generator).to(torch.uint8)
        return [
            TensorSpec("a", [MB_M, MB_K], DataType.FP8E4M3FN, init_value=a),
            TensorSpec(
                "a_scale",
                [MB_M, MB_K // MX_GROUP_SIZE],
                DataType.FP8E8M0,
                init_value=_pack_a_scale(a_codes).view(torch.float8_e8m0fnu),
            ),
            TensorSpec("b", [MB_K, MB_N], DataType.FP8E4M3FN, init_value=b),
            TensorSpec(
                "b_scale",
                [MB_K // MX_GROUP_SIZE, MB_N],
                DataType.FP8E8M0,
                init_value=_pack_b_scale(b_codes).view(torch.float8_e8m0fnu),
            ),
            TensorSpec("out", [MB_M, MB_N], DataType.FP32, is_output=True),
        ]

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = _matmul_mx_golden(
            tensors["a"],
            _unpack_a_scale(tensors["a_scale"].view(torch.uint8)),
            tensors["b"],
            _unpack_b_scale(tensors["b_scale"].view(torch.uint8)),
        )


class TestQuantizedMatmulMx(_JitMxTestCase):
    """On-chip A quantization followed by MX matmul."""

    __test__ = False
    entry = quantized_matmul_mx_onboard
    shape = (M, K, N)

    def __init__(self, *, platform: str | None = None):
        super().__init__(RunConfig(rtol=1e-5, atol=1e-3), platform=platform)

    def get_name(self) -> str:
        return "quantized_matmul_mx_16x64x32"

    def define_tensors(self) -> list[TensorSpec]:
        m, k, n = self.shape
        a = _exact_quantizable_matrix(m, k)
        b_nk = _exact_quantizable_matrix(n, k, transpose_pattern=True)
        b_q, b_codes = _host_prequant_b_nk(b_nk)
        return [
            TensorSpec("a", [m, k], DataType.FP32, init_value=a),
            TensorSpec("b", [k, n], DataType.FP8E4M3FN, init_value=b_q),
            TensorSpec(
                "b_scale",
                [k // MX_GROUP_SIZE, n],
                DataType.FP8E8M0,
                init_value=_pack_b_scale(b_codes).view(torch.float8_e8m0fnu),
            ),
            TensorSpec("out", [m, n], DataType.FP32, is_output=True),
        ]

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        _m, k, n = self.shape
        b_nk = _exact_quantizable_matrix(n, k, transpose_pattern=True)
        tensors["out"][:] = torch.matmul(tensors["a"], b_nk.T)


class TestQuantizedMatmulMxLayoutAB(_JitMxTestCase):
    """Public packed-layout quantization of both MX operands."""

    __test__ = False
    entry = quantized_matmul_mx_layout_ab_onboard

    def __init__(self, *, platform: str | None = None):
        super().__init__(RunConfig(rtol=1e-5, atol=1e-3), platform=platform)

    def get_name(self) -> str:
        return "quantized_matmul_mx_layout_ab_32x128x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "a", [BIG_M, BIG_K], DataType.FP32, init_value=_exact_quantizable_matrix(BIG_M, BIG_K)
            ),
            TensorSpec(
                "b_nk",
                [BIG_N, BIG_K],
                DataType.FP32,
                init_value=_exact_quantizable_matrix(BIG_N, BIG_K, transpose_pattern=True),
            ),
            TensorSpec("out", [BIG_M, BIG_N], DataType.FP32, is_output=True),
            TensorSpec("out_acc", [BIG_M, BIG_N], DataType.FP32, is_output=True),
        ]

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        expected = torch.matmul(tensors["a"], tensors["b_nk"].T)
        tensors["out"][:] = expected
        tensors["out_acc"][:] = 2 * expected


@pytest.mark.platforms("a5")
class TestMatmulMxOperations:
    """Numerical execution coverage for the Ascend950-only MX matmul path."""

    @pytest.mark.parametrize(
        "case_cls",
        [
            pytest.param(TestMatmulMx, id="base"),
            pytest.param(TestMatmulMxMultibox, id="multibox"),
            pytest.param(TestQuantizedMatmulMx, id="quantized"),
            pytest.param(TestQuantizedMatmulMxLayoutAB, id="layout-ab"),
        ],
    )
    @pytest.mark.parametrize("platform", [pytest.param("a5", id="a5")])
    def test_matmul_mx(self, test_runner, platform, case_cls):
        result = test_runner.run(case_cls(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
