# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""User-facing A5 runtime tests for MX quantization and matrix multiplication."""

import os
from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

# MIX_DIAG_MODE: "" (default varying A-scales) | "row" | "group" | "probe" | "probe_data".
# Default must exercise non-uniform A-scales (all-127 smoke is not a valid oracle).
# probe / probe_data: unique A (+ group-select B) to decode V2C remaps from out.
_MIX_DIAG_MODE = os.environ.get("MIX_DIAG_MODE", "").strip().lower()

M, K, N = 16, 64, 32
MIX_M, MIX_K, MIX_N = 32, 128, N
MIX_K_CHUNK = 64
MIX_K_CHUNKS = MIX_K // MIX_K_CHUNK
# Multi-box shapes: A-scale [32, 4] = 2x2 [16, 2] boxes; B-scale [4, 64] = 2x4
# [2, 16] boxes. Packing is no longer a byte-identity for these tensors.
MB_M, MB_K, MB_N = 32, 64, 64
# layout-ab uses single K-box packed quant (ExpandMxPackedQuant kb==1).
# Host-packed multi-box scales stay on MB_* via matmul_mx_multibox.
LAYOUT_M, LAYOUT_K, LAYOUT_N = 32, 64, 64
MX_GROUP_SIZE = 32
SCALE_BLOCK_SIZE = 16
SCALE_C0_SIZE = 2
MIX_DATA_SLOT_SIZE = MIX_M * MIX_K_CHUNK
# Per-chunk packed A-scale from quant_mx(MX_A_ZZ): [1, M*K_chunk/32] via GM.
MIX_SCALE_ELEMS = MIX_M * MIX_K_CHUNK // MX_GROUP_SIZE
# Per-chunk packed scales live in rows of [MIX_K_CHUNKS, MIX_SCALE_ELEMS]
# (concat into [1, M*K/32] is NOT full MX_A_ZZ — see Phase0).
MIX_PIPE_DEPTH = MIX_K_CHUNKS
MIX_DATA_BUFFER_SIZE = MIX_DATA_SLOT_SIZE * MIX_PIPE_DEPTH
LAYOUT_G = LAYOUT_M * LAYOUT_K // MX_GROUP_SIZE
LAYOUT_BG = (LAYOUT_K // MX_GROUP_SIZE) * LAYOUT_N


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


def _mx_a_zz_flat_1x256(logical_mg: torch.Tensor) -> torch.Tensor:
    """Logical ``[M, K/32]`` → ``quant_mx(MX_A_ZZ)`` flat ``[1, M*K/32]``.

    Not ND/NZ: stacked ``[16, 2]`` boxes via
    ``reshape(M/16, 16, G/2, 2).permute(0, 2, 1, 3).reshape(1, M*G)``.
    For one K-chunk (``M=32, G=8``) that is 8 boxes × 32B = ``[1, 256]``.
    """
    return _pack_a_scale(logical_mg).reshape(1, -1)


def _fp32_atom_transpose_1x256(flat_u8: torch.Tensor) -> torch.Tensor:
    """Byte permute of ``[1, 256]`` as FP32 ``[8, 8]`` matrix transpose.

    A5 V2C/TINSERT stages ColMajor on the FP32 ``[8, 8]`` send view, which is
    this 4-byte-atom transpose. Applying it once on V cancels the staging so C
    receives the original ``[1, 256]`` byte order. Not expressible as MX ND/NZ.
    """
    atoms = flat_u8.reshape(-1).to(torch.uint8).view(torch.float32).reshape(8, 8)
    return atoms.T.contiguous().view(torch.uint8).reshape(1, 256)


def _v2c_scale_group_src_index(m: int = MIX_M, g_chunk: int = MIX_K_CHUNK // MX_GROUP_SIZE) -> torch.Tensor:
    """Per-chunk logical scale index map if V2C applies FP32-atom T and LeftScale unpacks ZZ.

    Legacy helper from the V2C-scale experiments; only defined for the old
    ``MIX_K_CHUNK=256`` geometry (``g_chunk==8``).
    """
    assert g_chunk == 8 and m == 32, "V2C scale remap helper only supports MIX_K_CHUNK=256"
    ids = torch.arange(m * g_chunk, dtype=torch.int64).reshape(m, g_chunk)
    packed = _pack_a_scale(ids)
    return _unpack_a_scale(
        _fp32_atom_transpose_1x256(packed.reshape(1, -1).to(torch.uint8)).reshape(m, g_chunk).to(torch.int64)
    )


def _gather_a_mx_groups(a: torch.Tensor, src_index_chunk: torch.Tensor) -> torch.Tensor:
    """Gather MX groups of ``a`` so each K-chunk follows ``src_index_chunk`` ``[M, 8]``."""
    rows, cols = a.shape
    g_total = cols // MX_GROUP_SIZE
    g_chunk = src_index_chunk.shape[1]
    blocks = a.reshape(rows, g_total, MX_GROUP_SIZE)
    out = torch.empty_like(blocks)
    for c0 in range(0, g_total, g_chunk):
        idx = src_index_chunk
        m_src = idx // g_chunk
        g_src = idx % g_chunk
        out[:, c0 : c0 + g_chunk, :] = blocks[m_src, c0 + g_src, :]
    return out.reshape(rows, cols).contiguous()


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


def _diag_row_scale_matrix(rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 A with mantissa 1.0 and per-row E8M0 exponents (constant across K-groups).

    Returns ``(a, exp_per_row)`` where ``exp_per_row[m] = (m % 8) - 3`` so
    ``a[m, :] == 2 ** exp_per_row[m]``. Easy to reverse-engineer which scale row
    Cube applied: ``actual[m, n] / K ≈ 2 ** e_used`` when B is all ones.
    """
    groups = cols // MX_GROUP_SIZE
    exp_per_row = ((torch.arange(rows) % 8) - 3).to(torch.float32)
    scales = torch.pow(2.0, exp_per_row).view(rows, 1, 1)
    a = scales.expand(rows, groups, MX_GROUP_SIZE).reshape(rows, cols).contiguous()
    return a, exp_per_row


def _diag_group_scale_matrix(rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 A with mantissa 1.0 and per-K-group exponents (same for every row).

    Returns ``(a, exp_per_group)`` with ``exp_per_group[g] = (g % 8) - 3``.
    """
    groups = cols // MX_GROUP_SIZE
    exp_per_group = ((torch.arange(groups) % 8) - 3).to(torch.float32)
    scales = torch.pow(2.0, exp_per_group).view(1, groups, 1)
    a = scales.expand(rows, groups, MX_GROUP_SIZE).reshape(rows, cols).contiguous()
    return a, exp_per_group


def _diag_unique_probe_matrix(rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Special A: unique float-safe E8M0 per ``(m,g)`` on the first 8 rows.

    ``code[m, g] = 100 + m * 8 + g`` for ``m,g in 0..7`` (codes 100..163, all
    distinct and float32-safe). Rows 8..31 and K beyond chunk-0 are 0. Pair with
    :func:`_diag_group_select_b`: ``out[m,g] = 32 * 2**(code_used-127)``.
    """
    g_chunk = MIX_K_CHUNK // MX_GROUP_SIZE
    # Probe modes were written for MIX_K_CHUNK=256 (8 groups); keep that geometry.
    assert rows == MIX_M
    assert g_chunk >= 2
    codes_chunk = torch.zeros((rows, g_chunk), dtype=torch.int64)
    a = torch.zeros((rows, cols), dtype=torch.float32)
    for m in range(8):
        for g in range(g_chunk):
            code = 100 + m * 8 + g
            codes_chunk[m, g] = code
            scale = float(2.0 ** (code - 127))
            lo = g * MX_GROUP_SIZE
            a[m, lo : lo + MX_GROUP_SIZE] = scale
    groups = cols // MX_GROUP_SIZE
    codes_full = torch.zeros((rows, groups), dtype=torch.uint8)
    codes_full[:, :g_chunk] = codes_chunk.to(torch.uint8)
    return a, codes_full, codes_chunk.to(torch.uint8)


def _diag_group_select_b(k: int = MIX_K, n: int = MIX_N) -> torch.Tensor:
    """B whose column ``g`` (g<8) is ones only on K-group ``g`` of chunk-0."""
    b = torch.zeros((k, n), dtype=torch.float32)
    for g in range(MIX_K_CHUNK // MX_GROUP_SIZE):
        lo = g * MX_GROUP_SIZE
        b[lo : lo + MX_GROUP_SIZE, g] = 1.0
    return b.to(torch.float8_e4m3fn)


def _diag_data_tag_matrix(rows: int, cols: int) -> torch.Tensor:
    """Unit-scale A with one-hot row tags inside each chunk-0 group (data-path probe).

    ``a[m, g*32 + m] = 1``, elsewhere 0 in chunk-0; rest of K zero. All scales → 127.
    With :func:`_diag_group_select_b`, ``out[m,g]==1`` iff the tag for source row
    ``m`` still sits in rx row ``m`` group ``g`` (else the 1 moves to another row).
    """
    g_chunk = MIX_K_CHUNK // MX_GROUP_SIZE
    a = torch.zeros((rows, cols), dtype=torch.float32)
    for m in range(rows):
        for g in range(g_chunk):
            a[m, g * MX_GROUP_SIZE + m] = 1.0
    return a


def _host_mx_quant_a(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Host mirror of on-chip ``quant_mx`` along K: FP8 data + logical E8M0 codes."""
    rows, cols = a.shape
    blocks = a.reshape(rows, cols // MX_GROUP_SIZE, MX_GROUP_SIZE)
    absmax = blocks.abs().amax(dim=2).clamp(min=1e-30)
    exp = torch.ceil(torch.log2(absmax)).to(torch.int32)
    scale_f = torch.pow(2.0, exp.to(torch.float32))
    a_q = (blocks / scale_f.unsqueeze(-1)).reshape(rows, cols).to(torch.float8_e4m3fn)
    return a_q, (exp + 127).clamp(0, 254).to(torch.uint8)


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


@pl.program
class QuantizedMatmulMxMixedProgram:
    """Explicit A5 mixed kernel: FP8 data on V2C; A-scale via per-chunk GM.

    ``MIX_K=128`` / ``MIX_K_CHUNK=64`` (2 chunks). Each chunk has its own
    packed ZZ scale tensor viewed as ``MX_A_ZZ`` ``[M, 2]`` (Phase0-C).
    """

    @pl.function(type=pl.FunctionType.AIV)
    def prefetch_a_scales(
        self,
        a: pl.Tensor[[MIX_M, MIX_K], pl.FP32],
        a_s0_gm: pl.Out[pl.Tensor[[1, MIX_SCALE_ELEMS], pl.FP8E8M0]],
        a_s1_gm: pl.Out[pl.Tensor[[1, MIX_SCALE_ELEMS], pl.FP8E8M0]],
    ):
        _q0, s0 = pl.quant_mx(pl.load(a, [0, 0], [MIX_M, MIX_K_CHUNK]), layout=pl.MX_A_ZZ)
        _q1, s1 = pl.quant_mx(
            pl.load(a, [0, MIX_K_CHUNK], [MIX_M, MIX_K_CHUNK]),
            layout=pl.MX_A_ZZ,
        )
        a_s0_gm = pl.store(s0, [0, 0], a_s0_gm)
        a_s1_gm = pl.store(s1, [0, 0], a_s1_gm)
        return a_s0_gm, a_s1_gm

    @pl.function(type=pl.FunctionType.AIV)
    def vector_quantize(
        self,
        a: pl.Tensor[[MIX_M, MIX_K], pl.FP32],
        b: pl.Tensor[[MIX_K, MIX_N], pl.FP8E4M3FN],
        b_scale: pl.Tensor[[1, MIX_K_CHUNK // MX_GROUP_SIZE * MIX_N], pl.FP8E8M0],
        a_s0_gm: pl.Tensor[[1, MIX_SCALE_ELEMS], pl.FP8E8M0],
        a_s1_gm: pl.Tensor[[1, MIX_SCALE_ELEMS], pl.FP8E8M0],
        out: pl.Out[pl.Tensor[[MIX_M, MIX_N], pl.FP32]],
    ):
        data_peer = pl.import_peer_buffer(name="v2c_mx_data_slot", peer_func="cube_matmul")
        pl.aiv_initialize_pipe(
            pl.const(0, pl.INT32),
            data_peer,
            dir_mask=2,
            slot_size=MIX_DATA_SLOT_SIZE,
            id=0,
        )
        for chunk in pl.range(0, MIX_K_CHUNKS):
            k_offset = chunk * MIX_K_CHUNK
            quant, _scale = pl.quant_mx(
                pl.load(a, [0, k_offset], [MIX_M, MIX_K_CHUNK]),
                layout=pl.MX_A_ZZ,
            )
            quant_nz = pl.move(
                quant,
                target_memory=pl.Mem.Vec,
                blayout=pl.TileLayout.col_major,
                slayout=pl.TileLayout.row_major,
            )
            pl.tpush_to_aic(quant_nz, split=0, id=0)

    @pl.function(type=pl.FunctionType.AIC)
    def cube_matmul(
        self,
        a: pl.Tensor[[MIX_M, MIX_K], pl.FP32],
        b: pl.Tensor[[MIX_K, MIX_N], pl.FP8E4M3FN],
        b_scale: pl.Tensor[[1, MIX_K_CHUNK // MX_GROUP_SIZE * MIX_N], pl.FP8E8M0],
        a_s0_gm: pl.Tensor[[1, MIX_SCALE_ELEMS], pl.FP8E8M0],
        a_s1_gm: pl.Tensor[[1, MIX_SCALE_ELEMS], pl.FP8E8M0],
        out: pl.Out[pl.Tensor[[MIX_M, MIX_N], pl.FP32]],
    ) -> pl.Tensor[[MIX_M, MIX_N], pl.FP32]:
        data_slot = pl.reserve_buffer(
            name="v2c_mx_data_slot",
            size=MIX_DATA_BUFFER_SIZE,
            base=pl.AUTO,
        )
        pl.aic_initialize_pipe(
            pl.const(0, pl.INT32),
            data_slot,
            dir_mask=2,
            slot_size=MIX_DATA_SLOT_SIZE,
            id=0,
        )
        g_chunk = MIX_K_CHUNK // MX_GROUP_SIZE
        b_scale_mx = pl.tensor.view(b_scale, [g_chunk, MIX_N], layout=pl.MX_B_NN)
        a_s0 = pl.tensor.view(a_s0_gm, [MIX_M, g_chunk], layout=pl.MX_A_ZZ)
        a_s1 = pl.tensor.view(a_s1_gm, [MIX_M, g_chunk], layout=pl.MX_A_ZZ)

        data_mat: pl.Tile[
            [MIX_M, MIX_K_CHUNK],
            pl.FP8E4M3FN,
            pl.Mem.Mat,
            pl.TileView(
                blayout=pl.TileLayout.col_major,
                slayout=pl.TileLayout.row_major,
                fractal=512,
            ),
        ] = pl.tpop_from_aiv(split=0, id=0)
        scale_mat = pl.load(
            a_s0, [0, 0], [MIX_M, g_chunk], target_memory=pl.Mem.Mat
        )
        lhs = pl.move(data_mat, target_memory=pl.Mem.Left)
        lhs_scale = pl.move(scale_mat, target_memory=pl.Mem.LeftScale)
        pl.tfree_to_aiv(data_mat, id=0)
        rhs = pl.move(
            pl.load(b, [0, 0], [MIX_K_CHUNK, MIX_N], target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.Right,
        )
        # Reload RightScale immediately before each matmul: address reuse can
        # otherwise sink a hoisted move past InsertMxScaleAddr's tget_scale_addr.
        rhs_scale = pl.move(
            pl.load(b_scale_mx, [0, 0], [g_chunk, MIX_N], target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.RightScale,
        )
        acc = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)

        # MIX_K=128 → one acc iteration; a_s1 is the second independent ZZ pack
        # (Phase0-C: not a concat into a full MX_A_ZZ view).
        for chunk, (acc_iter,) in pl.range(1, MIX_K_CHUNKS, init_values=(acc,)):
            k_offset = chunk * MIX_K_CHUNK
            data_mat: pl.Tile[
                [MIX_M, MIX_K_CHUNK],
                pl.FP8E4M3FN,
                pl.Mem.Mat,
                pl.TileView(
                    blayout=pl.TileLayout.col_major,
                    slayout=pl.TileLayout.row_major,
                    fractal=512,
                ),
            ] = pl.tpop_from_aiv(split=0, id=0)
            scale_mat = pl.load(
                a_s1, [0, 0], [MIX_M, g_chunk], target_memory=pl.Mem.Mat
            )
            lhs = pl.move(data_mat, target_memory=pl.Mem.Left)
            lhs_scale = pl.move(scale_mat, target_memory=pl.Mem.LeftScale)
            pl.tfree_to_aiv(data_mat, id=0)
            rhs = pl.move(
                pl.load(
                    b, [k_offset, 0], [MIX_K_CHUNK, MIX_N], target_memory=pl.Mem.Mat
                ),
                target_memory=pl.Mem.Right,
            )
            rhs_scale = pl.move(
                pl.load(
                    b_scale_mx, [0, 0], [g_chunk, MIX_N], target_memory=pl.Mem.Mat
                ),
                target_memory=pl.Mem.RightScale,
            )
            updated = pl.matmul_mx_acc(acc_iter, lhs, lhs_scale, rhs, rhs_scale)
            result = pl.yield_(updated)
        return pl.store(result, [0, 0], out)

    @pl.function(type=pl.FunctionType.Group)
    def group_func(
        self,
        a: pl.Tensor[[MIX_M, MIX_K], pl.FP32],
        b: pl.Tensor[[MIX_K, MIX_N], pl.FP8E4M3FN],
        b_scale: pl.Tensor[[1, MIX_K_CHUNK // MX_GROUP_SIZE * MIX_N], pl.FP8E8M0],
        a_s0_gm: pl.Tensor[[1, MIX_SCALE_ELEMS], pl.FP8E8M0],
        a_s1_gm: pl.Tensor[[1, MIX_SCALE_ELEMS], pl.FP8E8M0],
        out: pl.Out[pl.Tensor[[MIX_M, MIX_N], pl.FP32]],
    ) -> pl.Tensor[[MIX_M, MIX_N], pl.FP32]:
        self.vector_quantize(a, b, b_scale, a_s0_gm, a_s1_gm, out)
        return self.cube_matmul(a, b, b_scale, a_s0_gm, a_s1_gm, out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[MIX_M, MIX_K], pl.FP32],
        b: pl.Tensor[[MIX_K, MIX_N], pl.FP8E4M3FN],
        b_scale: pl.Tensor[[1, MIX_K_CHUNK // MX_GROUP_SIZE * MIX_N], pl.FP8E8M0],
        out: pl.Out[pl.Tensor[[MIX_M, MIX_N], pl.FP32]],
    ) -> pl.Tensor[[MIX_M, MIX_N], pl.FP32]:
        a_s0_gm = pl.create_tensor([1, MIX_SCALE_ELEMS], dtype=pl.FP8E8M0)
        a_s1_gm = pl.create_tensor([1, MIX_SCALE_ELEMS], dtype=pl.FP8E8M0)
        a_s0_gm, a_s1_gm = self.prefetch_a_scales(a, a_s0_gm, a_s1_gm)
        return self.group_func(a, b, b_scale, a_s0_gm, a_s1_gm, out)



@pl.jit.incore
def _quantized_matmul_mx_layout_quant_ab(
    a: pl.Tensor[[LAYOUT_M, LAYOUT_K], pl.FP32],
    b_nk: pl.Tensor[[LAYOUT_N, LAYOUT_K], pl.FP32],
    a_q_gm: pl.Out[pl.Tensor[[LAYOUT_M, LAYOUT_K], pl.FP8E4M3FN]],
    a_s_gm: pl.Out[pl.Tensor[[1, LAYOUT_G], pl.FP8E8M0]],
    b_q_gm: pl.Out[pl.Tensor[[LAYOUT_K, LAYOUT_N], pl.FP8E4M3FN]],
    b_s_gm: pl.Out[pl.Tensor[[1, LAYOUT_BG], pl.FP8E8M0]],
):
    """Quantize single-K-box A and B through the two public packed layouts."""
    a_q, a_s = pl.quant_mx(pl.load(a, [0, 0], [LAYOUT_M, LAYOUT_K]), layout=pl.MX_A_ZZ)
    b_q, b_s = pl.quant_mx(pl.load(b_nk, [0, 0], [LAYOUT_N, LAYOUT_K]), layout=pl.MX_B_NN)
    a_q_gm = pl.store(a_q, [0, 0], a_q_gm)
    a_s_gm = pl.store(a_s, [0, 0], a_s_gm)
    b_q_gm = pl.store(b_q, [0, 0], b_q_gm)
    b_s_gm = pl.store(b_s, [0, 0], b_s_gm)
    return a_q_gm, a_s_gm, b_q_gm, b_s_gm


@pl.jit.incore
def _quantized_matmul_mx_layout_mm(
    a_q_gm: pl.Tensor[[LAYOUT_M, LAYOUT_K], pl.FP8E4M3FN],
    a_s_gm: pl.Tensor[[1, LAYOUT_G], pl.FP8E8M0],
    b_q_gm: pl.Tensor[[LAYOUT_K, LAYOUT_N], pl.FP8E4M3FN],
    b_s_gm: pl.Tensor[[1, LAYOUT_BG], pl.FP8E8M0],
    out: pl.Out[pl.Tensor[[LAYOUT_M, LAYOUT_N], pl.FP32]],
    out_acc: pl.Out[pl.Tensor[[LAYOUT_M, LAYOUT_N], pl.FP32]],
):
    """Cube ``matmul_mx`` + ``matmul_mx_acc`` over continuous ZZ/NN scales."""
    a_s_mx = pl.tensor.view(a_s_gm, [LAYOUT_M, LAYOUT_K // MX_GROUP_SIZE], layout=pl.MX_A_ZZ)
    b_s_mx = pl.tensor.view(b_s_gm, [LAYOUT_K // MX_GROUP_SIZE, LAYOUT_N], layout=pl.MX_B_NN)
    lhs = pl.move(
        pl.load(a_q_gm, [0, 0], [LAYOUT_M, LAYOUT_K], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.Left,
    )
    lhs_scale = pl.move(
        pl.load(a_s_mx, [0, 0], [LAYOUT_M, LAYOUT_K // MX_GROUP_SIZE], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.LeftScale,
    )
    rhs = pl.move(
        pl.load(b_q_gm, [0, 0], [LAYOUT_K, LAYOUT_N], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.Right,
    )
    rhs_scale = pl.move(
        pl.load(b_s_mx, [0, 0], [LAYOUT_K // MX_GROUP_SIZE, LAYOUT_N], target_memory=pl.Mem.Mat),
        target_memory=pl.Mem.RightScale,
    )
    base = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)
    out = pl.store(base, [0, 0], out)
    accumulated = pl.matmul_mx_acc(base, lhs, lhs_scale, rhs, rhs_scale)
    out_acc = pl.store(accumulated, [0, 0], out_acc)
    return out, out_acc


@pl.jit
def quantized_matmul_mx_layout_ab_onboard(
    a: pl.Tensor[[LAYOUT_M, LAYOUT_K], pl.FP32],
    b_nk: pl.Tensor[[LAYOUT_N, LAYOUT_K], pl.FP32],
    out: pl.Out[pl.Tensor[[LAYOUT_M, LAYOUT_N], pl.FP32]],
    out_acc: pl.Out[pl.Tensor[[LAYOUT_M, LAYOUT_N], pl.FP32]],
):
    """Public single-K-box A/B quantization followed by MX matmul and accumulation."""
    a_q_gm = pl.create_tensor([LAYOUT_M, LAYOUT_K], dtype=pl.FP8E4M3FN)
    a_s_gm = pl.create_tensor([1, LAYOUT_G], dtype=pl.FP8E8M0)
    b_q_gm = pl.create_tensor([LAYOUT_K, LAYOUT_N], dtype=pl.FP8E4M3FN)
    b_s_gm = pl.create_tensor([1, LAYOUT_BG], dtype=pl.FP8E8M0)
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
        return "matmul_mx_multibox_32x64x64"

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


class TestQuantizedMatmulMxMixedKernel(TestQuantizedMatmulMx):
    """AIV quantization and AIC MX matmul within one mixed kernel."""

    __test__ = False
    shape = (MIX_M, MIX_K, MIX_N)

    def get_program(self) -> Any:
        return QuantizedMatmulMxMixedProgram

    def get_name(self) -> str:
        return "quantized_matmul_mx_mixed_32x128x32"

    def define_tensors(self) -> list[TensorSpec]:
        # B-scale stays unit (127); A-scales come from on-chip quant_mx and must vary.
        unit_scale = torch.full(
            (MIX_K_CHUNK // MX_GROUP_SIZE, MIX_N),
            127,
            dtype=torch.uint8,
        ).view(torch.float8_e8m0fnu)
        if _MIX_DIAG_MODE == "row":
            # Mantissa 1.0, scale = 2**((m%8)-3) on every K-group of row m.
            # With B=ones: expected[m, :] == MIX_K * 2**exp[m].
            a, _ = _diag_row_scale_matrix(MIX_M, MIX_K)
            b = torch.ones((MIX_K, MIX_N), dtype=torch.float32).to(torch.float8_e4m3fn)
        elif _MIX_DIAG_MODE == "group":
            a, _ = _diag_group_scale_matrix(MIX_M, MIX_K)
            b = torch.ones((MIX_K, MIX_N), dtype=torch.float32).to(torch.float8_e4m3fn)
        elif _MIX_DIAG_MODE == "probe":
            # Unique scales + group-select B → decode LeftScale map from out[m,g].
            a, _, _ = _diag_unique_probe_matrix(MIX_M, MIX_K)
            b = _diag_group_select_b()
        elif _MIX_DIAG_MODE == "probe_data":
            # Unit scales + one-hot tags + group-select B → decode data remap.
            a = _diag_data_tag_matrix(MIX_M, MIX_K)
            b = _diag_group_select_b()
        else:
            # Default: per-(m,g) power-of-two A-scales (exponents in {-1,0,1,2}).
            a = _exact_quantizable_matrix(MIX_M, MIX_K)
            generator = torch.Generator().manual_seed(29)
            b = torch.randint(-2, 3, (MIX_K, MIX_N), generator=generator).to(torch.float8_e4m3fn)
        return [
            TensorSpec(
                "a",
                [MIX_M, MIX_K],
                DataType.FP32,
                init_value=a,
            ),
            TensorSpec("b", [MIX_K, MIX_N], DataType.FP8E4M3FN, init_value=b),
            TensorSpec(
                "b_scale",
                [1, MIX_K_CHUNK // MX_GROUP_SIZE * MIX_N],
                DataType.FP8E8M0,
                init_value=unit_scale.reshape(1, -1),
            ),
            TensorSpec("out", [MIX_M, MIX_N], DataType.FP32, is_output=True),
        ]

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["out"][:] = torch.matmul(tensors["a"], tensors["b"].to(torch.float32))


@pytest.mark.platforms("a5")
class TestQuantizedMatmulMxLayoutAB(_JitMxTestCase):
    """Public packed-layout quantization of both MX operands."""

    __test__ = False
    entry = quantized_matmul_mx_layout_ab_onboard

    def __init__(self, *, platform: str | None = None):
        super().__init__(RunConfig(rtol=1e-5, atol=1e-3), platform=platform)

    def get_name(self) -> str:
        return "quantized_matmul_mx_layout_ab_32x64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec(
                "a",
                [LAYOUT_M, LAYOUT_K],
                DataType.FP32,
                init_value=_exact_quantizable_matrix(LAYOUT_M, LAYOUT_K),
            ),
            TensorSpec(
                "b_nk",
                [LAYOUT_N, LAYOUT_K],
                DataType.FP32,
                init_value=_exact_quantizable_matrix(LAYOUT_N, LAYOUT_K, transpose_pattern=True),
            ),
            TensorSpec("out", [LAYOUT_M, LAYOUT_N], DataType.FP32, is_output=True),
            TensorSpec("out_acc", [LAYOUT_M, LAYOUT_N], DataType.FP32, is_output=True),
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
            pytest.param(TestQuantizedMatmulMxMixedKernel, id="mixed"),
            pytest.param(TestQuantizedMatmulMxLayoutAB, id="layout-ab"),
        ],
    )
    @pytest.mark.parametrize("platform", [pytest.param("a5", id="a5")])
    def test_matmul_mx(self, test_runner, platform, case_cls):
        result = test_runner.run(case_cls(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
