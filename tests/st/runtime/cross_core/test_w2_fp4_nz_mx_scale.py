# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 W2 integration: packed weights, NZ slots and compact row-major scales.

The vector peers keep named RAW/NZ ping-pong allocations and transport nine
ordered weight packets, each assembled from both AIV lanes. The cube loads A
scales through MX_A_ND and B scales through MX_B_NN, preserving the [rows,72]
activation-scale GM ABI and dynamically selected partial expert rows.
"""

import re

import pypto.language as pl
import pytest
import torch
from harness import st
from harness.core.harness import DataType, TensorSpec
from pypto.pypto_core.passes import MemoryPlanner

_REQUIRED_TORCH_DTYPES = ("float8_e4m3fn",)
if not all(hasattr(torch, name) for name in _REQUIRED_TORCH_DTYPES):
    pytest.skip("torch FP8 E4M3 dtype required", allow_module_level=True)

ROWS, M, K, N = 64, 16, 2304, 256
K_TILE = 256
LANE_ROWS = K_TILE // 2
PACKETS = K // K_TILE
GROUPS = K // 32
SLOT_BYTES = K_TILE * N
BUFFER_BYTES = 2 * SLOT_BYTES
OUTPUT_SENTINEL = -999.0


@pl.jit.incore(level=pl.Level.AIV)
def w2_vector(weight: pl.Tensor[[K, N // 2], pl.UINT8]):
    # Shapes and lane offsets are already per-AIV; suppress another halving.
    pl.func_attr({"split_aiv": True, "split": pl.SplitMode.UP_DOWN})
    peer = pl.import_peer_buffer(name="w2_weights", peer_func="w2_cube")
    pl.aiv_initialize_pipe(pl.const(0, pl.INT32), peer, dir_mask=2, slot_size=SLOT_BYTES, slot_num=2)
    lane = pl.tile.get_subblock_idx()
    for packet in pl.range(PACKETS):
        # Packed UB aliases require static slot offsets. Both branches declare
        # slots of the same named allocations, preserving the ordered packets.
        if packet % 2 == 0:
            raw0: pl.Tile[
                [LANE_ROWS, N // 2],
                pl.UINT8,
                pl.MemRef("RAW", slots=2)[0],
                pl.Mem.Vec,
            ] = pl.tile.load(weight, [packet * K_TILE + lane * LANE_ROWS, 0], [LANE_ROWS, N // 2])
            packed0 = pl.reinterpret_view(raw0, pl.FP4E2M1X2)
            nz0: pl.Tile[
                [LANE_ROWS, N],
                pl.FP8E4M3FN,
                pl.MemRef("NZ", slots=2)[0],
                pl.Mem.Vec,
                pl.TileView(
                    blayout=pl.TileLayout.col_major,
                    slayout=pl.TileLayout.row_major,
                ),
            ] = pl.cast(packed0, pl.FP8E4M3FN, layout=pl.NZ)
            pl.tpush_to_aic(nz0, split=1)
        else:
            raw1: pl.Tile[
                [LANE_ROWS, N // 2],
                pl.UINT8,
                pl.MemRef("RAW", slots=2)[1],
                pl.Mem.Vec,
            ] = pl.tile.load(weight, [packet * K_TILE + lane * LANE_ROWS, 0], [LANE_ROWS, N // 2])
            packed1 = pl.reinterpret_view(raw1, pl.FP4E2M1X2)
            nz1: pl.Tile[
                [LANE_ROWS, N],
                pl.FP8E4M3FN,
                pl.MemRef("NZ", slots=2)[1],
                pl.Mem.Vec,
                pl.TileView(
                    blayout=pl.TileLayout.col_major,
                    slayout=pl.TileLayout.row_major,
                ),
            ] = pl.cast(packed1, pl.FP8E4M3FN, layout=pl.NZ)
            pl.tpush_to_aic(nz1, split=1)


@pl.jit.inline(auto_scope=False)
def w2_operands(
    a: pl.Tensor[[ROWS, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[ROWS, GROUPS], pl.FP8E8M0],
    b_scale: pl.Tensor[[GROUPS, N], pl.FP8E8M0],
    row_base: pl.Scalar[pl.INDEX],
    valid_rows: pl.Scalar[pl.INDEX],
    packet: pl.Scalar[pl.INDEX],
):
    # The GM source descriptors distinguish row-major activation scales from
    # the already-packed weight scales; these aliases never rearrange bytes.
    a_scale_mx = pl.tensor.view(a_scale, [ROWS, GROUPS], layout=pl.MX_A_ND)
    b_scale_mx = pl.tensor.view(b_scale, [GROUPS, N], layout=pl.MX_B_NN)
    rhs_mat = pl.tpop_from_aiv(shape=[K_TILE, N], dtype=pl.FP8E4M3FN, split=1)
    lhs_mat = pl.tile.load(
        a,
        [row_base, packet * K_TILE],
        [M, K_TILE],
        valid_shape=[valid_rows, K_TILE],
        target_memory=pl.Mem.Mat,
    )
    # These are two independent GM-to-L1 loads. Raw A scales must not be
    # described as prepacked MX_A_ZZ or gathered/packed by the vector producer.
    sa_mat = pl.tile.load(
        a_scale_mx,
        [row_base, packet * (K_TILE // 32)],
        [M, K_TILE // 32],
        valid_shape=[valid_rows, K_TILE // 32],
        target_memory=pl.Mem.Mat,
    )
    sb_mat = pl.tile.load(
        b_scale_mx, [packet * (K_TILE // 32), 0], [K_TILE // 32, N], target_memory=pl.Mem.Mat
    )
    lhs = pl.move(lhs_mat, target_memory=pl.Mem.Left)
    rhs = pl.move(rhs_mat, target_memory=pl.Mem.Right)
    sa = pl.move(sa_mat, target_memory=pl.Mem.LeftScale)
    sb = pl.move(sb_mat, target_memory=pl.Mem.RightScale)
    pl.tfree_to_aiv(rhs_mat)
    return lhs, sa, rhs, sb


@pl.jit.incore(level=pl.Level.AIC)
def w2_cube(
    a: pl.Tensor[[ROWS, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[ROWS, GROUPS], pl.FP8E8M0],
    b_scale: pl.Tensor[[GROUPS, N], pl.FP8E8M0],
    counts: pl.Tensor[[2], pl.INT32],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    pl.func_attr({"split_aiv": True, "split": pl.SplitMode.UP_DOWN})
    buffer = pl.reserve_buffer(name="w2_weights", size=BUFFER_BYTES, base=pl.AUTO)
    pl.aic_initialize_pipe(pl.const(0, pl.INT32), buffer, dir_mask=2, slot_size=SLOT_BYTES, slot_num=2)
    row_base = pl.cast(pl.tensor.read(counts, [0]), pl.INDEX)
    valid_rows = pl.cast(pl.tensor.read(counts, [1]), pl.INDEX)
    lhs, sa, rhs, sb = w2_operands(a, a_scale, b_scale, row_base, valid_rows, 0)
    initial = pl.matmul_mx(lhs, sa, rhs, sb)
    for packet, (running,) in pl.range(1, PACKETS, init_values=(initial,)):
        lhs, sa, rhs, sb = w2_operands(a, a_scale, b_scale, row_base, valid_rows, packet)
        updated = pl.matmul_mx_acc(running, lhs, sa, rhs, sb)
        accumulated = pl.yield_(updated)
    pl.tile.store(accumulated, [0, 0], out)
    return out


@pl.jit.incore(level=pl.Level.CORE_GROUP)
def w2_group(
    a: pl.Tensor[[ROWS, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[ROWS, GROUPS], pl.FP8E8M0],
    weight: pl.Tensor[[K, N // 2], pl.UINT8],
    b_scale: pl.Tensor[[GROUPS, N], pl.FP8E8M0],
    counts: pl.Tensor[[2], pl.INT32],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    out = w2_cube(a, a_scale, b_scale, counts, out)
    w2_vector(weight)
    return out


@pl.jit(auto_scope=False)
def routed_w2(
    a: pl.Tensor[[ROWS, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[ROWS, GROUPS], pl.FP8E8M0],
    weight: pl.Tensor[[K, N // 2], pl.UINT8],
    b_scale: pl.Tensor[[GROUPS, N], pl.FP8E8M0],
    counts: pl.Tensor[[2], pl.INT32],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    with pl.spmd(1):
        out = w2_group(a, a_scale, weight, b_scale, counts, out)
    return out


def _pack_b_scale(codes):
    """Pack logical weight-scale bytes into the existing MX_B_NN ABI."""
    groups, cols = codes.shape
    return (
        codes.reshape(groups // 2, 2, cols // 16, 16).permute(2, 0, 3, 1).contiguous().reshape(groups, cols)
    )


def _w2_inputs(row_base, valid_rows):
    generator = torch.Generator().manual_seed(3445)
    a = torch.randint(-2, 3, (ROWS, K), generator=generator).to(torch.float8_e4m3fn)
    # Distinct scales in every row/group expose treating raw ND bytes as ZZ.
    a_codes = ((torch.arange(ROWS)[:, None] + torch.arange(GROUPS)[None, :]) % 5 + 125).byte()
    b_codes = ((torch.arange(GROUPS)[:, None] * 3 + torch.arange(N)[None, :]) % 5 + 124).byte()
    # Every byte occurs in each AIV lane of every packet; packet-dependent
    # permutations expose stale ping-pong slots or a changed packet order.
    weight = (
        torch.arange(K * (N // 2)).reshape(K, N // 2) * 29
        + torch.arange(K)[:, None] * 13
        + torch.arange(K)[:, None] // K_TILE * 17
    ).byte()
    return (
        a,
        a_codes,
        weight,
        _pack_b_scale(b_codes),
        torch.tensor([row_base, valid_rows], dtype=torch.int32),
        torch.full((M, N), OUTPUT_SENTINEL, dtype=torch.float32),
    )


def _w2_specs(row_base, valid_rows):
    # Explicit byte-carrier TensorSpecs let annotation-only JIT specialization
    # retain native E8M0 parameters without requiring Torch's optional E8M0
    # dtype. The uploaded bytes are the E8M0 encodings, unchanged.
    names = ("a", "a_scale", "weight", "b_scale", "counts", "out")
    dtypes = (
        DataType.FP8E4M3FN,
        DataType.UINT8,
        DataType.UINT8,
        DataType.UINT8,
        DataType.INT32,
        DataType.FP32,
    )
    return [
        TensorSpec(name, list(value.shape), dtype, init_value=value, is_output=name == "out")
        for name, dtype, value in zip(names, dtypes, _w2_inputs(row_base, valid_rows))
    ]


def _w2_golden(tensors):
    """Independent E2M1 numerical decode and logical per-32-element scaling."""
    raw = tensors["weight"]
    codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2).long()
    levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    decoded = torch.tensor(levels + [-value for value in levels], dtype=torch.float64)[codes]
    a_codes = tensors["a_scale"].view(torch.uint8)
    packed_b = tensors["b_scale"].view(torch.uint8)
    b_codes = packed_b.reshape(N // 16, GROUPS // 2, 16, 2).permute(1, 3, 0, 2).reshape(GROUPS, N)
    base, rows = tensors["counts"].tolist()
    group_indices = torch.arange(K) // 32
    lhs = tensors["a"][base : base + rows].double()
    lhs *= torch.pow(2.0, a_codes[base : base + rows, group_indices].double() - 127)
    rhs = decoded * torch.pow(2.0, b_codes[group_indices].double() - 127)
    expected = torch.full((M, N), OUTPUT_SENTINEL, dtype=torch.float32)
    expected[:rows] = (lhs @ rhs).float()
    return {"out": expected}


@pytest.fixture
def _retained_w2_case(_st_case, test_runner, tmp_path, monkeypatch):
    monkeypatch.setattr(test_runner.config, "save_kernels", True)
    monkeypatch.setattr(test_runner.config, "save_kernels_dir", str(tmp_path))
    run = st.CaseRun(_st_case, test_runner.run(_st_case))
    return run, run.work_dir or (tmp_path / _st_case.name)


@pytest.mark.platforms("a5")
@st.cases(
    *[
        st.case(
            routed_w2,
            tensors=_w2_specs(base, rows),
            name=f"w2_fp4_nz_mx_a_nd_base{base}_rows{rows}",
            golden=_w2_golden,
            memory_planner=MemoryPlanner.PYPTO,
            rtol=0.0,
            atol=0.0,
        )
        for base, rows in ((5, 13), (17, 7), (31, 1))
    ]
)
def test_w2_nz_and_compact_scale_loads(_retained_w2_case):
    run, work_dir = _retained_w2_case
    run.assert_passed()
    sources = list((work_dir / "kernels").rglob("*.cpp"))
    assert sources, "retain generated C++ to verify the fused instruction and source layout"
    cpp = "\n".join(path.read_text() for path in sources)
    assert "Layout::MX_A_ND" in cpp
    assert "Layout::MX_B_NN" in cpp
    assert "TCVT" in cpp
    assert re.search(r"float4_e2m1x2_t", cpp)
    assert not re.search(r"Tile<\s*(?:pto::)?TileType::Vec,\s*(?:float|bfloat16_t)\s*,", cpp)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
