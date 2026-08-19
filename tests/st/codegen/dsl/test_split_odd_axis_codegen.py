# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""An ODD split axis across the AIC/AIV boundary: which code the transport carries.

pto-isa's Cube->Vector FIFO finds lane 1's band inside the slot from the popped
tile's own RUNTIME valid extent -- at ``e1`` cells for ``TILE_UP_DOWN``, at
``e1 + 1`` for ``TILE_UP_DOWN_ODD`` (``popVecTileFromGMFiFo``). So a split whose
two lanes differ by exactly one -- an odd extent -- is expressible only under the
_ODD codes (``split = 3`` / ``4``), and the compiler, not the author, picks them:
``pl.split`` names the axis, and ExpandMixedKernel derives the code from the
boundary tile's extents.

The two ways an odd axis reaches the boundary:

* an odd VALID extent inside an even physical box (``15`` of ``16`` rows: lanes
  hold 8 and 7) -- the shape a real kernel's ragged tail has, since an Acc box is
  fractal-aligned and therefore even;
* an odd physical BOX (``17`` rows: lanes hold 9 and 8), for tiles whose box is
  not fractal-bound.

A deeper tail (``13`` of ``16``) would leave the box partition's lanes 8 and 5 --
further than one cell apart, so pto-isa could place neither. The compiler
partitions the boundary's VALID region instead (7 and 6, ``lane_stride=7``),
which is both placeable and evenly balanced. That rebalance needs the whole
split body to derive from the one boundary, so a body that also splits an
independent value keeps the universal box partition and reports the unplaceable
extents.
"""

import re

import pypto.language as pl
import pytest

torch = pytest.importorskip("torch")

from pypto.runtime import RunConfig  # noqa: E402

ROWS, COLS, K = 16, 128, 128
HALF = ROWS // 2
ODD_VALID_M = 15  # lanes hold 8 and 7 -> TILE_UP_DOWN_ODD
DEEP_TAIL_VALID_M = 13  # box lanes 8 and 5 -> rebalanced to 7 and 6
ODD_BOX_ROWS = 17  # a fully-valid ODD box: lanes hold 9 and 8
ODD_BOX_HALF = (ODD_BOX_ROWS + 1) // 2


@pl.jit
def odd_rows(
    a: pl.Tensor[[ODD_VALID_M, K], pl.BF16],
    w: pl.Tensor[[COLS, K], pl.BF16],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
    """``a @ w.T`` split by rows, with 15 of the box's 16 rows real."""
    with pl.at(
        level=pl.Level.CORE_GROUP,
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        name_hint="odd_rows",
    ):
        a_tile = pl.slice(a, [ROWS, K], [0, 0], valid_shape=[ODD_VALID_M, K])
        acc = pl.matmul(a_tile, w[0:COLS, 0:K], b_trans=True, out_dtype=pl.FP32)
        out[:] = pl.exp(acc)
    return out


@pl.jit
def deep_tail_rows(
    a: pl.Tensor[[DEEP_TAIL_VALID_M, K], pl.BF16],
    w: pl.Tensor[[COLS, K], pl.BF16],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
    """13 real rows: the box partition's 8 / 5 is rebalanced to 7 / 6."""
    with pl.at(
        level=pl.Level.CORE_GROUP,
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        name_hint="deep_tail_rows",
    ):
        a_tile = pl.slice(a, [ROWS, K], [0, 0], valid_shape=[DEEP_TAIL_VALID_M, K])
        acc = pl.matmul(a_tile, w[0:COLS, 0:K], b_trans=True, out_dtype=pl.FP32)
        out[:] = pl.exp(acc)
    return out


@pl.jit
def deep_tail_with_independent_value(
    a: pl.Tensor[[DEEP_TAIL_VALID_M, K], pl.BF16],
    w: pl.Tensor[[COLS, K], pl.BF16],
    extra: pl.Tensor[[ROWS, COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    out2: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """The same tail, plus a value split independently of the boundary.

    ``extra`` spans all 16 rows, which the boundary's 13-row balanced partition
    would not cover — so the body stays on the box partition, whose 8 / 5 lanes
    have no placeable transport.
    """
    with pl.at(
        level=pl.Level.CORE_GROUP,
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        name_hint="deep_tail_mixed",
    ):
        a_tile = pl.slice(a, [ROWS, K], [0, 0], valid_shape=[DEEP_TAIL_VALID_M, K])
        acc = pl.matmul(a_tile, w[0:COLS, 0:K], b_trans=True, out_dtype=pl.FP32)
        out[:] = pl.exp(acc)
        out2[:] = pl.abs(extra)
    return out, out2


@pl.jit
def explicit_region_odd_box(
    a: pl.Tensor[[ODD_BOX_ROWS, K], pl.BF16],
    w: pl.Tensor[[COLS, K], pl.BF16],
    out: pl.Out[pl.Tensor[[ODD_BOX_ROWS, COLS], pl.FP32]],
) -> pl.Tensor[[ODD_BOX_ROWS, COLS], pl.FP32]:
    """An explicit ``pl.split_aiv`` region over a fully-valid ODD box (17 rows)."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="region_odd"):
        acc = pl.matmul(a[0:ODD_BOX_ROWS, 0:K], w[0:COLS, 0:K], b_trans=True, out_dtype=pl.FP32)
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            shard = pl.aiv_shard(acc)
            out[aiv_id * ODD_BOX_HALF : aiv_id * ODD_BOX_HALF + ODD_BOX_HALF, 0:COLS] = shard
    return out


def _args(valid_m: int):
    return [
        torch.randn(valid_m, K, dtype=torch.bfloat16),
        torch.randn(COLS, K, dtype=torch.bfloat16),
        torch.empty(ROWS, COLS, dtype=torch.float32),
    ]


@pytest.fixture(scope="session")
def odd_rows_pto(tmp_path_factory) -> str:
    """Codegen-only .pto for the odd-extent kernel, emitted once and shared.

    The dump directory comes from ``tmp_path_factory`` so that concurrent pytest
    workers never share (and delete) one another's output.

    PTO codegen writes the .pto before the downstream kernel-compilation step
    (simpler_setup), which is absent in a codegen-only CI env, so a post-codegen
    failure is captured rather than raised.
    """
    dump_dir = tmp_path_factory.mktemp("split_odd_axis")
    cfg = RunConfig(platform="a2a3", codegen_only=True, save_kernels=True, save_kernels_dir=str(dump_dir))
    error: Exception | None = None
    try:
        odd_rows(*_args(ODD_VALID_M), config=cfg)
    except Exception as e:  # noqa: BLE001 - see docstring
        error = e
    ptos = sorted(dump_dir.rglob("*.pto"))
    assert ptos, f"codegen emitted no .pto under {dump_dir}; compile raised: {error!r}"
    return ptos[0].read_text()


def _lowered_half(program_text: str, suffix: str) -> str:
    """One lowered function body from the printed program."""
    match = re.search(rf"def (\w+_{suffix})\(", program_text)
    assert match, program_text[:400]
    return program_text.split(f"def {match.group(1)}")[1].split("\n    def ")[0]


def _pto_half(pto_text: str, suffix: str) -> str:
    """One lowered kernel body. Split on the DEFINITION, not the bare symbol —
    the cube half references the vector kernel by name too."""
    match = re.search(rf"func\.func @(\w+_{suffix})\(", pto_text)
    assert match, pto_text[:400]
    return pto_text.split(f"func.func @{match.group(1)}(")[1].split("func.func")[0]


def test_odd_valid_extent_takes_the_odd_split_code(odd_rows_pto):
    """Both ends of the transport agree on ``TILE_UP_DOWN_ODD``.

    The push, the pop and the slot release all name the same pipe, so a code
    that differed between them would place the lanes on different bands.
    """
    pto = odd_rows_pto

    assert re.search(r"pto\.tpush_to_aiv\(.*?\) \{split = 3\}", _pto_half(pto, "aic")), pto
    aiv = _pto_half(pto, "aiv")
    assert re.search(r"pto\.tpop_from_aic\(%\w+, %\w+\) \{split = 3\}", aiv), aiv
    assert "pto.tfree_from_aic {split = 3}" in aiv, aiv


def test_odd_pop_carries_a_per_lane_row_extent_and_the_full_column_box(odd_rows_pto):
    """The ODD code is only half the contract: PTOAS also wants per-lane operands.

    The row extent is the lane's own (``clamp(15 - lane * 8, 0, 8)`` -> 8 / 7);
    the column extent stays the physical box, because the pop strides the GM slot
    with it and the producer wrote the box.
    """
    aiv = _pto_half(odd_rows_pto, "aiv")

    pop = next(line for line in aiv.splitlines() if "pto.tpop_from_aic" in line)
    row_ssa, col = re.search(r"pto\.tpop_from_aic\((%\w+), (%\w+)\)", pop).groups()
    assert col == f"%c{COLS}_index", pop
    # The row operand is a min/max clamp over the lane index, not a constant.
    assert re.search(rf"{re.escape(row_ssa)} = arith\.minsi", aiv), aiv
    assert f"arith.subi %c{ODD_VALID_M}_index" in aiv, aiv
    # The full-box transport + treshape path is for a STATIC narrowing; a
    # per-lane extent must reach pto-isa as an operand instead.
    assert "pto.treshape" not in aiv, f"a per-lane extent cannot survive a static restore:\n{aiv}"


def test_deep_tail_is_rebalanced_onto_the_valid_region():
    """13 valid rows split 7 / 6, not the box partition's 8 / 5.

    Every consumer follows the same partition: the pop's per-lane extents, the
    lane-localized compute, and the store offsets (``idx * 7``) — otherwise the
    two lanes would disagree about which rows they own.
    """
    printed = str(deep_tail_rows.lower(config=RunConfig(platform="a2a3")))
    aiv = _lowered_half(printed, "aiv")

    # The stride rides onto the transport so every consumer of the boundary —
    # including the torch reference runtime — cuts the lanes at the same row.
    assert "pl.tile.tpop_from_aic(split=3, lane_stride=7)" in aiv, aiv
    # clamp(13 - lane * 7, 0, 7) -> 7 and 6.
    assert re.search(r"pl\.const\(13, pl\.INDEX\)[^]]*?pl\.const\(7, pl\.INDEX\)", aiv), aiv
    assert re.search(r"pl\.tile\.store\([^)]*subblock_idx \* 7", aiv), aiv
    assert "subblock_idx * 8" not in aiv, f"the box partition must not survive:\n{aiv}"


def test_rebalance_is_declined_when_a_value_is_split_independently():
    """A body that also splits its own value keeps the box partition — and reports.

    ``extra`` spans the full 16-row box, which the boundary's balanced partition
    does not cover, so rebalancing is unsound here. The box partition's 8 / 5
    lanes then have no expressible transport, and the diagnostic must name both
    the blocking value and the extents that would work.
    """
    with pytest.raises(ValueError) as exc:
        deep_tail_with_independent_value.lower(config=RunConfig(platform="a2a3"))

    message = str(exc.value)
    assert "leaves the two AIV lanes 8 and 5 cells" in message
    # The diagnostic must hand the author a way out, not just a refusal.
    assert "independently split value" in message
    assert "pl.tile.set_validshape" in message
    assert "16 / 15" in message


def test_explicit_region_odd_box_localizes_the_per_lane_extent():
    """A fully-valid ODD box is per-lane in an explicit region too.

    ``ReshapeSplitAxis`` gives both lanes the ceil half (9), which is right for
    the physical box and wrong for the extents pto-isa reads off the popped tile:
    17 rows split 9 / 8. The region path used to take the "fully valid needs no
    repair" shortcut here, leaving both lanes at 9 while the transport picked the
    odd code — lane 1 would then have been placed one row too far.
    """
    printed = str(explicit_region_odd_box.lower(config=RunConfig(platform="a2a3")))
    aiv = _lowered_half(printed, "aiv")

    assert "pl.tile.tpop_from_aic(split=3)" in aiv, aiv
    # clamp(17 - aiv_id * 9, 0, 9) -> 9 and 8, on a 9-row box.
    assert re.search(r"pl\.const\(17, pl\.INDEX\)[^]]*?pl\.const\(9, pl\.INDEX\)", aiv), aiv
    assert re.search(r"pl\.tile\.store\([^)]*\* 9", aiv), aiv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
