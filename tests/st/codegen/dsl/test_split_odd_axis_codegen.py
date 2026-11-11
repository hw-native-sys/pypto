# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit shard transport codes for odd and ragged split-axis valid extents.

The author names the crossing and writes per-lane output offsets. The compiler
selects the transport code and localizes boundary valid extents, but no longer
rebalances an AUTO body. Unrepresentable explicit partitions are diagnosed.
"""

import re

import pypto.language as pl
import pytest

torch = pytest.importorskip("torch")

from pypto.runtime import RunConfig  # noqa: E402

ROWS, COLS, K = 16, 128, 128
HALF = ROWS // 2
ODD_VALID_M = 15  # lanes hold 8 and 7 -> TILE_UP_DOWN_ODD
DEEP_TAIL_VALID_M = 13  # box lanes 8 and 5 -> unrepresentable explicit partition
ODD_BOX_ROWS = 17  # an odd source extent; its Acc box is 32, so lanes hold 16 and 1
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
        name_hint="odd_rows",
    ):
        a_tile = pl.slice(a, [ROWS, K], [0, 0], valid_shape=[ODD_VALID_M, K])
        acc = pl.matmul(a_tile, w[0:COLS, 0:K], b_trans=True, out_dtype=pl.FP32)
        for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            half = pl.aiv_shard(acc)
            out[lane * HALF : lane * HALF + HALF, :] = pl.exp(half)
    return out


@pl.jit
def deep_tail_rows(
    a: pl.Tensor[[DEEP_TAIL_VALID_M, K], pl.BF16],
    w: pl.Tensor[[COLS, K], pl.BF16],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
    """13 real rows retain the explicit box partition (8 / 5), which is rejected."""
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="deep_tail_rows",
    ):
        a_tile = pl.slice(a, [ROWS, K], [0, 0], valid_shape=[DEEP_TAIL_VALID_M, K])
        acc = pl.matmul(a_tile, w[0:COLS, 0:K], b_trans=True, out_dtype=pl.FP32)
        for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            half = pl.aiv_shard(acc)
            out[lane * HALF : lane * HALF + HALF, :] = pl.exp(half)
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

    ``extra`` has authored lane-local slices. It does not alter the boundary's
    box partition, whose 8 / 5 lanes have no representable transport.
    """
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="deep_tail_mixed",
    ):
        a_tile = pl.slice(a, [ROWS, K], [0, 0], valid_shape=[DEEP_TAIL_VALID_M, K])
        acc = pl.matmul(a_tile, w[0:COLS, 0:K], b_trans=True, out_dtype=pl.FP32)
        for lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            half = pl.aiv_shard(acc)
            row = lane * HALF
            out[row : row + HALF, :] = pl.exp(half)
            out2[row : row + HALF, :] = pl.abs(extra[row : row + HALF, :])
    return out, out2


@pl.jit
def explicit_region_odd_box(
    a: pl.Tensor[[ODD_BOX_ROWS, K], pl.BF16],
    w: pl.Tensor[[COLS, K], pl.BF16],
    out: pl.Out[pl.Tensor[[ODD_BOX_ROWS, COLS], pl.FP32]],
) -> pl.Tensor[[ODD_BOX_ROWS, COLS], pl.FP32]:
    """An explicit ``pl.split_aiv`` region over an odd (17-row) source extent.

    The 17 rows reach the cube as a 32-row NZ-fractal box carrying a 17-row valid
    extent, so the region's box partition leaves the lanes 16 and 1 -- further
    than one cell apart, which pto-isa cannot place.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="region_odd"):
        acc = pl.matmul(a[0:ODD_BOX_ROWS, 0:K], w[0:COLS, 0:K], b_trans=True, out_dtype=pl.FP32)
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            shard = pl.aiv_shard(acc)
            out[aiv_id * ODD_BOX_HALF : aiv_id * ODD_BOX_HALF + ODD_BOX_HALF, 0:COLS] = shard
    return out


@pl.jit
def runtime_tail_rows(
    a: pl.Tensor[[ROWS, K], pl.BF16],
    w: pl.Tensor[[COLS, K], pl.BF16],
    vt: pl.Tensor[[1], pl.INDEX],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
    """A RUNTIME row extent across the boundary: the lanes cannot be compared.

    ``clamp(v - lane * 8, 0, 8)`` is 8 / 4 for ``v = 12`` and 8 / 8 for ``v = 16``
    — the compiler cannot tell which, so it cannot pick a code that places lane 1
    at its extent. The boundary op keeps the FULL box instead, which is where the
    producer wrote lane 1's rows, and the extent rides on the consumers.
    """
    v: pl.Scalar[pl.INDEX] = pl.tensor.read(vt, [0])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="runtime_tail"):
        a_tile = pl.slice(a, [ROWS, K], [0, 0], valid_shape=[v, K])
        acc = pl.matmul(a_tile, w[0:COLS, 0:K], b_trans=True, out_dtype=pl.FP32)
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            shard = pl.aiv_shard(acc)
            scaled = pl.mul(shard, 2.0)
            out[aiv_id * HALF : aiv_id * HALF + HALF, 0:COLS] = scaled
    return out


@pl.jit
def runtime_tail_fillpad_first(
    a: pl.Tensor[[ROWS, K], pl.BF16],
    w: pl.Tensor[[COLS, K], pl.BF16],
    vt: pl.Tensor[[1], pl.INDEX],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
    """A pad fill straight off the boundary, with nothing to fill up to."""
    v: pl.Scalar[pl.INDEX] = pl.tensor.read(vt, [0])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="fill_first"):
        a_tile = pl.slice(a, [ROWS, K], [0, 0], valid_shape=[v, K])
        acc = pl.matmul(a_tile, w[0:COLS, 0:K], b_trans=True, out_dtype=pl.FP32)
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            shard = pl.aiv_shard(acc)
            padded = pl.fillpad(shard, pad_value=pl.PadValue.zero)
            out[aiv_id * HALF : aiv_id * HALF + HALF, 0:COLS] = padded
    return out


@pl.jit
def runtime_tail_store_first(
    a: pl.Tensor[[ROWS, K], pl.BF16],
    w: pl.Tensor[[COLS, K], pl.BF16],
    vt: pl.Tensor[[1], pl.INDEX],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
    """The boundary tile stored directly, with no consumer to carry the extent."""
    v: pl.Scalar[pl.INDEX] = pl.tensor.read(vt, [0])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="store_first"):
        a_tile = pl.slice(a, [ROWS, K], [0, 0], valid_shape=[v, K])
        acc = pl.matmul(a_tile, w[0:COLS, 0:K], b_trans=True, out_dtype=pl.FP32)
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            shard = pl.aiv_shard(acc)
            out[aiv_id * HALF : aiv_id * HALF + HALF, 0:COLS] = shard
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
        odd_rows.compile(*_args(ODD_VALID_M), config=cfg)
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


def test_deep_tail_requires_an_explicit_representable_partition():
    """A 13-row tail in a 16-row box is not silently rebalanced to 7/6."""
    with pytest.raises(ValueError, match="leaves the two AIV lanes 8 and 5 cells"):
        deep_tail_rows.lower(config=RunConfig(platform="a2a3"))


def test_independent_lane_value_does_not_change_boundary_partition():
    """An independent lane-local value does not change explicit boundary admission."""
    with pytest.raises(ValueError) as exc:
        deep_tail_with_independent_value.lower(config=RunConfig(platform="a2a3"))

    message = str(exc.value)
    assert "leaves the two AIV lanes 8 and 5 cells" in message
    # The diagnostic must hand the author a way out, not just a refusal.
    assert "pl.tile.set_validshape" in message
    assert "16 / 15" in message
    assert "does not rebalance an explicit boundary" in message
    assert "independently split value" not in message


def test_explicit_region_odd_box_is_refused_with_unplaceable_lanes():
    """An odd physical box cannot reach a Cube -> Vector boundary at all.

    ``tile.aiv_shard``'s cube side is an ``Acc``, and ptoas rejects an ``Acc``
    allocation whose rows are not a multiple of 16, so the 17-row box this used to
    assert on was never a shape the backend would accept -- it only survived
    because ``--codegen-only`` skips assembly. The 17 rows now arrive as a 32-row
    fractal box carrying a 17-row valid extent, and the region's box partition
    leaves lane 0 with 16 cells and lane 1 with 1: further than one cell apart, so
    neither ``TILE_UP_DOWN`` nor its ``_ODD`` sibling can place lane 1's band.
    The compiler says so, and names the two authoring routes that do work.

    The reachable odd axis -- an odd VALID extent inside an even box -- keeps its
    positive coverage in ``test_odd_valid_extent_takes_the_odd_split_code`` and
    ``test_odd_pop_carries_a_per_lane_row_extent_and_the_full_column_box``.
    """
    with pytest.raises(ValueError) as exc:
        explicit_region_odd_box.lower(config=RunConfig(platform="a2a3"))

    message = str(exc.value)
    assert "leaves the two AIV lanes 16 and 1 cells" in message, message
    assert "pl.tile.set_validshape" in message, message


@pytest.fixture(scope="session")
def runtime_tail_pto(tmp_path_factory) -> str:
    """Codegen-only .pto for the runtime-extent kernel."""
    dump_dir = tmp_path_factory.mktemp("split_runtime_tail")
    cfg = RunConfig(platform="a2a3", codegen_only=True, save_kernels=True, save_kernels_dir=str(dump_dir))
    error: Exception | None = None
    try:
        runtime_tail_rows.compile(
            torch.randn(ROWS, K, dtype=torch.bfloat16),
            torch.randn(COLS, K, dtype=torch.bfloat16),
            torch.tensor([12], dtype=torch.int64),
            torch.empty(ROWS, COLS, dtype=torch.float32),
            config=cfg,
        )
    except Exception as e:  # noqa: BLE001 - see odd_rows_pto
        error = e
    ptos = sorted(dump_dir.rglob("*.pto"))
    assert ptos, f"codegen emitted no .pto under {dump_dir}; compile raised: {error!r}"
    return ptos[0].read_text()


def test_runtime_extent_pops_the_full_box(runtime_tail_pto):
    """The per-lane extent must NOT reach the pop.

    pto-isa reads lane 1's band offset off the popped tile's own split-axis
    extent while the producer transports the full physical box, so a pop
    declaring ``clamp(v - 8, 0, 8)`` sends lane 1 to row ``v - 8`` — four rows
    early for ``v = 12``, and silently, since nothing else disagrees. A pop of
    the full box carries no operands at all and lands on the box half.
    """
    aiv = _pto_half(runtime_tail_pto, "aiv")

    assert re.search(r"pto\.tpop_from_aic \{split = 1\}", aiv), aiv
    assert "pto.tpop_from_aic(" not in aiv, f"the pop must carry no per-lane extent:\n{aiv}"


def test_runtime_extent_lands_on_the_first_consumer(runtime_tail_pto):
    """Widening the pop must not LOSE the extent — it moves one op downstream."""
    aiv = _pto_half(runtime_tail_pto, "aiv")

    # clamp(v - lane * 8, 0, 8), materialized for the consumer rather than the pop.
    assert re.search(r"arith\.minsi", aiv), aiv
    assert re.search(r"arith\.maxsi", aiv), aiv
    consumer = re.search(r"pto\.alloc_tile[^\n]*valid_row = (%\w+)", aiv)
    assert consumer, f"no consumer tile carries a computed valid_row:\n{aiv}"
    assert not consumer.group(1).startswith("%c"), f"consumer extent is a constant:\n{aiv}"


def _runtime_tail_args(v: int):
    return [
        torch.randn(ROWS, K, dtype=torch.bfloat16),
        torch.randn(COLS, K, dtype=torch.bfloat16),
        torch.tensor([v], dtype=torch.int64),
        torch.empty(ROWS, COLS, dtype=torch.float32),
    ]


def test_runtime_extent_rejects_a_pad_fill_at_the_boundary():
    """The extent has to land on a consumer -- a pad fill is not one.

    ``tile.fillpad`` reads where the lane's data ENDS, and its result is fully
    valid by construction. Off a boundary that carries the transport's box it
    would fill nothing and hand the padding on as data, so it is refused with the
    two authoring routes that do work.
    """
    with pytest.raises(ValueError) as exc:
        runtime_tail_fillpad_first.compile(*_runtime_tail_args(12), config=RunConfig(platform="a2a3"))

    message = str(exc.value)
    assert "fills the padding of a Cube -> Vector boundary tile" in message, message
    assert "pl.fillpad(acc) ahead of pl.aiv_shard(acc)" in message, message


def test_runtime_extent_rejects_a_direct_store_of_the_boundary():
    """Storing the boundary tile itself would write the transport's padding."""
    with pytest.raises(ValueError) as exc:
        runtime_tail_store_first.compile(*_runtime_tail_args(12), config=RunConfig(platform="a2a3"))

    message = str(exc.value)
    assert "consumes a Cube -> Vector boundary tile directly" in message, message
    assert "narrow after the crossing" in message, message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
