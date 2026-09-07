# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A partially-valid accumulator crossing ``pl.aiv_shard``: what the boundary carries.

An accumulator whose valid region does not fill its physical box is the ragged
tail of a real kernel (a vocab or sequence remainder). Which of those shapes the
AIC/AIV boundary can carry follows from one fact about the Cube->Vector FIFO:
pto-isa strides the GM slot view with the popped tile's RUNTIME ``validCol`` but
takes the row count from the COMPILE-TIME box (``TLoadGm2ubNd2nd``:
``lenBurst = validCol``, ``gmGap = gStride3 - gShape4``, ``nBurst = gShape3``).

So the column field is spoken for and the row field is free:

* UP_DOWN makes the ROW extent per-lane -> it rides on the TPOP ``valid_row``
  operand, and this file guards that it does (and that the empty lane's store is
  skipped, since a zero-row TSTORE is outside pto-isa's contract).
* LEFT_RIGHT makes the COLUMN extent per-lane -> no carrier, so it must be
  rejected with an actionable message rather than silently truncated.

Before this was handled, both shapes silently truncated BOTH lanes to
``ceil(V / 2)`` — the split deducer's lane-agnostic guess.
"""

import re
import shutil
from pathlib import Path

import pypto.language as pl
import pytest

torch = pytest.importorskip("torch")

from pypto.runtime import RunConfig  # noqa: E402

DUMP_DIR = Path(__file__).resolve().parents[4] / "build_output" / "split_aiv_ragged_split_axis"

ROWS, COLS, K = 16, 256, 256
HALF = ROWS // 2
VALID_M = 5  # real accumulator rows: reaches lane 0 only
VALID_N = 32  # real accumulator columns


@pl.jit
def ragged_rows(
    a: pl.Tensor[[VALID_M, K], pl.BF16],
    w: pl.Tensor[[COLS, K], pl.BF16],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
    """UP_DOWN over an accumulator valid to VALID_M of ROWS rows."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="ragged_rows"):
        a_tile = pl.slice(a, [ROWS, K], [0, 0], valid_shape=[VALID_M, K])
        acc = pl.matmul(a_tile, w[0:COLS, 0:K], b_trans=True, out_dtype=pl.FP32)
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            shard = pl.aiv_shard(acc)
            out[aiv_id * HALF : aiv_id * HALF + HALF, 0:COLS] = shard
    return out


@pl.jit
def ragged_cols_left_right(
    a: pl.Tensor[[ROWS, K], pl.BF16],
    w: pl.Tensor[[VALID_N, K], pl.BF16],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
    """LEFT_RIGHT over an accumulator valid to VALID_N of COLS columns."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="ragged_cols"):
        w_tile = pl.slice(w, [COLS, K], [0, 0], valid_shape=[VALID_N, K])
        acc = pl.matmul(a[0:ROWS, 0:K], w_tile, b_trans=True, out_dtype=pl.FP32)
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
            shard = pl.aiv_shard(acc)
            out[0:ROWS, aiv_id * (COLS // 2) : aiv_id * (COLS // 2) + COLS // 2] = shard
    return out


@pl.jit
def ragged_rows_and_cols(
    a: pl.Tensor[[VALID_M, K], pl.BF16],
    w: pl.Tensor[[VALID_N, K], pl.BF16],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
    """UP_DOWN with BOTH a per-lane row extent and a narrowed column extent."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="ragged_both"):
        a_tile = pl.slice(a, [ROWS, K], [0, 0], valid_shape=[VALID_M, K])
        w_tile = pl.slice(w, [COLS, K], [0, 0], valid_shape=[VALID_N, K])
        acc = pl.matmul(a_tile, w_tile, b_trans=True, out_dtype=pl.FP32)
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            shard = pl.aiv_shard(acc)
            out[aiv_id * HALF : aiv_id * HALF + HALF, 0:COLS] = shard
    return out


@pl.jit
def ragged_rows_nested(
    a: pl.Tensor[[VALID_M, K], pl.BF16],
    w: pl.Tensor[[COLS, K], pl.BF16],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
    """The same crossing, with the store nested in a branch.

    The shard itself stays at region top level — both lanes must pop the slot —
    but its consumer sits inside control flow that survives to pass 23 (a
    ``pl.range`` with a small constant trip count would be unrolled away long
    before then).
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="ragged_nested"):
        a_tile = pl.slice(a, [ROWS, K], [0, 0], valid_shape=[VALID_M, K])
        acc = pl.matmul(a_tile, w[0:COLS, 0:K], b_trans=True, out_dtype=pl.FP32)
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            shard = pl.aiv_shard(acc)
            if aiv_id == 0:
                out[0:HALF, 0:COLS] = shard
    return out


@pl.jit
def ragged_rows_with_compute(
    a: pl.Tensor[[VALID_M, K], pl.BF16],
    w: pl.Tensor[[COLS, K], pl.BF16],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
    """A vector op between the shard and its store, so the extent must PROPAGATE."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="ragged_compute"):
        a_tile = pl.slice(a, [ROWS, K], [0, 0], valid_shape=[VALID_M, K])
        acc = pl.matmul(a_tile, w[0:COLS, 0:K], b_trans=True, out_dtype=pl.FP32)
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            shard = pl.aiv_shard(acc)
            scaled = pl.exp(shard)
            out[aiv_id * HALF : aiv_id * HALF + HALF, 0:COLS] = scaled
    return out


@pl.jit
def hand_authored_partial_gather(
    x: pl.Tensor[[ROWS, K], pl.BF16],
    q: pl.Tensor[[ROWS, K], pl.BF16],
    out: pl.Out[pl.Tensor[[ROWS, ROWS], pl.FP32]],
) -> pl.Tensor[[ROWS, ROWS], pl.FP32]:
    """Both lanes hold the SAME partial extent, so their bands do not abut."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="partial_gather", allow_early_resolve=True):
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            lane = pl.slice(x, [HALF, K], [aiv_id * HALF, 0], valid_shape=[VALID_M, K])
            joined = pl.aic_gather(lane)
        out[0:ROWS, 0:ROWS] = pl.matmul(q, joined, b_trans=True, out_dtype=pl.FP32)
    return out


def _aiv_body(program_text: str) -> str:
    """The lowered AIV half. The outlined name is <jit>_<name_hint>_aiv, so it is
    matched by suffix rather than spelled out."""
    match = re.search(r"def (\w+_aiv)\(", program_text)
    assert match, program_text[:400]
    return program_text.split(f"def {match.group(1)}")[1].split("\n    def ")[0]


def _pto_aiv(pto_text: str) -> str:
    """The AIV kernel body. Split on the DEFINITION, not the bare symbol — the
    cube half references the vector kernel by name too."""
    match = re.search(r"func\.func @(\w+_aiv)\(", pto_text)
    assert match, pto_text[:400]
    return pto_text.split(f"func.func @{match.group(1)}(")[1].split("func.func")[0]


_PTO_CACHE: dict[str, str] = {}


def _ragged_rows_pto() -> str:
    """Codegen-only .pto for `ragged_rows`, emitted once and shared.

    PTO codegen writes the .pto before the downstream kernel-compilation step
    (simpler_setup), which is absent in a codegen-only CI env, so a post-codegen
    failure is captured rather than raised. The result is cached because a second
    compile into the same directory tries to LOAD it as a built artifact.
    """
    if "pto" in _PTO_CACHE:
        return _PTO_CACHE["pto"]
    if DUMP_DIR.exists():
        shutil.rmtree(DUMP_DIR)
    cfg = RunConfig(platform="a2a3", codegen_only=True, save_kernels=True, save_kernels_dir=str(DUMP_DIR))
    error: Exception | None = None
    try:
        ragged_rows(*_ragged_rows_args(), config=cfg)
    except Exception as e:  # noqa: BLE001 - see docstring
        error = e
    ptos = sorted(DUMP_DIR.rglob("*.pto"))
    assert ptos, f"codegen emitted no .pto under {DUMP_DIR}; compile raised: {error!r}"
    _PTO_CACHE["pto"] = ptos[0].read_text()
    return _PTO_CACHE["pto"]


def _ragged_rows_args():
    return [
        torch.randn(VALID_M, K, dtype=torch.bfloat16),
        torch.randn(COLS, K, dtype=torch.bfloat16),
        torch.empty(ROWS, COLS, dtype=torch.float32),
    ]


def test_ragged_rows_localize_to_a_per_lane_extent():
    """Each lane gets clamp(V - lane*half, 0, half), not the deducer's ceil(V/2).

    ceil(5 / 2) = 3 would drop rows 3 and 4 on lane 0 and fabricate three rows on
    lane 1; the truth is 5 rows on lane 0 and none on lane 1.
    """
    aiv = _aiv_body(str(ragged_rows.lower(config=RunConfig(platform="a2a3"))))

    assert "pl.tile.tpop_from_aic(split=1)" in aiv
    # The per-lane extent is an expression over the region's own lane index.
    assert re.search(r"valid_shape=\[pl\.max\(pl\.const\(5.*?\) - \w+ \* pl\.const\(8", aiv), aiv
    assert "valid_shape=[3," not in aiv, f"ceil(V/2) must not survive:\n{aiv}"


def test_ragged_rows_transport_keeps_the_full_column_box():
    """The row extent rides on TPOP; the column extent stays the physical box.

    A narrowed column would collapse the FIFO's GM row gap (gmGap = gStride3 -
    gShape4) and mis-stride the pop, so a full column is what makes the per-lane
    row extent carriable at all — and it needs no pto.treshape to restore.
    """
    pto = _ragged_rows_pto()
    aiv = _pto_aiv(pto)

    pop = next(line for line in aiv.splitlines() if "pto.tpop_from_aic" in line)
    assert re.search(r"pto\.tpop_from_aic\(%\w+, %c256_index\) \{split = 1\}", pop), pop
    assert "pto.treshape" not in aiv, f"a full column extent needs no restore:\n{aiv}"


def test_empty_lane_skips_the_store_but_still_pops_and_frees():
    """A zero-row TSTORE is outside pto-isa's contract, so the store is guarded.

    The tpop and the tfree must stay UNCONDITIONAL: both lanes hold a slot and
    must release it, or the FIFO credit protocol desynchronizes.
    """
    pto = _ragged_rows_pto()
    aiv = _pto_aiv(pto)
    lines = [line.strip() for line in aiv.splitlines()]

    guard = next(i for i, line in enumerate(lines) if line.startswith("scf.if"))
    store = next(i for i, line in enumerate(lines) if "pto.tstore" in line)
    pop = next(i for i, line in enumerate(lines) if "pto.tpop_from_aic" in line)
    free = next(i for i, line in enumerate(lines) if "pto.tfree_from_aic" in line)

    assert guard < store, f"the store must sit inside the empty-lane guard:\n{aiv}"
    assert pop < guard, f"the pop must precede (and sit outside) the guard:\n{aiv}"
    assert free > store, f"the free must follow the store:\n{aiv}"
    # Neither pop nor free may be nested in the guard.
    assert not any(
        "pto.tpop_from_aic" in line or "pto.tfree_from_aic" in line for line in lines[guard:store]
    ), aiv


def test_nested_control_flow_is_repaired_too():
    """The repair must reach as deep as the region does.

    A flat walk over the region body would leave a shard nested in a loop or a
    branch on the deducer's ceil(V/2) — silently, since nothing downstream can
    tell a lane-agnostic extent from a lane-aware one.
    """
    aiv = _aiv_body(str(ragged_rows_nested.lower(config=RunConfig(platform="a2a3"))))

    lane = re.search(r"= pl\.tile\.get_subblock_idx\(\)", aiv)
    assert lane, aiv
    # The extent must be an expression over the lane index, not a constant.
    assert re.search(r"valid_shape=\[[^\]]*aiv_id[^\]]*,", aiv), (
        f"the nested consumer must still see a lane-aware extent:\n{aiv}"
    )
    assert "valid_shape=[3," not in aiv, f"ceil(V/2) must not survive nesting:\n{aiv}"


def test_per_lane_extent_propagates_through_a_vector_consumer():
    """An extent-preserving consumer carries the per-lane extent to the store.

    The store is what finally reads it, so an op between the shard and the store
    must pass it through — otherwise the store would write the deducer's
    ceil(V/2) rows.
    """
    aiv = _aiv_body(str(ragged_rows_with_compute.lower(config=RunConfig(platform="a2a3"))))

    exp_line = next((line for line in aiv.splitlines() if "pl.tile.exp(" in line), None)
    assert exp_line is not None, aiv
    # Both the shard and the op that consumes it carry a lane-aware extent.
    assert aiv.count("aiv_id") >= 2, aiv
    assert "valid_shape=[3," not in aiv, f"ceil(V/2) must not reach the consumer:\n{aiv}"


def test_hand_authored_partial_gather_is_rejected_where_the_lanes_are_known():
    """The band rule lives in lowering, where the lanes' TRUE extents are known.

    The diagnostic must quote the author's own extents, not the deducer's
    lane-agnostic guess.
    """
    with pytest.raises(ValueError) as exc:
        hand_authored_partial_gather.lower(config=RunConfig(platform="a2a3"))

    message = str(exc.value)
    assert "tile.aic_gather re-joins the two lanes positionally" in message
    assert f"({VALID_M} of {HALF})" in message, message
    assert "pl.fillpad" in message


def test_left_right_ragged_columns_are_rejected_with_a_dsl_fix():
    """The column field is pinned by the transport, so a per-lane column has no carrier."""
    with pytest.raises(ValueError) as exc:
        ragged_cols_left_right.lower(config=RunConfig(platform="a2a3"))

    message = str(exc.value)
    assert "LEFT_RIGHT splits the column axis" in message
    assert "valid column extent (32 of 256)" in message
    # The diagnostic must hand the author a way out, not just a refusal.
    assert "mode=pl.SplitMode.UP_DOWN" in message
    assert "pl.set_validshape" in message


def test_ragged_rows_with_narrowed_columns_are_rejected():
    """treshape restores the column extent statically and would clobber the per-lane row."""
    with pytest.raises(ValueError) as exc:
        ragged_rows_and_cols.lower(config=RunConfig(platform="a2a3"))

    message = str(exc.value)
    assert "per-lane row extent" in message
    assert "pl.set_validshape" in message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
