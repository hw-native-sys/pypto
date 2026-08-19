# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end codegen guard for a per-lane scattered ``pl.gather_row`` inside a
``pl.split_aiv(UP_DOWN)`` region (issue #2244).

The kernel is the sparse-attention KV assembly shape: a paged top-k index list
gives 128 scattered GM row offsets, the two AIV lanes each gather 64 of them into
their own UB half, and ``pl.aic_gather`` hands the reassembled ``[128, 512]`` tile
to the cube as a matmul B-operand. This replaces a GM staging buffer plus a
hand-written ``pl.system.set_ffts`` + ``sync_set``/``sync_wait`` barrier.

Two facts are guarded here, both of which the tensor-level spelling depends on:

1. ``tile.gather_row`` is admitted into the per-lane half-width dataflow on its
   lane-derived ``src_offset`` (``AddressArgs`` in LowerAutoVectorSplit). Before
   that, the region was rejected as "mixes explicit ... with plain full-width
   vector op(s) [tile.gather_row]" and there was no tensor-level spelling at all.
2. The cross-core ring must be sized down with ``pl.cross_core_slot``. The V2C
   ring defaults to 8 slots of the FULL popped tile (131072 x 8 = 1 MB), which is
   twice the 512KB L1 — so the default depth cannot express this shape.

Unit coverage of (1) lives in
``tests/ut/ir/transforms/test_lower_auto_vector_split.py``
(``test_region_admits_lane_localized_gather_row``); this test is the end-to-end
guard that the admitted region survives all the way through PTO codegen.
"""

import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import pypto.language as pl  # noqa: E402
from pypto.runtime import RunConfig  # noqa: E402

POOL_ROWS, HEAD_DIM = 16384, 512
ROWS, HALF, Q_ROWS = 128, 64, 16
DUMP_DIR = Path(__file__).resolve().parents[4] / "build_output" / "split_aiv_gather_row_codegen"


@pl.jit
def sparse_kv_qk(
    pool: pl.Tensor[[POOL_ROWS, HEAD_DIM], pl.BF16],
    idx: pl.Tensor[[ROWS], pl.INT32],
    q: pl.Tensor[[Q_ROWS, HEAD_DIM], pl.BF16],
    out: pl.Out[pl.Tensor[[Q_ROWS, ROWS], pl.FP32]],
):
    """Each AIV lane gathers half of a scattered row set into UB; the cube pops the
    reassembled tile and matmuls it as a b_trans B-operand."""
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="sparse_kv",
        allow_early_resolve=True,
        # 131072-byte slots; the 8-slot default would reserve 1MB of the 512KB L1.
        optimizations=[pl.cross_core_slot(slot_num=2)],
    ):
        for aiv in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            ub = pl.full([HALF, HEAD_DIM], dtype=pl.BF16, value=0.0)
            for k in pl.range(HALF):
                src = pl.cast(pl.read(idx, [aiv * HALF + k]), pl.INDEX)
                ub = pl.gather_row(ub, pool, [k, 0], [src, 0], [1, HEAD_DIM])
            kv = pl.aic_gather(ub)
        out[0:Q_ROWS, 0:ROWS] = pl.matmul(q, kv, b_trans=True, out_dtype=pl.FP32)
    return out


def test_per_lane_gather_row_lowers_to_split_aiv_kernels():
    """The region lowers to an AIV lane that gathers into a HALF UB tile and pushes
    it, and an AIC that pops the FULL tile — no GM staging, no manual barrier."""
    program = sparse_kv_qk.lower(config=RunConfig(platform="a2a3"))
    text = str(program)

    aiv = text.split("def sparse_kv_aiv")[1].split("def ")[0]
    # Half extent per lane, gathered at a lane-derived source offset.
    assert "pl.Tile[[64, 512], pl.BF16" in aiv
    assert "pl.tile.gather_row(" in aiv
    assert "pl.tile.tpush_to_aic(" in aiv

    aic = text.split("def sparse_kv_aic")[1].split("def ")[0]
    # The cube pops the reassembled FULL tile into L1 and matmuls it.
    assert "pl.tile.tpop_from_aiv(" in aic
    assert "pl.Tile[[128, 512], pl.BF16" in aic
    assert "pl.tile.matmul" in aic
    # The ring was sized by pl.cross_core_slot, not the 8-slot default.
    assert "slot_num=2" in aic
    assert "size=262144" in aic


def test_per_lane_gather_row_compiles_to_pto():
    """The lowered kernel survives PTO codegen and emits a .pto."""
    if DUMP_DIR.exists():
        shutil.rmtree(DUMP_DIR)

    cfg = RunConfig(
        platform="a2a3",
        codegen_only=True,
        save_kernels=True,
        save_kernels_dir=str(DUMP_DIR),
    )
    # PTO codegen writes the .pto before the downstream kernel-compilation step
    # (simpler_setup), which is absent in the codegen-only CI env. Capture any
    # such post-codegen failure — the guard below is on the emitted .pto, which
    # materializes first.
    compile_error: Exception | None = None
    try:
        sparse_kv_qk(
            torch.randn(POOL_ROWS, HEAD_DIM, dtype=torch.bfloat16),
            ((torch.arange(ROWS, dtype=torch.int64) * 977) % (POOL_ROWS - ROWS)).to(torch.int32),
            torch.randn(Q_ROWS, HEAD_DIM, dtype=torch.bfloat16),
            torch.empty(Q_ROWS, ROWS, dtype=torch.float32),
            config=cfg,
        )
    except Exception as e:  # noqa: BLE001 - see comment above
        compile_error = e

    ptos = sorted(DUMP_DIR.rglob("*.pto"))
    assert ptos, (
        f"codegen emitted no .pto under {DUMP_DIR} for a per-lane gather_row split_aiv region; "
        f"compile raised before .pto materialized: {compile_error!r}"
    )


@pl.jit
def _deep_ring_kv_qk(
    pool: pl.Tensor[[POOL_ROWS, HEAD_DIM], pl.BF16],
    idx: pl.Tensor[[ROWS], pl.INT32],
    q: pl.Tensor[[Q_ROWS, HEAD_DIM], pl.BF16],
    out: pl.Out[pl.Tensor[[Q_ROWS, ROWS], pl.FP32]],
):
    """Same kernel with an explicit 8-slot ring — 131072 x 8 = 1MB of a 512KB L1.

    The depth is stated rather than inherited: the automatic ring defaults to
    ``cross_core_pipe::kDefaultAutoPipeSlotNum`` (2), which fits here, so only an
    explicit request overflows L1 and exercises the diagnostic below.
    """
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="deep_ring",
        allow_early_resolve=True,
        optimizations=[pl.cross_core_slot(slot_num=8)],
    ):
        for aiv in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            ub = pl.full([HALF, HEAD_DIM], dtype=pl.BF16, value=0.0)
            for k in pl.range(HALF):
                src = pl.cast(pl.read(idx, [aiv * HALF + k]), pl.INDEX)
                ub = pl.gather_row(ub, pool, [k, 0], [src, 0], [1, HEAD_DIM])
            kv = pl.aic_gather(ub)
        out[0:Q_ROWS, 0:ROWS] = pl.matmul(q, kv, b_trans=True, out_dtype=pl.FP32)
    return out


def test_oversized_ring_depth_reports_the_reserved_bytes():
    """An 8-slot ring overflows L1, and the diagnostic must attribute the bytes and
    name the knob rather than reporting a bare number the author cannot act on.

    The reserve-buffer overflow is caught by AllocateMemoryAddresses' in-pass
    ``CHECK`` (``pypto::ValueError`` -> a builtin ``ValueError``), not by the
    ``AllocatedMemoryAddr`` verifier — which is why both carry the same note.
    """
    with pytest.raises(ValueError) as exc:
        _deep_ring_kv_qk.lower(config=RunConfig(platform="a2a3"))
    message = str(exc.value)
    assert "Mat buffer usage (1064960 bytes) exceeds platform limit (524288 bytes)" in message
    assert "The first 1048576 bytes of that space are reserved by system.reserve_buffer" in message
    assert "cross-core pipe ring" in message
    assert "pl.cross_core_slot(slot_num=N)" in message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
