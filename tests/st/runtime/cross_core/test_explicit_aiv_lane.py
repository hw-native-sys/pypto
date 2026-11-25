# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit logical lanes must select data independently of the physical AIV.

Compile and materialize goldens before borrowing a card. Each case transfers
twelve distinct tiles through a four-slot ring. Swapping the V2C lane must
swap output row halves. The A5 local-UB C2V path binds the current physical
slot, so its explicit pop lane matches the physical AIV. No cross-UB remap is
assumed. All comparisons use independent, exact integer-valued goldens.
"""

import pypto.language as pl
import pytest
import torch
from harness import st
from harness.core.harness import DataType, TensorSpec

ROWS, COLS, ITERATIONS = 32, 16, 12
SLOT_BYTES, SLOT_COUNT = ROWS * COLS * 4, 4
BUFFER_BYTES = SLOT_BYTES * SLOT_COUNT


def _program(kind, direction):
    flip_push = direction == "push"

    @pl.jit.inline(auto_scope=False)
    def static_push(value: pl.Tile[[ROWS, COLS], pl.BF16], physical: pl.Scalar[pl.INDEX]):
        if physical == 0:
            if flip_push:
                pl.tpush_to_aic(value, split=1, aiv_subblockid=1)
            else:
                pl.tpush_to_aic(value, split=1, aiv_subblockid=0)
        elif flip_push:
            pl.tpush_to_aic(value, split=1, aiv_subblockid=0)
        else:
            pl.tpush_to_aic(value, split=1, aiv_subblockid=1)

    @pl.jit.inline(auto_scope=False)
    def static_receive(
        physical: pl.Scalar[pl.INDEX],
        iteration: pl.Scalar[pl.INDEX],
        out: pl.Tensor[[ITERATIONS * ROWS, COLS], pl.FP32],
    ):
        # Pop/store/free stay together in each branch: merging a popped tile
        # through a phi would lose the verifier's per-value lifetime pairing.
        if physical == 0:
            received = pl.tpop_from_aic(shape=[ROWS, COLS], dtype=pl.FP32, split=1, aiv_subblockid=0)
            out = pl.store(received, [iteration * ROWS, 0], out)
            pl.tfree_to_aic(received)
        else:
            received = pl.tpop_from_aic(shape=[ROWS, COLS], dtype=pl.FP32, split=1, aiv_subblockid=1)
            out = pl.store(received, [iteration * ROWS, 0], out)
            pl.tfree_to_aic(received)
        return out

    def shared(
        a: pl.Tensor[[ITERATIONS * ROWS, COLS], pl.BF16],
        b: pl.Tensor[[COLS, COLS], pl.BF16],
        out: pl.Out[pl.Tensor[[ITERATIONS * ROWS, COLS], pl.FP32]],
        cube_role: pl.constexpr,
    ):
        pl.func_attr({"split": pl.SplitMode.UP_DOWN})
        if cube_role:
            v2c = pl.reserve_buffer(name="lane_v2c", size=BUFFER_BYTES)
            c2v = pl.import_peer_buffer(name="lane_c2v", peer_func="consumer")
            pl.aic_initialize_pipe(
                dir_mask=3,
                slot_size=SLOT_BYTES,
                slot_num=SLOT_COUNT,
                c2v_consumer_buf=c2v,
                v2c_consumer_buf=v2c,
            )
            for iteration in pl.range(ITERATIONS):
                incoming: pl.Tile[[ROWS, COLS], pl.BF16, pl.Mem.Mat] = pl.tpop_from_aiv(split=1)
                left = pl.move(incoming, target_memory=pl.Mem.Left)
                b_mat = pl.load(b, [0, 0], [COLS, COLS], target_memory=pl.Mem.Mat)
                right = pl.move(b_mat, target_memory=pl.Mem.Right)
                product = pl.matmul(left, right)
                pl.tfree_to_aiv(incoming)
                pl.tpush_to_aiv(product, split=1)
        else:
            c2v = pl.reserve_buffer(name="lane_c2v", size=BUFFER_BYTES)
            v2c = pl.import_peer_buffer(name="lane_v2c", peer_func="producer")
            pl.aiv_initialize_pipe(
                dir_mask=3,
                slot_size=SLOT_BYTES,
                slot_num=SLOT_COUNT,
                c2v_consumer_buf=c2v,
                v2c_consumer_buf=v2c,
            )
            physical = pl.tile.get_subblock_idx()
            lane = pl.cast(physical, pl.INT64)
            one: pl.Scalar[pl.INT64] = 1
            opposite = one - lane
            for iteration in pl.range(ITERATIONS):
                # Each pre-generated frame has distinct row and iteration tags.
                tagged = pl.load(a, [iteration * ROWS, 0], [ROWS, COLS])
                if kind == "omit":
                    pl.tpush_to_aic(tagged, split=1)
                    received = pl.tpop_from_aic(shape=[ROWS, COLS], dtype=pl.FP32, split=1)
                elif kind == "dynamic":
                    if flip_push:
                        pl.tpush_to_aic(tagged, split=1, aiv_subblockid=opposite)
                    else:
                        pl.tpush_to_aic(tagged, split=1, aiv_subblockid=lane)
                    received = pl.tpop_from_aic(
                        shape=[ROWS, COLS], dtype=pl.FP32, split=1, aiv_subblockid=lane
                    )
                else:
                    static_push(tagged, physical)
                    out = static_receive(physical, iteration, out)
                if kind != "static":
                    out = pl.store(received, [iteration * ROWS, 0], out)
                    pl.tfree_to_aic(received)
        return out

    producer = pl.jit.incore(level=pl.Level.AIC)(shared)
    consumer = pl.jit.incore(level=pl.Level.AIV)(shared)

    @pl.jit.incore(level=pl.Level.CORE_GROUP)
    def group(
        a: pl.Tensor[[ITERATIONS * ROWS, COLS], pl.BF16],
        b: pl.Tensor[[COLS, COLS], pl.BF16],
        out: pl.Out[pl.Tensor[[ITERATIONS * ROWS, COLS], pl.FP32]],
    ):
        produced = producer(a, b, out, True)
        return consumer(a, b, produced, False)

    @pl.jit
    def main(
        a: pl.Tensor[[ITERATIONS * ROWS, COLS], pl.BF16],
        b: pl.Tensor[[COLS, COLS], pl.BF16],
        out: pl.Out[pl.Tensor[[ITERATIONS * ROWS, COLS], pl.FP32]],
    ):
        with pl.spmd(1, sync_start=True):
            group(a, b, out)
        return out

    return main


def _case(kind, direction):
    # Nonuniform, exactly representable operands distinguish both lanes and
    # each ring reuse without needing a floating-point tolerance.
    base = (torch.arange(1, ROWS + 1).reshape(ROWS, 1) + torch.arange(COLS).reshape(1, COLS) % 4).to(
        torch.bfloat16
    )
    a = torch.cat([base + iteration for iteration in range(ITERATIONS)])
    b = (1 + torch.arange(COLS).reshape(COLS, 1) % 3 + torch.arange(COLS).reshape(1, COLS) % 5).to(
        torch.bfloat16
    )
    expected = []
    for iteration in range(ITERATIONS):
        frame = a[iteration * ROWS : (iteration + 1) * ROWS]
        product = frame.float() @ b.float()
        if direction == "push":
            product = torch.cat((product[ROWS // 2 :], product[: ROWS // 2]))
        expected.append(product)
    golden = torch.cat(expected)
    return st.case(
        _program(kind, direction),
        tensors=[
            TensorSpec("a", [ITERATIONS * ROWS, COLS], DataType.BF16, init_value=a),
            TensorSpec("b", [COLS, COLS], DataType.BF16, init_value=b),
            TensorSpec("out", [ITERATIONS * ROWS, COLS], DataType.FP32, is_output=True),
        ],
        name=f"explicit_lane_{kind}_{direction}",
        golden=lambda _: {"out": golden},
        rtol=0,
        atol=0,
    )


@pytest.mark.platforms("a2a3", "a5")
@st.cases(
    _case("omit", "none"),
    *(_case(kind, direction) for kind in ("dynamic", "static") for direction in ("push", "pop")),
)
def test_explicit_aiv_lane(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
