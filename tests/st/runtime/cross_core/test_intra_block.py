# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 single-AIV pipeline handshakes with distinct tags and repeated event reuse.

Run each lane case with the hardware harness and an external timeout. There is
no TPipe or SYNCALL, so local IDs 4/5 are owned exclusively by this protocol.
Cube uses 4/5 for AIV0 and 20/21 for AIV1; either AIV uses local 4/5.
The Cube waits on MTE1 and then drains PIPE_ALL before the next iteration,
so scalar dispatch cannot outrun acknowledgements. This legal-pipe protocol
runs independently of the named PIPE_S verifier support checked by the UT.
"""

import sys

import pypto.language as pl
import pytest
import torch
from harness import st

ITERATIONS = 8
LANE_WIDTH = 16


def _handshake_kernel(lane: int):
    """Specialize the selected AIV through lexical control, not a sync lane API."""

    @pl.jit
    def handshake(
        data: pl.Tensor[[ITERATIONS, 2 * LANE_WIDTH], pl.FP32],
        output: pl.InOut[pl.Tensor[[ITERATIONS, 2 * LANE_WIDTH], pl.FP32]],
    ):
        for _ in pl.spmd(1):
            # The two explicitly pinned loops become concurrent AIC/AIV kernels.
            # Keep the Cube barrier outside the AIV pipeline: mainline rejects
            # manual pipe barriers in pipeline bodies before named-event lowering.
            for iteration in pl.range(ITERATIONS):
                pl.system.set_intra_block(4 + 16 * lane, pipe=pl.PipeType.MTE1, core_type=pl.KernelType.AIC)
                pl.system.wait_intra_block(5 + 16 * lane, pipe=pl.PipeType.MTE1, core_type=pl.KernelType.AIC)
                pl.system.barrier(pipe=pl.PipeType.ALL, core_type=pl.KernelType.AIC)
            for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                if aiv_id == lane:
                    for iteration in pl.pipeline(ITERATIONS, stage=1):
                        pl.system.wait_intra_block(4, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                        tile = pl.load(data, [iteration, lane * LANE_WIDTH], [1, LANE_WIDTH])
                        tagged = pl.add(tile, 1000.0)
                        pl.store(tagged, [iteration, lane * LANE_WIDTH], output)
                        pl.system.set_intra_block(5, pipe=pl.PipeType.MTE3, core_type=pl.KernelType.AIV)
        return output

    return handshake


def _lane_case(lane: int):
    # Integer-valued FP32 tags are exactly representable, distinct by lane and
    # iteration; the unselected lane must remain zero throughout the handshake.
    data = torch.arange(ITERATIONS * 2 * LANE_WIDTH, dtype=torch.float32).reshape(ITERATIONS, -1)
    output = torch.zeros_like(data)

    def golden(tensors):
        expected = torch.zeros_like(tensors["output"])
        start = lane * LANE_WIDTH
        expected[:, start : start + LANE_WIDTH] = tensors["data"][:, start : start + LANE_WIDTH] + 1000
        return {"output": expected}

    return st.case(
        _handshake_kernel(lane),
        data,
        output,
        name=f"intra_block_aiv{lane}",
        golden=golden,
        rtol=0,
        atol=0,
    )


@pytest.mark.platforms("a5")
@st.cases(_lane_case(0), _lane_case(1))
def test_intra_block_single_aiv(case_run):
    """Each case must finish under the runner timeout and match every output tag."""
    case_run.assert_passed()


def _payload_kernel(lane: int):
    """Produce each tagged tile with Cube FIX, then reuse one GM payload slot."""

    @pl.jit.incore(level=pl.Level.AIC)
    def produce(
        data: pl.Tensor[[ITERATIONS * LANE_WIDTH, LANE_WIDTH], pl.FP16],
        identity: pl.Tensor[[LANE_WIDTH, LANE_WIDTH], pl.FP16],
        slot: pl.InOut[pl.Tensor[[LANE_WIDTH, LANE_WIDTH], pl.FP32]],
    ):
        for iteration in pl.range(ITERATIONS):
            source = pl.load(data, [iteration * LANE_WIDTH, 0], [LANE_WIDTH, LANE_WIDTH])
            unit = pl.load(identity, [0, 0], [LANE_WIDTH, LANE_WIDTH])
            produced = pl.matmul(source, unit, out_dtype=pl.FP32)
            pl.store(produced, [0, 0], slot)
            # FIX ready retires the Cube writeback before the peer's load.
            # ACK and PIPE_ALL protect the same slot on the next iteration.
            pl.system.set_intra_block(8 + 16 * lane, pipe=pl.PipeType.FIX, core_type=pl.KernelType.AIC)
            pl.system.wait_intra_block(9 + 16 * lane, pipe=pl.PipeType.MTE1, core_type=pl.KernelType.AIC)
            pl.system.barrier(pipe=pl.PipeType.ALL, core_type=pl.KernelType.AIC)
        return slot

    @pl.jit.incore(level=pl.Level.AIV)
    def consume(
        slot: pl.Tensor[[LANE_WIDTH, LANE_WIDTH], pl.FP32],
        output: pl.InOut[pl.Tensor[[ITERATIONS * LANE_WIDTH, LANE_WIDTH], pl.FP32]],
    ):
        # This explicit AIV kernel owns full-width tiles and lane selection.
        # Dispatch both lanes without applying automatic shape halving.
        pl.func_attr({"split_aiv": True, "split": pl.SplitMode.NONE})
        if pl.tile.get_subblock_idx() == lane:
            for iteration in pl.range(ITERATIONS):
                pl.system.wait_intra_block(8, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                tile = pl.load(slot, [0, 0], [LANE_WIDTH, LANE_WIDTH])
                pl.store(tile, [iteration * LANE_WIDTH, 0], output)
                pl.system.set_intra_block(9, pipe=pl.PipeType.MTE3, core_type=pl.KernelType.AIV)
        return output

    @pl.jit.incore(level=pl.Level.CORE_GROUP)
    def group(
        data: pl.Tensor[[ITERATIONS * LANE_WIDTH, LANE_WIDTH], pl.FP16],
        identity: pl.Tensor[[LANE_WIDTH, LANE_WIDTH], pl.FP16],
        slot: pl.InOut[pl.Tensor[[LANE_WIDTH, LANE_WIDTH], pl.FP32]],
        output: pl.InOut[pl.Tensor[[ITERATIONS * LANE_WIDTH, LANE_WIDTH], pl.FP32]],
    ):
        # Explicit peers own the GM ready/ack protocol. Keep it separate from
        # automatic C2V promotion of an unsynchronized mixed-kernel exchange.
        slot = produce(data, identity, slot)
        output = consume(slot, output)
        return slot, output

    @pl.jit(auto_scope=False)
    def payload_handshake(
        data: pl.Tensor[[ITERATIONS * LANE_WIDTH, LANE_WIDTH], pl.FP16],
        identity: pl.Tensor[[LANE_WIDTH, LANE_WIDTH], pl.FP16],
        slot: pl.InOut[pl.Tensor[[LANE_WIDTH, LANE_WIDTH], pl.FP32]],
        output: pl.InOut[pl.Tensor[[ITERATIONS * LANE_WIDTH, LANE_WIDTH], pl.FP32]],
    ):
        with pl.spmd(1):
            group(data, identity, slot, output)
        return slot, output

    return payload_handshake


def _payload_case(lane: int):
    # Integers 1..2048 and their negatives are exact in FP16. Multiplication
    # by identity is exact in FP32, so every output row has a strict golden.
    data = torch.arange(1, ITERATIONS * LANE_WIDTH * LANE_WIDTH + 1, dtype=torch.float16)
    data = data.reshape(ITERATIONS * LANE_WIDTH, LANE_WIDTH)
    if lane:
        data = -data
    identity = torch.eye(LANE_WIDTH, dtype=torch.float16)
    slot = torch.zeros((LANE_WIDTH, LANE_WIDTH), dtype=torch.float32)
    output = torch.zeros((ITERATIONS * LANE_WIDTH, LANE_WIDTH), dtype=torch.float32)

    def golden(tensors):
        expected = tensors["data"].to(torch.float32)
        return {"slot": expected[-LANE_WIDTH:].clone(), "output": expected}

    return st.case(
        _payload_kernel(lane),
        data,
        identity,
        slot,
        output,
        name=f"intra_block_payload_aiv{lane}",
        golden=golden,
        rtol=0,
        atol=0,
    )


@pytest.mark.platforms("a5")
@st.cases(_payload_case(0), _payload_case(1))
def test_intra_block_payload_reuse(case_run):
    """Every output row depends on ready and the ack protecting slot reuse."""
    case_run.assert_passed()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
