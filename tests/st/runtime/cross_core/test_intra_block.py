# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 single-AIV handshakes with distinct tags and repeated event reuse.

Run each lane case with the hardware harness and an external timeout. There is
no TPipe or SYNCALL, so local IDs 4/5 are owned exclusively by this protocol.
Cube uses 4/5 for AIV0 and 20/21 for AIV1; either AIV uses local 4/5.
The Cube waits on MTE1 and then drains PIPE_ALL before the next iteration,
so scalar dispatch cannot outrun acknowledgements. This legal-pipe protocol
runs independently of the named PIPE_S verifier support checked by the UT.
"""

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
            for iteration in pl.range(ITERATIONS):
                pl.system.set_intra_block(4 + 16 * lane, pipe=pl.PipeType.MTE1, core_type=pl.KernelType.AIC)
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    if aiv_id == lane:
                        pl.system.wait_intra_block(4, pipe=pl.PipeType.MTE2, core_type=pl.KernelType.AIV)
                        tile = pl.load(data, [iteration, lane * LANE_WIDTH], [1, LANE_WIDTH])
                        tagged = pl.add(tile, 1000.0)
                        pl.store(tagged, [iteration, lane * LANE_WIDTH], output)
                        pl.system.set_intra_block(5, pipe=pl.PipeType.MTE3, core_type=pl.KernelType.AIV)
                pl.system.wait_intra_block(5 + 16 * lane, pipe=pl.PipeType.MTE1, core_type=pl.KernelType.AIC)
                pl.system.bar_all()
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
