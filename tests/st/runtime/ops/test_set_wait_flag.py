# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-device MTE2→V and V→MTE3 handoff with explicit flags.

The static flags make PTOAS skip automatic synchronization for this kernel.
The numerical result therefore checks the handoffs used by the load, vector
operation, and store; the saved PTO checks that both authored pairs reached
the backend in the intended order.
"""

from dataclasses import replace

import pypto.language as pl
import pytest
import torch

ROWS = 1
COLS = 256


@pl.jit
def explicit_flags_double(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for _ in pl.spmd(1):
        loaded: pl.Tile[[ROWS, COLS], pl.FP32, pl.Mem.Vec] = pl.load(x, [0, 0], [ROWS, COLS])
        pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.V, 0)
        pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.V, 0)
        doubled: pl.Tile[[ROWS, COLS], pl.FP32, pl.Mem.Vec] = pl.add(loaded, loaded)
        pl.system.set_flag(pl.PipeType.V, pl.PipeType.MTE3, 0)
        pl.system.wait_flag(pl.PipeType.V, pl.PipeType.MTE3, 0)
        out = pl.store(doubled, [0, 0], out)
    return out


@pytest.mark.platforms("a2a3", "a2a3sim")
def test_flags_order_load_compute_store_on_device(test_config, tmp_path):
    """Manual flags preserve data through two asynchronous pipe handoffs."""
    config = replace(test_config, save_kernels=True, save_kernels_dir=str(tmp_path))
    x = torch.randn(ROWS, COLS, dtype=torch.float32)
    out = torch.zeros_like(x)

    compiled = explicit_flags_double.compile(x, out, config=config)
    compiled(x, out, config=config)

    assert torch.equal(out, x + x)
    pto_sources = [path.read_text() for path in tmp_path.rglob("*.pto")]
    assert pto_sources, "compilation did not save the PTO kernel"
    pto = "\n".join(pto_sources)
    handoffs = (
        'pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]',
        'pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]',
        'pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]',
        'pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]',
    )
    positions = [pto.find(op) for op in handoffs]
    assert all(position >= 0 for position in positions), "manual flag pair is missing from PTO output"
    assert positions == sorted(positions), "manual flag pairs crossed their authored order"
