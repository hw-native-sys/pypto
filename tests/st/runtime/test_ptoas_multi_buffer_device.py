# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-device numeric parity for the ptoas multi-buffer switch (use_ptoas_multi_buffer).

Runs a 2-stage pipeline that accumulates four i-dependent [64, N] vec tiles.
With the switch on, `ConvertToPtoasMultiBuffer` lowers the rotating load to a
ptoas multi-buffer region (slot i%N) and auto-selects memory_planner=PTOAS
(--pto-level=level2), where ptoas PlanMemory delivers the cross-iteration
double-buffer overlap. This verifies the lowered kernel is numerically correct
on real hardware and matches the switch-off (native pipeline) result.
"""

import dataclasses

import pypto.language as pl
import pytest
import torch

T, N = 256, 64


@pl.jit
def mbuf_accum(
    a: pl.Tensor[[T, N], pl.FP32],
    c: pl.Out[pl.Tensor[[T, N], pl.FP32]],
):
    """c[0:64] = a[0:64] + a[64:128] + a[128:192] + a[192:256]."""
    with pl.at(level=pl.Level.CORE_GROUP):
        acc = pl.load(a, [0, 0], [64, N])
        for _i in pl.pipeline(1, 4, stage=2):  # noqa: B007 - pipeline index
            t = pl.load(a, [_i * 64, 0], [64, N])
            acc = pl.add(acc, t)
        c = pl.store(acc, [0, 0], c)
    return c


def _reference(a: torch.Tensor) -> torch.Tensor:
    return a[0:64] + a[64:128] + a[128:192] + a[192:256]


class TestPtoasMultiBufferDevice:
    """On-device numeric parity for the ptoas multi-buffer switch."""

    def test_multi_buffer_numeric_parity(self, test_config):
        """Switch-on (multi-buffer, level2) must match the torch reference and
        the switch-off (native pipeline) result on real hardware."""
        a = torch.randn(T, N, dtype=torch.float32)
        expected = _reference(a)

        # Switch OFF: native pipeline ping-pong (default planner / level3).
        mbuf_accum._cache.clear()
        c_off = torch.zeros(T, N, dtype=torch.float32)
        mbuf_accum(a, c_off, config=dataclasses.replace(test_config, use_ptoas_multi_buffer=False))

        # Switch ON: ptoas multi-buffer (auto memory_planner=PTOAS / level2).
        mbuf_accum._cache.clear()
        c_on = torch.zeros(T, N, dtype=torch.float32)
        mbuf_accum(a, c_on, config=dataclasses.replace(test_config, use_ptoas_multi_buffer=True))

        assert torch.allclose(c_off[0:64], expected, rtol=1e-4, atol=1e-4), (
            f"switch-off diff = {(c_off[0:64] - expected).abs().max().item()}"
        )
        assert torch.allclose(c_on[0:64], expected, rtol=1e-4, atol=1e-4), (
            f"multi-buffer diff vs reference = {(c_on[0:64] - expected).abs().max().item()}"
        )
        assert torch.allclose(c_on[0:64], c_off[0:64], rtol=1e-5, atol=1e-5), (
            f"multi-buffer vs native diff = {(c_on[0:64] - c_off[0:64]).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
