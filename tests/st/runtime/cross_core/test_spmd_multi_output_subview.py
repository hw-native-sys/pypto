# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Regression test for issue #1723: fused multi-output spmd scope miscompiles.

A single spmd scope computes two outputs: a per-token combine (post/comb
weights) and a per-head reduction (transpose -> per-row reshape to [TILE, 1]
-> row_expand_mul). MemoryReuse retargets the transposed tile's buffer onto an
earlier dead buffer and used to propagate the new MemRef wholesale to every
sharing-group member, collapsing the per-head row subviews onto offset 0 — all
four heads then read head 0's coefficients and x_mixed comes out wrong. The
same reduction in its own spmd scope was unaffected.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.ir.pass_manager import OptimizationStrategy

TILE = 16
HC = 4
HC_PAD = 8
HD = 512  # per-head hidden dim
D = HC * HD
EPS = 1e-3
COMBINE_TOKENS = 2


@pl.program
class SPMDMultiOutputSubviewProgram:
    """Fused multi-output spmd scope: per-token combine + per-head reduction."""

    @pl.function(type=pl.FunctionType.InCore)
    def fused(
        self,
        x: pl.Tensor[[TILE, D], pl.FP32],
        pre: pl.Tensor[[TILE, HC_PAD], pl.FP32],
        post: pl.Tensor[[TILE, HC_PAD], pl.FP32],
        comb: pl.Tensor[[TILE, HC * HC], pl.FP32],
        y: pl.Out[pl.Tensor[[COMBINE_TOKENS, D], pl.FP32]],
        x_mixed: pl.Out[pl.Tensor[[TILE, HD], pl.FP32]],
    ) -> tuple[pl.Tensor[[COMBINE_TOKENS, D], pl.FP32], pl.Tensor[[TILE, HD], pl.FP32]]:
        # Per-token combine via post/comb weights — the other output that makes
        # this a fused multi-output scope (and creates the earlier dead buffers
        # MemoryReuse retargets onto).
        for tt in pl.range(COMBINE_TOKENS):
            x_row = pl.load(x, [tt, 0], [1, HD])
            for out_h in pl.range(HC):
                post_w = pl.read(post, [tt, out_h])
                y_row = pl.mul(x_row, post_w)
                for in_h in pl.range(HC):
                    comb_w = pl.read(comb, [tt, in_h * HC + out_h])
                    res_row = pl.load(x, [tt, in_h * HD], [1, HD])
                    y_row = pl.add(y_row, pl.mul(res_row, comb_w))
                y = pl.store(y_row, [tt, out_h * HD], y)
        # Per-head reduction: transposed scales sliced row-by-row into [TILE, 1]
        # subviews — these must keep distinct offsets within pre_t's buffer.
        pre_tile = pl.load(pre, [0, 0], [TILE, HC_PAD])
        pre_eps = pl.add(pre_tile, EPS)
        pre_t = pl.transpose(pre_eps, axis1=0, axis2=1)
        pre0 = pl.reshape(pre_t[0:1, 0:TILE], [TILE, 1])
        pre1 = pl.reshape(pre_t[1:2, 0:TILE], [TILE, 1])
        pre2 = pl.reshape(pre_t[2:3, 0:TILE], [TILE, 1])
        pre3 = pl.reshape(pre_t[3:4, 0:TILE], [TILE, 1])
        x0 = pl.load(x, [0, 0 * HD], [TILE, HD])
        x1 = pl.load(x, [0, 1 * HD], [TILE, HD])
        x2 = pl.load(x, [0, 2 * HD], [TILE, HD])
        x3 = pl.load(x, [0, 3 * HD], [TILE, HD])
        y0 = pl.row_expand_mul(x0, pre0)
        y1 = pl.row_expand_mul(x1, pre1)
        y2 = pl.row_expand_mul(x2, pre2)
        y3 = pl.row_expand_mul(x3, pre3)
        x_mixed = pl.store(pl.add(pl.add(y0, y1), pl.add(y2, y3)), [0, 0], x_mixed)
        return y, x_mixed

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        x: pl.Tensor[[TILE, D], pl.FP32],
        pre: pl.Tensor[[TILE, HC_PAD], pl.FP32],
        post: pl.Tensor[[TILE, HC_PAD], pl.FP32],
        comb: pl.Tensor[[TILE, HC * HC], pl.FP32],
        y: pl.Out[pl.Tensor[[COMBINE_TOKENS, D], pl.FP32]],
        x_mixed: pl.Out[pl.Tensor[[TILE, HD], pl.FP32]],
    ) -> tuple[pl.Tensor[[COMBINE_TOKENS, D], pl.FP32], pl.Tensor[[TILE, HD], pl.FP32]]:
        with pl.spmd(1, name_hint="fused_multi_out"):
            y, x_mixed = self.fused(x, pre, post, comb, y, x_mixed)
        return y, x_mixed


class SPMDMultiOutputSubviewTestCase(PTOTestCase):
    """Fused multi-output spmd: per-head scales must keep distinct buffers."""

    def get_name(self) -> str:
        return "spmd_multi_output_subview"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("x", [TILE, D], DataType.FP32, init_value=torch.randn),
            TensorSpec("pre", [TILE, HC_PAD], DataType.FP32, init_value=torch.rand),
            TensorSpec("post", [TILE, HC_PAD], DataType.FP32, init_value=torch.rand),
            TensorSpec("comb", [TILE, HC * HC], DataType.FP32, init_value=torch.rand),
            TensorSpec("y", [COMBINE_TOKENS, D], DataType.FP32, is_output=True),
            TensorSpec("x_mixed", [TILE, HD], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return SPMDMultiOutputSubviewProgram

    def compute_expected(self, tensors, params=None):
        x = tensors["x"]
        pre_eps = tensors["pre"] + EPS
        post = tensors["post"]
        comb = tensors["comb"]

        for t in range(COMBINE_TOKENS):
            for oh in range(HC):
                row = x[t, 0:HD] * post[t, oh]
                for ih in range(HC):
                    row = row + x[t, ih * HD : (ih + 1) * HD] * comb[t, ih * HC + oh]
                tensors["y"][t, oh * HD : (oh + 1) * HD] = row

        mixed = torch.zeros(TILE, HD)
        for h in range(HC):
            mixed += x[:, h * HD : (h + 1) * HD] * pre_eps[:, h : h + 1]
        tensors["x_mixed"][:] = mixed


class TestSPMDMultiOutputSubview:
    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_spmd_multi_output_subview(self, test_runner, platform):
        """Per-head [TILE,1] subview scales in a fused multi-output spmd scope."""
        result = test_runner.run(SPMDMultiOutputSubviewTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
