# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed runtime test: two independent chip tasks + SubWorker reduce.

Three hierarchy levels:
  - HOST Orchestrator: dispatches two independent chip workers + SubWorker reduce
  - Chip Orchestration: manages InCore kernel dispatch on device
  - InCore: tile-level vector add / sub kernels

Computation:
  - Chip task 1: sum_ab = a + b     (independent)
  - Chip task 2: diff_ab = a - b    (independent)
  - SubWorker:   f = sum_ab + diff_ab = 2a

Verifies: multiple independent chip tasks in DAG, SubWorker aggregation
of two chip outputs, cross-fork data visibility for multi-tensor flows.
"""

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig


@pl.program
class L3ParallelReduceProgram:
    """L3: HOST orch → 2 independent chip workers (a+b, a-b) → SubWorker (reduce)."""

    @pl.function(type=pl.FunctionType.InCore)
    def tile_add(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_f = pl.add(tile_a, tile_b)
        out_f = pl.store(tile_f, [0, 0], f)
        return out_f

    @pl.function(type=pl.FunctionType.InCore)
    def tile_sub(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        tile_f = pl.sub(tile_a, tile_b)
        out_f = pl.store(tile_f, [0, 0], f)
        return out_f

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch_add(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        out_f = self.tile_add(a, b, f)
        return out_f

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch_sub(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        out_f = self.tile_sub(a, b, f)
        return out_f

    @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
    def reduce_sum(
        sum_ab: pl.Tensor[[128, 128], pl.FP32],
        diff_ab: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        result = torch.add(sum_ab, diff_ab)
        f[:] = result
        return f

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        sum_ab: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], dtype=pl.FP32)
        diff_ab: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor([128, 128], dtype=pl.FP32)
        out_sum: pl.Tensor[[128, 128], pl.FP32] = self.chip_orch_add(a, b, sum_ab)
        out_diff: pl.Tensor[[128, 128], pl.FP32] = self.chip_orch_sub(a, b, diff_ab)
        out_f: pl.Tensor[[128, 128], pl.FP32] = self.reduce_sum(out_sum, out_diff, f)
        return out_f


class TestL3ParallelReduce:
    """L3 distributed runtime: two independent chip tasks + SubWorker reduce."""

    def test_execute(self, test_config):
        """End-to-end: compile + execute, verify f = (a+b) + (a-b) = 2a."""
        compiled = ir.compile(
            L3ParallelReduceProgram,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=[7],
                num_sub_workers=1,
                block_dim=3,
                aicpu_thread_num=4,
            ),
        )

        a = torch.full((128, 128), 2.0, dtype=torch.float32)
        b = torch.full((128, 128), 3.0, dtype=torch.float32)

        # Return-style: framework auto-allocates the Out tensor (f)
        # sum_ab and diff_ab are local intermediates created via pl.create_tensor
        f = compiled(a, b)

        # f = (a + b) + (a - b) = 2a = 4.0
        expected = torch.full((128, 128), 4.0, dtype=torch.float32)
        assert torch.allclose(f, expected, rtol=1e-5, atol=1e-5), (
            f"L3 parallel reduce test failed: expected f = 2a = 4.0, "
            f"got max diff = {(f - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
