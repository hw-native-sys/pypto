# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed runtime test: tuple-return worker calls.

Exercises the distributed codegen tuple_element_tensors_ mapping added in
commit c98c36fc ("feat(codegen): support tuple-return worker calls in
distributed codegen").

Three hierarchy levels:
  - HOST Orchestrator: dispatches chip worker, unpacks tuple return
  - Chip Orchestration: manages InCore kernel dispatch on device
  - InCore: tile-level kernel computing sum and diff simultaneously

Computation:
  - out_sum  = a + b
  - out_diff = a - b

Verifies: tuple-returning chip worker correctly maps each tuple element
to its corresponding Out parameter tensor via TupleGetItemExpr unpacking
in the distributed codegen.
"""

import sys

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig


@pl.program
class L3TupleReturnProgram:
    """L3: HOST orch → CHIP worker with tuple return (sum + diff)."""

    @pl.function(type=pl.FunctionType.InCore)
    def tile_sum_diff(
        self,
        a: pl.Tensor[[128, 64], pl.FP32],
        b: pl.Tensor[[128, 64], pl.FP32],
        out_sum: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
        out_diff: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
    ) -> tuple[pl.Tensor[[128, 64], pl.FP32], pl.Tensor[[128, 64], pl.FP32]]:
        # Peak Vec (UB) footprint is 3 live 128x64 FP32 tiles = 96KB (a, b, and
        # sum/diff). 128x128 would reach 192KB and is rejected by AllocateMemoryAddr
        # after pto-isa#170 lowered the safe a2a3 UB to 184KB (see soc.cpp).
        tile_a = pl.load(a, [0, 0], [128, 64])
        tile_b = pl.load(b, [0, 0], [128, 64])
        tile_sum = pl.add(tile_a, tile_b)
        tile_diff = pl.sub(tile_a, tile_b)
        r_sum = pl.store(tile_sum, [0, 0], out_sum)
        r_diff = pl.store(tile_diff, [0, 0], out_diff)
        return r_sum, r_diff

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        a: pl.Tensor[[128, 64], pl.FP32],
        b: pl.Tensor[[128, 64], pl.FP32],
        out_sum: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
        out_diff: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
    ) -> tuple[pl.Tensor[[128, 64], pl.FP32], pl.Tensor[[128, 64], pl.FP32]]:
        s, d = self.tile_sum_diff(a, b, out_sum, out_diff)
        return s, d

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        a: pl.Tensor[[128, 64], pl.FP32],
        b: pl.Tensor[[128, 64], pl.FP32],
        out_sum: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
        out_diff: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
    ) -> tuple[pl.Tensor[[128, 64], pl.FP32], pl.Tensor[[128, 64], pl.FP32]]:
        s, d = self.chip_orch(a, b, out_sum, out_diff)
        return s, d


class TestL3TupleReturn:
    """L3 distributed runtime: tuple-return worker calls."""

    def test_execute(self, test_config, device_ids):
        """End-to-end: compile + execute, verify out_sum = a+b, out_diff = a-b."""
        compiled = ir.compile(
            L3TupleReturnProgram,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:1],
                block_dim=3,
                aicpu_thread_num=4,
            ),
        )

        a = torch.full((128, 64), 2.0, dtype=torch.float32)
        b = torch.full((128, 64), 3.0, dtype=torch.float32)
        out_sum = torch.zeros((128, 64), dtype=torch.float32)
        out_diff = torch.zeros((128, 64), dtype=torch.float32)

        compiled(a, b, out_sum, out_diff)

        expected_sum = torch.full((128, 64), 5.0, dtype=torch.float32)
        expected_diff = torch.full((128, 64), -1.0, dtype=torch.float32)
        assert torch.allclose(out_sum, expected_sum, rtol=1e-5, atol=1e-5), (
            f"Tuple return sum failed: expected a + b = 5.0, "
            f"got max diff = {(out_sum - expected_sum).abs().max().item()}"
        )
        assert torch.allclose(out_diff, expected_diff, rtol=1e-5, atol=1e-5), (
            f"Tuple return diff failed: expected a - b = -1.0, "
            f"got max diff = {(out_diff - expected_diff).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
