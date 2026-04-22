# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed runtime test: HOST orchestrator → CHIP worker → SubWorker.

Equivalent to simpler's test_l3_dependency.py but written in PyPTO DSL.

Three hierarchy levels:
  - HOST Orchestrator: dispatches chip work + SubWorker verification
  - Chip Orchestration: manages InCore kernel dispatch on device
  - InCore: tile-level vector add kernel (a + b → f)

Computation: f = a + b
Verifies: TensorMap dependency inference, cross-fork data visibility,
DAG ordering (SubWorker runs after chip completes).
"""

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

SIZE = 128 * 128


@pl.program
class L3DependencyProgram:
    """L3: HOST orch → CHIP worker (a + b) → SubWorker (verify)."""

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

    @pl.function(type=pl.FunctionType.Orchestration)
    def chip_orch(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        out_f = self.tile_add(a, b, f)
        return out_f

    @pl.function(level=pl.Level.HOST, role=pl.Role.Worker)
    def verify(self, f: pl.Tensor[[128, 128], pl.FP32]):
        expected = torch.full((128, 128), 5.0, dtype=torch.float32)
        if not torch.allclose(f, expected, rtol=1e-5, atol=1e-5):
            raise AssertionError(
                f"SubWorker verify failed: expected 5.0, got max={f.max().item()}, min={f.min().item()}"
            )

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        out_f: pl.Tensor[[128, 128], pl.FP32] = self.chip_orch(a, b, f)
        self.verify(out_f)
        return out_f


@pl.program
class L3DependencyInlineProgram:
    """L3: all levels inlined into host_orch via pl.at() using tensor API."""

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host_orch(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        b: pl.Tensor[[128, 128], pl.FP32],
        f: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        with pl.at(level=pl.Level.CHIP, role=pl.Role.Orchestrator):
            with pl.at(level=pl.Level.CORE_GROUP):
                out = pl.add(a, b)
                out_f = pl.assemble(f, out, [0, 0])
        with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
            expected = torch.full((128, 128), 5.0, dtype=torch.float32)
            if not torch.allclose(out_f, expected, rtol=1e-5, atol=1e-5):
                raise AssertionError(
                    f"SubWorker verify failed: expected 5.0, "
                    f"got max={out_f.max().item()}, min={out_f.min().item()}"
                )
        return out_f


class TestL3Dependency:
    """L3 distributed runtime: compile and execute via Worker(level=3)."""

    def test_execute(self, test_config):
        """End-to-end: compile + execute via Worker(level=3), verify f = a + b."""
        compiled = ir.compile(
            L3DependencyProgram,
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
        f = torch.zeros((128, 128), dtype=torch.float32)

        compiled(a, b, f)

        expected = torch.full((128, 128), 5.0, dtype=torch.float32)
        assert torch.allclose(f, expected, rtol=1e-5, atol=1e-5), (
            f"L3 dependency test failed: expected f = a + b = 5.0, "
            f"got max diff = {(f - expected).abs().max().item()}"
        )

    def test_execute_inline(self, test_config):
        """End-to-end: all levels inlined via pl.at(), verify f = a + b."""
        compiled = ir.compile(
            L3DependencyInlineProgram,
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
        f = torch.zeros((128, 128), dtype=torch.float32)

        compiled(a, b, f)

        expected = torch.full((128, 128), 5.0, dtype=torch.float32)
        assert torch.allclose(f, expected, rtol=1e-5, atol=1e-5), (
            f"L3 inline dependency test failed: expected f = a + b = 5.0, "
            f"got max diff = {(f - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
