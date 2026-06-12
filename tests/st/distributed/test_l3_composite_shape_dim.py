# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: 2-device composite parameter shape dim (``M * 2``).

Two independent chip tasks run on two devices — ``device=0`` computes
``f = a + b`` and ``device=1`` computes ``g = a - b``. Every InCore / chip-orch
parameter type carries a composite dim ``M * 2`` (a ``Mul(Var, ConstInt)``
expression), exercising the SSA-verifier and PTO-codegen composite-shape-dim
fixes through a 2-device distributed compile + execute.

Whole tensors are passed to each chip task (no per-rank slicing) and outputs are
caller-provided ``Out`` params, so the composite dim is carried solely by the
parameter types. ``rows = 2 * half_rows`` so ``M * 2`` matches the actual tensor
extent. The two outputs (f = a + b, g = a - b) being distinct confirms both
devices ran their own task.
"""

import sys

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

M = pl.dynamic("M")
N = pl.dynamic("N")

NRANKS = 2


def _build_program():
    @pl.program
    class L3CompositeDimProgram:
        """L3: add on device 0, sub on device 1; all params use composite ``M * 2``."""

        @pl.function(type=pl.FunctionType.InCore)
        def tile_add(
            self,
            a: pl.Tensor[[M * 2, N], pl.FP32],
            b: pl.Tensor[[M * 2, N], pl.FP32],
            f: pl.Out[pl.Tensor[[M * 2, N], pl.FP32]],
        ) -> pl.Tensor[[M * 2, N], pl.FP32]:
            tile_f = pl.add(pl.load(a, [0, 0], [128, 128]), pl.load(b, [0, 0], [128, 128]))
            return pl.store(tile_f, [0, 0], f)

        @pl.function(type=pl.FunctionType.InCore)
        def tile_sub(
            self,
            a: pl.Tensor[[M * 2, N], pl.FP32],
            b: pl.Tensor[[M * 2, N], pl.FP32],
            g: pl.Out[pl.Tensor[[M * 2, N], pl.FP32]],
        ) -> pl.Tensor[[M * 2, N], pl.FP32]:
            tile_g = pl.sub(pl.load(a, [0, 0], [128, 128]), pl.load(b, [0, 0], [128, 128]))
            return pl.store(tile_g, [0, 0], g)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_add(
            self,
            a: pl.Tensor[[M * 2, N], pl.FP32],
            b: pl.Tensor[[M * 2, N], pl.FP32],
            f: pl.Out[pl.Tensor[[M * 2, N], pl.FP32]],
        ) -> pl.Tensor[[M * 2, N], pl.FP32]:
            return self.tile_add(a, b, f)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch_sub(
            self,
            a: pl.Tensor[[M * 2, N], pl.FP32],
            b: pl.Tensor[[M * 2, N], pl.FP32],
            g: pl.Out[pl.Tensor[[M * 2, N], pl.FP32]],
        ) -> pl.Tensor[[M * 2, N], pl.FP32]:
            return self.tile_sub(a, b, g)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            a: pl.Tensor[[M * 2, N], pl.FP32],
            b: pl.Tensor[[M * 2, N], pl.FP32],
            f: pl.Out[pl.Tensor[[M * 2, N], pl.FP32]],
            g: pl.Out[pl.Tensor[[M * 2, N], pl.FP32]],
        ) -> tuple[pl.Tensor[[M * 2, N], pl.FP32], pl.Tensor[[M * 2, N], pl.FP32]]:
            out_f = self.chip_orch_add(a, b, f, device=0)
            out_g = self.chip_orch_sub(a, b, g, device=1)
            return out_f, out_g

    return L3CompositeDimProgram


class TestL3CompositeShapeDim:
    """L3 distributed runtime (2 devices): composite parameter shape dim."""

    def test_execute(self, test_config, device_ids):
        """2-device end-to-end: device 0 -> a+b, device 1 -> a-b, with M * 2 dims."""
        if len(device_ids) < NRANKS:
            pytest.skip(f"composite-dim 2-device test needs {NRANKS} devices, got {device_ids}")

        compiled = ir.compile(
            _build_program(),
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:NRANKS],
                num_sub_workers=0,
                block_dim=3,
                aicpu_thread_num=4,
            ),
        )

        a = torch.full((128, 128), 5.0, dtype=torch.float32)
        b = torch.full((128, 128), 3.0, dtype=torch.float32)
        f = torch.zeros((128, 128), dtype=torch.float32)
        g = torch.zeros((128, 128), dtype=torch.float32)

        compiled(a, b, f, g)

        expected_f = torch.full((128, 128), 8.0, dtype=torch.float32)  # a + b
        expected_g = torch.full((128, 128), 2.0, dtype=torch.float32)  # a - b
        assert torch.allclose(f, expected_f, rtol=1e-5, atol=1e-5), (
            f"device-0 add failed: max diff = {(f - expected_f).abs().max().item()}"
        )
        assert torch.allclose(g, expected_g, rtol=1e-5, atol=1e-5), (
            f"device-1 sub failed: max diff = {(g - expected_g).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
