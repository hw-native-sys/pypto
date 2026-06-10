# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""L3 distributed st: per-rank child dispatch WITHOUT a distributed-window arg — regression for issue #1708.

A distributed HOST orchestrator submits a per-rank child with rank-sliced tensor
arguments and ``device=r``::

    for r in pl.range(pld.world_size()):
        self.child(x[r], y[r], device=r)

The child here has **no** ``pld.DistributedTensor`` / ``pld.window`` argument — it
is a plain per-rank elementwise ``+ 1``. Before the FlattenCallExpr fix, the
generated ``host_orch.py`` referenced the rank-slice temporaries (``t__tmp_v1``,
``t__tmp_v2``) but never materialized them, so the program compiled yet failed at
runtime with ``KeyError: 't__tmp_v1'`` before the child task could run. Adding an
unused dummy distributed-window argument used to be the only workaround.

Root cause was in ``FlattenCallExpr`` (pass 07): the ForStmt visitor discarded
hoisted temporaries for a single-statement (non-``SeqStmts``) loop body, so the
``x[r]`` / ``y[r]`` rank slices became undefined free vars. The fix routes the
loop body through ``FlattenScopeBody``, which wraps the body together with its
pending hoisted slice assignments into a ``SeqStmts``.

Golden: ``outputs[r] == inputs[r] + 1`` on every rank.

Driven by 2 devices via ``DistributedConfig(device_ids=device_ids[:2], ...)``.
"""

import sys

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto import ir
from pypto.ir.distributed_compiled_program import DistributedConfig

N_RANKS = 2
ROWS = 16
COLS = 32


def _build_rank_slice_program():
    """Build the per-rank ``+ 1`` dispatch program at call time.

    Deferred construction lets this file collect even if the embedded body
    is rejected by the parser.
    """

    @pl.program
    class RankSliceNoDummy:
        @pl.function(type=pl.FunctionType.InCore)
        def add_one(
            self,
            x: pl.Tensor[[ROWS, COLS], pl.FP32],
            y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
        ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
            for row in pl.parallel(ROWS):
                x_row = pl.slice(x, [1, COLS], [row, 0])
                y_row = pl.add(x_row, 1.0)
                y = pl.assemble(y, y_row, [row, 0])
            return y

        @pl.function(type=pl.FunctionType.Orchestration)
        def child(
            self,
            x: pl.Tensor[[ROWS, COLS], pl.FP32],
            y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
        ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
            # No pld.DistributedTensor / pld.window argument — the case that
            # used to drop the rank-slice materialization (issue #1708).
            y = self.add_one(x, y)
            return y

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            x: pl.Tensor[[N_RANKS, ROWS, COLS], pl.FP32],
            y: pl.Out[pl.Tensor[[N_RANKS, ROWS, COLS], pl.FP32]],
        ):
            # Single-statement loop body: the bare per-rank child dispatch whose
            # x[r] / y[r] rank slices must be materialized before TaskArgs.
            for r in pl.range(pld.world_size()):
                self.child(x[r], y[r], device=r)

    return RankSliceNoDummy


class TestL3RankSliceDispatch:
    """L3 distributed runtime: per-rank child dispatch with no distributed-window arg."""

    def test_rank_slice_no_dummy(self, test_config, device_ids):
        if len(device_ids) < N_RANKS:
            pytest.skip(f"rank-slice dispatch needs {N_RANKS} devices, got {device_ids}")

        program = _build_rank_slice_program()
        compiled = ir.compile(
            program,
            platform=test_config.platform,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:N_RANKS],
                num_sub_workers=0,
            ),
        )

        inputs = torch.randn((N_RANKS, ROWS, COLS), dtype=torch.float32)
        outputs = torch.zeros((N_RANKS, ROWS, COLS), dtype=torch.float32)

        compiled(inputs, outputs)

        expected = inputs + 1.0
        assert torch.allclose(outputs, expected, rtol=1e-5, atol=1e-5), (
            f"rank-slice dispatch mismatch: max diff = {(outputs - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", *sys.argv[1:]])
