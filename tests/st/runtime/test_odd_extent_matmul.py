# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end test for issue #1447: a matmul whose N is not a whole number of boxes.

``b`` is ``[K, 17]``. A boxed Mat/Right tile allocates whole inner boxes only, so
17 columns has no legal allocation and the assembler used to reject the
``pto.alloc_tile`` the *compiler itself* synthesized, naming a source line that
holds no tile::

    'pto.alloc_tile' op expects result boxed tile cols to be a multiple of
    innerCols (8), but got 17

``ConvertTensorToTileOps`` now allocates the next whole box on the operand's
declared paddable axis and marks only the natural extent valid, so the bridged
load reads 17 columns into a 32-column box. N is the output width, so the padded
cells land outside the result's valid region and never reach GM.

**Why this input is discriminating.** The padded columns 17..31 are never
written, so they hold whatever the previous kernel left. Every value here is a
small integer exactly representable in FP32 and the reduction is over K=32, so
the golden is exact: any padded lane that leaked into the stored region — or any
misplaced column stride from the wider box — mismatches immediately rather than
hiding under a tolerance.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.ir.pass_manager import OptimizationStrategy

M = 32
K = 32
N = 17  # deliberately not a multiple of the FP32 Mat/Right box granularity (16)


def _make_a() -> torch.Tensor:
    i = torch.arange(M, dtype=torch.float32).reshape(M, 1)
    k = torch.arange(K, dtype=torch.float32).reshape(1, K)
    return ((i + k) % 4.0 - 1.0).contiguous()


def _make_b() -> torch.Tensor:
    k = torch.arange(K, dtype=torch.float32).reshape(K, 1)
    j = torch.arange(N, dtype=torch.float32).reshape(1, N)
    return ((k * j) % 3.0 - 1.0).contiguous()


@pl.program
class OddNMatmulProgram:
    """Plain CORE_GROUP matmul with an odd N. No split: the defect was never
    split-specific, though issue #1447 first surfaced it under LEFT_RIGHT."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            out = pl.matmul(pl.add(a, 0.0), b)
            output = pl.assemble(output, out, [0, 0])
        return output


class OddNMatmulTestCase(PTOTestCase):
    """Issue #1447: N=17 matmul must compile and produce the exact product."""

    def get_name(self) -> str:
        return "odd_extent_matmul_1447"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, K], DataType.FP32, init_value=_make_a),
            TensorSpec("b", [K, N], DataType.FP32, init_value=_make_b),
            TensorSpec("output", [M, N], DataType.FP32, init_value=torch.zeros, is_output=True),
        ]

    def get_program(self) -> Any:
        return OddNMatmulProgram

    def compute_expected(self, tensors, params=None):
        tensors["output"][:, :] = tensors["a"] @ tensors["b"]


class TestOddExtentMatmul:
    """A matmul operand whose N is not a whole number of boxes (issue #1447)."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_odd_n_matmul(self, test_runner, platform):
        result = test_runner.run(OddNMatmulTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
