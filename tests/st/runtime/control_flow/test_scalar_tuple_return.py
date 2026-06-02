# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime test for an InCore kernel that returns ``tuple[Tensor, Scalar]`` (issue #631).

The ``producer`` kernel returns a Tensor (writeback to its ``Out`` param) **and**
a ``Scalar[INDEX]`` (``off = base * 2``) that is *not* backed by any ``Out``
param. The orchestration unpacks both and forwards the Scalar to ``consumer``,
which uses it as its valid column count (``valid_shapes=[M, off]``).

Before the fix the orchestration codegen mapped each tuple-return position to an
``Out``/``InOut`` param index; the Scalar element had no such index, so it was
dropped and the generated C++ referenced an undefined variable (or, after a
stop-gap, a wrong ``off = 0``) -> the kernel failed to compile. The fix
re-materializes the Scalar at the call site by inlining the callee's defining
expression with the callee param ``base`` substituted by the call-site arg.

Because the Scalar drives ``valid_shapes``, the result is sensitive to its
*value*: a dropped/zeroed Scalar zeroes the whole output and a wrong value shifts
the valid boundary, so the test guards both the compile failure and a
materialization-value regression. ``base`` is read from a tensor at runtime so
the Scalar stays a genuine runtime value (not a constant-folded literal).
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec

M = 16
N = 16
BASE = 4  # init value of base_t; producer returns off = base * 2
VALID_COLS = BASE * 2  # = 8: the consumer's valid column count (0 < VALID_COLS < N)


@pl.program
class ScalarTupleReturnProgram:
    """producer returns (mid, off=base*2); the orchestration forwards off to consumer."""

    @pl.function(type=pl.FunctionType.InCore)
    def producer(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        base: pl.Scalar[pl.INDEX],
        mid: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> tuple[pl.Tensor[[M, N], pl.FP32], pl.Scalar[pl.INDEX]]:
        """Copy ``a`` into ``mid`` and return it together with the Scalar ``base * 2``."""
        at = pl.load(a, [0, 0], [M, N], target_memory=pl.MemorySpace.Vec)
        rt = pl.store(at, [0, 0], mid)
        off: pl.Scalar[pl.INDEX] = base * 2
        return rt, off

    @pl.function(type=pl.FunctionType.InCore)
    def consumer(
        self,
        mid: pl.Tensor[[M, N], pl.FP32],
        b: pl.Tensor[[M, N], pl.FP32],
        n: pl.Scalar[pl.INDEX],
        c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        """Add ``mid`` and ``b`` over the valid region ``[M, n]``; outside stays zero."""
        mt = pl.load(mid, [0, 0], [M, N], valid_shapes=[M, n])
        bt = pl.load(b, [0, 0], [M, N], valid_shapes=[M, n])
        result = pl.add(mt, bt)
        return pl.store(result, [0, 0], c)

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        b: pl.Tensor[[M, N], pl.FP32],
        base_t: pl.Tensor[[1], pl.INDEX],
        mid: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        c: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        base: pl.Scalar[pl.INDEX] = pl.tensor.read(base_t, [0])
        # producer returns (mid_written, off=base*2); off has no Out slot and must
        # be re-materialized at the call site before it is forwarded to consumer.
        t, off = self.producer(a, base, mid)
        c_out = self.consumer(t, b, off, c)
        return c_out


def _expected(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    expected = torch.zeros((M, N), dtype=torch.float32)
    expected[:, :VALID_COLS] = a[:, :VALID_COLS] + b[:, :VALID_COLS]
    return expected


class ScalarTupleReturnTestCase(PTOTestCase):
    """InCore producer returns ``tuple[Tensor, Scalar]``; the Scalar is forwarded."""

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "scalar_tuple_return"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [M, N], DataType.FP32, init_value=2.0),
            TensorSpec("b", [M, N], DataType.FP32, init_value=3.0),
            TensorSpec("base_t", [1], DataType.INT64, init_value=torch.tensor([BASE], dtype=torch.int64)),
            TensorSpec("mid", [M, N], DataType.FP32, is_output=True),
            TensorSpec("c", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return ScalarTupleReturnProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        tensors["mid"][:] = tensors["a"]
        tensors["c"][:] = _expected(tensors["a"], tensors["b"])


class TestScalarTupleReturn:
    """Test an InCore tuple[Tensor, Scalar] return with the Scalar forwarded downstream (#631)."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_scalar_tuple_return(self, test_runner, platform):
        result = test_runner.run(ScalarTupleReturnTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
