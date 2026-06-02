# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""
Runtime test for an InCore kernel that returns ``tuple[Tensor, Scalar]`` and
forwards the Scalar to a downstream InCore kernel (issue #631).

Reproduces the orchestration-codegen scenario from issue #631: the producer
kernel returns a Tensor (writeback to its ``Out`` param) **and** a
``Scalar[INDEX]`` (``off = base * 2``) that is *not* backed by any ``Out``
param. The orchestration unpacks both and forwards the Scalar to the consumer,
which uses it as its valid column count (``valid_shapes=[rows, off]``).

Before the fix, ``OrchestrationStmtCodegen`` mapped each tuple-return position
to an ``Out``/``InOut`` param index; the Scalar element had no such index, so it
was dropped and the generated C++ referenced an undefined variable (or, after a
stop-gap, a wrong ``off = 0``) -> the kernel failed to compile. The fix
re-materializes the Scalar at the call site by inlining the callee's defining
expression with the callee param ``base`` substituted by the call-site arg.

Because the Scalar drives ``valid_shapes``, the result is sensitive to its
*value*: a dropped/zeroed Scalar would zero the whole output and a wrong value
would shift the valid boundary, so the test guards both the original compile
failure and a materialization-value regression.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

# (shape, base): the producer returns the Scalar ``off = base * 2``, which becomes
# the consumer's valid column count. ``base * 2`` must be strictly between 0 and
# ``cols`` so the valid region is partial (discriminates drop / zero / full-region
# bugs). For (16, 16) with base=4 the valid region is the first 8 columns.
_SHAPE_BASE = [((16, 16), 4)]


class ScalarTupleReturnTestCase(PTOTestCase):
    """InCore producer returns ``tuple[Tensor, Scalar]``; the Scalar is forwarded.

    - ``producer`` copies ``a`` into ``mid`` and returns ``(mid, off=base*2)``.
    - ``consumer`` adds ``mid`` and ``b`` using ``valid_shapes=[rows, off]`` and
      writes ``c``; elements outside the valid region stay zero.
    """

    __test__ = False

    def __init__(
        self,
        shape: tuple[int, int],
        base: int,
        *,
        platform: str | None = None,
        config: RunConfig | None = None,
    ):
        super().__init__(config, platform=platform)
        self._rows, self._cols = shape
        self._base = base

    def get_name(self) -> str:
        return f"scalar_tuple_return_{self._rows}x{self._cols}_base{self._base}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self._rows, self._cols], DataType.FP32, init_value=2.0),
            TensorSpec("b", [self._rows, self._cols], DataType.FP32, init_value=3.0),
            TensorSpec(
                "base_t",
                [1],
                DataType.INT64,
                init_value=torch.tensor([self._base], dtype=torch.int64),
            ),
            TensorSpec("mid", [self._rows, self._cols], DataType.FP32, is_output=True),
            TensorSpec("c", [self._rows, self._cols], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        rows = self._rows
        cols = self._cols

        @pl.program
        class ScalarTupleReturnProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def producer(
                self,
                a: pl.Tensor[[rows, cols], pl.FP32],
                base: pl.Scalar[pl.INDEX],
                mid: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            ) -> tuple[pl.Tensor[[rows, cols], pl.FP32], pl.Scalar[pl.INDEX]]:
                """Copy ``a`` into ``mid`` and return it together with ``base * 2``."""
                at = pl.load(a, [0, 0], [rows, cols], target_memory=pl.MemorySpace.Vec)
                rt = pl.store(at, [0, 0], mid)
                off: pl.Scalar[pl.INDEX] = base * 2
                return rt, off

            @pl.function(type=pl.FunctionType.InCore)
            def consumer(
                self,
                mid: pl.Tensor[[rows, cols], pl.FP32],
                b: pl.Tensor[[rows, cols], pl.FP32],
                n: pl.Scalar[pl.INDEX],
                c: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                """Add ``mid`` and ``b`` over the valid region ``[rows, n]``."""
                mt = pl.load(mid, [0, 0], [rows, cols], valid_shapes=[rows, n])
                bt = pl.load(b, [0, 0], [rows, cols], valid_shapes=[rows, n])
                result = pl.add(mt, bt)
                out = pl.store(result, [0, 0], c)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[rows, cols], pl.FP32],
                b: pl.Tensor[[rows, cols], pl.FP32],
                base_t: pl.Tensor[[1], pl.INDEX],
                mid: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
                c: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            ) -> pl.Tensor[[rows, cols], pl.FP32]:
                base: pl.Scalar[pl.INDEX] = pl.tensor.read(base_t, [0])
                # producer returns (mid_written, off=base*2); off has no Out slot
                # and must be re-materialized at the call site.
                t, off = self.producer(a, base, mid)
                c_out = self.consumer(t, b, off, c)
                return c_out

        return ScalarTupleReturnProgram

    def compute_expected(self, tensors, params=None):
        n = self._base * 2
        # producer copied a into mid
        tensors["mid"][:] = tensors["a"]
        # consumer added a + b over the valid [rows, n] region; outside stays zero
        tensors["c"][:, :n] = tensors["a"][:, :n] + tensors["b"][:, :n]


class TestScalarTupleReturn:
    """Pytest suite for issue #631 (Scalar tuple-return forwarding)."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    @pytest.mark.parametrize("shape,base", _SHAPE_BASE)
    def test_scalar_tuple_return(self, test_runner, shape, base, platform):
        """InCore ``tuple[Tensor, Scalar]`` return with the Scalar forwarded downstream."""
        result = test_runner.run(ScalarTupleReturnTestCase(shape, base, platform=platform))
        assert result.passed, f"Test failed for {shape} base={base}: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
