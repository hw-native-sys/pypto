# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""
Runtime test for an InCore kernel that returns ``tuple[Tensor, Scalar]`` whose
Scalar element is forwarded to a downstream InCore kernel (issue #631).

Reproduces the orchestration-codegen scenario from issue #631: the producer
kernel returns a Tensor (writeback to its ``Out`` param) **and** a
``Scalar[INDEX]`` (``off = base * 2``) that is *not* backed by any ``Out``
param. The orchestration unpacks both and forwards the Scalar to the consumer.

Before the fix, ``OrchestrationStmtCodegen`` mapped each tuple-return position
to an ``Out``/``InOut`` param index; the Scalar element had no such index, so it
was dropped and the generated C++ referenced an undefined variable (or, after a
stop-gap, a wrong ``off = 0``) -> the kernel failed to compile. The fix
re-materializes the Scalar at the call site by inlining the callee's defining
expression with the callee param ``base`` substituted by the call-site arg.

The Scalar drives the consumer's ``valid_shapes`` so the result is sensitive to
the *value* as well: the valid region is ``[16, off]`` where ``off = base * 2``.
A dropped/zeroed Scalar (``off == 0``) would zero the whole output and a wrong
value would shift the valid boundary, so this test guards both the original
compile failure and a materialization-value regression.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

_ROWS = 16
_COLS = 16
# Runtime scalar read into the orchestration; the producer returns base * 2,
# which becomes the consumer's valid column count (4 * 2 = 8, a partial region
# strictly between 0 and _COLS so the test discriminates drop/zero/full bugs).
_BASE = 4


class ScalarTupleReturnTestCase(PTOTestCase):
    """InCore producer returns ``tuple[Tensor, Scalar]``; the Scalar is forwarded.

    - ``producer`` copies ``a`` into ``mid`` and returns ``(mid, off=base*2)``.
    - ``consumer`` adds ``mid`` and ``b`` using ``valid_shapes=[16, off]`` and
      writes ``c``; elements outside the valid region stay zero.
    """

    __test__ = False

    def __init__(self, *, platform: str | None = None, config: RunConfig | None = None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return f"scalar_tuple_return_{_ROWS}x{_COLS}_base{_BASE}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [_ROWS, _COLS], DataType.FP32, init_value=2.0),
            TensorSpec("b", [_ROWS, _COLS], DataType.FP32, init_value=3.0),
            TensorSpec("base_t", [1], DataType.INT64, init_value=torch.tensor([_BASE], dtype=torch.int64)),
            TensorSpec("mid", [_ROWS, _COLS], DataType.FP32, is_output=True),
            TensorSpec("c", [_ROWS, _COLS], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        rows = _ROWS
        cols = _COLS

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
        n = int(tensors["base_t"][0]) * 2
        # producer copied a into mid
        tensors["mid"][:] = tensors["a"]
        # consumer add over the valid [rows, n] region; outside stays zero
        tensors["c"][:, :n] = tensors["a"][:, :n] + tensors["b"][:, :n]


class TestScalarTupleReturn:
    """Pytest suite for issue #631 (Scalar tuple-return forwarding)."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_scalar_tuple_return(self, test_runner, platform):
        """InCore ``tuple[Tensor, Scalar]`` return with the Scalar forwarded downstream."""
        result = test_runner.run(ScalarTupleReturnTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
