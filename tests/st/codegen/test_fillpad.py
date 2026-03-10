# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Fillpad operation hardware test.

Tests pl.fillpad with zero pad mode on actual hardware.
When valid_shape == tile shape, fillpad is a copy operation.

Note: Only zero mode is supported by the PTO backend (pto.tfillpad).
Modes max/min are not supported by the ptoas assembler for tfillpad.
"""

import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

import pypto.language as pl
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy


class FillpadZero(PTOTestCase):
    """Fillpad with zero mode: load tile, fillpad(mode='zero'), store back."""

    __test__ = False

    def get_name(self) -> str:
        return "fillpad_zero_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src", [128, 128], DataType.FP32, init_value=3.14),
            TensorSpec("dst", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[128, 128], pl.FP32],
                dst: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(src, [0, 0], [128, 128])
                t2: pl.Tile[[128, 128], pl.FP32] = pl.fillpad(t, mode="zero")
                return pl.store(t2, [0, 0], dst)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, src: pl.Tensor[[128, 128], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                dst: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor(
                    [128, 128], dtype=pl.FP32
                )
                dst = self.kernel(src, dst)
                return dst

        return Prog

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.PTOAS

    def get_backend_type(self) -> BackendType:
        return BackendType.PTO

    def compute_expected(self, tensors, params=None):
        # With full tile (valid_shape == tile shape), fillpad is a copy
        tensors["dst"][:] = tensors["src"]


class TestFillpad:
    """Fillpad hardware test suite — verifies fillpad with zero pad mode."""

    def test_fillpad_zero(self, test_runner):
        """Fillpad with zero pad mode on hardware."""
        result = test_runner.run(FillpadZero())
        assert result.passed, f"Fillpad zero failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
