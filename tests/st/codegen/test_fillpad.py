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
Uses partial valid_shape so the padded region is observable.

Note: Only zero mode is supported by the PTO backend (pto.tfillpad).
"""

import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

import pypto.language as pl
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy


class FillpadZeroPartial(PTOTestCase):
    """Fillpad with valid_shape [64,64] < tile shape [128,128].

    Loads a [128,128] tile with only [64,64] valid data from a tensor
    filled with 3.14, then applies fillpad(mode='zero'). The valid
    region keeps 3.14; the padded region is filled with 0.0.

    This is a meaningful test: without fillpad the padded region would
    contain uninitialized/garbage data, not guaranteed zeros. The test
    proves fillpad actually writes zeros to the padded lanes.
    """

    __test__ = False

    def get_name(self) -> str:
        return "fillpad_zero_partial_128x128_valid_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("src", [128, 128], DataType.FP32, init_value=3.14),
            TensorSpec(
                "valid_shape",
                [2],
                DataType.INT64,
                init_value=torch.tensor([64, 64], dtype=torch.int64),
            ),
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
                m: pl.Scalar[pl.INDEX],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                t: pl.Tile[[128, 128], pl.FP32] = pl.load(
                    src, [0, 0], [128, 128], valid_shapes=[m, n]
                )
                t2: pl.Tile[[128, 128], pl.FP32] = pl.fillpad(t, mode="zero")
                return pl.store(t2, [0, 0], dst)

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src: pl.Tensor[[128, 128], pl.FP32],
                vs: pl.Tensor[[2], pl.INDEX],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                dst: pl.Tensor[[128, 128], pl.FP32] = pl.create_tensor(
                    [128, 128], dtype=pl.FP32
                )
                m: pl.Scalar[pl.INDEX] = pl.tensor.read(vs, [0])
                n: pl.Scalar[pl.INDEX] = pl.tensor.read(vs, [1])
                dst = self.kernel(src, dst, m, n)
                return dst

        return Prog

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.PTOAS

    def get_backend_type(self) -> BackendType:
        return BackendType.PTO

    def compute_expected(self, tensors, params=None):
        vr = int(tensors["valid_shape"][0])
        vc = int(tensors["valid_shape"][1])
        # Valid region keeps source data (3.14)
        tensors["dst"][:vr, :vc] = tensors["src"][:vr, :vc]
        # Padded region must be zero — fillpad guarantees this
        tensors["dst"][vr:, :] = 0.0
        tensors["dst"][:vr, vc:] = 0.0


class TestFillpad:
    """Fillpad hardware test suite."""

    def test_fillpad_zero_partial(self, test_runner):
        """Fillpad zero mode with partial valid_shape — verifies actual padding."""
        result = test_runner.run(FillpadZeroPartial())
        assert result.passed, f"Fillpad zero partial failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
