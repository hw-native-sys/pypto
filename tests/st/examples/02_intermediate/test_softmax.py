# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Softmax System Tests for PyPTO.

One tile reduction pattern is demonstrated:
  1. Softmax    — exp(x - max(x)) / sum(exp(x - max(x)))
"""

from typing import Any

import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec

from examples.language.intermediate.softmax import TileSoftmaxProgram


class TestTileSoftmax(PTOTestCase):
    """Test row-wise softmax: output[i] = exp(a[i] - max(a[i])) / sum(exp(a[i] - max(a[i])))."""

    __test__ = False

    def get_name(self) -> str:
        return "tile_softmax_64x64"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("output", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return TileSoftmaxProgram

    def compute_expected(self, tensors, params=None):
        tensors["output"][:] = torch.softmax(tensors["a"], dim=1)


class TestReductionOps:
    """Test suite for reduction-based tile ops."""

    def test_tile_softmax(self, test_runner):
        """Test row-wise softmax."""
        result = test_runner.run(TestTileSoftmax())
        assert result.passed, (
            f"tile_softmax failed: {result.error}\n"
            f"  max_abs_error={result.max_abs_error}, max_rel_error={result.max_rel_error}\n"
            f"  mismatches={result.mismatch_count}, sample_indices={result.mismatch_indices}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
