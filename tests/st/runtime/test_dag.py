# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Tests for DAG (Directed Acyclic Graph) operations using PyPTO frontend.

This test validates complex multi-kernel orchestration with mixed operations,
ensuring correct code generation and execution for DAG-structured computations.

The program definition is imported from examples/models/vector_dag.py
to keep a single source of truth and ensure examples are guarded by tests.
"""

from typing import Any

import pytest
from examples.models.vector_dag import VectorDAGProgram
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec


class VectorDAGTestCase(PTOTestCase):
    """Test case for vector DAG computation.

    Implements the formula: f = (a + b + 1)(a + b + 2) + (a + b)

    Task graph:
      t0: c = kernel_add(a, b)
      t1: d = kernel_add_scalar(c, 1.0)
      t2: e = kernel_add_scalar(c, 2.0)
      t3: g = kernel_mul(d, e)
      t4: f = kernel_add(g, c)
    """

    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "vector_dag_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=2.0),
            TensorSpec("b", [128, 128], DataType.FP32, init_value=3.0),
            TensorSpec("f", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return VectorDAGProgram

    def compute_expected(self, tensors, params=None):
        """Compute expected result: f = (a + b + 1)(a + b + 2) + (a + b)"""
        c = tensors["a"] + tensors["b"]
        d = c + 1.0
        e = c + 2.0
        g = d * e
        tensors["f"][:] = g + c


class TestDAGOperations:
    """Test suite for DAG operations."""

    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_vector_dag(self, test_runner, platform):
        """Test vector DAG computation with 128x128 shape."""
        result = test_runner.run(VectorDAGTestCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
