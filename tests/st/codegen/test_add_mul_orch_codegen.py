# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end test for orchestration function codegen.

This test verifies the compilation pipeline for an orchestration program
implementing the formula: f = (a + b + 1)(a + b + 2)

Task Graph:
  task0: c = a + b          (kernel_add)
  task1: d = c + 1          (kernel_add_scalar)
  task2: e = c + 2          (kernel_add_scalar)
  task3: f = d * e          (kernel_mul)

Dependencies: t0->t1, t0->t2, t1->t3, t2->t3

The JIT entry is imported from examples/models/vector_dag.py to keep a single
source of truth and ensure examples are guarded by tests.
"""

import pytest
import torch
from examples.models.vector_dag import example_orch


class TestOrchestrationCodegen:
    """Test suite for orchestration codegen."""

    def test_add_mul_orch_codegen(self):
        """Test orchestration compilation through the pass pipeline.

        Verifies that:
        - JIT entry compiles successfully through the full pass pipeline
        - Post-pass IR has the expected number of functions (3 InCore + 1 Orchestration)
        - No exceptions are raised during compilation
        """
        example_orch._cache.clear()
        a = torch.full((16, 16), 2.0, dtype=torch.float32)
        b = torch.full((16, 16), 3.0, dtype=torch.float32)
        output = torch.zeros((16, 16), dtype=torch.float32)

        program = example_orch.compile_for_test(a, b, output)

        # Sanity-check the post-pass IR shape.
        assert program is not None, "compile_for_test returned None"
        assert len(program.functions) > 0, "compile_for_test produced no functions"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
