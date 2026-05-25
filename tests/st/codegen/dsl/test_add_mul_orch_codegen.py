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
from pypto.ir import FunctionType


class TestOrchestrationCodegen:
    """Test suite for orchestration codegen."""

    def test_add_mul_orch_codegen(self):
        """Test orchestration compilation through the pass pipeline.

        Verifies that:
        - JIT entry compiles successfully through the full pass pipeline
        - Post-pass IR has 3 outlined InCore (AIV) functions + 1 Orchestration
        - No exceptions are raised during compilation
        """
        example_orch._cache.clear()
        a = torch.full((16, 16), 2.0, dtype=torch.float32)
        b = torch.full((16, 16), 3.0, dtype=torch.float32)
        output = torch.zeros((16, 16), dtype=torch.float32)

        program = example_orch.compile_for_test(a, b, output)

        # Verify post-pass IR shape: the example_orch entry composes three
        # @pl.jit.incore helpers (kernel_add_16, kernel_add_scalar_16,
        # kernel_mul_16); after OutlineIncoreScopes / pass pipeline the program
        # should hold exactly one Orchestration function plus three on-chip
        # (AIV) functions outlined from the incore scopes.
        assert program is not None, "compile_for_test returned None"
        types = [fn.func_type for fn in program.functions.values()]
        orch_count = sum(1 for t in types if t == FunctionType.Orchestration)
        aiv_count = sum(1 for t in types if t == FunctionType.AIV)
        assert orch_count == 1, f"expected 1 Orchestration function, got {orch_count} (types={types})"
        assert aiv_count == 3, f"expected 3 AIV functions, got {aiv_count} (types={types})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
