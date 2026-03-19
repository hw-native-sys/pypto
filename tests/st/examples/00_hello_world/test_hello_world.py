# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Hello World Example for PyPTO — element-wise tensor addition.

This is the simplest end-to-end PyPTO program:
  1. Load two tiles from global memory into local registers.
  2. Add them element-wise on the AI Vector core.
  3. Store the result back to global memory.

Run:
    pytest tests/st/examples/00_hello_world/hello_world.py -v --forked --platform=a2a3sim
    pytest tests/st/examples/00_hello_world/hello_world.py -v --forked --platform=a2a3 --device=0
"""

from typing import Any

import pytest
from harness.core.harness import DataType, PTOTestCase, TensorSpec

from examples.language.beginner.hello_world import HelloWorldProgram


class HelloWorldAdd(PTOTestCase):
    """Hello World: add two [128, 128] FP32 tensors element-wise.

    Program structure
    -----------------
    InCore function  ``tile_add``
        - Loads tile_a and tile_b from global memory (GM) into registers (UB).
        - Computes tile_c = tile_a + tile_b using the vector unit.
        - Stores tile_c back to the output tensor in GM.

    Orchestration function  ``orchestrator``
        - Calls ``tile_add`` once to process the whole tensor in one shot.
    """

    __test__ = False  # Prevent pytest from collecting this base class directly

    def get_name(self) -> str:
        return "hello_world_add_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=2.0),
            TensorSpec("b", [128, 128], DataType.FP32, init_value=3.0),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return HelloWorldProgram

    def compute_expected(self, tensors, params=None):
        """Expected: c = a + b (element-wise)."""
        tensors["c"][:] = tensors["a"] + tensors["b"]


# =============================================================================
# pytest test functions
# =============================================================================


class TestHelloWorld:
    """Hello World test suite — verifies the simplest PyPTO kernel."""

    def test_hello_world_add(self, test_runner):
        """Compile and run element-wise addition; compare result to torch reference."""
        test_case = HelloWorldAdd()
        result = test_runner.run(test_case)
        assert result.passed, f"Hello world add failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
