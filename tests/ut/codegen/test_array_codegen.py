# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Codegen tests for ArrayType operations.

Verifies that ``array.create`` / ``array.get_element`` / ``array.update_element``
lower to bare C stack arrays (``dtype name[N]``, no STL dependency — the device
CPU codegen does not pull ``<array>``), and that the SSA-functional
update_element correctly aliases the LHS to the input array so in-place
mutations land on the same backing storage.
"""

import pypto.language as pl
import pytest
from pypto import codegen, passes
from pypto.pypto_core import ir


def _generate_orch(src: str) -> str:
    """Parse a program, derive call directions, and codegen the orchestration func."""
    prog = pl.parse_program(src)
    prog = passes.derive_call_directions()(prog)
    for func in prog.functions.values():
        if func.func_type == ir.FunctionType.Orchestration:
            return codegen.generate_orchestration(prog, func).code
    raise AssertionError("no Orchestration function found in program")


def test_array_create_emits_std_array_declaration():
    src = """
@pl.program
class P:
    @pl.function(type=pl.FunctionType.Orchestration)
    def k(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        arr = pl.array.create(8, pl.INT32)
        return x
"""
    code = _generate_orch(src)
    # Bare C array, not std::array — device CPU codegen does not pull in STL.
    assert "#include <array>" not in code
    assert "int32_t arr[8] = {0};" in code


def test_array_write_read_with_constant_index():
    src = """
@pl.program
class P:
    @pl.function(type=pl.FunctionType.Orchestration)
    def k(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        arr = pl.array.create(8, pl.INT32)
        arr[0] = 7
        arr[3] = 42
        v0 = arr[0]
        v1 = arr[3]
        return x
"""
    code = _generate_orch(src)
    # Update_element + alias -> in-place writes on the same `arr`
    assert "arr[0] = 7;" in code
    assert "arr[3] = 42;" in code
    # get_element -> scalar reads
    assert "int32_t v0 = arr[0];" in code
    assert "int32_t v1 = arr[3];" in code


def test_array_write_with_dynamic_scalar_index():
    """Writes/reads driven by a runtime scalar index must emit ``arr[i]``."""
    src = """
@pl.program
class P:
    @pl.function(type=pl.FunctionType.Orchestration)
    def k(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        arr = pl.array.create(4, pl.INT32)
        i: pl.Scalar[pl.INT32] = 1
        arr[i] = 99
        v = arr[i]
        return x
"""
    code = _generate_orch(src)
    assert "int32_t arr[4] = {0};" in code
    # Update_element with dynamic index
    assert "arr[i] = 99;" in code
    # get_element with dynamic index
    assert "int32_t v = arr[i];" in code


def test_array_sequential_writes_share_backing_storage():
    """Multiple update_element calls must all target the same C variable (no copies)."""
    src = """
@pl.program
class P:
    @pl.function(type=pl.FunctionType.Orchestration)
    def k(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        arr = pl.array.create(4, pl.INT32)
        arr[0] = 10
        arr[1] = 20
        arr[2] = 30
        arr[3] = 40
        return x
"""
    code = _generate_orch(src)
    # Exactly one array declaration — all writes alias back to it.
    assert code.count("int32_t arr[4]") == 1
    for i, v in [(0, 10), (1, 20), (2, 30), (3, 40)]:
        assert f"arr[{i}] = {v};" in code


def test_array_codegen_in_for_loop():
    """Array reads/writes inside a for-loop. The array dtype is INT64 to match
    ``pl.range``'s INDEX loop variable — like ``tensor.write``, ``array.update_element``
    requires exact dtype match between the value and the array element type.
    """
    src = """
@pl.program
class P:
    @pl.function(type=pl.FunctionType.Orchestration)
    def k(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        arr = pl.array.create(4, pl.INT64)
        for i in pl.range(4):
            arr[i] = i
        return x
"""
    code = _generate_orch(src)
    assert "int64_t arr[4] = {0};" in code
    # for-loop body must contain the update_element write to arr[i]
    assert "arr[i] = i;" in code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
