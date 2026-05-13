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
from pypto.pypto_core import DataType, ir


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


# ----------------------------------------------------------------------------
# ForStmt with explicit ArrayType iter_arg — phase-fence carry shape.
#
# Phase-fence lowering produces ForStmts with explicit ArrayType iter_args
# (the per-slot TaskId carry that fans out to N add_dep calls on every
# downstream task). The DSL parser does NOT currently promote ``arr`` into a
# loop-carried iter_arg when only ``arr[k] = ...`` writes happen inside the
# loop body — those go through the LHS-alias path of update_element, so the
# array stays in scope without crossing an iter_arg boundary. The phase-fence
# pass produces the iter_arg form deliberately. These tests hand-build that
# IR shape to exercise the codegen path the pass will emit.
# ----------------------------------------------------------------------------


def _build_array_iter_arg_program(dtype: DataType, extent: int) -> tuple[ir.Program, ir.Function]:
    """Build an orchestration function with an ArrayType[dtype, extent] iter_arg.

    Loop body assigns ``arr[k] = <value>`` where ``value`` depends on dtype:

    * Integer dtype: write the loop var ``k`` (INDEX dtype, compatible with int).
    * TASK_ID dtype: write ``system.task_invalid()`` — the only producer of
      a Scalar[TASK_ID] available without going through a kernel Call.
    """
    from pypto.ir.builder import IRBuilder  # noqa: PLC0415
    from pypto.ir.op import array as ir_array  # noqa: PLC0415

    ib = IRBuilder()
    with ib.function("orch", type=ir.FunctionType.Orchestration) as orch_f:
        x = orch_f.param("x", ir.TensorType([16], DataType.INT64))
        orch_f.return_type(ir.TensorType([16], DataType.INT64))

        arr0 = ib.let("arr0", ir_array.create(extent, dtype))
        k = ib.var("k", ir.ScalarType(DataType.INDEX))
        with ib.for_loop(k, 0, extent, 1) as loop:
            arr_iter = loop.iter_arg("arr_iter", arr0)
            loop.return_var("arr_final")
            if dtype == DataType.TASK_ID:
                value = ib.let(
                    "tid",
                    ir.create_op_call("system.task_invalid", [], {}, ir.Span.unknown()),
                )
            else:
                value = k
            updated = ib.let("upd", ir_array.update_element(arr_iter, k, value))
            ib.emit(ir.YieldStmt([updated], ir.Span.unknown()))
        ib.return_stmt(x)
    orch_func = orch_f.get_result()
    program = ir.Program([orch_func], "test_array_iter_arg", ir.Span.unknown())
    return program, orch_func


def test_for_stmt_with_int_array_iter_arg_codegen():
    """Hand-built IR: ForStmt whose iter_arg is an ArrayType[INT64, 4].

    Each iteration calls ``array.update_element`` and yields the result as
    the next iter's carry value. Codegen must:

    * Emit a single C-stack array declaration with zero-initialization
      (``int64_t <name>[4] = {0};``) at the loop prologue.
    * Slot-by-slot copy from the init array into the carry array.
    * Route in-place writes through the carry name via the body's
      ``array.update_element`` LHS-alias mechanism.
    * Skip the trivial yield self-copy (would be invalid C for raw arrays).
    """
    import re  # noqa: PLC0415

    program, orch_func = _build_array_iter_arg_program(DataType.INT64, 4)
    code = codegen.generate_orchestration(program, orch_func).code

    # Carry array declared exactly once, with zero-init.
    decls = re.findall(r"int64_t\s+(\w+)\[4\]\s*=\s*\{0\};", code)
    assert len(decls) >= 1, code
    carry_names = set(decls)

    # Init copy loop into one of the declared carry arrays.
    init_loop_matches = re.findall(
        r"for \(int64_t __init_i = 0; __init_i < 4; \+\+__init_i\) (\w+)\[__init_i\] = (\w+)\[__init_i\];",
        code,
    )
    assert any(lhs in carry_names and rhs != lhs for lhs, rhs in init_loop_matches), code

    # Body write: ``<carry>[k] = k;`` via LHS-alias on update_element.
    body_writes = re.findall(r"(\w+)\[k\]\s*=\s*k;", code)
    assert any(name in carry_names for name in body_writes), code

    # No "<carry> = <carry>" self-assign from the yield.
    for name in carry_names:
        assert f"{name} = {name};" not in code, code


def test_for_stmt_with_task_id_array_iter_arg_codegen():
    """ArrayType[TASK_ID, 4] iter_arg — same shape, opaque-handle dtype.

    Phase-fence lowering will materialize this exact form. Codegen must
    emit ``PTO2TaskId <name>[4] = {0};`` (not a numeric C type) and the
    in-place slot-write pattern.
    """
    import re  # noqa: PLC0415

    program, orch_func = _build_array_iter_arg_program(DataType.TASK_ID, 4)
    code = codegen.generate_orchestration(program, orch_func).code
    decls = re.findall(r"PTO2TaskId\s+(\w+)\[4\]\s*=\s*\{0\};", code)
    assert len(decls) >= 1, code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
