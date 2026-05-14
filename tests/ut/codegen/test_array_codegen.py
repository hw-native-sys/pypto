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
# Phase-fence carries produce ForStmts with explicit ArrayType iter_args
# (the per-slot TaskId carry that fills N slots of the downstream task's
# ``set_dependencies`` array). The DSL parser does NOT currently promote ``arr`` into a
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
    the next iter's carry value. An ArrayType carry is in-place-update
    semantics, so codegen reuses the ``array.create`` backing array directly:

    * Exactly one C-stack array declaration (the ``array.create`` result) —
      the iter_arg and return_var alias it, no fresh carry array is emitted.
    * No slot-by-slot copy-in / copy-out and no yield self-copy.
    * In-place writes route through the shared array via the body's
      ``array.update_element`` LHS-alias mechanism.
    """
    import re  # noqa: PLC0415

    program, orch_func = _build_array_iter_arg_program(DataType.INT64, 4)
    code = codegen.generate_orchestration(program, orch_func).code

    # Exactly one INT64[4] array is declared — the array.create result.
    decls = re.findall(r"int64_t\s+(\w+)\[4\]", code)
    assert len(decls) == 1, code
    arr = decls[0]

    # The iter_arg/return_var reuse it: no slot-by-slot copy loop is emitted.
    assert "__init_i" not in code, code
    assert "__yield_i" not in code, code

    # Body write lands in-place on the shared array.
    assert f"{arr}[k] = k;" in code, code

    # No "<arr> = <arr>" self-assign from the yield.
    assert f"{arr} = {arr};" not in code, code


def test_for_stmt_with_task_id_array_iter_arg_codegen():
    """ArrayType[TASK_ID, 4] iter_arg — same shape, opaque-handle dtype.

    Phase-fence lowering materialises this exact form. Codegen must emit
    ``PTO2TaskId <name>[4]`` (not a numeric C type) and the in-place
    slot-write pattern.
    """
    import re  # noqa: PLC0415

    program, orch_func = _build_array_iter_arg_program(DataType.TASK_ID, 4)
    code = codegen.generate_orchestration(program, orch_func).code
    # ``array.create`` op codegen must special-case TASK_ID so the
    # declaration uses ``PTO2TaskId``, not the ``unknown`` fallback that
    # ``DataType::TASK_ID.ToCTypeString`` would otherwise return.
    assert re.search(r"PTO2TaskId\s+\w+\[4\]", code), code
    assert "unknown" not in code, code


def test_array_create_task_id_uses_invalid_sentinel():
    """``array.create(N, TASK_ID)`` lowers to a ``PTO2TaskId[N]`` declaration
    plus a per-slot fill with ``PTO2TaskId::invalid()``.

    Critical correctness: ``PTO2TaskId`` is an opaque handle whose
    "invalid" sentinel is NOT bit-zero. Zero-initialising would silently
    mark every slot as a real "task id 0" reference, causing the runtime
    fence to wait on a bogus dep on the first parallel iteration. The
    legacy codegen explicitly broadcast ``PTO2TaskId::invalid()`` over the
    array; this regression test pins the same behaviour for the
    pass-emitted path.
    """
    import re  # noqa: PLC0415

    from pypto.ir.builder import IRBuilder  # noqa: PLC0415
    from pypto.ir.op import array as ir_array  # noqa: PLC0415

    ib = IRBuilder()
    with ib.function("orch", type=ir.FunctionType.Orchestration) as orch_f:
        x = orch_f.param("x", ir.TensorType([16], DataType.INT64))
        orch_f.return_type(ir.TensorType([16], DataType.INT64))
        ib.let("arr", ir_array.create(4, DataType.TASK_ID))
        ib.return_stmt(x)
    orch_func = orch_f.get_result()
    program = ir.Program([orch_func], "test_array_create_task_id", ir.Span.unknown())
    code = codegen.generate_orchestration(program, orch_func).code
    assert re.search(r"PTO2TaskId\s+\w+\[4\];", code), code
    # Per-slot init with the invalid sentinel — NOT ``= {0};`` (which
    # would zero-byte-init, valid for integer dtypes but wrong here).
    assert re.search(r"\w+\[__init_i\]\s*=\s*PTO2TaskId::invalid\(\);", code), code
    assert "unknown" not in code, code


def test_array_create_int_still_uses_zero_init():
    """Non-TASK_ID dtypes keep the compact ``= {0};`` aggregate-init form
    (zero is a valid value for integer / BOOL arrays).
    """
    import re  # noqa: PLC0415

    from pypto.ir.builder import IRBuilder  # noqa: PLC0415
    from pypto.ir.op import array as ir_array  # noqa: PLC0415

    ib = IRBuilder()
    with ib.function("orch", type=ir.FunctionType.Orchestration) as orch_f:
        x = orch_f.param("x", ir.TensorType([16], DataType.INT32))
        orch_f.return_type(ir.TensorType([16], DataType.INT32))
        ib.let("arr", ir_array.create(8, DataType.INT32))
        ib.return_stmt(x)
    orch_func = orch_f.get_result()
    program = ir.Program([orch_func], "test_array_create_int", ir.Span.unknown())
    code = codegen.generate_orchestration(program, orch_func).code
    assert re.search(r"int32_t\s+\w+\[8\]\s*=\s*\{0\};", code), code


def test_array_get_element_task_id_uses_pto2_task_id_type():
    """``array.get_element`` on a TASK_ID array emits a ``PTO2TaskId`` local,
    not the ``unknown`` fallback of ``DataType::ToCTypeString``.
    """
    import re  # noqa: PLC0415

    from pypto.ir.builder import IRBuilder  # noqa: PLC0415
    from pypto.ir.op import array as ir_array  # noqa: PLC0415

    ib = IRBuilder()
    with ib.function("orch", type=ir.FunctionType.Orchestration) as orch_f:
        x = orch_f.param("x", ir.TensorType([16], DataType.INT64))
        orch_f.return_type(ir.TensorType([16], DataType.INT64))
        arr = ib.let("arr", ir_array.create(4, DataType.TASK_ID))
        idx = ir.ConstInt(0, DataType.INT32, ir.Span.unknown())
        ib.let("v", ir_array.get_element(arr, idx))
        ib.return_stmt(x)
    orch_func = orch_f.get_result()
    program = ir.Program([orch_func], "test_array_get_element_task_id", ir.Span.unknown())
    code = codegen.generate_orchestration(program, orch_func).code
    # The local for the get_element result must be ``PTO2TaskId``, not ``unknown``.
    assert re.search(r"PTO2TaskId\s+v\s*=\s*\w+\[", code), code
    assert "unknown" not in code, code


def _build_nested_array_iter_arg_program(
    dtype: DataType, n_outer: int, n_inner: int
) -> tuple[ir.Program, ir.Function]:
    """Build the Phase-B-target shape: outer SEQ x inner PARALLEL, both with ArrayType iter_args.

    The outer iter_arg's init is a freshly allocated array; the *inner* iter_arg's
    init is the outer iter_arg itself. The inner body writes ``task_invalid()`` /
    a loop var into slot ``branch``. The outer yields the inner's rv (an
    ArrayType-typed value).

    Codegen for this shape must:

    * Declare an OUTER carry array distinct from the init (not aliased — each
      ArrayType iter_arg owns fresh storage, so the alias-closure logic that
      treats inner_rv ~= outer_iter_arg for tensor buffers must NOT fire here).
    * Init-copy the outer carry from the init array.
    * At each outer iter, declare the INNER carry and init-copy slot-by-slot
      from the OUTER carry (not from the initial array).
    * At outer yield, slot-by-slot copy the inner carry back into the outer
      carry so state propagates across iterations.
    """
    from pypto.ir.builder import IRBuilder  # noqa: PLC0415
    from pypto.ir.op import array as ir_array  # noqa: PLC0415

    ib = IRBuilder()
    with ib.function("orch", type=ir.FunctionType.Orchestration) as orch_f:
        x = orch_f.param("x", ir.TensorType([16], DataType.INT64))
        orch_f.return_type(ir.TensorType([16], DataType.INT64))
        arr0 = ib.let("arr0", ir_array.create(n_inner, dtype))
        phase = ib.var("phase", ir.ScalarType(DataType.INDEX))
        with ib.for_loop(phase, 0, n_outer, 1, kind=ir.ForKind.Sequential) as outer:
            outer_arr = outer.iter_arg("outer_arr", arr0)
            outer.return_var("outer_arr_final")
            branch = ib.var("branch", ir.ScalarType(DataType.INDEX))
            with ib.for_loop(branch, 0, n_inner, 1, kind=ir.ForKind.Parallel) as inner:
                inner_arr = inner.iter_arg("inner_arr", outer_arr)
                inner.return_var("inner_arr_final")
                if dtype == DataType.TASK_ID:
                    value = ib.let(
                        "tid",
                        ir.create_op_call("system.task_invalid", [], {}, ir.Span.unknown()),
                    )
                else:
                    value = branch
                updated = ib.let("upd", ir_array.update_element(inner_arr, branch, value))
                ib.emit(ir.YieldStmt([updated], ir.Span.unknown()))
            inner_for = inner.get_result()
            inner_rv = inner_for.return_vars[0]
            ib.emit(ir.YieldStmt([inner_rv], ir.Span.unknown()))
        ib.return_stmt(x)
    orch_func = orch_f.get_result()
    program = ir.Program([orch_func], "test_nested_array_iter_arg", ir.Span.unknown())
    return program, orch_func


def test_nested_seq_parallel_task_id_array_carry_codegen():
    """Nested shape: outer SEQ x inner PARALLEL ArrayType[TASK_ID, N] carry.

    An ArrayType carry is in-place-update semantics, so all SSA renames of
    the logical array (the ``array.create`` result, the outer carry, the
    inner carry) collapse onto one C-stack array. Pins: (1) PTO2TaskId, not
    'unknown'; (2) exactly one backing array, declared with the
    ``PTO2TaskId::invalid()`` sentinel; (3) no copy-in / copy-out / yield
    self-copy between distinct arrays.
    """
    import re  # noqa: PLC0415

    n_outer = 3
    n_inner = 4
    program, orch_func = _build_nested_array_iter_arg_program(DataType.TASK_ID, n_outer, n_inner)
    code = codegen.generate_orchestration(program, orch_func).code

    # No fallback "unknown" dtype anywhere.
    assert "unknown" not in code, code

    # Exactly one PTO2TaskId[N] array — the array.create result, reused by
    # both loop carries.
    decls = re.findall(rf"PTO2TaskId\s+(\w+)\[{n_inner}\]", code)
    assert len(decls) == 1, code
    arr = decls[0]
    # ``array.create``'s output must use the invalid sentinel — anything
    # else (notably ``= {0};``) silently produces a "task id 0" reference
    # and breaks the runtime fence.
    assert re.search(rf"{arr}\[__init_i\]\s*=\s*PTO2TaskId::invalid\(\);", code), code

    # No slot-by-slot copy-in / copy-out between distinct arrays — the carries
    # alias the single backing array.
    assert not re.search(r"(\w+)\[__init_i\] = (\w+)\[__init_i\];", code), code
    assert "__yield_i" not in code, code

    # Inner body write lands in-place on the shared array.
    assert re.search(rf"{arr}\[branch\]\s*=\s*tid;", code), code

    # No "<arr> = <arr>;" self-assignment.
    assert f"{arr} = {arr};" not in code, code


def test_nested_seq_parallel_int_array_carry_codegen():
    """Same nested shape with INT64 dtype — the non-TASK_ID branch of
    ``array.create``'s codegen, with the same single-backing-array reuse."""
    import re  # noqa: PLC0415

    program, orch_func = _build_nested_array_iter_arg_program(DataType.INT64, 3, 4)
    code = codegen.generate_orchestration(program, orch_func).code
    # Exactly one INT64[4] array — the array.create result, reused by both
    # loop carries; no copy-in / copy-out loops.
    decls = re.findall(r"int64_t\s+(\w+)\[4\]", code)
    assert len(decls) == 1, code
    arr = decls[0]
    assert "__init_i" not in code, code
    assert "__yield_i" not in code, code
    assert f"{arr}[branch] = branch;" in code, code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
