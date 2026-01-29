# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for VerifySSAPass."""

from pypto import ir
from pypto.ir import builder
from pypto.pypto_core import DataType, passes


def test_verify_ssa_valid():
    """Test VerifySSAPass with valid SSA IR."""
    ib = builder.IRBuilder()

    with ib.function("test_valid_ssa") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        b = f.param("b", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        # Each variable assigned only once
        x = ib.let("x", a)
        _y = ib.let("y", b)  # Intentionally unused
        z = ib.let("z", x)

        ib.return_stmt(z)

    func = f.get_result()

    # Run verification
    verify_pass = passes.VerifySSAPass()
    verify_pass.run(func)

    # Check no errors
    assert not verify_pass.has_errors(), f"Expected no errors, but got:\n{verify_pass.get_report()}"
    report = verify_pass.get_report()
    assert "PASSED" in report
    print("✓ Valid SSA test passed")


def test_verify_ssa_multiple_assignment():
    """Test VerifySSAPass detects multiple assignments."""
    ib = builder.IRBuilder()

    with ib.function("test_multiple_assignment") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        # First assignment (intentionally unused)
        _x = ib.let("x", a)
        # Second assignment to same variable name (violates SSA)
        x2 = ib.let("x", a)

        ib.return_stmt(x2)

    func = f.get_result()

    # Run verification
    verify_pass = passes.VerifySSAPass()
    verify_pass.run(func)

    # Check for errors
    assert verify_pass.has_errors(), "Expected MULTIPLE_ASSIGNMENT error"
    errors = verify_pass.get_errors()
    assert len(errors) > 0
    assert any(e.type == passes.SSAErrorType.MULTIPLE_ASSIGNMENT for e in errors)
    assert "x" in errors[0].message
    assert "2 times" in errors[0].message
    report = verify_pass.get_report()
    assert "FAILED" in report
    assert "MULTIPLE_ASSIGNMENT" in report
    print("✓ Multiple assignment test passed")


def test_verify_ssa_multiple_assignment_three_times():
    """Test VerifySSAPass detects three assignments to same variable."""
    ib = builder.IRBuilder()

    with ib.function("test_three_assignments") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        # Three assignments to 'y' (intentionally unused)
        _y1 = ib.let("y", a)
        _y2 = ib.let("y", a)
        y3 = ib.let("y", a)

        ib.return_stmt(y3)

    func = f.get_result()

    verify_pass = passes.VerifySSAPass()
    verify_pass.run(func)

    assert verify_pass.has_errors()
    errors = verify_pass.get_errors()
    # Should have 2 errors (2nd and 3rd assignments)
    assert len(errors) >= 2
    assert all(e.type == passes.SSAErrorType.MULTIPLE_ASSIGNMENT for e in errors)
    print("✓ Three assignments test passed")


def test_verify_ssa_name_shadowing_in_for():
    """Test VerifySSAPass detects name shadowing in ForStmt."""
    ib = builder.IRBuilder()

    with ib.function("test_shadow_for") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        # Declare outer variable 'i'
        outer_i = ib.let("i", a)

        # Create ForStmt with loop var also named 'i' (shadows outer 'i')
        loop_var = ib.var("i", ir.ScalarType(DataType.INT64))
        with ib.for_loop(loop_var, 0, 5, 1):
            # Simple body (intentionally unused)
            _tmp = ib.let("tmp", loop_var)

        ib.return_stmt(outer_i)

    func = f.get_result()

    # Run verification
    verify_pass = passes.VerifySSAPass()
    verify_pass.run(func)

    assert verify_pass.has_errors(), "Expected NAME_SHADOWING error"
    errors = verify_pass.get_errors()
    assert any(e.type == passes.SSAErrorType.NAME_SHADOWING for e in errors), (
        f"Expected NAME_SHADOWING error, got {[e.type for e in errors]}"
    )
    report = verify_pass.get_report()
    assert "NAME_SHADOWING" in report
    print("✓ Name shadowing in ForStmt test passed")


def test_verify_ssa_name_shadowing_iter_arg():
    """Test VerifySSAPass detects name shadowing with iter_arg."""
    span = ir.Span.unknown()

    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

    # Declare outer variable 'acc'
    acc_outer = ir.Var("acc", ir.ScalarType(DataType.INT64), span)
    assign_acc = ir.AssignStmt(acc_outer, a, span)

    # Create ForStmt with iter_arg named 'acc' (shadows outer 'acc')
    loop_var = ir.Var("j", ir.ScalarType(DataType.INT64), span)
    iter_arg = ir.IterArg("acc", ir.ScalarType(DataType.INT64), acc_outer, span)
    body = ir.YieldStmt([iter_arg], span)
    result_var = ir.Var("result", ir.ScalarType(DataType.INT64), span)

    for_stmt = ir.ForStmt(
        loop_var,
        ir.ConstInt(0, DataType.INT64, span),
        ir.ConstInt(5, DataType.INT64, span),
        ir.ConstInt(1, DataType.INT64, span),
        [iter_arg],
        body,
        [result_var],
        span,
    )

    func_body = ir.SeqStmts([assign_acc, for_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_shadow_iter", params, return_types, func_body, span)

    # Run verification
    verify_pass = passes.VerifySSAPass()
    verify_pass.run(func)

    assert verify_pass.has_errors(), "Expected NAME_SHADOWING error"
    errors = verify_pass.get_errors()
    assert any(e.type == passes.SSAErrorType.NAME_SHADOWING for e in errors)
    print("✓ Name shadowing with iter_arg test passed")


def test_verify_ssa_for_missing_yield():
    """Test VerifySSAPass detects missing yield in ForStmt with iter_args."""
    span = ir.Span.unknown()

    # Create a simple function
    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

    # Create ForStmt with iter_arg but no yield in body
    loop_var = ir.Var("i", ir.ScalarType(DataType.INT64), span)
    iter_arg = ir.IterArg("sum", ir.ScalarType(DataType.INT64), a, span)
    # Body WITHOUT yield - this violates SSA
    body = ir.AssignStmt(ir.Var("dummy", ir.ScalarType(DataType.INT64), span), loop_var, span)
    result_var = ir.Var("result", ir.ScalarType(DataType.INT64), span)

    for_stmt = ir.ForStmt(
        loop_var,
        ir.ConstInt(0, DataType.INT64, span),
        ir.ConstInt(10, DataType.INT64, span),
        ir.ConstInt(1, DataType.INT64, span),
        [iter_arg],  # Has iter_arg
        body,  # But no yield!
        [result_var],
        span,
    )

    func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_missing_yield", params, return_types, func_body, span)

    # Run verification
    verify_pass = passes.VerifySSAPass()
    verify_pass.run(func)

    assert verify_pass.has_errors(), "Expected MISSING_YIELD error"
    errors = verify_pass.get_errors()
    assert any(e.type == passes.SSAErrorType.MISSING_YIELD for e in errors)
    assert "YieldStmt" in errors[0].message
    print("✓ ForStmt missing yield test passed")


def test_verify_ssa_for_type_mismatch():
    """Test VerifySSAPass detects type mismatch in ForStmt."""
    span = ir.Span.unknown()

    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

    # Create ForStmt with type mismatch
    loop_var = ir.Var("i", ir.ScalarType(DataType.INT64), span)
    # iter_arg with INT64 type
    iter_arg = ir.IterArg("sum", ir.ScalarType(DataType.INT64), a, span)
    # Yield with FP32 type (MISMATCH!)
    yield_value = ir.ConstFloat(1.0, DataType.FP32, span)
    body = ir.YieldStmt([yield_value], span)
    # return_var with INT64 type
    result_var = ir.Var("result", ir.ScalarType(DataType.INT64), span)

    for_stmt = ir.ForStmt(
        loop_var,
        ir.ConstInt(0, DataType.INT64, span),
        ir.ConstInt(10, DataType.INT64, span),
        ir.ConstInt(1, DataType.INT64, span),
        [iter_arg],
        body,
        [result_var],
        span,
    )

    func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_type_mismatch", params, return_types, func_body, span)

    # Run verification
    verify_pass = passes.VerifySSAPass()
    verify_pass.run(func)

    assert verify_pass.has_errors(), "Expected CONTROL_FLOW_TYPE_MISMATCH error"
    errors = verify_pass.get_errors()
    assert any(e.type == passes.SSAErrorType.CONTROL_FLOW_TYPE_MISMATCH for e in errors)
    # Should detect multiple mismatches
    assert len(errors) >= 1
    print("✓ ForStmt type mismatch test passed")


def test_verify_ssa_if_missing_yield():
    """Test VerifySSAPass detects missing yield in IfStmt."""
    span = ir.Span.unknown()

    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

    # Create IfStmt with return_vars but no yield in branches
    condition = ir.Gt(a, ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span)
    # then_body WITHOUT yield
    then_body = ir.AssignStmt(ir.Var("tmp1", ir.ScalarType(DataType.INT64), span), a, span)
    # else_body WITHOUT yield
    else_body = ir.AssignStmt(ir.Var("tmp2", ir.ScalarType(DataType.INT64), span), a, span)
    result_var = ir.Var("result", ir.ScalarType(DataType.INT64), span)

    if_stmt = ir.IfStmt(condition, then_body, else_body, [result_var], span)

    func_body = ir.SeqStmts([if_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_if_no_yield", params, return_types, func_body, span)

    # Run verification
    verify_pass = passes.VerifySSAPass()
    verify_pass.run(func)

    assert verify_pass.has_errors(), "Expected MISSING_YIELD error"
    errors = verify_pass.get_errors()
    assert any(e.type == passes.SSAErrorType.MISSING_YIELD for e in errors)
    # Should have errors for both then and else branches
    assert len(errors) >= 2
    print("✓ IfStmt missing yield test passed")


def test_verify_ssa_if_missing_else():
    """Test VerifySSAPass detects missing else branch when return_vars exist."""
    span = ir.Span.unknown()

    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

    condition = ir.Gt(a, ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span)
    then_body = ir.YieldStmt([a], span)
    result_var = ir.Var("result", ir.ScalarType(DataType.INT64), span)

    # IfStmt with return_vars but NO else branch
    if_stmt = ir.IfStmt(condition, then_body, None, [result_var], span)

    func_body = ir.SeqStmts([if_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_if_no_else", params, return_types, func_body, span)

    # Run verification
    verify_pass = passes.VerifySSAPass()
    verify_pass.run(func)

    assert verify_pass.has_errors(), "Expected MISSING_YIELD error for missing else"
    errors = verify_pass.get_errors()
    assert any(e.type == passes.SSAErrorType.MISSING_YIELD for e in errors)
    assert "else" in errors[0].message.lower()
    print("✓ IfStmt missing else branch test passed")


def test_verify_ssa_if_type_mismatch():
    """Test VerifySSAPass detects type mismatch between then and else yields."""
    span = ir.Span.unknown()

    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

    condition = ir.Gt(a, ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span)
    # then yields INT64
    then_body = ir.YieldStmt([ir.ConstInt(1, DataType.INT64, span)], span)
    # else yields FP32 (MISMATCH!)
    else_body = ir.YieldStmt([ir.ConstFloat(2.0, DataType.FP32, span)], span)
    result_var = ir.Var("result", ir.ScalarType(DataType.INT64), span)

    if_stmt = ir.IfStmt(condition, then_body, else_body, [result_var], span)

    func_body = ir.SeqStmts([if_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_if_type_mismatch", params, return_types, func_body, span)

    # Run verification
    verify_pass = passes.VerifySSAPass()
    verify_pass.run(func)

    assert verify_pass.has_errors(), "Expected CONTROL_FLOW_TYPE_MISMATCH error"
    errors = verify_pass.get_errors()
    assert any(e.type == passes.SSAErrorType.CONTROL_FLOW_TYPE_MISMATCH for e in errors)
    assert "then" in errors[0].message.lower() or "else" in errors[0].message.lower()
    print("✓ IfStmt type mismatch test passed")


def test_verify_ssa_multiple_errors():
    """Test VerifySSAPass collects multiple errors in one pass."""
    ib = builder.IRBuilder()

    with ib.function("test_multi_errors") as f:
        a = f.param("a", ir.ScalarType(DataType.INT64))
        f.return_type(ir.ScalarType(DataType.INT64))

        # Error 1: Multiple assignment (intentionally unused)
        _x = ib.let("x", a)
        x2 = ib.let("x", a)

        # Error 2: Another multiple assignment (intentionally unused)
        _y = ib.let("y", a)
        _y2 = ib.let("y", a)

        ib.return_stmt(x2)

    func = f.get_result()

    # Run verification
    verify_pass = passes.VerifySSAPass()
    verify_pass.run(func)

    assert verify_pass.has_errors()
    errors = verify_pass.get_errors()
    # Should have at least 2 errors (one for x, one for y)
    assert len(errors) >= 2
    assert all(e.type == passes.SSAErrorType.MULTIPLE_ASSIGNMENT for e in errors)

    report = verify_pass.get_report()
    assert "FAILED" in report
    assert str(len(errors)) in report
    print(f"✓ Multiple errors test passed (collected {len(errors)} errors)")


def test_verify_ssa_valid_for_with_iter_args():
    """Test valid ForStmt with iter_args passes verification."""
    span = ir.Span.unknown()

    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

    # Valid ForStmt with matching types
    loop_var = ir.Var("i", ir.ScalarType(DataType.INT64), span)
    iter_arg = ir.IterArg("sum", ir.ScalarType(DataType.INT64), a, span)
    # Yield with same type as iter_arg
    yield_value = ir.Add(iter_arg, loop_var, DataType.INT64, span)
    body = ir.YieldStmt([yield_value], span)
    result_var = ir.Var("result", ir.ScalarType(DataType.INT64), span)

    for_stmt = ir.ForStmt(
        loop_var,
        ir.ConstInt(0, DataType.INT64, span),
        ir.ConstInt(10, DataType.INT64, span),
        ir.ConstInt(1, DataType.INT64, span),
        [iter_arg],
        body,
        [result_var],
        span,
    )

    func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_valid_for", params, return_types, func_body, span)

    # Run verification
    verify_pass = passes.VerifySSAPass()
    verify_pass.run(func)

    assert not verify_pass.has_errors(), f"Expected no errors, but got:\n{verify_pass.get_report()}"
    assert "PASSED" in verify_pass.get_report()
    print("✓ Valid ForStmt with iter_args test passed")


def test_verify_ssa_valid_if_with_yields():
    """Test valid IfStmt with matching yields passes verification."""
    span = ir.Span.unknown()

    a = ir.Var("a", ir.ScalarType(DataType.INT64), span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [ir.ScalarType(DataType.INT64)]

    condition = ir.Gt(a, ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span)
    # Both branches yield INT64
    then_body = ir.YieldStmt([ir.ConstInt(1, DataType.INT64, span)], span)
    else_body = ir.YieldStmt([ir.ConstInt(2, DataType.INT64, span)], span)
    result_var = ir.Var("result", ir.ScalarType(DataType.INT64), span)

    if_stmt = ir.IfStmt(condition, then_body, else_body, [result_var], span)

    func_body = ir.SeqStmts([if_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_valid_if", params, return_types, func_body, span)

    # Run verification
    verify_pass = passes.VerifySSAPass()
    verify_pass.run(func)

    assert not verify_pass.has_errors(), f"Expected no errors, but got:\n{verify_pass.get_report()}"
    assert "PASSED" in verify_pass.get_report()
    print("✓ Valid IfStmt with yields test passed")


def test_verify_ssa_tensor_shape_mismatch():
    """Test VerifySSAPass detects shape mismatch in TensorType."""
    span = ir.Span.unknown()

    # Create TensorType with different shapes
    shape1 = [ir.ConstInt(10, DataType.INT64, span), ir.ConstInt(20, DataType.INT64, span)]
    shape2 = [ir.ConstInt(10, DataType.INT64, span), ir.ConstInt(30, DataType.INT64, span)]  # Different!

    tensor_type1 = ir.TensorType(shape1, DataType.FP32)
    tensor_type2 = ir.TensorType(shape2, DataType.FP32)

    a = ir.Var("a", tensor_type1, span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [tensor_type1]

    # Create ForStmt with shape mismatch
    loop_var = ir.Var("i", ir.ScalarType(DataType.INT64), span)
    iter_arg = ir.IterArg("tensor", tensor_type1, a, span)
    # Yield with different shape
    yield_value = ir.Var("temp", tensor_type2, span)
    body = ir.YieldStmt([yield_value], span)
    result_var = ir.Var("result", tensor_type1, span)

    for_stmt = ir.ForStmt(
        loop_var,
        ir.ConstInt(0, DataType.INT64, span),
        ir.ConstInt(10, DataType.INT64, span),
        ir.ConstInt(1, DataType.INT64, span),
        [iter_arg],
        body,
        [result_var],
        span,
    )

    func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_shape_mismatch", params, return_types, func_body, span)

    # Run verification
    verify_pass = passes.VerifySSAPass()
    verify_pass.run(func)

    assert verify_pass.has_errors(), "Expected CONTROL_FLOW_TYPE_MISMATCH error for shape"
    errors = verify_pass.get_errors()
    assert any(e.type == passes.SSAErrorType.CONTROL_FLOW_TYPE_MISMATCH for e in errors)
    report = verify_pass.get_report()
    assert "Shape mismatch" in report or "dimension" in report.lower()
    print("✓ TensorType shape mismatch test passed")


def test_verify_ssa_tensor_shape_dimension_count_mismatch():
    """Test VerifySSAPass detects dimension count mismatch."""
    span = ir.Span.unknown()

    # Create TensorType with different number of dimensions
    shape1 = [ir.ConstInt(10, DataType.INT64, span), ir.ConstInt(20, DataType.INT64, span)]
    shape2 = [ir.ConstInt(10, DataType.INT64, span)]  # Only 1 dimension!

    tensor_type1 = ir.TensorType(shape1, DataType.FP32)
    tensor_type2 = ir.TensorType(shape2, DataType.FP32)

    a = ir.Var("a", tensor_type1, span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [tensor_type1]

    condition = ir.Gt(
        ir.ConstInt(1, DataType.INT64, span), ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span
    )
    then_body = ir.YieldStmt([ir.Var("t1", tensor_type1, span)], span)
    else_body = ir.YieldStmt([ir.Var("t2", tensor_type2, span)], span)  # Different dimensions!
    result_var = ir.Var("result", tensor_type1, span)

    if_stmt = ir.IfStmt(condition, then_body, else_body, [result_var], span)

    func_body = ir.SeqStmts([if_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_dim_mismatch", params, return_types, func_body, span)

    # Run verification
    verify_pass = passes.VerifySSAPass()
    verify_pass.run(func)

    assert verify_pass.has_errors(), "Expected CONTROL_FLOW_TYPE_MISMATCH error for dimension count"
    errors = verify_pass.get_errors()
    assert any(e.type == passes.SSAErrorType.CONTROL_FLOW_TYPE_MISMATCH for e in errors)
    report = verify_pass.get_report()
    assert "dimensions" in report.lower()
    print("✓ TensorType dimension count mismatch test passed")


def test_verify_ssa_tile_shape_mismatch():
    """Test VerifySSAPass detects shape mismatch in TileType."""
    span = ir.Span.unknown()

    # Create TileType with different shapes
    shape1 = [ir.ConstInt(16, DataType.INT64, span), ir.ConstInt(16, DataType.INT64, span)]
    shape2 = [ir.ConstInt(16, DataType.INT64, span), ir.ConstInt(32, DataType.INT64, span)]  # Different!

    tile_type1 = ir.TileType(shape1, DataType.FP16)
    tile_type2 = ir.TileType(shape2, DataType.FP16)

    a = ir.Var("a", tile_type1, span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [tile_type1]

    # Create IfStmt with shape mismatch
    condition = ir.Gt(
        ir.ConstInt(1, DataType.INT64, span), ir.ConstInt(0, DataType.INT64, span), DataType.BOOL, span
    )
    then_body = ir.YieldStmt([ir.Var("tile1", tile_type1, span)], span)
    else_body = ir.YieldStmt([ir.Var("tile2", tile_type2, span)], span)  # Different shape!
    result_var = ir.Var("result", tile_type1, span)

    if_stmt = ir.IfStmt(condition, then_body, else_body, [result_var], span)

    func_body = ir.SeqStmts([if_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_tile_shape_mismatch", params, return_types, func_body, span)

    # Run verification
    verify_pass = passes.VerifySSAPass()
    verify_pass.run(func)

    assert verify_pass.has_errors(), "Expected CONTROL_FLOW_TYPE_MISMATCH error for tile shape"
    errors = verify_pass.get_errors()
    assert any(e.type == passes.SSAErrorType.CONTROL_FLOW_TYPE_MISMATCH for e in errors)
    report = verify_pass.get_report()
    assert "Shape mismatch" in report or "dimension" in report.lower()
    print("✓ TileType shape mismatch test passed")


def test_verify_ssa_valid_tensor_same_shape():
    """Test valid TensorType with same shape passes verification."""
    span = ir.Span.unknown()

    # Create TensorType with same shape
    shape = [ir.ConstInt(10, DataType.INT64, span), ir.ConstInt(20, DataType.INT64, span)]

    tensor_type = ir.TensorType(shape, DataType.FP32)

    a = ir.Var("a", tensor_type, span)
    params: list[ir.Var] = [a]
    return_types: list[ir.Type] = [tensor_type]

    # Create ForStmt with matching shape
    loop_var = ir.Var("i", ir.ScalarType(DataType.INT64), span)
    iter_arg = ir.IterArg("tensor", tensor_type, a, span)
    yield_value = ir.Var("temp", tensor_type, span)
    body = ir.YieldStmt([yield_value], span)
    result_var = ir.Var("result", tensor_type, span)

    for_stmt = ir.ForStmt(
        loop_var,
        ir.ConstInt(0, DataType.INT64, span),
        ir.ConstInt(10, DataType.INT64, span),
        ir.ConstInt(1, DataType.INT64, span),
        [iter_arg],
        body,
        [result_var],
        span,
    )

    func_body = ir.SeqStmts([for_stmt, ir.ReturnStmt([result_var], span)], span)
    func = ir.Function("test_valid_tensor_shape", params, return_types, func_body, span)

    # Run verification
    verify_pass = passes.VerifySSAPass()
    verify_pass.run(func)

    assert not verify_pass.has_errors(), f"Expected no errors, but got:\n{verify_pass.get_report()}"
    assert "PASSED" in verify_pass.get_report()
    print("✓ Valid TensorType with same shape test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Running SSA Verification Pass Test Suite")
    print("=" * 60)

    test_verify_ssa_valid()
    test_verify_ssa_multiple_assignment()
    test_verify_ssa_multiple_assignment_three_times()
    test_verify_ssa_name_shadowing_in_for()
    test_verify_ssa_name_shadowing_iter_arg()
    test_verify_ssa_for_missing_yield()
    test_verify_ssa_for_type_mismatch()
    test_verify_ssa_if_missing_yield()
    test_verify_ssa_if_missing_else()
    test_verify_ssa_if_type_mismatch()
    test_verify_ssa_multiple_errors()
    test_verify_ssa_valid_for_with_iter_args()
    test_verify_ssa_valid_if_with_yields()
    test_verify_ssa_tensor_shape_mismatch()
    test_verify_ssa_tensor_shape_dimension_count_mismatch()
    test_verify_ssa_tile_shape_mismatch()
    test_verify_ssa_valid_tensor_same_shape()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
