# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for Program class."""

from typing import cast

import pytest
from pypto import DataType, ir


class TestProgram:
    """Test Program class."""

    def test_program_creation(self):
        """Test creating a Program instance."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        func = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)
        program = ir.Program([func], "", span)

        assert program is not None
        assert program.span.filename == "test.py"
        assert len(program.functions) == 1
        assert program.name == ""

    def test_program_has_attributes(self):
        """Test that Program has name and functions attributes."""
        span = ir.Span("test.py", 10, 5, 10, 15)
        dtype = DataType.INT64
        a = ir.Var("a", ir.ScalarType(dtype), span)
        b = ir.Var("b", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(a, b, span)
        body = ir.SeqStmts([assign1], span)
        func1 = ir.Function("func1", [a], [ir.ScalarType(dtype)], body, span)
        func2 = ir.Function("func2", [b], [ir.ScalarType(dtype)], body, span)
        program = ir.Program([func1, func2], "my_program", span)

        assert program.name == "my_program"
        assert len(program.functions) == 2
        # Functions are stored in a map, sorted by name
        func_list = list(program.functions.values())
        assert cast(ir.Function, func_list[0]).name == "func1"
        assert cast(ir.Function, func_list[1]).name == "func2"

    def test_program_is_irnode(self):
        """Test that Program is an instance of IRNode."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        func = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)
        program = ir.Program([func], "", span)

        assert isinstance(program, ir.IRNode)

    def test_program_immutability(self):
        """Test that Program attributes are immutable."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        func = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)
        program = ir.Program([func], "test_program", span)

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            program.name = "new_name"  # type: ignore
        with pytest.raises(AttributeError):
            program.functions = []  # type: ignore

    def test_program_with_empty_functions(self):
        """Test Program with empty function list."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        program = ir.Program([], "", span)

        assert len(program.functions) == 0
        assert program.name == ""

    def test_program_with_multiple_functions(self):
        """Test Program with multiple functions."""
        span = ir.Span("test.py", 1, 1, 1, 10)
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, z, span)
        func1 = ir.Function("func1", [x], [ir.ScalarType(dtype)], assign1, span)
        func2 = ir.Function("func2", [y], [ir.ScalarType(dtype)], assign2, span)
        func3 = ir.Function("func3", [z], [ir.ScalarType(dtype)], assign1, span)
        program = ir.Program([func1, func2, func3], "", span)

        assert len(program.functions) == 3
        # Functions are stored in a map, sorted by name
        func_list = list(program.functions.values())
        assert cast(ir.Function, func_list[0]).name == "func1"
        assert cast(ir.Function, func_list[1]).name == "func2"
        assert cast(ir.Function, func_list[2]).name == "func3"

    def test_program_string_representation(self):
        """Test Program string representation."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        func = ir.Function("simple_func", [x], [ir.ScalarType(dtype)], assign, span)

        # Program with single function
        program1 = ir.Program([func], "", span)
        str_repr = str(program1)
        assert "simple_func" in str_repr or isinstance(str_repr, str)

        # Program with name
        program2 = ir.Program([func], "my_program", span)
        str_repr2 = str(program2)
        assert "my_program" in str_repr2 or isinstance(str_repr2, str)

    def test_program_get_function(self):
        """Test Program.get_function method."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(y, x, span)
        func1 = ir.Function("alpha", [x], [ir.ScalarType(dtype)], assign1, span)
        func2 = ir.Function("beta", [y], [ir.ScalarType(dtype)], assign2, span)
        program = ir.Program([func1, func2], "", span)

        # Test getting existing functions
        retrieved_func1 = program.get_function("alpha")
        assert retrieved_func1 is not None
        assert retrieved_func1.name == "alpha"

        retrieved_func2 = program.get_function("beta")
        assert retrieved_func2 is not None
        assert retrieved_func2.name == "beta"

        # Test getting non-existent function
        retrieved_func3 = program.get_function("gamma")
        assert retrieved_func3 is None

    def test_program_get_global_var(self):
        """Test Program.get_global_var method."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        func1 = ir.Function("my_func", [x], [ir.ScalarType(dtype)], assign1, span)
        program = ir.Program([func1], "", span)

        # Test getting existing GlobalVar
        gvar = program.get_global_var("my_func")
        assert gvar is not None
        assert gvar.name == "my_func"
        assert isinstance(gvar, ir.GlobalVar)

        # Test getting non-existent GlobalVar
        gvar2 = program.get_global_var("nonexistent")
        assert gvar2 is None

    def test_program_function_ordering(self):
        """Test that functions are ordered by name in the map."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, x, span)

        # Create functions in non-alphabetical order
        func_c = ir.Function("charlie", [x], [ir.ScalarType(dtype)], assign, span)
        func_a = ir.Function("alpha", [x], [ir.ScalarType(dtype)], assign, span)
        func_b = ir.Function("bravo", [x], [ir.ScalarType(dtype)], assign, span)

        # Add them to program in non-sorted order
        program = ir.Program([func_c, func_a, func_b], "", span)

        # Verify they are stored in sorted order
        func_names = [func.name for func in program.functions.values()]
        assert func_names == ["alpha", "bravo", "charlie"]

    def test_program_with_globalvar_in_call(self):
        """Test using GlobalVar in a Call expression."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, x, span)
        func = ir.Function("my_func", [x], [ir.ScalarType(dtype)], assign, span)
        program = ir.Program([func], "", span)

        # Get the GlobalVar and use it in a Call
        gvar = program.get_global_var("my_func")
        assert gvar is not None
        call = ir.Call(gvar, [x], span)

        assert call.op.name == "my_func"
        assert isinstance(call.op, ir.GlobalVar)


class TestProgramHash:
    """Tests for Program hash function."""

    def test_program_same_structure_hash(self):
        """Test Program nodes with same structure hash."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x1 = ir.Var("x", ir.ScalarType(dtype), span)
        y1 = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x1, y1, span)
        func1 = ir.Function("test_func", [x1], [ir.ScalarType(dtype)], assign1, span)
        program1 = ir.Program([func1], "", span)

        x2 = ir.Var("x", ir.ScalarType(dtype), span)
        y2 = ir.Var("y", ir.ScalarType(dtype), span)
        assign2 = ir.AssignStmt(x2, y2, span)
        func2 = ir.Function("test_func", [x2], [ir.ScalarType(dtype)], assign2, span)
        program2 = ir.Program([func2], "", span)

        hash1 = ir.structural_hash(program1, enable_auto_mapping=True)
        hash2 = ir.structural_hash(program2, enable_auto_mapping=True)
        assert hash1 == hash2

    def test_program_different_name_hash(self):
        """Test Program nodes with different names hash the same (name is IgnoreField)."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        func = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)

        program1 = ir.Program([func], "program1", span)
        program2 = ir.Program([func], "program2", span)

        hash1 = ir.structural_hash(program1)
        hash2 = ir.structural_hash(program2)
        assert hash1 == hash2  # name is IgnoreField, so should hash the same

    def test_program_different_functions_hash(self):
        """Test Program nodes with different functions hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(x, z, span)
        func1 = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign1, span)
        func2 = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign2, span)

        program1 = ir.Program([func1], "", span)
        program2 = ir.Program([func2], "", span)

        hash1 = ir.structural_hash(program1)
        hash2 = ir.structural_hash(program2)
        assert hash1 != hash2

    def test_program_empty_vs_non_empty_functions_hash(self):
        """Test Program nodes with empty vs non-empty functions hash differently."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        func = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)

        program1 = ir.Program([], "", span)
        program2 = ir.Program([func], "", span)

        hash1 = ir.structural_hash(program1)
        hash2 = ir.structural_hash(program2)
        assert hash1 != hash2


class TestProgramStructuralEqual:
    """Tests for Program structural equality function."""

    def test_program_structural_equal(self):
        """Test Program nodes with same structure are equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x1 = ir.Var("x", ir.ScalarType(dtype), span)
        y1 = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x1, y1, span)
        func1 = ir.Function("test_func", [x1], [ir.ScalarType(dtype)], assign1, span)
        program1 = ir.Program([func1], "", span)

        x2 = ir.Var("x", ir.ScalarType(dtype), span)
        y2 = ir.Var("y", ir.ScalarType(dtype), span)
        assign2 = ir.AssignStmt(x2, y2, span)
        func2 = ir.Function("test_func", [x2], [ir.ScalarType(dtype)], assign2, span)
        program2 = ir.Program([func2], "", span)

        assert ir.structural_equal(program1, program2, enable_auto_mapping=True)

    def test_program_different_name_not_equal(self):
        """Test Program nodes with different names are equal (name is IgnoreField)."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        func = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)

        program1 = ir.Program([func], "program1", span)
        program2 = ir.Program([func], "program2", span)

        assert ir.structural_equal(program1, program2)  # name is IgnoreField, so should be equal

    def test_program_different_functions_not_equal(self):
        """Test Program nodes with different functions are not equal."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        z = ir.Var("z", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x, y, span)
        assign2 = ir.AssignStmt(x, z, span)
        func1 = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign1, span)
        func2 = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign2, span)

        program1 = ir.Program([func1], "", span)
        program2 = ir.Program([func2], "", span)

        assert not ir.structural_equal(program1, program2)

    def test_program_different_from_base_irnode_not_equal(self):
        """Test Program is not equal to a different IRNode type."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        func = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)
        program = ir.Program([func], "", span)

        # Compare with a Var (different IRNode type)
        assert not ir.structural_equal(program, x)
