# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for PassManager and Pass classes."""

from pypto import DataType, ir


class TestOptimizationStrategy:
    """Test OptimizationStrategy enum."""

    def test_optimization_strategy_values(self):
        """Test that all optimization strategies exist."""
        assert ir.OptimizationStrategy.Default is not None
        assert ir.OptimizationStrategy.PTOAS is not None

    def test_optimization_strategy_values_are_different(self):
        """Test that optimization strategies have different values."""
        strategies = [
            ir.OptimizationStrategy.Default,
            ir.OptimizationStrategy.PTOAS,
        ]
        assert len(strategies) == len(set(strategies))


class TestPassManagerBasics:
    """Test basic PassManager functionality."""

    def test_pass_manager_get_strategy_ptoa(self):
        """Test getting PTOAS strategy PassManager."""
        pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.PTOAS)
        assert pm is not None
        assert pm.strategy == ir.OptimizationStrategy.PTOAS
        # PTOAS has 3 passes: InitMemRef, MemoryReuse, AddAlloc
        assert len(pm.passes) == 3
        assert len(pm.pass_names) == 3
        assert pm.pass_names[0] == "InitMemRef"
        assert pm.pass_names[1] == "MemoryReuse"
        assert pm.pass_names[2] == "AddAlloc"


class TestPassManagerExecution:
    """Test PassManager execution functionality."""

    def test_run_with_implicit_default_strategy(self):
        """Test running PassManager with implicit default strategy."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        func = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)
        pm = ir.PassManager.get_strategy()
        program = ir.Program([func], "test_run_with_implicit_default_strategy", ir.Span.unknown())
        result = pm.run_passes(program)
        func = list(result.functions.values())[0]
        # Default strategy runs InitMemRef, MemoryReuse, InsertSync, AddAlloc; function name unchanged
        assert pm.strategy == ir.OptimizationStrategy.Default
        assert result is not program
        assert func.name == "test_func"


class TestPassManagerMultipleInstances:
    """Test that multiple PassManager instances work independently."""

    def test_multiple_instances_same_strategy(self):
        """Test creating multiple instances of the same strategy."""
        pm1 = ir.PassManager.get_strategy(ir.OptimizationStrategy.PTOAS)
        pm2 = ir.PassManager.get_strategy(ir.OptimizationStrategy.PTOAS)

        # Should be different instances
        assert pm1 is not pm2

        # But should have the same strategy
        assert pm1.strategy == pm2.strategy

        # And same pass names
        assert pm1.get_pass_names() == pm2.get_pass_names()


class TestPassManagerWithProgram:
    """Test PassManager execution with Program input."""

    def test_run_passes_on_program_with_ptoa_strategy(self):
        """Test running PassManager with PTOAS strategy on a Program."""
        span = ir.Span.unknown()
        dtype = DataType.INT64

        # Create first function
        x1 = ir.Var("x", ir.ScalarType(dtype), span)
        y1 = ir.Var("y", ir.ScalarType(dtype), span)
        assign1 = ir.AssignStmt(x1, y1, span)
        func1 = ir.Function("func1", [x1], [ir.ScalarType(dtype)], assign1, span)

        # Create second function
        x2 = ir.Var("x", ir.ScalarType(dtype), span)
        y2 = ir.Var("y", ir.ScalarType(dtype), span)
        assign2 = ir.AssignStmt(x2, y2, span)
        func2 = ir.Function("func2", [x2], [ir.ScalarType(dtype)], assign2, span)

        # Create program with both functions
        program = ir.Program([func1, func2], "test_program", span)

        pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.PTOAS)
        result = pm.run_passes(program)

        # PTOAS runs InitMemRef, MemoryReuse, AddAlloc; function names unchanged
        assert isinstance(result, ir.Program)
        assert result.name == "test_program"
        assert len(result.functions) == 2

        func_names = [func.name for func in result.functions.values()]
        assert "func1" in func_names
        assert "func2" in func_names

    def test_run_passes_on_single_function_program(self):
        """Test running PassManager on a Program with a single function."""
        span = ir.Span.unknown()
        dtype = DataType.INT64

        # Create a single function
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        func = ir.Function("single_func", [x], [ir.ScalarType(dtype)], assign, span)

        # Create program with single function
        program = ir.Program([func], "single_func_program", span)

        pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.PTOAS)
        result = pm.run_passes(program)

        assert isinstance(result, ir.Program)
        assert result.name == "single_func_program"
        assert len(result.functions) == 1

        func_names = [func.name for func in result.functions.values()]
        assert "single_func" in func_names
