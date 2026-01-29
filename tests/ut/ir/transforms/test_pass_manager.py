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
        assert ir.OptimizationStrategy.Custom1 is not None
        assert ir.OptimizationStrategy.Custom2 is not None

    def test_optimization_strategy_values_are_different(self):
        """Test that optimization strategies have different values."""
        strategies = [
            ir.OptimizationStrategy.Default,
            ir.OptimizationStrategy.Custom1,
            ir.OptimizationStrategy.Custom2,
        ]
        assert len(strategies) == len(set(strategies))


class TestPassManagerBasics:
    """Test basic PassManager functionality."""

    def test_pass_manager_get_strategy_default(self):
        """Test getting Default strategy PassManager."""
        pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.Default)
        assert pm is not None
        assert pm.strategy == ir.OptimizationStrategy.Default
        # Default should have no passes
        assert len(pm.passes) == 0
        assert len(pm.pass_names) == 0

    def test_pass_manager_get_strategy_custom1(self):
        """Test getting Custom1 strategy PassManager."""
        pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.Custom1)
        assert pm is not None
        assert pm.strategy == ir.OptimizationStrategy.Custom1
        # Custom1 should have 1 pass
        assert len(pm.passes) == 1
        assert len(pm.pass_names) == 1
        assert pm.pass_names[0] == "IdentityPass_1"

    def test_pass_manager_get_strategy_custom2(self):
        """Test getting Custom2 strategy PassManager."""
        pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.Custom2)
        assert pm is not None
        assert pm.strategy == ir.OptimizationStrategy.Custom2
        # Custom2 should have 2 passes
        assert len(pm.passes) == 2
        assert len(pm.pass_names) == 2
        assert pm.pass_names[0] == "IdentityPass_1"
        assert pm.pass_names[1] == "IdentityPass_2"


class TestPassManagerExecution:
    """Test PassManager execution functionality."""

    def test_run_with_default_strategy(self):
        """Test running PassManager with Default strategy (no passes)."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        func = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)

        pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.Default)
        program = ir.Program([func], "test_run_with_default_strategy", ir.Span.unknown())
        result = pm.run_passes(program)
        func = list(result.functions.values())[0]

        # Default has no passes, should return the same function unchanged
        assert result is program
        assert func.name == "test_func"

    def test_run_with_custom1_strategy(self):
        """Test running PassManager with Custom1 strategy and verify pass execution."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        func = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)
        program = ir.Program([func], "test_run_with_custom1_strategy", ir.Span.unknown())
        pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.Custom1)
        result = pm.run_passes(program)
        func = list(result.functions.values())[0]

        # Custom1 has 1 IdentityPass, should append "_identity" once
        assert result is not program
        assert func.name == "test_func_identity"

    def test_run_with_custom2_strategy(self):
        """Test running PassManager with Custom2 strategy and verify pass execution."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        func = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)
        program = ir.Program([func], "test_run_with_custom2_strategy", ir.Span.unknown())
        pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.Custom2)
        result = pm.run_passes(program)
        func = list(result.functions.values())[0]

        # Custom2 has 2 IdentityPasses, should append "_identity" twice
        assert result is not program
        assert func.name == "test_func_identity_identity"

    def test_run_with_implicit_default_strategy(self):
        """Test running PassManager with implicit default strategy (no passes)."""
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
        # Default strategy has no passes, so the function should be unchanged.
        assert pm.strategy == ir.OptimizationStrategy.Default
        assert func.name == "test_func"


class TestPassManagerMultipleInstances:
    """Test that multiple PassManager instances work independently."""

    def test_multiple_instances_same_strategy(self):
        """Test creating multiple instances of the same strategy."""
        pm1 = ir.PassManager.get_strategy(ir.OptimizationStrategy.Custom2)
        pm2 = ir.PassManager.get_strategy(ir.OptimizationStrategy.Custom2)

        # Should be different instances
        assert pm1 is not pm2

        # But should have the same strategy
        assert pm1.strategy == pm2.strategy

        # And same pass names
        assert pm1.get_pass_names() == pm2.get_pass_names()

    def test_multiple_instances_different_strategies(self):
        """Test creating instances of different strategies."""
        pm_custom1 = ir.PassManager.get_strategy(ir.OptimizationStrategy.Custom1)
        pm_custom2 = ir.PassManager.get_strategy(ir.OptimizationStrategy.Custom2)

        # Should have different strategies
        assert pm_custom1.strategy != pm_custom2.strategy

        # Should have different pass counts
        assert len(pm_custom1.passes) < len(pm_custom2.passes)

        # Verify pass names are properly configured
        assert pm_custom1.get_pass_names() == ["IdentityPass_1"]
        assert pm_custom2.get_pass_names() == ["IdentityPass_1", "IdentityPass_2"]


class TestPassManagerWithProgram:
    """Test PassManager execution with Program input."""

    def test_run_passes_on_program_with_custom2_strategy(self):
        """Test running PassManager with Custom2 strategy on a Program."""
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

        pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.Custom2)
        result = pm.run_passes(program)

        # Custom2 has 2 IdentityPasses, should append "_identity" twice to each function
        assert isinstance(result, ir.Program)
        assert result.name == "test_program"
        assert len(result.functions) == 2

        # Get functions from result
        func_names = [func.name for func in result.functions.values()]
        assert "func1_identity_identity" in func_names
        assert "func2_identity_identity" in func_names

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

        pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.Custom1)
        result = pm.run_passes(program)

        # Should have one function with "_identity" suffix
        assert isinstance(result, ir.Program)
        assert result.name == "single_func_program"
        assert len(result.functions) == 1

        func_names = [func.name for func in result.functions.values()]
        assert "single_func_identity" in func_names
