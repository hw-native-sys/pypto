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
        assert ir.OptimizationStrategy.O1 is not None
        assert ir.OptimizationStrategy.O2 is not None
        assert ir.OptimizationStrategy.O3 is not None

    def test_optimization_strategy_values_are_different(self):
        """Test that optimization strategies have different values."""
        strategies = [
            ir.OptimizationStrategy.Default,
            ir.OptimizationStrategy.O1,
            ir.OptimizationStrategy.O2,
            ir.OptimizationStrategy.O3,
        ]
        assert len(strategies) == len(set(strategies))


class TestPassManagerBasics:
    """Test basic PassManager functionality."""

    def test_pass_manager_get_strategy_o0(self):
        """Test getting Default strategy PassManager."""
        pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.Default)
        assert pm is not None
        assert pm.strategy == ir.OptimizationStrategy.Default
        # Default should have no passes
        assert len(pm.passes) == 0
        assert len(pm.pass_names) == 0

    def test_pass_manager_get_strategy_o1(self):
        """Test getting O1 strategy PassManager."""
        pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.O1)
        assert pm is not None
        assert pm.strategy == ir.OptimizationStrategy.O1
        # O1 should have 1 pass
        assert len(pm.passes) == 1
        assert len(pm.pass_names) == 1
        assert pm.pass_names[0] == "IdentityPass_1"

    def test_pass_manager_get_strategy_o2(self):
        """Test getting O2 strategy PassManager."""
        pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.O2)
        assert pm is not None
        assert pm.strategy == ir.OptimizationStrategy.O2
        # O2 should have 2 passes
        assert len(pm.passes) == 2
        assert len(pm.pass_names) == 2
        assert pm.pass_names[0] == "IdentityPass_1"
        assert pm.pass_names[1] == "IdentityPass_2"

    def test_pass_manager_get_strategy_o3(self):
        """Test getting O3 strategy PassManager."""
        pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.O3)
        assert pm is not None
        assert pm.strategy == ir.OptimizationStrategy.O3
        # O3 should have 3 passes
        assert len(pm.passes) == 3
        assert len(pm.pass_names) == 3
        assert pm.pass_names[0] == "IdentityPass_1"
        assert pm.pass_names[1] == "IdentityPass_2"
        assert pm.pass_names[2] == "IdentityPass_3"


class TestPassManagerExecution:
    """Test PassManager execution functionality."""

    def test_run_with_o0_strategy(self):
        """Test running PassManager with Default strategy (no passes)."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        func = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)

        pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.Default)
        result = pm.run(func)

        # Default has no passes, should return the same function unchanged
        assert result is func
        assert result.name == "test_func"

    def test_run_with_o1_strategy(self):
        """Test running PassManager with O1 strategy and verify pass execution."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        func = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)

        pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.O1)
        result = pm.run(func)

        # O1 has 1 IdentityPass, should append "_identity" once
        assert result is not func
        assert result.name == "test_func_identity"

    def test_run_with_o2_strategy(self):
        """Test running PassManager with O2 strategy and verify pass execution."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        func = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)

        pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.O2)
        result = pm.run(func)

        # O2 has 2 IdentityPasses, should append "_identity" twice
        assert result is not func
        assert result.name == "test_func_identity_identity"

    def test_run_with_o3_strategy(self):
        """Test running PassManager with O3 strategy and verify pass execution."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        func = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)

        pm = ir.PassManager.get_strategy(ir.OptimizationStrategy.O3)
        result = pm.run(func)

        # O3 has 3 IdentityPasses, should append "_identity" three times
        assert result is not func
        assert result.name == "test_func_identity_identity_identity"

    def test_run_with_default_strategy(self):
        """Test running PassManager with the default strategy (no passes)."""
        span = ir.Span.unknown()
        dtype = DataType.INT64
        x = ir.Var("x", ir.ScalarType(dtype), span)
        y = ir.Var("y", ir.ScalarType(dtype), span)
        assign = ir.AssignStmt(x, y, span)
        func = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)
        pm = ir.PassManager.get_strategy()
        result = pm.run(func)
        # Default strategy has no passes, so the function should be unchanged.
        assert pm.strategy == ir.OptimizationStrategy.Default
        assert result.name == "test_func"


class TestPassManagerMultipleInstances:
    """Test that multiple PassManager instances work independently."""

    def test_multiple_instances_same_strategy(self):
        """Test creating multiple instances of the same strategy."""
        pm1 = ir.PassManager.get_strategy(ir.OptimizationStrategy.O2)
        pm2 = ir.PassManager.get_strategy(ir.OptimizationStrategy.O2)

        # Should be different instances
        assert pm1 is not pm2

        # But should have the same strategy
        assert pm1.strategy == pm2.strategy

        # And same pass names
        assert pm1.get_pass_names() == pm2.get_pass_names()

    def test_multiple_instances_different_strategies(self):
        """Test creating instances of different strategies."""
        pm_o1 = ir.PassManager.get_strategy(ir.OptimizationStrategy.O1)
        pm_o2 = ir.PassManager.get_strategy(ir.OptimizationStrategy.O2)
        pm_o3 = ir.PassManager.get_strategy(ir.OptimizationStrategy.O3)

        # Should have different strategies
        assert pm_o1.strategy != pm_o2.strategy
        assert pm_o2.strategy != pm_o3.strategy
        assert pm_o1.strategy != pm_o3.strategy

        # Should have different pass counts
        assert len(pm_o1.passes) < len(pm_o2.passes)
        assert len(pm_o2.passes) < len(pm_o3.passes)

        # Verify pass names are properly configured
        assert pm_o1.get_pass_names() == ["IdentityPass_1"]
        assert pm_o2.get_pass_names() == ["IdentityPass_1", "IdentityPass_2"]
        assert pm_o3.get_pass_names() == ["IdentityPass_1", "IdentityPass_2", "IdentityPass_3"]
