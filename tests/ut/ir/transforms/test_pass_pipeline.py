# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for PassPipeline."""

import pypto.language as pl
import pytest
from pypto import DataType, ir
from pypto.pypto_core import passes


def _make_simple_program():
    """Create a simple valid program for testing."""
    span = ir.Span.unknown()
    dtype = DataType.INT64
    x = ir.Var("x", ir.ScalarType(dtype), span)
    y = ir.Var("y", ir.ScalarType(dtype), span)
    assign = ir.AssignStmt(x, y, span)
    func = ir.Function("test_func", [x], [ir.ScalarType(dtype)], assign, span)
    return ir.Program([func], "test_program", span)


class TestPassPipeline:
    """Test PassPipeline creation and basic operations."""

    def test_empty_pipeline(self):
        """Test creating an empty pipeline."""
        pipeline = passes.PassPipeline()
        assert pipeline.get_pass_names() == []

    def test_add_passes(self):
        """Test adding passes to pipeline."""
        pipeline = passes.PassPipeline()
        pipeline.add_pass(passes.convert_to_ssa())
        pipeline.add_pass(passes.flatten_call_expr())
        assert pipeline.get_pass_names() == ["ConvertToSSA", "FlattenCallExpr"]

    def test_run_empty_pipeline(self):
        """Test running an empty pipeline returns the same program."""
        pipeline = passes.PassPipeline()
        program = _make_simple_program()
        result = pipeline.run(program)
        assert result is not None

    def test_run_single_pass(self):
        """Test running a pipeline with a single pass."""
        pipeline = passes.PassPipeline()
        pipeline.add_pass(passes.convert_to_ssa())
        program = _make_simple_program()
        result = pipeline.run(program)
        assert result is not None


class TestPassPipelineNoEnforcement:
    """Test that PassPipeline does not enforce required properties as prerequisites."""

    def test_run_succeeds_without_required_properties(self):
        """Test that Run succeeds even when required properties are not tracked."""
        pipeline = passes.PassPipeline()
        pipeline.add_pass(passes.basic_memory_reuse())
        program = _make_simple_program()
        result = pipeline.run(program)
        assert result is not None

    def test_run_succeeds_with_initial_properties(self):
        """Test Run succeeds with initial properties set."""
        pipeline = passes.PassPipeline()
        initial = passes.IRPropertySet()
        initial.insert(passes.IRProperty.HasMemRefs)
        pipeline.set_initial_properties(initial)
        pipeline.add_pass(passes.basic_memory_reuse())
        program = _make_simple_program()
        result = pipeline.run(program)
        assert result is not None


class TestVerificationMode:
    """Test VerificationMode enum."""

    def test_verification_mode_values(self):
        """Test that all verification modes exist."""
        assert passes.VerificationMode.NONE is not None
        assert passes.VerificationMode.BEFORE is not None
        assert passes.VerificationMode.AFTER is not None
        assert passes.VerificationMode.BEFORE_AND_AFTER is not None

    def test_set_verification_mode(self):
        """Test setting verification mode on pipeline."""
        pipeline = passes.PassPipeline()
        pipeline.set_verification_mode(passes.VerificationMode.BEFORE_AND_AFTER)
        # Should not raise
        assert True


def _make_non_ssa_program():
    """Create a program with SSA violations (duplicate assignment)."""

    @pl.program
    class NonSSA:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result = pl.add(x, 1.0)
            result = pl.add(result, 2.0)  # noqa: F841 - SSA violation
            return result

    return NonSSA


def _make_valid_ssa_program():
    """Create a valid SSA program (unique variable names)."""

    @pl.program(strict_ssa=True)
    class ValidSSA:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

    return ValidSSA


class TestVerificationModeAfter:
    """Test AFTER verification mode: checks produced properties after each pass."""

    def test_after_mode_succeeds_on_valid_pipeline(self):
        """AFTER mode succeeds when pass actually produces its claimed property."""
        pipeline = passes.PassPipeline()
        pipeline.set_verification_mode(passes.VerificationMode.AFTER)
        pipeline.add_pass(passes.convert_to_ssa())
        program = _make_non_ssa_program()
        result = pipeline.run(program)
        assert result is not None

    def test_after_mode_succeeds_with_multiple_passes(self):
        """AFTER mode verifies produced properties after each pass in sequence."""
        pipeline = passes.PassPipeline()
        pipeline.set_verification_mode(passes.VerificationMode.AFTER)
        pipeline.add_pass(passes.convert_to_ssa())
        pipeline.add_pass(passes.flatten_call_expr())
        program = _make_non_ssa_program()
        result = pipeline.run(program)
        assert result is not None

    def test_after_mode_succeeds_with_normalize_and_flatten(self):
        """AFTER mode verifies NormalizedStmtStructure and FlattenedSingleStmt."""
        pipeline = passes.PassPipeline()
        pipeline.set_verification_mode(passes.VerificationMode.AFTER)
        pipeline.add_pass(passes.normalize_stmt_structure())
        program = _make_valid_ssa_program()
        result = pipeline.run(program)  # type: ignore[arg-type]  # @pl.program returns Program at runtime
        assert result is not None


class TestVerificationModeBefore:
    """Test BEFORE verification mode: checks required properties before each pass."""

    def test_before_mode_catches_false_ssa_claim(self):
        """BEFORE mode detects that claimed SSAForm doesn't actually hold."""
        pipeline = passes.PassPipeline()
        pipeline.set_verification_mode(passes.VerificationMode.BEFORE)
        # Lie about initial properties — claim SSAForm holds
        initial = passes.IRPropertySet()
        initial.insert(passes.IRProperty.SSAForm)
        pipeline.set_initial_properties(initial)
        # OutlineIncoreScopes requires SSAForm — BEFORE mode will verify it
        pipeline.add_pass(passes.outline_incore_scopes())
        program = _make_non_ssa_program()
        with pytest.raises(Exception, match="Pre-verification failed"):
            pipeline.run(program)

    def test_before_mode_succeeds_when_property_holds(self):
        """BEFORE mode passes when the claimed property actually holds."""
        pipeline = passes.PassPipeline()
        pipeline.set_verification_mode(passes.VerificationMode.BEFORE)
        # First produce SSAForm, then use a pass that requires it
        pipeline.add_pass(passes.convert_to_ssa())
        pipeline.add_pass(passes.outline_incore_scopes())
        program = _make_non_ssa_program()
        result = pipeline.run(program)
        assert result is not None

    def test_none_mode_skips_verification_of_false_claim(self):
        """NONE mode doesn't verify — false initial properties pass unchecked."""
        pipeline = passes.PassPipeline()
        pipeline.set_verification_mode(passes.VerificationMode.NONE)
        # Same false claim as above, but NONE mode won't check
        initial = passes.IRPropertySet()
        initial.insert(passes.IRProperty.SSAForm)
        pipeline.set_initial_properties(initial)
        pipeline.add_pass(passes.outline_incore_scopes())
        program = _make_non_ssa_program()
        # Should NOT raise — no verification is performed
        result = pipeline.run(program)
        assert result is not None


class TestVerificationModeBeforeAndAfter:
    """Test BEFORE_AND_AFTER verification mode."""

    def test_before_and_after_succeeds_on_valid_pipeline(self):
        """BEFORE_AND_AFTER mode succeeds when all properties are correct."""
        pipeline = passes.PassPipeline()
        pipeline.set_verification_mode(passes.VerificationMode.BEFORE_AND_AFTER)
        pipeline.add_pass(passes.convert_to_ssa())
        pipeline.add_pass(passes.flatten_call_expr())
        program = _make_non_ssa_program()
        result = pipeline.run(program)
        assert result is not None

    def test_before_and_after_catches_pre_violation(self):
        """BEFORE_AND_AFTER catches pre-pass property violations."""
        pipeline = passes.PassPipeline()
        pipeline.set_verification_mode(passes.VerificationMode.BEFORE_AND_AFTER)
        initial = passes.IRPropertySet()
        initial.insert(passes.IRProperty.SSAForm)
        pipeline.set_initial_properties(initial)
        pipeline.add_pass(passes.outline_incore_scopes())
        program = _make_non_ssa_program()
        with pytest.raises(Exception, match="Pre-verification failed"):
            pipeline.run(program)

    def test_before_and_after_full_default_strategy(self):
        """Full default strategy succeeds with BEFORE_AND_AFTER verification."""
        pipeline = passes.PassPipeline()
        pipeline.set_verification_mode(passes.VerificationMode.BEFORE_AND_AFTER)
        pipeline.add_pass(passes.convert_to_ssa())
        pipeline.add_pass(passes.flatten_call_expr())
        pipeline.add_pass(passes.normalize_stmt_structure())
        pipeline.add_pass(passes.flatten_single_stmt())
        program = _make_non_ssa_program()
        result = pipeline.run(program)
        assert result is not None


class TestPassManagerWithPipeline:
    """Test PassManager uses PassPipeline correctly."""

    def test_pass_manager_with_verification_mode(self):
        """Test PassManager with verification mode."""
        pm = ir.PassManager.get_strategy(
            ir.OptimizationStrategy.Default,
            verification_mode=ir.VerificationMode.AFTER,
        )
        assert pm is not None
