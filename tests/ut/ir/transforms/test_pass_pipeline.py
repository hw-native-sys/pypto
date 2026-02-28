# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for PassPipeline and PassContext."""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes


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


def _make_non_ssa_program() -> ir.Program:
    """Create a program that has not been through ConvertToSSA.

    Uses the DSL without strict_ssa, producing a valid program that
    passes pointer-based SSA checks but has not been SSA-converted.
    """

    @pl.program
    class NonSSA:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

    return NonSSA  # type: ignore[return-value]


def _make_ssa_violating_program() -> ir.Program:
    """Create a program with a genuine SSA violation (same Var assigned twice)."""
    span = ir.Span.unknown()
    tensor_type = ir.TensorType([64], DataType.FP32)
    x = ir.Var("x", tensor_type, span)
    result = ir.Var("result", tensor_type, span)

    # Assign to the same Var pointer twice — genuine SSA violation
    assign1 = ir.AssignStmt(result, x, span)
    assign2 = ir.AssignStmt(result, x, span)
    return_stmt = ir.ReturnStmt([result], span)
    body = ir.SeqStmts([assign1, assign2, return_stmt], span)

    func = ir.Function("main", [x], [tensor_type], body, span)
    return ir.Program([func], "test_program", span)


def _make_valid_ssa_program() -> ir.Program:
    """Create a valid SSA program (unique variable names)."""

    @pl.program(strict_ssa=True)
    class ValidSSA:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            return result

    return ValidSSA  # type: ignore[return-value]


class TestPassContext:
    """Test PassContext and instrument system."""

    def test_context_is_active_from_conftest(self):
        """The conftest autouse fixture sets up a context for all tests."""
        assert passes.PassContext.current() is not None

    def test_context_nesting_overrides_outer(self):
        """Inner context overrides outer context (conftest provides the outer)."""
        outer_ctx = passes.PassContext.current()
        assert outer_ctx is not None
        inner = passes.PassContext([])  # no instruments
        with inner:
            # Inner context is now active, overriding conftest's
            assert passes.PassContext.current() is not None
            assert passes.PassContext.current() is not outer_ctx
        # Outer (conftest) context restored
        assert passes.PassContext.current() is outer_ctx

    def test_after_mode_succeeds_on_valid_pipeline(self):
        """AFTER mode succeeds when pass actually produces its claimed property."""
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.AFTER)]):
            result = passes.convert_to_ssa()(_make_non_ssa_program())
            assert result is not None

    def test_after_mode_succeeds_with_multiple_passes(self):
        """AFTER mode verifies produced properties after each pass in sequence."""
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.AFTER)]):
            program = _make_non_ssa_program()
            program = passes.convert_to_ssa()(program)
            program = passes.flatten_call_expr()(program)
            assert program is not None

    def test_after_mode_succeeds_with_normalize(self):
        """AFTER mode verifies NormalizedStmtStructure."""
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.AFTER)]):
            program = _make_valid_ssa_program()
            result = passes.normalize_stmt_structure()(program)
            assert result is not None

    def test_before_mode_catches_false_ssa_claim(self):
        """BEFORE mode detects that required SSAForm doesn't actually hold."""
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE)]):
            # Same Var assigned twice — genuine SSA violation
            program = _make_ssa_violating_program()
            with pytest.raises(Exception, match="Pre-verification failed"):
                passes.outline_incore_scopes()(program)

    def test_before_mode_succeeds_when_property_holds(self):
        """BEFORE mode passes when the required property actually holds."""
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE)]):
            program = _make_non_ssa_program()
            program = passes.convert_to_ssa()(program)
            result = passes.outline_incore_scopes()(program)
            assert result is not None

    def test_empty_context_disables_verification(self):
        """Empty instrument list overrides conftest's verification context."""
        with passes.PassContext([]):
            # OutlineIncoreScopes requires SSAForm, but empty context = no check
            program = _make_non_ssa_program()
            result = passes.outline_incore_scopes()(program)
            assert result is not None

    def test_before_and_after_succeeds_on_valid_pipeline(self):
        """BEFORE_AND_AFTER mode succeeds when all properties are correct."""
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
            program = _make_non_ssa_program()
            program = passes.convert_to_ssa()(program)
            program = passes.flatten_call_expr()(program)
            assert program is not None

    def test_before_and_after_catches_pre_violation(self):
        """BEFORE_AND_AFTER catches pre-pass property violations."""
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
            # Same Var assigned twice — genuine SSA violation
            program = _make_ssa_violating_program()
            with pytest.raises(Exception, match="Pre-verification failed"):
                passes.outline_incore_scopes()(program)

    def test_pipeline_with_context(self):
        """PassPipeline respects active PassContext instruments."""
        pipeline = passes.PassPipeline()
        pipeline.add_pass(passes.convert_to_ssa())
        pipeline.add_pass(passes.flatten_call_expr())

        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.AFTER)]):
            result = pipeline.run(_make_non_ssa_program())
            assert result is not None


class TestCallbackInstrument:
    """Test CallbackInstrument with user-provided callbacks."""

    def test_after_callback_fires_for_each_pass(self):
        """After callback is invoked once per pass."""
        log: list[str] = []

        def after_cb(p: passes.Pass, _program: ir.Program) -> None:
            log.append(p.get_name())

        instrument = passes.CallbackInstrument(after_pass=after_cb)

        pipeline = passes.PassPipeline()
        pipeline.add_pass(passes.convert_to_ssa())
        pipeline.add_pass(passes.flatten_call_expr())

        with passes.PassContext([instrument]):
            pipeline.run(_make_non_ssa_program())

        assert log == ["ConvertToSSA", "FlattenCallExpr"]

    def test_before_callback_fires_for_each_pass(self):
        """Before callback is invoked once per pass."""
        log: list[str] = []

        def before_cb(p: passes.Pass, _program: ir.Program) -> None:
            log.append(p.get_name())

        instrument = passes.CallbackInstrument(before_pass=before_cb)

        pipeline = passes.PassPipeline()
        pipeline.add_pass(passes.convert_to_ssa())
        pipeline.add_pass(passes.flatten_call_expr())

        with passes.PassContext([instrument]):
            pipeline.run(_make_non_ssa_program())

        assert log == ["ConvertToSSA", "FlattenCallExpr"]

    def test_none_callbacks_no_error(self):
        """None callbacks are silently skipped."""
        instrument = passes.CallbackInstrument()

        pipeline = passes.PassPipeline()
        pipeline.add_pass(passes.convert_to_ssa())

        with passes.PassContext([instrument]):
            result = pipeline.run(_make_non_ssa_program())

        assert result is not None

    def test_callback_instrument_name(self):
        """Custom name is returned by get_name()."""
        instrument = passes.CallbackInstrument(name="MyInstrument")
        assert instrument.get_name() == "MyInstrument"

    def test_default_name(self):
        """Default name is CallbackInstrument."""
        instrument = passes.CallbackInstrument()
        assert instrument.get_name() == "CallbackInstrument"


class TestVerifiedProperties:
    """Test verified property configuration."""

    @pytest.fixture(autouse=True)
    def pass_verification_context(self):
        """Override conftest: no PassContext for verification tests."""
        yield

    def test_verified_properties_contains_expected(self):
        """Verified properties include SSAForm, TypeChecked, AllocatedMemoryAddr."""
        props = passes.get_verified_properties()
        assert props.contains(passes.IRProperty.SSAForm)
        assert props.contains(passes.IRProperty.TypeChecked)
        assert props.contains(passes.IRProperty.AllocatedMemoryAddr)

    def test_allocated_memory_addr_exists_in_enum(self):
        """AllocatedMemoryAddr is accessible in IRProperty enum."""
        prop = passes.IRProperty.AllocatedMemoryAddr
        assert prop is not None


class TestVerificationLevel:
    """Test verification level via PassContext."""

    @pytest.fixture(autouse=True)
    def pass_verification_context(self):
        """Override conftest: no PassContext for verification tests."""
        yield

    def test_pass_context_default_level_is_basic(self):
        """PassContext defaults to Basic verification level."""
        ctx = passes.PassContext([])
        assert ctx.get_verification_level() == passes.VerificationLevel.BASIC

    def test_pass_context_accepts_none_level(self):
        """PassContext can be created with NONE verification level."""
        ctx = passes.PassContext([], passes.VerificationLevel.NONE)
        assert ctx.get_verification_level() == passes.VerificationLevel.NONE

    def test_verification_level_exposed_on_ir_module(self):
        """VerificationLevel is re-exported from pypto.ir."""
        assert hasattr(ir, "VerificationLevel")
        assert ir.VerificationLevel.BASIC == passes.VerificationLevel.BASIC


class TestAutoVerificationPipeline:
    """Test automatic verification in PassPipeline."""

    @pytest.fixture(autouse=True)
    def pass_verification_context(self):
        """Override conftest: no PassContext for verification tests."""
        yield

    def test_valid_program_passes_verification_no_context(self):
        """A valid program passes verification (no PassContext)."""
        assert passes.PassContext.current() is None
        pipeline = passes.PassPipeline()
        pipeline.add_pass(passes.convert_to_ssa())
        pipeline.add_pass(passes.flatten_call_expr())

        program = _make_non_ssa_program()
        result = pipeline.run(program)
        assert result is not None

    def test_valid_program_passes_verification_with_context(self):
        """A valid program passes verification via PassContext."""
        with passes.PassContext([], passes.VerificationLevel.BASIC):
            pipeline = passes.PassPipeline()
            pipeline.add_pass(passes.convert_to_ssa())
            pipeline.add_pass(passes.flatten_call_expr())

            program = _make_non_ssa_program()
            result = pipeline.run(program)
            assert result is not None

    def test_none_level_disables_verification(self):
        """VerificationLevel.NONE disables verification entirely."""
        with passes.PassContext([], passes.VerificationLevel.NONE):
            pipeline = passes.PassPipeline()
            pipeline.add_pass(passes.convert_to_ssa())

            program = _make_ssa_violating_program()
            result = pipeline.run(program)
            assert result is not None

    def test_full_pipeline_skips_already_verified_properties(self):
        """TypeChecked is produced by both ConvertToSSA and FlattenCallExpr,
        but already-verified properties are not re-checked."""
        assert passes.PassContext.current() is None

        pipeline = passes.PassPipeline()
        pipeline.add_pass(passes.convert_to_ssa())
        pipeline.add_pass(passes.flatten_call_expr())

        program = _make_non_ssa_program()
        # Should succeed — TypeChecked verified after ConvertToSSA,
        # not re-verified after FlattenCallExpr
        result = pipeline.run(program)
        assert result is not None


class TestVerifyProperties:
    """Test the verify_properties helper directly."""

    @pytest.fixture(autouse=True)
    def pass_verification_context(self):
        """Override conftest: no PassContext for verification tests."""
        yield

    def test_valid_program_passes(self):
        """Helper does not throw for valid programs."""
        program = _make_non_ssa_program()
        # Run ConvertToSSA to establish SSAForm + TypeChecked
        ssa_program = passes.convert_to_ssa()(program)

        props = passes.IRPropertySet()
        props.insert(passes.IRProperty.SSAForm)
        props.insert(passes.IRProperty.TypeChecked)

        # Should not raise
        passes.verify_properties(props, ssa_program, "ConvertToSSA")

    def test_invalid_program_throws(self):
        """Helper throws for programs with SSA violations (pre-SSA-conversion)."""
        # Verify SSAForm on a program that has NOT been through ConvertToSSA
        program = _make_ssa_violating_program()

        props = passes.IRPropertySet()
        props.insert(passes.IRProperty.SSAForm)

        with pytest.raises(Exception, match="Verification failed"):
            passes.verify_properties(props, program, "TestPass")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
