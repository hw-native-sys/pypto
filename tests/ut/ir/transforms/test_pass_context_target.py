# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for TargetType in PassContext."""

import os

import pytest
from pypto.backend import TargetType
from pypto.pypto_core import passes


class TestTargetType:
    """Test TargetType enum."""

    def test_enum_values_exist(self):
        assert TargetType.ASCEND_910B is not None
        assert TargetType.ASCEND_910C is not None
        assert TargetType.ASCEND_950 is not None

    def test_enum_values_are_distinct(self):
        assert TargetType.ASCEND_910B != TargetType.ASCEND_910C
        assert TargetType.ASCEND_910B != TargetType.ASCEND_950
        assert TargetType.ASCEND_910C != TargetType.ASCEND_950


class TestPassContextTarget:
    """Test PassContext target configuration."""

    @pytest.fixture(autouse=True)
    def clean_env(self):
        """Ensure PYPTO_TARGET is unset before/after each test."""
        old = os.environ.pop("PYPTO_TARGET", None)
        yield
        if old is not None:
            os.environ["PYPTO_TARGET"] = old
        else:
            os.environ.pop("PYPTO_TARGET", None)

    def test_default_has_no_target(self):
        ctx = passes.PassContext()
        assert ctx.has_target() is False

    def test_explicit_target(self):
        ctx = passes.PassContext(TargetType.ASCEND_910B)
        assert ctx.has_target() is True
        assert ctx.get_target() == TargetType.ASCEND_910B

    def test_explicit_target_910c(self):
        ctx = passes.PassContext(TargetType.ASCEND_910C)
        assert ctx.get_target() == TargetType.ASCEND_910C

    def test_explicit_target_950(self):
        ctx = passes.PassContext(TargetType.ASCEND_950)
        assert ctx.get_target() == TargetType.ASCEND_950

    def test_get_target_raises_without_target(self):
        ctx = passes.PassContext()
        with pytest.raises(ValueError, match="No target configured"):
            ctx.get_target()

    def test_inner_context_inherits_outer_target(self):
        """Inner PassContext without explicit target inherits outer target."""
        with passes.PassContext(TargetType.ASCEND_910B):
            outer_ctx = passes.PassContext.current()
            assert outer_ctx is not None
            assert outer_ctx.get_target() == TargetType.ASCEND_910B

            # Inner context without target inherits from outer via chain walking
            with passes.PassContext():
                inner_ctx = passes.PassContext.current()
                assert inner_ctx is not None
                assert inner_ctx.has_target() is False
                assert inner_ctx.get_target() == TargetType.ASCEND_910B

    def test_nesting_preserves_target(self):
        outer = passes.PassContext(TargetType.ASCEND_910B)
        inner = passes.PassContext(TargetType.ASCEND_950)
        with outer:
            ctx = passes.PassContext.current()
            assert ctx is not None
            assert ctx.get_target() == TargetType.ASCEND_910B
            with inner:
                inner_ctx = passes.PassContext.current()
                assert inner_ctx is not None
                assert inner_ctx.get_target() == TargetType.ASCEND_950
            restored_ctx = passes.PassContext.current()
            assert restored_ctx is not None
            assert restored_ctx.get_target() == TargetType.ASCEND_910B


class TestPassContextCurrentTarget:
    """Test static PassContext.current_target()."""

    @pytest.fixture(autouse=True)
    def pass_verification_context(self):
        """Override conftest: no PassContext for target tests."""
        yield

    def test_current_target_with_active_context(self):
        with passes.PassContext(TargetType.ASCEND_910B):
            assert passes.PassContext.current_target() == TargetType.ASCEND_910B

    def test_current_target_raises_without_context(self):
        assert passes.PassContext.current() is None
        with pytest.raises(ValueError, match="No target configured"):
            passes.PassContext.current_target()


class TestTargetFromEnv:
    """Test PYPTO_TARGET environment variable."""

    @pytest.fixture(autouse=True)
    def pass_verification_context(self):
        """Override conftest: no PassContext for env var tests."""
        yield

    @pytest.fixture(autouse=True)
    def clean_env(self):
        """Ensure PYPTO_TARGET is unset before/after each test."""
        old = os.environ.pop("PYPTO_TARGET", None)
        yield
        if old is not None:
            os.environ["PYPTO_TARGET"] = old
        else:
            os.environ.pop("PYPTO_TARGET", None)

    def test_env_var_fallback(self):
        os.environ["PYPTO_TARGET"] = "910B"
        ctx = passes.PassContext()
        assert ctx.get_target() == TargetType.ASCEND_910B

    def test_env_var_current_target(self):
        os.environ["PYPTO_TARGET"] = "950"
        assert passes.PassContext.current_target() == TargetType.ASCEND_950

    def test_invalid_env_var_raises(self):
        os.environ["PYPTO_TARGET"] = "INVALID"
        ctx = passes.PassContext()
        with pytest.raises(ValueError, match="Unknown target type"):
            ctx.get_target()

    def test_explicit_target_overrides_env(self):
        os.environ["PYPTO_TARGET"] = "910B"
        ctx = passes.PassContext(TargetType.ASCEND_950)
        assert ctx.get_target() == TargetType.ASCEND_950


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
