# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ScopeManager."""

import pytest
from pypto.language.parser.scope_manager import ScopeManager, SSAViolationError


class TestScopeManager:
    """Tests for ScopeManager class."""

    def test_initialization(self):
        """Test ScopeManager initializes correctly."""
        sm = ScopeManager()

        assert len(sm.scopes) == 1  # Global scope
        assert sm.current_scope_type() == "global"

    def test_enter_exit_scope(self):
        """Test entering and exiting scopes."""
        sm = ScopeManager()

        sm.enter_scope("function")
        assert len(sm.scopes) == 2
        assert sm.current_scope_type() == "function"

        sm.exit_scope()
        assert len(sm.scopes) == 1
        assert sm.current_scope_type() == "global"

    def test_nested_scopes(self):
        """Test nested scope management."""
        sm = ScopeManager()

        sm.enter_scope("function")
        sm.enter_scope("for")
        sm.enter_scope("if")

        assert len(sm.scopes) == 4  # global + function + for + if
        assert sm.current_scope_type() == "if"

        sm.exit_scope()
        assert sm.current_scope_type() == "for"

        sm.exit_scope()
        assert sm.current_scope_type() == "function"

    def test_define_var(self):
        """Test defining variables in scope."""
        sm = ScopeManager()

        sm.enter_scope("function")
        sm.define_var("x", "value_x")

        assert sm.is_defined("x")
        assert sm.lookup_var("x") == "value_x"

    def test_ssa_violation(self):
        """Test SSA violation detection when strict_ssa=True."""
        sm = ScopeManager(strict_ssa=True)

        sm.enter_scope("function")
        sm.define_var("x", "value1")

        # Trying to redefine should raise SSAViolationError
        with pytest.raises(SSAViolationError, match="already defined"):
            sm.define_var("x", "value2")

    def test_allow_redef(self):
        """Test allowing redefinition for special cases."""
        sm = ScopeManager()

        sm.enter_scope("function")
        sm.define_var("x", "value1", allow_redef=True)
        sm.define_var("x", "value2", allow_redef=True)  # Should not raise

        assert sm.lookup_var("x") == "value2"

    def test_variable_shadowing(self):
        """Test that variables in inner scopes shadow outer scopes."""
        sm = ScopeManager()

        sm.enter_scope("function")
        sm.define_var("x", "outer")

        sm.enter_scope("for")
        sm.define_var("x", "inner")

        # Inner scope should see inner value
        assert sm.lookup_var("x") == "inner"

        sm.exit_scope()
        # Outer scope should see outer value
        assert sm.lookup_var("x") == "outer"

    def test_lookup_undefined_var(self):
        """Test looking up undefined variable returns None."""
        sm = ScopeManager()

        assert sm.lookup_var("undefined") is None
        assert not sm.is_defined("undefined")

    def test_mark_yielded(self):
        """Test marking variables as yielded."""
        sm = ScopeManager()

        sm.enter_scope("for")
        sm.mark_yielded("result")

        yielded = sm.get_yielded_vars()
        assert "result" in yielded

    def test_in_scope_type(self):
        """Test checking if in specific scope type."""
        sm = ScopeManager()

        assert sm.in_scope_type("global")
        assert not sm.in_scope_type("function")

        sm.enter_scope("function")
        assert sm.in_scope_type("function")
        assert sm.in_scope_type("global")  # Still in global too

        sm.enter_scope("for")
        assert sm.in_scope_type("for")
        assert sm.in_scope_type("function")

    def test_exit_global_scope_error(self):
        """Test that exiting global scope raises error."""
        sm = ScopeManager()

        with pytest.raises(RuntimeError, match="Cannot exit global scope"):
            sm.exit_scope()

    def test_scope_isolation(self):
        """Test that scope variables are properly isolated."""
        sm = ScopeManager()

        sm.enter_scope("function")
        sm.define_var("x", "func_var")

        sm.enter_scope("for")
        sm.define_var("y", "loop_var")

        # Both variables should be accessible in inner scope
        assert sm.is_defined("x")
        assert sm.is_defined("y")

        scope_vars = sm.exit_scope()

        # After exiting, loop variable should not be in function scope
        assert "y" in scope_vars
        assert sm.is_defined("x")
        # y is no longer accessible after exiting its scope
        assert not sm.is_defined("y")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
