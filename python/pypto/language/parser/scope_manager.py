# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scope management and SSA verification for IR parsing."""

from typing import Any, Optional

from pypto.ir import Span

from .diagnostics import ScopeIsolationError, SSAViolationError


class ScopeManager:
    """Manages variable scopes and optionally enforces SSA properties."""

    def __init__(self, strict_ssa: bool = False):
        """Initialize scope manager.

        Args:
            strict_ssa: If True, enforce SSA (single assignment per variable).
                       If False (default), allow variable reassignment.
        """
        self.strict_ssa = strict_ssa
        self.scopes: list[dict[str, Any]] = [{}]  # Stack of scope dictionaries
        self.assignments: dict[str, int] = {}  # Track assignment count per variable
        self.scope_types: list[str] = ["global"]  # Track type of each scope
        self.yielded_vars: dict[int, set[str]] = {}  # Track yielded variables per scope
        self.var_spans: list[dict[str, Span]] = [{}]  # Track span for each variable definition

    def enter_scope(self, scope_type: str) -> None:
        """Enter a new scope.

        Args:
            scope_type: Type of scope ('function', 'for', 'if')
        """
        self.scopes.append({})
        self.var_spans.append({})
        self.scope_types.append(scope_type)
        scope_id = len(self.scopes) - 1
        self.yielded_vars[scope_id] = set()

    def exit_scope(self, leak_vars: bool = False) -> dict[str, Any]:
        """Exit current scope and return defined variables.

        Args:
            leak_vars: If True, copy all variables from exiting scope to parent scope.
                       Used for plain syntax where variables should be visible after for/if.

        Returns:
            Dictionary of variables defined in the exited scope
        """
        if len(self.scopes) <= 1:
            raise RuntimeError("Cannot exit global scope")

        scope_vars = self.scopes.pop()
        self.var_spans.pop()
        self.scope_types.pop()
        scope_id = len(self.scopes)

        # Leak variables to parent scope if requested
        if leak_vars and self.scopes:
            parent_scope = self.scopes[-1]
            for name, value in scope_vars.items():
                parent_scope[name] = value

        # Clean up yielded vars tracking
        if scope_id in self.yielded_vars:
            del self.yielded_vars[scope_id]

        return scope_vars

    def define_var(
        self, name: str, value: Any, allow_redef: bool = False, span: Optional[Any] = None
    ) -> None:
        """Define a variable in the current scope.

        Args:
            name: Variable name
            value: Variable value/node
            allow_redef: If True, allow redefinition (for iter_args and parameters)
            span: Optional source location span for error reporting

        Raises:
            SSAViolationError: If strict_ssa=True and variable is already defined in current scope
        """
        current_scope = self.scopes[-1]
        current_spans = self.var_spans[-1]

        # Check SSA: variable should not already be defined in current scope
        if name in current_scope and not allow_redef:
            if self.strict_ssa:
                # Get the previous definition span
                previous_span = current_spans.get(name)

                raise SSAViolationError(
                    f"Variable '{name}' is already defined",
                    span=span,
                    previous_span=previous_span,
                    hint="Use a different variable name for each assignment (SSA form requires unique names)",
                    note="Each variable can only be assigned once per scope",
                )
            # In non-SSA mode: just update the variable (no error)

        current_scope[name] = value
        if span is not None:
            current_spans[name] = span

        # Track assignment count globally
        if name not in self.assignments:
            self.assignments[name] = 0
        self.assignments[name] += 1

    def lookup_var(self, name: str) -> Optional[Any]:
        """Lookup variable in scope chain.

        Args:
            name: Variable name to look up

        Returns:
            Variable value/node if found, None otherwise
        """
        # Search from innermost to outermost scope
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def is_defined(self, name: str) -> bool:
        """Check if variable is defined in any accessible scope.

        Args:
            name: Variable name

        Returns:
            True if variable is defined, False otherwise
        """
        return self.lookup_var(name) is not None

    def mark_yielded(self, var_name: str) -> None:
        """Mark a variable as yielded from current scope.

        Args:
            var_name: Variable name that is yielded
        """
        scope_id = len(self.scopes) - 1
        if scope_id not in self.yielded_vars:
            self.yielded_vars[scope_id] = set()
        self.yielded_vars[scope_id].add(var_name)

    def get_yielded_vars(self, scope_id: Optional[int] = None) -> set[str]:
        """Get variables yielded from a specific scope.

        Args:
            scope_id: Scope ID (defaults to current scope)

        Returns:
            Set of yielded variable names
        """
        if scope_id is None:
            scope_id = len(self.scopes) - 1
        return self.yielded_vars.get(scope_id, set())

    def current_scope_type(self) -> str:
        """Get the type of the current scope.

        Returns:
            Scope type string ('global', 'function', 'for', 'if')
        """
        return self.scope_types[-1]

    def in_scope_type(self, scope_type: str) -> bool:
        """Check if currently in a specific scope type.

        Args:
            scope_type: Type to check for

        Returns:
            True if in the specified scope type
        """
        return scope_type in self.scope_types


__all__ = ["ScopeManager", "SSAViolationError", "ScopeIsolationError"]
