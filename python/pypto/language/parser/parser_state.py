# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit parser service contracts and state with distinct lifetimes."""

import ast
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pypto.ir import IRBuilder
from pypto.ir.builder import ForLoopBuilder, IfStmtBuilder, WhileLoopBuilder
from pypto.pypto_core import ir

from .expr_evaluator import ExprEvaluator
from .scope_manager import ScopeManager
from .span_tracker import SpanTracker
from .type_resolver import (
    TypeResolver,  # noqa: PLC2701
)


class AssignCallback(Protocol):
    """Bind a value using the entry parser's existing SSA/aliasing rules."""

    def __call__(
        self,
        var_name: str,
        value_expr: ir.Expr,
        span: ir.Span,
        override_type: ir.Type | None = None,
    ) -> ir.Var: ...


@dataclass(frozen=True)
class ParserServices:
    """Stable per-parser services; components must not snapshot their mutable contents.

    Inline expansion temporarily changes the evaluator's closure and the span
    tracker's source coordinates. All components observe those changes through
    these same service instances.
    """

    span_tracker: SpanTracker
    scope_manager: ScopeManager
    expr_evaluator: ExprEvaluator
    type_resolver: TypeResolver
    builder: IRBuilder


@dataclass(frozen=True)
class ParseCallbacks:
    """Recursive entry points without a back-reference to the whole ASTParser.

    Calls are deferred until parsing starts, after all components are composed.
    Assignment stays centralized so indexing and submit preserve Var identity.
    """

    expression: Callable[[ast.expr], ir.Expr]
    statement: Callable[[ast.stmt], None]
    body: Callable[[Sequence[ast.stmt]], None]
    assign: AssignCallback
    bind_unpack_target: Callable[[ast.expr, ir.Expr, ir.Span], list[ir.Var]]


class FunctionContext:
    """Function metadata shared by parsing components; reset at function entry."""

    def __init__(self) -> None:
        # Forward-sticky ``pl.dump_tag`` set (per function, reset at function
        # entry). Holds the bound Vars whose subsequent kernel-call uses get a
        # per-call ``dump_vars`` entry. See ``_handle_dump_tag``.
        self.dump_tagged_vars: list[Any] = []

        # Current function name (set during parse_function)
        self.func_name: str = ""

        # Current function level (set during parse_function). Drives
        # context-scoped op constraints (e.g. pld.system.world_size is host-only).
        self.func_level: ir.Level | None = None

        # Current function type (set during parse_function). Used by markers
        # whose validity is scoped to a specific function type — e.g.
        # ``pl.dump_tag`` only makes sense in Orchestration functions.
        self.func_type: ir.FunctionType = ir.FunctionType.Opaque

        # Dyn-dim symbols (``pl.dynamic()`` Vars) the current signature declares as a
        # bare extent of a tensor param, keyed by ``Var.unique_id`` (see
        # ``_fold_tensor_dim``). Per-function; reset in parse_function.
        self.param_dim_symbols: set[int] = set()

        # Fresh tensor.dim scalar aliases are invalidated when their binding changes.
        self.shape_dim_aliases: dict[int, ir.Call] = {}

        # Current function's auto_scope flag (set during parse_function). When
        # True (default) the compiler owns AUTO scope placement, so a hand-placed
        # AUTO `with pl.scope()` is rejected; set False to take control. MANUAL
        # scopes are allowed regardless (they are a dependency-semantics choice).
        self.func_auto_scope: bool = True


class ControlFlowState:
    """Active control-flow state shared with explicit split regions."""

    def __init__(self) -> None:
        # Track context for handling yields and returns
        self.in_for_loop = False

        self.in_while_loop = False

        self.in_if_stmt = False

        self.current_if_builder: IfStmtBuilder | None = None

        self.current_loop_builder: ForLoopBuilder | WhileLoopBuilder | None = None

        # Track loop kinds for break/continue validation
        self.loop_kind_stack: list[str] = []

        # Yield tracking state — None means tracking is inactive (outside loops/ifs)
        self.current_yield_vars: list[str] | None = None

        self.current_yield_types: dict[str, ir.Type] | None = None


class InlineState:
    """Inline expansion state read by return parsing and restored after expansion."""

    def __init__(self) -> None:
        # Inline function expansion state
        self.inline_mode = False

        self.inline_return_expr: ir.Expr | None = None


class ProgramBindings:
    """Program-wide names and identities passed through from the decorator."""

    def __init__(
        self,
        *,
        global_vars: dict[str, ir.GlobalVar] | None,
        gvar_to_func: dict[ir.GlobalVar, ir.Function] | None,
        buffer_name_meta: dict[tuple[str, str], dict[str, Any]] | None,
        alloc_window_buffer_names: set[str] | None,
    ) -> None:
        self.global_vars = global_vars or {}  # Track GlobalVars for cross-function calls

        self.gvar_to_func = gvar_to_func or {}  # Track parsed functions for type inference

        self.external_funcs: dict[str, ir.Function] = {}  # Track external functions referenced

        # Secondary index:
        # (func_name, buffer_name) → metadata (for cross-function import_peer_buffer resolution)
        # Shared across parser instances when parsing a @pl.program with multiple functions.
        self.buffer_name_meta: dict[tuple[str, str], dict[str, Any]] = (
            buffer_name_meta if buffer_name_meta is not None else {}
        )

        # Declared pld.tensor.alloc_window_buffer names — globally unique across
        # all functions in a @pl.program (the decorator shares this set).
        self.alloc_window_buffer_names: set[str] = (
            alloc_window_buffer_names if alloc_window_buffer_names is not None else set()
        )
