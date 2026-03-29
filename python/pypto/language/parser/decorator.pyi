# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Typing facade for DSL decorators.

This stub intentionally models ``@pl.function``-decorated methods as permissive
bound callables. The DSL parser accepts literal arguments like ``2.0`` at call
sites and lowers them to IR constants, but plain Python type checking cannot
express that coercion on arbitrary decorated methods without broadening the
callable boundary.
"""

from ast import FunctionDef
from collections.abc import Callable
from typing import Any, Generic, TypeVar, overload

from pypto.pypto_core import ir

_R_co = TypeVar("_R_co", covariant=True)

class InlineFunction:
    """Stores AST and metadata for a function to be inlined at call sites."""

    name: str
    func_def: FunctionDef
    param_names: list[str]
    source_file: str
    source_lines: list[str]
    line_offset: int
    col_offset: int
    closure_vars: dict[str, Any]

class DSLFunction(ir.Function, Generic[_R_co]):
    """Typing-only view of a decorated DSL function.

    At module scope, decorated functions behave like ``ir.Function`` objects.
    When accessed as class attributes through an instance inside ``@pl.program``
    bodies, treat them as permissive callables so pyright does not reject DSL
    literal-to-scalar coercions at call sites. The return type of the decorated
    function is preserved even though the argument list is intentionally broad.
    """

    @overload
    def __get__(self, instance: None, owner: type[Any] | None = None) -> DSLFunction[_R_co]: ...
    @overload
    def __get__(self, instance: object, owner: type[Any] | None = None) -> Callable[..., _R_co]: ...
    def __call__(self, *args: Any, **kwargs: Any) -> _R_co: ...

@overload
def function(func: Callable[..., _R_co], /) -> DSLFunction[_R_co]: ...
@overload
def function(
    *,
    type: ir.FunctionType = ...,
    level: ir.Level | None = ...,
    role: ir.Role | None = ...,
    strict_ssa: bool = ...,
) -> Callable[[Callable[..., _R_co]], DSLFunction[_R_co]]: ...
def function(
    func: Callable[..., _R_co] | None = None,
    *,
    type: ir.FunctionType = ...,
    level: ir.Level | None = ...,
    role: ir.Role | None = ...,
    strict_ssa: bool = ...,
) -> DSLFunction[_R_co] | Callable[[Callable[..., _R_co]], DSLFunction[_R_co]]: ...
def inline(func: Callable[..., Any]) -> InlineFunction: ...
@overload
def program(cls: type[Any], /, *, strict_ssa: bool = ...) -> ir.Program: ...
@overload
def program(cls: None = None, /, *, strict_ssa: bool = ...) -> Callable[[type[Any]], ir.Program]: ...
def program(
    cls: type[Any] | None = None,
    *,
    strict_ssa: bool = ...,
) -> ir.Program | Callable[[type[Any]], ir.Program]: ...
