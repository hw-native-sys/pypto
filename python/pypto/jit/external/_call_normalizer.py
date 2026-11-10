# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Materialize external results and hidden writes before parsing and SSA conversion.

Each statement and expression is visited once. Calls are indexed by generated
function name. Control-flow bodies retain their evaluation sites; calls in lazy
expressions and control-flow headers are diagnosed rather than eagerly hoisted.
"""

import ast
import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from ._alias_analysis import AliasTracker, _AliasRoots
from .normalization import CallPlan


@dataclass(frozen=True)
class CallBinding:
    plan: CallPlan
    public_tuple: bool = False
    # None preserves the legacy ABI's alias policy. Provider contracts must
    # explicitly permit any known overlap involving a written parameter.
    alias_permissions: frozenset[tuple[int, int]] | None = None


def normalize_calls(
    body: list[ast.stmt], bindings: Mapping[str, CallBinding], parameter_names: list[str]
) -> list[ast.stmt]:
    """Lower statement-position calls, keeping public values separate from writes.

    Tensor arguments are evaluated once, including views. Their projected
    updates reuse the same storage; existing inline/SSA passes propagate those
    writes through aliases without leaking a helper's local name rebindings.
    """
    if not bindings:
        return body
    used = set(parameter_names)
    for statement in body:
        used.update(node.id for node in ast.walk(statement) if isinstance(node, ast.Name))
    serial = 0

    def fresh() -> str:
        nonlocal serial
        while True:
            name = f"_external_result_{serial}"
            serial += 1
            if name not in used:
                used.add(name)
                return name

    aliases = AliasTracker(parameter_names, fresh)

    def binding_for(node: ast.AST) -> CallBinding | None:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            return bindings.get(node.func.attr)
        return None

    def reject_nested(node: ast.AST) -> None:
        for child in ast.walk(node):
            if binding_for(child) is not None:
                raise ValueError(
                    "External calls must be standalone statements or the complete value of an "
                    "assignment/return; bind the call result before using it in a nested expression "
                    "or control-flow header"
                )

    def assign(name: str, value: ast.expr, origin: ast.AST) -> ast.Assign:
        return ast.copy_location(
            ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=value), origin
        )

    def lower_call(statement: ast.stmt, value: ast.AST | None, binding: CallBinding) -> list[ast.stmt]:
        result: list[ast.stmt] = []

        assert isinstance(value, ast.Call)
        plan = binding.plan
        if value.keywords or len(value.args) != len(plan.parameter_directions):
            raise ValueError(f"External call {ast.unparse(value.func)!r} has unbound arguments")
        for argument in value.args:
            reject_nested(argument)
        aliases.check_aliases(value, binding)
        if not plan.writeback_slots and plan.public_slots == (0,) and not binding.public_tuple:
            aliases.record_assignment(statement, [aliases.alias_of(value.args[plan.result_parameters[0]])])
            return [statement]
        public_roots = [
            aliases.alias_of(value.args[plan.result_parameters[slot]]) for slot in plan.public_slots
        ]
        aliases.record_assignment(statement, public_roots)
        # Freeze arguments before any update. A view expression cannot be
        # used as an assignment target and must not be evaluated twice.
        arguments: list[ast.expr] = []
        for argument in value.args:
            if isinstance(argument, (ast.Name, ast.Constant)):
                arguments.append(argument)
            else:
                name = fresh()
                result.append(assign(name, argument, statement))
                arguments.append(ast.Name(id=name, ctx=ast.Load()))
        value.args = arguments
        if not plan.result_parameters:
            result.append(ast.copy_location(ast.Expr(value=value), statement))
        else:
            temporary = fresh()
            result.append(assign(temporary, value, statement))

            def projection(slot: int) -> ast.expr:
                expr: ast.expr = ast.Name(id=temporary, ctx=ast.Load())
                if len(plan.result_parameters) > 1:
                    expr = ast.Subscript(value=expr, slice=ast.Constant(slot), ctx=ast.Load())
                return expr

            for parameter, slot in plan.writeback_slots:
                target = arguments[parameter]
                if not isinstance(target, ast.Name):
                    raise ValueError(
                        f"External tensor writeback requires a tensor argument, got {ast.unparse(target)}"
                    )
                result.append(assign(target.id, projection(slot), statement))
            public = [projection(slot) for slot in plan.public_slots]
            if not isinstance(statement, ast.Expr) and public:
                assert isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Return))
                statement.value = (
                    ast.Tuple(elts=public, ctx=ast.Load()) if binding.public_tuple else public[0]
                )
                result.append(statement)
                return result
        if isinstance(statement, ast.Return):
            result.append(ast.copy_location(ast.Return(value=None), statement))
        elif not isinstance(statement, ast.Expr):
            raise ValueError("A void external call has no public value to assign")
        return result

    def block(statements: list[ast.stmt]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for statement in statements:
            value = (
                statement.value
                if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Return, ast.Expr))
                else None
            )
            binding = binding_for(value) if value is not None else None
            if binding is None:
                # Recurse only through statement blocks, so nested blocks are
                # not rescanned when checking their parent's header.
                rebound: set[str] = set()
                for field, item in ast.iter_fields(statement):
                    if isinstance(item, list) and item and isinstance(item[0], ast.stmt):
                        # A plain dependency scope executes its body once. Its
                        # assignments retain provenance after the scope; SPMD
                        # and other potentially zero-trip contexts still join.
                        if isinstance(statement, ast.With) and all(
                            isinstance(context.context_expr, ast.Call)
                            and ast.unparse(context.context_expr.func) in {"pl.scope", "pl.manual_scope"}
                            for context in statement.items
                        ):
                            setattr(statement, field, block(cast(list[ast.stmt], item)))
                            continue
                        journal: dict[str, _AliasRoots] = {}
                        aliases.journals.append(journal)
                        setattr(statement, field, block(cast(list[ast.stmt], item)))
                        aliases.journals.pop()
                        rebound.update(journal)
                        # Each arm starts from the incoming environment.
                        aliases.aliases.update(journal)
                    elif isinstance(item, ast.AST):
                        reject_nested(item)
                    elif isinstance(item, list):
                        for element in cast(list[object], item):
                            if isinstance(element, ast.AST):
                                reject_nested(element)
                # A branch/loop may not execute. Do not claim an alias
                # relationship for a rebound name after its boundary.
                for name in rebound:
                    aliases.bind(name, None)
                result.append(statement)
                aliases.record_assignment(statement)
                continue
            result.extend(lower_call(statement, value, binding))

        return result

    return block(copy.deepcopy(body))
