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
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from .._call_syntax import SUBMIT_DISPATCH_KEYWORDS, submit_target
from ._alias_analysis import AliasTracker, _AliasRoots, _AliasValue
from .normalization import CallPlan


@dataclass(frozen=True)
class CallBinding:
    plan: CallPlan
    public_tuple: bool = False
    # None preserves the legacy ABI's alias policy. Provider contracts must
    # explicitly permit any known overlap involving a written parameter.
    alias_permissions: frozenset[tuple[int, int]] | None = None


def _assign(name: str, value: ast.expr, origin: ast.AST) -> ast.Assign:
    return ast.copy_location(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=value), origin)


def _bind_submit_targets(
    statement: ast.stmt,
    value: ast.Call,
    plan: CallPlan,
    occupied: set[str],
    aliases: AliasTracker,
    public_roots: Sequence[_AliasValue],
) -> tuple[ast.Assign, list[ast.expr], list[ast.Assign]] | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    if not isinstance(target, ast.Tuple):
        return None
    if len(target.elts) != 2:
        raise ValueError("An external submission must bind (result, task_id)")
    public, tid = target.elts
    if not isinstance(tid, ast.Name):
        raise ValueError("An external submission must bind a TaskId variable")
    targets = public.elts if isinstance(public, ast.Tuple) else [public]
    if len(targets) != len(plan.public_slots) or any(not isinstance(x, ast.Name) for x in targets):
        raise ValueError("External submission targets must match the public return arity")

    result_names = [aliases.fresh() for _ in plan.result_parameters]
    selected: dict[int, str] = {}
    names = [item.id for item in targets if isinstance(item, ast.Name)]
    counts = Counter(names)
    for slot, name in zip(plan.public_slots, names, strict=True):
        # Keep result names on the Submit so the parser can validate their
        # producer dependencies. Delay writes to aliased assignment targets.
        if slot not in selected and name not in occupied and counts[name] == 1 and name != tid.id:
            selected[slot] = name
            result_names[slot] = name
    late = [
        _assign(name, ast.Name(id=result_names[slot], ctx=ast.Load()), statement)
        for slot, name in zip(plan.public_slots, names, strict=True)
        if name != selected.get(slot)
    ]
    task = tid.id if tid.id not in occupied and tid.id not in counts else aliases.fresh()
    if task != tid.id:
        late.append(_assign(tid.id, ast.Name(id=task, ctx=ast.Load()), statement))
    result_root = tuple(public_roots) if isinstance(public, ast.Tuple) else public_roots[0]
    aliases.record_assignment(statement, [result_root, None])
    outputs: list[ast.expr] = [ast.Name(id=name, ctx=ast.Store()) for name in result_names]
    lowered_target = ast.Tuple(
        elts=[ast.Tuple(elts=outputs, ctx=ast.Store()), ast.Name(id=task, ctx=ast.Store())],
        ctx=ast.Store(),
    )
    lowered = ast.copy_location(ast.Assign(targets=[lowered_target], value=value), statement)
    return lowered, [ast.Name(id=name, ctx=ast.Load()) for name in result_names], late


def _lower_call(  # noqa: PLR0912
    statement: ast.stmt,
    value: ast.Call,
    binding: CallBinding,
    operands: ast.Call,
    aliases: AliasTracker,
    reject_nested: Callable[[ast.AST], None],
) -> list[ast.stmt]:
    result: list[ast.stmt] = []
    fresh = aliases.fresh
    plan = binding.plan
    submitted = value is not operands
    if operands.keywords or len(operands.args) != len(plan.parameter_directions):
        raise ValueError(f"External call {ast.unparse(operands.func)!r} has unbound arguments")
    for operand in [*operands.args, *(kw.value for kw in value.keywords)]:
        reject_nested(operand)
    aliases.check_aliases(operands, binding)
    roots = [aliases.alias_of(operands.args[parameter]) for parameter in plan.result_parameters]
    public_roots = [roots[slot] for slot in plan.public_slots]
    if not submitted and not plan.writeback_slots and plan.public_slots == (0,) and not binding.public_tuple:
        aliases.record_assignment(statement, public_roots)
        return [statement]

    # Freeze arguments before any update. A view expression cannot be
    # used as an assignment target and must not be evaluated twice.
    arguments: list[ast.expr] = []
    for argument in operands.args:
        if isinstance(argument, (ast.Name, ast.Constant)):
            arguments.append(argument)
        else:
            name = fresh()
            result.append(_assign(name, argument, statement))
            arguments.append(ast.Name(id=name, ctx=ast.Load()))

    value.args = [value.args[0], *arguments] if submitted else arguments
    occupied = {
        arg.id for parameter, _ in plan.writeback_slots if isinstance(arg := arguments[parameter], ast.Name)
    }
    unpack = (
        _bind_submit_targets(statement, value, plan, occupied, aliases, public_roots) if submitted else None
    )
    late: list[ast.Assign] = []
    temporary: str | None = None
    if unpack is not None:
        launch, projections, late = unpack
        result.append(launch)
    elif submitted or plan.result_parameters:
        temporary = fresh()
        result.append(_assign(temporary, value, statement))

        def projection(slot: int, name: str = temporary) -> ast.expr:
            expr: ast.expr = ast.Name(id=name, ctx=ast.Load())
            if submitted or len(plan.result_parameters) > 1:
                expr = ast.Subscript(value=expr, slice=ast.Constant(slot), ctx=ast.Load())
            return expr

        projections = [projection(slot) for slot in range(len(plan.result_parameters) + int(submitted))]
    else:
        result.append(ast.copy_location(ast.Expr(value=value), statement))
        projections = []

    for parameter, slot in plan.writeback_slots:
        target = arguments[parameter]
        if not isinstance(target, ast.Name):
            raise ValueError(
                f"External tensor writeback requires a tensor argument, got {ast.unparse(target)}"
            )
        result.append(_assign(target.id, projections[slot], statement))

    if unpack is not None:
        return [*result, *late]

    if submitted:
        if not isinstance(statement, ast.Expr):
            assert isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Return))
            assert temporary is not None
            public = [projections[slot] for slot in plan.public_slots]
            statement.value = ast.Tuple(elts=[*public, projections[-1]], ctx=ast.Load())
            aliases.bind(temporary, (*roots, None))
            aliases.record_assignment(statement)
            result.append(statement)
        return result

    aliases.record_assignment(statement, public_roots)
    public = [projections[slot] for slot in plan.public_slots]
    if not isinstance(statement, ast.Expr) and public:
        assert isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Return))
        statement.value = ast.Tuple(elts=public, ctx=ast.Load()) if binding.public_tuple else public[0]
        result.append(statement)
    elif isinstance(statement, ast.Return):
        result.append(ast.copy_location(ast.Return(value=None), statement))
    elif not isinstance(statement, ast.Expr):
        raise ValueError("A void external call has no public value to assign")
    return result


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

    def binding_for(node: ast.AST) -> tuple[CallBinding, ast.Call] | None:
        if not isinstance(node, ast.Call):
            return None
        target = submit_target(node)
        callee = target if target is not None else node.func
        if (
            isinstance(callee, ast.Attribute)
            and isinstance(callee.value, ast.Name)
            and callee.value.id == "self"
            and (binding := bindings.get(callee.attr)) is not None
        ):
            operands = (
                node
                if target is None
                else ast.Call(
                    func=callee,
                    args=node.args[1:],
                    keywords=[kw for kw in node.keywords if kw.arg not in SUBMIT_DISPATCH_KEYWORDS],
                )
            )
            return binding, operands
        return None

    def reject_nested(node: ast.AST) -> None:
        for child in ast.walk(node):
            if binding_for(child) is not None:
                raise ValueError(
                    "External calls must be standalone statements or the complete value of an "
                    "assignment/return; bind the call result before using it in a nested expression "
                    "or control-flow header"
                )

    def block(statements: list[ast.stmt]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for statement in statements:
            value = (
                statement.value
                if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Return, ast.Expr))
                else None
            )
            match = binding_for(value) if value is not None else None
            if match is None:
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
            assert isinstance(value, ast.Call)
            result.extend(_lower_call(statement, value, *match, aliases, reject_nested))

        return result

    return block(copy.deepcopy(body))
