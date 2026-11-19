# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Resolve explicit squeeze-store state provenance into canonical memory operands."""

import ast
import copy
from typing import cast

from pypto.pypto_core import ir

from ..op.reg_ops import _SQUEEZE_STORE_STATE_OPS
from .diagnostics import ParserSyntaxError


def expand_squeeze_state_ops(body: list[ast.stmt], span: ir.Span) -> list[ast.stmt]:
    """Bind each state name to its init destination before ordinary SSA conversion.

    The public API carries only state/value/mask. Canonical step and flush also
    name the destination as an operand so alias analysis, DCE and memory planning
    can see the write without consulting mutable frontend objects or type fields.
    This mapping resolves provenance, not liveness: the IR verifier checks
    consumption order, loop carries, missing flush and destination rebinding.
    """
    destinations: dict[str, ast.Name] = {}

    def public_op(node: ast.AST) -> str | None:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "reg"
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "pl"
        ):
            for name in ("init", "step", "flush"):
                if node.func.attr == f"squeeze_store_{name}":
                    return name
        return None

    def arguments(call: ast.Call, names: tuple[str, ...], required: int) -> list[ast.expr]:
        if len(call.args) > len(names):
            raise ParserSyntaxError(f"Too many arguments for {ast.unparse(call.func)}", span=span)
        bound = dict(zip(names, call.args))
        for kw in call.keywords:
            if kw.arg not in names or kw.arg in bound:
                raise ParserSyntaxError(
                    f"Invalid or duplicate argument {kw.arg!r} for squeeze store", span=span
                )
            bound[kw.arg] = kw.value
        if any(name not in bound for name in names[:required]):
            raise ParserSyntaxError(f"Expected {', '.join(names)} for {ast.unparse(call.func)}", span=span)
        return [bound.get(name, ast.Constant(0)) for name in names]

    def destination(state: ast.expr) -> ast.Name:
        if not isinstance(state, ast.Name) or state.id not in destinations:
            raise ParserSyntaxError(
                "Squeeze-store state must be a named result of squeeze_store_init/step", span=span
            )
        return copy.deepcopy(destinations[state.id])

    def lower(call: ast.Call, name: str) -> tuple[ast.Call, ast.Name]:
        if name == "init":
            args = arguments(call, ("dst", "offset"), 1)
            dst = args[0]
            if not isinstance(dst, ast.Name):
                raise ParserSyntaxError("squeeze_store_init requires a named destination Tile", span=span)
        elif name == "step":
            args = arguments(call, ("state", "value", "mask"), 3)
            dst = destination(args[0])
            args.insert(1, dst)
        else:
            args = arguments(call, ("state",), 1)
            dst = destination(args[0])
            args.append(dst)
        rewritten = copy.deepcopy(call)
        assert isinstance(rewritten.func, ast.Attribute)
        rewritten.func.attr = _SQUEEZE_STORE_STATE_OPS[name].__name__
        rewritten.args = args
        rewritten.keywords = []
        return rewritten, dst

    def reject_state_copy(value: ast.expr | None) -> None:
        if isinstance(value, ast.Name) and value.id in destinations:
            raise ParserSyntaxError(
                "Squeeze-store state cannot be copied; bind the successor returned by step", span=span
            )

    class States(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign) -> ast.AST:
            name = public_op(node.value)
            if name is not None:
                if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                    raise ParserSyntaxError("Bind squeeze-store operations to a single name", span=span)
                assert isinstance(node.value, ast.Call)
                node.value, dst = lower(node.value, name)
                if name != "flush":
                    destinations[node.targets[0].id] = dst
                return node
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "yield_"
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "pl"
                and len(node.targets) == 1
            ):
                targets = node.targets[0].elts if isinstance(node.targets[0], ast.Tuple) else node.targets
                for target, source in zip(targets, node.value.args):
                    if isinstance(target, ast.Name) and isinstance(source, ast.Name):
                        if source.id in destinations:
                            destinations[target.id] = destinations[source.id]
            reject_state_copy(node.value)
            return self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
            if node.value is not None and (name := public_op(node.value)) is not None:
                if not isinstance(node.target, ast.Name):
                    raise ParserSyntaxError("Bind squeeze-store operations to a single name", span=span)
                assert isinstance(node.value, ast.Call)
                node.value, dst = lower(node.value, name)
                if name != "flush":
                    destinations[node.target.id] = dst
                return node
            reject_state_copy(node.value)
            return self.generic_visit(node)

        def visit_Expr(self, node: ast.Expr) -> ast.AST:
            name = public_op(node.value)
            if name is None:
                return self.generic_visit(node)
            if name != "flush":
                raise ParserSyntaxError("Assign the returned squeeze-store state", span=span)
            assert isinstance(node.value, ast.Call)
            call, _ = lower(node.value, name)
            # Register writebacks use named results internally, even when the
            # public flush statement discards the destination alias.
            result = f"_pypto_squeeze_flush_{node.lineno}_{node.col_offset}"
            return ast.copy_location(
                ast.Assign(targets=[ast.Name(id=result, ctx=ast.Store())], value=call), node
            )

        def visit_For(self, node: ast.For) -> ast.AST:
            # Explicit init_values syntax can rename the state at the loop
            # header. Simple loops need no special handling; ConvertToSSA adds
            # carries for the names rebound by step.
            if isinstance(node.iter, ast.Call) and isinstance(node.target, ast.Tuple):
                for kw in node.iter.keywords:
                    if kw.arg != "init_values" or not isinstance(kw.value, (ast.Tuple, ast.List)):
                        continue
                    if len(node.target.elts) == 2 and isinstance(node.target.elts[1], ast.Tuple):
                        for target, initial in zip(node.target.elts[1].elts, kw.value.elts):
                            if isinstance(target, ast.Name) and isinstance(initial, ast.Name):
                                if initial.id in destinations:
                                    destinations[target.id] = destinations[initial.id]
            return self.generic_visit(node)

    result = [cast(ast.stmt, States().visit(copy.deepcopy(node))) for node in body]
    for node in result:
        ast.fix_missing_locations(node)
    return result
