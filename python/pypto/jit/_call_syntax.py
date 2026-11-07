# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Call syntax shared without importing JIT specialization state."""

import ast

# Submit dispatch metadata belongs to the launch, never to the callee signature.
SUBMIT_DISPATCH_KEYWORDS = frozenset(
    {
        "deps",
        "dumps",
        "allow_early_resolve",
        "predicate",
        "timing_slot",
        "core_num",
        "sync_start",
        "device",
        "attrs",
    }
)


def dependency_call(node: ast.AST) -> ast.Call | None:
    """View a bare call or a DSL submission as a call to its JIT dependency.

    Keep the original AST immutable. A submission's first argument names the
    dependency; its remaining arguments bind the callee. Call-site identity and
    positions stay with the original node wherever analyses need source ordering.
    """
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        return node
    if (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pl"
        and node.func.attr in {"submit", "spmd_submit"}
        and node.args
        and isinstance(node.args[0], ast.Name)
    ):
        return ast.copy_location(
            ast.Call(
                func=node.args[0],
                args=node.args[1:],
                keywords=[kw for kw in node.keywords if kw.arg not in SUBMIT_DISPATCH_KEYWORDS],
            ),
            node,
        )
    return None


def call_operand(call: ast.Call, index: int | None, name: str) -> ast.expr | None:
    """The argument ``call`` binds to parameter ``name``.

    Args:
        call: The operator call
        index: The parameter's positional slot, or None when it is keyword-only
        name: The parameter's keyword name

    Returns:
        The argument expression, or None when the call does not supply it
    """
    if index is not None and len(call.args) > index:
        return call.args[index]
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def writeback_return_operand(call: ast.Call) -> ast.expr | None:
    """A store returns its exact destination; opaque calls never guess an Out."""
    if ast.unparse(call.func) not in {"pl.store", "pl.tile.store"}:
        return None
    return call_operand(call, 2, "output_tensor")
