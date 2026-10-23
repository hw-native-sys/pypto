# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Attribute values shared by function directives, calls, operators and loops."""

import ast
from typing import Any, cast

from pypto.pypto_core import ir

from .diagnostics import (
    ParserSyntaxError,
    ParserTypeError,
)
from .enum_utils import (
    extract_enum_value,
)
from .parser_state import (
    ParseCallbacks,
    ParserServices,
)
from .syntax_helpers import ENUM_VALUE_TYPES

# Call attrs that never appear inside a printed ``attrs={...}`` dict. On
# ``system.task_dummy`` they ride bespoke surfaces (``manual_dep_edges`` prints
# as ``deps=[...]``; ``dummy_task`` is re-derived from the op), and the printer
# rejects them on every other op. Accepting either from a generic attrs dict
# would build IR that cannot be printed back.
TASK_DUMMY_ONLY_ATTRS = frozenset({"manual_dep_edges", "dummy_task"})


class AttrParser:
    """Attribute values shared by function directives, calls, operators and loops."""

    def __init__(
        self,
        *,
        callbacks: ParseCallbacks,
        services: ParserServices,
    ) -> None:
        self.callbacks = callbacks
        self.services = services

    def parse_attrs_dict(self, node: ast.Dict) -> dict[str, object]:
        """Parse an attrs dict literal from AST.

        Supports string keys with values that are:
        - Integer/float/bool/string constants
        """
        # Map of known enum attr keys to their (enum_map, enum_name, qualified) configs
        _ENUM_ATTRS: dict[str, tuple[dict[str, object], str, str]] = {}

        result: dict[str, object] = {}
        for key_node, value_node in zip(node.keys, node.values):
            if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                raise ParserSyntaxError(
                    "attrs keys must be string literals",
                    span=self.services.span_tracker.get_span(key_node) if key_node else None,
                )
            key = key_node.value

            if key in _ENUM_ATTRS:
                enum_map, enum_name, qualified = _ENUM_ATTRS[key]
                result[key] = extract_enum_value(value_node, enum_map, enum_name, qualified)
            elif isinstance(value_node, ast.Constant):
                result[key] = value_node.value
            else:
                raise ParserSyntaxError(
                    f"Unsupported value type for attrs key '{key}'",
                    span=self.services.span_tracker.get_span(value_node),
                    hint="Supported values: integer, float, bool, or string",
                )
        return result

    def get_call_attrs_dict(
        self,
        method_name: str,
        keywords: list[ast.keyword],
        span: ir.Span,
    ) -> ast.Dict | None:
        attrs_kw: ast.keyword | None = next((kw for kw in keywords if kw.arg == "attrs"), None)
        if attrs_kw is None:
            return None

        if not isinstance(attrs_kw.value, ast.Dict):
            raise ParserTypeError(
                f"'attrs=' on call to '{method_name}' must be a dict literal",
                span=self.services.span_tracker.get_span(attrs_kw.value),
                hint='e.g. attrs={"arg_directions": [pl.adir.input, ...]}',
            )
        # No key allowlist: every attrs key round-trips. The well-known keys
        # (arg_directions / manual_dep_edges / dump_vars) keep dedicated
        # extractors; every other key is recovered generically by
        # ``_extract_generic_call_attrs`` -> ``_parse_attr_value`` (the printer's
        # ``PrintAttrValue`` is the matching writer). Only the string-literal-key
        # invariant is enforced here.
        for key_node in attrs_kw.value.keys:
            if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
                raise ParserSyntaxError(
                    f"'attrs=' on call to '{method_name}' must use string-literal keys",
                    span=self.services.span_tracker.get_span(key_node) if key_node else span,
                )
        return attrs_kw.value

    def parse_attr_value(self, method_name: str, key: str, value_node: ast.expr) -> Any:
        """Reconstruct one generic ``attrs={...}`` value from its Python AST node.

        The matching writer is ``PrintAttrValue`` in the C++ printer
        (``src/ir/transforms/python_printer.cpp``). The value type is inferred
        from syntax (the "syntax inference only" round-trip contract): scalars
        from constants, ``[int, ...]`` -> index list, ``[pl.adir.X, ...]`` ->
        direction list, printed enum attributes -> their original enum values,
        and ``[name, ...]`` / bare ``name`` -> Var(s). Anything else becomes an
        IR expression. Syntactically ambiguous shapes (empty / mixed-kind lists)
        are REJECTED rather than guessed — never silently dropped — mirroring
        the printer's fail-loud behaviour.
        """
        from pypto.language.arg_direction import NAME_TO_DIRECTION  # noqa: PLC0415

        node_span = self.services.span_tracker.get_span(value_node)
        if (
            isinstance(value_node, ast.UnaryOp)
            and isinstance(value_node.op, (ast.USub, ast.UAdd))
            and isinstance(value_node.operand, ast.Constant)
            and type(value_node.operand.value) in (int, float)
        ):
            # A sign is a UnaryOp in Python's AST, even for a scalar literal.
            # Explicit pl.const(...) still takes the IR-expression path below.
            value = cast(int | float, value_node.operand.value)
            return -value if isinstance(value_node.op, ast.USub) else value
        if isinstance(value_node, ast.Constant):
            v = value_node.value
            # bool is a subclass of int, so this also covers True/False.
            if isinstance(v, (bool, int, float, str)):
                return v
            raise ParserSyntaxError(
                f"attrs['{key}'] on call to '{method_name}': unsupported constant "
                f"'{ast.unparse(value_node)}'",
                span=node_span,
            )
        if isinstance(value_node, (ast.List, ast.Tuple)):
            elts = list(value_node.elts)
            if not elts:
                # Empty list: element type is not inferable from syntax, but the
                # binding (ConvertKwargsDict) types it by attr key (e.g.
                # arg_direction_overrides -> vector<int32_t>, the Var-list keys
                # -> vector<VarPtr>). Return [] so a printed empty vector attr
                # round-trips instead of raising.
                return []
            # All integer literals -> index list (e.g. arg_direction_overrides).
            if all(
                isinstance(e, ast.Constant) and isinstance(e.value, int) and not isinstance(e.value, bool)
                for e in elts
            ):
                return [e.value for e in elts]  # type: ignore[attr-defined]
            # All pl.adir.<name> -> ArgDirection list.
            if all(
                isinstance(e, ast.Attribute)
                and isinstance(e.value, ast.Attribute)
                and e.value.attr == "adir"
                and isinstance(e.value.value, ast.Name)
                and e.value.value.id == "pl"
                for e in elts
            ):
                dirs: list[ir.ArgDirection] = []
                for e in elts:
                    assert isinstance(e, ast.Attribute)
                    if e.attr not in NAME_TO_DIRECTION:
                        raise ParserSyntaxError(
                            f"attrs['{key}'] on call to '{method_name}': unknown direction "
                            f"'pl.adir.{e.attr}'",
                            span=self.services.span_tracker.get_span(e),
                        )
                    dirs.append(NAME_TO_DIRECTION[e.attr])
                return dirs
            # All bare names -> Var list (resolved in the current scope).
            if all(isinstance(e, ast.Name) for e in elts):
                return [self.callbacks.expression(e) for e in elts]
            # All 2-tuples -> the cache-policy pair lists: ``[(idx, policy), ...]``
            # for the outlined-function attr and ``[(name, policy), ...]`` for the
            # scope attr. Both halves are recovered by the same syntax rules used
            # above, so the pair shape alone distinguishes them from every other
            # list attr. Written by PrintAttrValue's matching arms.
            pair_nodes = [e for e in elts if isinstance(e, ast.Tuple) and len(e.elts) == 2]  # noqa: PLR2004
            if len(pair_nodes) == len(elts):
                return [self.parse_attr_pair(method_name, key, e) for e in pair_nodes]
            raise ParserSyntaxError(
                f"attrs['{key}'] on call to '{method_name}': list elements are mixed or of an "
                "unsupported kind (expected all ints, all pl.adir.<name>, all names, or all "
                "(name_or_int, int) pairs)",
                span=node_span,
            )
        # Printed enum attrs must remain enum attrs. ``parse_expression`` either
        # rejects these standalone attribute forms or, for MemorySpace, lowers
        # them to ConstInt and silently changes the stored attr type.
        if isinstance(value_node, ast.Attribute):
            success, value = self.services.expr_evaluator.try_eval_expr(value_node)
            if success and isinstance(value, ENUM_VALUE_TYPES):
                return value

            # Keep the dtype resolver's diagnostic for unknown ``pl.<DTYPE>``
            # spellings when expression evaluation cannot resolve the value.
            try:
                return self.services.type_resolver.resolve_dtype(value_node)
            except ParserTypeError:
                pass
        # Bare name -> Var; any other expression -> the parsed IR expression.
        return self.callbacks.expression(value_node)

    def parse_attr_pair(self, method_name: str, key: str, node: ast.Tuple) -> "tuple[Any, int]":
        """Reconstruct one ``(reference, int)`` attr tuple from its AST node.

        The pair shape backs the cache-policy carriers: ``(param_index, policy)``
        on an outlined Function and ``(tensor_name, policy)`` on a ScopeStmt.
        The second half is always a plain int (the ``CachePolicy`` value); the
        first is either an int index or a bare name resolved to its Var.
        """
        first_node, second_node = node.elts
        if not (
            isinstance(second_node, ast.Constant)
            and isinstance(second_node.value, int)
            and not isinstance(second_node.value, bool)
        ):
            raise ParserSyntaxError(
                f"attrs['{key}'] on call to '{method_name}': the second element of a pair must be "
                "an integer literal",
                span=self.services.span_tracker.get_span(second_node),
            )
        if isinstance(first_node, ast.Constant) and isinstance(first_node.value, int):
            return first_node.value, second_node.value
        if isinstance(first_node, ast.Name):
            return self.callbacks.expression(first_node), second_node.value
        raise ParserSyntaxError(
            f"attrs['{key}'] on call to '{method_name}': the first element of a pair must be an "
            "integer literal or a bare variable name",
            span=self.services.span_tracker.get_span(first_node),
        )

    def extract_generic_call_attrs(
        self,
        method_name: str,
        keywords: list[ast.keyword],
        span: ir.Span,
    ) -> dict[str, Any]:
        """Recover every non-bespoke ``attrs={...}`` key via ``_parse_attr_value``.

        The well-known keys (arg_directions / manual_dep_edges / dump_vars) keep
        their dedicated extractors and are skipped here; this is the generic
        round-trip path for everything else (arg_direction_overrides,
        dummy_task, and any future attr the printer emits via ``PrintAttrValue``).
        """
        attrs_dict = self.get_call_attrs_dict(method_name, keywords, span)
        if attrs_dict is None:
            return {}
        bespoke = {"arg_directions", "manual_dep_edges", "dump_vars"}
        result: dict[str, Any] = {}
        for key_node, value_node in zip(attrs_dict.keys, attrs_dict.values):
            # _get_call_attrs_dict already validated string-literal keys.
            assert isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)
            key = key_node.value
            if key in bespoke:
                continue
            result[key] = self.parse_attr_value(method_name, key, value_node)
        return result

    def parse_op_attrs(self, call: ast.Call) -> dict[str, object] | None:
        """Extract a generic ``attrs={...}`` kwarg from an op call, if present.

        The python printer surfaces compiler-internal op-call attrs (e.g.
        ``pipeline_membership``) as a trailing ``attrs={...}`` dict. The op DSL
        wrappers / IR builders take no attrs parameter, so the dispatch helpers
        parse it here and re-attach it via ``ir.set_call_attrs`` after building
        the call. Returns ``None`` when no ``attrs=`` kwarg is present.

        Values are read by ``_parse_attr_value`` — the same open-world reader the
        GlobalVar-call / Submit paths use — so every type the printer's
        ``PrintAttrValue`` can emit round-trips. No key allowlist: the writer
        (``print_serialized_attrs``) is a denylist, so a key without a bespoke
        surface must be recoverable here or the round-trip silently loses it.
        """
        for keyword in call.keywords:
            if keyword.arg != "attrs":
                continue
            if not isinstance(keyword.value, ast.Dict):
                raise ParserSyntaxError(
                    "op attrs must be a dict literal",
                    span=self.services.span_tracker.get_span(keyword.value),
                )
            return self.parse_generic_attrs_dict(ast.unparse(call.func), keyword.value)
        return None

    def parse_generic_attrs_dict(self, method_name: str, node: ast.Dict) -> dict[str, object]:
        """Read an open-world ``attrs={...}`` dict literal via ``_parse_attr_value``.

        Only the string-literal-key invariant is enforced; each value's type is
        inferred from syntax by ``_parse_attr_value``, whose matching writer is
        ``PrintAttrValue`` in the C++ python printer.
        """
        result: dict[str, object] = {}
        for key_node, value_node in zip(node.keys, node.values):
            if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
                raise ParserSyntaxError(
                    "attrs keys must be string literals",
                    span=self.services.span_tracker.get_span(key_node) if key_node else None,
                )
            key = key_node.value
            if key in TASK_DUMMY_ONLY_ATTRS:
                # The printer never emits these into ``attrs={...}``: on
                # ``system.task_dummy`` they ride the bespoke ``deps=`` surface
                # (and ``dummy_task`` is re-derived from the op), and on any
                # other op the printer rejects them outright. Accepting one here
                # would build IR that cannot be printed back.
                raise ParserSyntaxError(
                    f"attrs['{key}'] on call to '{method_name}' is not a writable attr; "
                    "manual dependency edges are written as deps=[...] on "
                    "pl.system.task_dummy or pl.submit",
                    span=self.services.span_tracker.get_span(value_node),
                )
            result[key] = self.parse_attr_value(method_name, key, value_node)
        return result

    @staticmethod
    def attach_op_attrs(result: ir.Expr, attrs: dict[str, object] | None) -> ir.Expr:
        """Re-attach parsed generic attrs to a freshly built op Call."""
        if attrs and isinstance(result, ir.Call):
            return ir.set_call_attrs(result, attrs)
        return result
