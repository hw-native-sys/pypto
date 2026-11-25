# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Stateful AST body rewriting for JIT source specialization.

Each transformer owns one function's tensor metadata, renaming, and lexical
scope state. Metadata previews subclass the same transformer so their folding
and rebinding rules stay aligned with emitted source.
"""

import ast
from bisect import bisect_left, bisect_right
from collections import Counter, deque
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, TypeVar, cast

from pypto.language.diagnostics import ParserError
from pypto.pypto_core import DataType, Error, arith, ir

from . import _specialize_values as values
from ._call_syntax import (
    PLACEMENT_KEYWORDS,
    SUBMIT_DISPATCH_KEYWORDS,
    dependency_call,
    writeback_return_operand,
)
from ._specialize_types import DynDim, TensorMeta, UnpackMetaLookup, merge_tensor_metas

if TYPE_CHECKING:
    from .specializer import SpecializeContext

_BindingValue = TypeVar("_BindingValue")
# None denotes a write whose tensor type cannot be inferred by copying a name.
_TensorWrites = dict[str, set[str | ast.If | None]]


def _restore_bindings(
    current: dict[str, _BindingValue], before: dict[str, _BindingValue], names: set[str]
) -> None:
    for name in names:
        if name in before:
            current[name] = before[name]
        else:
            current.pop(name, None)


def _scope_call_name(value: ast.expr | None) -> str | None:
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
        return value.func.attr
    return None


def _with_scope_names(node: ast.With) -> set[str | None]:
    return {_scope_call_name(item.context_expr) for item in node.items}


def _constant_scalar(node: ast.expr) -> int | bool | None:
    """Resolve literal integer and boolean expressions using the DSL's semantics."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, int) else None
    if not isinstance(node, (ast.Compare, ast.BoolOp, ast.UnaryOp, ast.BinOp)):
        return None
    literal_nodes = (
        ast.Constant,
        ast.Compare,
        ast.BoolOp,
        ast.UnaryOp,
        ast.BinOp,
        ast.cmpop,
        ast.boolop,
        ast.unaryop,
        ast.operator,
    )
    if not all(isinstance(child, literal_nodes) for child in ast.walk(node)):
        return None

    from pypto.language.parser.ast_parser import ASTParser  # noqa: PLC0415

    # Names and calls would need the actual function scope. Let the normal
    # parser handle those conditions, and retain diagnostics for invalid DSL.
    try:
        expr = ASTParser("<JIT constant>", []).parse_expression(ast.fix_missing_locations(node))
        folded = arith.Analyzer().simplify(expr)
    except (ParserError, TypeError, ValueError, OverflowError):
        return None
    return folded.value if isinstance(folded, (ir.ConstInt, ir.ConstBool)) else None


def _constant_condition(node: ast.expr) -> bool | None:
    value = _constant_scalar(node)
    return value if isinstance(value, bool) else None


class _SourceStores:
    """Index lexical stores before rewriting a compound statement or its children."""

    def __init__(self) -> None:
        self.spans: dict[ast.stmt, tuple[int, int]] = {}
        self.names: list[str] = []
        self.sites: dict[str, list[int]] = {}

    def _index(self, node: ast.AST) -> None:
        start = len(self.names)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            self.sites.setdefault(node.id, []).append(start)
            self.names.append(node.id)
        for child in ast.iter_child_nodes(node):
            self._index(child)
        if isinstance(node, ast.stmt):
            self.spans[node] = start, len(self.names)

    def matching(self, node: ast.stmt, candidates: Collection[str]) -> set[str]:
        if not candidates:
            return set()
        if node not in self.spans:
            self._index(node)
        start, end = self.spans[node]
        if end - start < len(candidates):
            return {name for name in self.names[start:end] if name in candidates}
        result: set[str] = set()
        for name in candidates:
            if sites := self.sites.get(name):
                index = bisect_left(sites, start)
                if index < len(sites) and sites[index] < end:
                    result.add(name)
        return result


class BodyTransformer(ast.NodeTransformer):
    """Rewrites a JIT function body for use inside @pl.program.

    Transformations applied:
    - Remove ``param.bind_dynamic(...)`` statements.
    - Remove ``M = pl.dynamic(...)`` assignments (promoted to module level).
    - Replace ``a.shape`` with a tuple literal ``(128, 128)``.
    - Replace ``a.shape[i]`` with an integer literal.
    - Replace ``M, N = a.shape`` tuple-unpack with individual assignments.
    - Replace ``a.dtype`` with the pl.XXX dtype name.
    - Multi-function: rewrite bare ``dep_func(...)`` calls to ``self.dep_func(...)``.
    """

    def __init__(
        self,
        tensor_meta: dict[str, TensorMeta],
        dep_names: set[str],
        param_names: list[str] | None = None,
        initial_used_names: set[str] | None = None,
        py_globals: Mapping[str, Any] | None = None,
        callee_contexts: "Mapping[str, SpecializeContext] | None" = None,
        dep_func_names: dict[str, str] | None = None,
        constexpr_values: Mapping[str, str] | None = None,
        dep_call_targets: Mapping[int, str] | None = None,
        unpack_metas: UnpackMetaLookup | None = None,
    ) -> None:
        super().__init__()
        # Dimension reads on a formal follow its emitted parameter type. A
        # narrowed view can be passed under its parent's static allocation
        # shape. Inferred helper results still carry their descriptor extents.
        self._meta = {
            name: replace(meta, inexact_axes=frozenset(), runtime_dims={}) if meta.inexact_axes else meta
            for name, meta in tensor_meta.items()
        }
        self._constexpr = dict(constexpr_values or {})
        self._callee_contexts = callee_contexts or {}
        self._unpack_metas = unpack_metas
        self._reads_tensor_properties = True
        self._untyped_unpack_names: set[str] = set()
        self._operator_result_types: dict[int, TensorMeta] = {}
        self._single_store_names: set[str] = set()
        self._operator_shape_names: set[str] = set()
        self._untyped_shape_sources: set[str] = set()
        self._untyped_shape_reads: dict[ast.expr, tuple[str, int]] = {}
        self._dep_names = dep_names
        # Call name → generated function name, for deps this body reaches
        # under a different name (an aliased import). ``visit_Call`` resolves
        # through it so ``kern(...)`` becomes ``self.kernel(...)``; a name
        # absent from the map is its own generated name.
        self._dep_func_names = dep_func_names or {}
        # ``id(call node) → generated function name``. Consulted first, so two
        # call sites of one name that were compiled against different
        # ``pl.constexpr`` values reach their own generated function instead of
        # both resolving to whichever one the name happens to map to. Empty
        # unless some name really does reach more than one function.
        self._dep_call_targets = dict(dep_call_targets or {})
        # DynVar name → (anchor_param, anchor_dim_idx). ``visit_Name`` uses
        # this to rewrite runtime references like ``pl.create_tensor([M, ...])``
        # via ``_dyn_dim_expr`` so the annotation-only DynVar doesn't leak past
        # SSA conversion. Each DynVar anchors at the first parameter dim it's
        # bound to; the AST is built fresh per use site to avoid node sharing.
        self._dynvar_anchors: dict[str, tuple[str, int]] = {}
        for pname, meta in tensor_meta.items():
            for i, dim in enumerate(meta.shape):
                if isinstance(dim, DynDim) and dim.name not in self._dynvar_anchors:
                    self._dynvar_anchors[dim.name] = (pname, i)
        # Module-level globals from the originating function. Used by
        # ``visit_Name`` to inline imported int/float/bool constants
        # (e.g. ``BATCH = 16`` from a config module) at their use sites.
        self._py_globals = py_globals or {}
        # Maps local variable names (from `M, N = a.shape`) to their concrete
        # constant values when all dimensions are static.  Used by visit_Name
        # to inline constants and by visit_Assign to suppress the assignment.
        self._shape_inlined: dict[str, int] = {}
        self._proven_empty_loops: set[ast.For | ast.While] = set()
        # Alpha-renaming support: tracks how many times each local has been assigned
        # (so we can generate x_v1, x_v2, ... on rebindings).
        self._assign_count: dict[str, int] = {}
        # Maps local variable name → current (latest) renamed alias.
        # Empty until the variable is assigned a second time.
        self._var_renames: dict[str, str] = {}
        # Reverse map: generated alias → original user variable name.
        # Used to rewrite error messages so users see their original names.
        self._alias_to_original: dict[str, str] = {}
        self._conditional_inputs: dict[ast.stmt, set[str]] = {}
        self._static_arms: dict[ast.If, bool] = {}
        self._source_stores = _SourceStores()
        self._tensor_writes: dict[ast.stmt, _TensorWrites] = {}
        self._tensor_arms: dict[ast.If, tuple[_TensorWrites, _TensorWrites]] = {}
        self._emitted_writes: dict[ast.stmt, tuple[set[str], bool]] = {}
        # Tracks the scope depth at which each variable was first assigned.
        # Rebindings at a deeper scope (e.g. inside a for/with) are loop-carried
        # updates and must NOT be renamed.
        self._assign_depth: dict[str, int] = {}
        self._scope_depth: int = 0
        self._optional_depth = 0
        self._graph_captures: dict[str, str] | None = None
        self.graph_rebinds: dict[str, str | int] = {}
        self.has_scope_capture = False
        self._simt_depth: int = 0
        self._simt_runtime_names: set[str] = set()
        self._while_runtime_names: set[str] = set()
        # Pre-register function parameters as "already assigned at depth 0" so
        # that body assignments like `x = pl.load(x, ...)` are treated as
        # rebindings and get alpha-renamed to `x_v1`.
        for name in param_names or []:
            self._assign_count[name] = 1
            self._assign_depth[name] = 0
        # Track all names pre-defined by the user (params + all Store targets) to
        # avoid generating aliases that collide with user-defined variables.
        self._used_names: set[str] = (initial_used_names or set()) | set(param_names or [])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record_first_assign(self, var_name: str) -> None:
        """Record a first (or tuple-unpack) assignment so later rebindings can be renamed."""
        if var_name not in self._assign_count:
            self._assign_count[var_name] = 1
            self._assign_depth[var_name] = self._scope_depth
        self._used_names.add(var_name)

    def _fresh_alias(self, var_name: str) -> str:
        """Reserve a unique spelling without changing the current binding."""
        count = self._assign_count[var_name]
        new_name = f"{var_name}_v{count}"
        # Skip any candidate that collides with a user-defined name.
        while new_name in self._used_names:
            count += 1
            new_name = f"{var_name}_v{count}"
        self._assign_count[var_name] = count + 1
        self._alias_to_original[new_name] = var_name
        self._used_names.add(new_name)
        return new_name

    def _rebind(self, var_name: str) -> str:
        """Generate a fresh name for a rebinding of ``var_name``.

        Returns the new name and updates ``_var_renames`` so subsequent reads
        of ``var_name`` resolve to the new alias.  Skips candidate names that
        are already used by the user to avoid collisions.
        """
        new_name = self._fresh_alias(var_name)
        if self._graph_captures is not None:
            folded = self._active_shape_fold(var_name)
            previous = self._var_renames.get(var_name, var_name)
            self.graph_rebinds[new_name] = previous if folded is None else folded
        self._var_renames[var_name] = new_name
        return new_name

    @property
    def rename_map(self) -> dict[str, str]:
        """Return mapping from generated alias → original user variable name."""
        return dict(self._alias_to_original)

    def _known_tensor_name(self, name: str) -> str | None:
        """Resolve known metadata without falling back to a rebound parameter."""
        current = self._var_renames.get(name, name)
        if current == name and name in self._var_renames:
            return None
        return current if current in self._meta else None

    def _record_tensor_meta(self, name: str, meta: TensorMeta | None, *, reuses_name: bool = False) -> None:
        """Keep metadata per emitted SSA name; invalidate conflicting scoped writes."""
        if reuses_name:
            meta = merge_tensor_metas(self._meta.get(name), meta)
            if meta is None and name in self._meta and name not in self._alias_to_original:
                # Keep the original version's metadata for its ABI. A later
                # scoped write cannot reestablish a fact on the skipped path.
                self._var_renames[name] = name
                return
            if meta is not None and self._var_renames.get(name) == name:
                self._var_renames.pop(name)
        if meta is None:
            self._meta.pop(name, None)
        else:
            self._meta[name] = meta

    def _active_shape_fold(self, name: str) -> int | None:
        # A fold from a forgotten sibling binding can still be needed after
        # a constexpr branch, but cannot serve as this scope's predecessor.
        if name in self._assign_count and name not in self._var_renames:
            return self._shape_inlined.get(name)
        return None

    def _is_folded_graph_capture(self, name: str) -> bool:
        # A capture of a folded name has an alias for the TaskId but no
        # runtime predecessor for an assignment to reuse.
        alias = self._var_renames.get(name)
        return (
            self._graph_captures is not None
            and alias is not None
            and self._graph_captures.get(name) == alias
            and isinstance(self.graph_rebinds.get(alias), int)
        )

    def _needs_alpha_rename(self, name: str) -> bool:
        # A folded name has no IR binding for the Graph body to assign to.
        return self._scope_depth == self._assign_depth[name] or (
            self._graph_captures is not None
            and (self._active_shape_fold(name) is not None or self._is_folded_graph_capture(name))
        )

    def _bind_assign_name(self, target: ast.Name, value_meta: TensorMeta | None) -> ast.Name:
        """Bind a store target after the RHS has been visited."""
        name = target.id
        if self._simt_depth:
            self._shape_inlined.pop(name, None)
        if name not in self._assign_count:
            self._record_first_assign(name)
        elif self._needs_alpha_rename(name):
            target = ast.copy_location(ast.Name(id=self._rebind(name), ctx=ast.Store()), target)
        else:
            # Loop-carried writes must update the active outer binding.
            current = self._var_renames.get(name, name)
            self._record_tensor_meta(current, value_meta, reuses_name=True)
            return ast.copy_location(ast.Name(id=current, ctx=ast.Store()), target)
        self._record_tensor_meta(target.id, value_meta)
        return target

    def _visit_simple_assign(self, node: ast.Assign) -> ast.stmt:
        """Handle a single-target name assignment with alpha-renaming support."""
        source = self._known_tensor_name(node.value.id) if isinstance(node.value, ast.Name) else None
        value_meta = self._meta.get(source) if source is not None else None
        node.value = self.visit(node.value)
        node.targets[0] = self._bind_assign_name(cast(ast.Name, node.targets[0]), value_meta)
        return node

    def _visit_chained_assign(self, node: ast.Assign) -> list[ast.stmt]:
        """Evaluate a chain's RHS once before rebinding and copying to its targets."""
        unchanged: set[str] = set()
        if self._shape_inlined:
            _, unchanged, _ = self._shape_write_effects(node, set())
        value = node.value
        assignments: list[ast.stmt] = []
        for target in node.targets:
            name = cast(ast.Name, target).id
            if name in unchanged:
                continue
            assignment = ast.copy_location(ast.Assign(targets=[target], value=value), node)
            assignments.append(self._visit_simple_assign(assignment))
            self._shape_inlined.pop(name, None)
            if len(assignments) == 1:
                value = ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node.value)
        return assignments

    def _same_shape_value(self, name: str, value: ast.expr | None) -> bool:
        folded = self._active_shape_fold(name)
        if folded is None or value is None:
            return False
        constant = _constant_scalar(value)
        return type(constant) is int and folded == constant

    def _restore_shape_folds(self, folded: dict[str, int]) -> None:
        """Resume constant reads after all reaching paths restore a dimension."""
        for name, value in folded.items():
            self._var_renames.pop(name, None)
            self._shape_inlined[name] = value

    def _static_call_meta(self, call: ast.Call) -> TensorMeta | None:
        """Use the DSL operator's IR type rule for an otherwise unmodeled unpacked value."""
        import pypto.language as pl  # noqa: PLC0415
        from pypto.language.op import tensor_ops  # noqa: PLC0415
        from pypto.language.typing import Tensor  # noqa: PLC0415

        fn = call.func
        if not isinstance(fn, ast.Attribute) or fn.attr not in tensor_ops.__all__:
            return None
        namespace = fn.value
        if isinstance(namespace, ast.Attribute) and namespace.attr == "tensor":
            namespace = namespace.value
            operation = getattr(tensor_ops, fn.attr)
        else:
            operation = getattr(pl, fn.attr, None)
        if not isinstance(namespace, ast.Name) or namespace.id != "pl" or operation is None:
            return None

        symbols: dict[int, ir.Var] = {}
        dimensions: dict[int, DynDim] = {}

        def tensor_operand(name: str, meta: TensorMeta) -> Any:
            if self._var_renames.get(name) == name or meta.is_distributed or meta.inexact_axes:
                return NotImplemented
            shape: list[int | ir.Expr] = []
            for dim in meta.shape:
                if isinstance(dim, int):
                    shape.append(dim)
                else:
                    # Bounds are cache observations. Only the original symbol can
                    # prove that an output dimension still denotes this input axis.
                    if id(dim) not in symbols:
                        symbol = ir.Var(dim.name, ir.ScalarType(DataType.INDEX), ir.Span.unknown())
                        symbols[id(dim)] = symbol
                        dimensions[symbol.unique_id] = dim
                    shape.append(symbols[id(dim)])
            view = ir.TensorView([], meta.layout) if meta.layout is not None else None
            typ = ir.TensorType(shape, meta.dtype, tensor_view=view)
            return Tensor(expr=ir.Var(name, typ, ir.Span.unknown()))

        def operand(expr: ast.expr) -> Any:
            if isinstance(expr, ast.Name) and (meta := self._meta.get(expr.id)) is not None:
                return tensor_operand(expr.id, meta)
            if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name) and expr.value.id == "pl":
                value = getattr(pl, expr.attr, None)
                if isinstance(value, DataType):
                    return value
            if isinstance(expr, ast.Constant):
                return expr.value
            if isinstance(expr, (ast.List, ast.Tuple)) and all(
                isinstance(e, ast.Constant) for e in expr.elts
            ):
                return ast.literal_eval(expr)
            return NotImplemented

        if any(kw.arg is None for kw in call.keywords):
            return None
        try:
            args = [operand(arg) for arg in call.args]
            kwargs = {kw.arg: operand(kw.value) for kw in call.keywords if kw.arg is not None}
            values = [*args, *kwargs.values()]
            if any(v is NotImplemented for v in values) or not any(isinstance(v, Tensor) for v in values):
                return None
            result = operation(*args, **kwargs)
        except (
            TypeError,
            ValueError,
            RuntimeError,
            AttributeError,
            IndexError,
            ArithmeticError,
            AssertionError,
            Error,
        ):
            # Parsing the real call owns its diagnostics and handles runtime operands.
            return None
        if not isinstance(result, Tensor):
            return None
        result_type = result.unwrap().type
        if type(result_type) is not ir.TensorType:
            return None
        dims: list[int | DynDim] = []
        for extent in result_type.shape:
            if isinstance(extent, ir.ConstInt):
                dims.append(extent.value)
            elif isinstance(extent, ir.Var) and extent.unique_id in dimensions:
                dims.append(dimensions[extent.unique_id])
            else:
                return None
        layout = result_type.tensor_view.layout if result_type.tensor_view is not None else None
        return TensorMeta(tuple(dims), result_type.dtype, layout)

    def _unpack_tensor_metas(self, target: ast.expr, value: ast.expr) -> dict[ast.expr, TensorMeta]:
        """Read every RHS slot in the incoming scope, before any simultaneous stores."""
        if isinstance(target, ast.Name) and isinstance(value, ast.Name):
            meta = self._meta.get(value.id)
            if meta is not None and self._var_renames.get(value.id) != value.id:
                return {target: meta}
        if isinstance(target, (ast.Tuple, ast.List)):
            slots: dict[ast.expr, TensorMeta] = {}
            literal = isinstance(value, (ast.Tuple, ast.List)) and len(target.elts) == len(value.elts)
            if isinstance(value, (ast.Tuple, ast.List)) and literal:
                for elt, rhs in zip(target.elts, value.elts, strict=True):
                    slots.update(self._unpack_tensor_metas(elt, rhs))
                if all(isinstance(rhs, (ast.Name, ast.Constant)) for rhs in value.elts):
                    return slots
            if self._reads_tensor_properties and self._unpack_metas is not None and hasattr(target, "lineno"):
                metas = self._unpack_metas().get((target.lineno, target.col_offset))
                if metas is not None and len(metas) == len(target.elts):
                    for index, (elt, meta) in enumerate(zip(target.elts, metas, strict=True)):
                        if (
                            literal
                            and isinstance(value, (ast.Tuple, ast.List))
                            and isinstance(value.elts[index], ast.Name)
                        ):
                            continue
                        if isinstance(elt, ast.Name) and meta is not None:
                            slots.setdefault(elt, meta)
            if self._reads_tensor_properties and isinstance(value, (ast.Tuple, ast.List)) and literal:
                for elt, rhs in zip(target.elts, value.elts, strict=True):
                    if isinstance(elt, ast.Name) and elt not in slots and isinstance(rhs, ast.Call):
                        if (meta := self._static_call_meta(rhs)) is not None:
                            slots[elt] = meta
                            self._operator_result_types[id(meta)] = meta
            return slots
        return {}

    def _visit_unpack_target(
        self, target: ast.expr, value: ast.expr | None, tensors: Mapping[ast.expr, TensorMeta]
    ) -> ast.expr:
        """Bind tuple elements using the same scope policy as a simple assignment."""
        if isinstance(target, (ast.Tuple, ast.List)):
            if isinstance(value, (ast.Tuple, ast.List)) and len(target.elts) == len(value.elts):
                target.elts = [
                    self._visit_unpack_target(elt, rhs, tensors)
                    for elt, rhs in zip(target.elts, value.elts, strict=True)
                ]
            else:
                target.elts = [self._visit_unpack_target(elt, None, tensors) for elt in target.elts]
        elif isinstance(target, ast.Starred):
            target.value = self._visit_unpack_target(target.value, None, tensors)
        elif isinstance(target, ast.Name):
            # The RHS has already been visited in the incoming scope. Keep
            # the full unpack so sibling values and calls are evaluated once.
            if self._same_shape_value(target.id, value):
                return target
            meta = tensors.get(target)
            bound = self._bind_assign_name(target, meta)
            if meta is None:
                self._untyped_unpack_names.add(bound.id)
            self._shape_inlined.pop(target.id, None)
            return bound
        else:
            target = cast(ast.expr, self.visit(target))
        return target

    def _dyn_dim_expr(self, param_name: str, dim_idx: int) -> ast.expr:
        """Emit ``pl.tensor.dim(<param>, <dim_idx>)`` for a dynamic dim.

        Used by ``_shape_tuple_node`` / ``_shape_dim_node`` / the
        ``M, N = a.shape`` Case 2 emission so DynVar references never
        materialize as bare names in runtime expressions. Going through the
        anchor expression directly keeps these emitted nodes outside the
        scope where ``visit_Name``'s DynVar→anchor substitution would have
        rewritten them anyway, and avoids the "Variable 'M' used outside
        its defining scope" SSA error.
        """
        pl_tensor = ast.Attribute(
            value=ast.Name(id="pl", ctx=ast.Load()),
            attr="tensor",
            ctx=ast.Load(),
        )
        return ast.fix_missing_locations(
            ast.Call(
                func=ast.Attribute(value=pl_tensor, attr="dim", ctx=ast.Load()),
                args=[ast.Name(id=param_name, ctx=ast.Load()), ast.Constant(value=dim_idx)],
                keywords=[],
            )
        )

    def _shape_tuple_node(self, param_name: str) -> ast.Tuple:
        meta = self._meta[param_name]
        elts = [self._shape_dim_node(param_name, i) for i in range(len(meta.shape))]
        return ast.Tuple(elts=elts, ctx=ast.Load())

    def _shape_dim_node(self, param_name: str, dim_idx: int) -> ast.expr:
        meta = self._meta[param_name]
        axis = dim_idx if dim_idx >= 0 else len(meta.shape) + dim_idx
        if not 0 <= axis < len(meta.shape):
            raise IndexError(dim_idx)
        dim = meta.actual_dim(axis)
        return ast.Constant(value=dim) if isinstance(dim, int) else self._dyn_dim_expr(param_name, axis)

    # ------------------------------------------------------------------
    # Statement-level transforms
    # ------------------------------------------------------------------

    def visit_Expr(self, node: ast.Expr) -> ast.stmt | None:
        """Remove bind_dynamic(...) and dynvar assignment statements."""
        if isinstance(node.value, ast.Call):
            call = node.value
            func = call.func
            # param.bind_dynamic(dim, dynvar) → delete
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "bind_dynamic"
                and isinstance(func.value, ast.Name)
                and func.value.id in self._meta
            ):
                return None
        return cast("ast.stmt", self.generic_visit(node))

    def _needs_runtime_shape_write(self, name: str) -> bool:
        return (
            name in self._while_runtime_names
            or (self._simt_depth > 0 and name in self._simt_runtime_names)
            or (
                self._assign_depth.get(name, self._scope_depth) < self._optional_depth
                and self._active_shape_fold(name) is None
            )
        )

    def _expand_shape_unpacking(
        self, node: ast.Assign, param_name: str, target_names: list[ast.expr]
    ) -> list[ast.stmt]:
        """Expand known shape dimensions without introducing tuple values."""
        meta = self._meta[param_name]
        stmts: list[ast.stmt] = []
        for i, tgt in enumerate(target_names):
            dim = meta.actual_dim(i)
            if not isinstance(tgt, ast.Name):
                continue
            if tgt.id in self._var_renames or self._needs_runtime_shape_write(tgt.id):
                # Keep the current binding updated after scalar renaming or a
                # conditional write. Folding only the source name would leave
                # reads pointing at its old alias. Expand tuples into scalars
                # because a tuple temporary is illegal inside SIMT.
                value = self._shape_dim_node(param_name, i)
                assignment = ast.Assign(
                    targets=[tgt], value=value, lineno=node.lineno, col_offset=node.col_offset
                )
                stmts.append(self._visit_simple_assign(assignment))
            elif not isinstance(dim, int):
                # Dynamic dim: emit M = pl.tensor.dim(<param>, i) so the
                # DynVar name never appears as a runtime expression.
                lhs_name = tgt.id
                if lhs_name in self._assign_count:
                    lhs_name = self._rebind(lhs_name)
                else:
                    self._record_first_assign(tgt.id)
                stmts.append(
                    ast.Assign(
                        targets=[ast.Name(id=lhs_name, ctx=ast.Store())],
                        value=self._dyn_dim_expr(param_name, i),
                        lineno=node.lineno,
                        col_offset=node.col_offset,
                    )
                )
            else:
                # Static dim: inline constant, suppress assignment
                self._remember_operator_extent(tgt.id, meta)
                self._record_first_assign(tgt.id)
                if self._is_folded_graph_capture(tgt.id):
                    self._var_renames.pop(tgt.id)
                self._shape_inlined[tgt.id] = dim
        return stmts

    def _remember_operator_extent(self, name: str, meta: TensorMeta) -> None:
        # An extent introduced by the typed-unpack fallback may be copied
        # through stable locals. Ordinary scalar copies keep their existing
        # runtime representation, including when used in slice descriptors.
        if name in self._single_store_names and id(meta) in self._operator_result_types:
            self._operator_shape_names.add(name)

    def _assigned_shape_extent(self, target: str, value: ast.expr) -> int | None:
        if isinstance(value, ast.Name) and value.id in self._operator_shape_names:
            dim = self._active_shape_fold(value.id)
            if dim is not None and target in self._single_store_names:
                self._operator_shape_names.add(target)
            return dim
        if (
            isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Attribute)
            and value.value.attr == "shape"
            and isinstance(value.value.value, ast.Name)
            and (tensor := self._known_tensor_name(value.value.value.id)) is not None
            and isinstance(value.slice, ast.Constant)
            and isinstance(value.slice.value, int)
        ):
            extent = self._shape_dim_node(tensor, value.slice.value)
            if isinstance(extent, ast.Constant) and isinstance(extent.value, int):
                self._remember_operator_extent(target, self._meta[tensor])
                return extent.value
        return None

    def visit_Assign(self, node: ast.Assign) -> ast.stmt | list[ast.stmt] | None:
        """Handle special assignment patterns.

        1. ``M = pl.dynamic("M")`` → delete (promoted to module level).
        2. ``M, N = a.shape`` → expand into individual assignments.
        """
        # Case 1: M = pl.dynamic("M") → remove
        if isinstance(node.value, ast.Call):
            func = node.value.func
            is_pl_dynamic = (
                isinstance(func, ast.Attribute)
                and func.attr == "dynamic"
                and isinstance(func.value, ast.Name)
                and func.value.id == "pl"
            ) or (isinstance(func, ast.Name) and func.id == "dynamic")
            if is_pl_dynamic:
                anchor = next(
                    (
                        self._dynvar_anchors[target.id]
                        for target in node.targets
                        if isinstance(target, ast.Name) and target.id in self._dynvar_anchors
                    ),
                    None,
                )
                if len(node.targets) == 1 or anchor is None:
                    return None
                # The declaration is promoted, but its chained aliases still
                # need the same runtime value even when only one is anchored.
                node.value = ast.copy_location(self._dyn_dim_expr(*anchor), node.value)

        if len(node.targets) > 1 and all(isinstance(target, ast.Name) for target in node.targets):
            return self._visit_chained_assign(node)

        # Case 2: M, N = a.shape  →  individual assignments (dynamic) or inline (static)
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Tuple)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "shape"
            and isinstance(node.value.value, ast.Name)
            and (param_name := self._known_tensor_name(node.value.value.id)) is not None
        ):
            meta = self._meta[param_name]
            target_names = node.targets[0].elts
            if len(target_names) == len(meta.shape):
                return self._expand_shape_unpacking(node, param_name, target_names) or None

        # A copy of a folded extent has the same binding rules as a direct shape read.
        if (
            len(node.targets) == 1
            and isinstance(target := node.targets[0], ast.Name)
            and (target.id not in self._var_renames or self._is_folded_graph_capture(target.id))
            and not self._needs_runtime_shape_write(target.id)
            and (dim := self._assigned_shape_extent(target.id, node.value)) is not None
        ):
            self._var_renames.pop(target.id, None)
            self._shape_inlined[target.id] = dim
            self._record_first_assign(target.id)
            return None

        # Default: visit the RHS first (applies existing renames to operands),
        # then handle rebinding rename on the LHS.
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            return self._visit_simple_assign(node)

        if len(node.targets) == 1 and isinstance(node.targets[0], (ast.Tuple, ast.List)):
            folded = {}
            if self._shape_inlined:
                _, unchanged, _ = self._shape_write_effects(node, set())
                folded = {name: self._shape_inlined[name] for name in unchanged}
            node.value = self.visit(node.value)
            target = node.targets[0]
            node.targets[0] = self._visit_unpack_target(
                target, node.value, self._unpack_tensor_metas(target, node.value)
            )
            self._restore_shape_folds(folded)
            return node

        if self._while_runtime_names:
            for target in node.targets:
                self._retarget_while_store(target)
        return cast("ast.stmt", self.generic_visit(node))

    def _retarget_while_store(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name) and target.id in self._while_runtime_names:
            target.id = self._var_renames.get(target.id, target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._retarget_while_store(item)
        elif isinstance(target, ast.Starred):
            self._retarget_while_store(target.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.stmt | list[ast.stmt]:
        if not isinstance(node.target, ast.Name):
            return cast("ast.AugAssign", self.generic_visit(node))
        name = node.target.id
        node.value = self.visit(node.value)
        prefix = self._make_bridge_assignments([name]) if self._active_shape_fold(name) is not None else []
        node.target.id = self._var_renames.get(name, name)
        return [*prefix, node] if prefix else node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.stmt | None:
        """Handle annotated assignments (``x: T = value``) with alpha-renaming."""
        if node.value is None:
            return cast("ast.stmt", self.generic_visit(node))
        if not isinstance(node.target, ast.Name):
            return cast("ast.stmt", self.generic_visit(node))
        source = self._known_tensor_name(node.value.id) if isinstance(node.value, ast.Name) else None
        value_meta = self._meta.get(source) if source is not None else None
        node.value = self.visit(node.value)
        # Inline shape constants in the local annotation too (e.g. a body-level
        # ``x: pl.Tile[[1, W_PAD], pl.FP32]`` where ``W_PAD`` is a module-level
        # int). Without this the un-inlined name leaks into the generated source
        # and the parser rejects it ("Unknown shape variable: W_PAD").
        node.annotation = self.visit(node.annotation)
        node.target = self._bind_assign_name(node.target, value_meta)
        return cast("ast.stmt", node)

    # ------------------------------------------------------------------
    # Expression-level transforms
    # ------------------------------------------------------------------

    def visit_Name(self, node: ast.Name) -> ast.expr:
        """Replace inlined shape constants, renamed rebindings, DynVar runtime
        references, and free names bound to renderable values.

        Scalar *parameters* are deliberately not replaced: they are runtime
        values and must survive as real ``pl.Scalar`` parameters. A body that
        needs a compile-time constant either declares a ``pl.constexpr``
        parameter (folded below) or reads a module-level or closure name (folded
        by the free-name rule further down) — the two share one renderer, so
        they admit exactly the same values.
        """
        if isinstance(node.ctx, ast.Load):
            # Check active renames first — a rebinding supersedes any earlier inlining.
            if node.id in self._var_renames:
                return ast.Name(id=self._var_renames[node.id], ctx=ast.Load())
            # A constexpr parameter folds to its call-site value. This must
            # precede the ``_used_names`` guard below: a parameter is always a
            # local, so that guard would skip it and the name would survive into
            # the generated source as an undefined free variable.
            constexpr_text = self._constexpr.get(node.id)
            if constexpr_text is not None:
                folded = ast.parse(constexpr_text, mode="eval").body
                return ast.fix_missing_locations(ast.copy_location(folded, node))
            if node.id in self._shape_inlined:
                return ast.Constant(value=self._shape_inlined[node.id])
            # DynVar runtime references — e.g. pl.create_tensor([M, HIDDEN], ...).
            # Substitute M with pl.tensor.dim(P, k) where (P, k) is the first
            # parameter dim bound to M. This keeps the IR free of annotation-only
            # DynVar references in runtime expressions; without it ConvertToSSA
            # raises "Variable 'M' used outside its defining scope".
            #
            # Skipped when the name was assigned earlier in the body (a
            # ``M, N = a.shape`` Case 2 rebind, the ``M = pl.dynamic("M")``
            # assignment, or any user shadowing) — ``_used_names`` covers all
            # those cases. Otherwise the substitution fires for the bare
            # module-level DynVar reference.
            if node.id in self._dynvar_anchors and node.id not in self._used_names:
                pname, dim_idx = self._dynvar_anchors[node.id]
                return self._dyn_dim_expr(pname, dim_idx)
            # Free-name inlining: only when the name is not also a local (function
            # param or assigned-in-body) — those take priority. ``_py_globals``
            # is the defining function's own globals plus its closure cells, so a
            # constant from an enclosing factory folds the same as a module one.
            if node.id not in self._used_names:
                folded = values.fold_free_name(node.id, self._py_globals, node)
                if folded is not None:
                    return folded
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        """Replace a.shape → tuple and a.dtype → pl.XXX."""
        reference = values.math_constant_reference(node)
        if reference is not None and reference.split(".", 1)[0] not in self._used_names:
            folded = values.fold_free_name(reference, self._py_globals, node)
            if folded is not None:
                return folded
        if (
            isinstance(node.value, ast.Name)
            and (param_name := self._known_tensor_name(node.value.id)) is not None
        ):
            if node.attr == "shape":
                return self._shape_tuple_node(param_name)
            if node.attr == "dtype":
                dtype_s = values.dtype_str(self._meta[param_name].dtype)
                # Build attribute access: pl.FP32 → Attribute(Name("pl"), "FP32")
                parts = dtype_s.split(".")
                result: ast.expr = ast.Name(id=parts[0], ctx=ast.Load())
                for part in parts[1:]:
                    result = ast.Attribute(value=result, attr=part, ctx=ast.Load())
                return result
        if node.attr == "shape" and isinstance(node.value, ast.Name):
            current = self._var_renames.get(node.value.id, node.value.id)
            if current in self._untyped_unpack_names:
                self._untyped_shape_sources.add(current)
        return cast("ast.expr", self.generic_visit(node))

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        """Replace a.shape[i] → integer constant."""
        if (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "shape"
            and isinstance(node.value.value, ast.Name)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
        ):
            name = node.value.value.id
            current = self._known_tensor_name(name)
            if current is not None:
                return self._shape_dim_node(current, node.slice.value)
            if name in self._meta and name in self._var_renames:
                current = self._var_renames[name]
                read = self._dyn_dim_expr(current, node.slice.value)
                if current in self._untyped_unpack_names:
                    self._untyped_shape_sources.add(current)
                    self._untyped_shape_reads[read] = current, node.slice.value
                return read
        return cast("ast.expr", self.generic_visit(node))

    def visit_Call(self, node: ast.Call) -> ast.expr:
        """Rewrite dep_func(args) → self.dep_func(args) for multi-function JIT.

        The attribute is the dep's *generated* function name, which differs
        from the name at the call site when the body reaches the dep through
        an aliased import (``kern(...)`` → ``self.kernel(...)``) and when this
        site's ``pl.constexpr`` values select one of several compilations of
        the same dep (``helper(x, 16)`` → ``self.helper(...)`` while
        ``helper(x, 32)`` → ``self.helper__2(...)``). The per-site map is
        consulted first for exactly that reason.

        Keyword args are normalised to positional based on the dep's
        parameter order (so ``dep(a, out=out)`` becomes ``self.dep(a, out)``).
        Inter-function calls inside ``@pl.program`` only accept positional
        args — preserving keyword form would make the parser reject the call.
        """
        original = node
        normalized = dependency_call(node)
        if (
            normalized is not None
            and isinstance(normalized.func, ast.Name)
            and normalized.func.id in self._dep_names
        ):
            callee_name = normalized.func.id
            node = normalized
            dep_name = self._dep_call_targets.get(
                id(original), self._dep_func_names.get(callee_name, callee_name)
            )
            new_func = ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load()),
                attr=dep_name,
                ctx=ast.Load(),
            )
            is_submit = original is not normalized
            callee = self._callee_contexts.get(dep_name)
            # Only HOST entries discover chip orchestrators as JIT dependencies.
            if is_submit:
                dispatch_keys = SUBMIT_DISPATCH_KEYWORDS
            elif callee is not None and callee.func_type in (None, "orchestration"):
                dispatch_keys = PLACEMENT_KEYWORDS
            else:
                dispatch_keys = frozenset[str]()
            callee_keywords = [kw for kw in original.keywords if kw.arg not in dispatch_keys]
            if callee is not None and (callee_keywords or callee.constexpr_values):
                param_order = callee.param_names
                constexpr_params = callee.constexpr_values
                bound_args = [
                    (param_order[index] if index < len(param_order) else None, arg)
                    for index, arg in enumerate(node.args)
                ]
                kw_by_name = {kw.arg: kw.value for kw in callee_keywords if kw.arg is not None}
                # Append keyword-bound args in dep's parameter order, skipping
                # params already covered by positional args.
                for param in param_order[len(node.args) :]:
                    if param in kw_by_name:
                        bound_args.append((param, kw_by_name.pop(param)))
                # Any remaining keywords (e.g. **kwargs splats or unknown names)
                # are kept as-is so we don't silently drop them; the parser
                # will surface them as a clear error.
                remaining_kw = [kw for kw in callee_keywords if kw.arg is None or kw.arg in kw_by_name]
                if constexpr_params:
                    # Drop the callee's compile-time arguments. It resolved them
                    # during specialization and its generated signature no longer
                    # declares them, so passing one would be an arity mismatch.
                    # Preserve formal identities before compacting omitted Out
                    # slots: a later keyword constexpr is not the missing output.
                    bound_args = [(param, arg) for param, arg in bound_args if param not in constexpr_params]
                    remaining_kw = [kw for kw in remaining_kw if kw.arg not in constexpr_params]
                if is_submit:
                    # Submit can allocate a suffix of omitted outputs. The
                    # native ABI has no way to encode a hole in that prefix.
                    runtime_params = [param for param in param_order if param not in constexpr_params]
                    supplied = [param for param, _ in bound_args if param is not None]
                    if supplied != runtime_params[: len(supplied)]:
                        raise ValueError(
                            f"Submit to {callee_name!r} cannot skip a runtime argument before a supplied one"
                        )
                new_node = ast.Call(func=new_func, args=[arg for _, arg in bound_args], keywords=remaining_kw)
            else:
                new_node = ast.Call(func=new_func, args=node.args, keywords=callee_keywords)
            if is_submit:
                new_node = ast.Call(
                    func=original.func, args=[new_func, *new_node.args], keywords=new_node.keywords
                )
            new_node.keywords.extend(kw for kw in original.keywords if kw.arg in dispatch_keys)
            ast.copy_location(new_node, original)
            return cast("ast.expr", self.generic_visit(new_node))
        return cast("ast.expr", self.generic_visit(node))

    @classmethod
    def _terminates_suite(cls, stmt: ast.stmt) -> bool:
        if isinstance(stmt, (ast.Break, ast.Continue, ast.Return)):
            return True
        if isinstance(stmt, ast.If):
            condition = values.literal_condition(stmt.test)
            if condition is not None:
                return any(map(cls._terminates_suite, stmt.body if condition else stmt.orelse))
            return any(map(cls._terminates_suite, stmt.orelse)) and any(map(cls._terminates_suite, stmt.body))
        return isinstance(stmt, ast.With) and any(map(cls._terminates_suite, stmt.body))

    def _visit_scoped_body(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        """Visit statements within a nested scope, flattening lists and dropping None."""
        result: list[ast.stmt] = []
        for stmt in statements:
            visited = self.visit(stmt)
            if visited is None:
                pass
            elif isinstance(visited, list):
                result.extend(cast("list[ast.stmt]", visited))
            else:
                result.append(visited)
            if result and self._terminates_suite(result[-1]):
                break
        return result or [ast.Pass()]

    @staticmethod
    def _names_by_ctx(node: ast.AST, ctx_type: type) -> set[str]:
        """Collect ``Name`` ids appearing under the given context (Load/Store)."""
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ctx_type)}

    def _live_statements(self, body: list[ast.stmt], *, reverse: bool = False) -> Iterator[ast.stmt]:
        """Flatten statically selected arms in execution (or liveness) order."""
        for stmt in reversed(body) if reverse else body:
            if isinstance(stmt, ast.If) and stmt in self._static_arms:
                arm = stmt.body if self._static_arms[stmt] else stmt.orelse
                yield from self._live_statements(arm, reverse=reverse)
            else:
                yield stmt

    def _collect_upward_exposed(
        self,
        statements: list[ast.stmt],
        written: set[str],
        exposed: set[str],
        *,
        may_skip_loops: bool = False,
        loop_exits: list[set[str]] | None = None,
    ) -> bool:
        """Record names *read before being written* within ``statements``.

        A name whose first appearance (in execution order) is a read is
        *upward-exposed* — live on entry to this body, hence a genuine
        loop-carried value.  A name first written is a fresh body-local.
        ``_classify_loop_locals`` uses this to tell genuine carries apart
        from fresh per-loop locals that merely share a name with a sibling
        scope's local: a write-before-read name in a sibling loop is a new
        variable, and bridging it (``x_v1 = x``) would emit a read of ``x``
        outside its defining scope.

        ``written`` and ``exposed`` are accumulators threaded through nested
        bodies so execution order is respected across compound statements.
        Return whether control can reach the statement after this suite.

        SIMT and branch-merge callers set ``may_skip_loops`` because a loop
        body cannot initialize a local when the loop runs zero times.

        Names bound only inside a comprehension or lambda are not
        statement-level assignments, so they never enter ``_assign_count`` and
        are never ``_classify_loop_locals`` candidates.  ``_names_by_ctx`` does
        walk into those nested scopes, but only statement *targets* feed
        ``written`` here — a comprehension/lambda local can at most leak a
        harmless extra entry into ``exposed``, never affecting classification.
        """
        for stmt in self._live_statements(statements):
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                # RHS is evaluated before the target binds.
                if stmt.value is not None:
                    exposed.update(self._names_by_ctx(stmt.value, ast.Load) - written)
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                for tgt in targets:
                    # Subscript/attribute targets read their base (``t[i] = ...``).
                    exposed.update(self._names_by_ctx(tgt, ast.Load) - written)
                    written.update(self._names_by_ctx(tgt, ast.Store))
            elif isinstance(stmt, ast.AugAssign):
                exposed.update(self._names_by_ctx(stmt.value, ast.Load) - written)
                # ``x += v`` reads then writes ``x``.
                tgt_names = self._names_by_ctx(stmt.target, ast.Store)
                exposed.update(tgt_names - written)
                written.update(tgt_names)
            elif isinstance(stmt, ast.For) or (may_skip_loops and isinstance(stmt, ast.While)):
                self._collect_loop_writes(stmt, written, exposed, may_skip_loops)
            elif isinstance(stmt, ast.If):
                exposed.update(self._names_by_ctx(stmt.test, ast.Load) - written)
                # then/else are alternative paths: analyze each from the same
                # incoming ``written`` so a read in one branch is not masked by
                # a write in the other.  A name is exposed if read-before-write
                # on either path. Only arms reaching the next statement can
                # supply its value. Copies before an unconditional transfer
                # must also be valid when both arms terminate.
                then_written = set(written)
                then_reaches = self._collect_upward_exposed(
                    stmt.body, then_written, exposed, may_skip_loops=may_skip_loops, loop_exits=loop_exits
                )
                else_written = set(written)
                else_reaches = self._collect_upward_exposed(
                    stmt.orelse, else_written, exposed, may_skip_loops=may_skip_loops, loop_exits=loop_exits
                )
                if not may_skip_loops or then_reaches == else_reaches:
                    written.update(then_written & else_written)
                else:
                    written.update(then_written if then_reaches else else_written)
            elif isinstance(stmt, ast.With):
                for item in stmt.items:
                    exposed.update(self._names_by_ctx(item.context_expr, ast.Load) - written)
                    if item.optional_vars is not None:
                        written.update(self._names_by_ctx(item.optional_vars, ast.Store))
                # SIMT can read enclosing values, but its writes are private.
                body_written = written.copy() if "simt_scope" in _with_scope_names(stmt) else written
                self._collect_upward_exposed(
                    stmt.body, body_written, exposed, may_skip_loops=may_skip_loops, loop_exits=loop_exits
                )
            else:
                # Expr / Return / Assert / ... — pure reads.
                exposed.update(self._names_by_ctx(stmt, ast.Load) - written)
            if loop_exits is not None and isinstance(stmt, (ast.Break, ast.Continue)):
                loop_exits.append(written.copy())
            if self._terminates_suite(stmt):
                return False
        return True

    def _collect_loop_writes(
        self, stmt: ast.For | ast.While, written: set[str], exposed: set[str], may_skip_loops: bool
    ) -> None:
        header = stmt.iter if isinstance(stmt, ast.For) else stmt.test
        exposed.update(self._names_by_ctx(header, ast.Load) - written)
        body_written = written.copy() if may_skip_loops else written
        targets = self._names_by_ctx(stmt.target, ast.Store) if isinstance(stmt, ast.For) else set[str]()
        body_written.update(targets)
        exits: list[set[str]] = []
        self._collect_upward_exposed(
            stmt.body,
            body_written,
            exposed,
            may_skip_loops=may_skip_loops,
            loop_exits=exits if may_skip_loops else None,
        )
        if may_skip_loops:
            # An explicit carry has an initial value even when the loop is
            # skipped. Ordinary locals must also be defined on early exits.
            outputs = self._loop_yield_targets(stmt.body)
            if isinstance(stmt, ast.For) and self._private_loop_body(stmt, outputs):
                written.update(outputs)
            elif self._loop_enters(stmt) is True:
                written.update(body_written.intersection(*exits) - targets)
        self._collect_upward_exposed(stmt.orelse, written, exposed, may_skip_loops=may_skip_loops)

    def _classify_loop_locals(self, statements: list[ast.stmt]) -> tuple[list[str], list[str]]:
        """Classify same-depth name collisions in a loop body into ``(carried, fresh)``.

        A name is considered only if it (1) is already in ``_assign_count`` and
        (2) had its first assignment at the current scope depth — i.e. it
        collides with an existing same-depth binding.  Such a name is:

        - *carried* — a genuine loop-carried rebind — when it is upward-exposed
          (read before written) in the body.  It needs a bridge assignment
          (``x_v1 = x``) so the carried value threads across the loop boundary.
        - *fresh* — a new body-local that merely reuses a sibling scope's name
          (both sit at the same ``_scope_depth``).  It must be neither bridged
          (the source ``x`` would be read outside its defining scope) nor
          renamed (the alias would be undefined when the loop runs zero times).

        Depth alone cannot tell these apart — two sibling loop bodies share a
        depth.  Fresh writes reuse the existing spelling throughout the loop,
        leaving a valid predecessor intact when the loop executes zero times.
        """
        carried: list[str] = []
        fresh: list[str] = []
        seen: set[str] = set()
        candidates: list[str] = []
        for stmt in self._live_statements(statements):
            if self._terminates_suite(stmt):
                break
            if isinstance(stmt, ast.Assign):
                targets = stmt.targets
            elif isinstance(stmt, ast.AnnAssign):
                targets = [stmt.target]
            else:
                continue
            # Collect every stored name — covers plain, tuple-unpacking
            # (``x, y = ...``) and chained (``a = b = ...``) targets.
            # Subscript/attribute targets contribute nothing: their base is a
            # Load, not a rebind.
            for target in targets:
                for name in self._names_by_ctx(target, ast.Store):
                    if name in seen:
                        continue
                    if name in self._assign_count and self._scope_depth == self._assign_depth.get(name, -1):
                        seen.add(name)
                        candidates.append(name)
        if candidates:
            exposed: set[str] = set()
            self._collect_upward_exposed(statements, set(), exposed)
            for name in candidates:
                (carried if name in exposed else fresh).append(name)
        return carried, fresh

    def _forget_rename_bookkeeping(self, name: str) -> None:
        """Drop all rename bookkeeping for ``name``.

        After this call ``name`` is unknown to the renamer, so the next
        assignment to it is treated as a first binding (no rename, no bridge).
        Used to discard sibling-scope locals — fresh loop-body locals and
        then-branch locals — that must not leak into a sibling scope.
        """
        self._assign_count.pop(name, None)
        self._assign_depth.pop(name, None)
        self._var_renames.pop(name, None)
        self._meta.pop(name, None)

    def _binding_value(self, name: str) -> str | int:
        folded = self._active_shape_fold(name)
        return folded if folded is not None else self._var_renames.get(name, name)

    @staticmethod
    def _copy_binding(target: str, value: str | int) -> ast.Assign:
        return ast.fix_missing_locations(
            ast.Assign(
                targets=[ast.Name(id=target, ctx=ast.Store())],
                value=ast.Constant(value=value)
                if isinstance(value, int)
                else ast.Name(id=value, ctx=ast.Load()),
                lineno=0,
                col_offset=0,
            )
        )

    def _make_bridge_assignments(self, rebind_vars: list[str]) -> list[ast.stmt]:
        """Create bridge assignments (``x_v1 = x``) and apply renames for each variable.

        After this call, ``_var_renames`` is updated so that subsequent reads of the
        original name resolve to the new alias.  The ``_assign_depth`` is updated to
        the current (outer) scope depth so that assignments inside the loop body are
        treated as deeper-scope (loop-carried) and NOT renamed again.

        The returned statements should be inserted before the scope that contains
        the rebinding assignments.
        """
        bridges: list[ast.stmt] = []
        for var_name in rebind_vars:
            # Get the current read-name for this variable (could already be an alias)
            value = self._binding_value(var_name)
            source = self._known_tensor_name(var_name)
            value_meta = self._meta.get(source) if source is not None else None
            new_name = self._rebind(var_name)
            if isinstance(value, int):
                self._shape_inlined.pop(var_name)
            self._record_tensor_meta(new_name, None if isinstance(value, int) else value_meta)
            # Update assign_depth to the outer scope so the loop body sees a
            # depth mismatch and does NOT trigger another rebind.
            self._assign_depth[var_name] = self._scope_depth
            bridges.append(self._copy_binding(new_name, value))
        return bridges

    def _materialize_folded_writes(self, statements: list[ast.stmt]) -> tuple[list[ast.stmt], dict[str, int]]:
        """Give folded dimensions scalar bindings before conditional assignments."""
        if not self._shape_inlined:
            return [], {}
        written = set[str]().union(
            *(self._source_stores.matching(stmt, self._shape_inlined) for stmt in statements)
        )
        folded = {name for name in written if self._active_shape_fold(name) is not None}
        if not folded:
            return [], {}
        live, restored = self._reachable_folded_writes(statements)
        recovered = {name: self._shape_inlined[name] for name in folded & restored}
        return self._make_bridge_assignments(sorted(folded & live)), recovered

    @staticmethod
    def _initial_while_test(stmt: ast.For, call: ast.Call) -> ast.expr | None:
        """Express the first pl.cond test in terms of the incoming initializers."""
        from pypto.language.parser.syntax_helpers import is_pl_call  # noqa: PLC0415

        first = stmt.body[0] if stmt.body else None
        init = next((kw.value for kw in call.keywords if kw.arg == "init_values"), None)
        if not (
            isinstance(stmt.target, ast.Tuple)
            and isinstance(init, (ast.Tuple, ast.List))
            and len(stmt.target.elts) == len(init.elts)
            and all(isinstance(elt, ast.Name) for elt in stmt.target.elts)
            and isinstance(first, ast.Expr)
            and is_pl_call(first.value, "cond")
            and len(first.value.args) == 1
            and not first.value.keywords
        ):
            return None
        # Substitute simultaneously: each initializer sees the outer scope,
        # even when names are swapped or shadow a folded dimension.
        initial = {
            cast(ast.Name, arg).id: value for arg, value in zip(stmt.target.elts, init.elts, strict=True)
        }

        class InitialValues(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.expr:
                return initial.get(node.id, node)

        return InitialValues().visit(deepcopy(first.value.args[0]))

    def _loop_enters(self, stmt: ast.For | ast.While, unstable: set[str] | None = None) -> bool | None:
        """Resolve the first iteration from the header, or return None for a dynamic bound."""
        from pypto.language.parser.control_flow_parser import ITERATOR_TO_KIND  # noqa: PLC0415
        from pypto.language.parser.syntax_helpers import is_pl_call  # noqa: PLC0415

        explicit_while = False
        if isinstance(stmt, ast.While):
            values_to_check = [stmt.test]
        elif is_pl_call(stmt.iter, "while_"):
            test = self._initial_while_test(stmt, stmt.iter)
            if test is None:
                return None
            explicit_while = True
            values_to_check = [test]
        else:
            call = stmt.iter
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "pl"
                and call.func.attr in ITERATOR_TO_KIND
            ):
                return None
            values_to_check = call.args
            if call.func.attr == "hardware_loop" and not values_to_check:
                values_to_check = [kw.value for kw in call.keywords if kw.arg == "stop"]
            if not 1 <= len(values_to_check) <= 3:
                return None
        if any(
            isinstance(part, ast.Call)
            or (isinstance(part, ast.Name) and unstable is not None and part.id in unstable)
            for expr in values_to_check
            for part in ast.walk(expr)
        ):
            return None
        # Reachability previews source expressions. Branch-merge analysis sees
        # the specialized header, whose names must not be rewritten a second time.
        bounds = (
            values_to_check if unstable is None else [self.visit(deepcopy(expr)) for expr in values_to_check]
        )
        if isinstance(stmt, ast.While) or explicit_while:
            folded = _constant_scalar(bounds[0])
            return bool(folded) if folded is not None else values.literal_condition(bounds[0])
        args: list[int] = []
        for bound in bounds:
            value = _constant_scalar(bound)
            if type(value) is not int:
                return None
            args.append(value)
        return None if len(args) == 3 and args[2] == 0 else bool(range(*args))

    def _empty_loop(self, stmt: ast.For | ast.While, unstable: set[str]) -> bool:
        return self._loop_enters(stmt, unstable) is False and (
            (isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Tuple))
            or not self._loop_yield_targets(stmt.body)
        )

    def _preview_shape_rhs(self, value: ast.expr, changed: set[str]) -> ast.expr | None:
        if any(
            isinstance(part, (ast.Call, ast.NamedExpr)) or (isinstance(part, ast.Name) and part.id in changed)
            for part in ast.walk(value)
        ):
            return None
        return self.visit(deepcopy(value))

    def _shape_write_effects(
        self, stmt: ast.Assign | ast.AnnAssign, changed: set[str]
    ) -> tuple[set[str], set[str], bool]:
        """Classify final stores against the entry folds, using the pre-assignment RHS."""
        stores = {item.id for item in self._yield_stores(stmt)}
        if stmt.value is None or not any(self._active_shape_fold(name) is not None for name in stores):
            return stores, set(), False
        unchanged: set[str] = set()

        def inspect(target: ast.expr, rhs: ast.expr) -> bool:
            if isinstance(target, (ast.Tuple, ast.List)):
                # Inspect explicit elements separately: a dynamic sibling
                # cannot invalidate a known constant. A shape can also fold
                # to a tuple during the preview.
                value = (
                    rhs if isinstance(rhs, (ast.Tuple, ast.List)) else self._preview_shape_rhs(rhs, changed)
                )
                if isinstance(value, (ast.Tuple, ast.List)) and len(target.elts) == len(value.elts):
                    untouched = True
                    for lhs, elt in zip(target.elts, value.elts, strict=True):
                        if not inspect(lhs, elt):
                            untouched = False
                    return untouched
            elif isinstance(target, ast.Name):
                if self._same_shape_value(target.id, self._preview_shape_rhs(rhs, changed)):
                    unchanged.add(target.id)
                    return target.id not in changed
            unchanged.difference_update(self._names_by_ctx(target, ast.Store))
            return False

        # Always emit annotated stores so the parser still validates the type.
        elide = isinstance(stmt, ast.Assign)
        for target in stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]:
            if not inspect(target, stmt.value):
                elide = False
        return stores - unchanged, unchanged, elide

    def _constant_outputs(
        self,
        statements: list[ast.stmt],
        changed: set[str],
        *,
        loop: ast.For | None = None,
        empty: bool = False,
    ) -> dict[str, int]:
        """Prove an integer at each owned yield and, for loops, on the zero-trip path."""
        bindings = self._yield_bindings(statements)
        if not bindings:
            return {}
        # None shadows an outer constant. A carry name represents an unknown
        # iteration's input: forwarding that same slot preserves its initializer.
        local: dict[str, int | str | None] = {}
        unknown = changed.copy()

        class LocalConstants(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.expr:
                value = local.get(node.id)
                return ast.Constant(value=value) if type(value) is int else node

        def value_of(arg: ast.expr) -> int | str | None:
            if isinstance(arg, ast.Constant):
                return arg.value if type(arg.value) is int else None
            if isinstance(arg, ast.Call):
                return None
            if isinstance(arg, ast.Name) and arg.id in local:
                return local[arg.id]
            expr = LocalConstants().visit(deepcopy(arg)) if local else arg
            preview = self._preview_shape_rhs(expr, unknown)
            value = _constant_scalar(preview) if preview is not None else None
            return value if type(value) is int else None

        initial: list[int | str | None] = []
        carries: list[ast.Name] = []
        if loop is not None:
            call = cast(ast.Call, loop.iter)
            init = next((kw.value for kw in call.keywords if kw.arg == "init_values"), None)
            if not isinstance(init, (ast.Tuple, ast.List)):
                return {}
            initial = [value_of(arg) for arg in init.elts]
            induction, carries = self._loop_header_names(loop, _scope_call_name(call) == "while_")
            if len(carries) != len(initial):
                return {}
            unknown.update(self._names_by_ctx(loop.target, ast.Store))
            local.update((carry.id, carry.id) for carry in carries)
            if induction is not None:
                local[induction] = None

        results: dict[str, int | None] = {}

        def observe(stmt: ast.Assign | ast.AnnAssign) -> None:
            names = bindings[stmt]
            args = cast(ast.Call, stmt.value).args
            if len(names) != len(args) or (loop is not None and len(names) != len(initial)):
                results.update(dict.fromkeys(names))
                return
            for index, (name, arg) in enumerate(zip(names, args, strict=True)):
                value = initial[index] if empty else value_of(arg)
                if loop is not None:
                    if value == carries[index].id:
                        value = initial[index]
                    if value != initial[index]:
                        value = None
                results[name] = value if type(value) is int and results.get(name, value) == value else None

        def assigned_values(stmt: ast.Assign | ast.AnnAssign) -> dict[str, int | str | None]:
            assigned: dict[str, int | str | None] = {}
            if stmt.value is not None:
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        assigned[target.id] = value_of(stmt.value)
                    elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(
                        stmt.value, (ast.Tuple, ast.List)
                    ):
                        if len(target.elts) == len(stmt.value.elts) and all(
                            isinstance(t, ast.Name) for t in target.elts
                        ):
                            assigned.update(
                                (cast(ast.Name, t).id, value_of(v))
                                for t, v in zip(target.elts, stmt.value.elts, strict=True)
                            )
            return assigned

        if empty:
            for output in bindings:
                observe(output)
        else:
            from pypto.language.parser.control_flow_parser import ControlFlowParser  # noqa: PLC0415

            pending = list(reversed(statements))
            while pending:
                stmt = pending.pop()
                if isinstance(stmt, ast.With) and ControlFlowParser.is_runtime_scope_with(stmt):
                    for item in stmt.items:
                        if item.optional_vars is not None:
                            names = self._names_by_ctx(item.optional_vars, ast.Store)
                            unknown.update(names)
                            local.update(dict.fromkeys(names))
                    pending.extend(reversed(stmt.body))
                    continue
                if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt in bindings:
                    observe(stmt)
                    continue
                # Read all RHS values before applying stores, including repeated
                # targets. For compound statements retain only unaffected facts.
                wrapped = (
                    self._yield_bindings([stmt])
                    if isinstance(stmt, ast.For) and _scope_call_name(stmt.iter) == "split_aiv"
                    else {}
                )
                # split_aiv is transparent to yield ownership. A yield target
                # names the outer result, not a private write before the yield.
                output_targets = {target for output in wrapped for target in self._yield_stores(output)}
                stores = {
                    part.id
                    for part in ast.walk(stmt)
                    if isinstance(part, ast.Name)
                    and isinstance(part.ctx, ast.Store)
                    and part not in output_targets
                }
                assigned = assigned_values(stmt) if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else {}
                unknown.update(stores)
                local.update(dict.fromkeys(stores))
                local.update(assigned)
                for output in wrapped:
                    observe(output)
        return {name: value for name, value in results.items() if value is not None}

    def _branch_output_constants(self, stmt: ast.If, changed: set[str]) -> dict[str, int]:
        left = self._constant_outputs(stmt.body, changed)
        if not left:
            return {}
        right = self._constant_outputs(stmt.orelse, changed)
        return {name: value for name, value in left.items() if right.get(name) == value}

    @staticmethod
    def _loop_entry_changes(
        body: list[ast.stmt],
        entry: set[str],
        transfer: Callable[[ast.stmt, set[str]], tuple[set[str] | None, set[str]]],
    ) -> set[str]:
        """Propagate newly unstable names to their next use or definition around the loop."""
        # A fallthrough statement passes untouched names through implicitly.
        # A transfer can bypass later stores, so route all changing names
        # through it, including names that do not occur in its syntax.
        stop = len(body)
        mentions: dict[str, list[int]] = {}
        next_jump = [stop] * (stop + 1)
        for index, stmt in enumerate(body):
            for part in ast.walk(stmt):
                if isinstance(part, ast.Name):
                    sites = mentions.setdefault(part.id, [])
                    if not sites or sites[-1] != index:
                        sites.append(index)
                elif isinstance(part, (ast.Break, ast.Continue, ast.Return)):
                    next_jump[index] = index
        for index in reversed(range(stop)):
            next_jump[index] = min(next_jump[index], next_jump[index + 1])

        def next_use(name: str, index: int) -> int:
            sites = mentions.get(name, [])
            pos = bisect_right(sites, index)
            return min(sites[pos], next_jump[index + 1]) if pos < len(sites) else next_jump[index + 1]

        before: list[set[str]] = [set() for _ in body]
        reached = 0
        work = deque([0])
        queued = {0}

        def schedule(index: int) -> None:
            if index <= reached and index not in queued:
                queued.add(index)
                work.append(index)

        def deliver(name: str, index: int) -> None:
            if index < stop and name not in before[index]:
                before[index].add(name)
                schedule(index)

        for name in entry:
            deliver(name, next_use(name, -1))
        while work:
            index = work.popleft()
            queued.remove(index)
            after, continued = transfer(body[index], before[index])
            if after is not None:
                if reached == index and reached + 1 < stop:
                    reached += 1
                    schedule(reached)
                for name in after:
                    successor = next_use(name, index)
                    if successor == stop:
                        continued.add(name)
                    else:
                        deliver(name, successor)
            for name in continued - entry:
                entry.add(name)
                deliver(name, next_use(name, -1))
        return entry

    def _reachable_folded_writes(self, statements: list[ast.stmt]) -> tuple[set[str], set[str]]:
        """Track possible writes separately along each arm and loop backedge."""

        # A dimension can need a real binding for intermediate reads even
        # when every path restores its entry value before leaving the scope.
        temporary: set[str] = set()

        def publish(
            changed: set[str], outputs: set[str], constants: dict[str, int], record: bool
        ) -> set[str]:
            same = {name for name, val in constants.items() if self._active_shape_fold(name) == val}
            if record:
                temporary.update(changed & same)
            return (changed | outputs) - same

        def join_paths(left: set[str] | None, right: set[str] | None) -> set[str] | None:
            if left is None:
                return right
            return left if right is None else left | right

        def visit_branch(
            stmt: ast.If, changed: set[str], exits: dict[type[ast.stmt], set[str]], record: bool
        ) -> set[str] | None:
            condition = None
            if not any(
                isinstance(part, ast.Call) or (isinstance(part, ast.Name) and part.id in changed)
                for part in ast.walk(stmt.test)
            ):
                test = self.visit(deepcopy(stmt.test))
                condition = values.literal_condition(test)
                if condition is None:
                    condition = _constant_condition(test)
                if record and condition is not None:
                    # Retain the path's incoming constants when another
                    # arm later needs a scalar binding for this name.
                    stmt.test = test
            if condition is not None:
                after = visit_suite(stmt.body if condition else stmt.orelse, changed, exits, record)
            elif outputs := self._loop_yield_targets([*stmt.body, *stmt.orelse]):
                # Both arms have private bindings and explicit outputs.
                after = publish(changed, outputs, self._branch_output_constants(stmt, changed), record)
            else:
                left = visit_suite(stmt.body, changed, exits, record)
                right = visit_suite(stmt.orelse, changed, exits, record)
                after = join_paths(left, right)
            return after

        def visit_loop(
            stmt: ast.For | ast.While, changed: set[str], exits: dict[type[ast.stmt], set[str]], record: bool
        ) -> set[str] | None:
            outputs = self._loop_yield_targets(stmt.body)
            breaks = None
            if self._empty_loop(stmt, changed):
                if record:
                    self._proven_empty_loops.add(stmt)
                if isinstance(stmt, ast.For) and outputs:
                    changed = publish(
                        changed,
                        outputs,
                        self._constant_outputs(stmt.body, changed, loop=stmt, empty=True),
                        record,
                    )
            elif isinstance(stmt, ast.For) and self._private_loop_body(stmt, outputs):
                changed = publish(
                    changed, outputs, self._constant_outputs(stmt.body, changed, loop=stmt), record
                )
            else:
                scoped = (
                    self._names_by_ctx(stmt.target, ast.Store) if isinstance(stmt, ast.For) else set[str]()
                )
                entry = changed | scoped
                transfers: dict[type[ast.stmt], set[str]] = {}
                end = visit_suite(stmt.body, entry, transfers, False)
                backedge = (end or set[str]()) | transfers.get(ast.Continue, set())
                grew = not backedge <= entry
                if grew:

                    def transfer(item: ast.stmt, before: set[str]) -> tuple[set[str] | None, set[str]]:
                        exits: dict[type[ast.stmt], set[str]] = {}
                        after = visit_suite([item], before, exits, False)
                        return after, exits.get(ast.Continue, set())

                    entry = self._loop_entry_changes(stmt.body, entry | backedge, transfer)
                # Recompute exits with the stable header before recording any
                # elided stores or folding conditions in the actual AST.
                if record or grew:
                    transfers = {}
                    visit_suite(stmt.body, entry, transfers, record)
                # Breaks exit this loop. Returns exit the surrounding
                # function. Header names stay local unless yielded.
                if ast.Return in transfers:
                    exits.setdefault(ast.Return, set()).update((transfers[ast.Return] - scoped) | changed)
                hidden = scoped - outputs
                if ast.Break in transfers:
                    breaks = changed | (transfers[ast.Break] - hidden)
                changed |= entry - hidden
            # A break bypasses the loop's else suite.
            after = join_paths(breaks, visit_suite(stmt.orelse, changed, exits, record))
            return after

        def visit_suite(
            suite: list[ast.stmt], incoming: set[str], exits: dict[type[ast.stmt], set[str]], record: bool
        ) -> set[str] | None:
            # Each arm starts with the facts at its own entry. Only paths
            # that can reach a subsequent statement affect its condition.
            changed = incoming.copy()
            for index, stmt in enumerate(suite):
                if isinstance(stmt, (ast.Break, ast.Continue, ast.Return)):
                    exits.setdefault(type(stmt), set()).update(changed)
                    return None
                if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
                    writes, unchanged, elide = self._shape_write_effects(stmt, changed)
                    if record:
                        temporary.update(changed & unchanged)
                        if elide:
                            suite[index] = ast.copy_location(ast.Pass(), stmt)
                    changed.difference_update(unchanged)
                    changed.update(writes)
                elif isinstance(stmt, ast.AugAssign):
                    changed.update(self._names_by_ctx(stmt.target, ast.Store))
                elif isinstance(stmt, ast.If):
                    after = visit_branch(stmt, changed, exits, record)
                    if after is None:
                        return None
                    changed = after
                elif isinstance(stmt, (ast.For, ast.While)):
                    after = visit_loop(stmt, changed, exits, record)
                    if after is None:
                        return None
                    changed = after
                elif isinstance(stmt, ast.With) and not any(
                    _scope_call_name(item.context_expr) == "simt_scope" for item in stmt.items
                ):
                    for item in stmt.items:
                        if item.optional_vars is not None:
                            changed.update(self._names_by_ctx(item.optional_vars, ast.Store))
                    after = visit_suite(stmt.body, changed, exits, record)
                    if after is None:
                        return None
                    changed = after
            return changed

        exits: dict[type[ast.stmt], set[str]] = {}
        end = visit_suite(statements, set(), exits, True)
        # Transfers can observe a changed value even if the fallthrough paths
        # restore it. Keep their runtime binding until the enclosing scope exits.
        live = (end or set[str]()).union(*exits.values())
        return live | temporary, temporary - live

    @staticmethod
    def _yield_stores(stmt: ast.Assign | ast.AnnAssign) -> list[ast.Name]:
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        return [
            item
            for target in targets
            for item in ast.walk(target)
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
        ]

    @classmethod
    def _loop_yield_targets(cls, statements: list[ast.stmt]) -> set[str]:
        """Collect outputs using the parser's yield ownership boundaries."""
        from pypto.language.parser.control_flow_parser import ControlFlowParser  # noqa: PLC0415

        return {
            item.id
            for stmt in ControlFlowParser.iter_yield_assignments(statements)
            for item in cls._yield_stores(stmt)
        }

    @staticmethod
    def _private_loop_body(node: ast.For, outputs: set[str]) -> bool:
        """Mirror the parser scopes that export only assignment-form yields."""
        from pypto.language.parser.control_flow_parser import ITERATOR_TO_KIND  # noqa: PLC0415

        kind = _scope_call_name(node.iter)
        return kind == "while_" or (
            kind in ITERATOR_TO_KIND and (isinstance(node.target, ast.Tuple) or bool(outputs))
        )

    @staticmethod
    def _loop_header_names(node: ast.For, is_explicit_while: bool) -> tuple[str | None, list[ast.Name]]:
        """Return the induction name (range only) and ordered loop-carried arguments."""
        if is_explicit_while:
            induction, carries = None, node.target
        elif isinstance(node.target, ast.Tuple) and len(node.target.elts) == 2:
            induction, carries = node.target.elts
        else:
            induction, carries = node.target, None
        return (
            induction.id if isinstance(induction, ast.Name) else None,
            [elt for elt in carries.elts if isinstance(elt, ast.Name)]
            if isinstance(carries, ast.Tuple)
            else [],
        )

    def _publish_loop_updates(
        self, yields: dict[ast.Assign | ast.AnnAssign, list[str]], incoming: set[str]
    ) -> list[ast.stmt]:
        """Copy private loop results into the bindings visible on the enclosing scope's skipped path."""
        # Loop return_vars have their own IR identity. An ordinary store to
        # the incoming binding lets the enclosing SSA join see the update.
        results = {name: self._fresh_alias(name) for name in sorted(incoming)}
        for stmt, names in yields.items():
            for target, name in zip(self._yield_stores(stmt), names, strict=True):
                if name in results:
                    target.id = results[name]
        copies: list[ast.stmt] = []
        for name, result in results.items():
            bound = self._bind_assign_name(ast.Name(id=name, ctx=ast.Store()), None)
            self._shape_inlined.pop(name, None)
            copies.append(self._copy_binding(bound.id, result))
        return copies

    @classmethod
    def _for_scoped_names(cls, node: ast.For) -> set[str]:
        if _scope_call_name(node.iter) == "while_":
            return cls._names_by_ctx(node.target, ast.Store)
        target = node.target
        if isinstance(target, ast.Tuple) and target.elts:
            target = target.elts[0]
        return {target.id} if isinstance(target, ast.Name) else set()

    @classmethod
    def _record_tensor_write(cls, result: _TensorWrites, target: ast.expr, value: ast.expr | None) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            operands = value.elts if isinstance(value, (ast.Tuple, ast.List)) else []
            for index, element in enumerate(target.elts):
                rhs = operands[index] if len(operands) == len(target.elts) else None
                cls._record_tensor_write(result, element, rhs)
        elif isinstance(target, ast.Name):
            if isinstance(value, ast.Call):
                value = writeback_return_operand(value)
            result.setdefault(target.id, set()).add(value.id if isinstance(value, ast.Name) else None)

    def _suite_tensor_writes(self, statements: list[ast.stmt]) -> _TensorWrites:
        writes: _TensorWrites = {}
        for stmt in statements:
            for name, sources in self._statement_tensor_writes(stmt).items():
                writes.setdefault(name, set()).update(sources)
            if self._terminates_suite(stmt):
                break
        return writes

    def _conditional_tensor_writes(self, stmt: ast.If) -> _TensorWrites:
        writes: _TensorWrites = {}
        left = self._suite_tensor_writes(stmt.body)
        right = self._suite_tensor_writes(stmt.orelse)
        self._tensor_arms[stmt] = left, right
        for name in left.keys() | right.keys():
            sources = left.get(name)
            if sources is not None and sources == right.get(name):
                writes[name] = sources
            else:
                # The guard can depend on a folded value in this loop's
                # incoming environment. Select its arm when consuming it.
                writes[name] = {stmt}
        return writes

    def _statement_tensor_writes(self, stmt: ast.stmt) -> _TensorWrites:
        """Summarize visible copy sources before visiting a lexical subtree."""
        if stmt in self._tensor_writes:
            return self._tensor_writes[stmt]
        writes: _TensorWrites = {}
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for target in targets:
                self._record_tensor_write(writes, target, stmt.value)
        elif isinstance(stmt, ast.If):
            writes = self._conditional_tensor_writes(stmt)
        elif isinstance(stmt, (ast.For, ast.While)):
            outputs = self._loop_yield_targets(stmt.body)
            if isinstance(stmt, ast.For) and self._private_loop_body(stmt, outputs):
                writes = {name: {None} for name in outputs}
            else:
                writes = self._suite_tensor_writes(stmt.body)
                if isinstance(stmt, ast.For):
                    for name in self._for_scoped_names(stmt) - outputs:
                        writes.pop(name, None)
            for name, sources in self._suite_tensor_writes(stmt.orelse).items():
                writes.setdefault(name, set()).update(sources)
        elif isinstance(stmt, ast.With):
            if "simt_scope" not in _with_scope_names(stmt):
                writes = self._suite_tensor_writes(stmt.body)
            for item in stmt.items:
                if item.optional_vars is not None:
                    self._record_tensor_write(writes, item.optional_vars, None)
        else:
            writes = {name: {None} for name in self._names_by_ctx(stmt, ast.Store)}
        self._tensor_writes[stmt] = writes
        return writes

    def _forget_variant_tensors(self, names: set[str]) -> None:
        for name in names:
            current = self._var_renames.get(name, name)
            self._record_tensor_meta(current, None, reuses_name=True)

    def _loop_variant_tensors(self, loop: ast.For | ast.While) -> set[str]:
        """Prove which tensor copies preserve their incoming type on every backedge."""
        if not self._meta or not self._reads_tensor_properties:
            return set()
        writes = self._statement_tensor_writes(loop)
        written = set(writes)
        incoming = {
            name: self._meta[current]
            for name in written
            if (current := self._known_tensor_name(name)) is not None
        }
        if not incoming:
            return set()
        variants: set[str] = set()
        dependents: dict[str, set[str]] = {}

        conditions: dict[ast.If, bool | None] = {}
        for name, original in incoming.items():
            pending = list(writes[name])
            seen: set[str | ast.If | None] = set()
            while pending:
                item = pending.pop()
                if item in seen:
                    continue
                seen.add(item)
                if isinstance(item, ast.If):
                    if item not in conditions:
                        test = self._preview_shape_rhs(item.test, written)
                        conditions[item] = _constant_condition(test) if test is not None else None
                    selected = conditions[item]
                    left, right = self._tensor_arms[item]
                    if selected is not False:
                        pending.extend(left.get(name, ()))
                    if selected is not True:
                        pending.extend(right.get(name, ()))
                    continue
                source = self._known_tensor_name(item) if item is not None else None
                meta = self._meta.get(source) if source is not None else None
                merged = merge_tensor_metas(original, meta)
                if merged is None or merged.inexact_axes != original.inexact_axes:
                    variants.add(name)
                    break
                if item is not None and item in incoming:
                    dependents.setdefault(item, set()).add(name)
        work = list(variants)
        while work:
            affected = dependents.get(work.pop(), set()) - variants
            variants.update(affected)
            work.extend(affected)
        return variants

    def visit_For(self, node: ast.For) -> ast.stmt | list[ast.stmt]:
        """Visit For loop, incrementing scope depth for its body.

        Before entering the body, classify same-depth name collisions
        (:meth:`_classify_loop_locals`).  A *carried* name gets a bridge
        assignment (``x_v1 = x``) before the loop plus a rename inside it,
        preserving loop-carried semantics across two sibling loops.  A *fresh*
        name keeps its incoming spelling for every write without an entry read.
        """
        is_explicit_while = _scope_call_name(node.iter) == "while_"
        induction, carries = self._loop_header_names(node, is_explicit_while)
        after_loop: list[ast.stmt] = []
        recovered: dict[str, int] = {}
        if node in self._proven_empty_loops or (self._shape_inlined and self._empty_loop(node, set())):
            # Keep the header for initializer evaluation and iterator validation
            # (e.g. pipeline stage or hardware-loop policy). Only the initial
            # carries can reach the parser's yield-LHS names on a zero-trip path.
            from pypto.language.parser.control_flow_parser import ControlFlowParser  # noqa: PLC0415

            recovered = {
                name: value
                for name, value in self._constant_outputs(node.body, set(), loop=node, empty=True).items()
                if self._active_shape_fold(name) == value or self._scope_depth == 0
            }
            outputs = list(ControlFlowParser.iter_yield_assignments(node.body))
            for output in outputs:
                call = cast(ast.Call, output.value)
                call.args = [
                    ast.copy_location(ast.Name(id=carry.id, ctx=ast.Load()), carry) for carry in carries
                ]
            node.body = (node.body[:1] if is_explicit_while else []) + outputs
            after_loop, node.orelse = node.orelse, []
            self._tensor_writes.pop(node, None)
        elif self._shape_inlined and carries:
            recovered = {
                name: value
                for name, value in self._constant_outputs(node.body, set(), loop=node).items()
                if self._active_shape_fold(name) == value
            }
        node.iter = self.visit(node.iter)
        # Range-like iterators bind an induction variable; pl.while_ binds only
        # carries. Both shadow incoming aliases/folds inside the body, but the
        # parser publishes assignment-form yield targets as post-loop results.
        scoped_names = {carry.id for carry in carries}
        if induction is not None:
            scoped_names.add(induction)
        yields = self._yield_bindings(node.body)
        yielded_names = {name for names in yields.values() for name in names}
        private = self._private_loop_body(node, yielded_names)
        # An optional enclosing scope needs the pre-loop binding on its
        # skipped path. Keep private body writes separate from that binding.
        incoming_outputs = {
            name
            for name in yielded_names
            if private and self._assign_depth.get(name, self._scope_depth) < self._scope_depth
        }
        local_names = scoped_names.copy()
        if private:
            for stmt in node.body:
                local_names.update(self._names_by_ctx(stmt, ast.Store))
        restore_names = (local_names - yielded_names) | incoming_outputs
        meta_names = restore_names | {self._var_renames.get(name, name) for name in restore_names}
        saved = (
            {name: self._meta[name] for name in meta_names if name in self._meta},
            {name: self._shape_inlined[name] for name in restore_names if name in self._shape_inlined},
            {name: self._assign_count[name] for name in restore_names if name in self._assign_count},
            {name: self._assign_depth[name] for name in restore_names if name in self._assign_depth},
            {name: self._var_renames[name] for name in restore_names if name in self._var_renames},
        )
        captures = self._graph_captures.copy() if private and self._graph_captures is not None else None
        outer_depth = self._scope_depth
        for name in scoped_names:
            self._forget_rename_bookkeeping(name)
            self._shape_inlined.pop(name, None)
            self._var_renames[name] = name  # Also shadows constexpr/free-name folding.
            self._assign_count[name] = 1
            # Carry updates belong to the loop's incoming binding. Marking
            # them body-local would alpha-rename the yield target unnecessarily.
            self._assign_depth[name] = outer_depth + 1 if name == induction else outer_depth
        variant_tensors: set[str] = set() if private else self._loop_variant_tensors(node)
        self._forget_variant_tensors(variant_tensors)
        fresh_depths: dict[str, int] = {}
        outer_optional_depth = self._optional_depth
        try:
            self._scope_depth += 1
            bridges: list[ast.stmt] = []
            if private:
                # Each iteration reads outer values until it shadows them.
                # Only yield outputs are visible after leaving the parser scope.
                for name in (local_names - scoped_names) & self._assign_depth.keys():
                    self._assign_depth[name] = self._scope_depth
            else:
                carried, fresh = self._classify_loop_locals(node.body)
                self._scope_depth -= 1
                bridges = self._make_bridge_assignments(
                    [name for name in carried if name not in scoped_names]
                )
                # A fresh write reuses the incoming spelling on both the zero-trip
                # and executed paths without reading a sibling's private local.
                fresh_depths = {name: self._assign_depth[name] for name in fresh if name not in scoped_names}
                self._assign_depth.update(dict.fromkeys(fresh_depths, outer_depth))
                folded_bridges, recovered = self._materialize_folded_writes([node])
                bridges.extend(folded_bridges)
                self._scope_depth += 1
                self._optional_depth = self._scope_depth
            node.body = self._visit_scoped_body(node.body)
        finally:
            self._optional_depth = outer_optional_depth
            self._scope_depth = outer_depth
            self._assign_depth.update(fresh_depths)
            if captures is not None and self._graph_captures is not None:
                _restore_bindings(self._graph_captures, captures, restore_names)
            _restore_bindings(self._meta, saved[0], meta_names)
            _restore_bindings(self._shape_inlined, saved[1], restore_names)
            _restore_bindings(self._assign_count, saved[2], restore_names)
            _restore_bindings(self._assign_depth, saved[3], restore_names)
            _restore_bindings(self._var_renames, saved[4], restore_names)
        self._forget_variant_tensors(variant_tensors)
        copies = self._publish_loop_updates(yields, incoming_outputs) if incoming_outputs else []
        self._restore_shape_folds(recovered)
        if after_loop:
            copies.extend(self._visit_scoped_body(after_loop))
        return [*bridges, node, *copies] if bridges or copies else node

    def inline_untyped_shape_reads(self, body: list[ast.stmt], parameters: list[str]) -> None:
        """Let the parser read a stable unpack result's shape from its actual IR type.

        Some operations have no Python tensor-metadata rule. A local copy of a
        shape extent would hide its ConstInt from operations requiring literal
        dimensions. Follow the binding through scopes only while it still denotes that dimension.
        """
        if not self._untyped_shape_sources:
            return
        writes = Counter(parameters)
        for stmt in body:
            writes.update(
                n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            )
        sources = {name for name in self._untyped_shape_sources if writes[name] == 1}
        if not sources:
            return

        _ShapeAliasInliner(self, sources).visit_suite(body)

    def _record_static_arms(self, statements: list[ast.stmt]) -> bool:
        """Collect erased arms for liveness using the emitter's binding and folding rules."""
        conditions = [
            node for stmt in statements for node in ast.walk(stmt) if isinstance(node, (ast.If, ast.While))
        ]
        tests = [node.test for node in conditions if isinstance(node, ast.If)]
        if not tests:
            return bool(conditions)

        # An untouched runtime argument survives visit_Name. A predicate that
        # directly compares it cannot select a static arm during specialization.
        runtime = self._assign_count.keys() - self._constexpr.keys() - self._shape_inlined.keys()
        for stmt in statements:
            runtime.difference_update(self._source_stores.matching(stmt, runtime))

        def runtime_test(test: ast.expr) -> bool:
            operands = (test.left, *test.comparators) if isinstance(test, ast.Compare) else (test,)
            return any(isinstance(arg, ast.Name) and arg.id in runtime for arg in operands)

        if all(runtime_test(test) for test in tests):
            return True

        class Preview(BodyTransformer):
            def _visit_literal_if(self, node: ast.If, condition: bool) -> list[ast.stmt]:
                self._static_arms[node] = condition
                return super()._visit_literal_if(node, condition)

            @contextmanager
            def _optional_facts(self) -> Iterator[None]:
                incoming = set(self._assign_count)
                yield
                # A name first assigned on a possibly skipped path does not
                # dominate a later predicate. Leave the lexical name shadowed.
                for name in self._assign_count.keys() - incoming:
                    self._shape_inlined.pop(name, None)
                    self._meta.pop(self._var_renames.get(name, name), None)

            def _visit_if_branches(self, node: ast.If) -> None:
                with self._optional_facts():
                    super()._visit_if_branches(node)

            def visit_For(self, node: ast.For) -> ast.stmt | list[ast.stmt]:
                with self._optional_facts():
                    return super().visit_For(node)

            def visit_While(self, node: ast.While) -> ast.stmt | list[ast.stmt]:
                with self._optional_facts():
                    return super().visit_While(node)

        # Dep variants are keyed by AST identity. The copy has its own nodes
        # and SSA state so analysis cannot modify the source or consume names.
        memo: dict[int, Any] = {}
        body = deepcopy(statements, memo)
        preview = Preview(
            self._meta,
            self._dep_names,
            list(self._assign_count),
            self._used_names,
            self._py_globals,
            self._callee_contexts,
            self._dep_func_names,
            self._constexpr,
            {id(memo[key]): target for key, target in self._dep_call_targets.items() if key in memo},
            self._unpack_metas,
        )
        preview._reads_tensor_properties = self._reads_tensor_properties
        preview._single_store_names = self._single_store_names
        try:
            for stmt in body:
                preview.visit(stmt)
        except (IndexError, KeyError, TypeError, ValueError, ParserError):
            # Liveness can change sibling bindings. Diagnose the actual source
            # in the emitting pass and retain both arms wherever preview stopped.
            pass
        self._static_arms.update(
            (node, preview._static_arms[memo[id(node)]])
            for node in conditions
            if isinstance(node, ast.If) and memo[id(node)] in preview._static_arms
        )
        return True

    def track_conditional_inputs(self, statements: list[ast.stmt]) -> None:
        """Record which sibling bindings must keep their spelling across each conditional."""
        self._reads_tensor_properties = any(
            isinstance(node, ast.Attribute) and node.attr in {"shape", "dtype"}
            for stmt in statements
            for node in ast.walk(stmt)
        )
        if self._reads_tensor_properties:
            stores = Counter(
                node.id
                for stmt in statements
                for node in ast.walk(stmt)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            )
            self._single_store_names = {
                name for name, count in stores.items() if count == 1
            } - self._assign_count.keys()
        if not self._record_static_arms(statements):
            return

        # GEN(body) is independent of its live-outs. Loop entry already
        # includes every live-out (zero trips), so only these upward-exposed
        # reads need to be added to account for a second iteration.
        loop_reads: dict[ast.For | ast.While, set[str]] = {}

        def loop_live_before(
            stmt: ast.For | ast.While, live: set[str], loop_live: set[str], *, record: bool
        ) -> set[str]:
            # A back edge observes reads before the body's first write. A
            # break can take the old value straight to the continuation.
            header = stmt.iter if isinstance(stmt, ast.For) else stmt.test
            loop_inputs = live_before(stmt.orelse, live, loop_live, record=record) | live
            if isinstance(stmt, ast.For):
                # Explicit carries publish their yield results even on zero trips.
                outputs = self._loop_yield_targets(stmt.body)
                if self._private_loop_body(stmt, outputs):
                    loop_inputs.difference_update(outputs)
            loop_inputs.update(self._names_by_ctx(header, ast.Load))
            # Find the reads before a write in one iteration, then use
            # those as live-outs of the preceding iteration.
            if stmt not in loop_reads:
                body_reads = live_before(stmt.body, set(), set(), record=False)
                if isinstance(stmt, ast.For):
                    # The iterator supplies the induction variable and carries on every entry.
                    body_reads.difference_update(self._names_by_ctx(stmt.target, ast.Store))
                loop_reads[stmt] = body_reads
            live = loop_inputs | loop_reads[stmt]
            if record:
                live_before(stmt.body, live, live, record=True)
                if isinstance(stmt, ast.While):
                    self._conditional_inputs[stmt] = live.copy()
            return live

        def scope_live_before(
            stmt: ast.With, live: set[str], loop_live: set[str], *, record: bool
        ) -> set[str]:
            # Graph completion captures bind after recording. Dispatch captures
            # bind on entry, after evaluating their dependency expressions.
            scopes = _with_scope_names(stmt)
            for item in stmt.items:
                if item.optional_vars is not None and _scope_call_name(item.context_expr) == "graph":
                    live.difference_update(self._names_by_ctx(item.optional_vars, ast.Store))
            body_live = live_before(stmt.body, live, loop_live, record=record)
            if scopes & {"simt_scope", "graph"}:
                body_live.update(live)  # Private region bindings can shadow the caller.
            live = body_live
            for item in reversed(stmt.items):
                if item.optional_vars is not None and _scope_call_name(item.context_expr) != "graph":
                    live.difference_update(self._names_by_ctx(item.optional_vars, ast.Store))
                live.update(self._names_by_ctx(item.context_expr, ast.Load))
            return live

        def live_before(
            statements: list[ast.stmt], live_out: set[str], loop_live: set[str], *, record: bool
        ) -> set[str]:
            live: set[str] = live_out.copy()
            for stmt in self._live_statements(statements, reverse=True):
                if isinstance(stmt, ast.If):
                    then_live = live_before(stmt.body, live, loop_live, record=record)
                    else_live = live_before(stmt.orelse, live, loop_live, record=record)
                    inputs = then_live | else_live | self._names_by_ctx(stmt.test, ast.Load)
                    # Even when both arms overwrite a live-out, they need a common target.
                    if record:
                        self._conditional_inputs[stmt] = live | inputs
                    live = inputs
                elif isinstance(stmt, (ast.For, ast.While)):
                    live = loop_live_before(stmt, live, loop_live, record=record)
                elif isinstance(stmt, ast.With):
                    live = scope_live_before(stmt, live, loop_live, record=record)
                elif isinstance(stmt, (ast.Break, ast.Continue)):
                    live = loop_live.copy()
                else:
                    if isinstance(stmt, ast.Return):
                        live.clear()
                    if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
                        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                        for target in targets:
                            live.difference_update(self._names_by_ctx(target, ast.Store))
                    elif isinstance(stmt, ast.AugAssign):
                        live.update(self._names_by_ctx(stmt.target, ast.Store))
                    live.update(self._names_by_ctx(stmt, ast.Load))
            return live

        live_before(statements, set(), set(), record=True)

    @contextmanager
    def _conditional_scope(self, stmt: ast.If | ast.While) -> Iterator[None]:
        """Treat visible bindings as incoming on every path through an if or while."""
        # Sibling temporaries can have different types. Share their spelling
        # only when the conditional reads the predecessor or exports the result.
        incoming = self._conditional_inputs.get(stmt)
        sibling_depths = {
            name: depth
            for name, depth in self._assign_depth.items()
            if depth > self._scope_depth and (incoming is None or name in incoming)
        }
        self._assign_depth.update(dict.fromkeys(sibling_depths, self._scope_depth))
        self._scope_depth += 1
        outer_optional_depth, self._optional_depth = self._optional_depth, self._scope_depth
        try:
            yield
        finally:
            self._optional_depth = outer_optional_depth
            self._scope_depth -= 1
            self._assign_depth.update(sibling_depths)

    def visit_While(self, node: ast.While) -> ast.stmt | list[ast.stmt]:
        """Keep while-body writes on the binding used by the loop condition."""
        if node in self._proven_empty_loops or (self._shape_inlined and self._empty_loop(node, set())):
            return self._visit_scoped_body(node.orelse)
        variant_tensors = self._loop_variant_tensors(node)
        self._forget_variant_tensors(variant_tensors)
        bridges, recovered = self._materialize_folded_writes([node])
        node.test = self.visit(node.test)
        with self._conditional_scope(node):
            outer_runtime_names = self._while_runtime_names
            if not self._simt_depth:
                # An incoming scalar needs a real store on each taken iteration.
                # A fresh sibling temporary can still fold to a constant shape.
                written = self._statement_tensor_writes(node).keys() & self._assign_depth.keys()
                self._while_runtime_names = outer_runtime_names | {
                    name
                    for name in written
                    if self._assign_depth[name] < self._scope_depth and self._active_shape_fold(name) is None
                }
            try:
                node.body = self._visit_scoped_body(node.body)
            finally:
                self._while_runtime_names = outer_runtime_names
        self._forget_variant_tensors(variant_tensors)
        if node.orelse:
            node.orelse = self._visit_scoped_body(node.orelse)
        self._restore_shape_folds(recovered)
        return [*bridges, node] if bridges else node

    @staticmethod
    def _is_attribute_call(expr: ast.expr | None, name: str) -> bool:
        return isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == name

    @classmethod
    def selected_yield_assignments(
        cls, statements: list[ast.stmt]
    ) -> Iterator[tuple[ast.Assign | ast.AnnAssign, ast.expr]]:
        """Yield the ordinary RHS for each explicit result of a selected if arm."""
        for stmt in statements:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                value = stmt.value
                if not isinstance(value, ast.Call) or not cls._is_attribute_call(value, "yield_"):
                    continue
                if isinstance(stmt, ast.Assign) and len(stmt.targets) != 1:
                    continue
                target = stmt.targets[0] if isinstance(stmt, ast.Assign) else stmt.target
                if isinstance(target, ast.Tuple) and len(target.elts) == len(value.args):
                    yield stmt, ast.copy_location(ast.Tuple(elts=value.args, ctx=ast.Load()), value)
                elif isinstance(target, ast.Name) and len(value.args) == 1:
                    yield stmt, value.args[0]
            # These parser scopes are transparent to the enclosing if's yields.
            # Nested ifs and real loops keep ownership of their own yields.
            elif isinstance(stmt, ast.With) and any(
                cls._is_attribute_call(item.context_expr, name)
                for item in stmt.items
                for name in ("scope", "manual_scope")
            ):
                yield from cls.selected_yield_assignments(stmt.body)
            elif isinstance(stmt, ast.For) and cls._is_attribute_call(stmt.iter, "split_aiv"):
                yield from cls.selected_yield_assignments(stmt.body)

    def _visit_isolated_branch(
        self, statements: list[ast.stmt], *, materialize: bool = False
    ) -> list[ast.stmt]:
        """Transform a branch whose local bindings do not escape."""
        state = (
            self._meta.copy(),
            self._shape_inlined.copy(),
            self._assign_count.copy(),
            self._assign_depth.copy(),
            self._var_renames.copy(),
        )
        graph_captures = self._graph_captures
        if graph_captures is not None:
            self._graph_captures = graph_captures.copy()
        try:
            prefix, _ = self._materialize_folded_writes(statements) if materialize else ([], {})
            return prefix + self._visit_scoped_body(statements)
        finally:
            self._meta, self._shape_inlined, self._assign_count, self._assign_depth, self._var_renames = state
            self._graph_captures = graph_captures

    def _visit_literal_if(self, node: ast.If, condition: bool) -> list[ast.stmt]:
        """Emit only the live arm, binding explicit results without an IfStmt."""
        selected = node.body if condition else node.orelse
        for assignment, value in self.selected_yield_assignments(selected):
            assignment.value = value
        return self._visit_scoped_body(selected)

    def _visit_yielding_if(self, node: ast.If, yields: dict[ast.Assign | ast.AnnAssign, list[str]]) -> ast.If:
        # The parser exports only yield targets. Give both arms the same
        # output names while keeping their temporaries inside each arm.
        recovered = {}
        if self._shape_inlined:
            recovered = {
                name: value
                for name, value in self._branch_output_constants(node, set()).items()
                if self._active_shape_fold(name) == value
            }
        self._scope_depth += 1
        try:
            node.body = self._visit_isolated_branch(node.body, materialize=True)
            if node.orelse:
                node.orelse = self._visit_isolated_branch(node.orelse, materialize=True)
        finally:
            self._scope_depth -= 1
        published: dict[str, str] = {}
        for stmt, names in yields.items():
            for target, name in zip(self._yield_stores(stmt), names, strict=True):
                if name not in published:
                    published[name] = self._bind_assign_name(ast.Name(id=name, ctx=ast.Store()), None).id
                    self._shape_inlined.pop(name, None)
                target.id = published[name]
        self._restore_shape_folds(recovered)
        return node

    @classmethod
    def _yield_bindings(cls, statements: list[ast.stmt]) -> dict[ast.Assign | ast.AnnAssign, list[str]]:
        from pypto.language.parser.control_flow_parser import ControlFlowParser  # noqa: PLC0415

        return {
            stmt: [target.id for target in cls._yield_stores(stmt)]
            for stmt in ControlFlowParser.iter_yield_assignments(statements)
        }

    def visit_If(self, node: ast.If) -> ast.stmt | list[ast.stmt]:
        """Visit If; treat the two branches as mutually-exclusive sibling scopes.

        A name first-assigned inside the then-branch is branch-local: the
        else-branch can never *read* it, so an else-branch assignment of the
        same name is a fresh local, not a rebind.  Drop then-branch-introduced
        names before visiting the else-branch so it re-declares them cleanly —
        no cross-branch rename.  Without this the else-branch assignment is
        treated as a same-depth rebind (then/else share one ``_scope_depth``)
        and aliased to ``x_v1``, which is then read out of its defining branch.

        This mirrors :meth:`_classify_loop_locals` for sibling loops.  Per-branch
        leaks and the conditional merge are handled by the Parser
        (``exit_scope(leak_vars=True)``) and ConvertToSSA, as for ``@pl.program``.
        """
        node.test = self.visit(node.test)
        capture_condition = _constant_condition(node.test)
        if capture_condition is not None and any(
            isinstance(item, ast.withitem) and isinstance(item.optional_vars, ast.Name)
            for stmt in (*node.body, *node.orelse)
            for item in ast.walk(stmt)
        ):
            # Keep both bodies for the parser's yield handling. Isolate the
            # inactive capture so it cannot overwrite the live binding.
            self._scope_depth += 1
            if capture_condition:
                if node.orelse:
                    node.orelse = self._visit_isolated_branch(node.orelse)
                node.body = self._visit_scoped_body(node.body)
            else:
                node.body = self._visit_isolated_branch(node.body)
                if node.orelse:
                    node.orelse = self._visit_scoped_body(node.orelse)
            self._scope_depth -= 1
            return node

        condition = (
            capture_condition if capture_condition is not None else values.literal_condition(node.test)
        )
        if condition is not None:
            return self._visit_literal_if(node, condition)
        statements = [*node.body, *node.orelse]
        yields = self._yield_bindings(statements)
        if yields:
            return self._visit_yielding_if(node, yields)
        bridges, recovered = self._materialize_folded_writes([node])
        with self._conditional_scope(node):
            self._visit_if_branches(node)
        self._restore_shape_folds(recovered)
        return [*bridges, node] if bridges else node

    def _publish_branch_alias(self, body: list[ast.stmt], alias: str, value: str | int) -> None:
        """Keep copies on executable paths with a control transfer at the tail."""
        last = body[-1]
        if not self._terminates_suite(last):
            body.append(self._copy_binding(alias, value))
        elif isinstance(last, ast.If):
            self._emitted_writes.pop(last, None)
            self._publish_branch_alias(last.body, alias, value)
            if last.orelse:
                self._publish_branch_alias(last.orelse, alias, value)
        elif isinstance(last, ast.With):
            self._emitted_writes.pop(last, None)
            self._publish_branch_alias(last.body, alias, value)
        else:
            body.insert(-1, self._copy_binding(alias, value))

    def _definite_branch_writes(self, body: list[ast.stmt]) -> tuple[set[str], bool]:
        """Summarize finalized branch exits once, including any emitted alias copies."""
        written: set[str] = set()
        for stmt in self._live_statements(body):
            if stmt not in self._emitted_writes:
                if isinstance(stmt, ast.If):
                    left, then_reaches = self._definite_branch_writes(stmt.body)
                    right, else_reaches = self._definite_branch_writes(stmt.orelse)
                    if then_reaches == else_reaches:
                        definite = left & right
                    else:
                        definite = left if then_reaches else right
                    reaches = not self._terminates_suite(stmt)
                else:
                    definite = set[str]()
                    reaches = self._collect_upward_exposed([stmt], definite, set(), may_skip_loops=True)
                self._emitted_writes[stmt] = definite, reaches
            definite, reaches = self._emitted_writes[stmt]
            written.update(definite)
            if not reaches:
                return written, False
        return written, True

    def _visit_if_branches(self, node: ast.If) -> None:
        """Merge implicit outputs of runtime branches at their shared scope depth."""
        before = set(self._assign_count)
        written = self._source_stores.matching(node, before)

        def live_meta(name: str) -> TensorMeta | None:
            current = self._known_tensor_name(name)
            return self._meta.get(current) if current is not None else None

        def set_live_meta(name: str, meta: TensorMeta | None) -> None:
            current = self._var_renames.get(name, name)
            if current == name and name in self._meta and meta is None:
                self._var_renames[name] = name
            else:
                self._record_tensor_meta(current, meta)
                if meta is not None and self._var_renames.get(name) == name:
                    self._var_renames.pop(name)

        incoming = {name: live_meta(name) for name in written}
        node.body = self._visit_scoped_body(node.body)
        then_tensors = {name: live_meta(name) for name in written}
        if not node.orelse:
            for name, meta in incoming.items():
                set_live_meta(name, merge_tensor_metas(meta, then_tensors[name]))
        else:
            for name, meta in incoming.items():
                set_live_meta(name, meta)
            # then-branch locals are invisible to the else-branch: forget them
            # so a same-named else-branch assignment is a fresh first binding.
            then_locals = {name: self._binding_value(name) for name in set(self._assign_count) - before}
            then_meta = {
                name: self._meta.get(value) for name, value in then_locals.items() if isinstance(value, str)
            }
            for name in then_locals:
                self._forget_rename_bookkeeping(name)
                self._shape_inlined.pop(name, None)
            node.orelse = self._visit_scoped_body(node.orelse)
            for name, meta in then_tensors.items():
                set_live_meta(name, merge_tensor_metas(meta, live_meta(name)))
            candidates: dict[str, str | int] = {}
            for name in then_locals.keys() & self._assign_count.keys():
                right = self._binding_value(name)
                if then_locals[name] != right:
                    candidates[name] = right
                elif isinstance(right, str):
                    # The arms can reuse a spelling for different tensors.
                    self._record_tensor_meta(
                        right, merge_tensor_metas(then_meta.get(name), self._meta.get(right))
                    )
            if not candidates:
                return
            # A visited alias can belong to just one arm of a nested if. A
            # merge copy reads its source unconditionally at the branch exit.
            then_written, _ = self._definite_branch_writes(node.body)
            else_written, _ = self._definite_branch_writes(node.orelse)
            for name, right in sorted(candidates.items()):
                left = then_locals[name]
                if (isinstance(left, str) and left not in then_written) or (
                    isinstance(right, str) and right not in else_written
                ):
                    continue
                merged = self._rebind(name)
                self._publish_branch_alias(node.body, merged, left)
                self._publish_branch_alias(node.orelse, merged, right)
                if isinstance(right, str):
                    self._record_tensor_meta(
                        merged, merge_tensor_metas(then_meta.get(name), self._meta.get(right))
                    )
                self._shape_inlined.pop(name, None)
                self._assign_depth[name] = self._scope_depth - 1

    def _visit_with_captures(self, node: ast.With) -> tuple[dict[str, str], list[ast.stmt]]:
        """Rewrite capture names and materialize folded host predecessors."""
        captures: dict[str, str] = {}
        prefix: list[ast.stmt] = []
        for item in node.items:
            item.context_expr = self.visit(item.context_expr)
            if isinstance(item.optional_vars, ast.Name):
                self.has_scope_capture = True
                original = item.optional_vars
                # Use a fresh capture alias when the caller only has a
                # folded literal, so the parser can restore that predecessor.
                folded = self._active_shape_fold(original.id)
                captured = (
                    self._rebind(original.id)
                    if self._graph_captures is not None and folded is not None
                    else self._var_renames.get(original.id, original.id)
                )
                # An outer dispatch may be followed by a write to this scalar.
                # Materialize the folded predecessor so the parser can restore
                # it when the temporary TaskId binding is overwritten.
                if self._graph_captures is None and folded is not None:
                    assignment = ast.Assign(
                        targets=[ast.Name(id=captured, ctx=ast.Store())], value=ast.Constant(value=folded)
                    )
                    prefix.append(ast.copy_location(assignment, node))
                if self._graph_captures is not None:
                    self._graph_captures[original.id] = captured
                item.optional_vars = ast.copy_location(ast.Name(id=captured, ctx=ast.Store()), original)
                captures[original.id] = captured
                if _scope_call_name(item.context_expr) != "graph":
                    self._var_renames[original.id] = captured
                    self._shape_inlined.pop(original.id, None)
                    self._record_tensor_meta(captured, None)
            elif item.optional_vars is not None:
                item.optional_vars = self.visit(item.optional_vars)
        return captures, prefix

    def visit_With(self, node: ast.With) -> ast.stmt | list[ast.stmt]:
        """Visit With block, incrementing scope depth for its body."""
        scope_names = _with_scope_names(node)
        graph = "graph" in scope_names
        captures, capture_prefix = self._visit_with_captures(node)
        simt = "simt_scope" in scope_names
        # SIMT assignments are per-thread. Folding a local shape or renaming a
        # shadowed scalar must not change the caller's later specialization.
        # Generated-name reservations and diagnostic aliases remain global to
        # this function so distinct regions never reuse a generated name.
        state = None
        if simt or graph:
            state = (
                self._meta.copy(),
                self._shape_inlined.copy(),
                self._assign_count.copy(),
                self._assign_depth.copy(),
                self._var_renames.copy(),
                self._simt_runtime_names,
            )
        if simt:
            self._simt_runtime_names = set()
            pending: list[tuple[ast.AST, bool]] = [(stmt, False) for stmt in node.body]
            while pending:
                current, conditional = pending.pop()
                conditional = conditional or isinstance(current, (ast.If, ast.For, ast.While))
                if conditional and isinstance(current, ast.Name) and isinstance(current.ctx, ast.Store):
                    self._simt_runtime_names.add(current.id)
                pending.extend((child, conditional) for child in ast.iter_child_nodes(current))
            # A private binding is a runtime scalar even when its incoming
            # value came from a folded shape. Materialize live-in constants,
            # and retain conditionally rebound scalar assignments so conditional
            # writes cannot leave either stale constants or undefined names.
            written = self._names_by_ctx(node, ast.Store)
            exposed: set[str] = set()
            self._collect_upward_exposed(node.body, set(), exposed, may_skip_loops=True)
            incoming = [(name, value) for name, value in self._shape_inlined.items() if name in written]
            prefix: list[ast.stmt] = []
            for name, value in incoming:
                self._shape_inlined.pop(name)
                if name in exposed:
                    assignment = ast.Assign(
                        targets=[ast.Name(id=name, ctx=ast.Store())], value=ast.Constant(value=value)
                    )
                    prefix.append(ast.fix_missing_locations(ast.copy_location(assignment, node)))
            node.body = prefix + node.body
            self._simt_depth += 1
        previous_graph = self._graph_captures
        if graph:
            self._graph_captures = {}
        self._scope_depth += 1
        try:
            node.body = self._visit_scoped_body(node.body)
        finally:
            self._scope_depth -= 1
            graph_locals = self._graph_captures if graph else None
            self._graph_captures = previous_graph
            if simt:
                self._simt_depth -= 1
            if state is not None and simt:
                (
                    self._meta,
                    self._shape_inlined,
                    self._assign_count,
                    self._assign_depth,
                    self._var_renames,
                    self._simt_runtime_names,
                ) = state
            elif state is not None and graph_locals:
                old_meta, old_shapes, old_counts, old_depths, old_renames, _ = state
                # Keep a new shape fold or a later alpha-renamed live-out.
                # The parser handles any alpha-renamed local TaskIds.
                graph_locals = {
                    name
                    for name in graph_locals
                    if self._active_shape_fold(name) is None
                    and (
                        (current := self._var_renames.get(name, name)) == old_renames.get(name, name)
                        or current not in self.graph_rebinds
                    )
                }
                meta_names = (
                    graph_locals
                    | {old_renames.get(name, name) for name in graph_locals}
                    | {self._var_renames.get(name, name) for name in graph_locals}
                )
                _restore_bindings(self._meta, old_meta, meta_names)
                _restore_bindings(self._shape_inlined, old_shapes, graph_locals)
                _restore_bindings(self._assign_count, old_counts, graph_locals)
                _restore_bindings(self._assign_depth, old_depths, graph_locals)
                _restore_bindings(self._var_renames, old_renames, graph_locals)
                # The parser may recover a tensor written by the captured
                # dispatch under the old spelling. Resolve later shape and
                # dtype reads from that typed binding.
                for name in graph_locals:
                    self._meta.pop(self._var_renames.get(name, name), None)
        # A capture binds the dispatch result, not a body-local reassignment
        # of the same spelling. Do not create another alpha version here:
        # the parser reuses the source Var and native SSA builds carries/phis.
        for original, captured in captures.items():
            self._var_renames[original] = captured
            self._shape_inlined.pop(original, None)
            self._record_tensor_meta(captured, None)
        return [*capture_prefix, node] if capture_prefix else node


class _PeerReferenceTransformer(ast.NodeTransformer):
    """Rewrite symbolic peer names after ordinary body specialization."""

    def __init__(
        self, names: Mapping[str, str | None], py_globals: Mapping[str, Any], *, ambiguous_scope: bool = False
    ) -> None:
        self._peer_func_names = names
        self._py_globals = py_globals
        self._ambiguous_scope = ambiguous_scope

    def visit_Call(self, node: ast.Call) -> ast.Call:
        node = cast(ast.Call, self.generic_visit(node))
        # Only DSL imports carry Group peer references; unrelated user
        # functions may also have a peer_func keyword.
        if not any(kw.arg == "peer_func" for kw in node.keywords):
            return node
        attrs: list[str] = []
        root: ast.expr = node.func
        while isinstance(root, ast.Attribute):
            attrs.append(root.attr)
            root = root.value
        target = self._py_globals.get(root.id) if isinstance(root, ast.Name) else None
        for attr in reversed(attrs):
            target = getattr(target, attr, None)
        if (
            getattr(target, "__module__", None)
            not in {"pypto.ir.op.system_ops", "pypto.language.op.system_ops"}
            or getattr(target, "__name__", None) != "import_peer_buffer"
        ):
            return node
        for kw in node.keywords:
            if kw.arg != "peer_func" or not isinstance(kw.value, ast.Constant):
                continue
            name = kw.value.value
            if not isinstance(name, str):
                continue
            if self._ambiguous_scope:
                raise ValueError(f"peer_func={name!r} is ambiguous across groups; use separate peer helpers")
            if name not in self._peer_func_names:
                continue
            generated = self._peer_func_names[name]
            if generated is None:
                raise ValueError(f"peer_func={name!r} is ambiguous across groups; use separate peer helpers")
            kw.value = ast.copy_location(ast.Constant(value=generated), kw.value)
        return node


def remap_peer_references(
    body: list[ast.stmt],
    names: Mapping[str, str | None],
    py_globals: Mapping[str, Any],
    *,
    ambiguous_scope: bool = False,
) -> None:
    """Resolve Group-local symbols in one additional linear AST walk."""
    if not names and not ambiguous_scope:
        return
    transformer = _PeerReferenceTransformer(names, py_globals, ambiguous_scope=ambiguous_scope)
    for stmt in body:
        transformer.visit(stmt)


# The pair denotes either a tensor binding (axis=None) or one of its dimensions.
_ShapeBinding = tuple[str, int | None]


class _ShapeAliasInliner(ast.NodeTransformer):
    """Expose dominating IR tensor dimensions at their scalar uses before parsing."""

    def __init__(self, transformer: BodyTransformer, sources: set[str]):
        self.transformer = transformer
        self.sources = sources
        self.bindings: dict[str, _ShapeBinding] = {}
        self.loop_exits: list[dict[str, _ShapeBinding]] = []

    def visit_suite(self, statements: list[ast.stmt]) -> bool:
        for statement in statements:
            self.visit(statement)
            if BodyTransformer._terminates_suite(statement):
                return False
        return True

    def _forget(self, names: set[str]) -> None:
        for name, (source, _) in tuple(self.bindings.items()):
            if name in names or source in names:
                del self.bindings[name]

    @staticmethod
    def _stores(node: ast.AST) -> set[str]:
        return BodyTransformer._names_by_ctx(node, ast.Store)

    @staticmethod
    def _shape_read(name: str, axis: int, location: ast.expr) -> ast.Subscript:
        tensor = ast.Name(id=name, ctx=ast.Load())
        shape = ast.Attribute(value=tensor, attr="shape", ctx=ast.Load())
        return ast.copy_location(
            ast.Subscript(value=shape, slice=ast.Constant(value=axis), ctx=ast.Load()), location
        )

    def _tensor_source(self, expr: ast.expr) -> str | None:
        if isinstance(expr, ast.Name):
            bound = self.bindings.get(expr.id)
            if bound is not None and bound[1] is None:
                return bound[0]
            if expr.id in self.sources:
                return expr.id
        return None

    def _assigned_bindings(self, target: ast.expr, rhs: ast.expr) -> Iterator[tuple[str, _ShapeBinding]]:
        shape = rhs.value if isinstance(rhs, ast.Subscript) else rhs
        if (
            isinstance(shape, ast.Attribute)
            and shape.attr == "shape"
            and (src := self._tensor_source(shape.value))
        ):
            if isinstance(target, ast.Tuple) and rhs is shape:
                for axis, elt in enumerate(target.elts):
                    if isinstance(elt, ast.Name):
                        yield elt.id, (src, axis)
            elif isinstance(target, ast.Name) and isinstance(rhs, ast.Subscript):
                if isinstance(rhs.slice, ast.Constant) and type(rhs.slice.value) is int:
                    yield target.id, (src, rhs.slice.value)
        elif isinstance(target, ast.Name):
            if isinstance(rhs, ast.Name) and rhs.id in self.bindings:
                yield target.id, self.bindings[rhs.id]
            elif src := self._tensor_source(rhs):
                yield target.id, (src, None)
            elif (read := self.transformer._untyped_shape_reads.get(rhs)) and read[0] in self.sources:
                yield target.id, read

    def _forget_loop_writes(self, node: ast.For | ast.While, writes: set[str]) -> None:
        # A backedge can preserve an incoming dimension even when a conditional
        # exits before the current iteration writes it again.
        candidates = {
            name: value for name, value in self.bindings.items() if name in writes and value[0] not in writes
        }
        self._forget(writes)
        if not candidates:
            return
        unproven = Counter(
            part.id
            for part in ast.walk(node)
            if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store)
        )
        for part in ast.walk(node):
            if isinstance(part, (ast.Assign, ast.AnnAssign)) and part.value is not None:
                targets = part.targets if isinstance(part, ast.Assign) else [part.target]
                for target in targets:
                    for name, value in self._assigned_bindings(target, part.value):
                        if candidates.get(name) == value:
                            unproven[name] -= 1
        self.bindings.update((name, value) for name, value in candidates.items() if unproven[name] == 0)

    @staticmethod
    def _join(left: dict[str, _ShapeBinding], right: dict[str, _ShapeBinding]) -> dict[str, _ShapeBinding]:
        # When each arm publishes its tensor into the same result, a scalar
        # holding the selected tensor's dimension can use that result's type.
        shared = left.keys() & right.keys()
        tensors = {
            (left[name][0], right[name][0]): name
            for name in sorted(shared)
            if left[name][1] is None and right[name][1] is None
        }
        result: dict[str, _ShapeBinding] = {}
        for name in shared:
            lhs, axis = left[name]
            rhs, other_axis = right[name]
            source = lhs if lhs == rhs else tensors.get((lhs, rhs))
            if source is not None and axis == other_axis:
                result[name] = source, axis
        return result

    def visit_Name(self, node: ast.Name) -> ast.expr:
        fact = self.bindings.get(node.id)
        if isinstance(node.ctx, ast.Load) and fact is not None and fact[1] is not None:
            return self._shape_read(fact[0], fact[1], node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.expr:
        fact = self.transformer._untyped_shape_reads.get(node)
        if fact is not None and fact[0] in self.sources:
            return self._shape_read(fact[0], fact[1], node)
        return cast("ast.expr", self.generic_visit(node))

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        self.generic_visit(node)
        self._forget({name for target in node.targets for name in self._stores(target)})
        for target in node.targets:
            self.bindings.update(self._assigned_bindings(target, node.value))
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        self.generic_visit(node)
        if node.value is not None:
            self._forget(self._stores(node.target))
            self.bindings.update(self._assigned_bindings(node.target, node.value))
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AugAssign:
        self.generic_visit(node)
        self._forget(self._stores(node.target))
        return node

    def visit_If(self, node: ast.If) -> ast.If:
        node.test = self.visit(node.test)
        incoming = self.bindings.copy()
        then_reaches = self.visit_suite(node.body)
        left, self.bindings = self.bindings, incoming
        else_reaches = self.visit_suite(node.orelse)
        if then_reaches and else_reaches:
            self.bindings = self._join(left, self.bindings)
        elif then_reaches:
            self.bindings = left
        return node

    def visit_Break(self, node: ast.Break | ast.Continue) -> ast.Break | ast.Continue:
        self.loop_exits.append(self.bindings.copy())
        return node

    visit_Continue = visit_Break

    @staticmethod
    def _predicated(expr: ast.expr) -> bool:
        return isinstance(expr, ast.Call) and any(
            kw.arg == "predicate" and not (isinstance(kw.value, ast.Constant) and kw.value.value is None)
            for kw in expr.keywords
        )

    def _loop_enters(self, node: ast.For | ast.While) -> bool | None:
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and _scope_call_name(node.iter) == "spmd"
        ):
            count = _constant_scalar(node.iter.args[0]) if node.iter.args else None
            return count > 0 if type(count) is int and not self._predicated(node.iter) else None
        return self.transformer._loop_enters(node)

    def _visit_loop(self, node: ast.For | ast.While) -> ast.For | ast.While:
        if isinstance(node, ast.For):
            node.iter = self.visit(node.iter)
        enters = self._loop_enters(node)
        outputs = self.transformer._loop_yield_targets(node.body)
        private = isinstance(node, ast.For) and self.transformer._private_loop_body(node, outputs)
        incoming = self.bindings.copy()
        writes = self._stores(node)
        self._forget_loop_writes(node, writes)
        if isinstance(node, ast.While):
            node.test = self.visit(node.test)
        outer_exits, self.loop_exits = self.loop_exits, []
        if self.visit_suite(node.body):
            self.loop_exits.append(self.bindings)
        paths, self.loop_exits = self.loop_exits, outer_exits
        if private or enters is False:
            self.bindings = incoming
            self._forget(outputs)
        else:
            if enters is None:
                paths.append(incoming)
            self.bindings = paths.pop() if paths else {}
            for path in paths:
                self.bindings = self._join(self.bindings, path)
        if isinstance(node, ast.For) and not private and enters is not False:
            # Restore bindings shadowed by the induction variable, provided
            # the loop has not overwritten their scalar or tensor source.
            targets = self._stores(node.target)
            exposed_writes = writes - targets
            self._forget(targets)
            self.bindings.update(
                (name, fact)
                for name, fact in incoming.items()
                if (name in targets or fact[0] in targets)
                and name not in exposed_writes
                and fact[0] not in exposed_writes
            )
        self.visit_suite(node.orelse)
        return node

    visit_For = _visit_loop
    visit_While = _visit_loop

    def visit_With(self, node: ast.With) -> ast.With:
        private = "simt_scope" in _with_scope_names(node)
        predicated = any(self._predicated(item.context_expr) for item in node.items)
        incoming = self.bindings.copy() if private or predicated else None
        for item in node.items:
            item.context_expr = self.visit(item.context_expr)
            if item.optional_vars is not None and _scope_call_name(item.context_expr) != "graph":
                self._forget(self._stores(item.optional_vars))
        self.visit_suite(node.body)
        if incoming is not None:
            self.bindings = incoming if private else self._join(incoming, self.bindings)
        for item in node.items:
            if item.optional_vars is not None:
                self._forget(self._stores(item.optional_vars))
        return node
