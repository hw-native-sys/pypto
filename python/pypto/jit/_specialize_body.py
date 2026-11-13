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
import copy
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from pypto.language.parser.ast_parser import ASTParser
from pypto.language.parser.control_flow_parser import ControlFlowParser
from pypto.language.parser.diagnostics import ParserError
from pypto.pypto_core import arith, ir

from . import _specialize_values as values
from ._call_syntax import SUBMIT_DISPATCH_KEYWORDS, dependency_call, submit_target
from ._specialize_types import DynDim, TensorMeta

_BindingValue = TypeVar("_BindingValue")


@dataclass(frozen=True)
class _TensorBinding:
    """Descriptor and object kind of one emitted tensor value."""

    meta: TensorMeta
    distributed: bool


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


def _constant_condition(node: ast.expr) -> bool | None:
    """Resolve a literal-only condition using the DSL's expression semantics."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, bool) else None
    if not isinstance(node, (ast.Compare, ast.BoolOp, ast.UnaryOp)):
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

    # Names and calls would need the actual function scope. Let the normal
    # parser handle those conditions, and retain diagnostics for invalid DSL.
    try:
        condition = ASTParser("<JIT condition>", []).parse_expression(ast.fix_missing_locations(node))
        folded = arith.Analyzer().simplify(condition)
    except (ParserError, TypeError, ValueError, OverflowError):
        return None
    return folded.value if isinstance(folded, ir.ConstBool) else None


def _literal_if_body(node: ast.If) -> list[ast.stmt] | None:
    """Select the same literal arm during write analysis and body rewriting."""
    condition = values.literal_condition(node.test)
    if condition is None:
        condition = _constant_condition(node.test)
    return None if condition is None else node.body if condition else node.orelse


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
        dep_param_names: dict[str, list[str]] | None = None,
        dep_func_names: dict[str, str] | None = None,
        constexpr_values: Mapping[str, str] | None = None,
        dep_constexpr_params: dict[str, set[str]] | None = None,
        dep_call_targets: Mapping[int, str] | None = None,
    ) -> None:
        super().__init__()
        self._meta = {name: _TensorBinding(meta, False) for name, meta in tensor_meta.items()}
        self._constexpr = dict(constexpr_values or {})
        self._dep_constexpr_params = dep_constexpr_params or {}
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
        # Maps dep_name → ordered parameter list. Used by ``visit_Call`` to
        # normalise keyword args to positional, so generated
        # ``self.<dep>(a, out=out)`` calls become parser-accepted
        # ``self.<dep>(a, out)``.
        self._dep_param_names = dep_param_names or {}
        self._dep_parameter_returns: Mapping[str, tuple[str | None, ...]] = {}
        # Module-level globals from the originating function. Used by
        # ``visit_Name`` to inline imported int/float/bool constants
        # (e.g. ``BATCH = 16`` from a config module) at their use sites.
        self._py_globals = py_globals or {}
        # Maps local variable names (from `M, N = a.shape`) to their concrete
        # constant values when all dimensions are static.  Used by visit_Name
        # to inline constants and by visit_Assign to suppress the assignment.
        self._shape_inlined: dict[str, int] = {}
        # Alpha-renaming support: tracks how many times each local has been assigned
        # (so we can generate x_v1, x_v2, ... on rebindings).
        self._assign_count: dict[str, int] = {}
        # Maps local variable name → current (latest) renamed alias.
        # Empty until the variable is assigned a second time.
        self._var_renames: dict[str, str] = {}
        # Inactive regions reserve spellings without initializing a value on
        # their fallthrough path. A later first assignment needs a fresh name.
        self._guard_local_names: set[str] = set()
        # Reverse map: generated alias → original user variable name.
        # Used to rewrite error messages so users see their original names.
        self._alias_to_original: dict[str, str] = {}
        # Tracks the scope depth at which each variable was first assigned.
        # Rebindings at a deeper scope (e.g. inside a for/with) are loop-carried
        # updates and must NOT be renamed.
        self._assign_depth: dict[str, int] = {}
        self._scope_depth: int = 0
        # Yield targets name the enclosing control-flow result. Both branches
        # must use its incoming spelling, independent of body-local rebindings.
        self._yield_renames: dict[str, str] = {}
        self._branch_live_outs: dict[int, set[str]] = {}
        self._liveness_static_arms: dict[int, list[ast.stmt]] = {}
        self._loop_exit_bindings: dict[int, dict[str, tuple[str, int | None]]] = {}
        self._loop_exit_tensor_meta: dict[int, dict[str, _TensorBinding | None]] = {}
        self._initialized_loop_exports: dict[int, set[str]] = {}
        self._graph_captures: dict[str, str] | None = None
        self.graph_rebinds: dict[str, str | int] = {}
        self.has_scope_capture = False
        self._simt_depth: int = 0
        self._simt_runtime_names: set[str] = set()
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

    def _rebind(self, var_name: str, *, graph_predecessor: bool = True) -> str:
        """Generate a fresh name for a rebinding of ``var_name``.

        Returns the new name and updates ``_var_renames`` so subsequent reads
        of ``var_name`` resolve to the new alias.  Skips candidate names that
        are already used by the user to avoid collisions.
        """
        count = self._assign_count[var_name]
        new_name = f"{var_name}_v{count}"
        # Skip any candidate that collides with a user-defined name.
        while new_name in self._used_names:
            count += 1
            new_name = f"{var_name}_v{count}"
        self._assign_count[var_name] = count + 1
        if graph_predecessor and self._graph_captures is not None:
            folded = self._active_shape_fold(var_name)
            previous = self._var_renames.get(var_name, var_name)
            self.graph_rebinds[new_name] = previous if folded is None else folded
        self._var_renames[var_name] = new_name
        self._alias_to_original[new_name] = var_name
        self._used_names.add(new_name)
        return new_name

    @property
    def rename_map(self) -> dict[str, str]:
        """Return mapping from generated alias → original user variable name."""
        return dict(self._alias_to_original)

    @property
    def known_tensor_meta(self) -> dict[str, TensorMeta]:
        """Snapshot descriptors belonging to the emitted SSA spellings."""
        return {
            name: binding.meta for name, binding in self._meta.items() if self._var_renames.get(name) != name
        }

    @property
    def known_distributed_tensors(self) -> set[str]:
        """Snapshot distributed kinds from the same emitted-value bindings."""
        return {
            name
            for name, binding in self._meta.items()
            if binding.distributed and self._var_renames.get(name) != name
        }

    def initialize_tensor_bindings(
        self, names: set[str], dependency_returns: Mapping[str, tuple[str | None, ...]]
    ) -> None:
        """Seed signature kinds and proven callee parameter returns before rewriting."""
        for name in names & self._meta.keys():
            self._meta[name] = _TensorBinding(self._meta[name].meta, True)
        self._dep_parameter_returns = dependency_returns

    def _known_tensor_name(self, name: str) -> str | None:
        """Resolve known metadata without falling back to a rebound parameter."""
        current = self._var_renames.get(name, name)
        if current == name and name in self._var_renames:
            return None
        return current if current in self._meta else None

    def _record_tensor_meta(
        self, name: str, meta: _TensorBinding | None, *, reuses_name: bool = False
    ) -> None:
        """Keep metadata per emitted SSA name; invalidate conflicting scoped writes."""
        if not reuses_name:
            if meta is None:
                self._meta.pop(name, None)
            else:
                self._meta[name] = meta
        elif meta is None or meta != self._meta.get(name):
            if name in self._alias_to_original:
                self._meta.pop(name, None)
            elif name in self._meta:
                # Keep the original parameter metadata for its original version.
                self._var_renames[name] = name
        elif self._var_renames.get(name) == name:
            self._var_renames.pop(name)

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

    def _bind_assign_name(
        self, var_name: str, value_meta: _TensorBinding | None, *, yield_result: bool = False
    ) -> str:
        """Bind an assignment target after its RHS has been visited."""
        if yield_result:
            name = self._yield_renames.get(var_name, var_name)
            self._record_first_assign(var_name)
            if name == var_name:
                self._var_renames.pop(var_name, None)
            else:
                self._var_renames[var_name] = name
            self._record_tensor_meta(name, value_meta)
            return name
        if self._simt_depth:
            self._shape_inlined.pop(var_name, None)
        if var_name in self._assign_count:
            # A nested first binding must not force all later outer writes to
            # reuse one name, especially when a new value changes its type.
            # Captured Graph names instead reuse the tensor predecessor that
            # the parser restores after the temporary TaskId binding.
            if self._graph_captures is None or var_name not in self._graph_captures:
                self._assign_depth[var_name] = min(self._assign_depth[var_name], self._scope_depth)
            # Respect the shared scope decision, including Graph predecessors
            # that are folded constants rather than runtime bindings.
            if self._needs_alpha_rename(var_name):
                new_name = self._rebind(var_name)
                self._record_tensor_meta(new_name, value_meta)
                return new_name
            # Deeper scope: if the variable has an active rename (from a bridge
            # assignment), apply it to the LHS so the loop body consistently
            # uses the renamed alias (e.g. x_v1 = some_op(x_v1)).
            current_name = self._var_renames.get(var_name, var_name)
            self._record_tensor_meta(current_name, value_meta, reuses_name=True)
            return current_name
        # First assignment: record and keep original name.
        self._record_first_assign(var_name)
        if var_name in self._guard_local_names:
            name = self._rebind(var_name, graph_predecessor=False)
            self._record_tensor_meta(name, value_meta)
            return name
        self._record_tensor_meta(var_name, value_meta)
        return var_name

    def _dependency_result_operands(self, call: ast.Call) -> list[ast.expr | None] | None:
        """Map proven callee parameter returns onto this call's actual operands."""
        original = call
        submitted_target = submit_target(call)
        if submitted_target is not None:
            # Keep the original identity for per-call constexpr variants. The
            # first Submit operand names the callee; dispatch keywords do not
            # bind its formal parameters. Generated calls use self.<callee>.
            call = dependency_call(call) or ast.copy_location(
                ast.Call(
                    func=submitted_target,
                    args=call.args[1:],
                    keywords=[kw for kw in call.keywords if kw.arg not in SUBMIT_DISPATCH_KEYWORDS],
                ),
                call,
            )
        if isinstance(call.func, ast.Name) and call.func.id in self._dep_names:
            name = self._dep_call_targets.get(
                id(original), self._dep_func_names.get(call.func.id, call.func.id)
            )
            parameters = self._dep_param_names.get(name, [])
        elif (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"
        ):
            name = call.func.attr
            parameters = [
                param
                for param in self._dep_param_names.get(name, [])
                if param not in self._dep_constexpr_params.get(name, set())
            ]
        else:
            return None
        results = self._dep_parameter_returns.get(name)
        if results is None:
            return None
        arguments = dict(zip(parameters, call.args))
        arguments.update({kw.arg: kw.value for kw in call.keywords if kw.arg is not None})
        operands = [arguments.get(param) if param is not None else None for param in results]
        if submitted_target is None:
            return operands
        # Source syntax packs a multi-result Submit as ((a, b), task_id),
        # although its IR type is flat. A TaskId never carries tensor metadata.
        result = (
            operands[0]
            if len(operands) == 1
            else ast.Tuple(
                elts=[operand if operand is not None else ast.Constant(value=None) for operand in operands],
                ctx=ast.Load(),
            )
        )
        return [result, None]

    def _tensor_operand_meta(
        self, operand: ast.expr, snapshot: dict[int, _TensorBinding | None] | None = None
    ) -> _TensorBinding | None:
        """Snapshot a plain or single yielded tensor operand before rebinding."""
        if snapshot is not None and id(operand) in snapshot:
            return snapshot[id(operand)]
        if self._is_yield(operand) and isinstance(operand, ast.Call) and len(operand.args) == 1:
            operand = operand.args[0]
        if isinstance(operand, ast.Call):
            results = self._dependency_result_operands(operand)
            meta = (
                self._tensor_operand_meta(results[0], snapshot)
                if results is not None and len(results) == 1 and results[0] is not None
                else None
            )
        else:
            source = self._known_tensor_name(operand.id) if isinstance(operand, ast.Name) else None
            meta = self._meta.get(source) if source is not None else None
        if snapshot is not None:
            snapshot[id(operand)] = meta
        return meta

    def _visit_simple_assign(self, node: ast.Assign) -> ast.stmt:
        """Handle a single-target name assignment with alpha-renaming support."""
        target = cast(ast.Name, node.targets[0])
        value_meta = self._tensor_operand_meta(node.value)
        node.value = self.visit(node.value)
        name = self._bind_assign_name(target.id, value_meta, yield_result=self._is_yield(node.value))
        self._shape_inlined.pop(target.id, None)
        node.targets = [ast.copy_location(ast.Name(id=name, ctx=ast.Store()), target)]
        return node

    @staticmethod
    def _is_yield(value: ast.expr | None) -> bool:
        return (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "yield_"
        )

    def _visit_unpack_target(
        self,
        target: ast.expr,
        *,
        yield_result: bool = False,
        value: ast.expr | None = None,
        operand_meta: dict[int, _TensorBinding | None] | None = None,
    ) -> ast.expr:
        """Bind unpacked names left-to-right, including nested submit results."""
        if yield_result and isinstance(value, ast.Call):
            # Explicit result metadata belongs to the yielded operands, rather
            # than to the yield call itself. Snapshot before any LHS binds.
            value = ast.Tuple(elts=value.args, ctx=ast.Load())
        if operand_meta is None:
            # The RHS names have already been rewritten. Snapshot their
            # metadata before scoped/repeated targets can overwrite a source.
            operand_meta = {}
            if value is not None:
                for operand in ast.walk(value):
                    if isinstance(operand, (ast.Name, ast.Call)):
                        self._tensor_operand_meta(operand, operand_meta)
        if isinstance(target, ast.Name):
            if (
                not yield_result
                and isinstance(value, ast.Constant)
                and type(value.value) is int
                and value.value == self._active_shape_fold(target.id)
            ):
                # A tuple may write back the exact folded dimension. Retain
                # the literal for compile-time consumers outside this region.
                return target
            # The RHS was already visited. Binding must still see a folded
            # predecessor so Graph can record its restoration value before
            # the old shape fold is invalidated.
            name = self._bind_assign_name(target.id, operand_meta.get(id(value)), yield_result=yield_result)
            self._shape_inlined.pop(target.id, None)
            return ast.copy_location(ast.Name(id=name, ctx=ast.Store()), target)
        if isinstance(target, ast.Tuple):
            operands = (
                list(value.elts)
                if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(target.elts)
                else results
                if isinstance(value, ast.Call)
                and (results := self._dependency_result_operands(value)) is not None
                and len(results) == len(target.elts)
                else [None] * len(target.elts)
            )
            target.elts = [
                self._visit_unpack_target(
                    elt, yield_result=yield_result, value=operand, operand_meta=operand_meta
                )
                for elt, operand in zip(target.elts, operands, strict=True)
            ]
            return target
        return self.visit(target)

    def _visit_chained_assign(self, node: ast.Assign) -> list[ast.stmt]:
        """Evaluate a chain's RHS once before rebinding and copying to its targets."""
        first_name = cast(ast.Name, node.targets[0]).id
        value = node.value
        assignments: list[ast.stmt] = []
        for target in node.targets:
            assignment = ast.copy_location(ast.Assign(targets=[target], value=value), node)
            assignments.append(self._visit_simple_assign(assignment))
            # Clear an old inlined shape after reading the original RHS.
            self._shape_inlined.pop(cast(ast.Name, target).id, None)
            value = ast.copy_location(ast.Name(id=first_name, ctx=ast.Load()), node.value)
        return assignments

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
        meta = self._meta[param_name].meta
        elts = [self._shape_dim_node(param_name, i) for i in range(len(meta.shape))]
        return ast.Tuple(elts=elts, ctx=ast.Load())

    def _shape_dim_node(self, param_name: str, dim_idx: int) -> ast.expr:
        d = self._meta[param_name].meta.shape[dim_idx]
        # Known metadata belongs to this exact emitted value, including safe
        # tuple aliases. Renaming alone does not make a static dimension dynamic.
        if isinstance(d, DynDim):
            return self._dyn_dim_expr(param_name, dim_idx)
        return ast.Constant(value=d)

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

    def _expand_shape_unpacking(
        self, node: ast.Assign, param_name: str, target_names: list[ast.expr]
    ) -> list[ast.stmt]:
        """Expand known shape dimensions without introducing tuple values."""
        meta = self._meta[param_name].meta
        stmts: list[ast.stmt] = []
        for i, (tgt, dim) in enumerate(zip(target_names, meta.shape, strict=True)):
            if not isinstance(tgt, ast.Name):
                continue
            if tgt.id in self._var_renames or (self._simt_depth and tgt.id in self._simt_runtime_names):
                # Keep the current binding updated after scalar renaming or a
                # conditional write. Folding only the source name would leave
                # reads pointing at its old alias. Expand tuples into scalars
                # because a tuple temporary is illegal inside SIMT.
                value = (
                    self._dyn_dim_expr(param_name, i) if isinstance(dim, DynDim) else ast.Constant(value=dim)
                )
                assignment = ast.Assign(
                    targets=[tgt], value=value, lineno=node.lineno, col_offset=node.col_offset
                )
                stmts.append(self._visit_simple_assign(assignment))
            elif isinstance(dim, DynDim):
                # Dynamic dim: emit M = pl.tensor.dim(<param>, i) so the
                # DynVar name never appears as a runtime expression.
                assignment = ast.Assign(
                    targets=[tgt],
                    value=self._dyn_dim_expr(param_name, i),
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                )
                stmts.append(self._visit_simple_assign(assignment))
            else:
                # Static dim: inline constant, suppress assignment
                self._record_first_assign(tgt.id)
                if self._is_folded_graph_capture(tgt.id):
                    self._var_renames.pop(tgt.id)
                self._shape_inlined[tgt.id] = dim
        return stmts

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
            meta = self._meta[param_name].meta
            target_names = node.targets[0].elts
            if len(target_names) == len(meta.shape):
                return self._expand_shape_unpacking(node, param_name, target_names) or None

        # Case 3: K = a.shape[1]  →  inline static dim, suppress assignment
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            # An active alias normally needs a scalar write. A capture
            # backed by a folded shape has no scalar Var to update.
            and (
                node.targets[0].id not in self._var_renames
                or self._is_folded_graph_capture(node.targets[0].id)
            )
            and not (self._simt_depth and node.targets[0].id in self._simt_runtime_names)
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Attribute)
            and node.value.value.attr == "shape"
            and isinstance(node.value.value.value, ast.Name)
            and (param_name := self._known_tensor_name(node.value.value.value.id)) is not None
            and isinstance(node.value.slice, ast.Constant)
            and isinstance(node.value.slice.value, int)
        ):
            dim_idx = node.value.slice.value
            dim = self._meta[param_name].meta.shape[dim_idx]
            if not isinstance(dim, DynDim):
                self._var_renames.pop(node.targets[0].id, None)
                self._shape_inlined[node.targets[0].id] = dim
                self._record_first_assign(node.targets[0].id)
                return None

        # Default: visit the RHS first (applies existing renames to operands),
        # then handle rebinding rename on the LHS.
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            return self._visit_simple_assign(node)

        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
            # Snapshot all operand aliases before changing any target: swaps
            # and repeated targets must keep Python's single-RHS semantics.
            node.value = self.visit(node.value)
            node.targets = [
                self._visit_unpack_target(
                    node.targets[0], yield_result=self._is_yield(node.value), value=node.value
                )
            ]
            return node

        return cast("ast.stmt", self.generic_visit(node))

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.stmt | None:
        """Handle annotated assignments (``x: T = value``) with alpha-renaming."""
        if node.value is None:
            return cast("ast.stmt", self.generic_visit(node))
        if not isinstance(node.target, ast.Name):
            return cast("ast.stmt", self.generic_visit(node))
        var_name = node.target.id
        value_meta = self._tensor_operand_meta(node.value)
        node.value = self.visit(node.value)
        # Inline shape constants in the local annotation too (e.g. a body-level
        # ``x: pl.Tile[[1, W_PAD], pl.FP32]`` where ``W_PAD`` is a module-level
        # int). Without this the un-inlined name leaks into the generated source
        # and the parser rejects it ("Unknown shape variable: W_PAD").
        node.annotation = self.visit(node.annotation)
        name = self._bind_assign_name(var_name, value_meta, yield_result=self._is_yield(node.value))
        self._shape_inlined.pop(var_name, None)
        node.target = ast.copy_location(ast.Name(id=name, ctx=ast.Store()), node.target)
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

    def _visit_comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> None:
        """Rewrite generators in evaluation order within their private scope."""
        bound = {
            name for generator in node.generators for name in self._names_by_ctx(generator.target, ast.Store)
        }
        saved = (
            {name: self._meta[name] for name in bound if name in self._meta},
            {name: self._shape_inlined[name] for name in bound if name in self._shape_inlined},
            {name: self._var_renames[name] for name in bound if name in self._var_renames},
        )
        try:
            for generator in node.generators:
                # Each iterable sees the preceding scope, before its own
                # target shadows aliases, constexpr values and tensor metadata.
                generator.iter = self.visit(generator.iter)
                for name in self._names_by_ctx(generator.target, ast.Store):
                    self._meta.pop(name, None)
                    self._shape_inlined.pop(name, None)
                    self._var_renames[name] = name
                generator.ifs = [self.visit(condition) for condition in generator.ifs]
            if isinstance(node, ast.DictComp):
                node.key = self.visit(node.key)
                node.value = self.visit(node.value)
            else:
                node.elt = self.visit(node.elt)
        finally:
            _restore_bindings(self._meta, saved[0], bound)
            _restore_bindings(self._shape_inlined, saved[1], bound)
            _restore_bindings(self._var_renames, saved[2], bound)

    def visit_ListComp(self, node: ast.ListComp) -> ast.ListComp:
        self._visit_comprehension(node)
        return node

    def visit_SetComp(self, node: ast.SetComp) -> ast.SetComp:
        self._visit_comprehension(node)
        return node

    def visit_DictComp(self, node: ast.DictComp) -> ast.DictComp:
        self._visit_comprehension(node)
        return node

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.GeneratorExp:
        self._visit_comprehension(node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        """Replace a.shape → tuple and a.dtype → pl.XXX."""
        if (
            isinstance(node.value, ast.Name)
            and (param_name := self._known_tensor_name(node.value.id)) is not None
        ):
            if node.attr == "shape":
                return self._shape_tuple_node(param_name)
            if node.attr == "dtype":
                dtype_s = values.dtype_str(self._meta[param_name].meta.dtype)
                # Build attribute access: pl.FP32 → Attribute(Name("pl"), "FP32")
                parts = dtype_s.split(".")
                result: ast.expr = ast.Name(id=parts[0], ctx=ast.Load())
                for part in parts[1:]:
                    result = ast.Attribute(value=result, attr=part, ctx=ast.Load())
                return result
        return cast("ast.expr", self.generic_visit(node))

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        """Replace a.shape[i] → integer constant."""
        if (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "shape"
            and isinstance(node.value.value, ast.Name)
        ):
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                original = node.value.value.id
                if (param_name := self._known_tensor_name(original)) is not None:
                    return self._shape_dim_node(param_name, node.slice.value)
                if original in self._var_renames:
                    # The parameter's original shape no longer describes this value.
                    return self._dyn_dim_expr(self._var_renames[original], node.slice.value)
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
            param_order = self._dep_param_names.get(dep_name)
            constexpr_params = self._dep_constexpr_params.get(dep_name, set())
            if param_order is not None and (node.keywords or constexpr_params):
                bound_args = [
                    (param_order[index] if index < len(param_order) else None, arg)
                    for index, arg in enumerate(node.args)
                ]
                kw_by_name = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
                # Append keyword-bound args in dep's parameter order, skipping
                # params already covered by positional args.
                for param in param_order[len(node.args) :]:
                    if param in kw_by_name:
                        bound_args.append((param, kw_by_name.pop(param)))
                # Any remaining keywords (e.g. **kwargs splats or unknown names)
                # are kept as-is so we don't silently drop them; the parser
                # will surface them as a clear error.
                remaining_kw = [kw for kw in node.keywords if kw.arg is None or kw.arg in kw_by_name]
                if constexpr_params:
                    # Drop the callee's compile-time arguments. It resolved them
                    # during specialization and its generated signature no longer
                    # declares them, so passing one would be an arity mismatch.
                    # Preserve formal identities before compacting omitted Out
                    # slots: a later keyword constexpr is not the missing output.
                    bound_args = [(param, arg) for param, arg in bound_args if param not in constexpr_params]
                    remaining_kw = [kw for kw in remaining_kw if kw.arg not in constexpr_params]
                if original is not normalized:
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
                new_node = ast.Call(func=new_func, args=node.args, keywords=node.keywords)
            if original is not normalized:
                # Keep dispatch metadata on Submit while specializing the callee's
                # constexpr/keyword arguments exactly like an ordinary JIT call.
                dispatch_keywords = [kw for kw in original.keywords if kw.arg in SUBMIT_DISPATCH_KEYWORDS]
                new_node = ast.Call(
                    func=original.func,
                    args=[new_func, *new_node.args],
                    keywords=[*new_node.keywords, *dispatch_keywords],
                )
            ast.copy_location(new_node, original)
            return cast("ast.expr", self.generic_visit(new_node))
        return cast("ast.expr", self.generic_visit(node))

    def _visit_scoped_body(
        self, statements: list[ast.stmt], *, yield_renames: dict[str, str] | None = None
    ) -> list[ast.stmt]:
        """Visit statements within a nested scope, flattening lists and dropping None."""
        previous = self._yield_renames
        if yield_renames is not None:
            self._yield_renames = yield_renames
        try:
            return self._visit_statements(statements)
        finally:
            self._yield_renames = previous

    def _exported_loop_names(self, statements: list[ast.stmt], *, initialized_only: bool = False) -> set[str]:
        """Find loop exports through branches that the parser statically selects."""
        names: set[str] = set()
        pending = list(statements)
        while pending:
            statement = pending.pop()
            if isinstance(statement, ast.If):
                if (selected := _literal_if_body(statement)) is not None:
                    pending.extend(selected)
            elif initialized_only and isinstance(statement, (ast.For, ast.While)):
                names |= self._initialized_loop_exports.get(id(statement), set())
            elif isinstance(statement, ast.While):
                names |= self._names_by_ctx(statement, ast.Store)
            elif isinstance(statement, ast.For):
                yielded = self._loop_yield_targets(statement.body)
                if isinstance(statement.target, ast.Tuple):
                    names |= yielded
                elif self._is_inactive_plain_loop(statement):
                    continue
                elif isinstance(statement.target, ast.Name) and not yielded:
                    names |= self._names_by_ctx(statement, ast.Store) - self._loop_scoped_names(statement)
        return {self._alias_to_original.get(name, name) for name in names}

    @staticmethod
    def _literal_range_has_iterations(node: ast.For) -> bool | None:
        """Prove whether a literal range has trips without evaluating arbitrary calls."""
        call = node.iter
        if not isinstance(call, ast.Call) or not 1 <= len(call.args) <= 3:
            return None
        if isinstance(call.func, ast.Name):
            supported = call.func.id == "range"
        else:
            supported = (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "pl"
                and call.func.attr in {"range", "unroll", "parallel"}
            )
        if not supported:
            return None
        bounds: list[int] = []
        for argument in call.args:
            sign = 1
            if isinstance(argument, ast.UnaryOp) and isinstance(argument.op, (ast.UAdd, ast.USub)):
                sign = -1 if isinstance(argument.op, ast.USub) else 1
                argument = argument.operand
            if not isinstance(argument, ast.Constant) or type(argument.value) is not int:
                return None
            bounds.append(sign * argument.value)
        start, stop = (0, bounds[0]) if len(bounds) == 1 else (bounds[0], bounds[1])
        step = bounds[2] if len(bounds) == 3 else 1
        if step == 0:
            return None
        return (step > 0 and start < stop) or (step < 0 and start > stop)

    def _is_inactive_plain_loop(self, node: ast.For) -> bool:
        """Recognize empty loops that do not publish carried initial values."""
        return (
            isinstance(node.target, ast.Name)
            and self._literal_range_has_iterations(node) is False
            and not self._loop_yield_targets(node.body)
        )

    def _definite_loop_writes(self, statements: list[ast.stmt], after: set[str]) -> set[str]:
        """Find stores present on every iteration exit, including early transfers.

        Nested loops use their already computed summaries, so an enclosing
        loop does not rescan their bodies. Return/raise paths are treated
        conservatively; they cannot establish a post-loop binding.
        """
        written = after.copy()
        for statement in reversed(statements):
            if isinstance(statement, (ast.Break, ast.Continue, ast.Return, ast.Raise)):
                written.clear()
            elif isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                for target in targets:
                    written |= self._names_by_ctx(target, ast.Store)
            elif isinstance(statement, ast.If):
                selected = _literal_if_body(statement)
                written = (
                    self._definite_loop_writes(selected, written)
                    if selected is not None
                    else self._definite_loop_writes(statement.body, written)
                    & self._definite_loop_writes(statement.orelse, written)
                )
            elif isinstance(statement, (ast.For, ast.While)):
                written |= self._initialized_loop_exports.get(id(statement), set())
            elif isinstance(statement, ast.With) and not any(
                _scope_call_name(item.context_expr) in {"graph", "simt_scope"} for item in statement.items
            ):
                written = self._definite_loop_writes(statement.body, written)
                for item in statement.items:
                    if item.optional_vars is not None:
                        written |= self._names_by_ctx(item.optional_vars, ast.Store)
        return written

    def _remember_initialized_loop_exports(self, statement: ast.stmt) -> None:
        names: set[str]
        if isinstance(statement, ast.For):
            if isinstance(statement.target, ast.Tuple):
                # Explicit carry results are initialized even on zero trips.
                names = self._loop_yield_targets(statement.body)
            elif self._literal_range_has_iterations(statement) is True:
                names = self._definite_loop_writes(statement.body, set())
                names -= self._loop_scoped_names(statement)
            else:
                names = set()
        elif isinstance(statement, ast.While) and values.literal_condition(statement.test) is True:
            names = self._definite_loop_writes(statement.body, set())
        else:
            return
        self._initialized_loop_exports[id(statement)] = names

    def _visit_statements(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        loop_exports: set[str] = set()
        initialized_exports: set[str] = set()
        for stmt in statements:
            if isinstance(stmt, ast.If):
                # Simple loops export body locals, and explicit carry loops
                # export owned yield results, including zero-trip initial values.
                # A following conditional updates those incoming bindings.
                # Keep the original loop depth until this boundary: sibling
                # loops and Graph kernels still
                # need it to distinguish fresh locals from genuine carries.
                self._promote_scope_bindings(loop_exports)
            elif isinstance(stmt, (ast.For, ast.While)):
                # An active version from an earlier loop is an incoming value
                # even when this loop writes before reading or takes zero trips.
                # A nonempty predecessor loop can initialize an unversioned
                # value too. Its first alias may be created only in this loop.
                # Unknown-trip sibling locals keep their fresh-local handling.
                self._promote_scope_bindings(
                    {name for name in loop_exports if self._var_renames.get(name, name) != name}
                    | initialized_exports
                )
            exported = self._exported_loop_names([stmt])
            if isinstance(stmt, (ast.Break, ast.Continue)):
                self._loop_exit_bindings[id(stmt)] = {
                    name: (self._var_renames.get(name, name), self._active_shape_fold(name))
                    for name in self._assign_count
                }
                self._loop_exit_tensor_meta[id(stmt)] = {
                    name: (
                        self._meta.get(known)
                        if (known := self._known_tensor_name(name)) is not None
                        else None
                    )
                    for name in self._assign_count
                }
            visited = self.visit(stmt)
            loop_exports |= exported
            if visited is None:
                pass
            elif isinstance(visited, list):
                emitted = cast("list[ast.stmt]", visited)
                result.extend(emitted)
                for statement in emitted:
                    self._remember_initialized_loop_exports(statement)
                # A constexpr arm is flattened into this statement list. Carry
                # its loop exports along with the emitted statements.
                loop_exports |= self._exported_loop_names(emitted)
                initialized_exports |= self._exported_loop_names(emitted, initialized_only=True)
            else:
                result.append(visited)
                self._remember_initialized_loop_exports(visited)
                loop_exports |= self._exported_loop_names([visited])
                initialized_exports |= self._exported_loop_names([visited], initialized_only=True)
        return result or [ast.Pass()]

    def transform_body(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        """Rewrite a complete function body, merging only live branch outputs."""
        self._preview_liveness_conditions(statements, self._used_names - self._constexpr.keys())
        self._collect_branch_liveness(statements, set())
        return self._visit_scoped_body(statements)

    def _preview_liveness_conditions(self, statements: list[ast.stmt], runtime_names: set[str]) -> None:
        """Index the static arms using the same binding rules as the actual rewrite.

        A descriptor may survive copies, joins and invariant loop backedges.
        Reusing the real visitor avoids a second, weaker metadata analysis.
        The copied visitor uses only literal/external conditions for its own
        liveness, so previewing does not recursively start another preview.
        """
        owner = self
        source = ast.Module(body=statements, type_ignores=[])
        copied = copy.deepcopy(source)
        originals: dict[int, ast.If] = {}
        call_targets: dict[int, str] = {}
        for original, clone in zip(ast.walk(source), ast.walk(copied), strict=True):
            if isinstance(original, ast.If):
                originals[id(clone)] = original
            if (target := self._dep_call_targets.get(id(original))) is not None:
                # Variant dispatch is keyed by AST identity. Preserve each
                # call's selected constexpr specialization on the copied AST.
                call_targets[id(clone)] = target

        class Preview(BodyTransformer):
            def _preview_liveness_conditions(
                self, statements: list[ast.stmt], runtime_names: set[str]
            ) -> None:
                for statement in statements:
                    for node in ast.walk(statement):
                        if isinstance(node, ast.If):
                            selected = self._static_if_body(node, runtime_names)
                            if selected is not None:
                                self._liveness_static_arms[id(node)] = selected

            def visit_If(self, node: ast.If) -> ast.stmt | list[ast.stmt]:
                transformed = super().visit_If(node)
                selected = values.literal_condition(node.test)
                if selected is None:
                    selected = _constant_condition(node.test)
                if selected is not None:
                    original = originals[id(node)]
                    owner._liveness_static_arms[id(original)] = original.body if selected else original.orelse
                return transformed

        preview = Preview(
            tensor_meta={name: binding.meta for name, binding in self._meta.items()},
            dep_names=self._dep_names,
            param_names=list(self._assign_count),
            initial_used_names=self._used_names.copy(),
            py_globals=self._py_globals,
            dep_param_names=self._dep_param_names,
            dep_func_names=self._dep_func_names,
            constexpr_values=self._constexpr,
            dep_constexpr_params=self._dep_constexpr_params,
            dep_call_targets=call_targets,
        )
        preview.initialize_tensor_bindings(self.known_distributed_tensors, self._dep_parameter_returns)
        preview.transform_body(copied.body)

    def _has_owned_loop_break(self, statements: list[ast.stmt]) -> bool:
        """Find a reachable break without crossing a nested loop's ownership."""
        pending = list(statements)
        while pending:
            statement = pending.pop()
            if isinstance(statement, ast.Break):
                return True
            if isinstance(statement, ast.If):
                selected = self._liveness_static_arms.get(id(statement), _literal_if_body(statement))
                pending.extend(selected if selected is not None else [*statement.body, *statement.orelse])
            elif isinstance(statement, ast.With):
                pending.extend(statement.body)
        return False

    def _collect_loop_liveness(
        self,
        statement: ast.For | ast.While,
        live_after: set[str],
        outer_exits: tuple[set[str], set[str]] | None,
    ) -> set[str]:
        """Analyze one loop with distinct break and next-iteration successors."""
        tail = self._collect_branch_liveness(statement.orelse, live_after, outer_exits)
        carried: set[str] = set()
        self._collect_upward_exposed(statement.body, set(), carried, may_skip_loops=True)
        control = statement.iter if isinstance(statement, ast.For) else statement.test
        reads = self._names_by_ctx(control, ast.Load)
        continue_live = tail | carried
        if isinstance(statement, ast.While):
            continue_live |= reads
        if self._has_owned_loop_break(statement.body):
            # A later iteration can break before a loop-else overwrite. Its
            # exit values must therefore survive a preceding continue too.
            continue_live |= live_after
        body = self._collect_branch_liveness(statement.body, continue_live, (live_after, continue_live))
        if isinstance(statement, ast.For):
            body -= self._names_by_ctx(statement.target, ast.Store)
        return tail | body | reads

    def _collect_branch_liveness(
        self,
        statements: list[ast.stmt],
        live_after: set[str],
        loop_exits: tuple[set[str], set[str]] | None = None,
    ) -> set[str]:
        """Summarize blocks backwards before aliases mutate the source AST.

        Branch-local versions need a common export only when their value can
        be read later. Exporting dead temporaries would unnecessarily merge
        independent types from sibling branches. Loop live-outs also include
        reads before writes, so exports survive the next iteration's backedge.
        """
        live = live_after.copy()
        for stmt in reversed(statements):
            if isinstance(stmt, ast.If):
                self._branch_live_outs[id(stmt)] = live.copy()
                # Use the source-ordered descriptor preview: a read in a
                # statically dead arm cannot require a cross-type export.
                selected = self._liveness_static_arms.get(id(stmt))
                if selected is not None:
                    live = self._collect_branch_liveness(selected, live, loop_exits)
                else:
                    live = (
                        self._collect_branch_liveness(stmt.body, live, loop_exits)
                        | self._collect_branch_liveness(stmt.orelse, live, loop_exits)
                        | self._names_by_ctx(stmt.test, ast.Load)
                    )
            elif isinstance(stmt, (ast.For, ast.While)):
                self._branch_live_outs[id(stmt)] = live.copy()
                live = self._collect_loop_liveness(stmt, live, loop_exits)
            elif isinstance(stmt, ast.With):
                private = any(
                    _scope_call_name(item.context_expr) in {"graph", "simt_scope"} for item in stmt.items
                )
                body = self._collect_branch_liveness(
                    stmt.body, set() if private else live, None if private else loop_exits
                )
                for item in reversed(stmt.items):
                    if item.optional_vars is not None:
                        body -= self._names_by_ctx(item.optional_vars, ast.Store)
                    body |= self._names_by_ctx(item.context_expr, ast.Load)
                live = body | live if private else body
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                for target in targets:
                    live -= self._names_by_ctx(target, ast.Store)
                    live |= self._names_by_ctx(target, ast.Load)
                if stmt.value is not None:
                    live |= self._names_by_ctx(stmt.value, ast.Load)
            elif isinstance(stmt, ast.Return):
                live = self._names_by_ctx(stmt, ast.Load)
            elif isinstance(stmt, (ast.Break, ast.Continue)) and loop_exits is not None:
                # Transfers bypass the remaining sibling statements. Their
                # destination, not a later store on the fallthrough path,
                # determines the values that need an export before the exit.
                live = loop_exits[isinstance(stmt, ast.Continue)].copy()
            elif isinstance(stmt, ast.AugAssign):
                live |= self._names_by_ctx(stmt.target, ast.Store) | self._names_by_ctx(stmt, ast.Load)
            else:
                live |= self._names_by_ctx(stmt, ast.Load)
        return live

    @staticmethod
    def _names_in_scope(node: ast.AST, ctx_type: type) -> list[ast.Name]:
        """Collect names visible to the caller, respecting comprehension scopes."""
        names: list[ast.Name] = []

        class ScopeNames(ast.NodeVisitor):
            bound: set[str] = set()

            def visit_Name(self, node: ast.Name) -> None:
                if isinstance(node.ctx, ctx_type) and node.id not in self.bound:
                    names.append(node)

            def visit_comprehension_scope(
                self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
            ) -> None:
                before = self.bound
                self.bound = before.copy()
                try:
                    for generator in node.generators:
                        self.visit(generator.iter)
                        self.bound |= {
                            item.id for item in ast.walk(generator.target) if isinstance(item, ast.Name)
                        }
                        for condition in generator.ifs:
                            self.visit(condition)
                    if isinstance(node, ast.DictComp):
                        self.visit(node.key)
                        self.visit(node.value)
                    else:
                        self.visit(node.elt)
                finally:
                    self.bound = before

            visit_ListComp = visit_comprehension_scope
            visit_SetComp = visit_comprehension_scope
            visit_DictComp = visit_comprehension_scope
            visit_GeneratorExp = visit_comprehension_scope

        ScopeNames().visit(node)
        return names

    @classmethod
    def _names_by_ctx(cls, node: ast.AST, ctx_type: type) -> set[str]:
        """Collect visible ``Name`` ids under the given context (Load/Store)."""
        return {item.id for item in cls._names_in_scope(node, ctx_type)}

    def _collect_upward_exposed(
        self,
        statements: list[ast.stmt],
        written: set[str],
        exposed: set[str],
        *,
        may_skip_loops: bool = False,
    ) -> None:
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

        SIMT callers set ``may_skip_loops`` because an assignment in a loop
        body does not initialize a private scalar when the loop runs zero times.

        Names bound only inside a comprehension or lambda are not
        statement-level assignments, so they never enter ``_assign_count`` and
        are never ``_classify_loop_locals`` candidates.  ``_names_by_ctx`` does
        walk into those nested scopes, but only statement *targets* feed
        ``written`` here — a comprehension/lambda local can at most leak a
        harmless extra entry into ``exposed``, never affecting classification.
        """
        for stmt in statements:
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
            elif isinstance(stmt, ast.For):
                exposed.update(self._names_by_ctx(stmt.iter, ast.Load) - written)
                body_written = set(written) if may_skip_loops else written
                body_written.update(self._names_by_ctx(stmt.target, ast.Store))
                self._collect_upward_exposed(stmt.body, body_written, exposed, may_skip_loops=may_skip_loops)
                self._collect_upward_exposed(stmt.orelse, written, exposed, may_skip_loops=may_skip_loops)
                if isinstance(stmt.target, ast.Tuple):
                    # Explicit carries publish their initialized result even
                    # if the body is skipped. This is a definite branch write.
                    written.update(self._loop_yield_targets(stmt.body))
            elif isinstance(stmt, ast.While) and may_skip_loops:
                exposed.update(self._names_by_ctx(stmt.test, ast.Load) - written)
                self._collect_upward_exposed(stmt.body, set(written), exposed, may_skip_loops=may_skip_loops)
                self._collect_upward_exposed(stmt.orelse, written, exposed, may_skip_loops=may_skip_loops)
            elif isinstance(stmt, ast.If):
                if (selected := self._liveness_static_arms.get(id(stmt), _literal_if_body(stmt))) is not None:
                    self._collect_upward_exposed(selected, written, exposed, may_skip_loops=may_skip_loops)
                    continue
                exposed.update(self._names_by_ctx(stmt.test, ast.Load) - written)
                # then/else are alternative paths: analyze each from the same
                # incoming ``written`` so a read in one branch is not masked by
                # a write in the other.  A name is exposed if read-before-write
                # on either path; it is definitely written after the ``if``
                # only if written on both (no ``else`` → no definite write).
                then_written = set(written)
                self._collect_upward_exposed(stmt.body, then_written, exposed, may_skip_loops=may_skip_loops)
                if stmt.orelse:
                    else_written = set(written)
                    self._collect_upward_exposed(
                        stmt.orelse, else_written, exposed, may_skip_loops=may_skip_loops
                    )
                    written.update(then_written & else_written)
            elif isinstance(stmt, ast.With):
                for item in stmt.items:
                    exposed.update(self._names_by_ctx(item.context_expr, ast.Load) - written)
                    if item.optional_vars is not None:
                        written.update(self._names_by_ctx(item.optional_vars, ast.Store))
                self._collect_upward_exposed(stmt.body, written, exposed, may_skip_loops=may_skip_loops)
            else:
                # Expr / Return / Assert / ... — pure reads.
                exposed.update(self._names_by_ctx(stmt, ast.Load) - written)

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
        depth.  ``visit_For`` drops *fresh* names from the rename bookkeeping so
        the body re-declares them cleanly, exactly how the ``@pl.program``
        parser scopes sibling loop bodies.
        """
        carried: list[str] = []
        fresh: list[str] = []
        seen: set[str] = set()
        # Names read before written in the body — the only valid bridge sources.
        exposed: set[str] = set()
        self._collect_upward_exposed(statements, set(), exposed)
        for stmt in statements:
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

    def _promote_scope_bindings(self, names: set[str]) -> None:
        """Match parser scopes that export body bindings to their caller."""
        for name in names & self._assign_depth.keys():
            self._assign_depth[name] = min(self._assign_depth[name], self._scope_depth)

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
            current_name = self._var_renames.get(var_name, var_name)
            source = self._known_tensor_name(var_name)
            value_meta = self._meta.get(source) if source is not None else None
            new_name = self._rebind(var_name)
            self._record_tensor_meta(new_name, value_meta)
            # Update assign_depth to the outer scope so the loop body sees a
            # depth mismatch and does NOT trigger another rebind.
            self._assign_depth[var_name] = self._scope_depth
            bridge = ast.Assign(
                targets=[ast.Name(id=new_name, ctx=ast.Store())],
                value=ast.Name(id=current_name, ctx=ast.Load()),
                lineno=0,
                col_offset=0,
            )
            ast.fix_missing_locations(bridge)
            bridges.append(bridge)
        return bridges

    @classmethod
    def _loop_scoped_names(cls, node: ast.For) -> set[str]:
        """Names shadowed by the parser's induction/carry binding contract."""
        return cls._names_by_ctx(node.target, ast.Store)

    @classmethod
    def _loop_yield_targets(cls, statements: list[ast.stmt]) -> set[str]:
        """Collect outputs using the parser's yield ownership boundaries."""
        names: set[str] = set()
        for stmt in ControlFlowParser.iter_yield_assignments(statements):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for target in targets:
                names.update(cls._names_by_ctx(target, ast.Store))
        return names

    def _preserved_shape_targets(self, node: ast.Assign, modified_sources: set[str]) -> set[int]:
        """Recognize shape writes that the existing assignment folder suppresses."""
        if len(node.targets) != 1:
            return set()
        target = node.targets[0]
        value = node.value
        dimensions: list[tuple[ast.expr, ast.expr]] = []
        if isinstance(target, ast.Tuple) and isinstance(value, (ast.Tuple, ast.List)):
            pending = [(target, value)]
            while pending:
                unpack, operands = pending.pop()
                if len(unpack.elts) != len(operands.elts):
                    continue
                for item, operand in zip(unpack.elts, operands.elts, strict=True):
                    if isinstance(item, ast.Tuple) and isinstance(operand, (ast.Tuple, ast.List)):
                        pending.append((item, operand))
                    elif isinstance(operand, ast.Name) and operand.id not in modified_sources:
                        folded = self._active_shape_fold(operand.id)
                        if folded is not None:
                            dimensions.append((item, ast.Constant(value=folded)))
                    elif isinstance(operand, ast.Constant) and type(operand.value) is int:
                        dimensions.append((item, operand))
        if isinstance(value, ast.Attribute) and value.attr == "shape" and isinstance(value.value, ast.Name):
            if value.value.id in modified_sources:
                return set()
            source = self._known_tensor_name(value.value.id)
            if (
                source is not None
                and isinstance(target, ast.Tuple)
                and len(target.elts) == len(self._meta[source].meta.shape)
            ):
                dimensions = [(item, self._shape_dim_node(source, i)) for i, item in enumerate(target.elts)]
        elif (
            isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Attribute)
            and value.value.attr == "shape"
            and isinstance(value.value.value, ast.Name)
            and isinstance(value.slice, ast.Constant)
            and isinstance(value.slice.value, int)
        ):
            if value.value.value.id in modified_sources:
                return set()
            source = self._known_tensor_name(value.value.value.id)
            if source is not None and -len(self._meta[source].meta.shape) <= value.slice.value < len(
                self._meta[source].meta.shape
            ):
                dimensions = [(target, self._shape_dim_node(source, value.slice.value))]
        return {
            id(target)
            for target, dimension in dimensions
            if isinstance(target, ast.Name)
            and isinstance(dimension, ast.Constant)
            and dimension.value == self._active_shape_fold(target.id)
        }

    def _static_if_body(
        self, node: ast.If, shadowed: set[str], integer_bounds: Mapping[str, tuple[int, int]] | None = None
    ) -> list[ast.stmt] | None:
        """Preview external constants without mutating the body or folding runtime locals."""
        owner = self

        class ExternalNames(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.expr:
                if node.id not in shadowed and (
                    node.id in owner._constexpr
                    or node.id not in owner._used_names
                    or owner._active_shape_fold(node.id) is not None
                ):
                    return owner.visit_Name(node)
                return node

            def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
                if (
                    isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id not in shadowed
                    and owner._known_tensor_name(node.value.value.id) is not None
                ):
                    # Descriptor conditions are safe until a preceding write
                    # changes their tensor source on any incoming path.
                    return owner.visit_Subscript(node)
                return cast(ast.expr, self.generic_visit(node))

            def visit_Compare(self, node: ast.Compare) -> ast.expr:
                node = cast(ast.Compare, self.generic_visit(node))
                # A fold can stay invariant even when a loop writes it: every
                # possible literal assignment may remain on one side of a
                # threshold. Only monotone comparisons admit this endpoint proof.
                if (
                    integer_bounds is None
                    or len(node.ops) != 1
                    or not isinstance(node.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE))
                ):
                    return node
                left, right = node.left, node.comparators[0]
                varying = left if isinstance(left, ast.Name) else right
                fixed = right if varying is left else left
                if (
                    not isinstance(varying, ast.Name)
                    or varying.id not in integer_bounds
                    or not isinstance(fixed, ast.Constant)
                    or type(fixed.value) is not int
                ):
                    return node
                outcomes: list[bool | None] = []
                for bound in integer_bounds[varying.id]:
                    literal = ast.Constant(value=bound)
                    comparison = ast.Compare(
                        left=literal if varying is left else fixed,
                        ops=node.ops,
                        comparators=[fixed if varying is left else literal],
                    )
                    outcomes.append(_constant_condition(comparison))
                if outcomes[0] is not None and outcomes[0] == outcomes[1]:
                    return ast.copy_location(ast.Constant(value=outcomes[0]), node)
                return node

        preview = copy.copy(node)
        preview.test = ExternalNames().visit(copy.deepcopy(node.test))
        return _literal_if_body(preview)

    def _literal_shape_bounds(
        self, assignments: list[ast.Assign | ast.AnnAssign], targets: list[ast.Name]
    ) -> dict[str, tuple[int, int]]:
        """Bound folded integers only when every region write is an integer literal.

        Entry values and all literal writes form an inductive range without a
        fixed-point scan. An unknown write or a scoped induction/carry target
        discards the proof, rather than inheriting the enclosing fold.
        """
        literals: dict[int, int] = {}
        for assignment in assignments:
            # Annotated writes can cast their literal to a different scalar
            # dtype. Keep them unknown until those cast semantics are modeled.
            if isinstance(assignment, ast.AnnAssign):
                continue
            pending = [(target, assignment.value) for target in assignment.targets]
            while pending:
                target, value = pending.pop()
                if (
                    isinstance(target, ast.Name)
                    and isinstance(value, ast.Constant)
                    and type(value.value) is int
                ):
                    literals[id(target)] = value.value
                elif (
                    isinstance(target, ast.Tuple)
                    and isinstance(value, (ast.Tuple, ast.List))
                    and len(target.elts) == len(value.elts)
                ):
                    pending.extend(zip(target.elts, value.elts, strict=True))
        bounds = {
            name: (value, value)
            for name in self._shape_inlined
            if (value := self._active_shape_fold(name)) is not None
        }
        for target in targets:
            if target.id not in bounds:
                continue
            if id(target) not in literals:
                bounds.pop(target.id)
            else:
                lower, upper = bounds[target.id]
                value = literals[id(target)]
                bounds[target.id] = min(lower, value), max(upper, value)
        return bounds

    def _preserved_tensor_targets(
        self, node: ast.Assign | ast.AnnAssign, modified_sources: set[str] | None = None
    ) -> set[int]:
        """Recognize identities or descriptor-equal copies from unchanged sources."""
        if node.value is None:
            return set()
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        pending = [(target, node.value) for target in targets]
        preserved: set[int] = set()
        while pending:
            target, value = pending.pop()
            if (
                isinstance(target, ast.Name)
                and isinstance(value, ast.Name)
                and (source := self._tensor_operand_meta(value)) is not None
                and (
                    target.id == value.id
                    if modified_sources is None
                    else value.id not in modified_sources and source == self._tensor_operand_meta(target)
                )
            ):
                preserved.add(id(target))
            elif (
                isinstance(target, ast.Tuple)
                and isinstance(value, (ast.Tuple, ast.List))
                and len(target.elts) == len(value.elts)
            ):
                pending.extend(zip(target.elts, value.elts, strict=True))
        return preserved

    def _conditional_shape_assignments(
        self, node: ast.If | ast.For | ast.While, shadowed: set[str]
    ) -> tuple[list[ast.Assign | ast.AnnAssign], list[ast.Name], set[int]]:
        """Collect reachable writes while excluding targets restored by local scopes."""
        assignments: list[ast.Assign | ast.AnnAssign] = []
        targets: list[ast.Name] = []
        pending: list[ast.AST] = [*node.body, *node.orelse]
        scoped_targets: set[int] = set()
        while pending:
            current = pending.pop()
            if isinstance(current, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                # Their binders do not write any enclosing shape binding.
                continue
            if (
                isinstance(current, ast.If)
                and (selected := self._static_if_body(current, shadowed)) is not None
            ):
                pending.extend(selected)
                continue
            if isinstance(current, ast.With) and any(
                _scope_call_name(item.context_expr) in {"graph", "simt_scope"} for item in current.items
            ):
                # These regions restore the caller's bindings themselves.
                continue
            if isinstance(current, ast.For):
                if self._is_inactive_plain_loop(current):
                    pending.extend(current.orelse)
                    continue
                restored = self._loop_scoped_names(current) - self._loop_yield_targets(current.body)
                scoped_targets.update(
                    id(item)
                    for item in ast.walk(current)
                    if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store) and item.id in restored
                )
            if isinstance(current, (ast.Assign, ast.AnnAssign)):
                assignments.append(current)
            if isinstance(current, ast.Name) and isinstance(current.ctx, ast.Store):
                targets.append(current)
            pending.extend(ast.iter_child_nodes(current))
        return assignments, targets, scoped_targets

    def _materialize_conditional_shapes(self, node: ast.If | ast.For | ast.While) -> list[ast.stmt]:
        """Give conditionally overwritten folds a runtime predecessor on every path."""
        if self._simt_depth or not self._shape_inlined:
            return []
        # Only stable external names may fold during this preview. A local
        # write or loop header can shadow a free name before its nested test;
        # ordinary body visitation will resolve those bindings in order.
        shadowed = self._names_by_ctx(node, ast.Store)
        assignments, targets, scoped_targets = self._conditional_shape_assignments(node, shadowed)
        entry_preserved: set[int] = set()
        for assignment in assignments:
            if isinstance(assignment, ast.Assign):
                entry_preserved |= self._preserved_shape_targets(assignment, set())
            entry_preserved |= self._preserved_tensor_targets(assignment)
            entry_preserved |= self._preserved_tensor_targets(assignment, shadowed)
        # Loop operands may already have changed on an earlier iteration.
        changed_targets = {
            id(target): target.id for target in targets if id(target) not in entry_preserved | scoped_targets
        }
        loop_writes = set(changed_targets.values())
        integer_bounds = self._literal_shape_bounds(assignments, targets)

        escaped_writes: set[str] = set()

        def scan(
            statements: list[ast.stmt],
            modified: set[str],
            allowed: set[str] | None,
            incoming_written: set[str] | None = None,
        ) -> set[str]:
            written: set[str] = set() if incoming_written is None else incoming_written.copy()
            for stmt in statements:
                if isinstance(stmt, ast.For) and self._is_inactive_plain_loop(stmt):
                    written = scan(stmt.orelse, modified, allowed, written)
                    continue
                if isinstance(stmt, (ast.Break, ast.Continue, ast.Return, ast.Raise)):
                    # A later restore cannot kill writes on a path that has
                    # already left this region. Keep these exits conservative.
                    escaped_writes.update(written)
                    break
                if isinstance(stmt, ast.With):
                    if any(
                        _scope_call_name(item.context_expr) in {"graph", "simt_scope"} for item in stmt.items
                    ):
                        continue
                    written = scan(stmt.body, modified, allowed, written)
                elif (
                    isinstance(stmt, ast.If)
                    and (selected := self._static_if_body(stmt, modified, integer_bounds)) is not None
                ):
                    # The proof must also select the emitted arm. Otherwise a
                    # retained runtime branch could still bind a folded name
                    # whose predecessor this analysis deliberately omitted.
                    stmt.test = ast.copy_location(ast.Constant(value=selected is stmt.body), stmt.test)
                    written = scan(selected, modified, allowed, written)
                elif isinstance(stmt, (ast.If, ast.For, ast.While)):
                    child_live = self._branch_live_outs.get(id(stmt))
                    child_allowed = (
                        allowed
                        if child_live is None
                        else child_live
                        if allowed is None
                        else allowed & child_live
                    )
                    left, right = modified.copy(), modified.copy()
                    if isinstance(stmt, (ast.For, ast.While)):
                        left |= {
                            changed_targets[id(item)]
                            for item in ast.walk(stmt)
                            if id(item) in changed_targets
                        }
                    restored: set[str] = set()
                    if isinstance(stmt, ast.For):
                        scoped = self._loop_scoped_names(stmt)
                        restored = scoped - self._loop_yield_targets(stmt.body)
                        left |= scoped
                    # Induction/carry bindings shadow caller folds inside the
                    # loop, then restore them unless an owned yield publishes.
                    written = (scan(stmt.body, left, child_allowed, written) - restored) | scan(
                        stmt.orelse, right, child_allowed, written
                    )
                    modified.clear()
                    modified.update((left - restored) | right)
                else:
                    stores = {id(item): item.id for item in self._names_in_scope(stmt, ast.Store)}
                    preserved: set[int] = (
                        self._preserved_shape_targets(stmt, modified)
                        if isinstance(stmt, ast.Assign)
                        else set()
                    )
                    changed = {name for target, name in stores.items() if target not in preserved}
                    # A definite write of the entry value kills an earlier
                    # overwrite on this path, including direct sibling writes.
                    restored_values = {name for target, name in stores.items() if target in preserved}
                    written -= restored_values
                    modified -= restored_values
                    written |= changed if allowed is None else changed & allowed
                    identities: set[int] = set()
                    same_meta: set[int] = set()
                    if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                        identities = self._preserved_tensor_targets(stmt)
                        same_meta = self._preserved_tensor_targets(stmt, modified)
                    modified -= {name for target, name in stores.items() if target in same_meta}
                    modified |= {
                        name
                        for target, name in stores.items()
                        if target not in preserved | identities | same_meta
                    }
            return written

        incoming: set[str] = loop_writes if isinstance(node, (ast.For, ast.While)) else set()
        written = scan(node.body, incoming.copy(), None) | scan(node.orelse, incoming.copy(), None)
        written |= escaped_writes
        if (live := self._branch_live_outs.get(id(node))) is not None:
            relevant = live.copy()
            if isinstance(node, (ast.For, ast.While)):
                # A scalar can also be live on the next loop iteration. TaskId
                # shadows used only in this body retain Graph restoration and
                # must not overwrite a newly materialized integer Var.
                self._collect_upward_exposed(node.body, set(), relevant, may_skip_loops=True)
                if isinstance(node, ast.While):
                    relevant |= self._names_by_ctx(node.test, ast.Load)
            written &= relevant
        prefix: list[ast.stmt] = []
        for name in sorted(written & self._shape_inlined.keys()):
            folded = self._active_shape_fold(name)
            if folded is None:
                continue
            # A fresh alias also prevents a later shape assignment in the
            # branch from being folded unconditionally into the caller.
            alias = self._rebind(name)
            self._assign_depth[name] = self._scope_depth
            self._shape_inlined.pop(name)
            assignment = ast.Assign(
                targets=[ast.Name(id=alias, ctx=ast.Store())], value=ast.Constant(value=folded)
            )
            prefix.append(ast.fix_missing_locations(ast.copy_location(assignment, node)))
        return prefix

    def _prepare_loop_bindings(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        """Bridge loop carries while discarding fresh sibling-scope locals."""
        self._scope_depth += 1
        try:
            carried, fresh = self._classify_loop_locals(statements)
        finally:
            self._scope_depth -= 1
        bridges = self._make_bridge_assignments(carried)
        for name in fresh:
            self._forget_rename_bookkeeping(name)
        return bridges

    def _explicit_carry_meta(self, node: ast.For) -> list[tuple[str, _TensorBinding | None]]:
        """Pair flat carry binders with the already visited initial operands."""
        if not isinstance(node.target, ast.Tuple) or not isinstance(node.iter, ast.Call):
            return []
        initial = next((kw.value for kw in node.iter.keywords if kw.arg == "init_values"), None)
        if not isinstance(initial, (ast.Tuple, ast.List)):
            return []
        carries = (
            node.target
            if _scope_call_name(node.iter) == "while_"
            else node.target.elts[1]
            if len(node.target.elts) == 2
            else None
        )
        if not isinstance(carries, ast.Tuple) or len(carries.elts) != len(initial.elts):
            return []
        owned = list(ControlFlowParser.iter_yield_assignments(node.body))
        yield_stores: set[int] = set()
        for assignment in owned:
            target = assignment.targets[0] if isinstance(assignment, ast.Assign) else assignment.target
            yield_stores.update(id(item) for item in self._names_in_scope(target, ast.Store))
        written = {
            item.id
            for item in self._names_in_scope(ast.Module(body=node.body, type_ignores=[]), ast.Store)
            if id(item) not in yield_stores
        }
        bound_carries = self._names_by_ctx(carries, ast.Store)
        body_stores = self._names_by_ctx(ast.Module(body=node.body, type_ignores=[]), ast.Store)
        result: list[tuple[str, _TensorBinding | None]] = []
        for index, (target, value) in enumerate(zip(carries.elts, initial.elts, strict=True)):
            if not isinstance(target, ast.Name):
                continue
            initial_meta = self._tensor_operand_meta(value)
            stable = target.id not in written and all(
                isinstance(assignment.value, ast.Call)
                and index < len(assignment.value.args)
                and isinstance(operand := assignment.value.args[index], ast.Name)
                and (
                    operand.id == target.id
                    or (
                        operand.id not in body_stores | bound_carries
                        and self._tensor_operand_meta(operand) == initial_meta
                    )
                )
                for assignment in owned
            )
            # A changing backedge cannot inherit the initial descriptor for
            # later iterations. Stable external tensors with equal descriptors
            # preserve the invariant just as an identity carry does.
            result.append((target.id, initial_meta if stable else None))
        return result

    def _shadow_loop_bindings(self, scoped_names: set[str], induction_names: set[str]) -> None:
        """Hide caller bindings behind this loop's induction and carry parameters."""
        for name in scoped_names:
            self._forget_rename_bookkeeping(name)
            self._shape_inlined.pop(name, None)
            self._var_renames[name] = name  # Also shadows constexpr/free-name folding.
            self._assign_count[name] = 1
            # Carry updates belong to the incoming binding; induction updates
            # belong to the body and may need their own local versions.
            self._assign_depth[name] = self._scope_depth + (name in induction_names)

    def visit_For(self, node: ast.For) -> ast.stmt | list[ast.stmt]:
        """Visit For loop, incrementing scope depth for its body.

        Before entering the body, classify same-depth name collisions
        (:meth:`_classify_loop_locals`).  A *carried* name gets a bridge
        assignment (``x_v1 = x``) before the loop plus a rename inside it,
        preserving loop-carried semantics across two sibling loops.  A *fresh*
        name — a new body-local that merely reuses a sibling loop's name — is
        dropped from the rename bookkeeping so the body re-declares it cleanly:
        neither bridged nor renamed.
        """
        node.iter = self.visit(node.iter)
        if self._is_inactive_plain_loop(node):
            # Keep the loop for parser validation, but unreachable assignments
            # cannot replace the caller's aliases, descriptors, or shape folds.
            node.body = self._visit_inactive_branch(
                node.body,
                yield_renames=self._var_renames.copy(),
                reserve_locals=True,
                induction_names=self._names_by_ctx(node.target, ast.Store),
            )
            if node.orelse:
                node.orelse = self._visit_scoped_body(node.orelse)
            return node
        # Range-like iterators bind an induction variable; pl.while_ binds only
        # carries. Both shadow incoming aliases/folds inside the body, but the
        # parser publishes assignment-form yield targets as post-loop results.
        is_explicit_while = _scope_call_name(node.iter) == "while_"
        scoped_names = self._loop_scoped_names(node)
        induction = node.target.elts[0] if isinstance(node.target, ast.Tuple) else node.target
        induction_names: set[str] = (
            self._names_by_ctx(induction, ast.Store) if not is_explicit_while else set()
        )
        yielded_names = self._loop_yield_targets(node.body)
        carry_meta = self._explicit_carry_meta(node)
        # A published output must agree with its zero-trip initial value.
        # Unknown or conflicting operands cannot acquire static metadata merely
        # because the body happens to yield a known tensor.
        output_meta: dict[str, _TensorBinding | None] = {}
        for assignment in ControlFlowParser.iter_yield_assignments(node.body):
            target = assignment.targets[0] if isinstance(assignment, ast.Assign) else assignment.target
            outputs = target.elts if isinstance(target, ast.Tuple) else [target]
            for output, (_, meta) in zip(outputs, carry_meta):
                if isinstance(output, ast.Name):
                    output_meta[output.id] = meta
        saved = (
            {name: self._meta[name] for name in scoped_names if name in self._meta},
            {name: self._shape_inlined[name] for name in scoped_names if name in self._shape_inlined},
            {name: self._assign_count[name] for name in scoped_names if name in self._assign_count},
            {name: self._assign_depth[name] for name in scoped_names if name in self._assign_depth},
            {name: self._var_renames[name] for name in scoped_names if name in self._var_renames},
        )
        outer_depth = self._scope_depth
        self._shadow_loop_bindings(scoped_names, induction_names)
        for name, meta in carry_meta:
            if meta is not None:
                self._meta[name] = meta
                # The identity rename marks an unknown shadow. Known tensor
                # carries instead resolve their own descriptor through _meta.
                self._var_renames.pop(name, None)
        try:
            bridges = self._materialize_conditional_shapes(node)
            self._scope_depth += 1
            carried, fresh = self._classify_loop_locals(node.body)
            carried = [name for name in carried if name not in scoped_names]
            fresh = [name for name in fresh if name not in scoped_names]
            self._scope_depth -= 1
            # Carried names: bridge + rename so the carried value threads across.
            bridges += self._make_bridge_assignments(carried)
            # Fresh names: forget the same-depth sibling binding so the body's
            # assignment is treated as a first binding (no rename, no bridge).
            for name in fresh:
                self._forget_rename_bookkeeping(name)
            self._scope_depth += 1
            # split_aiv yields belong to the enclosing control-flow scope.
            node.body = self._visit_scoped_body(
                node.body,
                yield_renames=(
                    self._yield_renames
                    if _scope_call_name(node.iter) == "split_aiv"
                    else self._var_renames.copy()
                ),
            )
            self._scope_depth -= 1
            for name, initial_meta in output_meta.items():
                emitted = self._var_renames.get(name, name)
                if self._meta.get(emitted) != initial_meta:
                    self._meta.pop(emitted, None)
            if isinstance(node.target, ast.Name) and not yielded_names:
                self._export_loop_exit_bindings(node)
            if bridges:
                return bridges + [node]
            return node
        finally:
            self._scope_depth = outer_depth
            # A yield-LHS is an output definition even when it reuses a target
            # name. Only unpublished loop-local bindings restore the outer value.
            restore_names = scoped_names - yielded_names
            _restore_bindings(self._meta, saved[0], restore_names)
            _restore_bindings(self._shape_inlined, saved[1], restore_names)
            _restore_bindings(self._assign_count, saved[2], restore_names)
            _restore_bindings(self._assign_depth, saved[3], restore_names)
            _restore_bindings(self._var_renames, saved[4], restore_names)

    def visit_While(self, node: ast.While) -> ast.stmt | list[ast.stmt]:
        """Keep ordinary and SIMT while updates on the incoming carried binding."""
        bridges = self._materialize_conditional_shapes(node)
        if not self._simt_depth:
            # The condition is read before the body writes, so it participates
            # in carry classification and must use any resulting bridge alias.
            bridges += self._prepare_loop_bindings([ast.Expr(value=node.test), *node.body])
        node.test = self.visit(node.test)
        self._scope_depth += 1
        try:
            node.body = self._visit_scoped_body(node.body, yield_renames=self._var_renames.copy())
        finally:
            self._scope_depth -= 1
        if node.orelse:
            node.orelse = self._visit_scoped_body(node.orelse)
        self._export_loop_exit_bindings(node)
        return bridges + [node] if bridges else node

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

    def _visit_inactive_branch(
        self,
        statements: list[ast.stmt],
        *,
        yield_renames: dict[str, str],
        reserve_locals: bool = False,
        induction_names: set[str] | None = None,
    ) -> list[ast.stmt]:
        """Transform a nonjoining region without publishing its caller bindings."""
        state = (
            self._meta.copy(),
            self._shape_inlined.copy(),
            self._assign_count.copy(),
            self._assign_depth.copy(),
            self._var_renames.copy(),
        )
        graph_captures = self._graph_captures
        outer_depth = self._scope_depth
        if graph_captures is not None:
            self._graph_captures = graph_captures.copy()
        try:
            if induction_names:
                self._shadow_loop_bindings(induction_names, induction_names)
                self._scope_depth += 1
                yield_renames = self._var_renames.copy()
            return self._visit_scoped_body(statements, yield_renames=yield_renames)
        finally:
            self._scope_depth = outer_depth
            if reserve_locals:
                self._guard_local_names.update(self._assign_count.keys() - state[2].keys())
            self._meta, self._shape_inlined, self._assign_count, self._assign_depth, self._var_renames = state
            self._graph_captures = graph_captures

    def _visit_literal_if(self, node: ast.If, condition: bool) -> list[ast.stmt]:
        """Emit only the live arm, binding explicit results without an IfStmt."""
        selected = node.body if condition else node.orelse
        for assignment, value in self.selected_yield_assignments(selected):
            assignment.value = value
        return self._visit_scoped_body(selected)

    def _forget_dead_partial_bindings(self, node: ast.If) -> None:
        """Allow a complete branch to replace dead locals from an earlier partial one."""
        if not node.orelse:
            return
        candidates = {name for name, depth in self._assign_depth.items() if depth > self._scope_depth}
        for body in (node.body, node.orelse):
            written: set[str] = set()
            exposed: set[str] = set()
            self._collect_upward_exposed(body, written, exposed, may_skip_loops=True)
            candidates &= written - exposed
        # Both arms overwrite these names before reading them. Their previous
        # conditional binding is not an incoming value for this branch's join.
        for name in candidates:
            self._forget_rename_bookkeeping(name)
            self._shape_inlined.pop(name, None)

    def _normalize_nonjoining_if(self, node: ast.If) -> list[ast.stmt] | None:
        """Hoist the sole reaching arm so its definitions belong to the join's scope."""
        if not node.orelse or any(
            self._is_yield(item)
            if isinstance(item, ast.Call)
            else isinstance(item, ast.withitem) and item.optional_vars is not None
            for item in ast.walk(node)
        ):
            return None
        then_reaches = self._can_reach_join(node.body)
        else_reaches = self._can_reach_join(node.orelse)
        if then_reaches == else_reaches:
            return None
        terminal, reaching = (node.orelse, node.body) if then_reaches else (node.body, node.orelse)
        if then_reaches:
            node.test = ast.copy_location(ast.UnaryOp(op=ast.Not(), operand=node.test), node.test)
        self._scope_depth += 1
        try:
            node.body = self._visit_inactive_branch(
                terminal, yield_renames=self._var_renames.copy(), reserve_locals=True
            )
        finally:
            self._scope_depth -= 1
        node.orelse = []
        return [node, *self._visit_statements(reaching)]

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
        selected = _constant_condition(node.test)
        captures = selected is not None and any(
            isinstance(item, ast.withitem) and isinstance(item.optional_vars, ast.Name)
            for stmt in (*node.body, *node.orelse)
            for item in ast.walk(stmt)
        )
        # Keep main's literal-branch pruning before return-type inference;
        # captured scopes still need the parser's two-branch yield contract.
        if not captures and (condition := values.literal_condition(node.test)) is not None:
            return self._visit_literal_if(node, condition)
        if selected is None:
            self._forget_dead_partial_bindings(node)
            if (normalized := self._normalize_nonjoining_if(node)) is not None:
                return normalized
        prefix = self._materialize_conditional_shapes(node) if selected is None else []
        yield_renames = self._var_renames.copy()
        self._scope_depth += 1
        if selected is not None:
            # Keep both bodies for the parser's yield handling, but publish only
            # the selected branch's aliases and shape folds. Materializing its
            # static shape writes would turn legal unroll bounds into IR Vars.
            if selected:
                if node.orelse:
                    node.orelse = self._visit_inactive_branch(node.orelse, yield_renames=yield_renames)
                node.body = self._visit_scoped_body(node.body, yield_renames=yield_renames)
            else:
                node.body = self._visit_inactive_branch(node.body, yield_renames=yield_renames)
                if node.orelse:
                    node.orelse = self._visit_scoped_body(node.orelse, yield_renames=yield_renames)
            self._scope_depth -= 1
            return prefix + [node] if prefix else node

        before = set(self._assign_count)
        node.body = self._visit_scoped_body(node.body, yield_renames=yield_renames)
        then_bindings = {
            name: (
                self._var_renames.get(name, name),
                self._meta.get(self._var_renames.get(name, name)),
                self._active_shape_fold(name),
            )
            for name in set(self._assign_count) - before
        }
        if node.orelse:
            # then-branch locals are invisible to the else-branch: forget them
            # so a same-named else-branch assignment is a fresh first binding.
            for name in set(self._assign_count) - before:
                self._forget_rename_bookkeeping(name)
            node.orelse = self._visit_scoped_body(node.orelse, yield_renames=yield_renames)
        self._scope_depth -= 1
        if node.orelse:
            self._merge_branch_exports(node, before, then_bindings)
        return prefix + [node] if prefix else node

    @classmethod
    def _can_reach_join(cls, statements: list[ast.stmt]) -> bool:
        """Recognize transfers without crossing a nested loop's ownership."""
        for statement in statements:
            if isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                return False
            if isinstance(statement, ast.If):
                selected = _literal_if_body(statement)
                arms = [selected] if selected is not None else [statement.body, statement.orelse]
                if not any(cls._can_reach_join(arm) for arm in arms):
                    return False
            elif (
                isinstance(statement, ast.With)
                and not any(
                    _scope_call_name(item.context_expr) in {"graph", "simt_scope"} for item in statement.items
                )
                and not cls._can_reach_join(statement.body)
            ):
                return False
        return True

    def _merge_branch_exports(
        self,
        node: ast.If,
        before: set[str],
        then_bindings: dict[str, tuple[str, _TensorBinding | None, int | None]],
    ) -> None:
        """Export live new locals through the same fresh name in both branches."""
        candidates = then_bindings.keys() & (self._assign_count.keys() - before)
        live = self._branch_live_outs.get(id(node))
        if live is not None:
            candidates &= live
        then_reaches = self._can_reach_join(node.body)
        else_reaches = self._can_reach_join(node.orelse)
        exit_exports: dict[str, str] = {}
        for name in sorted(candidates):
            then_alias, then_meta, then_fold = then_bindings[name]
            else_alias = self._var_renames.get(name, name)
            else_meta = self._meta.get(else_alias)
            else_fold = self._active_shape_fold(name)
            joining_meta = [
                meta for meta, reaches in [(then_meta, then_reaches), (else_meta, else_reaches)] if reaches
            ]
            merged_meta = (
                joining_meta[0]
                if joining_meta and all(meta == joining_meta[0] for meta in joining_meta)
                else None
            )
            # Both arms define this live binding, including when no export
            # assignment is needed. Later conditionals must see it as an
            # enclosing variable rather than another branch-local first write.
            self._assign_depth[name] = self._scope_depth
            if then_fold is not None and then_fold == else_fold:
                continue
            if then_alias == else_alias and then_fold is None and else_fold is None:
                self._record_tensor_meta(else_alias, merged_meta)
                continue
            # There is no single Graph predecessor: each arm supplies its own
            # value. A fresh export also permits a local type-changing rebind.
            exported = self._rebind(name, graph_predecessor=False)
            exit_exports[name] = exported
            self._shape_inlined.pop(name, None)
            self._record_tensor_meta(exported, merged_meta)
            for body, alias, folded in [
                (node.body, then_alias, then_fold),
                (node.orelse, else_alias, else_fold),
            ]:
                value = ast.Name(id=alias, ctx=ast.Load()) if folded is None else ast.Constant(value=folded)
                assignment = ast.Assign(targets=[ast.Name(id=exported, ctx=ast.Store())], value=value)
                if not isinstance(body[-1], (ast.Break, ast.Continue, ast.Return, ast.Raise)):
                    body.append(ast.copy_location(assignment, node))
        if exit_exports:
            self._insert_loop_exit_exports(node.body, exit_exports)
            self._insert_loop_exit_exports(node.orelse, exit_exports)

    def _insert_loop_exit_exports(
        self, body: list[ast.stmt], exports: Mapping[str, str], *, join_exit_metadata: bool = False
    ) -> None:
        """Export the value at each loop exit, before later writes can change it."""
        statements: list[ast.stmt] = []
        for stmt in body:
            if isinstance(stmt, (ast.Break, ast.Continue)):
                for name, exported in exports.items():
                    if join_exit_metadata:
                        # A transfer skips the branch join, but can still
                        # contribute a differently shaped post-loop value.
                        exit_meta = self._loop_exit_tensor_meta[id(stmt)].get(name)
                        if exit_meta != self._meta.get(exported):
                            self._meta.pop(exported, None)
                    binding = self._loop_exit_bindings[id(stmt)].get(name)
                    if binding is not None:
                        alias, folded = binding
                        if alias != exported or folded is not None:
                            value = (
                                ast.Name(id=alias, ctx=ast.Load())
                                if folded is None
                                else ast.Constant(value=folded)
                            )
                            assignment = ast.Assign(
                                targets=[ast.Name(id=exported, ctx=ast.Store())], value=value
                            )
                            statements.append(ast.copy_location(assignment, stmt))
            elif isinstance(stmt, ast.If):
                self._insert_loop_exit_exports(stmt.body, exports, join_exit_metadata=join_exit_metadata)
                self._insert_loop_exit_exports(stmt.orelse, exports, join_exit_metadata=join_exit_metadata)
            elif isinstance(stmt, ast.With):
                self._insert_loop_exit_exports(stmt.body, exports, join_exit_metadata=join_exit_metadata)
            # A nested loop owns its own break/continue statements.
            statements.append(stmt)
        body[:] = statements

    def _export_loop_exit_bindings(self, node: ast.For | ast.While) -> None:
        """Publish exit values under the final spelling used after this loop.

        A later fallthrough assignment can create a new alias after an earlier
        branch's break/continue has already been visited. Resolve the target
        only once the complete body is rewritten, then use the recorded value
        at each transfer. The inserter deliberately skips nested loops.
        """
        live = self._branch_live_outs.get(id(node), set())
        exports = {
            name: self._var_renames.get(name, name)
            for name in sorted(live & self._exported_loop_names([node]))
        }
        if exports:
            self._insert_loop_exit_exports(node.body, exports, join_exit_metadata=True)

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
        scope_names = {_scope_call_name(item.context_expr) for item in node.items}
        graph = "graph" in scope_names
        exported_names = self._names_by_ctx(node, ast.Store)
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
        if scope_names & {"scope", "manual_scope"}:
            # Transparent with scopes export their bindings to the caller. A
            # later sibling if must update that binding on its false path too.
            self._promote_scope_bindings(exported_names)
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
