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
from collections.abc import Mapping
from typing import Any, cast

from . import _specialize_values as values
from ._specialize_types import DynDim, TensorMeta


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
        self._meta = dict(tensor_meta)
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
        # Reverse map: generated alias → original user variable name.
        # Used to rewrite error messages so users see their original names.
        self._alias_to_original: dict[str, str] = {}
        # Tracks the scope depth at which each variable was first assigned.
        # Rebindings at a deeper scope (e.g. inside a for/with) are loop-carried
        # updates and must NOT be renamed.
        self._assign_depth: dict[str, int] = {}
        self._scope_depth: int = 0
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

    def _rebind(self, var_name: str) -> str:
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
        self._var_renames[var_name] = new_name
        self._alias_to_original[new_name] = var_name
        self._used_names.add(new_name)
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

    def _visit_simple_assign(self, node: ast.Assign) -> ast.stmt:
        """Handle a single-target name assignment with alpha-renaming support."""
        var_name = cast(ast.Name, node.targets[0]).id
        source = self._known_tensor_name(node.value.id) if isinstance(node.value, ast.Name) else None
        value_meta = self._meta.get(source) if source is not None else None
        visited_value = self.visit(node.value)
        if var_name in self._assign_count:
            # Only rename at the same scope where the variable was first defined.
            # Rebindings at a deeper scope (inside for/with) are loop-carried
            # updates — do not rename them.
            if self._scope_depth == self._assign_depth[var_name]:
                new_name = self._rebind(var_name)
                self._record_tensor_meta(new_name, value_meta)
                new_node = ast.Assign(
                    targets=[ast.Name(id=new_name, ctx=ast.Store())],
                    value=visited_value,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                )
                ast.fix_missing_locations(new_node)
                return new_node
            # Deeper scope: if the variable has an active rename (from a bridge
            # assignment), apply it to the LHS so the loop body consistently
            # uses the renamed alias (e.g. x_v1 = some_op(x_v1)).
            if var_name in self._var_renames:
                renamed = self._var_renames[var_name]
                node.targets = [ast.Name(id=renamed, ctx=ast.Store())]
            self._record_tensor_meta(self._var_renames.get(var_name, var_name), value_meta, reuses_name=True)
            node.value = visited_value
            return node
        # First assignment: record and keep original name.
        self._record_first_assign(var_name)
        self._record_tensor_meta(var_name, value_meta)
        node.value = visited_value
        return node

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
        meta = self._meta[param_name]
        elts = [self._shape_dim_node(param_name, i) for i in range(len(meta.shape))]
        return ast.Tuple(elts=elts, ctx=ast.Load())

    def _shape_dim_node(self, param_name: str, dim_idx: int) -> ast.expr:
        d = self._meta[param_name].shape[dim_idx]
        if isinstance(d, DynDim) or param_name in self._alias_to_original:
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
                stmts: list[ast.stmt] = []
                for i, (tgt, dim) in enumerate(zip(target_names, meta.shape, strict=True)):
                    if not isinstance(tgt, ast.Name):
                        continue
                    if isinstance(dim, DynDim) or param_name in self._alias_to_original:
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
                        self._record_first_assign(tgt.id)
                        self._shape_inlined[tgt.id] = dim
                return stmts if stmts else None

        # Case 3: K = a.shape[1]  →  inline static dim, suppress assignment
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Attribute)
            and node.value.value.attr == "shape"
            and isinstance(node.value.value.value, ast.Name)
            and node.value.value.value.id in self._meta
            and node.value.value.value.id not in self._var_renames
            and isinstance(node.value.slice, ast.Constant)
            and isinstance(node.value.slice.value, int)
        ):
            param_name = node.value.value.value.id
            dim_idx = node.value.slice.value
            dim = self._meta[param_name].shape[dim_idx]
            if not isinstance(dim, DynDim):
                self._shape_inlined[node.targets[0].id] = dim
                self._record_first_assign(node.targets[0].id)
                return None

        # Default: visit the RHS first (applies existing renames to operands),
        # then handle rebinding rename on the LHS.
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            return self._visit_simple_assign(node)

        return cast("ast.stmt", self.generic_visit(node))

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.stmt | None:
        """Handle annotated assignments (``x: T = value``) with alpha-renaming."""
        if node.value is None:
            return cast("ast.stmt", self.generic_visit(node))
        if not isinstance(node.target, ast.Name):
            return cast("ast.stmt", self.generic_visit(node))
        var_name = node.target.id
        source = self._known_tensor_name(node.value.id) if isinstance(node.value, ast.Name) else None
        value_meta = self._meta.get(source) if source is not None else None
        node.value = self.visit(node.value)
        # Inline shape constants in the local annotation too (e.g. a body-level
        # ``x: pl.Tile[[1, W_PAD], pl.FP32]`` where ``W_PAD`` is a module-level
        # int). Without this the un-inlined name leaks into the generated source
        # and the parser rejects it ("Unknown shape variable: W_PAD").
        node.annotation = self.visit(node.annotation)
        if var_name in self._assign_count:
            if self._scope_depth == self._assign_depth[var_name]:
                new_name = self._rebind(var_name)
                self._record_tensor_meta(new_name, value_meta)
                new_node = ast.AnnAssign(
                    target=ast.Name(id=new_name, ctx=ast.Store()),
                    annotation=node.annotation,
                    value=node.value,
                    simple=node.simple,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                )
                ast.fix_missing_locations(new_node)
                return new_node
            current_name = self._var_renames.get(var_name, var_name)
            node.target = ast.copy_location(ast.Name(id=current_name, ctx=ast.Store()), node.target)
            self._record_tensor_meta(current_name, value_meta, reuses_name=True)
        else:
            self._record_first_assign(var_name)
            self._record_tensor_meta(var_name, value_meta)
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
        return cast("ast.expr", self.generic_visit(node))

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        """Replace a.shape[i] → integer constant."""
        if (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "shape"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id in self._meta
        ):
            param_name = node.value.value.id
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                if param_name in self._var_renames:
                    # The parameter's original shape no longer describes this value.
                    return self._dyn_dim_expr(self._var_renames[param_name], node.slice.value)
                return self._shape_dim_node(param_name, node.slice.value)
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
        if isinstance(node.func, ast.Name) and node.func.id in self._dep_names:
            dep_name = self._dep_call_targets.get(
                id(node), self._dep_func_names.get(node.func.id, node.func.id)
            )
            new_func = ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load()),
                attr=dep_name,
                ctx=ast.Load(),
            )
            param_order = self._dep_param_names.get(dep_name)
            constexpr_params = self._dep_constexpr_params.get(dep_name, set())
            if param_order is not None and (node.keywords or constexpr_params):
                pos_args = list(node.args)
                kw_by_name = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
                # Append keyword-bound args in dep's parameter order, skipping
                # params already covered by positional args.
                for param in param_order[len(pos_args) :]:
                    if param in kw_by_name:
                        pos_args.append(kw_by_name.pop(param))
                # Any remaining keywords (e.g. **kwargs splats or unknown names)
                # are kept as-is so we don't silently drop them; the parser
                # will surface them as a clear error.
                remaining_kw = [kw for kw in node.keywords if kw.arg is None or kw.arg in kw_by_name]
                if constexpr_params:
                    # Drop the callee's compile-time arguments. It resolved them
                    # during specialization and its generated signature no longer
                    # declares them, so passing one would be an arity mismatch.
                    # Filter by position against the *declared* order, which still
                    # includes them; anything beyond it is left alone for the
                    # parser to report.
                    pos_args = [
                        arg
                        for index, arg in enumerate(pos_args)
                        if index >= len(param_order) or param_order[index] not in constexpr_params
                    ]
                    remaining_kw = [kw for kw in remaining_kw if kw.arg not in constexpr_params]
                new_node = ast.Call(func=new_func, args=pos_args, keywords=remaining_kw)
            else:
                new_node = ast.Call(func=new_func, args=node.args, keywords=node.keywords)
            ast.copy_location(new_node, node)
            return cast("ast.expr", self.generic_visit(new_node))
        return cast("ast.expr", self.generic_visit(node))

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
        return result or [ast.Pass()]

    @staticmethod
    def _names_by_ctx(node: ast.AST, ctx_type: type) -> set[str]:
        """Collect ``Name`` ids appearing under the given context (Load/Store)."""
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ctx_type)}

    def _collect_upward_exposed(
        self, statements: list[ast.stmt], written: set[str], exposed: set[str]
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
                written.update(self._names_by_ctx(stmt.target, ast.Store))
                self._collect_upward_exposed(stmt.body, written, exposed)
                self._collect_upward_exposed(stmt.orelse, written, exposed)
            elif isinstance(stmt, ast.If):
                exposed.update(self._names_by_ctx(stmt.test, ast.Load) - written)
                # then/else are alternative paths: analyze each from the same
                # incoming ``written`` so a read in one branch is not masked by
                # a write in the other.  A name is exposed if read-before-write
                # on either path; it is definitely written after the ``if``
                # only if written on both (no ``else`` → no definite write).
                then_written = set(written)
                self._collect_upward_exposed(stmt.body, then_written, exposed)
                if stmt.orelse:
                    else_written = set(written)
                    self._collect_upward_exposed(stmt.orelse, else_written, exposed)
                    written.update(then_written & else_written)
            elif isinstance(stmt, ast.With):
                for item in stmt.items:
                    exposed.update(self._names_by_ctx(item.context_expr, ast.Load) - written)
                    if item.optional_vars is not None:
                        written.update(self._names_by_ctx(item.optional_vars, ast.Store))
                self._collect_upward_exposed(stmt.body, written, exposed)
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
        self._scope_depth += 1
        carried, fresh = self._classify_loop_locals(node.body)
        self._scope_depth -= 1
        # Carried names: bridge + rename so the carried value threads across.
        bridges = self._make_bridge_assignments(carried)
        # Fresh names: forget the same-depth sibling binding so the body's
        # assignment is treated as a first binding (no rename, no bridge).
        for name in fresh:
            self._forget_rename_bookkeeping(name)
        self._scope_depth += 1
        node.body = self._visit_scoped_body(node.body)
        self._scope_depth -= 1
        if bridges:
            return bridges + [node]
        return node

    def visit_If(self, node: ast.If) -> ast.stmt:
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
        self._scope_depth += 1
        before = set(self._assign_count)
        node.body = self._visit_scoped_body(node.body)
        if node.orelse:
            # then-branch locals are invisible to the else-branch: forget them
            # so a same-named else-branch assignment is a fresh first binding.
            for name in set(self._assign_count) - before:
                self._forget_rename_bookkeeping(name)
            node.orelse = self._visit_scoped_body(node.orelse)
        self._scope_depth -= 1
        return node

    def visit_With(self, node: ast.With) -> ast.stmt:
        """Visit With block, incrementing scope depth for its body."""
        node.items = [self.visit(item) for item in node.items]
        self._scope_depth += 1
        node.body = self._visit_scoped_body(node.body)
        self._scope_depth -= 1
        return node
