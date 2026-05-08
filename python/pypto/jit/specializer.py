# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""AST specializer: transform @pl.jit source into @pl.program source.

Given concrete tensor shapes/dtypes and scalar values collected from a call
site, this module rewrites a JIT-decorated function (and its @pl.jit.incore
dependencies) into valid @pl.program / @pl.function source code that the
existing parser can consume unchanged.

Transformation rules
--------------------
User writes (JIT style)            Generated DSL (@pl.program style)
─────────────────────────────────  ──────────────────────────────────
a: pl.Tensor                       a: pl.Tensor[[128, 128], pl.FP32]
param: pl.INDEX                    param: pl.Scalar[pl.INDEX]  (value substituted)
M = pl.dynamic("M")               Promoted to module level; shared dynvar_cache
a.bind_dynamic(0, M)              Deleted (info already used in annotation)
K = a.shape[1]                    K = 128
M, N = a.shape                    M = 128\\nN = 128  (or Var name for dynamic)
a.shape                            (128, 128)
a.dtype                            pl.FP32
other_jit_func(a, b)              self.other_jit_func(a, b)  (multi-function only)
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Any, cast

from pypto.pypto_core import DataType

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TensorMeta:
    """Runtime metadata for a single tensor parameter.

    Attributes:
        shape: Concrete shape tuple from the torch tensor.
        dtype: DataType resolved from the torch tensor.
    """

    shape: tuple[int, ...]
    dtype: DataType


@dataclass
class SpecializeContext:
    """All information needed to specialize a single JIT function.

    Attributes:
        func_name: Python function name.
        source: Dedented source code of the function.
        func_type: 'orchestration' | 'incore' | 'inline' | 'opaque' | None (auto).
        level: pl.Level value or None.
        param_names: Ordered parameter names (excluding 'self').
        tensor_meta: TensorMeta per tensor param name.
        scalar_values: Concrete value per scalar param name.
        scalar_dtypes: DataType annotation per scalar param name.
        dynamic_dims: Set of (param_name, dim_index) pairs marked dynamic.
        dep_names: Names of dep functions called from this function.
        py_globals: The originating function's ``__globals__``. The specializer
            uses this to resolve module-level int/float/bool constants (e.g.
            ``BATCH``, ``HIDDEN`` imported from a config module) by inlining
            them at the use site.
    """

    func_name: str
    source: str
    func_type: str | None
    level: Any
    param_names: list[str]
    tensor_meta: dict[str, TensorMeta]
    scalar_values: dict[str, int | float | bool]
    scalar_dtypes: dict[str, DataType]
    dynamic_dims: set[tuple[str, int]] = field(default_factory=set)
    dep_names: list[str] = field(default_factory=list)
    py_globals: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DataType → pl.XXX string mapping
# ---------------------------------------------------------------------------

_DTYPE_TO_PL: dict[DataType, str] = {
    DataType.FP4: "pl.FP4",
    DataType.FP8E4M3FN: "pl.FP8E4M3FN",
    DataType.FP8E5M2: "pl.FP8E5M2",
    DataType.FP16: "pl.FP16",
    DataType.FP32: "pl.FP32",
    DataType.BF16: "pl.BF16",
    DataType.HF4: "pl.HF4",
    DataType.HF8: "pl.HF8",
    DataType.INT4: "pl.INT4",
    DataType.INT8: "pl.INT8",
    DataType.INT16: "pl.INT16",
    DataType.INT32: "pl.INT32",
    DataType.INT64: "pl.INT64",
    DataType.UINT4: "pl.UINT4",
    DataType.UINT8: "pl.UINT8",
    DataType.UINT16: "pl.UINT16",
    DataType.UINT32: "pl.UINT32",
    DataType.UINT64: "pl.UINT64",
    DataType.BOOL: "pl.BOOL",
    DataType.INDEX: "pl.INDEX",
}


# Set of bare dtype name strings (e.g. "FP32", "INT8") for annotation parsing.
# Derived from _DTYPE_TO_PL to avoid duplication.
_DTYPE_NAMES: frozenset[str] = frozenset(v.split(".")[1] for v in _DTYPE_TO_PL.values())


def _dtype_str(dt: DataType) -> str:
    if dt not in _DTYPE_TO_PL:
        raise ValueError(f"Unsupported DataType: {dt}")
    return _DTYPE_TO_PL[dt]


# ---------------------------------------------------------------------------
# Pre-scan: collect bind_dynamic calls before body transformation
# ---------------------------------------------------------------------------


def _collect_dynamic_dims(
    func_def: ast.FunctionDef,
    param_names: set[str],
) -> set[tuple[str, int]]:
    """Scan a function body for ``param.bind_dynamic(dim_idx, dynvar)`` calls.

    Returns a set of (param_name, dim_index) pairs.
    """
    result: set[tuple[str, int]] = set()
    for node in ast.walk(func_def):
        if not isinstance(node, ast.Expr):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "bind_dynamic"
            and isinstance(func.value, ast.Name)
            and func.value.id in param_names
        ):
            continue
        if len(call.args) < 1:
            continue
        dim_node = call.args[0]
        if isinstance(dim_node, ast.Constant) and isinstance(dim_node.value, int):
            result.add((func.value.id, dim_node.value))
    return result


def _collect_dynvar_names(func_def: ast.FunctionDef) -> dict[str, str]:
    """Collect dynvar assignments from pl.dynamic(...) calls in the function body.

    Returns a dict mapping Python variable name → string literal passed to
    pl.dynamic().  For example, ``rows = pl.dynamic("M")`` produces
    ``{"rows": "M"}``.  When the literal cannot be determined statically, the
    variable name itself is used as the fallback.
    """
    result: dict[str, str] = {}
    for node in ast.walk(func_def):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        call = node.value
        func = call.func
        is_pl_dynamic = (
            isinstance(func, ast.Attribute)
            and func.attr == "dynamic"
            and isinstance(func.value, ast.Name)
            and func.value.id == "pl"
        ) or (isinstance(func, ast.Name) and func.id == "dynamic")
        if not is_pl_dynamic:
            continue
        # Extract the string literal argument, e.g. pl.dynamic("M") → "M"
        dyn_literal: str | None = None
        if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
            dyn_literal = call.args[0].value
        for target in node.targets:
            if isinstance(target, ast.Name):
                result[target.id] = dyn_literal if dyn_literal is not None else target.id
    return result


def _collect_dep_names(func_def: ast.FunctionDef, jit_func_names: set[str]) -> list[str]:
    """Return names of @pl.jit.incore functions called in this function body."""
    deps: list[str] = []
    seen: set[str] = set()
    for node in ast.walk(func_def):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in jit_func_names and func.id not in seen:
            deps.append(func.id)
            seen.add(func.id)
    return deps


# ---------------------------------------------------------------------------
# Build annotated type string for a parameter
# ---------------------------------------------------------------------------


def _build_tensor_annotation(
    param_name: str,
    meta: TensorMeta,
    dynamic_dims: set[tuple[str, int]],
    dynvar_names: dict[str, str],
    is_out: bool,
) -> str:
    """Build the full type annotation string for a tensor parameter.

    Args:
        param_name: Parameter name (for dynamic dim lookup).
        meta: TensorMeta with concrete shape and dtype.
        dynamic_dims: Set of (param_name, dim_idx) dynamic pairs.
        dynvar_names: Maps dimension variable name → DynVar Python variable name
            used in the generated module scope.
        is_out: Whether the parameter is annotated Out[...].

    Returns:
        Annotation string such as ``pl.Tensor[[M_dyn, 128], pl.FP32]`` or
        ``pl.Out[pl.Tensor[[128, 128], pl.FP32]]``.
    """
    dims: list[str] = []
    for i, dim in enumerate(meta.shape):
        if (param_name, i) in dynamic_dims:
            # Find the dynvar name for this dim.  We look up by searching
            # dynvar_names for a matching variable; if multiple dynvars cover
            # different dims of the same param we rely on the caller having
            # collected them correctly.  Fall back to a generated name.
            dv_name = _dynvar_name_for_dim(param_name, i, dynvar_names)
            dims.append(dv_name)
        else:
            dims.append(str(dim))
    inner = f"pl.Tensor[[{', '.join(dims)}], {_dtype_str(meta.dtype)}]"
    if is_out:
        return f"pl.Out[{inner}]"
    return inner


def _dynvar_name_for_dim(
    param_name: str,
    dim_idx: int,
    dynvar_names: dict[str, str],
) -> str:
    """Return the Python variable name to use for a dynamic dimension.

    ``dynvar_names`` maps (param_name, dim_idx) encoded as
    ``"<param>__<idx>"`` to the DynVar variable name.  Falls back to a
    generated name if no explicit binding is found.
    """
    key = f"{param_name}__{dim_idx}"
    return dynvar_names.get(key, f"_dyn_{param_name}_{dim_idx}")


# ---------------------------------------------------------------------------
# AST node transformer
# ---------------------------------------------------------------------------


class _BodyTransformer(ast.NodeTransformer):
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
        scalar_values: dict[str, int | float | bool],
        dynamic_dims: set[tuple[str, int]],
        dynvar_python_names: dict[str, str],
        dep_names: set[str],
        dynvar_var_names: set[str],
        param_names: list[str] | None = None,
        initial_used_names: set[str] | None = None,
        py_globals: dict[str, Any] | None = None,
        dep_param_names: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__()
        self._meta = tensor_meta
        self._scalars = scalar_values
        self._dynamic_dims = dynamic_dims
        # Maps "<param>__<dim_idx>" → python var name for the DynVar
        self._dv_names = dynvar_python_names
        self._dep_names = dep_names
        self._dynvar_var_names = dynvar_var_names
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

    def _visit_simple_assign(self, node: ast.Assign) -> ast.stmt:
        """Handle a single-target name assignment with alpha-renaming support."""
        var_name = cast(ast.Name, node.targets[0]).id
        visited_value = self.visit(node.value)
        if var_name in self._assign_count:
            # Only rename at the same scope where the variable was first defined.
            # Rebindings at a deeper scope (inside for/with) are loop-carried
            # updates — do not rename them.
            if self._scope_depth == self._assign_depth[var_name]:
                new_name = self._rebind(var_name)
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
            node.value = visited_value
            return node
        # First assignment: record and keep original name.
        self._record_first_assign(var_name)
        node.value = visited_value
        return node

    def _shape_tuple_node(self, param_name: str) -> ast.Tuple:
        meta = self._meta[param_name]
        elts: list[ast.expr] = []
        for i, dim in enumerate(meta.shape):
            if (param_name, i) in self._dynamic_dims:
                dv = _dynvar_name_for_dim(param_name, i, self._dv_names)
                elts.append(ast.Name(id=dv, ctx=ast.Load()))
            else:
                elts.append(ast.Constant(value=dim))
        return ast.Tuple(elts=elts, ctx=ast.Load())

    def _shape_dim_node(self, param_name: str, dim_idx: int) -> ast.expr:
        meta = self._meta[param_name]
        if (param_name, dim_idx) in self._dynamic_dims:
            dv = _dynvar_name_for_dim(param_name, dim_idx, self._dv_names)
            return ast.Name(id=dv, ctx=ast.Load())
        return ast.Constant(value=meta.shape[dim_idx])

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
                return None

        # Case 2: M, N = a.shape  →  individual assignments (dynamic) or inline (static)
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Tuple)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "shape"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id in self._meta
        ):
            param_name = node.value.value.id
            meta = self._meta[param_name]
            target_names = node.targets[0].elts
            if len(target_names) == len(meta.shape):
                stmts: list[ast.stmt] = []
                for tgt, (i, dim) in zip(target_names, enumerate(meta.shape)):
                    if isinstance(tgt, ast.Name):
                        if (param_name, i) in self._dynamic_dims:
                            # Dynamic dim: emit assignment (M = <dynvar ref>),
                            # but skip if it would be a no-op (LHS name == RHS name).
                            val: ast.expr = self._shape_dim_node(param_name, i)
                            if not (isinstance(val, ast.Name) and val.id == tgt.id):
                                lhs_name = tgt.id
                                if lhs_name in self._assign_count:
                                    lhs_name = self._rebind(lhs_name)
                                else:
                                    self._record_first_assign(tgt.id)
                                stmts.append(
                                    ast.Assign(
                                        targets=[ast.Name(id=lhs_name, ctx=ast.Store())],
                                        value=val,
                                        lineno=node.lineno,
                                        col_offset=node.col_offset,
                                    )
                                )
                            else:
                                self._record_first_assign(tgt.id)
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
            and isinstance(node.value.slice, ast.Constant)
            and isinstance(node.value.slice.value, int)
        ):
            param_name = node.value.value.value.id
            dim_idx = node.value.slice.value
            if (param_name, dim_idx) not in self._dynamic_dims:
                meta = self._meta[param_name]
                self._shape_inlined[node.targets[0].id] = meta.shape[dim_idx]
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
        node.value = self.visit(node.value)
        if var_name in self._assign_count:
            if self._scope_depth == self._assign_depth[var_name]:
                new_name = self._rebind(var_name)
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
        else:
            self._record_first_assign(var_name)
        return cast("ast.stmt", node)

    # ------------------------------------------------------------------
    # Expression-level transforms
    # ------------------------------------------------------------------

    def visit_Name(self, node: ast.Name) -> ast.expr:
        """Replace scalar param references, inlined shape constants, renamed rebindings,
        and module-level int/float/bool constants imported from globals."""
        if isinstance(node.ctx, ast.Load):
            # Check active renames first — a rebinding supersedes any earlier inlining.
            if node.id in self._var_renames:
                return ast.Name(id=self._var_renames[node.id], ctx=ast.Load())
            if node.id in self._scalars:
                return ast.Constant(value=self._scalars[node.id])
            if node.id in self._shape_inlined:
                return ast.Constant(value=self._shape_inlined[node.id])
            # Module-level constant inlining: only when the name is not also a
            # local (function param or assigned-in-body) — those take priority.
            if node.id not in self._used_names:
                value = self._py_globals.get(node.id)
                # Accept int/float (excluding bool, which would otherwise be picked
                # up here despite being a separate semantic — but also bool is
                # commonly used as a literal flag, so include it).
                if isinstance(value, (int, float, bool)) and not isinstance(value, type):
                    return ast.Constant(value=value)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        """Replace a.shape → tuple and a.dtype → pl.XXX."""
        if isinstance(node.value, ast.Name) and node.value.id in self._meta:
            param_name = node.value.id
            if node.attr == "shape":
                return self._shape_tuple_node(param_name)
            if node.attr == "dtype":
                dtype_s = _dtype_str(self._meta[param_name].dtype)
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
                return self._shape_dim_node(param_name, node.slice.value)
        return cast("ast.expr", self.generic_visit(node))

    def visit_Call(self, node: ast.Call) -> ast.expr:
        """Rewrite dep_func(args) → self.dep_func(args) for multi-function JIT.

        Keyword args are normalised to positional based on the dep's
        parameter order (so ``dep(a, out=out)`` becomes ``self.dep(a, out)``).
        Inter-function calls inside ``@pl.program`` only accept positional
        args — preserving keyword form would make the parser reject the call.
        """
        if isinstance(node.func, ast.Name) and node.func.id in self._dep_names:
            dep_name = node.func.id
            new_func = ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load()),
                attr=dep_name,
                ctx=ast.Load(),
            )
            param_order = self._dep_param_names.get(dep_name)
            if param_order is not None and node.keywords:
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
                result.extend(visited)
            else:
                result.append(visited)
        return result or [ast.Pass()]

    def _scan_for_rebinds(self, statements: list[ast.stmt]) -> list[str]:
        """Scan a body for variable names that would trigger alpha-renaming.

        Returns variable names that are already in ``_assign_count`` and whose
        first assignment was at the same scope depth as the current depth.
        These need bridge assignments (``x_v1 = x``) before entering the scope
        so that loop-carried reads inside the body resolve to the new alias.
        """
        rebind_vars: list[str] = []
        seen: set[str] = set()
        for stmt in statements:
            # Check direct single-target assignments
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name):
                    name = target.id
                    if (
                        name not in seen
                        and name in self._assign_count
                        and self._scope_depth == self._assign_depth.get(name, -1)
                    ):
                        rebind_vars.append(name)
                        seen.add(name)
            # Check annotated assignments
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                name = stmt.target.id
                if (
                    name not in seen
                    and name in self._assign_count
                    and self._scope_depth == self._assign_depth.get(name, -1)
                ):
                    rebind_vars.append(name)
                    seen.add(name)
        return rebind_vars

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
            new_name = self._rebind(var_name)
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

        Before entering the body, scan for variables that would be rebound at
        the same depth.  For each such variable, insert a bridge assignment
        (``x_v1 = x``) before the loop and apply the rename so that both LHS
        and RHS inside the loop body use the new alias.  This preserves
        loop-carried semantics when two parallel for loops assign to the same
        variable.
        """
        node.iter = self.visit(node.iter)
        self._scope_depth += 1
        # Scan for rebinds and insert bridge assignments before the loop
        rebind_vars = self._scan_for_rebinds(node.body)
        self._scope_depth -= 1
        bridges = self._make_bridge_assignments(rebind_vars)
        self._scope_depth += 1
        node.body = self._visit_scoped_body(node.body)
        self._scope_depth -= 1
        if bridges:
            return bridges + [node]
        return node

    def visit_If(self, node: ast.If) -> ast.stmt:
        """Visit If statement, incrementing scope depth for both branches.

        If-branch assignments are loop-carried in the Parser's view (variables
        leak to outer scope via exit_scope(leak_vars=True)) so they must NOT be
        alpha-renamed at this layer — the Parser and ConvertToSSA handle them.
        """
        node.test = self.visit(node.test)
        self._scope_depth += 1
        node.body = self._visit_scoped_body(node.body)
        if node.orelse:
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


# ---------------------------------------------------------------------------
# Return-type inference (Option A: from Out params)
# ---------------------------------------------------------------------------


def _infer_return_type(
    func_def: ast.FunctionDef,
    tensor_meta: dict[str, TensorMeta],
    dynamic_dims: set[tuple[str, int]],
    dynvar_names: dict[str, str],
    out_params: list[str],
) -> str | None:
    """Infer the return type annotation string from Out parameters.

    Examines the function's return statement.  If it returns a name that
    is in out_params, uses that param's tensor type (without Out[]).
    Falls back to the first Out param if the return statement is absent or
    not a simple name.

    Returns None if no Out params exist (return type cannot be inferred).
    """
    if not out_params:
        return None

    # Try to find a bare return statement
    returned_name: str | None = None
    for node in ast.walk(func_def):
        if isinstance(node, ast.Return) and node.value is not None:
            if isinstance(node.value, ast.Name):
                returned_name = node.value.id
            break

    target_param = returned_name if returned_name in out_params else out_params[0]

    if target_param not in tensor_meta:
        return None

    meta = tensor_meta[target_param]
    return _build_tensor_annotation(target_param, meta, dynamic_dims, dynvar_names, is_out=False)


# ---------------------------------------------------------------------------
# Annotation classifier: detect Out[], plain Tensor, Scalar params
# ---------------------------------------------------------------------------


def _classify_params(
    func_def: ast.FunctionDef,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Classify function parameters.

    Returns:
        (out_params, tensor_params, scalar_dtype_strs)
        - out_params: names annotated Out[pl.Tensor]
        - tensor_params: all names annotated pl.Tensor (including Out ones)
        - scalar_dtype_strs: param_name → dtype string for scalar params
    """
    out_params: list[str] = []
    tensor_params: list[str] = []
    scalar_dtype_strs: dict[str, str] = {}

    for arg in func_def.args.args:
        name = arg.arg
        if name == "self":
            continue
        ann = arg.annotation
        if ann is None:
            continue

        # Detect Out[pl.Tensor] or pl.Out[pl.Tensor]
        if isinstance(ann, ast.Subscript):
            outer = ann.value
            is_out = (isinstance(outer, ast.Name) and outer.id == "Out") or (
                isinstance(outer, ast.Attribute) and outer.attr == "Out"
            )
            # The inner subscript value
            inner = ann.slice
            is_tensor = _is_tensor_annotation(inner)
            if is_out and is_tensor:
                out_params.append(name)
                tensor_params.append(name)
                continue
            # Check for Scalar dtype annotations: pl.INDEX, pl.FP32, etc.
            is_scalar_ann = _is_scalar_annotation(outer)
            if is_scalar_ann:
                scalar_dtype_strs[name] = _ast_to_str(inner)
                continue

        # Detect pl.Tensor (bare, no subscript)
        if _is_tensor_annotation(ann):
            tensor_params.append(name)
            continue

        # Detect bare dtype annotation: pl.INDEX, pl.FP32, etc. (no Scalar wrapper)
        dtype_str = _extract_bare_dtype(ann)
        if dtype_str is not None:
            scalar_dtype_strs[name] = dtype_str

    return out_params, tensor_params, scalar_dtype_strs


def _is_tensor_annotation(node: ast.expr) -> bool:
    """Return True if the AST node represents pl.Tensor or Tensor (bare or subscripted)."""
    if isinstance(node, ast.Name):
        return node.id == "Tensor"
    if isinstance(node, ast.Attribute):
        return node.attr == "Tensor"
    # Handle subscripted form: pl.Tensor[[...], dtype] or Tensor[[...], dtype]
    if isinstance(node, ast.Subscript):
        return _is_tensor_annotation(node.value)
    return False


def _is_scalar_annotation(node: ast.expr) -> bool:
    """Return True if the AST node represents pl.Scalar or Scalar."""
    if isinstance(node, ast.Name):
        return node.id == "Scalar"
    if isinstance(node, ast.Attribute):
        return node.attr == "Scalar"
    return False


def _extract_bare_dtype(node: ast.expr) -> str | None:
    """If node is a bare dtype like pl.FP32 or INDEX, return its string."""
    if isinstance(node, ast.Name) and node.id in _DTYPE_NAMES:
        return f"pl.{node.id}"
    if isinstance(node, ast.Attribute) and node.attr in _DTYPE_NAMES:
        return f"pl.{node.attr}"
    return None


def _ast_to_str(node: ast.expr) -> str:
    """Render a simple AST expression back to source."""
    return ast.unparse(node)


# ---------------------------------------------------------------------------
# InCore scope detection
# ---------------------------------------------------------------------------


def _has_incore_scope(func_def: ast.FunctionDef) -> bool:
    """Return True if the function body contains a ``with pl.incore():`` scope.

    Detects ``with pl.incore():`` and ``with pl.auto_incore():`` — the forms
    that OutlineIncoreScopes processes.  Does not recurse into nested function
    definitions.
    """
    for node in ast.walk(func_def):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            ctx_expr = item.context_expr
            if not isinstance(ctx_expr, ast.Call):
                continue
            func = ctx_expr.func
            # pl.incore() or pl.auto_incore()
            if isinstance(func, ast.Attribute) and func.attr in ("incore", "auto_incore"):
                return True
            # bare incore() or auto_incore() (less common)
            if isinstance(func, ast.Name) and func.id in ("incore", "auto_incore"):
                return True
    return False


# ---------------------------------------------------------------------------
# Main specializer
# ---------------------------------------------------------------------------


class Specializer:
    """Transform a collection of SpecializeContext objects into @pl.program source.

    Usage::

        s = Specializer(class_name, contexts, dynvar_bindings)
        source_code = s.specialize()
        program = pl.parse(source_code)
    """

    def __init__(
        self,
        class_name: str,
        contexts: list[SpecializeContext],
        dynvar_bindings: dict[str, str],
        dynvar_literals: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            class_name: Name for the generated @pl.program class.
            contexts: List of SpecializeContext, one per function.
                The entry-point (Orchestration) function must be last.
            dynvar_bindings: Maps "<param>__<dim_idx>" → DynVar Python variable
                name used in the generated source.
            dynvar_literals: Maps DynVar Python variable name → string literal
                passed to pl.dynamic().  For example, if the user wrote
                ``rows = pl.dynamic("M")``, this should be ``{"rows": "M"}``
                so the generated code emits ``rows = pl.dynamic("M")`` rather
                than ``rows = pl.dynamic("rows")``.  Defaults to using the
                variable name as the literal when not provided.
        """
        self._class_name = class_name
        self._contexts = contexts
        self._dv_bindings = dynvar_bindings
        self._dv_literals = dynvar_literals or {}
        # Accumulated alias→original map across all specialized functions.
        self._rename_map: dict[str, str] = {}
        # Param order per function — lets the body transformer rewrite
        # ``self.<dep>(a, out=out)`` keyword args into all-positional form
        # (the parser rejects keyword args on inter-function calls).
        self._dep_param_names: dict[str, list[str]] = {
            ctx.func_name: list(ctx.param_names) for ctx in contexts
        }

    @property
    def rename_map(self) -> dict[str, str]:
        """Return mapping from generated alias → original user variable name.

        Only populated after ``specialize()`` has been called.
        """
        return dict(self._rename_map)

    def specialize(self) -> str:
        """Generate @pl.program source code string.

        Returns:
            Python source string ready to pass to ``pl.parse()``.
        """
        lines: list[str] = [
            "import pypto.language as pl",
            "",
        ]

        # Emit module-level DynVar declarations (deduplicated)
        dv_seen: set[str] = set()
        for ctx in self._contexts:
            for dv_varname in self._iter_dynvar_names(ctx):
                if dv_varname not in dv_seen:
                    dv_seen.add(dv_varname)
                    # Use the original string literal if available, else fall back to var name
                    dv_literal = self._dv_literals.get(dv_varname, dv_varname)
                    lines.append(f'{dv_varname} = pl.dynamic("{dv_literal}")')
        if dv_seen:
            lines.append("")

        # Open @pl.program class
        lines.append("@pl.program")
        lines.append(f"class {self._class_name}:")

        for ctx in self._contexts:
            method_lines = self._specialize_function(ctx)
            for ml in method_lines:
                lines.append("    " + ml)
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_dynvar_names(self, ctx: SpecializeContext) -> list[str]:
        """Return all DynVar Python variable names referenced in ctx."""
        names: list[str] = []
        for param, dim_idx in ctx.dynamic_dims:
            dv = _dynvar_name_for_dim(param, dim_idx, self._dv_bindings)
            if dv not in names:
                names.append(dv)
        return names

    def _specialize_function(self, ctx: SpecializeContext) -> list[str]:
        """Generate lines for a single @pl.function method."""
        # Parse the source to AST
        src = textwrap.dedent(ctx.source)
        tree = ast.parse(src)
        func_def = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == ctx.func_name
        )

        # Classify parameters
        out_params, tensor_params, scalar_dtype_strs = _classify_params(func_def)

        # Collect all param names (excluding self)
        all_param_names = [arg.arg for arg in func_def.args.args if arg.arg != "self"]

        # Build decorator
        decorator = self._build_decorator(ctx, func_def)

        # Build parameter list with updated annotations
        params = self._build_params(all_param_names, out_params, tensor_params, scalar_dtype_strs, ctx)

        # Infer return type
        ret_type = _infer_return_type(
            func_def,
            ctx.tensor_meta,
            ctx.dynamic_dims,
            self._dv_bindings,
            out_params,
        )
        ret_ann = f" -> {ret_type}" if ret_type else ""

        # Transform body
        dep_names = set(ctx.dep_names)
        dynvar_var_names = set(self._iter_dynvar_names(ctx))
        # Collect all names defined anywhere in the function to seed collision avoidance.
        all_defined = {
            node.id
            for node in ast.walk(func_def)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        transformer = _BodyTransformer(
            tensor_meta=ctx.tensor_meta,
            scalar_values=ctx.scalar_values,
            dynamic_dims=ctx.dynamic_dims,
            dynvar_python_names=self._dv_bindings,
            dep_names=dep_names,
            dynvar_var_names=dynvar_var_names,
            param_names=all_param_names,
            initial_used_names=all_defined,
            py_globals=ctx.py_globals,
            dep_param_names=self._dep_param_names,
        )
        new_body = [transformer.visit(stmt) for stmt in func_def.body]
        # Accumulate alias→original renames for error message rewriting.
        # Don't overwrite entries from earlier functions — in multi-function JIT,
        # two functions may independently generate the same alias (e.g. t_v1) for
        # different user variables.  First-seen wins; per-function context is more
        # accurate than a global override.
        for alias, original in transformer.rename_map.items():
            self._rename_map.setdefault(alias, original)
        # Filter out None (deleted statements) and flatten lists
        flat_body: list[ast.stmt] = []
        for item in new_body:
            if item is None:
                continue
            if isinstance(item, list):
                flat_body.extend(item)
            else:
                flat_body.append(item)

        if not flat_body:
            flat_body = [ast.Pass()]

        # Reconstruct a minimal FunctionDef to unparse
        new_func = ast.FunctionDef(
            name=ctx.func_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self")] + [ast.arg(arg=p) for p in all_param_names],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=flat_body,
            decorator_list=[],
            returns=None,
            lineno=1,
            col_offset=0,
        )
        ast.fix_missing_locations(new_func)

        body_src = ast.unparse(new_func)
        # ast.unparse gives us "def func_name(self, ...):\n    ..."
        # We need to insert the decorator and updated signature
        body_lines = body_src.splitlines()

        # Replace the def line with our annotated signature
        sig = f"def {ctx.func_name}(self, {', '.join(params)}){ret_ann}:"
        result_lines = [decorator, sig]
        # body_lines[0] is "def ...:", rest are body
        result_lines.extend(body_lines[1:])

        return result_lines

    def _build_decorator(self, ctx: SpecializeContext, func_def: ast.FunctionDef | None = None) -> str:
        """Build the @pl.function(...) decorator line.

        Entry functions (func_type is None or 'orchestration') are emitted as
        Opaque when they contain ``with pl.incore():`` scopes so that
        OutlineIncoreScopes can outline them and promote the function to
        Orchestration.  Entry functions without InCore scopes (multi-function
        style B) are emitted directly as Orchestration.
        """
        if ctx.func_type is None or ctx.func_type == "orchestration":
            if func_def is not None and _has_incore_scope(func_def):
                return "@pl.function(type=pl.FunctionType.Opaque)"
            return "@pl.function(type=pl.FunctionType.Orchestration)"
        if ctx.func_type == "inline":
            return "@pl.function(type=pl.FunctionType.Inline)"
        if ctx.func_type == "opaque":
            return "@pl.function(type=pl.FunctionType.Opaque)"
        # InCore
        if ctx.level is None:
            return "@pl.function(type=pl.FunctionType.InCore)"
        # Level present
        level_str = _level_to_str(ctx.level)
        return f"@pl.function(type=pl.FunctionType.InCore, level={level_str})"

    def _build_params(
        self,
        all_param_names: list[str],
        out_params: list[str],
        tensor_params: list[str],
        scalar_dtype_strs: dict[str, str],
        ctx: SpecializeContext,
    ) -> list[str]:
        """Build the annotated parameter strings for the function signature."""
        result: list[str] = []
        for name in all_param_names:
            if name in tensor_params:
                is_out = name in out_params
                meta = ctx.tensor_meta.get(name)
                if meta is None:
                    raise ValueError(
                        f"@pl.jit: missing inferred tensor metadata for parameter '{name}'. "
                        "This usually means the tensor's shape/dtype could not be statically "
                        "determined. Pass the tensor directly as a function argument, or "
                        "ensure any intermediate pl.create_tensor() used for this parameter "
                        "has a statically inferable shape and dtype."
                    )
                ann = _build_tensor_annotation(
                    name,
                    meta,
                    ctx.dynamic_dims,
                    self._dv_bindings,
                    is_out=is_out,
                )
                result.append(f"{name}: {ann}")
            elif name in scalar_dtype_strs:
                dtype_s = scalar_dtype_strs[name]
                result.append(f"{name}: pl.Scalar[{dtype_s}]")
            else:
                result.append(name)
        return result


def _level_to_str(level: Any) -> str:
    """Convert a pl.Level enum value to its source string."""
    name = level.name if hasattr(level, "name") else str(level)
    return f"pl.Level.{name}"


# ---------------------------------------------------------------------------
# Top-level entry point used by JITFunction
# ---------------------------------------------------------------------------


def specialize(
    class_name: str,
    contexts: list[SpecializeContext],
    dynvar_bindings: dict[str, str],
    dynvar_literals: dict[str, str] | None = None,
) -> str:
    """Specialize one or more JIT functions into @pl.program source.

    Args:
        class_name: Name for the generated @pl.program class.
        contexts: SpecializeContext per function (deps first, entry last).
        dynvar_bindings: Maps "<param>__<dim_idx>" → DynVar variable name.
        dynvar_literals: Maps DynVar variable name → string literal passed to
            pl.dynamic().  Falls back to variable name when not provided.

    Returns:
        Source code string ready to pass to ``pl.parse()``.
    """
    return Specializer(class_name, contexts, dynvar_bindings, dynvar_literals).specialize()


def build_specialize_context(
    func: Any,
    func_name: str,
    func_type: str | None,
    level: Any,
    tensor_meta: dict[str, TensorMeta],
    scalar_values: dict[str, int | float | bool],
    scalar_dtypes: dict[str, DataType],
    dynamic_dims: set[tuple[str, int]],
    dep_names: list[str],
) -> SpecializeContext:
    """Build a SpecializeContext from a Python function and call-site data.

    Args:
        func: The original Python function object.
        func_name: The function name.
        func_type: 'orchestration', 'incore', or None.
        level: pl.Level enum or None.
        tensor_meta: TensorMeta per tensor param name.
        scalar_values: Concrete scalar values from the call site.
        scalar_dtypes: DataType per scalar param name.
        dynamic_dims: Set of (param_name, dim_idx) dynamic pairs.
        dep_names: Names of @pl.jit.incore functions called from this function.

    Returns:
        SpecializeContext ready for use in Specializer.
    """
    try:
        source = inspect.getsource(func)
    except OSError as e:
        raise OSError(
            f"@pl.jit cannot retrieve source code for '{func.__name__}'. "
            "Source code must be available on disk. "
            "Interactive shells, Jupyter notebooks, and exec/eval-generated "
            f"functions are not supported. (Original error: {e})"
        ) from e
    source = textwrap.dedent(source)

    param_names = [p for p in inspect.signature(func).parameters if p != "self"]

    return SpecializeContext(
        func_name=func_name,
        source=source,
        func_type=func_type,
        level=level,
        param_names=param_names,
        tensor_meta=tensor_meta,
        scalar_values=scalar_values,
        scalar_dtypes=scalar_dtypes,
        dynamic_dims=dynamic_dims,
        dep_names=dep_names,
        py_globals=getattr(func, "__globals__", {}),
    )


__all__ = [
    "SpecializeContext",
    "TensorMeta",
    "Specializer",
    "build_specialize_context",
    "specialize",
]
