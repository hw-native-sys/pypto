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
a: pl.Tensor[[M, 128], pl.FP32]   a: pl.Tensor[[M, 128], pl.FP32]  (M kept dynamic)
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
import functools
import inspect
import textwrap
import warnings
from dataclasses import dataclass, field
from typing import Any, cast

from pypto.pypto_core import DataType

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DynDim:
    """A shape dim bound to a DynVar at JIT specialization time.

    Attributes:
        name:        Python variable name used in user source (e.g. ``"M"``).
        literal:     String passed to ``pl.dynamic("...")``; usually equals
                     ``name`` but may diverge (e.g. ``rows = pl.dynamic("M")``).
        static_bound: Concrete extent at this specialization. Replaced by
                     ``None`` in the cache key so different bounds reuse the
                     same compilation, but still available as the static-shape
                     fallback wherever a numeric dim is required (e.g.
                     ``pl.slice`` parent-dim inheritance, ``ir.compile``).
    """

    name: str
    literal: str
    static_bound: int


ShapeDim = int | DynDim


@dataclass
class TensorMeta:
    """Runtime metadata for a single tensor parameter.

    Attributes:
        shape: Per-dim entries. Each entry is either a static ``int`` or a
            :class:`DynDim` capturing a JIT-time-bound dynamic dim.
        dtype: DataType resolved from the torch tensor.
    """

    shape: tuple[ShapeDim, ...]
    dtype: DataType

    def static_shape(self) -> tuple[int, ...]:
        """Concrete shape tuple — DynDim dims collapse to their static_bound."""
        return tuple(d.static_bound if isinstance(d, DynDim) else d for d in self.shape)

    def dynamic_dim_indices(self) -> set[int]:
        """Indices of dims that are DynDim-bound at this specialization."""
        return {i for i, d in enumerate(self.shape) if isinstance(d, DynDim)}


@dataclass
class SpecializeContext:
    """All information needed to specialize a single JIT function.

    Dynamic dims live as :class:`DynDim` entries inside each meta's
    ``shape`` tuple — the legacy ``(param, dim_idx)`` view is derived
    on demand via the :attr:`dynamic_dims` property.

    Attributes:
        func_name: Python function name.
        source: Dedented source code of the function.
        func_type: 'orchestration' | 'incore' | 'inline' | 'opaque' | None (auto).
        level: pl.Level value or None.
        param_names: Ordered parameter names (excluding 'self').
        tensor_meta: TensorMeta per tensor param name.
        scalar_values: Concrete value per scalar param name.
        scalar_dtypes: DataType annotation per scalar param name.
        dep_names: Names of dep functions called from this function.
        py_globals: The originating function's ``__globals__``. The specializer
            uses this to resolve module-level int/float/bool constants (e.g.
            ``BATCH``, ``HIDDEN`` imported from a config module) by inlining
            them at the use site.
        orig_file: Path to the user's real source file (``inspect.getsourcefile``),
            or ``None`` when the function has no on-disk source (REPL / exec).
            Used to map generated diagnostics back to the user's ``.py`` (#1612).
        orig_start_line: 1-based file line of the function's first source line
            (its first decorator), i.e. the line ``inspect.getsourcelines``
            starts at. Anchors the generated→original line remap.
        orig_col_offset: Indentation (in columns) stripped by ``textwrap.dedent``
            from the original source — added back to recover original columns.
    """

    func_name: str
    source: str
    func_type: str | None
    level: Any
    param_names: list[str]
    tensor_meta: dict[str, TensorMeta]
    scalar_values: dict[str, int | float | bool]
    scalar_dtypes: dict[str, DataType]
    dep_names: list[str] = field(default_factory=list)
    py_globals: dict[str, Any] = field(default_factory=dict)
    orig_file: str | None = None
    orig_start_line: int = 1
    orig_col_offset: int = 0

    @property
    def dynamic_dims(self) -> set[tuple[str, int]]:
        """The legacy ``(param_name, dim_idx)`` set, derived from tensor_meta."""
        return {(p, i) for p, m in self.tensor_meta.items() for i in m.dynamic_dim_indices()}

    def dynvar_for(self, param: str, dim_idx: int) -> str | None:
        """Return the DynVar variable name bound to ``param``'s ``dim_idx``, or None."""
        meta = self.tensor_meta.get(param)
        if meta is None or dim_idx >= len(meta.shape):
            return None
        d = meta.shape[dim_idx]
        return d.name if isinstance(d, DynDim) else None

    def dynvar_literals(self) -> dict[str, str]:
        """Map ``dynvar_name → pl.dynamic("...") literal`` across all metas.

        Used by :class:`Specializer` to emit module-level
        ``M = pl.dynamic("M")`` declarations. A given Python name should not
        be rebound to a different literal within a function — if two metas
        disagree, the last one wins. (``pl.dynamic()`` itself is not
        singleton-cached; the constraint is on user source, not on the
        runtime object identity.)
        """
        out: dict[str, str] = {}
        for meta in self.tensor_meta.values():
            for d in meta.shape:
                if isinstance(d, DynDim):
                    out[d.name] = d.literal
        return out


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


def _collect_annotation_dynamic_dims(
    func: Any,
    param_names: set[str],
) -> tuple[set[tuple[str, int]], dict[str, str], dict[str, str]]:
    """Scan a JIT function's parameter annotations for ``pl.dynamic()`` dims.

    This makes ``@pl.jit`` honour the same dynamic-shape contract as
    ``@pl.program``: a :class:`~pypto.language.typing.dynamic.DynVar` used
    directly in a tensor annotation (e.g. ``pl.Tensor[[M, 128], pl.FP32]``)
    marks that dimension runtime-dynamic, with no ``bind_dynamic`` call
    required.  It complements :func:`_collect_dynamic_dims`; callers take the
    union of both sources.

    Args:
        func: The JIT-decorated Python function.
        param_names: Parameter names to consider (excludes ``self``).

    Returns:
        ``(dims, bindings, literals)`` where:

        - ``dims``: ``{(param_name, dim_idx)}`` marked dynamic via annotation.
        - ``bindings``: ``{"<param>__<dim_idx>": dynvar_var_name}``.
        - ``literals``: ``{dynvar_var_name: pl.dynamic() string literal}``.

        ``DynVar.name`` is a validated identifier, so it doubles as both the
        generated module-level variable name and the ``pl.dynamic("...")``
        literal.
    """
    dims, bindings, literals = _collect_annotation_dynamic_dims_cached(func, frozenset(param_names))
    # Return copies so callers can freely mutate without corrupting the cache.
    return set(dims), dict(bindings), dict(literals)


@functools.lru_cache(maxsize=512)
def _collect_annotation_dynamic_dims_cached(
    func: Any,
    param_names: frozenset[str],
) -> tuple[set[tuple[str, int]], dict[str, str], dict[str, str]]:
    """Cached core of :func:`_collect_annotation_dynamic_dims` (hot path: runs
    per JIT call). A function object's signature is fixed, so memoize on it."""
    from pypto.language.typing.dynamic import DynVar  # noqa: PLC0415
    from pypto.language.typing.tensor import Tensor  # noqa: PLC0415

    dims: set[tuple[str, int]] = set()
    bindings: dict[str, str] = {}
    literals: dict[str, str] = {}

    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return dims, bindings, literals

    # When the user's module uses ``from __future__ import annotations`` the
    # annotations arrive as strings; evaluate them via get_type_hints so the
    # Tensor[...] subscripts resolve to real objects carrying the DynVars.
    resolved_hints: dict[str, Any] | None = None
    if any(isinstance(p.annotation, str) for p in sig.parameters.values()):
        try:
            import typing  # noqa: PLC0415

            resolved_hints = typing.get_type_hints(func)
        except Exception:  # noqa: BLE001 - best effort; fall back to raw annotations
            resolved_hints = None

    for name, param in sig.parameters.items():
        if name not in param_names:
            continue
        annotation = param.annotation
        if resolved_hints is not None and name in resolved_hints:
            annotation = resolved_hints[name]
        if not isinstance(annotation, Tensor):
            continue
        shape = annotation.shape
        if shape is None:
            continue
        for dim_idx, dim in enumerate(shape):
            if isinstance(dim, DynVar):
                dims.add((name, dim_idx))
                bindings[f"{name}__{dim_idx}"] = dim.name
                literals[dim.name] = dim.name
    return dims, bindings, literals


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


def _build_tensor_annotation(meta: TensorMeta, is_out: bool, is_distributed: bool = False) -> str:
    """Build the type annotation string for a tensor parameter.

    Static dims emit as integer literals; DynDim entries emit as their
    DynVar variable name. When ``is_distributed`` is True the head type
    becomes ``pld.DistributedTensor`` instead of ``pl.Tensor`` — same
    shape/dtype subscript form, only the IR ObjectKind differs (see
    ``pld.DistributedTensor``).

    Returns:
        Annotation string such as ``pl.Tensor[[M, 128], pl.FP32]``,
        ``pl.Out[pl.Tensor[[128, 128], pl.FP32]]``, or
        ``pld.DistributedTensor[[256], pl.INT8]``.
    """
    dims = [d.name if isinstance(d, DynDim) else str(d) for d in meta.shape]
    head = "pld.DistributedTensor" if is_distributed else "pl.Tensor"
    inner = f"{head}[[{', '.join(dims)}], {_dtype_str(meta.dtype)}]"
    return f"pl.Out[{inner}]" if is_out else inner


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
        dep_names: set[str],
        param_names: list[str] | None = None,
        initial_used_names: set[str] | None = None,
        py_globals: dict[str, Any] | None = None,
        dep_param_names: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__()
        self._meta = tensor_meta
        self._scalars = scalar_values
        self._dep_names = dep_names
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
        elts: list[ast.expr] = []
        for i, d in enumerate(meta.shape):
            if isinstance(d, DynDim):
                elts.append(self._dyn_dim_expr(param_name, i))
            else:
                elts.append(ast.Constant(value=d))
        return ast.Tuple(elts=elts, ctx=ast.Load())

    def _shape_dim_node(self, param_name: str, dim_idx: int) -> ast.expr:
        d = self._meta[param_name].shape[dim_idx]
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
                for i, (tgt, dim) in enumerate(zip(target_names, meta.shape, strict=True)):
                    if not isinstance(tgt, ast.Name):
                        continue
                    if isinstance(dim, DynDim):
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
        DynVar runtime references, and module-level int/float/bool constants from globals."""
        if isinstance(node.ctx, ast.Load):
            # Check active renames first — a rebinding supersedes any earlier inlining.
            if node.id in self._var_renames:
                return ast.Name(id=self._var_renames[node.id], ctx=ast.Load())
            if node.id in self._scalars:
                return ast.Constant(value=self._scalars[node.id])
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


# ---------------------------------------------------------------------------
# Return-type inference (Option A: from Out params)
# ---------------------------------------------------------------------------


def _infer_return_type(
    func_def: ast.FunctionDef,
    tensor_meta: dict[str, TensorMeta],
    out_params: list[str],
    distributed_params: set[str] | None = None,
) -> str | None:
    """Infer the return type annotation string from the return statement.

    Examines the function's return statement:
    - ``return a, b, ...``  -> ``tuple[T_a, T_b, ...]`` when every element is a
      tensor-typed name we can resolve via ``tensor_meta`` (covers both ``pl.Out``
      and bare ``pl.Tensor`` params).
    - ``return name``       -> that name's tensor type (without ``Out[]``) when
      ``name`` is a tensor-typed param.
    - ``return f(...)``     -> ``None`` (return type depends on f, which we
      can't resolve here).
    - No tensor-typed params -> ``None``.

    Falls back to the first ``Out`` param's tensor type as a last resort —
    matches the legacy single-return convention before bare ``pl.Tensor``
    inline params were supported.

    ``distributed_params`` carries the subset of returnable names whose head
    type is ``pld.DistributedTensor`` rather than ``pl.Tensor`` — propagated
    here so a function returning a window-bound view does not leak as plain
    ``pl.Tensor`` (the two kinds have distinct IR ``ObjectKind``).
    """
    dist_set = distributed_params or set()

    # Find the first return-with-value
    return_node: ast.Return | None = None
    for node in ast.walk(func_def):
        if isinstance(node, ast.Return) and node.value is not None:
            return_node = node
            break

    # Multi-return: `return a, b, ...` -> emit `tuple[T_a, T_b, ...]`
    if return_node is not None and isinstance(return_node.value, ast.Tuple):
        elt_annotations: list[str] = []
        for elt in return_node.value.elts:
            if not isinstance(elt, ast.Name) or elt.id not in tensor_meta:
                return None  # Can't infer this element — drop the annotation
            elt_annotations.append(
                _build_tensor_annotation(tensor_meta[elt.id], is_out=False, is_distributed=elt.id in dist_set)
            )
        return f"tuple[{', '.join(elt_annotations)}]"

    # `return f(...)`: the result type depends on f's declared return types,
    # which we don't have here. Emit no annotation rather than wrongly assuming
    # `out_params[0]` — the caller may be a multi-return function.
    if return_node is not None and isinstance(return_node.value, ast.Call):
        return None

    # `return name`: use that name's tensor type if known.
    if (
        return_node is not None
        and isinstance(return_node.value, ast.Name)
        and return_node.value.id in tensor_meta
    ):
        name = return_node.value.id
        return _build_tensor_annotation(tensor_meta[name], is_out=False, is_distributed=name in dist_set)

    # Fallback: first Out param's tensor type (legacy single-return convention).
    if out_params and out_params[0] in tensor_meta:
        name = out_params[0]
        return _build_tensor_annotation(tensor_meta[name], is_out=False, is_distributed=name in dist_set)

    return None


# ---------------------------------------------------------------------------
# Annotation classifier: detect Out[], plain Tensor, Scalar params
# ---------------------------------------------------------------------------


def _classify_params(
    func_def: ast.FunctionDef,
) -> tuple[list[str], list[str], dict[str, str], set[str]]:
    """Classify function parameters.

    Returns:
        (out_params, tensor_params, scalar_dtype_strs, distributed_params)
        - out_params: names annotated Out[pl.Tensor] / Out[pld.DistributedTensor]
        - tensor_params: all names annotated as tensor-like — pl.Tensor or
          pld.DistributedTensor (including Out ones)
        - scalar_dtype_strs: param_name → dtype string for scalar params
        - distributed_params: subset of tensor_params whose annotation uses
          ``DistributedTensor`` (and should round-trip as
          ``pld.DistributedTensor[...]`` in the generated @pl.program source)
    """
    out_params: list[str] = []
    tensor_params: list[str] = []
    scalar_dtype_strs: dict[str, str] = {}
    distributed_params: set[str] = set()

    for arg in func_def.args.args:
        name = arg.arg
        if name == "self":
            continue
        ann = arg.annotation
        if ann is None:
            continue

        # Detect Out[pl.Tensor] / Out[pld.DistributedTensor] etc.
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
                if _is_distributed_tensor_annotation(inner):
                    distributed_params.add(name)
                continue
            # Check for Scalar dtype annotations: pl.INDEX, pl.FP32, etc.
            is_scalar_ann = _is_scalar_annotation(outer)
            if is_scalar_ann:
                scalar_dtype_strs[name] = _ast_to_str(inner)
                continue

        # Detect pl.Tensor / pld.DistributedTensor (bare or subscripted)
        if _is_tensor_annotation(ann):
            tensor_params.append(name)
            if _is_distributed_tensor_annotation(ann):
                distributed_params.add(name)
            continue

        # Detect bare dtype annotation: pl.INDEX, pl.FP32, etc. (no Scalar wrapper)
        dtype_str = _extract_bare_dtype(ann)
        if dtype_str is not None:
            scalar_dtype_strs[name] = dtype_str

    return out_params, tensor_params, scalar_dtype_strs, distributed_params


def _is_tensor_annotation(node: ast.expr) -> bool:
    """Return True if the AST node represents a tensor-like annotation.

    Matches both ``pl.Tensor`` / ``Tensor`` and ``pld.DistributedTensor`` /
    ``DistributedTensor`` (bare or subscripted). ``DistributedTensor`` shares
    the same ``[shape, dtype, ...]`` subscript form as ``Tensor`` — only the
    IR ObjectKind differs — so treating both as tensor-like at the classifier
    level lets the rest of the specializer reuse the same plumbing. Call
    :func:`_is_distributed_tensor_annotation` to distinguish the two kinds.
    """
    if isinstance(node, ast.Name):
        return node.id in ("Tensor", "DistributedTensor")
    if isinstance(node, ast.Attribute):
        return node.attr in ("Tensor", "DistributedTensor")
    if isinstance(node, ast.Subscript):
        return _is_tensor_annotation(node.value)
    return False


def _is_distributed_tensor_annotation(node: ast.expr) -> bool:
    """Return True iff the annotation specifically names DistributedTensor."""
    if isinstance(node, ast.Name):
        return node.id == "DistributedTensor"
    if isinstance(node, ast.Attribute):
        return node.attr == "DistributedTensor"
    if isinstance(node, ast.Subscript):
        return _is_distributed_tensor_annotation(node.value)
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


def _dfs_stmts(node: ast.AST) -> list[ast.stmt]:
    """Statement descendants of ``node`` in pre-order (DFS).

    Used to correlate the transformed AST with the reparsed generated AST when
    building the source map (#1612): both share statement count and order, so
    zipping their pre-order statement lists pairs each generated statement with
    its originating user statement.
    """
    result: list[ast.stmt] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.stmt):
            result.append(child)
            result.extend(_dfs_stmts(child))
    return result


class Specializer:
    """Transform a collection of SpecializeContext objects into @pl.program source.

    Usage::

        s = Specializer(class_name, contexts)
        source_code = s.specialize()
        program = pl.parse(source_code)

    DynVar names and ``pl.dynamic("...")`` literals are read directly from
    each context's :class:`TensorMeta`-embedded :class:`DynDim` entries (see
    :meth:`SpecializeContext.dynvar_for` and
    :meth:`SpecializeContext.dynvar_literals`).
    """

    def __init__(
        self,
        class_name: str,
        contexts: list[SpecializeContext],
    ) -> None:
        """
        Args:
            class_name: Name for the generated @pl.program class.
            contexts: List of SpecializeContext, one per function.
                The entry-point (Orchestration) function must be last.
        """
        self._class_name = class_name
        self._contexts = contexts
        # Accumulated alias→original map across all specialized functions.
        self._rename_map: dict[str, str] = {}
        # Param order per function — lets the body transformer rewrite
        # ``self.<dep>(a, out=out)`` keyword args into all-positional form
        # (the parser rejects keyword args on inter-function calls).
        self._dep_param_names: dict[str, list[str]] = {
            ctx.func_name: list(ctx.param_names) for ctx in contexts
        }
        # generated-program absolute line → (orig_file, orig_line, orig_col),
        # built by specialize() so diagnostics map back to the user's .py (#1612).
        self._source_map: dict[int, tuple[str, int, int]] = {}
        # (ctx, original (lineno, col_offset) per transformed statement in DFS
        # order) per function — captured BEFORE ast.fix_missing_locations so
        # synthesized statements read as (0, 0), and zipped against the reparsed
        # generated statements in _build_source_map after assembly.
        self._fn_origin: list[tuple[SpecializeContext, list[tuple[int, int]]]] = []
        # Set by _build_params when any param uses pld.DistributedTensor; the
        # corresponding ``import pypto.language.distributed as pld`` is then
        # emitted in the generated module preamble so the source is
        # self-contained (mirrors the ``import pypto.language as pl`` line).
        self._needs_pld_import: bool = False

    @property
    def rename_map(self) -> dict[str, str]:
        """Return mapping from generated alias → original user variable name.

        Only populated after ``specialize()`` has been called.
        """
        return dict(self._rename_map)

    @property
    def source_map(self) -> dict[int, tuple[str, int, int]]:
        """Map generated-program line → ``(orig_file, orig_line, orig_col)``.

        Statement-granular; only entries for functions with on-disk source and
        a clean structural match appear. Empty until ``specialize()`` runs.
        Passed to ``pl.parse(..., source_map=...)`` so IR spans (and thus parse
        and compile error diagnostics) point at the user's real source (#1612).
        """
        return dict(self._source_map)

    def specialize(self) -> str:
        """Generate @pl.program source code string.

        Returns:
            Python source string ready to pass to ``pl.parse()``.
        """
        self._fn_origin = []
        self._needs_pld_import = False

        # Specialize bodies first so per-context state (DynVars,
        # _needs_pld_import) is fully populated before assembling the preamble.
        body_lines: list[str] = []
        for ctx in self._contexts:
            method_lines = self._specialize_function(ctx)
            for ml in method_lines:
                body_lines.append("    " + ml)
            body_lines.append("")

        lines: list[str] = ["import pypto.language as pl"]
        if self._needs_pld_import:
            lines.append("import pypto.language.distributed as pld")
        lines.append("")

        # Emit module-level DynVar declarations (deduplicated across contexts)
        dv_seen: set[str] = set()
        for ctx in self._contexts:
            for dv_name, dv_literal in ctx.dynvar_literals().items():
                if dv_name not in dv_seen:
                    dv_seen.add(dv_name)
                    lines.append(f'{dv_name} = pl.dynamic("{dv_literal}")')
        if dv_seen:
            lines.append("")

        # Open @pl.program class
        lines.append("@pl.program")
        lines.append(f"class {self._class_name}:")
        lines.extend(body_lines)

        src = "\n".join(lines)
        self._source_map = self._build_source_map(src)
        return src

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_source_map(self, src: str) -> dict[int, tuple[str, int, int]]:
        """Map each generated-program line to the user's original source location.

        Reparses the assembled program to recover absolute line numbers (which
        equal the spans ``SpanTracker`` will emit), then zips each function's
        reparsed statements against the original (dedented) positions captured
        per statement in :meth:`_specialize_function`. Per-statement granularity:
        a multi-line user statement collapses to its start line; synthesized
        statements (captured lineno 0) are left unmapped so their spans keep the
        generated coordinates. See issue #1612.
        """
        out: dict[int, tuple[str, int, int]] = {}
        try:
            prog = ast.parse(src)
        except SyntaxError:
            # Malformed generated source is a specializer bug; let pl.parse
            # surface it with full diagnostics rather than crashing here.
            return out
        classdef = next((n for n in prog.body if isinstance(n, ast.ClassDef)), None)
        if classdef is None:
            return out
        gen_funcs = {n.name: n for n in classdef.body if isinstance(n, ast.FunctionDef)}
        for ctx, orig_positions in self._fn_origin:
            gen_func = gen_funcs.get(ctx.func_name)
            if ctx.orig_file is None or gen_func is None:
                continue
            gen_stmts = _dfs_stmts(gen_func)
            # Reparsing the unparsed transform must preserve statement count and
            # DFS order; if it somehow diverges, skip this function rather than
            # emit a wrong mapping (its spans then fall back to generated coords).
            if len(gen_stmts) != len(orig_positions):
                continue
            for gen_stmt, (orig_line, orig_col) in zip(gen_stmts, orig_positions, strict=True):
                if orig_line <= 0:
                    continue  # synthesized statement — no original location
                out[gen_stmt.lineno] = (
                    ctx.orig_file,
                    orig_line + ctx.orig_start_line - 1,
                    orig_col + ctx.orig_col_offset,
                )
        return out

    def _specialize_function(self, ctx: SpecializeContext) -> list[str]:
        """Generate lines for a single @pl.function method."""
        # Parse the source to AST
        src = textwrap.dedent(ctx.source)
        tree = ast.parse(src)
        func_def = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == ctx.func_name
        )

        # Classify parameters
        out_params, tensor_params, scalar_dtype_strs, distributed_params = _classify_params(func_def)

        # Inline helpers are spliced at the call site before SSA conversion,
        # so their parameters are already in-place aliases of the caller's
        # variables — `pl.Out[...]` is redundant ceremony there. Warn the user
        # so they migrate to bare `pl.Tensor[...]`, and drop the wrapper from
        # the generated source so downstream passes see the simpler form.
        is_inline = ctx.func_type == "inline"
        if is_inline and out_params:
            warnings.warn(
                f"@pl.jit.inline helper '{ctx.func_name}' uses pl.Out[...] on "
                f"parameter(s) {out_params!r}. pl.Out annotations are deprecated "
                f"for inline helpers because the body is spliced at the call "
                f"site before SSA conversion — the parameter is already an "
                f"in-place alias of the caller's variable. Drop the pl.Out "
                f"wrapper; bare pl.Tensor[...] works the same.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Collect all param names (excluding self)
        all_param_names = [arg.arg for arg in func_def.args.args if arg.arg != "self"]

        # Build decorator
        decorator = self._build_decorator(ctx, func_def)

        params = self._build_params(
            all_param_names,
            out_params,
            tensor_params,
            scalar_dtype_strs,
            distributed_params,
            ctx,
            is_inline=is_inline,
        )

        # Infer return type
        ret_type = _infer_return_type(func_def, ctx.tensor_meta, out_params, distributed_params)
        ret_ann = f" -> {ret_type}" if ret_type else ""

        # Transform body
        dep_names = set(ctx.dep_names)
        # Collect all names defined anywhere in the function to seed collision avoidance.
        all_defined = {
            node.id
            for node in ast.walk(func_def)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        transformer = _BodyTransformer(
            tensor_meta=ctx.tensor_meta,
            scalar_values=ctx.scalar_values,
            dep_names=dep_names,
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
        # Capture each statement's original (dedented) position in DFS order
        # BEFORE fix_missing_locations runs (#1612). Surviving/expanded statements
        # carry their original linenos; synthesized statements (loop bridges, the
        # empty-body `pass`, transformer-built nodes) have no lineno yet and read
        # as (0, 0) — fix_missing_locations would otherwise backfill them with the
        # function's lineno=1, which would mis-map them to the decorator line.
        # specialize() zips this against the reassembled program's statements.
        orig_positions = [
            (getattr(s, "lineno", 0), getattr(s, "col_offset", 0)) for s in _dfs_stmts(new_func)
        ]
        self._fn_origin.append((ctx, orig_positions))
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
        distributed_params: set[str],
        ctx: SpecializeContext,
        is_inline: bool = False,
    ) -> list[str]:
        """Build the annotated parameter strings for the function signature.

        When ``is_inline`` is True, ``pl.Out[...]`` wrappers are stripped from
        tensor params — inline helpers don't have a calling convention boundary,
        so the direction tag carries no information.

        Params listed in ``distributed_params`` round-trip as
        ``pld.DistributedTensor[...]`` and trigger the corresponding import in
        the generated module preamble.
        """
        result: list[str] = []
        for name in all_param_names:
            if name in tensor_params:
                is_out = (name in out_params) and not is_inline
                is_distributed = name in distributed_params
                if is_distributed:
                    self._needs_pld_import = True
                meta = ctx.tensor_meta.get(name)
                if meta is None:
                    raise ValueError(
                        f"@pl.jit: missing inferred tensor metadata for parameter '{name}'. "
                        "This usually means the tensor's shape/dtype could not be statically "
                        "determined. Pass the tensor directly as a function argument, or "
                        "ensure any intermediate pl.create_tensor() used for this parameter "
                        "has a statically inferable shape and dtype."
                    )
                ann = _build_tensor_annotation(meta, is_out=is_out, is_distributed=is_distributed)
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


def specialize(class_name: str, contexts: list[SpecializeContext]) -> str:
    """Specialize one or more JIT functions into @pl.program source.

    Args:
        class_name: Name for the generated @pl.program class.
        contexts: SpecializeContext per function (deps first, entry last).

    Returns:
        Source code string ready to pass to ``pl.parse()``.
    """
    return Specializer(class_name, contexts).specialize()


def build_specialize_context(
    func: Any,
    func_name: str,
    func_type: str | None,
    level: Any,
    tensor_meta: dict[str, TensorMeta],
    scalar_values: dict[str, int | float | bool],
    scalar_dtypes: dict[str, DataType],
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
        dep_names: Names of @pl.jit.incore functions called from this function.

    Dynamic dims live inside ``tensor_meta`` as :class:`DynDim` entries —
    no separate set is passed in.

    Returns:
        SpecializeContext ready for use in Specializer.
    """
    try:
        raw_lines, orig_start_line = inspect.getsourcelines(func)
    except OSError as e:
        raise OSError(
            f"@pl.jit cannot retrieve source code for '{func.__name__}'. "
            "Source code must be available on disk. "
            "Interactive shells, Jupyter notebooks, and exec/eval-generated "
            f"functions are not supported. (Original error: {e})"
        ) from e
    raw_source = "".join(raw_lines)
    source = textwrap.dedent(raw_source)

    # Real-source anchors for diagnostic provenance (#1612). The generated
    # source is re-derived via ast.unparse, so spans default to the synthesized
    # text; these let the specializer map statements back to the user's .py.
    src_file = inspect.getsourcefile(func)
    orig_file = src_file if (src_file and not src_file.startswith("<")) else None
    orig_col_offset = (len(raw_lines[0]) - len(raw_lines[0].lstrip())) if raw_lines else 0

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
        dep_names=dep_names,
        py_globals=getattr(func, "__globals__", {}),
        orig_file=orig_file,
        orig_start_line=orig_start_line,
        orig_col_offset=orig_col_offset,
    )


__all__ = [
    "DynDim",
    "ShapeDim",
    "SpecializeContext",
    "TensorMeta",
    "Specializer",
    "build_specialize_context",
    "specialize",
]
