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
@pl.jit                            @pl.function(type=pl.FunctionType.Orchestration)
@pl.jit.host                       @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
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
import copy
import functools
import importlib
import inspect
import keyword
import textwrap
import types
import unicodedata
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

from pypto._external_source import EXTERNAL_INCLUDE_DIRS_ATTR, encode_external_include_dirs
from pypto._function_attrs import DUAL_AIV_DISPATCH_ATTR
from pypto.language.typing.array import Array as _LangArray
from pypto.language.typing.dynamic import DynVar as _LangDynVar
from pypto.language.typing.scalar import Scalar as _LangScalar
from pypto.language.typing.tensor import Tensor as _LangTensor
from pypto.language.typing.tile import Tile as _LangTile
from pypto.language.typing.tuple import Tuple as _LangTuple
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir
from pypto.pypto_core.ir import TensorLayout

from . import _specialize_values as values
from ._source import _UNBOUND, cache_in_snapshot, function_namespace
from ._specialize_body import BodyTransformer
from ._specialize_types import DynDim, ShapeDim, TensorMeta

# Preserve existing imports while source emission and cache hashing share the
# exact renderer objects defined in the private values module.
constant_expr = values.constant_expr
constant_source = values.constant_source
free_name_source = values.free_name_source

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TileMetadataOverride:
    """One original metadata expression, its cache spelling and proven eager kind."""

    kind: str
    source: str
    canonical_source: str
    path: tuple[int, ...] = ()


@dataclass
class _EagerTileAnnotations:
    """Request-local annotation templates shared by specialization and cache hashing."""

    sources: dict[str, str] = field(default_factory=dict[str, str])
    dynvars: dict[str, str] = field(default_factory=dict[str, str])
    fingerprints: dict[str, str] = field(default_factory=dict[str, str])
    overrides: dict[str, tuple[_TileMetadataOverride, ...]] = field(
        default_factory=dict[str, tuple[_TileMetadataOverride, ...]]
    )
    parameter_placeholders: dict[str, dict[str, str]] = field(default_factory=dict[str, dict[str, str]])


@dataclass
class SpecializeContext:
    """All information needed to specialize a single JIT function.

    Dynamic dims live as :class:`DynDim` entries inside each meta's
    ``shape`` tuple — the legacy ``(param, dim_idx)`` view is derived
    on demand via the :attr:`dynamic_dims` property.

    Attributes:
        func_name: Name of the generated ``@pl.function`` method. Normally the
            Python function's ``__name__``, but the JIT layer uniquifies it
            when two distinct deps share a name (see ``source_func_name``).
        source: Dedented source code of the function.
        func_type: 'orchestration' | 'incore' | 'inline' | 'opaque' | 'graph' | None (auto).
        level: pl.Level value or None.
        param_names: Ordered parameter names (excluding 'self').
        tensor_meta: TensorMeta per tensor param name.
        scalar_dtypes: DataType annotation per scalar param name. A scalar
            parameter is a runtime value: it survives into the generated
            program as a real ``pl.Scalar`` parameter and is never folded, so
            only its type is carried here.
        constexpr_values: ``pl.constexpr`` param name → the generated-source
            text its call-site value folds to. These are the opposite of a
            scalar: every use is replaced by that text and the parameter is
            dropped from the generated signature, so it reaches neither the IR
            nor the dispatch ABI.
        dep_names: Names this function's source calls its deps by. Under an
            aliased import (``from mod import kernel as kern``) that is the
            alias, not the callee's ``__name__`` — see ``dep_func_names``.
        dep_func_names: ``call name -> generated function name``, for every
            call name that differs from the function it resolves to. The body
            transformer consults it when rewriting ``kern(...)`` into
            ``self.kernel(...)``; an absent entry means the two agree.
        dep_call_variants: ``(call name, ordinal) -> generated function name``,
            consulted before ``dep_func_names`` so two call sites of one name
            can reach different generated functions. See the field comment.
        py_globals: A snapshot of the originating function's globals and closure
            bindings. During JIT compilation this is the same namespace used
            for cache-key construction and annotation resolution. The specializer
            inlines referenced int/float/bool constants at their use sites.
        eager_tile_annotations: Read-only request snapshot of eager inline Tile
            sources, deferred metadata and fresh parameter names. Independent
            factory values come from Python's evaluated annotations; call-owned
            expressions are substituted separately for each specialization.
        annotation_dynvars: Scalar symbols used by Tile annotation shapes or
            metadata. These need generated declarations independently of any
            Tensor parameter's dynamic-dimension bindings.
        orig_file: Path to the user's real source file (``inspect.getsourcefile``),
            or ``None`` when the function has no on-disk source (REPL / exec).
            Used to map generated diagnostics back to the user's ``.py`` (#1612).
        orig_start_line: 1-based file line of the function's first source line
            (its first decorator), i.e. the line ``inspect.getsourcelines``
            starts at. Anchors the generated→original line remap.
        orig_col_offset: Indentation (in columns) stripped by ``textwrap.dedent``
            from the original source — added back to recover original columns.
        auto_scope: Whether the compiler auto-inserts AUTO runtime scopes
            (SIMPLER_SCOPE). ``True`` by default; ``False`` emits
            ``@pl.function(..., auto_scope=False)`` so the body places scopes
            by hand. Honored for the Orchestration entry, HOST orchestrator,
            and inline sub-function decorators (see
            :meth:`Specializer._build_decorator`).
    """

    func_name: str
    source: str
    func_type: str | None
    level: Any
    param_names: list[str]
    tensor_meta: dict[str, TensorMeta]
    scalar_dtypes: dict[str, DataType]
    dep_names: list[str] = field(default_factory=list[str])
    py_globals: Mapping[str, Any] = field(default_factory=dict[str, Any])
    orig_file: str | None = None
    orig_start_line: int = 1
    orig_col_offset: int = 0
    # Appended at the tail to preserve positional construction of this exported
    # dataclass for external callers (auto_scope is keyword-only in practice).
    auto_scope: bool = True
    # External C++ kernel backing (func_type == "extern"). core_type is
    # "aic" | "aiv" (single) or "mixed" (AIC+AIV pair dispatched as one
    # MixedKernels submit). The specializer renders the corresponding
    # ``@pl.function(external_source=...)`` declaration(s), plus a Group wrapper
    # for "mixed". Paths are absolute .cpp files.
    external_core_type: str | None = None
    external_aic_source: str | None = None
    external_aiv_source: str | None = None
    external_dual_aiv_dispatch: bool = False
    external_include_dirs: tuple[str, ...] = ()
    # Also appended at the tail (see above): ``call name -> generated function
    # name`` for the deps this function reaches under a different name.
    dep_func_names: dict[str, str] = field(default_factory=dict[str, str])
    # Name of the ``def`` inside ``source`` — the Python ``__name__``. It differs
    # from ``func_name`` whenever the JIT layer had to uniquify the generated
    # name (two deps sharing a ``__name__``). Anything that looks the definition
    # up in the *original* source must use ``source_def_name``; anything naming
    # the *generated* function uses ``func_name``.
    source_func_name: str | None = None
    # Also appended at the tail (see above): ``pl.constexpr`` param name -> the
    # generated-source text its call-site value folds to.
    constexpr_values: dict[str, str] = field(default_factory=dict[str, str])
    # Also appended at the tail (see above): ``(call name, ordinal) -> generated
    # function name``, where the ordinal is the call's position among
    # same-named calls in source order. A dep called at two different
    # ``pl.constexpr`` values is compiled once per value, so the two call sites
    # resolve to different generated functions and the call name alone no
    # longer identifies the callee. Empty when every call site of a name
    # reaches the same function, which ``dep_func_names`` already covers.
    dep_call_variants: dict[tuple[str, int], str] = field(default_factory=dict[tuple[str, int], str])
    # Appended for positional compatibility. These sources share the request's
    # annotation snapshot with the cache key; parameter-dependent components
    # remain expressions so call bindings and local dimensions take precedence.
    eager_tile_annotations: _EagerTileAnnotations = field(default_factory=_EagerTileAnnotations)
    annotation_dynvars: dict[str, str] = field(default_factory=dict[str, str])

    @property
    def source_def_name(self) -> str:
        """Name of the ``def`` to find in :attr:`source` (the Python name)."""
        return self.source_func_name or self.func_name

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
        out: dict[str, str] = dict(self.annotation_dynvars)
        for meta in self.tensor_meta.values():
            for d in meta.shape:
                if isinstance(d, DynDim):
                    out[d.name] = d.literal
        return out


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------


def func_name_lookup(func: Any) -> Mapping[str, Any]:
    """Return ``func.__globals__`` merged with closure free-var bindings.

    A function defined inside a factory, a test method, or any other enclosing
    scope captures the names it uses (``HIDDEN``, ``ROWS``, a factory's ``nr``,
    ...) as closure free vars, not as globals — both namespaces have to be
    inspected to find them. Closure bindings override globals, matching Python's
    own name-resolution order at the function's call site.

    This is the namespace every ``@pl.jit`` name lookup must use. The generated
    ``@pl.program`` source is ``exec``'d in a fresh module holding only ``pl``
    and ``pld``, so a free name that survives into it is undefined there; both
    static shape resolution and body constant-folding therefore resolve against
    this mapping rather than ``__globals__`` alone.
    """
    return function_namespace(func)


def _reject_constexpr_rebinding(func_def: ast.FunctionDef, ctx: SpecializeContext) -> None:
    """Refuse a body that assigns to one of its own ``pl.constexpr`` parameters.

    The name is folded to its call-site value at every load, so an assignment to
    it cannot take effect: later reads would keep returning the original
    constant while the assignment itself leaked into the generated body as a
    stray runtime local. Silently compiling the wrong extent is the worst
    outcome available here, so the rebind is rejected instead.

    Only ``Store`` occurrences bind. A target can *read* names too —
    ``out[0:BLOCK] = tile`` stores through a subscript whose slice loads
    ``BLOCK`` — and rejecting those would refuse the most natural use of a
    compile-time extent while the same statement written with a literal
    compiles.

    Raises:
        ValueError: naming the parameter and the alternatives.
    """
    if not ctx.constexpr_values:
        return
    for node in ast.walk(func_def):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif isinstance(node, ast.For):
            targets = [node.target]
        for target in targets:
            for leaf in ast.walk(target):
                if (
                    isinstance(leaf, ast.Name)
                    and isinstance(leaf.ctx, ast.Store)
                    and leaf.id in ctx.constexpr_values
                ):
                    raise ValueError(
                        f"@pl.jit function '{ctx.source_def_name}': '{leaf.id}' is a "
                        f"'pl.constexpr' parameter, so it is a compile-time constant and cannot "
                        f"be assigned to. Compute the derived value where it is used "
                        f"(e.g. '[{leaf.id} // 2, {leaf.id}]'), bind it to a different local "
                        f"name, or annotate '{leaf.id}' as 'pl.Scalar[dtype]' if it should be a "
                        f"runtime value."
                    )


# ---------------------------------------------------------------------------
# Pre-scan: collect bind_dynamic calls before body transformation
# ---------------------------------------------------------------------------


# Cross-module private helper: production (jit/decorator.py) and the test suite
# reach it by underscore import; pyright's unused analysis discounts those.
def _collect_dynamic_dims(  # pyright: ignore[reportUnusedFunction]
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


def _resolve_dep_call_targets(
    func_def: ast.FunctionDef,
    dep_names: set[str],
    per_site: Mapping[tuple[str, int], str],
) -> dict[int, str]:
    """Map ``id(call node)`` → the generated function that call site reaches.

    The JIT layer folds each dep call site's ``pl.constexpr`` arguments
    separately, so two calls to one name can be compiled against different
    constants and must be rewritten to different generated functions. It cannot
    hand over the call nodes themselves — it parsed its own copy of the source
    — so it keys its answer by ``(call name, ordinal)``, the call's position
    among same-named calls in source order. Both sides enumerate ``ast.walk``
    matches sorted by position over the same dedented source, so the Nth call
    of a name means the same call in both. See ``_dep_call_nodes`` in
    ``decorator.py``; this resolves that answer back onto local nodes.
    """
    by_name: dict[str, list[ast.Call]] = {}
    for node in ast.walk(func_def):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in dep_names:
            by_name.setdefault(node.func.id, []).append(node)
    targets: dict[int, str] = {}
    for name, calls in by_name.items():
        for index, call in enumerate(sorted(calls, key=lambda call: (call.lineno, call.col_offset))):
            generated = per_site.get((name, index))
            if generated is not None:
                targets[id(call)] = generated
    return targets


# Cross-module private helper: production (jit/decorator.py) and the test suite
# reach it by underscore import; pyright's unused analysis discounts those.
def _collect_dynvar_names(  # pyright: ignore[reportUnusedFunction]
    func_def: ast.FunctionDef,
) -> dict[str, str]:
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


# Cross-module private helper: production (jit/decorator.py) and the test suite
# reach it by underscore import; pyright's unused analysis discounts those.
def _collect_annotation_dynamic_dims(  # pyright: ignore[reportUnusedFunction]
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


# ---------------------------------------------------------------------------
# Build annotated type string for a parameter
# ---------------------------------------------------------------------------


def _build_tensor_annotation(
    meta: TensorMeta, is_out: bool, is_distributed: bool = False, is_inout: bool = False
) -> str:
    """Build the type annotation string for a tensor parameter.

    Static dims emit as integer literals; DynDim entries emit as their
    DynVar variable name. When ``is_distributed`` is True the head type
    becomes ``pld.DistributedTensor`` instead of ``pl.Tensor`` — same
    shape/dtype subscript form, only the IR ObjectKind differs (see
    ``pld.DistributedTensor``).

    ``is_inout`` wraps the type in ``pl.InOut[...]`` (read-write parameter);
    it takes precedence over ``is_out``. Both round-trip the direction so the
    generated ``@pl.function`` source declares the same ``ir.ParamDirection``
    the user wrote.

    A non-ND ``meta.layout`` is emitted as the third slot, so the layout the
    user annotated survives into the generated source. Whether that layout is
    legal in a bare slot is the type resolver's call, not ours — ``pl.DN``, for
    instance, is rejected there with a migration hint.

    Returns:
        Annotation string such as ``pl.Tensor[[M, 128], pl.FP32]``,
        ``pl.Tensor[[64, 128], pl.FP16, pl.NZ]``,
        ``pl.Out[pl.Tensor[[128, 128], pl.FP32]]``,
        ``pl.InOut[pl.Tensor[[128, 128], pl.FP32]]``, or
        ``pld.DistributedTensor[[256], pl.INT8]``.
    """
    dims = [d.name if isinstance(d, DynDim) else str(d) for d in meta.shape]
    head = "pld.DistributedTensor" if is_distributed else "pl.Tensor"
    slots = [f"[{', '.join(dims)}]", values.dtype_str(meta.dtype)]
    layout_slot = values.layout_str(meta.layout) if meta.layout is not None else None
    if layout_slot is not None:
        slots.append(layout_slot)
    inner = f"{head}[{', '.join(slots)}]"
    if is_inout:
        return f"pl.InOut[{inner}]"
    return f"pl.Out[{inner}]" if is_out else inner


# ---------------------------------------------------------------------------
# Return-type inference (Option A: from Out params)
# ---------------------------------------------------------------------------


def _annotation_names(node: ast.AST) -> set[str]:
    """Collect names used by an annotation expression."""
    return {part.id for part in ast.walk(node) if isinstance(part, ast.Name)}


def _fold_const_names(
    node: ast.expr,
    py_globals: Mapping[str, Any],
    constexpr_values: Mapping[str, str] | None = None,
) -> ast.expr:
    """Replace ``Name`` nodes bound to renderable free values with their source form.

    An explicit tuple / scalar return annotation is copied into the generated
    ``@pl.program`` source verbatim (the return element has no ``TensorMeta`` to
    render from). Without folding, a symbolic shape dim (e.g.
    ``pl.Tensor[[BATCH, VOCAB], pl.FP32]``) would reach the parser as an
    unresolved name (``NameError: name 'BATCH' is not defined``). Fold the
    function's own free names — including the values the body transformer inlines
    at use sites — so the emitted annotation
    carries concrete extents and a dtype held in a constant (``pl.Tensor[[64],
    DTYPE]``) survives too.

    A helper's own ``pl.constexpr`` bindings take precedence over free names.
    Public DSL module/type aliases also become canonical paths, so ``p.Tile``
    and ``Tile[..., MS.Vec]`` remain defined in the generated module. This
    rewrites the source AST rather than rebuilding the type from shape/dtype;
    trailing metadata such as TileView and MemRef must survive unchanged.

    The input AST is left untouched (a deep copy is transformed) because
    ``func_def`` is shared with the body specialization pass.
    """

    class _Folder(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            if constexpr_values and node.id in constexpr_values:
                return ast.copy_location(ast.parse(constexpr_values[node.id], mode="eval").body, node)
            value = py_globals.get(node.id, _UNBOUND)
            folded = _render_annotation_value(value)
            if folded is not None:
                return ast.copy_location(folded, node)
            path = _annotation_symbol_index().get(id(value))
            return ast.copy_location(ast.parse(path, mode="eval").body, node) if path else node

    return _Folder().visit(copy.deepcopy(node))


@functools.lru_cache(maxsize=1)
def _annotation_symbol_index() -> dict[int, str]:
    """Map public DSL symbol identities to paths available in generated annotations.

    Unlike body-value folding, annotation rendering must also recognize module
    and constructor aliases, including TileView/MemRef and memory-space enums.
    Only exports of the canonical DSL modules qualify; arbitrary user objects
    keep their original names and annotation-evaluation behavior.
    """
    paths: dict[int, str] = {}
    for prefix, module_name in (("pl", "pypto.language"), ("pld", "pypto.language.distributed")):
        namespace = importlib.import_module(module_name)
        paths[id(namespace)] = prefix
        for name in sorted(namespace.__all__):
            paths.setdefault(id(getattr(namespace, name)), f"{prefix}.{name}")
    return paths


def _validate_dynamic_symbol_name(name: str, *, declaration: bool = False) -> None:
    """Reject names that cannot safely appear in generated annotation source."""
    if not name.isidentifier() or keyword.iskeyword(name) or name == "__debug__":
        raise ValueError(
            f"JIT dynamic symbol '{name}' must be an assignable Python identifier, not a keyword; "
            "choose a different pl.dynamic() label"
        )
    normalized = unicodedata.normalize("NFKC", name)
    if name != normalized:
        raise ValueError(
            f"JIT dynamic symbol '{name}' changes to '{normalized}' when Python parses identifiers; "
            "choose a different pl.dynamic() label with stable Unicode normalization"
        )
    if declaration and name in {"pl", "pld"}:
        raise ValueError(
            f"JIT dynamic symbol '{name}' is reserved for generated DSL imports; "
            "choose a different dynamic symbol name"
        )


def _tuple_annotation_elements(annotation: Any) -> tuple[Any, ...] | None:
    """Read the DSL Tuple wrapper or Python's built-in tuple annotation."""
    if isinstance(annotation, _LangTuple):
        return cast("tuple[Any, ...]", getattr(cast("Any", annotation), "_types", ()))
    if isinstance(annotation, types.GenericAlias) and annotation.__origin__ is tuple:
        return annotation.__args__
    return None


def _contains_tile_annotation(annotation: Any) -> bool:
    if annotation is _LangTile or isinstance(annotation, _LangTile):
        return True
    elements = _tuple_annotation_elements(annotation)
    return elements is not None and any(_contains_tile_annotation(element) for element in elements)


def _inline_annotation_ir_type(annotation: Any, path: tuple[int, ...] = ()) -> _ir.Type | None:
    """Build canonical inline annotation types, retaining mixed Tuple elements."""
    elements = _tuple_annotation_elements(annotation)
    if elements is not None:
        children: list[_ir.Type] = []
        for index, element in enumerate(elements):
            child_path = (*path, index)
            child = _inline_annotation_ir_type(element, child_path)
            if child is None:
                raise ValueError(
                    "Inline Tuple annotation contains an incomplete or unsupported type "
                    f"at path {child_path!r}"
                )
            children.append(child)
        return _ir.TupleType(children)
    if isinstance(annotation, DataType):
        return _ir.ScalarType(annotation)
    if isinstance(annotation, _LangScalar):
        return _ir.ScalarType(annotation.dtype) if annotation.dtype is not None else None
    if isinstance(annotation, _LangArray):
        if annotation.dtype is None or annotation.extent is None:
            return None
        return _ir.ArrayType(annotation.dtype, annotation.extent)
    # Tensor subclasses can carry a different IR object kind. Do not
    # silently erase that contract when retaining a mixed Tuple.
    if not isinstance(annotation, (_LangTile, _LangTensor)) or (
        isinstance(annotation, _LangTensor) and type(annotation) is not _LangTensor
    ):
        return None
    if annotation.shape is None or annotation.dtype is None:
        return None
    # Validate before the native printer's output reaches ast.parse, including
    # eager snapshots and cache fingerprints. Local Scalar symbols are not
    # declarations, so import-name conflicts are checked only at emission.
    _tile_annotation_dynvars(annotation)
    shape: list[_ir.Expr] = []
    for extent in cast("Sequence[Any]", annotation.shape):
        if isinstance(extent, _LangScalar):
            extent = extent.unwrap()
        # Native shaped-type constructors accept all-int or all-Expr shapes,
        # so a generic [M, 64] annotation must normalize its static axes too.
        if isinstance(extent, int):
            extent = _ir.ConstInt(extent, DataType.INDEX, _ir.Span.unknown())
        if not isinstance(extent, _ir.Expr):
            return None
        shape.append(extent)
    if isinstance(annotation, _LangTile):
        return _ir.TileType(
            shape,
            annotation.dtype,
            memref=annotation.memref,
            tile_view=annotation.tile_view,
            memory_space=annotation.memory_space,
        )
    view = annotation.layout
    if isinstance(view, TensorLayout):
        view = _ir.TensorView(layout=view)
    return _ir.TensorType(shape, annotation.dtype, memref=annotation.memref, tensor_view=view)


def _tile_annotation_source(annotation: Any) -> str | None:
    """Render a complete annotation through the native printer, including Tuple leaves."""
    annotation_type = _inline_annotation_ir_type(annotation)
    return _ir.python_print_type(annotation_type, format=False) if annotation_type is not None else None


def _render_annotation_value(value: Any) -> ast.expr | None:
    """Render annotation constants, including generic shapes and prebuilt metadata.

    Cache fingerprints must serialize these values even when a different
    component references an as-yet-unbound helper parameter. Use the same
    native printer as complete Tile annotations; body-value folding continues
    to accept only its existing operator argument values.
    """
    if isinstance(value, _LangDynVar):
        # The Python alias need not match the DynVar's emitted name.
        _validate_dynamic_symbol_name(value.name, declaration=True)
        return ast.Name(id=value.name, ctx=ast.Load())
    if isinstance(value, (_LangTile, _LangTensor)) or _tuple_annotation_elements(value) is not None:
        source = _tile_annotation_source(value)
        return ast.parse(source, mode="eval").body if source is not None else None
    if isinstance(value, TensorLayout):
        return ast.parse(f"pl.TensorLayout.{value.name}", mode="eval").body
    if isinstance(value, (list, tuple)):
        elements = [_render_annotation_value(element) for element in cast("Sequence[Any]", value)]
        if any(element is None for element in elements):
            return None
        rendered = cast("list[ast.expr]", elements)
        return (
            ast.List(elts=rendered, ctx=ast.Load())
            if isinstance(value, list)
            else ast.Tuple(elts=rendered, ctx=ast.Load())
        )
    if isinstance(value, (_ir.MemRef, _ir.TileView, _ir.TensorView)):
        # A fresh symbolic extent cannot equal an existing view's valid
        # shape by identity, preventing TileType from clearing that field
        # relative to the holder's unrelated physical shape.
        extent = _ir.Var("_annotation_extent", _ir.ScalarType(DataType.INDEX), _ir.Span.unknown())
        # Tile's public shape hint describes integer axes; this internal
        # annotation holder also accepts the Expr axes normalized above.
        shape = cast("Sequence[Any]", [extent])
        holder = (
            _LangTensor(shape, DataType.FP32, layout=cast("Any", value), _annotation_only=True)
            if isinstance(value, _ir.TensorView)
            else _LangTile(
                shape,
                DataType.FP32,
                memref=value if isinstance(value, _ir.MemRef) else None,
                tile_view=value if isinstance(value, _ir.TileView) else None,
                memory_space=_ir.MemorySpace.Vec if isinstance(value, _ir.MemRef) else None,
                _annotation_only=True,
            )
        )
        source = _tile_annotation_source(holder)
        assert source is not None
        node = ast.parse(source, mode="eval").body
        assert isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Tuple)
        # A holder has exactly one metadata object. MemRef precedes the
        # required memory-space enum; a view is its only trailing argument.
        default_view = (
            "pl.TensorView(layout=pl.TensorLayout.ND)"
            if isinstance(value, _ir.TensorView)
            else "pl.TileView()"
        )
        rendered = (
            node.slice.elts[2] if len(node.slice.elts) > 2 else ast.parse(default_view, mode="eval").body
        )
        if isinstance(value, _ir.TileView):
            assert isinstance(rendered, ast.Call)
            # The destination's memory space may change at a call. A layout
            # omitted relative to this neutral holder would then be inferred
            # differently (for example row_major Vec versus col_major Mat).
            fields = {
                "blayout": f"pl.TileLayout.{value.blayout.name}",
                "slayout": f"pl.TileLayout.{value.slayout.name}",
                "fractal": str(value.fractal),
                "pad": f"pl.PadValue.{value.pad.name}",
                "compact": f"pl.CompactMode.{value.compact.name}",
            }
            existing = {keyword.arg for keyword in rendered.keywords}
            rendered.keywords.extend(
                ast.keyword(arg=name, value=ast.parse(source, mode="eval").body)
                for name, source in fields.items()
                if name not in existing
            )
        return rendered
    if isinstance(value, (_LangScalar, _ir.Expr)):
        source = _tile_annotation_source(
            _LangTile(cast("Sequence[Any]", [value]), DataType.FP32, _annotation_only=True)
        )
        if source is None:
            return None
        node = ast.parse(source, mode="eval").body
        assert isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Tuple)
        shape = node.slice.elts[0]
        assert isinstance(shape, ast.List)
        return shape.elts[0]
    return values.render_free_value(value)


def _free_scalar_name_collision(name: str) -> ValueError:
    """Explain a free/local name ambiguity that canonical source cannot distinguish."""
    return ValueError(
        f"Free dynamic symbol '{name}' conflicts with a distinct Scalar parameter; "
        "choose a unique pl.dynamic() label for the free symbol"
    )


def _scalar_annotation_variable_ids(
    ann_globals: Mapping[str, Any], scalar_param_names: set[str]
) -> dict[str, int]:
    """Identify actual local Scalar bindings rather than matching their spelling alone."""
    variables: dict[str, int] = {}
    for name in scalar_param_names:
        value = ann_globals.get(name)
        if isinstance(value, _LangScalar):
            expr = value.unwrap()
            if isinstance(expr, _ir.Var):
                variables[name] = expr.unique_id
    return variables


def _tile_annotation_dynvars(
    annotation: Any, scalar_variable_ids: Mapping[str, int] | None = None
) -> dict[str, str]:
    """Collect scalar symbols from annotations or metadata, excluding allocation bases."""
    declarations: dict[str, str] = {}

    class Collector(_ir.IRVisitor):
        def visit_var(self, op: _ir.Var) -> None:
            if isinstance(op.type, _ir.ScalarType):
                _validate_dynamic_symbol_name(op.name_hint)
                if (
                    scalar_variable_ids is not None
                    and op.name_hint in scalar_variable_ids
                    and op.unique_id != scalar_variable_ids[op.name_hint]
                ):
                    raise _free_scalar_name_collision(op.name_hint)
                declarations[op.name_hint] = op.name_hint
            super().visit_var(op)

    collector = Collector()

    def collect_expr(value: Any) -> None:
        if isinstance(value, _LangScalar):
            value = value.unwrap()
        if isinstance(value, _ir.Expr):
            collector.visit_expr(value)

    def collect(value: Any) -> None:
        elements = _tuple_annotation_elements(value)
        if elements is not None:
            for element in elements:
                collect(element)
        elif isinstance(value, (_LangTile, _LangTensor)):
            for extent in value.shape or ():
                collect_expr(extent)
            collect(value.tile_view if isinstance(value, _LangTile) else value.layout)
            collect(value.memref)
        elif isinstance(value, (_ir.TileView, _ir.TensorView)):
            for extent in (*value.valid_shape, *value.stride):
                collect_expr(extent)
            if isinstance(value, _ir.TileView):
                collect_expr(value.start_offset)
        elif isinstance(value, _ir.MemRef):
            collect_expr(value.byte_offset_)
            collect_expr(value.slot_index_)

    collect(annotation)
    return declarations


def _tile_metadata_kind(value: Any) -> str | None:
    """Identify trailing Tile components by semantic type, independent of order."""
    for kind, cls in (("MemRef", _ir.MemRef), ("MemorySpace", _ir.MemorySpace), ("TileView", _ir.TileView)):
        if isinstance(value, cls):
            return kind
    if isinstance(value, (_ir.TensorView, TensorLayout)):
        return "TensorView"
    return None


def _tile_metadata_expression_kind(node: ast.expr, namespace: Mapping[str, Any]) -> str | None:
    """Identify an existing value or constructor without requiring its operands."""
    kind = _tile_metadata_kind(_eval_annotation(ast.unparse(node), namespace))
    if kind is not None:
        return kind
    root = node
    while isinstance(root, ast.Subscript):
        root = root.value
    if isinstance(root, ast.Call):
        constructor = _eval_annotation(ast.unparse(root.func), namespace)
        language = importlib.import_module("pypto.language")
        for kind, cls in (("MemRef", _ir.MemRef), ("TileView", _ir.TileView), ("TensorView", _ir.TensorView)):
            if constructor is cls or constructor is getattr(language, kind):
                return kind
    return None


def _eager_metadata_override_kinds(
    extras: list[ast.expr], annotation: Any, namespace: Mapping[str, Any], param_names: set[str]
) -> dict[int, str]:
    """Prove originating kinds before deferred metadata can replace an eager slot.

    Eager Tile objects retain one value of each kind but lose source order.
    Resolve independent values/constructors first, then use a unique remaining
    kind. Multiple erased aliases cannot be associated safely by position.
    """
    values = (
        (annotation.memref, annotation.memory_space, annotation.tile_view)
        if isinstance(annotation, _LangTile)
        else (annotation.memref, annotation.layout)
    )
    present = {kind for value in values if (kind := _tile_metadata_kind(value)) is not None}
    # Tile[...] records validated original order before factory aliases are
    # lost. Constructor-created/mutated objects can lack a usable record;
    # validate it against the current fields before using the fallback proof.
    recorded = cast("tuple[str, ...] | None", getattr(annotation, "_annotation_extra_kinds", None))
    if recorded is not None and len(recorded) == len(extras) == len(present) and set(recorded) == present:
        kinds: dict[int, str | None] = dict(enumerate(recorded))
    else:
        kinds = {
            index: _tile_metadata_expression_kind(extra, namespace) for index, extra in enumerate(extras)
        }
    remaining = present - {kind for kind in kinds.values() if kind is not None}
    unknown = [index for index, kind in kinds.items() if kind is None]
    if len(unknown) == len(remaining) == 1:
        kinds[unknown[0]] = remaining.pop()
    result: dict[int, str] = {}
    for index, extra in enumerate(extras):
        if not _annotation_names(extra) & param_names:
            continue
        kind = kinds[index]
        if kind is None:
            raise ValueError(
                f"Cannot determine original metadata kind of parameter-dependent Tile component "
                f"{ast.unparse(extra)!r}; make independent metadata constructors explicit"
            )
        result[index] = kind
    return result


def _unwrap_direction_annotation(node: ast.expr, namespace: Mapping[str, Any]) -> ast.expr:
    """Unwrap the DSL's identity direction markers, including imported aliases."""
    language = importlib.import_module("pypto.language")
    while isinstance(node, ast.Subscript):
        wrapper = _eval_annotation(ast.unparse(node.value), namespace)
        if wrapper is not language.Out and wrapper is not language.InOut:
            break
        node = node.slice
    return node


def _annotation_parameter_components(
    namespace: Mapping[str, Any],
    param_names: set[str],
    reserved_names: set[str],
) -> tuple[Callable[[ast.expr], ast.expr], dict[str, str]]:
    """Restore call-owned components with fresh names distinct from free DynVars."""
    # Mark original parameter references before folding free values. A free
    # M = dynamic("ROWS") may print as ROWS too, but is not the ROWS formal.
    parameter_placeholders: dict[str, str] = {}
    markers_by_param: dict[str, str] = {}

    class ParameterMarker(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            if node.id not in param_names:
                return node
            if node.id not in markers_by_param:
                marker = f"_pypto_tile_param_{len(parameter_placeholders)}"
                while marker in reserved_names:
                    marker = f"_{marker}"
                reserved_names.add(marker)
                markers_by_param[node.id] = marker
                parameter_placeholders[marker] = node.id
            return ast.copy_location(ast.Name(id=markers_by_param[node.id], ctx=ast.Load()), node)

    def parameter_component(node: ast.expr) -> ast.expr:
        missing = _annotation_names(node) - param_names - namespace.keys()
        if missing:
            # Eager evaluation has erased a mixed expression's operands; its
            # numeric result cannot reveal them once a parameter changes.
            raise ValueError(
                f"Cannot recover annotation-only factory value(s) {sorted(missing)!r} "
                f"in parameter-dependent Tile annotation component {ast.unparse(node)!r}; "
                "pass them as pl.constexpr arguments or reference them in the helper body"
            )
        marked = ParameterMarker().visit(copy.deepcopy(node))
        return _fold_const_names(marked, namespace)

    return parameter_component, parameter_placeholders


def _parameterized_eager_tile_leaf_source(
    original: ast.expr,
    eager_source: str,
    annotation: Any,
    namespace: Mapping[str, Any],
    param_names: set[str],
    reserved_names: set[str],
) -> tuple[str, tuple[_TileMetadataOverride, ...], dict[str, str]] | None:
    """Restore call-owned axes/dtype/metadata while keeping independent eager values.

    Overlay whole components: eager evaluation cannot recover a factory-only
    operand inside an expression such as ``ROWS + factory_padding``.
    """
    if not isinstance(original, ast.Subscript) or not isinstance(original.slice, ast.Tuple):
        return None
    original_parts = original.slice.elts
    if len(original_parts) < 2:
        return None
    merged = ast.parse(eager_source, mode="eval").body
    assert isinstance(merged, ast.Subscript) and isinstance(merged.slice, ast.Tuple)
    parameter_component, parameter_placeholders = _annotation_parameter_components(
        namespace, param_names, reserved_names
    )

    def depends_on_parameter(node: ast.expr) -> bool:
        return bool(_annotation_names(node) & param_names)

    raw_view = (
        annotation.tile_view if isinstance(annotation, _LangTile) else getattr(annotation, "layout", None)
    )
    if isinstance(raw_view, (_ir.TileView, _ir.TensorView)):
        # A view full relative to the ambient shape must stay explicit when
        # a call supplies a different shape.
        merged.slice.elts[2:] = [
            part
            for part in merged.slice.elts[2:]
            if not (
                isinstance(part, ast.Call)
                and isinstance(part.func, ast.Attribute)
                and part.func.attr in {"TileView", "TensorView"}
            )
        ]
        view = _render_annotation_value(raw_view)
        assert view is not None
        merged.slice.elts.append(view)

    merged_parts = merged.slice.elts
    original_shape, merged_shape = original_parts[0], merged_parts[0]
    if (
        isinstance(original_shape, (ast.List, ast.Tuple))
        and isinstance(merged_shape, (ast.List, ast.Tuple))
        and len(original_shape.elts) == len(merged_shape.elts)
    ):
        for index, axis in enumerate(original_shape.elts):
            if depends_on_parameter(axis):
                merged_shape.elts[index] = parameter_component(axis)
    elif depends_on_parameter(original_shape):
        merged_parts[0] = parameter_component(original_shape)
    if depends_on_parameter(original_parts[1]):
        merged_parts[1] = parameter_component(original_parts[1])
    overrides: list[_TileMetadataOverride] = []
    extras = original_parts[2:]
    if any(depends_on_parameter(extra) for extra in extras):
        for index, kind in _eager_metadata_override_kinds(extras, annotation, namespace, param_names).items():
            extra = extras[index]
            overrides.append(
                _TileMetadataOverride(kind, ast.unparse(extra), ast.unparse(parameter_component(extra)))
            )
    return ast.unparse(merged), tuple(overrides), parameter_placeholders


def _annotation_node_at_path(node: ast.expr, path: tuple[int, ...]) -> ast.Subscript:
    for index in path:
        if not isinstance(node, ast.Subscript):
            raise ValueError(f"Invalid inline Tuple annotation path {path!r}")
        elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if not 0 <= index < len(elements):
            raise ValueError(f"Invalid inline Tuple annotation path {path!r}")
        node = elements[index]
    if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Tuple):
        raise ValueError(f"Incomplete inline annotation at Tuple path {path!r}")
    return node


def _parameterized_eager_tile_source(
    original: ast.expr,
    eager_source: str,
    annotation: Any,
    namespace: Mapping[str, Any],
    param_names: set[str],
    reserved_names: set[str] | None = None,
) -> tuple[str, tuple[_TileMetadataOverride, ...], dict[str, str]] | None:
    """Apply the existing Tile leaf merge recursively under Tuple containers."""
    original = _unwrap_direction_annotation(original, namespace)
    if reserved_names is None:
        # Folding can reveal DynVar labels absent from both raw source and
        # eager expressions simplified under an ambient parameter value.
        reserved_names = (
            _annotation_names(original)
            | _annotation_names(ast.parse(eager_source, mode="eval").body)
            | _annotation_names(_fold_const_names(original, namespace))
            | namespace.keys()
        )
    elements = _tuple_annotation_elements(annotation)
    if elements is not None:
        merged = ast.parse(eager_source, mode="eval").body
        if not isinstance(original, ast.Subscript) or not isinstance(merged, ast.Subscript):
            return None
        originals = original.slice.elts if isinstance(original.slice, ast.Tuple) else [original.slice]
        rendered = merged.slice.elts if isinstance(merged.slice, ast.Tuple) else [merged.slice]
        if len(originals) != len(elements) or len(rendered) != len(elements):
            raise ValueError("Inline Tuple annotation has inconsistent element counts")
        overrides: list[_TileMetadataOverride] = []
        placeholders: dict[str, str] = {}
        for index, (child, source_node) in enumerate(zip(elements, originals, strict=True)):
            if not _annotation_names(source_node) & param_names:
                continue
            result = _parameterized_eager_tile_source(
                source_node, ast.unparse(rendered[index]), child, namespace, param_names, reserved_names
            )
            if result is None:
                raise ValueError(f"Cannot recover parameter-dependent inline Tuple element {index}")
            child_source, child_overrides, child_placeholders = result
            rendered[index] = ast.parse(child_source, mode="eval").body
            overrides.extend(replace(override, path=(index, *override.path)) for override in child_overrides)
            placeholders.update(child_placeholders)
        merged.slice = ast.Tuple(elts=rendered, ctx=ast.Load())
        return ast.unparse(merged), tuple(overrides), placeholders
    if isinstance(annotation, (_LangScalar, DataType)):
        if isinstance(annotation, _LangScalar) and annotation.dtype is None:
            return None
        dtype_node = original.slice if isinstance(original, ast.Subscript) else original
        if (
            isinstance(original, ast.Call)
            and original.args
            and _eval_annotation(ast.unparse(original.func), namespace) is _LangScalar
        ):
            dtype_node = original.args[0]
        parameter_component, placeholders = _annotation_parameter_components(
            namespace, param_names, reserved_names
        )
        return f"pl.Scalar[{ast.unparse(parameter_component(dtype_node))}]", (), placeholders
    return _parameterized_eager_tile_leaf_source(
        original, eager_source, annotation, namespace, param_names, reserved_names
    )


def _unreplaced_template_names(source: str, overrides: tuple[_TileMetadataOverride, ...]) -> set[str]:
    """Ignore stale eager metadata that a deferred expression is guaranteed to replace."""
    node = ast.parse(source, mode="eval").body
    namespace: dict[str, Any] = {"pl": importlib.import_module("pypto.language"), "__builtins__": {}}
    for override in overrides:
        leaf = _annotation_node_at_path(node, override.path)
        assert isinstance(leaf.slice, ast.Tuple)
        leaf.slice.elts[2:] = [
            extra
            for extra in leaf.slice.elts[2:]
            if _tile_metadata_expression_kind(extra, namespace) != override.kind
        ]
    return _annotation_names(node)


@cache_in_snapshot
def _eager_tile_annotation_data(func: Any) -> _EagerTileAnnotations:
    """Snapshot eager Tile contracts without freezing helper constexpr references.

    Only the evaluated annotation retains factory locals used exclusively in
    annotations: Python does not put those names in the function's closure.
    Keep canonical source in the same request snapshot used by cache hashing.
    Parameter-dependent shape/dtype components come from the source AST:
    constexpr bindings and runtime scalar dimensions belong to the call,
    while independent components retain factory values from eager evaluation.
    """
    annotations = inspect.get_annotations(func, eval_str=False)
    if not any(_contains_tile_annotation(annotation) for annotation in annotations.values()):
        return _EagerTileAnnotations()
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    definition = _find_func_def(tree, func.__name__)
    assert definition is not None, f"specialize: no def named {func.__name__!r} in its own source"
    param_names = {arg.arg for arg in definition.args.args}
    scalar_param_names = set(_classify_params(definition)[3])
    data = _EagerTileAnnotations()
    namespace = {name: value for name, value in func_name_lookup(func).items() if name not in param_names}
    for arg in definition.args.args:
        annotation = annotations.get(arg.arg)
        if not _contains_tile_annotation(annotation) or arg.annotation is None:
            continue
        source = _tile_annotation_source(annotation)
        overrides: tuple[_TileMetadataOverride, ...] = ()
        placeholders: dict[str, str] = {}
        if source is not None and _annotation_names(arg.annotation) & param_names:
            merged = _parameterized_eager_tile_source(
                arg.annotation, source, annotation, namespace, param_names
            )
            if merged is None:
                source = None
                data.fingerprints[arg.arg] = ast.unparse(_fold_const_names(arg.annotation, namespace))
            else:
                source, overrides, placeholders = merged
        if source is not None:
            data.sources[arg.arg] = source
            data.overrides[arg.arg] = overrides
            data.parameter_placeholders[arg.arg] = placeholders
            # Both the factory-derived base and deferred expressions affect
            # generated types. Constexpr values remain request-key-owned.
            data.fingerprints[arg.arg] = repr(
                (
                    source,
                    tuple(
                        (override.path, override.kind, override.canonical_source) for override in overrides
                    ),
                    tuple(placeholders.items()),
                )
            )
            used_names = _annotation_names(ast.parse(source))
            symbols = _tile_annotation_dynvars(annotation)
            # A metadata override may replace an ambient same-named Scalar
            # with the true local parameter; validate that value at emission.
            collisions = _unreplaced_template_names(source, overrides) & symbols.keys() & scalar_param_names
            if collisions:
                raise _free_scalar_name_collision(sorted(collisions)[0])
            data.dynvars.update(
                {
                    name: literal
                    for name, literal in symbols.items()
                    # Only Scalar formals supply local shape values. A Tile
                    # or Tensor formal with the same spelling is not in scope
                    # while its own annotation resolves this module symbol.
                    if name in used_names and name not in scalar_param_names
                }
            )
    return data


@cache_in_snapshot
def tile_annotation_fingerprints(func: Any) -> dict[str, str]:
    """Hash the canonical annotation spelling, including postponed DynVar aliases.

    Generic annotation aliases have no body-value rendering. Their canonical
    names still change generated source, so they need annotation-only cache
    records in both the memory and persistent keys.
    """
    fingerprints = dict(_eager_tile_annotation_data(func).fingerprints)
    namespace = dict(func_name_lookup(func))
    ann_globals: dict[str, Any] = {
        "pl": importlib.import_module("pypto.language"),
        "tuple": tuple,
        **namespace,
        "__builtins__": {},
    }
    param_names = set(inspect.signature(func).parameters)
    render_globals = {name: value for name, value in namespace.items() if name not in param_names}
    for name, annotation in inspect.get_annotations(func, eval_str=False).items():
        if name == "return" or not isinstance(annotation, str):
            continue
        try:
            node = ast.parse(annotation, mode="eval").body
        except SyntaxError:
            continue
        if not _annotation_names(node) & param_names:
            # A complete Tile may be named by an alias such as CONTRACT,
            # without any subscript in this annotation. Emission evaluates
            # that object too, so both cache paths must include its value.
            ann_obj = _eval_annotation(ast.unparse(node), ann_globals)
            source = _tile_annotation_source(ann_obj) if _contains_tile_annotation(ann_obj) else None
            if source is not None:
                fingerprints[name] = source
                continue
        node = _unwrap_direction_annotation(node, render_globals)
        if not isinstance(node, ast.Subscript):
            continue
        if _eval_annotation(ast.unparse(node.value), ann_globals) in (_LangTile, _LangTuple, tuple):
            # With unbound formals, fold independent aggregates/metadata and
            # keep formal expressions for the per-call constexpr identity
            # layered onto this function hash.
            fingerprints[name] = ast.unparse(_fold_const_names(node, render_globals))
    return fingerprints


def _specialize_eager_tile_annotation(
    ctx: SpecializeContext,
    param_name: str,
    ann_globals: Mapping[str, Any],
    scalar_param_names: set[str],
) -> str:
    """Apply per-call metadata values without losing the factory-derived base."""
    annotations = ctx.eager_tile_annotations
    substitutions = {
        marker: ctx.constexpr_values.get(param, param)
        for marker, param in annotations.parameter_placeholders.get(param_name, {}).items()
    }
    node = _fold_const_names(ast.parse(annotations.sources[param_name], mode="eval").body, {}, substitutions)
    assert isinstance(node, ast.Subscript)
    canonical_globals: dict[str, Any] = {
        "pl": importlib.import_module("pypto.language"),
        "__builtins__": {},
    }
    for override in annotations.overrides.get(param_name, ()):
        leaf = _annotation_node_at_path(node, override.path)
        assert isinstance(leaf.slice, ast.Tuple)
        # Original spellings must evaluate in the original namespace: a
        # Python DynVar alias can differ from its emitted canonical name.
        value = _eval_annotation(override.source, ann_globals)
        kind = _tile_metadata_kind(value)
        if kind != override.kind:
            raise ValueError(
                f"@pl.jit.inline helper '{ctx.source_def_name}' Tile parameter '{param_name}': "
                f"metadata component {override.source!r} must retain its original "
                f"{override.kind} kind, got {type(value).__name__}"
            )
        rendered = _render_annotation_value(value)
        assert rendered is not None
        # The printer orders fields by kind, not by original source position.
        # Replace the proven originating kind instead of adding a duplicate.
        leaf.slice.elts[2:] = [
            extra
            for extra in leaf.slice.elts[2:]
            if _tile_metadata_expression_kind(extra, canonical_globals) != kind
        ]
        leaf.slice.elts.append(rendered)
        # An evaluated override can select a different free DynVar from the
        # eager branch. Declare its actual symbols, keeping Scalar formals local.
        ctx.annotation_dynvars.update(
            {
                name: literal
                for name, literal in _tile_annotation_dynvars(
                    value, _scalar_annotation_variable_ids(ann_globals, scalar_param_names)
                ).items()
                if name not in scalar_param_names
            }
        )
    return ast.unparse(node)


def _infer_return_type(
    func_def: ast.FunctionDef,
    tensor_meta: dict[str, TensorMeta],
    out_params: list[str],
    distributed_params: set[str] | None = None,
    py_globals: Mapping[str, Any] | None = None,
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

    ``py_globals`` are the originating function's module globals. When a return
    element is filled from the explicit annotation (a local with no
    ``TensorMeta``, e.g. ``logits`` unpacked from a dep call), any module int
    constant in its shape (``pl.Tensor[[BATCH, VOCAB], pl.FP32]``) is folded to a
    literal via :func:`_fold_const_names`; otherwise the symbolic name would
    reach the parser unresolved.
    """
    dist_set = distributed_params or set()

    # Find the first return-with-value
    return_node: ast.Return | None = None
    for node in ast.walk(func_def):
        if isinstance(node, ast.Return) and node.value is not None:
            return_node = node
            break

    # Multi-return: `return a, b, ...` -> emit `tuple[T_a, T_b, ...]`.
    # Tensor elements are specialized from runtime metadata. Non-tensor elements
    # (for example a TASK_ID captured by ``with pl.spmd(...) as tid``) must be
    # supplied by a matching explicit tuple return annotation because they have
    # no TensorMeta entry.
    if return_node is not None and isinstance(return_node.value, ast.Tuple):
        return_elts = return_node.value.elts
        explicit_elements: list[ast.expr] | None = None
        if isinstance(func_def.returns, ast.Subscript) and _is_tuple_annotation(func_def.returns.value):
            annotation_slice = func_def.returns.slice
            explicit_elements = (
                annotation_slice.elts if isinstance(annotation_slice, ast.Tuple) else [annotation_slice]
            )
            if len(explicit_elements) != len(return_elts):
                explicit_elements = None

        def _element_uninferable(i: int) -> bool:
            e = return_elts[i]
            return not (isinstance(e, ast.Name) and e.id in tensor_meta)

        # A *complete* (subscripted) explicit element for an un-inferrable return
        # — e.g. ``pl.Scalar[pl.TASK_ID]`` — is one the parser cannot re-derive
        # from a value, so the annotation is required and must not be dropped. A
        # plain tensor local, in contrast, is re-inferable once the annotation is
        # gone (the pre-#2016 behavior).
        annotation_required = explicit_elements is not None and any(
            _element_uninferable(i) and isinstance(explicit_elements[i], ast.Subscript)
            for i in range(len(return_elts))
        )

        elt_annotations: list[str] = []
        has_bare_element = False
        for index, elt in enumerate(return_elts):
            if isinstance(elt, ast.Name) and elt.id in tensor_meta:
                elt_annotations.append(
                    _build_tensor_annotation(
                        tensor_meta[elt.id],
                        is_out=False,
                        is_distributed=elt.id in dist_set,
                    )
                )
            elif explicit_elements is not None and isinstance(explicit_elements[index], ast.Subscript):
                # Complete explicit element; fold module int constants in its shape
                # to literals so the emitted source carries concrete extents.
                folded = _fold_const_names(explicit_elements[index], py_globals or {})
                elt_annotations.append(ast.unparse(folded))
            elif explicit_elements is not None:
                # Un-inferrable element with a bare (shapeless) explicit annotation.
                elt_annotations.append(ast.unparse(explicit_elements[index]))
                has_bare_element = True
            else:
                return None  # no explicit annotation at all — drop (pre-#2016)

        # Drop the annotation only when it is entirely dispensable: a bare element
        # is present AND no sibling element requires the annotation. That keeps a
        # bare tensor-local tuple parsing again without stripping a TASK_ID an
        # adjacent element depends on (the bare element then surfaces as a clear
        # ``Incomplete type annotation`` the author must shape).
        if has_bare_element and not annotation_required:
            return None
        return f"tuple[{', '.join(elt_annotations)}]"

    # A TASK_ID (or another explicit Scalar) has no TensorMeta entry. Preserve
    # its declared type for any scalar expression so an inline helper can
    # return a task ID, constant, or computed scalar to its caller rather than
    # being treated as a void function.
    if (
        return_node is not None
        and isinstance(func_def.returns, ast.Subscript)
        and _is_scalar_annotation(func_def.returns.value)
    ):
        return ast.unparse(_fold_const_names(func_def.returns, py_globals or {}))

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
) -> tuple[list[str], list[str], list[str], dict[str, str], set[str]]:
    """Classify function parameters.

    Returns:
        (out_params, inout_params, tensor_params, scalar_dtype_strs, distributed_params)
        - out_params: names annotated Out[pl.Tensor] / Out[pld.DistributedTensor]
        - inout_params: names annotated InOut[pl.Tensor] / InOut[pld.DistributedTensor]
        - tensor_params: all names annotated as tensor-like — pl.Tensor or
          pld.DistributedTensor (including Out and InOut ones)
        - scalar_dtype_strs: param_name → dtype string for scalar params
        - distributed_params: subset of tensor_params whose annotation uses
          ``DistributedTensor`` (and should round-trip as
          ``pld.DistributedTensor[...]`` in the generated @pl.program source)
    """
    out_params: list[str] = []
    inout_params: list[str] = []
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

        # Detect Out[pl.Tensor] / InOut[pl.Tensor] (or the pld.DistributedTensor
        # variants). Both are direction wrappers around a tensor-like inner type;
        # they round-trip so the generated source keeps the user's direction.
        if isinstance(ann, ast.Subscript):
            outer = ann.value
            is_out = (isinstance(outer, ast.Name) and outer.id == "Out") or (
                isinstance(outer, ast.Attribute) and outer.attr == "Out"
            )
            is_inout = (isinstance(outer, ast.Name) and outer.id == "InOut") or (
                isinstance(outer, ast.Attribute) and outer.attr == "InOut"
            )
            # The inner subscript value
            inner = ann.slice
            is_tensor = _is_tensor_annotation(inner)
            if (is_out or is_inout) and is_tensor:
                (out_params if is_out else inout_params).append(name)
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

    return out_params, inout_params, tensor_params, scalar_dtype_strs, distributed_params


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


def _is_tuple_annotation(node: ast.expr) -> bool:
    """Return True if the AST node represents tuple, Tuple, or typing.Tuple."""
    if isinstance(node, ast.Name):
        return node.id in ("tuple", "Tuple")
    if isinstance(node, ast.Attribute):
        return node.attr in ("tuple", "Tuple")
    return False


def _eval_annotation(source: str, ann_globals: Mapping[str, Any]) -> Any:
    """Evaluate annotation source in its builtins-free namespace; None when it does not evaluate.

    Trusted input — the kernel's own annotation, evaluated where it was written
    with builtins stripped, so only type construction can run.
    """
    try:
        return eval(source, dict(ann_globals))  # noqa: S307
    except Exception:  # noqa: BLE001 — best effort; the caller keeps the annotation as written
        return None


def _scalar_dtype_source(dtype_text: str, ann_globals: Mapping[str, Any]) -> str:
    """The ``pl.<NAME>`` a scalar annotation's dtype evaluates to, else ``dtype_text``.

    A scalar's dtype may be spelled through a module or closure constant
    (``n: pl.Scalar[IDX]``). The generated module binds only ``pl`` and ``pld``,
    so the spelling itself would be undefined there.

    Args:
        dtype_text: The dtype as written in the annotation
        ann_globals: The function's builtins-free annotation namespace

    Returns:
        The rendered dtype, or ``dtype_text`` unchanged when it does not
        evaluate to a ``DataType``
    """
    value = _eval_annotation(dtype_text, ann_globals)
    rendered = constant_source(value) if isinstance(value, DataType) else None
    return dtype_text if rendered is None else rendered


def _bind_scalar_annotation_names(
    scalar_dtypes: Mapping[str, str],
    ann_globals: dict[str, Any],
    canonical_globals: Mapping[str, Any],
    is_inline: bool,
) -> None:
    """Use symbolic local bindings when earlier Scalar parameters size later Tiles."""
    if not is_inline:
        return
    # A same-named ambient constant must not select the Tile's physical size.
    # These bindings classify the type; the emitted AST keeps the local name.
    for name, dtype_source in scalar_dtypes.items():
        dtype = _eval_annotation(dtype_source, canonical_globals)
        if isinstance(dtype, DataType):
            var = _ir.Var(name, _ir.ScalarType(dtype), _ir.Span.unknown())
            ann_globals[name] = _LangScalar(expr=var)


def _extract_bare_dtype(node: ast.expr) -> str | None:
    """If node is a bare dtype like pl.FP32 or INDEX, return its string."""
    if isinstance(node, ast.Name) and node.id in values.DTYPE_NAMES:
        return f"pl.{node.id}"
    if isinstance(node, ast.Attribute) and node.attr in values.DTYPE_NAMES:
        return f"pl.{node.attr}"
    return None


def _ast_to_str(node: ast.expr) -> str:
    """Render a simple AST expression back to source."""
    return ast.unparse(node)


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


def _find_func_def(tree: ast.AST, func_name: str) -> ast.FunctionDef | None:
    """The ``FunctionDef`` named ``func_name`` anywhere under ``tree``."""
    return next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func_name), None)


def _original_def_line(ctx: SpecializeContext) -> int | None:
    """Line of the ``def`` inside a context's dedented source, 1-based.

    ``ctx.orig_start_line`` points at the function's *first decorator*, so the
    ``def`` itself sits some lines below it. Callers add
    ``orig_start_line - 1`` to convert to an absolute file line.

    Args:
        ctx: The specialization context whose source to inspect

    Returns:
        The 1-based line of the ``def``, or None if the source cannot be parsed
    """
    try:
        tree = ast.parse(textwrap.dedent(ctx.source))
    except SyntaxError:
        return None
    node = _find_func_def(tree, ctx.source_def_name)
    return None if node is None else node.lineno


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
        # Constexpr params per function. ``visit_Call`` drops their arguments
        # when rewriting a call: the callee resolved them at compile time and no
        # longer declares them, so passing one would be an arity mismatch.
        self._dep_constexpr_params: dict[str, set[str]] = {
            ctx.func_name: set(ctx.constexpr_values) for ctx in contexts
        }
        # generated-program absolute line → (orig_file, orig_line, orig_col),
        # built by specialize() so diagnostics map back to the user's .py (#1612).
        self._source_map: dict[int, tuple[str, int, int]] = {}
        # (ctx, original (lineno, col_offset) per transformed statement in DFS
        # order) per function — captured BEFORE ast.fix_missing_locations so
        # synthesized statements read as (0, 0), and zipped against the reparsed
        # generated statements in _build_source_map after assembly.
        self._fn_origin: list[tuple[SpecializeContext, list[tuple[int, int]]]] = []
        # Set when a parameter or transformed body references pld; the
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
                    _validate_dynamic_symbol_name(dv_name, declaration=True)
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
            # The signature line carries the parameter annotations, so an
            # annotation-level parse error (e.g. a rejected layout) reports
            # there — map it too, or such errors would point at the generated
            # source instead of the user's file.
            orig_def_line = _original_def_line(ctx)
            if orig_def_line is not None:
                out[gen_func.lineno] = (
                    ctx.orig_file,
                    orig_def_line + ctx.orig_start_line - 1,
                    ctx.orig_col_offset,
                )
            gen_stmts = _dfs_stmts(gen_func)
            # Reparsing the unparsed transform must preserve statement count and
            # DFS order; if it somehow diverges, skip this function's *statement*
            # mappings rather than emit wrong ones (those spans then fall back to
            # generated coords). The signature entry above is independent of the
            # body, so it stands either way.
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
        func_def = _find_func_def(tree, ctx.source_def_name)
        assert func_def is not None, f"specialize: no def named {ctx.source_def_name!r} in its own source"

        # Classify parameters
        out_params, inout_params, tensor_params, scalar_dtype_strs, distributed_params = _classify_params(
            func_def
        )

        # Inline helpers share the caller's tensor storage, so writes do not
        # need Out/InOut annotations at this eliminated call boundary. Local
        # name rebindings remain local; InlineFunctions separates their handles.
        # Warn on the redundant wrapper and emit bare tensor annotations.
        is_inline = ctx.func_type == "inline"
        if is_inline and (out_params or inout_params):
            warnings.warn(
                f"@pl.jit.inline helper '{ctx.source_def_name}' uses pl.Out[...]/pl.InOut[...] on "
                f"parameter(s) {(out_params + inout_params)!r}. Direction annotations are "
                f"deprecated for inline helpers because the body is spliced at the call "
                f"site before SSA conversion — tensor writes reach the caller's "
                f"storage while name rebindings stay local. Drop the wrapper; "
                f"bare pl.Tensor[...] works the same.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Collect all param names (excluding self). A ``pl.constexpr`` parameter
        # is dropped: its every use has been folded to a literal, so it has no
        # IR type to declare and no slot in the dispatch ABI. Keeping it would
        # emit an unannotated parameter the parser rejects.
        all_param_names = [
            arg.arg for arg in func_def.args.args if arg.arg != "self" and arg.arg not in ctx.constexpr_values
        ]
        _reject_constexpr_rebinding(func_def, ctx)

        # Build decorator
        decorator = self._build_decorator(ctx)

        # Evaluate original annotations in their own namespace. Folding a
        # dtype first would introduce ``pl.BF16`` even when the user imports
        # the DSL as ``p`` and binds an unrelated object to ``pl``.
        canonical_globals: dict[str, Any] = {
            "pl": importlib.import_module("pypto.language"),
            "pld": importlib.import_module("pypto.language.distributed"),
            "tuple": tuple,
            "__builtins__": {},
        }
        ann_globals: dict[str, Any] = {
            **canonical_globals,
            **ctx.py_globals,
            "__builtins__": {},
        }
        # Constexpr sources are compiler-rendered canonical expressions. Decode
        # them separately before overriding the corresponding original names,
        # so helper bindings win without replacing the user's module aliases.
        ann_globals.update(
            {name: _eval_annotation(value, canonical_globals) for name, value in ctx.constexpr_values.items()}
        )
        scalar_dtype_strs = {
            name: _scalar_dtype_source(text, ann_globals) for name, text in scalar_dtype_strs.items()
        }
        _bind_scalar_annotation_names(scalar_dtype_strs, ann_globals, canonical_globals, is_inline)
        scalar_param_names = set(scalar_dtype_strs)
        extra_param_anns: dict[str, str] = {}
        for arg in func_def.args.args:
            if (
                arg.arg == "self"
                or arg.annotation is None
                or arg.arg in tensor_params
                or arg.arg in scalar_dtype_strs
            ):
                continue
            # Eager annotations preserve their defining factory's values even
            # after it returns (or when a module global has the same name).
            # Parameter-dependent axes remain source expressions in this
            # snapshot; resolve this call's constexpr variant only now.
            if is_inline and arg.arg in ctx.eager_tile_annotations.sources:
                extra_param_anns[arg.arg] = _specialize_eager_tile_annotation(
                    ctx, arg.arg, ann_globals, scalar_param_names
                )
                continue
            ann_obj = _eval_annotation(ast.unparse(arg.annotation), ann_globals)
            if isinstance(ann_obj, _LangArray) and ann_obj.extent is not None and ann_obj.dtype is not None:
                extra_param_anns[arg.arg] = (
                    f"pl.Array[{ann_obj.extent}, {values.array_dtype_str(ann_obj.dtype)}]"
                )
            elif _contains_tile_annotation(ann_obj):
                # Keep the declared type independently of the function kind.
                # Inlining reuses the caller's actual Tile without a Tile copy.
                # Render the evaluated object so prebuilt TileView/MemRef and
                # aggregate shapes cannot leave undefined module aliases.
                # Symbolic Scalar bindings above preserve local dimensions.
                source = _tile_annotation_source(ann_obj)
                if source is not None:
                    extra_param_anns[arg.arg] = source
                ctx.annotation_dynvars.update(
                    {
                        name: literal
                        for name, literal in _tile_annotation_dynvars(
                            ann_obj, _scalar_annotation_variable_ids(ann_globals, scalar_param_names)
                        ).items()
                        if name not in scalar_param_names
                    }
                )

        params = self._build_params(
            all_param_names,
            out_params,
            inout_params,
            tensor_params,
            scalar_dtype_strs,
            distributed_params,
            ctx,
            is_inline=is_inline,
            extra_anns=extra_param_anns,
        )

        # Infer return type
        ret_type = _infer_return_type(
            func_def, ctx.tensor_meta, out_params, distributed_params, ctx.py_globals
        )
        ret_ann = f" -> {ret_type}" if ret_type else ""

        # External C++ kernel: emit header-only declaration(s) backed by the
        # hand-written source instead of a transformed DSL body. A "mixed"
        # kernel expands to an AIC member + AIV member + Group wrapper so the
        # entry's call lowers to a single MixedKernels submit.
        if ctx.func_type == "extern":
            return self._render_external(ctx, params, ret_ann, all_param_names)

        # Transform body
        dep_names = set(ctx.dep_names)
        # Collect all names defined anywhere in the function to seed collision avoidance.
        all_defined = {
            node.id
            for node in ast.walk(func_def)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        transformer = BodyTransformer(
            tensor_meta=ctx.tensor_meta,
            constexpr_values=ctx.constexpr_values,
            dep_names=dep_names,
            param_names=all_param_names,
            initial_used_names=all_defined,
            py_globals=ctx.py_globals,
            dep_param_names=self._dep_param_names,
            dep_constexpr_params=self._dep_constexpr_params,
            dep_func_names=ctx.dep_func_names,
            dep_call_targets=(
                _resolve_dep_call_targets(func_def, dep_names, ctx.dep_call_variants)
                if ctx.dep_call_variants
                else {}
            ),
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
                flat_body.extend(cast("list[ast.stmt]", item))
            else:
                flat_body.append(item)

        if not flat_body:
            flat_body = [ast.Pass()]

        if any(
            isinstance(node, ast.Name) and node.id == "pld" and isinstance(node.ctx, ast.Load)
            for stmt in flat_body
            for node in ast.walk(stmt)
        ):
            self._needs_pld_import = True

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

    def _render_external(
        self,
        ctx: SpecializeContext,
        params: list[str],
        ret_ann: str,
        all_param_names: list[str],
    ) -> list[str]:
        """Render an external C++ kernel declaration as @pl.function source lines.

        Single core (``core_type`` "aic"/"aiv") emits one header-only function.
        "mixed" emits an AIC member, an AIV member, and a Group wrapper that
        calls both — reproducing the ``@pl.program`` ``pl.group`` route so the
        entry's call lowers to a single MixedKernels submit.
        """
        name = ctx.func_name
        sig_params = ", ".join(params)
        call_args = ", ".join(all_param_names)
        header = f"def {name}(self, {sig_params}){ret_ann}:"

        def _member(
            member_name: str,
            core_upper: str,
            source: str,
            *,
            dual_aiv_dispatch: bool = False,
        ) -> list[str]:
            function_attrs: list[str] = []
            if ctx.external_include_dirs:
                encoded_include_dirs = encode_external_include_dirs(ctx.external_include_dirs)
                function_attrs.append(f'"{EXTERNAL_INCLUDE_DIRS_ATTR}": {encoded_include_dirs!r}')
            if dual_aiv_dispatch:
                function_attrs.append(f'"{DUAL_AIV_DISPATCH_ATTR}": True')
            attrs = f", attrs={{{', '.join(function_attrs)}}}" if function_attrs else ""
            return [
                f"@pl.function(type=pl.FunctionType.{core_upper}, external_source={source!r}{attrs})",
                f"def {member_name}(self, {sig_params}){ret_ann}:",
                "    ...",
            ]

        # The @pl.jit.extern decorator guarantees the source(s) for the selected
        # core_type are set; assert to narrow str | None -> str for the type checker.
        core = ctx.external_core_type
        if core == "aic":
            assert ctx.external_aic_source is not None
            return _member(name, "AIC", ctx.external_aic_source)
        if core == "aiv":
            assert ctx.external_aiv_source is not None
            return _member(name, "AIV", ctx.external_aiv_source)

        # Mixed: two members + a Group wrapper named after the extern so the
        # entry's ``self.<name>(...)`` call resolves to the group.
        assert ctx.external_aic_source is not None and ctx.external_aiv_source is not None
        lines = _member(f"{name}_aic", "AIC", ctx.external_aic_source)
        lines += _member(
            f"{name}_aiv",
            "AIV",
            ctx.external_aiv_source,
            dual_aiv_dispatch=ctx.external_dual_aiv_dispatch,
        )
        lines.append("@pl.function(type=pl.FunctionType.Group)")
        lines.append(header)
        # AIV lanes never capture a return; the AIC lane echoes the outputs, so
        # a value-returning kernel threads through the AIC member (matching the
        # @pl.program group example). A void kernel emits both calls uncaptured.
        if ret_ann:
            lines.append(f"    _r = self.{name}_aic({call_args})")
            lines.append(f"    self.{name}_aiv({call_args})")
            lines.append("    return _r")
        else:
            lines.append(f"    self.{name}_aic({call_args})")
            lines.append(f"    self.{name}_aiv({call_args})")
        return lines

    def _build_decorator(self, ctx: SpecializeContext) -> str:
        """Build the @pl.function(...) decorator line.

        Entry functions (func_type is None or 'orchestration') are emitted as
        Orchestration.  OutlineIncoreScopes outlines any inner
        ``with pl.at(level=pl.Level.CORE_GROUP):`` scopes into per-core InCore
        kernels.

        Host orchestrators (func_type == 'host') emit as
        ``@pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)`` —
        the canonical spelling for HOST-level orchestration that owns
        ``pld.alloc_window_buffer`` / ``pld.window`` / ``pld.world_size()``
        and dispatches chip orchestrators with ``device=`` (see
        ``tests/st/distributed/test_l3_allreduce.py``). ``type=`` is left at
        the default ``FunctionType.Opaque``; the C++ ``Function`` constructor
        only auto-derives level/role for InCore/Group/Orchestration types
        (``include/pypto/ir/function.h``), so a HOST Opaque function keeps
        the explicit ``role=Orchestrator``.
        """
        # auto_scope=False is meaningful for orchestration-level entries (the
        # Orchestration entry and the HOST orchestrator) and for inline
        # sub-functions, whose bodies are spliced into the caller so hand-placed
        # scopes land there. incore/opaque reject it at the decorator layer, so
        # ctx.auto_scope is always True for them and the suffix stays empty.
        auto_scope_suffix = "" if ctx.auto_scope else ", auto_scope=False"
        if ctx.func_type == "host":
            return f"@pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator{auto_scope_suffix})"
        if ctx.func_type is None or ctx.func_type == "orchestration":
            return f"@pl.function(type=pl.FunctionType.Orchestration{auto_scope_suffix})"
        if ctx.func_type == "inline":
            return f"@pl.function(type=pl.FunctionType.Inline{auto_scope_suffix})"
        if ctx.func_type == "opaque":
            return "@pl.function(type=pl.FunctionType.Opaque)"
        if ctx.func_type == "graph":
            return "@pl.function(type=pl.FunctionType.Graph)"
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
        inout_params: list[str],
        tensor_params: list[str],
        scalar_dtype_strs: dict[str, str],
        distributed_params: set[str],
        ctx: SpecializeContext,
        is_inline: bool = False,
        extra_anns: dict[str, str] | None = None,
    ) -> list[str]:
        """Build the annotated parameter strings for the function signature.

        ``extra_anns`` carries pre-rendered annotations for params that are
        neither tensors nor scalars (e.g. ``pl.Array[N, pl.TASK_ID]`` TaskId
        arrays) — without it those params would be emitted bare and the
        generated source would fail to parse.

        When ``is_inline`` is True, ``pl.Out[...]`` / ``pl.InOut[...]`` wrappers
        are stripped from tensor params — inlining preserves writes to shared
        storage and separates local name rebindings without a direction tag.

        Params listed in ``distributed_params`` round-trip as
        ``pld.DistributedTensor[...]`` and trigger the corresponding import in
        the generated module preamble.
        """
        result: list[str] = []
        for name in all_param_names:
            if name in tensor_params:
                is_out = (name in out_params) and not is_inline
                is_inout = (name in inout_params) and not is_inline
                is_distributed = name in distributed_params
                if is_distributed:
                    self._needs_pld_import = True
                meta = ctx.tensor_meta.get(name)
                if meta is None:
                    raise ValueError(
                        f"@pl.jit: missing inferred tensor metadata for parameter '{name}' "
                        f"of '{ctx.func_name}'. This usually means the tensor's shape/dtype "
                        "could not be statically determined, or that the call site passing it "
                        f"was not resolved (no call to '{ctx.func_name}' found in its caller). "
                        "Pass the tensor directly as a function argument, or ensure any "
                        "intermediate pl.create_tensor() used for this parameter has a "
                        "statically inferable shape and dtype."
                    )
                ann = _build_tensor_annotation(
                    meta, is_out=is_out, is_distributed=is_distributed, is_inout=is_inout
                )
                result.append(f"{name}: {ann}")
            elif name in scalar_dtype_strs:
                dtype_s = scalar_dtype_strs[name]
                result.append(f"{name}: pl.Scalar[{dtype_s}]")
            elif extra_anns and name in extra_anns:
                result.append(f"{name}: {extra_anns[name]}")
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


def build_specialize_context(  # noqa: PLR0913 — pass-through assembler; each arg maps to a SpecializeContext field
    func: Any,
    func_name: str,
    func_type: str | None,
    level: Any,
    tensor_meta: dict[str, TensorMeta],
    scalar_dtypes: dict[str, DataType],
    dep_names: list[str],
    auto_scope: bool = True,
    external_core_type: str | None = None,
    external_aic_source: str | None = None,
    external_aiv_source: str | None = None,
    external_dual_aiv_dispatch: bool = False,
    external_include_dirs: tuple[str, ...] = (),
    dep_func_names: dict[str, str] | None = None,
    constexpr_values: dict[str, str] | None = None,
    dep_call_variants: dict[tuple[str, int], str] | None = None,
) -> SpecializeContext:
    """Build a SpecializeContext from a Python function and call-site data.

    Args:
        func: The original Python function object.
        func_name: Name for the generated ``@pl.function``. Normally
            ``func.__name__``; the JIT layer passes a uniquified name when two
            distinct deps share a ``__name__``. The ``def`` is located in
            ``func``'s own source by ``func.__name__`` either way.
        func_type: 'orchestration', 'incore', or None.
        level: pl.Level enum or None.
        tensor_meta: TensorMeta per tensor param name.
        scalar_dtypes: DataType per scalar param name.
        dep_names: Names this function's source calls its @pl.jit deps by
            (the alias, under an aliased import).
        auto_scope: Whether the compiler auto-inserts AUTO runtime scopes.
            Forwarded to the generated ``@pl.function`` decorator; the
            Orchestration entry, HOST orchestrator, and inline sub-functions
            honor ``False``.
        dep_func_names: ``call name -> generated function name`` for the deps
            this function reaches under a different name; names that agree may
            be omitted.
        constexpr_values: Folded source text per ``pl.constexpr`` param name.
        dep_call_variants: ``(call name, ordinal) -> generated function name``
            for bodies where one call name reaches more than one generated
            function, which happens when a dep is compiled once per
            ``pl.constexpr`` value it is called with.

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
    eager_data = _eager_tile_annotation_data(func) if func_type == "inline" else _EagerTileAnnotations()

    return SpecializeContext(
        func_name=func_name,
        source_func_name=func.__name__,
        source=source,
        func_type=func_type,
        level=level,
        param_names=param_names,
        tensor_meta=tensor_meta,
        scalar_dtypes=scalar_dtypes,
        constexpr_values=constexpr_values or {},
        dep_names=dep_names,
        dep_func_names=dep_func_names or {},
        dep_call_variants=dep_call_variants or {},
        auto_scope=auto_scope,
        py_globals=func_name_lookup(func),
        eager_tile_annotations=eager_data,
        annotation_dynvars=dict(eager_data.dynvars),
        orig_file=orig_file,
        orig_start_line=orig_start_line,
        orig_col_offset=orig_col_offset,
        external_core_type=external_core_type,
        external_aic_source=external_aic_source,
        external_aiv_source=external_aiv_source,
        external_dual_aiv_dispatch=external_dual_aiv_dispatch,
        external_include_dirs=external_include_dirs,
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
