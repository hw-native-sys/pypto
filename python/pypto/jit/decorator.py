# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""@pl.jit decorator implementation.

Public API
----------
    # Single-function: one @jit entry with pl.at(level=pl.Level.CORE_GROUP) scope
    @pl.jit
    def kernel(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP):
            M, N = a.shape
            ...

    # Multi-function: @jit entry + one or more sub-function deps. Three flavours:
    @pl.jit.incore
    def sub_kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):     # FunctionType.InCore
        ...

    @pl.jit.inline
    def util(a: pl.Tensor, c: pl.Out[pl.Tensor]):           # FunctionType.Inline
        ...                                                 # spliced at call site

    @pl.jit.opaque
    def opaque_util(a: pl.Tensor, c: pl.Out[pl.Tensor]):    # FunctionType.Opaque
        for i in pl.parallel(...):                          # may wrap orchestration
            with pl.at(level=pl.Level.CORE_GROUP):          # loops + pl.at scopes
                ...

    @pl.jit
    def entry(a: pl.Tensor, c: pl.Out[pl.Tensor]):
        c = sub_kernel(a, c)   # dep discovered automatically (any of incore/inline/opaque)
        return c

JITFunction.__call__ flow
-------------------------
1. Lazily discover deps (incore/inline/opaque) from entry's globals (once).
2. Classify args: tensor vs scalar.
3. Extract TensorMeta from torch.Tensor arguments.
4. Scan entry + dep ASTs for bind_dynamic declarations.
5. Build CacheKey (dynamic dims → None in shape tuple).
6. Cache hit  → execute cached CompiledProgram on device → return result.
7. Cache miss → specialize (entry + deps) → pl.parse() → ir.compile() → cache → execute → return.
"""

from __future__ import annotations

import ast
import copy
import functools
import inspect
import os
import re
import shutil
import textwrap
from typing import Any, NamedTuple

from pypto.pypto_core import DataType

from .cache import CacheKey, compute_source_hash, make_cache_key
from .specializer import (
    DynDim,
    ShapeDim,
    SpecializeContext,
    Specializer,
    TensorMeta,
    _classify_params,
    _collect_annotation_dynamic_dims,
    _collect_dynvar_names,
    build_specialize_context,
)

# ---------------------------------------------------------------------------
# Error message rewriting for JIT compilation
# ---------------------------------------------------------------------------


def _rewrite_jit_error(exc: Exception, rename_map: dict[str, str]) -> Exception:
    """Replace internal alpha-renamed aliases (e.g. ``x_v1``) with the user's
    original variable name (``x``) in the exception message.

    Uses word-boundary matching to avoid partial replacements (e.g. replacing
    ``max_v1`` when the alias is ``x_v1``). Sorts aliases longest-first so
    longer aliases are matched before any shorter prefix aliases.
    """
    if not rename_map:
        return exc
    msg = str(exc)
    for alias in sorted(rename_map, key=len, reverse=True):
        original = rename_map[alias]
        msg = re.sub(rf"\b{re.escape(alias)}\b", original, msg)
    if msg == str(exc):
        return exc
    # Use copy.copy to preserve all exception fields (span, hint, note,
    # source_lines for ParserError subclasses) then patch the message.
    try:
        new_exc = copy.copy(exc)
        new_exc.args = (msg,)
        if hasattr(new_exc, "message"):
            object.__setattr__(new_exc, "message", msg)
    except Exception:  # noqa: BLE001
        # If copy fails (e.g. non-standard __init__), fall back to plain Exception.
        new_exc = Exception(msg)
    return new_exc


# ---------------------------------------------------------------------------
# torch-optional dtype conversion
# ---------------------------------------------------------------------------

# Sentinel list: empty means not yet loaded; [None] means torch unavailable;
# [torch_module] means torch is loaded.
_TORCH_CACHE: list[Any] = []
_TORCH_DTYPE_MAP: dict[Any, DataType] = {}


def _get_torch() -> Any:
    """Return the torch module, or None if not installed. Result is cached."""
    if not _TORCH_CACHE:
        try:
            import torch  # noqa: PLC0415

            _TORCH_CACHE.append(torch)
            _TORCH_DTYPE_MAP.update(
                {
                    torch.float16: DataType.FP16,
                    torch.float32: DataType.FP32,
                    torch.bfloat16: DataType.BF16,
                    torch.int8: DataType.INT8,
                    torch.int16: DataType.INT16,
                    torch.int32: DataType.INT32,
                    torch.int64: DataType.INT64,
                    torch.uint8: DataType.UINT8,
                    torch.bool: DataType.BOOL,
                }
            )
        except ImportError:
            _TORCH_CACHE.append(None)
    return _TORCH_CACHE[0]


def _torch_dtype_to_pypto(torch_dtype: Any) -> DataType:
    _get_torch()
    if torch_dtype not in _TORCH_DTYPE_MAP:
        raise TypeError(
            f"Unsupported torch dtype {torch_dtype}. "
            "Supported: float16, float32, bfloat16, int8/16/32/64, uint8, bool."
        )
    return _TORCH_DTYPE_MAP[torch_dtype]


def _ptoas_available() -> bool:
    """Return True if the ptoas binary is available on this machine."""
    ptoas_root = os.environ.get("PTOAS_ROOT")
    if ptoas_root:
        return os.path.isfile(os.path.join(ptoas_root, "ptoas"))
    return shutil.which("ptoas") is not None


def _is_tensor(obj: Any) -> bool:
    """Return True if obj is a torch.Tensor (without hard-importing torch)."""
    torch = _get_torch()
    if torch is None:
        return False
    return isinstance(obj, torch.Tensor)


def _extract_tensor_meta(
    tensor: Any,
    dyn_dims: dict[int, DynDim] | None = None,
) -> TensorMeta:
    """Extract TensorMeta from a torch.Tensor.

    ``dyn_dims`` maps ``dim_idx → DynDim`` for dims declared dynamic at this
    parameter (via ``bind_dynamic`` or an annotation-embedded ``pl.dynamic()``).
    The DynDim's ``static_bound`` is filled from the actual tensor extent at
    this call site.
    """
    dyn = dyn_dims or {}
    shape: list[ShapeDim] = []
    for i, d in enumerate(tensor.shape):
        extent = int(d)
        bound = dyn.get(i)
        if bound is None:
            shape.append(extent)
        else:
            shape.append(DynDim(name=bound.name, literal=bound.literal, static_bound=extent))
    return TensorMeta(shape=tuple(shape), dtype=_torch_dtype_to_pypto(tensor.dtype))


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=512)
def _get_func_def(func: Any) -> ast.FunctionDef:
    """Parse func source and return its FunctionDef node.

    Memoised on ``func`` identity: ``inspect.getsource`` + ``ast.parse`` are
    expensive and called repeatedly per JIT invocation (dep discovery, call-site
    extraction, dynamic-dim scan, local-meta inference) for the same functions.
    The returned node is shared across callers, so callers MUST treat it as
    read-only — every in-tree consumer only walks/reads it.

    Raises:
        OSError: If the source code cannot be retrieved (e.g. interactive REPL,
            Jupyter notebook, or exec/eval-generated functions).
    """
    try:
        src = textwrap.dedent(inspect.getsource(func))
    except OSError as e:
        raise OSError(
            f"@pl.jit cannot retrieve source code for '{func.__name__}'. "
            "Source code must be available on disk. "
            "Interactive shells, Jupyter notebooks, and exec/eval-generated "
            f"functions are not supported. (Original error: {e})"
        ) from e
    tree = ast.parse(src)
    func_def = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func.__name__),
        None,
    )
    if func_def is None:
        raise OSError(
            f"@pl.jit could not locate function definition '{func.__name__}' "
            "in its own source file. This may happen with heavily wrapped functions."
        )
    return func_def


def _collect_all_called_names(func_def: ast.FunctionDef) -> list[str]:
    """Return names used as bare (non-method) function calls in func_def body."""
    names: list[str] = []
    seen: set[str] = set()
    for node in ast.walk(func_def):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name not in seen:
                names.append(name)
                seen.add(name)
    return names


def _collect_bind_dynamic_bindings(
    func_def: ast.FunctionDef,
    param_names: set[str],
) -> dict[tuple[str, int], str]:
    """Scan ``param.bind_dynamic(dim, dynvar_var)`` calls.

    Returns ``(param_name, dim_idx) → dynvar_variable_name``.
    """
    result: dict[tuple[str, int], str] = {}
    for node in ast.walk(func_def):
        if not isinstance(node, ast.Expr):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        fn = call.func
        if not (
            isinstance(fn, ast.Attribute)
            and fn.attr == "bind_dynamic"
            and isinstance(fn.value, ast.Name)
            and fn.value.id in param_names
        ):
            continue
        if len(call.args) < 2:
            continue
        dim_node, dv_node = call.args[0], call.args[1]
        if (
            isinstance(dim_node, ast.Constant)
            and isinstance(dim_node.value, int)
            and isinstance(dv_node, ast.Name)
        ):
            result[(fn.value.id, dim_node.value)] = dv_node.id
    return result


@functools.lru_cache(maxsize=512)
def _build_dyndim_map_for_func(
    func: Any,
    param_names: tuple[str, ...],
) -> dict[str, dict[int, DynDim]]:
    """For each tensor param of ``func``, return ``dim_idx → DynDim``.

    Unions the three sources of "this dim is dynamic" declarations:

    1. ``param.bind_dynamic(dim, dynvar_var)`` — gives both ``(param, dim)``
       and the dynvar Python variable name.
    2. Annotation-embedded ``pl.dynamic()`` (``pl.Tensor[[M, …], …]``) —
       gives the same info; bind_dynamic takes precedence on overlap.
    3. ``M = pl.dynamic("M_literal")`` body assignment — maps each dynvar
       variable name to its ``pl.dynamic()`` string literal (these usually
       match but can differ, e.g. ``rows = pl.dynamic("M")``).

    ``DynDim.static_bound`` is filled with ``0`` here as a placeholder; the
    real per-call extent is injected by :func:`_extract_tensor_meta` from the
    actual ``torch.Tensor`` argument.
    """
    func_def = _get_func_def(func)
    pset = set(param_names)
    bd_bindings = _collect_bind_dynamic_bindings(func_def, pset)
    _, ann_bindings, ann_literals = _collect_annotation_dynamic_dims(func, pset)
    dyn_literals = _collect_dynvar_names(func_def)

    out: dict[str, dict[int, DynDim]] = {}

    def _literal_for(dv_name: str) -> str:
        return dyn_literals.get(dv_name) or ann_literals.get(dv_name, dv_name)

    # bind_dynamic source (authoritative when present)
    for (p, i), dv_name in bd_bindings.items():
        out.setdefault(p, {})[i] = DynDim(name=dv_name, literal=_literal_for(dv_name), static_bound=0)

    # annotation source fills dims not covered by bind_dynamic
    for key, dv_name in ann_bindings.items():
        p, idx_str = key.rsplit("__", 1)
        i = int(idx_str)
        per_param = out.setdefault(p, {})
        if i not in per_param:
            per_param[i] = DynDim(name=dv_name, literal=_literal_for(dv_name), static_bound=0)
    return out


def _scan_dynamic_dims(func: Any, param_names: list[str]) -> set[tuple[str, int]]:
    """Return dynamic ``(param, dim)`` pairs declared in ``func`` (union of all sources)."""
    dyn_map = _build_dyndim_map_for_func(func, tuple(param_names))
    return {(p, i) for p, dims in dyn_map.items() for i in dims}


def _compute_per_func_dyndim_maps(
    entry_func: Any,
    entry_param_names: list[str],
    deps: list[Any],
    callers_by_dep_id: dict[int, list[Any]],
    call_args_cache: dict[tuple[int, str], list[tuple[str | None, str | _SlicedArg | None]] | None],
) -> dict[int, dict[str, dict[int, DynDim]]]:
    """Per JIT function in the dep graph, return ``param → dim_idx → DynDim``.

    Each function's map starts from its own declarations
    (:func:`_build_dyndim_map_for_func`) and is augmented leaf-first with
    DynDim entries cascaded from every dep it calls: if a dep param
    ``a.dim=0`` is dynamic and the caller passes its arg ``x`` to that
    param, then ``x.dim=0`` is marked dynamic at the caller too. This
    keeps the entry's cache key DynDim-aware even when the entry itself
    has no ``bind_dynamic`` or annotation dynvar.

    The dep's own declarations take precedence at the caller — caller
    bindings only fill dims the caller didn't already specify.
    """
    out: dict[int, dict[str, dict[int, DynDim]]] = {
        id(entry_func): {
            p: dict(dims)
            for p, dims in _build_dyndim_map_for_func(entry_func, tuple(entry_param_names)).items()
        }
    }
    for dep in deps:
        out[id(dep._func)] = {
            p: dict(dims)
            for p, dims in _build_dyndim_map_for_func(dep._func, tuple(dep._param_names())).items()
        }

    # Leaf-first cascade: a dep's dynamic dim flows up to every recorded caller's arg.
    for dep in deps:
        dep_map = out[id(dep._func)]
        if not dep_map:
            continue
        for caller_func in callers_by_dep_id.get(id(dep._func), ()):
            call_args = call_args_cache.get((id(caller_func), dep.__name__))
            if call_args is None:
                continue
            param_mapping = _build_param_mapping(dep._param_names(), call_args)
            caller_map = out.get(id(caller_func))
            if caller_map is None:
                continue
            for dep_param, dim_to_dyn in dep_map.items():
                caller_arg = param_mapping.get(dep_param)
                # A per-rank sliced arg (x[r]) is keyed by a _SlicedArg, not a
                # caller variable name; DynDim does not flow through the dropped
                # leading dim, so skip it here.
                if caller_arg is None or isinstance(caller_arg, _SlicedArg):
                    continue
                target = caller_map.setdefault(caller_arg, {})
                for i, dyn in dim_to_dyn.items():
                    target.setdefault(i, dyn)
    return out


_PL_DTYPE_MAP: dict[str, Any] = {}


def _get_pl_dtype_map() -> dict[str, Any]:
    """Build a mapping from pl dtype attribute name (e.g. 'FP32') to DataType."""
    if not _PL_DTYPE_MAP:
        import pypto.language as _pl  # noqa: PLC0415
        from pypto.pypto_core import DataType as _DataType  # noqa: PLC0415

        _PL_DTYPE_MAP.update(
            {name: getattr(_pl, name) for name in dir(_pl) if isinstance(getattr(_pl, name), _DataType)}
        )
    return _PL_DTYPE_MAP


def _scan_dim_aliases(func_def: ast.FunctionDef) -> dict[str, tuple[str, int]]:
    """Map ``var → (param, dim_idx)`` for every ``var = pl.tensor.dim(P, k)``.

    The walker in :func:`_extract_local_tensor_metas` doesn't otherwise visit
    this assignment via its create_tensor/slice/dep branches, so the alias
    table is built up-front and consulted from ``_resolve_shape_elt``.

    The scan is flow-insensitive but safe: if a name is later reassigned to
    anything that is *not* another ``pl.tensor.dim(P, k)`` call, its alias is
    dropped from the table — so patterns like ``tokens = pl.tensor.dim(x, 0);
    tokens = tokens - 1`` don't leave a stale entry that would stamp the
    wrong DynDim onto downstream ``pl.create_tensor`` shapes.
    """
    aliases: dict[str, tuple[str, int]] = {}
    rebound: set[str] = set()
    for node in ast.walk(func_def):
        if isinstance(node, ast.Assign):
            targets, value = list(node.targets), node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            new_alias: tuple[str, int] | None = None
            if isinstance(value, ast.Call):
                fn = value.func
                if (
                    isinstance(fn, ast.Attribute)
                    and fn.attr == "dim"
                    and isinstance(fn.value, ast.Attribute)
                    and fn.value.attr == "tensor"
                    and isinstance(fn.value.value, ast.Name)
                    and fn.value.value.id == "pl"
                    and len(value.args) >= 2
                ):
                    src_arg, dim_arg = value.args[0], value.args[1]
                    if (
                        isinstance(src_arg, ast.Name)
                        and isinstance(dim_arg, ast.Constant)
                        and isinstance(dim_arg.value, int)
                    ):
                        new_alias = (src_arg.id, dim_arg.value)
            if new_alias is not None and target.id not in rebound:
                aliases[target.id] = new_alias
            else:
                # Reassigned to something other than pl.tensor.dim(...) —
                # drop any earlier alias and mark the name poisoned.
                aliases.pop(target.id, None)
                rebound.add(target.id)
    return aliases


def _build_dynvar_anchor_index(
    seed_meta: dict[str, TensorMeta],
) -> dict[str, list[tuple[str, int]]]:
    """Inverse map ``DynVar name → list of (param, dim_idx) anchor sites``.

    Lets ``[M, HIDDEN]`` (where ``M`` is a DynVar bound to a seeded param's
    dim) resolve via :func:`_extract_local_tensor_metas` to the parent dim's
    :class:`DynDim`.
    """
    anchors: dict[str, list[tuple[str, int]]] = {}
    for pname, meta in seed_meta.items():
        for i, dim in enumerate(meta.shape):
            if isinstance(dim, DynDim):
                anchors.setdefault(dim.name, []).append((pname, i))
    return anchors


def _func_name_lookup(func: Any) -> dict[str, Any]:
    """Return ``func.__globals__`` merged with closure free-var bindings.

    A function defined inside a test method (or any other enclosing scope)
    captures module-level helpers (``HIDDEN``, ``ROWS``, ...) as closure free
    vars, not as globals — both namespaces have to be inspected for static
    shape-element resolution to find them. Closure bindings override globals,
    matching Python's own name-resolution order at the function's call site.
    """
    out: dict[str, Any] = dict(getattr(func, "__globals__", {}))
    co_freevars = getattr(getattr(func, "__code__", None), "co_freevars", ())
    closure = getattr(func, "__closure__", None) or ()
    for fv_name, cell in zip(co_freevars, closure, strict=True):
        try:
            out[fv_name] = cell.cell_contents
        except ValueError:
            # Unbound closure cell — skip silently (matches _discover_deps).
            pass
    return out


def _scan_dep_io(
    func: Any, caller_func_type: str = "orchestration"
) -> dict[str, tuple[list[str], list[str]]]:
    """Return ``dep_name → (param_names, output_param_names)`` for every @pl.jit
    dep called from ``func``'s body.

    Used by :func:`_extract_local_tensor_metas` to propagate metas through
    ``v1, ..., vk = dep(args)`` assignments (each ``vi`` inherits the meta of
    the caller arg bound to the i-th output-like parameter).

    ``output_param_names`` covers both ``pl.Out[...]`` and ``pl.InOut[...]``
    params — a caller can capture either from ``v = dep(...)`` — and is kept in
    declaration order so it stays aligned with the callee's return order (the
    positional target<->param zip in :func:`_propagate_dep_out_metas`).

    ``caller_func_type`` mirrors :func:`_discover_deps`'s gating: a host
    orchestrator also admits ``orchestration`` deps (its chip orchestrators).
    """
    out: dict[str, tuple[list[str], list[str]]] = {}
    for dep in _discover_deps(func, caller_func_type):
        try:
            out_params, inout_params, _, _, _ = _classify_params(_get_func_def(dep._func))
        except OSError:
            continue
        param_names = dep._param_names()
        output_set = set(out_params) | set(inout_params)
        output_params = [p for p in param_names if p in output_set]
        out[dep.__name__] = (param_names, output_params)
    return out


def _propagate_dep_out_metas(
    call: ast.Call,
    dep_name: str,
    target: ast.expr,
    dep_io: dict[str, tuple[list[str], list[str]]],
    local: dict[str, TensorMeta],
) -> None:
    """For ``v1, ..., vk = dep(args)`` where ``dep`` has ``k`` ``Out`` params,
    bind each ``vi``'s meta to the caller arg passed to the matching ``Out``
    parameter. The local meta table is mutated in place.

    Mapping handles both positional and keyword args. No-op when the dep has
    no ``Out`` params or when target/arity don't match.
    """
    dep_params, out_params = dep_io[dep_name]
    if isinstance(target, ast.Name):
        names = [target.id]
    elif isinstance(target, ast.Tuple) and all(isinstance(e, ast.Name) for e in target.elts):
        names = [e.id for e in target.elts if isinstance(e, ast.Name)]
    else:
        return
    if not out_params or len(names) != len(out_params):
        return
    mapping: dict[str, str | None] = {}
    for i, arg in enumerate(call.args):
        if i < len(dep_params):
            mapping[dep_params[i]] = arg.id if isinstance(arg, ast.Name) else None
    for kw in call.keywords:
        if kw.arg is not None:
            mapping[kw.arg] = kw.value.id if isinstance(kw.value, ast.Name) else None
    for vname, out_param in zip(names, out_params, strict=True):
        caller_arg = mapping.get(out_param)
        if caller_arg is not None and caller_arg in local:
            local[vname] = local[caller_arg]


def _fold_int_arith(op: ast.operator, lhs: int, rhs: int) -> int | None:
    """Fold a binary arithmetic op over two Python ints, or return None.

    Used by ``_extract_local_tensor_metas._resolve_shape_elt`` to keep the
    shape-element resolver under the per-function branch limit. Anything
    involving a :class:`DynDim` operand is rejected upstream — this helper
    only sees ``int·int``.
    """
    if isinstance(op, ast.Add):
        return lhs + rhs
    if isinstance(op, ast.Sub):
        return lhs - rhs
    if isinstance(op, ast.Mult):
        return lhs * rhs
    if isinstance(op, ast.FloorDiv) and rhs != 0:
        return lhs // rhs
    if isinstance(op, ast.Mod) and rhs != 0:
        return lhs % rhs
    if isinstance(op, ast.Pow) and rhs >= 0:
        return lhs**rhs
    return None


def _extract_local_tensor_metas(
    func: Any,
    seed_meta: dict[str, TensorMeta] | None = None,
    seed_scalars: dict[str, int | float | bool] | None = None,
    caller_func_type: str = "orchestration",
) -> dict[str, TensorMeta]:
    """Infer ``TensorMeta`` for the local tensor variables in ``func``'s body.

    Walks the body in source order, tracking the three ways a local tensor can
    be produced inside a JIT function:

    1. ``var = pl.create_tensor([shape], dtype=pl.XXX)`` — shape from the
       literal list (literal ints, ``Name`` refs to int globals / seeded
       scalars, and simple int arithmetic over those), dtype from ``dtype=``.
       A shape element that resolves through a dynamic alias — either
       ``tokens = pl.tensor.dim(P, k)`` for a seeded param ``P`` whose dim
       ``k`` is :class:`DynDim`-bound, or a direct reference to a DynVar
       declared in the seed metas — stamps the matching ``DynDim`` onto the
       local's shape so the dynamic chain keeps flowing through subsequent
       deps.
    2. ``var = pl.slice(src, [shape], [...])`` — dtype inherited from ``src`` (a
       parameter or earlier local); each shape dim that is a static int is used
       as-is, and a non-static dim (e.g. a runtime ``valid_len``) falls back to
       ``src``'s corresponding dim, since a slice is bounded above by its
       parent. ``src`` dims that are themselves ``DynDim`` flow through
       transparently.
    3. ``v1, ..., vk = jit_dep(args)`` where ``jit_dep`` is an
       ``@pl.jit.incore`` / ``inline`` / ``opaque`` callee with ``k``
       ``pl.Out[...]`` parameters — each ``vi`` inherits the meta of the caller
       argument bound to the i-th ``Out`` parameter (the in-place-output
       convention every such kernel follows, and the same heuristic
       :func:`_infer_return_type` uses on the callee side).

    ``seed_meta`` pre-populates the table with the caller's parameter metas
    (including any :class:`DynDim` entries those carry) so a ``pl.slice`` of a
    parameter, a dep call passing a parameter through, or a local
    ``pl.create_tensor`` sized off a dynamic dim of a parameter all resolve;
    ``seed_scalars`` lets compile-time-specialized scalar parameters appear
    as shape dimensions. Anything not statically resolvable is skipped
    silently — the clear ``ValueError`` in ``Specializer._build_params`` then
    fires for that variable.
    """
    func_def = _get_func_def(func)
    local: dict[str, TensorMeta] = dict(seed_meta or {})
    dtype_map = _get_pl_dtype_map()
    func_globals = _func_name_lookup(func)
    scalars: dict[str, int | float | bool] = seed_scalars or {}
    dim_aliases = _scan_dim_aliases(func_def)
    dynvar_anchors = _build_dynvar_anchor_index(seed_meta or {})

    def _resolve_shape_elt(elt: ast.expr) -> ShapeDim | None:
        """Resolve a shape element to an ``int`` or a :class:`DynDim`.

        Dynamic resolution paths (added on top of the original static integer
        resolver):

        - ``Name`` that's a dim-alias for ``(P, k)`` where ``P`` is a seeded
          param with a :class:`DynDim` at dim ``k`` → returns that DynDim.
        - ``Name`` that's a DynVar declared on a seeded param → returns the
          DynDim of the (first) anchor site.

        Falls back to integer resolution for literal ints, int globals,
        seeded scalars, and arithmetic over those (the same combinations the
        original ``_resolve_int`` covered).
        """
        if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
            return elt.value
        if isinstance(elt, ast.Name):
            # Dim alias: tokens = pl.tensor.dim(P, k)
            alias = dim_aliases.get(elt.id)
            if alias is not None:
                p, k = alias
                src_meta = local.get(p)
                if src_meta is not None and k < len(src_meta.shape):
                    return src_meta.shape[k]
            # Direct DynVar reference (e.g. M used as a shape entry).
            anchors = dynvar_anchors.get(elt.id)
            if anchors:
                p, k = anchors[0]
                src_meta = local.get(p)
                if src_meta is not None and k < len(src_meta.shape):
                    d = src_meta.shape[k]
                    if isinstance(d, DynDim):
                        return d
            # Static int via globals or seeded scalars.
            value = func_globals.get(elt.id, scalars.get(elt.id))
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            return None
        if isinstance(elt, ast.BinOp):
            lhs = _resolve_shape_elt(elt.left)
            rhs = _resolve_shape_elt(elt.right)
            # Arithmetic over DynDim is not statically resolvable here; we
            # only fold int·int. Anything else (DynDim+int, DynDim*DynDim)
            # is left unresolved — the parent-dim fallback in _slice_meta /
            # the silent skip in _create_tensor_meta is the right behaviour.
            if isinstance(lhs, int) and isinstance(rhs, int):
                return _fold_int_arith(elt.op, lhs, rhs)
            return None
        if isinstance(elt, ast.UnaryOp):
            v = _resolve_shape_elt(elt.operand)
            if not isinstance(v, int):
                return None
            if isinstance(elt.op, ast.USub):
                return -v
            if isinstance(elt.op, ast.UAdd):
                return v
            return None
        return None

    def _resolve_int(elt: ast.expr) -> int | None:
        """Integer-only wrapper retained for _slice_meta's existing call site."""
        v = _resolve_shape_elt(elt)
        return v if isinstance(v, int) else None

    def _resolve_shape(node: ast.expr | None) -> tuple[ShapeDim, ...] | None:
        if not isinstance(node, ast.List):
            return None
        dims: list[ShapeDim] = []
        for elt in node.elts:
            v = _resolve_shape_elt(elt)
            if v is None:
                return None
            dims.append(v)
        return tuple(dims)

    def _dtype_from_kw(call: ast.Call) -> DataType | None:
        for kw in call.keywords:
            if (
                kw.arg == "dtype"
                and isinstance(kw.value, ast.Attribute)
                and isinstance(kw.value.value, ast.Name)
            ):
                return dtype_map.get(kw.value.attr)
        return None

    def _create_tensor_meta(call: ast.Call) -> TensorMeta | None:
        shape = _resolve_shape(call.args[0]) if call.args else None
        dtype_val = _dtype_from_kw(call)
        if shape is None or dtype_val is None:
            return None
        return TensorMeta(shape=shape, dtype=dtype_val)

    def _window_meta(call: ast.Call) -> TensorMeta | None:
        # pld.window(buffer, [shape], dtype=pl.XXX) — a distributed window view
        # over a window buffer. Shape is the 2nd positional arg; dtype is the
        # ``dtype=`` keyword (same spelling as create_tensor). Lets a host
        # orchestrator's per-rank window locals propagate their meta into the
        # ``pld.DistributedTensor`` parameters of the chip orchestrator it calls.
        shape = _resolve_shape(call.args[1]) if len(call.args) >= 2 else None
        dtype_val = _dtype_from_kw(call)
        if shape is None or dtype_val is None:
            return None
        return TensorMeta(shape=shape, dtype=dtype_val)

    def _reshape_meta(call: ast.Call) -> TensorMeta | None:
        # pl.reshape(input, shape) — dtype inherited from source tensor.
        src = (
            call.args[0] if call.args else next((kw.value for kw in call.keywords if kw.arg == "input"), None)
        )
        if not isinstance(src, ast.Name) or src.id not in local:
            return None
        src_meta = local[src.id]
        shape_node = (
            call.args[1]
            if len(call.args) >= 2
            else next((kw.value for kw in call.keywords if kw.arg == "shape"), None)
        )
        if shape_node is None:
            return None
        shape = _resolve_shape(shape_node)
        if shape is None:
            return None
        return TensorMeta(shape=shape, dtype=src_meta.dtype)

    def _slice_meta(call: ast.Call) -> TensorMeta | None:
        # pl.slice(tensor, shape, offset, ...) — shape is positional index 1 or kw `shape=`.
        src = call.args[0] if call.args else None
        if not isinstance(src, ast.Name) or src.id not in local:
            return None
        src_meta = local[src.id]
        shape_node = (
            call.args[1]
            if len(call.args) >= 2
            else next((kw.value for kw in call.keywords if kw.arg == "shape"), None)
        )
        if not isinstance(shape_node, ast.List) or len(shape_node.elts) != len(src_meta.shape):
            return None
        dims: list[ShapeDim] = []
        for elt, parent_dim in zip(shape_node.elts, src_meta.shape, strict=True):
            v = _resolve_int(elt)
            # A non-static slice dim (e.g. a runtime ``valid_len = pl.min(...)``)
            # is bounded above by the parent dim — advertise that static bound,
            # the way hand-written @pl.program code annotates a kernel that
            # consumes a narrowed view (see examples/models/04_paged_attention.py).
            # If the parent dim is itself a DynDim, it propagates through.
            dims.append(v if v is not None else parent_dim)
        return TensorMeta(shape=tuple(dims), dtype=src_meta.dtype)

    dep_io = _scan_dep_io(func, caller_func_type)

    def _record_dep_result_metas(call: ast.Call, dep_name: str, target: ast.expr) -> None:
        _propagate_dep_out_metas(call, dep_name, target, dep_io, local)

    # Dispatch table: pl.<attr>(...) → meta extraction function.
    # Replaces sequential if-chains, reducing branch and statement counts.
    _pl_attr_handlers = {
        "create_tensor": _create_tensor_meta,
        "slice": _slice_meta,
        "window": _window_meta,
        "reshape": _reshape_meta,
    }

    def _walk(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            # Descend into nested DSL scopes (for / if / with / while) first so
            # producers always run before their (same-or-deeper) consumers.
            for attr in ("body", "orelse", "finalbody"):
                sub = getattr(stmt, attr, None)
                if isinstance(sub, list):
                    _walk(sub)
            # Single-target ``v = ...`` and annotated ``v: T = ...`` both bind a
            # name we want to track (the latter is the common DSL style).
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target: ast.expr = stmt.targets[0]
            elif isinstance(stmt, ast.AnnAssign):
                target = stmt.target
            else:
                continue
            call = stmt.value
            if not isinstance(call, ast.Call):  # AnnAssign.value may be None
                continue
            fn = call.func
            if (
                isinstance(fn, ast.Attribute)
                and isinstance(fn.value, ast.Name)
                and isinstance(target, ast.Name)
            ):
                handler = _pl_attr_handlers.get(fn.attr)
                if handler is not None:
                    meta = handler(call)
                    if meta is not None:
                        local[target.id] = meta
                    continue
            if isinstance(fn, ast.Name) and fn.id in dep_io:
                _record_dep_result_metas(call, fn.id, target)

    _walk(func_def.body)
    return local


class _SlicedArg(NamedTuple):
    """A call-site argument of the form ``base[i]`` / ``base[i, j]`` that drops
    one or more leading dims of a Name (the per-rank ``chip_orch(x[r], ...)``
    dispatch pattern). ``drop`` counts the integer (non-slice) index elements;
    the dep parameter inherits ``base``'s meta with those leading dims removed.
    """

    base: str
    drop: int


def _arg_ref(arg: ast.expr) -> str | _SlicedArg | None:
    """Caller-side reference for a call argument.

    - ``ast.Name`` → the variable name (``str``).
    - ``ast.Subscript`` of a Name with integer indices (``x[r]``, ``x[r, 0]``)
      → a :class:`_SlicedArg` recording the base name and how many leading
      dims the indexing drops. Slice indices (``x[r:r+1]``) keep their dim and
      are not counted.
    - anything else (literal, attribute, computed expr) → ``None``.
    """
    if isinstance(arg, ast.Name):
        return arg.id
    if isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name):
        sl = arg.slice
        elts = sl.elts if isinstance(sl, ast.Tuple) else [sl]
        drop = sum(1 for e in elts if not isinstance(e, ast.Slice))
        if drop > 0:
            return _SlicedArg(arg.value.id, drop)
    return None


def _extract_call_args_for_dep(
    entry_func: Any, dep_name: str
) -> list[tuple[str | None, str | _SlicedArg | None]] | None:
    """Find the arguments passed to ``dep_name`` in ``entry_func``'s body.

    Returns a unified list of ``(param_name, arg_ref)`` pairs:

    - ``param_name`` is ``None`` for a positional argument (the consumer
      pairs it with the dep's parameter list by index) and the keyword
      name for a keyword argument.
    - ``arg_ref`` is the caller-side reference: a variable name (``str``), a
      :class:`_SlicedArg` for a per-rank subscript (``x[r]``), or ``None`` for
      other non-``Name`` expressions (literals, attribute access, …).

    Mixed calls like ``dep(a, out=out)`` are preserved correctly. Returns
    ``None`` if no call site to ``dep_name`` is found. Only the first call
    site is examined.
    """
    func_def = _get_func_def(entry_func)
    for node in ast.walk(func_def):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == dep_name):
            continue
        result: list[tuple[str | None, str | _SlicedArg | None]] = [
            (None, _arg_ref(arg)) for arg in node.args
        ]
        result.extend(
            (kw.arg, _arg_ref(kw.value))
            for kw in node.keywords
            if kw.arg is not None  # skip **kwargs splats
        )
        return result
    return None


def _build_param_mapping(
    dep_param_names: list[str],
    call_args: list[tuple[str | None, str | _SlicedArg | None]],
) -> dict[str, str | _SlicedArg | None]:
    """Map dep parameter name → caller argument ref from call-site args.

    ``call_args`` is the unified form returned by
    ``_extract_call_args_for_dep``: a list of ``(param_name, arg_ref)``
    pairs where ``param_name is None`` marks a positional argument (paired
    with ``dep_param_names`` by index) and a string is a keyword name. The
    ``arg_ref`` may be a name (``str``), a :class:`_SlicedArg`, or ``None``.
    Mixed positional + keyword call sites collapse to the same dict.
    """
    mapping: dict[str, str | _SlicedArg | None] = {}
    pos_idx = 0
    for param_name, arg_name in call_args:
        if param_name is None:
            if pos_idx < len(dep_param_names):
                mapping[dep_param_names[pos_idx]] = arg_name
            pos_idx += 1
        else:
            mapping[param_name] = arg_name
    return mapping


def _resolve_dep_call_metadata(
    dep: JITFunction,
    caller_func: Any,
    caller_tensor_meta: dict[str, TensorMeta],
    caller_scalar_values: dict[str, int | float | bool],
    caller_scalar_dtypes: dict[str, DataType],
    dep_dyn_map: dict[str, dict[int, DynDim]],
    caller_func_type: str = "orchestration",
) -> tuple[
    dict[str, TensorMeta],
    dict[str, int | float | bool],
    dict[str, DataType],
]:
    """Map ``dep``'s parameter names to TensorMeta / scalar metadata using
    ``caller_func``'s call-site arguments.

    The caller may be the entry function or another dep (transitive case);
    in either case we look up the (first) ``dep(...)`` call site in
    ``caller_func``'s body and apply the positional-or-keyword mapping.
    Intermediate tensors produced in the caller — ``pl.create_tensor``,
    ``pl.slice`` views, and the return values of other ``@pl.jit`` deps — are
    folded into the metadata pool (see :func:`_extract_local_tensor_metas`).
    Falls back to name-based matching when call-site extraction fails.

    ``caller_func_type`` is forwarded to :func:`_extract_local_tensor_metas`
    so a host orchestrator's body can also recognise chip-orchestrator deps
    when walking ``v = chip_orch(...)`` return-capture assignments.
    """
    dep_param_names = dep._param_names()
    call_args = _extract_call_args_for_dep(caller_func, dep.__name__)
    intermediate_metas = _extract_local_tensor_metas(
        caller_func,
        seed_meta=caller_tensor_meta,
        seed_scalars=caller_scalar_values,
        caller_func_type=caller_func_type,
    )
    all_tensor_meta = {**intermediate_metas, **caller_tensor_meta}

    dep_tensor_meta: dict[str, TensorMeta] = {}
    dep_scalar_values: dict[str, int | float | bool] = {}
    dep_scalar_dtypes: dict[str, DataType] = {}

    if call_args is not None:
        for dep_param, caller_arg in _build_param_mapping(dep_param_names, call_args).items():
            if caller_arg is None:
                continue
            if isinstance(caller_arg, _SlicedArg):
                # Per-rank dispatch ``chip_orch(x[r], ...)``: the dep parameter
                # inherits the base tensor's meta with ``drop`` leading dims
                # removed (the subscripted dims selected by integer indices).
                base_meta = all_tensor_meta.get(caller_arg.base)
                if base_meta is not None and caller_arg.drop < len(base_meta.shape):
                    dep_tensor_meta[dep_param] = TensorMeta(
                        shape=base_meta.shape[caller_arg.drop :],
                        dtype=base_meta.dtype,
                    )
                continue
            if caller_arg in all_tensor_meta:
                dep_tensor_meta[dep_param] = all_tensor_meta[caller_arg]
            elif caller_arg in caller_scalar_values:
                dep_scalar_values[dep_param] = caller_scalar_values[caller_arg]
                if caller_arg in caller_scalar_dtypes:
                    dep_scalar_dtypes[dep_param] = caller_scalar_dtypes[caller_arg]
    else:
        # Fallback: name-based matching against the caller's metadata pool.
        dep_tensor_meta = {n: all_tensor_meta[n] for n in dep_param_names if n in all_tensor_meta}
        dep_scalar_values = {n: caller_scalar_values[n] for n in dep_param_names if n in caller_scalar_values}
        dep_scalar_dtypes = {n: caller_scalar_dtypes[n] for n in dep_param_names if n in caller_scalar_dtypes}

    # Overlay DynDim from the dep's own declarations (pre-computed in
    # _compute_per_func_dyndim_maps). Dims already carrying a DynDim from
    # the caller take precedence; we only fill plain int dims and pin
    # ``static_bound`` to the meta's current extent so cache keys stay coherent.
    for dep_param, dim_to_dyn in dep_dyn_map.items():
        meta = dep_tensor_meta.get(dep_param)
        if meta is None:
            continue
        new_shape: list[ShapeDim] = list(meta.shape)
        changed = False
        for i, dyn in dim_to_dyn.items():
            if i >= len(new_shape) or isinstance(new_shape[i], DynDim):
                continue
            existing = new_shape[i]
            assert isinstance(existing, int)
            new_shape[i] = DynDim(name=dyn.name, literal=dyn.literal, static_bound=existing)
            changed = True
        if changed:
            dep_tensor_meta[dep_param] = TensorMeta(shape=tuple(new_shape), dtype=meta.dtype)

    return dep_tensor_meta, dep_scalar_values, dep_scalar_dtypes


# ---------------------------------------------------------------------------
# RunConfig -> ir.compile() keyword forwarding
# ---------------------------------------------------------------------------


def _run_config_compile_kwargs(run_config: Any) -> dict[str, Any]:
    """Extract ``ir.compile()`` keyword arguments from a ``pypto.runtime.RunConfig``.

    Maps the compile-side fields of ``RunConfig`` onto the parameters
    ``ir.compile()`` accepts, so a ``@pl.jit`` kernel invoked with
    ``config=RunConfig(...)`` honours the same compile knobs that
    ``ir.compile(program, ...)`` does. Runtime-only fields (``device_id``,
    ``rtol`` / ``atol``, DFX toggles, ...) are not compile inputs and are
    consumed by ``CompiledProgram.__call__`` instead.

    ``backend_type`` is intentionally omitted: ``ir.compile()`` derives the
    codegen backend from ``platform``, which the JIT path already forwards;
    passing ``backend_type`` as well would be redundant and could conflict.

    ``block_dim`` is intentionally omitted too: although ``ir.compile()``
    accepts it (baking it into ``kernel_config.py``), the JIT runtime path
    always re-supplies ``RunConfig.block_dim`` at dispatch time
    (``execute_compiled``), which overrides the baked value. Forwarding it
    here would be redundant and would split the cache key on a value that
    never reaches the executed artifact.

    ``output_dir`` is forwarded only when set, so an unset value defers to
    ``ir.compile()``'s own default.

    ``distributed_config`` is likewise forwarded only when set. When supplied it
    makes ``ir.compile()`` emit a ``DistributedCompiledProgram`` (HOST-level
    ``@pl.jit.host`` kernels), which ``__call__`` then dispatches per-rank. An
    unset value defers to ``ir.compile()``'s default (a single-chip
    ``CompiledProgram``) and keeps it out of the cache key for non-distributed
    callers.

    ``analyze_auto_scopes_for_deps`` is forwarded because it changes the pass
    pipeline's dependency derivation and therefore the generated orchestration.
    """
    kwargs: dict[str, Any] = {
        "strategy": run_config.strategy,
        "dump_passes": run_config.dump_passes,
        "profiling": run_config.compile_profiling,
        "diagnostic_phase": run_config.diagnostic_phase,
        "disabled_diagnostics": run_config.disabled_diagnostics,
        "analyze_auto_scopes_for_deps": run_config.analyze_auto_scopes_for_deps,
    }
    if run_config.save_kernels_dir is not None:
        kwargs["output_dir"] = run_config.save_kernels_dir
    if run_config.distributed_config is not None:
        kwargs["distributed_config"] = run_config.distributed_config
    return kwargs


# ---------------------------------------------------------------------------


class JITFunction:
    """A JIT-compiled function with shape specialization and caching.

    Created by the ``@jit`` or ``@jit.incore`` decorators.

    Attributes:
        _func: Original Python function.
        _func_type: 'orchestration' | 'host' | 'incore' | 'inline' | 'opaque'.
            ``'host'`` is the HOST-level orchestrator produced by
            ``@pl.jit.host`` — it owns ``pld.alloc_window_buffer`` /
            ``pld.window`` / ``pld.world_size()`` and the per-rank
            ``device=`` dispatch loop. End-to-end runtime dispatch works when
            the caller supplies ``config=RunConfig(distributed_config=...)``:
            the config is forwarded through :meth:`_compile` → ``ir.compile()``
            (see :func:`_run_config_compile_kwargs`), which yields a
            ``DistributedCompiledProgram`` that :meth:`__call__` dispatches
            per-rank.
        _level: pl.Level or None.
        _auto_scope: Whether the compiler auto-inserts AUTO runtime scopes
            (PTO2_SCOPE) around the body and each for/if body. ``True`` by
            default; set ``False`` via ``@pl.jit(auto_scope=False)`` /
            ``@pl.jit.host(auto_scope=False)`` to place scopes by hand with
            ``with pl.scope()``. Also accepted on
            ``@pl.jit.inline(auto_scope=False)`` — after the ``InlineFunctions``
            pass splices the body, hand-placed scopes land in the caller.
            ``incore`` / ``opaque`` kinds reject it (they outline into
            separate kernels, so scopes never land in the caller).
        _dep_graph: Lazily-computed transitive JIT dep graph rooted here —
            ``(deps_topo, callers_by_dep_id, callees_by_func_id,
            call_args_cache)``.  ``None`` until first ``_get_dep_graph()``
            call.  See that method for the tuple's structure.
        _cache: L1 in-memory cache: CacheKey → CompiledProgram (post-pass ir.Program wrapped).
        _source_hash: Lazily-computed hash of func source + all dep sources.
    """

    def __init__(
        self,
        func: Any,
        func_type: str | None = None,
        level: Any = None,
        auto_scope: bool = True,
    ) -> None:
        self._func = func
        self._func_type = func_type or "orchestration"
        self._level = level
        self._auto_scope = auto_scope
        self._dep_graph: (
            tuple[
                list[JITFunction],
                dict[int, list[Any]],
                dict[int, list[str]],
                dict[tuple[int, str], list[tuple[str | None, str | _SlicedArg | None]] | None],
            ]
            | None
        ) = None
        self._cache: dict[CacheKey, Any] = {}  # CacheKey → CompiledProgram
        self._source_hash: str | None = None

        # Preserve function metadata
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__module__ = func.__module__

    @property
    def _diagnostic_filename(self) -> str:
        """Synthetic filename for the generated, specialized source.

        Statements that survive specialization are remapped to the user's real
        ``.py`` via the source map (see :meth:`Specializer.source_map`); this
        ``<jit:name>`` marker is only the fallback identity for synthesized
        statements that have no original location. Naming the kernel here is far
        more navigable than an anonymous ``<string>``. See issue #1612.
        """
        return f"<jit:{self.__name__}>"

    # ------------------------------------------------------------------
    # Lazy dep discovery
    # ------------------------------------------------------------------

    def _get_dep_graph(
        self,
    ) -> tuple[
        list[JITFunction],
        dict[int, list[Any]],
        dict[int, list[str]],
        dict[tuple[int, str], list[tuple[str | None, str | _SlicedArg | None]] | None],
    ]:
        """Return the transitive JIT dep graph rooted at this function.

        The graph is computed lazily on first access and cached for the
        lifetime of this ``JITFunction``. Returns:

        - ``deps_topo``: every reachable dep in leaf-first topological order
          (deduplicated by underlying Python function identity). The entry
          function is NOT included.
        - ``callers_by_dep_id``: for each dep, the list of Python functions
          whose bodies contain a call site to it. Recorded in DFS-discovery
          order; deduplicated within each list. The entry has no caller and
          does not appear as a key.

          Tensor / scalar metadata for a shared dep is still resolved
          through the first-recorded caller — call sites in other branches
          must agree on shapes/dtypes (otherwise one specialization would
          have to differ from another, which the one-context-per-function
          design doesn't support).
        - ``callees_by_func_id``: for each function (entry + every reached
          dep), the names of JIT deps it directly calls. Used to set each
          context's ``dep_names`` so the body transformer rewrites nested
          dep calls into the ``self.<dep>(...)`` form required by
          multi-function ``@pl.program``.
        - ``call_args_cache``: ``(id(caller_func), dep_name)`` → unified
          call-site arg list (see ``_extract_call_args_for_dep``) or
          ``None`` if the call site isn't found. Cached so metadata
          resolution doesn't re-walk caller ASTs on every JIT call.
        """
        if self._dep_graph is None:
            deps_topo: list[JITFunction] = []
            seen: set[int] = set()
            callers_by_dep_id: dict[int, list[Any]] = {}
            callees_by_func_id: dict[int, list[str]] = {}
            call_args_cache: dict[
                tuple[int, str], list[tuple[str | None, str | _SlicedArg | None]] | None
            ] = {}

            def visit(func: Any, caller_func_type: str) -> None:
                direct = _discover_deps(func, caller_func_type)
                callees_by_func_id[id(func)] = [d.__name__ for d in direct]
                for dep in direct:
                    # Key everything off ``id(dep._func)`` (the underlying
                    # Python function) — same key the downstream helpers
                    # use, and stable across multiple wrapper objects for
                    # the same source function.
                    callers = callers_by_dep_id.setdefault(id(dep._func), [])
                    if func not in callers:
                        callers.append(func)
                    # Memoise per-(caller, dep) call-site args once.
                    cache_key = (id(func), dep.__name__)
                    if cache_key not in call_args_cache:
                        call_args_cache[cache_key] = _extract_call_args_for_dep(func, dep.__name__)
                    if id(dep._func) in seen:
                        continue
                    # Mark before recursing — this also serves as a cycle
                    # guard (a self-recursive JIT function is unsupported
                    # but won't loop forever here).
                    seen.add(id(dep._func))
                    visit(dep._func, dep._func_type)
                    deps_topo.append(dep)

            visit(self._func, self._func_type)
            self._dep_graph = (
                deps_topo,
                callers_by_dep_id,
                callees_by_func_id,
                call_args_cache,
            )
        return self._dep_graph

    def _get_deps(self) -> list[JITFunction]:
        """Return all transitively-reachable JIT deps in leaf-first order."""
        return self._get_dep_graph()[0]

    # ------------------------------------------------------------------
    # Source hash (includes all dep sources; lazily computed after deps found)
    # ------------------------------------------------------------------

    def _get_source_hash(self) -> str:
        if self._source_hash is None:
            sources = [inspect.getsource(self._func)]
            for dep in self._get_deps():
                sources.append(inspect.getsource(dep._func))
            self._source_hash = compute_source_hash(sources)
        return self._source_hash

    # ------------------------------------------------------------------
    # Parameter introspection
    # ------------------------------------------------------------------

    def _param_names(self) -> list[str]:
        return [p for p in inspect.signature(self._func).parameters if p != "self"]

    def _bind_args(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[
        list[str],
        dict[str, Any],
        dict[str, TensorMeta],
        dict[str, int | float | bool],
        dict[str, DataType],
        dict[int, dict[str, dict[int, DynDim]]],
    ]:
        """Bind *args/**kwargs to param names and classify into tensor/scalar metadata.

        Tensor metas carry :class:`DynDim` entries for every param dim that is
        either declared dynamic at this function (``bind_dynamic`` / annotation
        ``pl.dynamic()``) **or** cascaded up from a dep's declarations.
        Cascading happens during ``_compute_per_func_dyndim_maps`` so the cache
        key reflects every dynamic dim reachable through the dep graph — two
        calls with different runtime extents reuse the same compilation.

        Returns the per-function DynDim map alongside the entry metadata so
        downstream specialization (``_resolve_dep_call_metadata``) can pull
        each dep's effective map without recomputing.
        """
        param_names = self._param_names()
        sig = inspect.signature(self._func)
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
        except TypeError as e:
            raise TypeError(f"@pl.jit function '{self.__name__}': {e}") from e

        arguments = dict(bound.arguments)

        deps, callers_by_id, _, call_args_cache = self._get_dep_graph()
        per_func_dyn_maps = _compute_per_func_dyndim_maps(
            self._func, param_names, deps, callers_by_id, call_args_cache
        )
        entry_dyn_map = per_func_dyn_maps[id(self._func)]
        tensor_meta: dict[str, TensorMeta] = {}
        scalar_values: dict[str, int | float | bool] = {}
        scalar_dtypes: dict[str, DataType] = {}

        for name, value in arguments.items():
            if _is_tensor(value):
                tensor_meta[name] = _extract_tensor_meta(value, entry_dyn_map.get(name))
            elif isinstance(value, (int, float, bool)):
                scalar_values[name] = value

        return param_names, arguments, tensor_meta, scalar_values, scalar_dtypes, per_func_dyn_maps

    # ------------------------------------------------------------------
    # Call
    # ------------------------------------------------------------------

    def _resolve_compiled(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, list[Any], Any | None]:
        """Bind args, look up or build the CompiledProgram, return it with the
        ordered positional arg list and the consumed RunConfig.

        Shared by :meth:`__call__` (which then dispatches) and :meth:`compile`
        (which then returns the CompiledProgram). Centralises:

        - ``config=`` keyword extraction (so it never leaks into the decorated
          function's signature)
        - ``_bind_args`` shape/dtype classification
        - cache-key construction (platform + strategy participate so artefacts
          for different targets never collide)
        - on-miss ``_compile()`` invocation

        Returns:
            ``(compiled, ordered_args, run_config)`` where ``ordered_args``
            is the positional list in declared parameter order — keyword
            callers like ``kernel(a=x, b=y)`` are normalised here so
            downstream dispatch is order-agnostic.
        """
        import pypto.language as pl  # noqa: PLC0415

        # Extract RunConfig without mutating *kwargs* — although the caller's
        # ``**kwargs`` dict is normally owned by Python at this scope, building
        # a fresh dict is the same cost and removes the ambiguity for readers
        # who don't track the calling convention.
        run_config = kwargs.get("config")
        if "config" in kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k != "config"}

        param_names, arguments, tensor_meta, scalar_values, scalar_dtypes, per_func_dyn = self._bind_args(
            args, kwargs
        )

        # Compile-side knobs (strategy, dump_passes, ...) come from the
        # RunConfig. Forwarding them lets a @pl.jit kernel honour the same
        # compile options as a direct ir.compile(program, ...) call.
        compile_kwargs = _run_config_compile_kwargs(run_config) if run_config is not None else {}

        # Build cache key. Platform and strategy are included so artifacts
        # compiled for different targets or optimization strategies never
        # collide in the same cache. With no RunConfig the strategy is
        # normalized to ir.compile()'s own default so the key holds the
        # effective strategy rather than a None sentinel.
        from pypto.ir.pass_manager import OptimizationStrategy  # noqa: PLC0415

        platform = run_config.platform if run_config is not None else None
        strategy = run_config.strategy if run_config is not None else OptimizationStrategy.Default
        # distributed_config is baked into the DistributedCompiledProgram and
        # drives per-rank dispatch, so it must split the cache: two @pl.jit.host
        # calls with different device_ids compile to distinct artifacts.
        distributed_config = run_config.distributed_config if run_config is not None else None
        analyze_auto_scopes_for_deps = (
            run_config.analyze_auto_scopes_for_deps if run_config is not None else False
        )
        key = make_cache_key(
            source_hash=self._get_source_hash(),
            param_names=param_names,
            tensor_shapes={n: m.static_shape() for n, m in tensor_meta.items()},
            tensor_dtypes={n: m.dtype for n, m in tensor_meta.items()},
            dynamic_dims={(n, i) for n, m in tensor_meta.items() for i in m.dynamic_dim_indices()},
            scalar_values=scalar_values,
            platform=platform,
            strategy=strategy,
            distributed_config=distributed_config,
            analyze_auto_scopes_for_deps=analyze_auto_scopes_for_deps,
        )

        # L1 cache lookup
        if key not in self._cache:
            self._cache[key] = self._compile(
                tensor_meta,
                scalar_values,
                scalar_dtypes,
                per_func_dyn,
                pl,
                platform=platform,
                **compile_kwargs,
            )

        # Use bound.arguments (in signature order) so keyword-style calls
        # like kernel(a=x, b=y) are routed correctly regardless of how the
        # caller passed them.
        compiled = self._cache[key]
        ordered_args = [arguments[n] for n in param_names if n in arguments]
        return compiled, ordered_args, run_config

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Specialize, compile (or serve from cache), and execute on device.

        On the first call for a given shape/dtype combination the function is
        specialized into ``@pl.program`` source, parsed, and compiled via
        ``ir.compile()`` (passes + codegen).  The resulting ``CompiledProgram``
        is stored in the L1 in-memory cache so subsequent calls with the same
        specialization key skip compilation entirely.

        The compiled kernel is then executed on the NPU device with the given
        torch tensor arguments (Triton-like API).

        A ``config=RunConfig(...)`` keyword argument is consumed here rather
        than passed to the decorated function: its compile-side fields
        (``strategy``, ``dump_passes``, diagnostics, ...) are forwarded to
        ``ir.compile()`` via :func:`_run_config_compile_kwargs`, and its
        runtime fields drive on-device execution.  ``strategy`` also takes
        part in the cache key so two strategies never share a cache entry.

        Args:
            *args: Positional arguments matching the decorated function's params.
            **kwargs: Keyword arguments.  A ``config`` keyword, if present, is
                a :class:`~pypto.runtime.runner.RunConfig` and is consumed by
                the JIT machinery (not forwarded to the decorated function).

        Returns:
            ``None`` for in-place calls (output tensors modified on device),
            or ``torch.Tensor`` / ``tuple[torch.Tensor, ...]`` for return-style
            calls. Per-run on-device timing is no longer surfaced as an
            attribute — read it from the runtime's ``[STRACE]`` log markers
            (simpler PR #1177).
        """
        compiled, ordered_args, run_config = self._resolve_compiled(args, kwargs)
        if run_config is not None:
            return compiled(*ordered_args, config=run_config)
        return compiled(*ordered_args)

    def compile(self, *args: Any, **kwargs: Any) -> Any:
        """Specialize + compile for the shape/dtype combination implied by *args*,
        and return the underlying :class:`~pypto.ir.compiled_program.CompiledProgram`.

        Same specialization / cache pipeline as :meth:`__call__`, minus the
        on-device dispatch. Use this when you want to drive execution through
        the runtime worker API directly:

        - :meth:`pypto.runtime.ChipWorker.run` / :meth:`~pypto.runtime.ChipWorker.register`
          for explicit L2 dispatch.
        - :attr:`CompiledProgram.chip_callable` / ``runtime_name`` / ``runtime_config``
          to drive a hand-constructed ``simpler.worker.Worker``.
        - :attr:`CompiledProgram.build_orch_args` / ``build_call_config`` to
          assemble the simpler dispatch tuple yourself.

        ``config=RunConfig(...)`` is still consumed (and its compile-side
        knobs forwarded to ``ir.compile()``) so the returned
        ``CompiledProgram`` honours the same options as a direct
        ``kernel(*args, config=...)`` call. Runtime-side fields on the
        ``RunConfig`` (``device_id``, DFX flags, ...) do not apply here —
        they affect dispatch, not the compiled artefact.

        Subsequent calls (either :meth:`__call__` or :meth:`compile`) with the
        same specialization key hit the L1 cache and return the same
        ``CompiledProgram`` instance.

        Example::

            @pl.jit
            def my_kernel(x, w, out):
                ...

            worker = ChipWorker(config=RunConfig(platform="a2a3"))
            compiled = my_kernel.compile(sample_x, sample_w, sample_out)
            w_dev = worker.alloc_tensor(real_w.shape, real_w.dtype, init=real_w)
            h = worker.register(compiled)
            for batch in stream:
                h(batch.x, w_dev, batch.out)

        Args:
            *args: Positional arguments matching the decorated function's
                params. Tensor values are inspected for shape/dtype only;
                their contents are not read.
            **kwargs: Keyword arguments. A ``config`` keyword, if present, is
                a :class:`~pypto.runtime.runner.RunConfig`.

        Returns:
            The cached :class:`CompiledProgram` for this specialization.
        """
        compiled, _ordered_args, _run_config = self._resolve_compiled(args, kwargs)
        return compiled

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def _compile(
        self,
        tensor_meta: dict[str, TensorMeta],
        scalar_values: dict[str, int | float | bool],
        scalar_dtypes: dict[str, DataType],
        per_func_dyn: dict[int, dict[str, dict[int, DynDim]]],
        pl: Any,
        platform: str | None = None,
        **ir_compile_kwargs: Any,
    ) -> Any:
        """Specialize entry + deps into @pl.program source, parse, and compile.

        Runs the full compilation pipeline: pass pipeline + codegen via
        ``ir.compile()``.  Returns a ``CompiledProgram``
        containing the post-pass ``ir.Program`` and the generated output
        artifacts (orchestration C++, kernel MLIR).

        ``per_func_dyn`` is the per-function effective DynDim map computed in
        :meth:`_bind_args`; reused here so :func:`_resolve_dep_call_metadata`
        doesn't re-walk the dep graph on every cache miss.

        ``ir_compile_kwargs`` are forwarded verbatim to ``ir.compile()`` —
        compile-side knobs (``strategy``, ``dump_passes``, ``output_dir``,
        ``profiling``, diagnostics, ...) that the JIT caller derives from a
        ``RunConfig`` via :func:`_run_config_compile_kwargs`.
        """
        from pypto.ir.compile import compile as ir_compile  # noqa: PLC0415

        contexts = self._build_contexts(tensor_meta, scalar_values, scalar_dtypes, per_func_dyn)
        class_name = f"_jit_{self.__name__}"
        specializer = Specializer(class_name, contexts)
        source = specializer.specialize()
        rename_map = specializer.rename_map
        try:
            parsed = pl.parse(source, filename=self._diagnostic_filename, source_map=specializer.source_map)
            skip_ptoas = not _ptoas_available()
            return ir_compile(parsed, skip_ptoas=skip_ptoas, platform=platform, **ir_compile_kwargs)
        except Exception as exc:
            rewritten = _rewrite_jit_error(exc, rename_map)
            if rewritten is exc:
                raise
            raise rewritten from exc

    def _compile_to_program(
        self,
        tensor_meta: dict[str, TensorMeta],
        scalar_values: dict[str, int | float | bool],
        scalar_dtypes: dict[str, DataType],
        per_func_dyn: dict[int, dict[str, dict[int, DynDim]]],
        pl: Any,
    ) -> Any:
        """Specialize entry + deps and return the parsed ir.Program (pre-pass).

        This method is intended for testing only — it lets tests inspect and
        compare the specialized IR without running the full pass pipeline or
        requiring the Ascend toolchain.
        """
        contexts = self._build_contexts(tensor_meta, scalar_values, scalar_dtypes, per_func_dyn)
        class_name = f"_jit_{self.__name__}"
        specializer = Specializer(class_name, contexts)
        source = specializer.specialize()
        rename_map = specializer.rename_map
        try:
            return pl.parse(source, filename=self._diagnostic_filename, source_map=specializer.source_map)
        except Exception as exc:
            rewritten = _rewrite_jit_error(exc, rename_map)
            if rewritten is exc:
                raise
            raise rewritten from exc

    def _build_contexts(
        self,
        tensor_meta: dict[str, TensorMeta],
        scalar_values: dict[str, int | float | bool],
        scalar_dtypes: dict[str, DataType],
        per_func_dyn: dict[int, dict[str, dict[int, DynDim]]],
    ) -> list[SpecializeContext]:
        """Build SpecializeContext list for entry + every transitive dep.

        Walks the JIT call graph from ``_get_dep_graph()``:

        - Tensor / scalar metadata is resolved top-down (caller-first), so
          each dep is built using its actual caller's resolved metadata.
          For nested deps the caller may itself be another dep, not the
          entry. DynDim entries inside the metas propagate naturally as
          metas are forwarded into deps via ``_resolve_dep_call_metadata``.
        - Each context's ``dep_names`` is the set of JIT deps that the
          context's function *directly* calls. The body transformer uses
          this to rewrite nested ``dep(args)`` calls into the
          ``self.dep(args)`` form required by multi-function ``@pl.program``.

        The returned list is in leaf-first order (deps before their
        callers) so the generated source defines callees before callers.
        """
        deps_topo, callers_by_id, callees_by_id, _ = self._get_dep_graph()
        empty_dyn: dict[str, dict[int, DynDim]] = {}

        # Map each Python function id → its JIT ``_func_type`` so meta
        # resolution downstream can gate dep discovery on the caller's type
        # (a host orchestrator additionally admits ``orchestration`` deps).
        func_type_by_id: dict[int, str] = {id(self._func): self._func_type}
        for d in deps_topo:
            func_type_by_id[id(d._func)] = d._func_type

        # Walk caller-first to resolve each dep's metadata from its actual
        # caller's already-resolved metadata.
        resolved: dict[
            int,
            tuple[
                dict[str, TensorMeta],
                dict[str, int | float | bool],
                dict[str, DataType],
            ],
        ] = {id(self._func): (tensor_meta, scalar_values, scalar_dtypes)}

        # Walk caller-first (reverse of leaf-first topo order) so each dep's
        # caller metadata is already resolved when we get to it; collect
        # contexts caller-first, then reverse to restore leaf-first emit
        # order.
        dep_contexts: list[SpecializeContext] = []
        for dep in reversed(deps_topo):
            # For metadata resolution we use the first-recorded caller. In a
            # diamond ``entry -> {A, B} -> shared`` only one specialization
            # of ``shared`` is emitted, so the call sites in other branches
            # must agree on shapes/dtypes anyway.
            caller_func = callers_by_id[id(dep._func)][0]
            c_meta, c_sv, c_sd = resolved[id(caller_func)]
            caller_ftype = func_type_by_id.get(id(caller_func), "orchestration")
            dep_meta, dep_sv, dep_sd = _resolve_dep_call_metadata(
                dep,
                caller_func,
                c_meta,
                c_sv,
                c_sd,
                per_func_dyn.get(id(dep._func), empty_dyn),
                caller_func_type=caller_ftype,
            )
            resolved[id(dep._func)] = (dep_meta, dep_sv, dep_sd)
            dep_contexts.append(
                build_specialize_context(
                    func=dep._func,
                    func_name=dep.__name__,
                    func_type=dep._func_type,
                    level=dep._level,
                    tensor_meta=dep_meta,
                    scalar_values=dep_sv,
                    scalar_dtypes=dep_sd,
                    dep_names=callees_by_id[id(dep._func)],
                    auto_scope=dep._auto_scope,
                )
            )
        dep_contexts.reverse()

        entry_ctx = build_specialize_context(
            func=self._func,
            func_name=self.__name__,
            func_type=self._func_type,
            level=self._level,
            tensor_meta=tensor_meta,
            scalar_values=scalar_values,
            scalar_dtypes=scalar_dtypes,
            dep_names=callees_by_id[id(self._func)],
            auto_scope=self._auto_scope,
        )
        return dep_contexts + [entry_ctx]

    def compile_for_test(self, *args: Any, **kwargs: Any) -> Any:
        """Specialize, compile, and return the post-pass ir.Program for testing.

        Runs the full pass pipeline and populates ``_cache`` with a
        ``CompiledProgram`` (via ``ir.compile()``), then returns the post-pass
        ``ir.Program`` for structural equality comparison in unit tests.

        Unlike ``__call__``, this method does not execute on device.

        Args:
            *args: Positional arguments matching the decorated function's params.
            **kwargs: Keyword arguments.

        Returns:
            ``ir.Program`` after the full pass pipeline, suitable for
            ``ir.assert_structural_equal`` comparison.
        """
        import pypto.language as pl  # noqa: PLC0415
        from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415

        param_names, _, tensor_meta, scalar_values, scalar_dtypes, per_func_dyn = self._bind_args(
            args, kwargs
        )

        key = make_cache_key(
            source_hash=self._get_source_hash(),
            param_names=param_names,
            tensor_shapes={n: m.static_shape() for n, m in tensor_meta.items()},
            tensor_dtypes={n: m.dtype for n, m in tensor_meta.items()},
            dynamic_dims={(n, i) for n, m in tensor_meta.items() for i in m.dynamic_dim_indices()},
            scalar_values=scalar_values,
            platform=None,  # compile_for_test is platform-agnostic (testing only)
            strategy=OptimizationStrategy.Default,  # _compile() uses the default strategy
        )

        # Populate cache via ir.compile() (codegen included) as a best-effort
        # side effect.  Two known failure modes are both acceptable here:
        #   (1) Single-function programs with incore scopes fail at
        #       OutlineIncoreScopes (the pass only handles Opaque functions).
        #   (2) Some programs fail at ptoas due to hardware-specific constraints
        #       (e.g. tinsert loc=acc/mat mismatch on assemble kernels).
        # In both cases the cache entry is simply left empty; the actual return
        # value of this method comes from _compile_to_program() below, which
        # runs only through the pass pipeline (no codegen) and always succeeds
        # for structurally valid IR.
        if key not in self._cache:
            try:
                self._cache[key] = self._compile(tensor_meta, scalar_values, scalar_dtypes, per_func_dyn, pl)
            except Exception:
                pass

        # Return the post-pass ir.Program via the lightweight path
        # (no codegen) for structural equality comparison.
        pre_pass = self._compile_to_program(tensor_meta, scalar_values, scalar_dtypes, per_func_dyn, pl)
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        return pm.run_passes(pre_pass)

    def __repr__(self) -> str:
        return f"JITFunction({self.__name__!r}, func_type={self._func_type!r})"


# ---------------------------------------------------------------------------
# Dep auto-discovery (defined after JITFunction to avoid forward reference)
# ---------------------------------------------------------------------------


def _discover_deps(func: Any, caller_func_type: str = "orchestration") -> list[JITFunction]:
    """Discover JIT dep functions called by ``func``.

    Scans the function's AST for bare function calls, then resolves each name
    against both module globals and closure variables (for deps defined in an
    enclosing scope, e.g. inside a test method or a factory function).

    The set of admissible dep ``_func_type`` values is gated by the caller:

    - A regular entry (``caller_func_type`` is ``'orchestration'`` or any
      ``incore`` / ``inline`` / ``opaque`` sub-function recursing into its
      own deps) admits ``incore``, ``inline``, ``opaque`` sub-functions.
    - A host orchestrator (``caller_func_type == 'host'``) additionally
      admits ``orchestration`` deps — the chip-level orchestrator a host
      entry dispatches with ``self.chip_orch(..., device=r)``. Plain
      ``@pl.jit`` entries never discover other ``@pl.jit`` entries; that
      would conflate two top-level kernels into a single program.

    Only top-level (non-method) calls are considered. The returned list
    preserves the order in which deps first appear in the source.
    """
    func_def = _get_func_def(func)

    called_names = _collect_all_called_names(func_def)

    # Module-level globals
    func_globals = getattr(func, "__globals__", {})

    # Closure variables (covers deps defined in an enclosing scope)
    closure_vars: dict[str, Any] = {}
    co_freevars = getattr(getattr(func, "__code__", None), "co_freevars", ())
    closure = getattr(func, "__closure__", None) or ()
    for name, cell in zip(co_freevars, closure):
        try:
            closure_vars[name] = cell.cell_contents
        except ValueError:
            pass

    all_vars = {**func_globals, **closure_vars}

    allowed_dep_types: set[str] = {"incore", "inline", "opaque"}
    if caller_func_type == "host":
        allowed_dep_types.add("orchestration")

    deps: list[JITFunction] = []
    seen: set[str] = set()
    for name in called_names:
        obj = all_vars.get(name)
        if isinstance(obj, JITFunction) and obj._func_type in allowed_dep_types and name not in seen:
            deps.append(obj)
            seen.add(name)
    return deps


# ---------------------------------------------------------------------------
# _JITDecorator — supports @jit, @jit.incore, @jit.incore(level=...)
# ---------------------------------------------------------------------------

# Sentinel distinguishing "auto_scope= was not passed" from an explicit value.
# Lets sub-decorators that don't support the kwarg reject ANY explicit
# auto_scope= (including auto_scope=True), not just non-True values.
_AUTO_SCOPE_UNSET: Any = object()


class _SubFunctionDecorator:
    """Sub-decorator factory for ``@jit.<kind>`` (host / incore / inline / opaque).

    Every kind supports both ``@jit.kind`` (bare) and ``@jit.kind()`` (parens).
    Only ``incore`` honors a ``level=`` kwarg; passing it to other kinds raises.
    Only ``host`` and ``inline`` honor an ``auto_scope=`` kwarg — ``inline``
    because its body is spliced into the caller, so hand-placed scopes land
    there; ``incore``/``opaque`` outline into separate kernels and reject it.

    See `_JITDecorator` for the kind semantics:
      - ``host``    → HOST Orchestrator entry (specialized to
        ``@pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)``).
      - ``incore``  → ``FunctionType.InCore`` (separate IR function, ``level`` selectable).
      - ``inline``  → ``FunctionType.Inline`` (spliced at every call site by the
        ``InlineFunctions`` IR pass).
      - ``opaque``  → ``FunctionType.Opaque`` (separate IR function; may wrap
        orchestration loops and ``pl.at`` scopes).
    """

    def __init__(self, func_type: str, *, allow_level: bool, allow_auto_scope: bool = False) -> None:
        self._func_type = func_type
        self._allow_level = allow_level
        self._allow_auto_scope = allow_auto_scope

    def __call__(self, func: Any = None, *, level: Any = None, auto_scope: Any = _AUTO_SCOPE_UNSET) -> Any:
        if level is not None and not self._allow_level:
            raise TypeError(f"@pl.jit.{self._func_type} does not accept a level= argument")
        if auto_scope is not _AUTO_SCOPE_UNSET and not self._allow_auto_scope:
            raise TypeError(
                f"@pl.jit.{self._func_type} does not accept an auto_scope= argument "
                "(auto_scope is only meaningful for the Orchestration entry, the "
                "HOST orchestrator, and inline sub-functions)"
            )
        resolved_auto_scope = True if auto_scope is _AUTO_SCOPE_UNSET else auto_scope
        if func is None:
            return lambda f: JITFunction(
                f, func_type=self._func_type, level=level, auto_scope=resolved_auto_scope
            )
        return JITFunction(func, func_type=self._func_type, level=None, auto_scope=resolved_auto_scope)


class _JITDecorator:
    """The ``pl.jit`` object.

    Supports::

        @pl.jit                               # entry-point (Orchestration)
        @pl.jit.host                          # entry-point (HOST Orchestrator)
        @pl.jit.incore                        # InCore sub-function
        @pl.jit.incore(level=pl.Level.AIC)   # InCore with explicit level
        @pl.jit.inline                        # Inline sub-function (spliced at call site)
        @pl.jit.opaque                        # Opaque sub-function (separate IR function)

    ``host`` is the L3+ entry variant: it authors the per-rank dispatch loop
    (``for r in pl.range(pld.world_size()): chip_orch(..., device=r)``) and
    window-buffer allocation that today only ``@pl.function(level=HOST,
    role=Orchestrator)`` inside ``@pl.program`` could express. It is keyed
    off ``_func_type='host'`` and specializes into
    ``@pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)``.
    """

    def __init__(self) -> None:
        self.host = _SubFunctionDecorator("host", allow_level=False, allow_auto_scope=True)
        self.incore = _SubFunctionDecorator("incore", allow_level=True)
        self.inline = _SubFunctionDecorator("inline", allow_level=False, allow_auto_scope=True)
        self.opaque = _SubFunctionDecorator("opaque", allow_level=False)

    def __call__(self, func: Any = None, *, auto_scope: bool = True) -> Any:
        """Decorate an entry-point JIT function (Orchestration).

        Supports both the bare ``@pl.jit`` form and the parenthesized
        ``@pl.jit(auto_scope=False)`` form. Setting ``auto_scope=False`` opts
        out of compiler-inserted AUTO runtime scopes so the body can place
        them by hand with ``with pl.scope()``.
        """
        if func is None:
            return lambda f: JITFunction(f, func_type="orchestration", level=None, auto_scope=auto_scope)
        return JITFunction(func, func_type="orchestration", level=None, auto_scope=auto_scope)


# Singleton decorator object exposed as ``pl.jit``
jit = _JITDecorator()


__all__ = ["JITFunction", "jit"]
