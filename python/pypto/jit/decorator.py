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
import inspect
import os
import re
import shutil
import textwrap
from typing import Any

from pypto.pypto_core import DataType

from .cache import CacheKey, compute_source_hash, make_cache_key
from .specializer import (
    SpecializeContext,
    Specializer,
    TensorMeta,
    _collect_dynamic_dims,
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


def _extract_tensor_meta(tensor: Any) -> TensorMeta:
    """Extract TensorMeta from a torch.Tensor."""
    return TensorMeta(
        shape=tuple(int(d) for d in tensor.shape),
        dtype=_torch_dtype_to_pypto(tensor.dtype),
    )


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _get_func_def(func: Any) -> ast.FunctionDef:
    """Parse func source and return its FunctionDef node.

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


def _scan_dynamic_dims(func: Any, param_names: list[str]) -> set[tuple[str, int]]:
    """Statically scan func for bind_dynamic calls and return dynamic (param, dim) pairs."""
    func_def = _get_func_def(func)
    return _collect_dynamic_dims(func_def, set(param_names))


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


def _extract_create_tensor_metas(func: Any) -> dict[str, TensorMeta]:
    """Scan func's AST for ``var = pl.create_tensor([...], dtype=pl.XXX)`` and return TensorMeta per var.

    Each shape element may be either a literal int (``42``) or a
    ``Name`` referencing an int in ``func.__globals__`` — covering the common
    case of model code that imports shape constants from a config module.
    Each dtype must be a ``pl.<name>`` attribute. Anything that cannot be
    statically resolved is skipped silently.
    """
    func_def = _get_func_def(func)
    result: dict[str, TensorMeta] = {}
    dtype_map = _get_pl_dtype_map()
    func_globals = getattr(func, "__globals__", {})

    def _resolve_int(elt: ast.expr) -> int | None:
        """Resolve ``elt`` to a Python int. Returns None if not statically resolvable.

        Handles literal ints, ``Name`` refs to module-level int globals, and
        simple integer arithmetic (``+``, ``-``, ``*``, ``//``, ``%``, ``**``)
        over any combination of those — covering shape expressions like
        ``BATCH * TOTAL_Q_GROUPS * Q_HEAD_PAD`` and ``2 ** N``. Also accepts
        leading unary ``+`` / ``-``.
        """
        if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
            return elt.value
        if isinstance(elt, ast.Name):
            value = func_globals.get(elt.id)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            return None
        if isinstance(elt, ast.BinOp):
            lhs = _resolve_int(elt.left)
            rhs = _resolve_int(elt.right)
            if lhs is None or rhs is None:
                return None
            op = elt.op
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
            # Pow: only int exponents to keep the result an int.
            if isinstance(op, ast.Pow) and rhs >= 0:
                return lhs**rhs
            return None
        if isinstance(elt, ast.UnaryOp):
            v = _resolve_int(elt.operand)
            if v is None:
                return None
            if isinstance(elt.op, ast.USub):
                return -v
            if isinstance(elt.op, ast.UAdd):
                return v
            return None
        return None

    for node in ast.walk(func_def):
        # Look for: var = pl.create_tensor([...], dtype=pl.XXX)
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        var_name = node.targets[0].id
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        fn = call.func
        if not (
            isinstance(fn, ast.Attribute) and fn.attr == "create_tensor" and isinstance(fn.value, ast.Name)
        ):
            continue
        # Extract shape: first positional arg must be a list whose elements are
        # literal ints or globals-resolved ints.
        if not call.args or not isinstance(call.args[0], ast.List):
            continue
        shape_node = call.args[0]
        resolved_shape: list[int] = []
        skip = False
        for elt in shape_node.elts:
            v = _resolve_int(elt)
            if v is None:
                skip = True
                break
            resolved_shape.append(v)
        if skip:
            continue
        shape = tuple(resolved_shape)
        # Extract dtype from keyword arg dtype=pl.XXX
        dtype_val = None
        for kw in call.keywords:
            if (
                kw.arg == "dtype"
                and isinstance(kw.value, ast.Attribute)
                and isinstance(kw.value.value, ast.Name)
            ):
                dtype_val = dtype_map.get(kw.value.attr)
                break
        if dtype_val is None:
            continue
        result[var_name] = TensorMeta(shape=shape, dtype=dtype_val)

    return result


def _extract_call_args_for_dep(entry_func: Any, dep_name: str) -> list[tuple[str | None, str | None]] | None:
    """Find the argument names passed to ``dep_name`` in ``entry_func``'s body.

    Returns a unified list of ``(param_name, arg_name)`` pairs:

    - ``param_name`` is ``None`` for a positional argument (the consumer
      pairs it with the dep's parameter list by index) and the keyword
      name for a keyword argument.
    - ``arg_name`` is the caller-side variable name, or ``None`` for
      non-``Name`` expressions (literals, attribute access, computed
      expressions, …).

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
        result: list[tuple[str | None, str | None]] = [
            (None, arg.id if isinstance(arg, ast.Name) else None) for arg in node.args
        ]
        result.extend(
            (kw.arg, kw.value.id if isinstance(kw.value, ast.Name) else None)
            for kw in node.keywords
            if kw.arg is not None  # skip **kwargs splats
        )
        return result
    return None


def _build_param_mapping(
    dep_param_names: list[str],
    call_args: list[tuple[str | None, str | None]],
) -> dict[str, str | None]:
    """Map dep parameter name → caller argument name from call-site args.

    ``call_args`` is the unified form returned by
    ``_extract_call_args_for_dep``: a list of ``(param_name, arg_name)``
    pairs where ``param_name is None`` marks a positional argument (paired
    with ``dep_param_names`` by index) and a string is a keyword name.
    Mixed positional + keyword call sites collapse to the same dict.
    """
    mapping: dict[str, str | None] = {}
    pos_idx = 0
    for param_name, arg_name in call_args:
        if param_name is None:
            if pos_idx < len(dep_param_names):
                mapping[dep_param_names[pos_idx]] = arg_name
            pos_idx += 1
        else:
            mapping[param_name] = arg_name
    return mapping


def _compute_per_func_dynamic_dims(
    entry_func: Any,
    deps: list[Any],
    entry_dynamic_dims: set[tuple[str, int]],
    callers_by_dep_id: dict[int, list[Any]],
    dep_dyn_cache: dict[int, set[tuple[str, int]]],
    call_args_cache: dict[tuple[int, str], list[tuple[str | None, str | None]] | None],
) -> dict[int, set[tuple[str, int]]]:
    """Resolve dynamic dims for every reachable JIT function.

    Each dep's own ``bind_dynamic`` calls seed its set, then leaf-first
    propagation maps every dep param marked dynamic to the corresponding
    arg at *every* recorded caller, so dyn dims hop through every
    intermediate caller until reaching the entry. ``deps`` must be in
    leaf-first topological order (as returned by ``_get_dep_graph``);
    a single pass then suffices, even for diamonds where a shared dep is
    reached from multiple branches.

    ``dep_dyn_cache`` and ``call_args_cache`` are the per-graph AST
    memoisations from ``_get_dep_graph`` — passed in so this function
    avoids re-walking ASTs on every JIT invocation.

    Returns a dict keyed by ``id(py_func)`` so callers can pull either the
    entry's set (cache key) or any individual dep's set (specialization).
    """
    per_func: dict[int, set[tuple[str, int]]] = {id(entry_func): set(entry_dynamic_dims)}
    for dep in deps:
        per_func[id(dep._func)] = set(dep_dyn_cache[id(dep._func)])
    for dep in deps:
        dep_dyn = per_func[id(dep._func)]
        for caller_func in callers_by_dep_id.get(id(dep._func), ()):
            call_args = call_args_cache[(id(caller_func), dep.__name__)]
            if call_args is None:
                continue
            mapping = _build_param_mapping(dep._param_names(), call_args)
            caller_set = per_func[id(caller_func)]
            for dep_param, dim_idx in dep_dyn:
                caller_arg = mapping.get(dep_param)
                if caller_arg is not None:
                    caller_set.add((caller_arg, dim_idx))
    return per_func


# ---------------------------------------------------------------------------
# DynVar binding table
# ---------------------------------------------------------------------------


def _build_dynvar_bindings(contexts: list[SpecializeContext]) -> tuple[dict[str, str], dict[str, str]]:
    """Build dynvar_bindings dict for the Specializer.

    Returns:
        (bindings, literals) where:
        - bindings: maps "<param>__<dim_idx>" → DynVar Python variable name.
        - literals: maps DynVar Python variable name → string literal passed to
          pl.dynamic().  Used to emit ``varname = pl.dynamic("literal")`` at
          module level.
    """
    bindings: dict[str, str] = {}
    literals: dict[str, str] = {}

    for ctx in contexts:
        if not ctx.dynamic_dims:
            continue
        func_def = None
        try:
            src = textwrap.dedent(ctx.source)
            tree = ast.parse(src)
            func_def = next(
                (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == ctx.func_name),
                None,
            )
        except SyntaxError:
            continue
        if func_def is None:
            continue

        # Collect dynvar name → literal from pl.dynamic("...") assignments
        from .specializer import _collect_dynvar_names  # noqa: PLC0415

        dyn_literals = _collect_dynvar_names(func_def)

        # Match bind_dynamic(dim, dynvar_varname) calls
        for node in ast.walk(func_def):
            if not isinstance(node, ast.Expr):
                continue
            call = node.value
            if not isinstance(call, ast.Call):
                continue
            fn = call.func
            if not (
                isinstance(fn, ast.Attribute) and fn.attr == "bind_dynamic" and isinstance(fn.value, ast.Name)
            ):
                continue
            if len(call.args) < 2:
                continue
            dim_node, dv_node = call.args[0], call.args[1]
            if not (isinstance(dim_node, ast.Constant) and isinstance(dv_node, ast.Name)):
                continue
            key = f"{fn.value.id}__{dim_node.value}"
            var_name = dv_node.id
            bindings[key] = var_name
            # Store literal: prefer the pl.dynamic("M") literal, fall back to var name
            literals[var_name] = dyn_literals.get(var_name, var_name)

    return bindings, literals


def _backfill_dynvar_bindings(
    deps: list[Any],
    callers_by_dep_id: dict[int, list[Any]],
    bindings: dict[str, str],
    literals: dict[str, str],
    call_args_cache: dict[tuple[int, str], list[tuple[str | None, str | None]] | None],
) -> None:
    """Cascade dynvar bindings up through the JIT call graph.

    When a dep declares ``param.bind_dynamic(dim, dynvar)`` but its caller
    (which may itself be another dep) doesn't, the caller's arg passed to
    that dep param must inherit the same binding so the generated tensor
    annotation references the right DynVar. ``deps`` is leaf-first, so a
    single pass cascades each binding from leaf -> mid -> ... -> entry,
    visiting *every* recorded caller of each dep so diamond branches all
    receive the same binding.

    ``call_args_cache`` is the per-graph AST memoisation from
    ``_get_dep_graph`` — avoids re-walking caller bodies on every JIT
    invocation.

    Mutates ``bindings`` and ``literals`` in place.
    """
    for dep in deps:
        for caller_func in callers_by_dep_id.get(id(dep._func), ()):
            call_args = call_args_cache[(id(caller_func), dep.__name__)]
            if call_args is None:
                continue
            mapping = _build_param_mapping(dep._param_names(), call_args)
            for dep_param, caller_arg in mapping.items():
                if caller_arg is None:
                    continue
                prefix = f"{dep_param}__"
                for key, var_name in list(bindings.items()):
                    if key.startswith(prefix):
                        dim_idx = key[len(prefix) :]
                        caller_key = f"{caller_arg}__{dim_idx}"
                        if caller_key not in bindings:
                            bindings[caller_key] = var_name
                            if var_name not in literals:
                                literals[var_name] = var_name


def _resolve_dep_call_metadata(
    dep: JITFunction,
    caller_func: Any,
    caller_tensor_meta: dict[str, TensorMeta],
    caller_scalar_values: dict[str, int | float | bool],
    caller_scalar_dtypes: dict[str, DataType],
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
    Intermediate tensors created via ``pl.create_tensor`` in the caller are
    folded into the metadata pool. Falls back to name-based matching when
    call-site extraction fails.
    """
    dep_param_names = dep._param_names()
    call_args = _extract_call_args_for_dep(caller_func, dep.__name__)
    intermediate_metas = _extract_create_tensor_metas(caller_func)
    all_tensor_meta = {**intermediate_metas, **caller_tensor_meta}

    dep_tensor_meta: dict[str, TensorMeta] = {}
    dep_scalar_values: dict[str, int | float | bool] = {}
    dep_scalar_dtypes: dict[str, DataType] = {}

    if call_args is not None:
        for dep_param, caller_arg in _build_param_mapping(dep_param_names, call_args).items():
            if caller_arg is None:
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

    return dep_tensor_meta, dep_scalar_values, dep_scalar_dtypes


# ---------------------------------------------------------------------------


class JITFunction:
    """A JIT-compiled function with shape specialization and caching.

    Created by the ``@jit`` or ``@jit.incore`` decorators.

    Attributes:
        _func: Original Python function.
        _func_type: 'orchestration' | 'incore' | 'inline' | 'opaque'.
        _level: pl.Level or None.
        _dep_graph: Lazily-computed transitive JIT dep graph rooted here —
            ``(deps_topo, callers_by_dep_id, callees_by_func_id,
            dep_dyn_cache, call_args_cache)``.  ``None`` until first
            ``_get_dep_graph()`` call.  See that method for the tuple's
            structure.
        _cache: L1 in-memory cache: CacheKey → CompiledProgram (post-pass ir.Program wrapped).
        _source_hash: Lazily-computed hash of func source + all dep sources.
    """

    def __init__(
        self,
        func: Any,
        func_type: str | None = None,
        level: Any = None,
    ) -> None:
        self._func = func
        self._func_type = func_type or "orchestration"
        self._level = level
        self._dep_graph: (
            tuple[
                list[JITFunction],
                dict[int, list[Any]],
                dict[int, list[str]],
                dict[int, set[tuple[str, int]]],
                dict[tuple[int, str], list[tuple[str | None, str | None]] | None],
            ]
            | None
        ) = None
        self._cache: dict[CacheKey, Any] = {}  # CacheKey → CompiledProgram
        self._source_hash: str | None = None

        # Preserve function metadata
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__module__ = func.__module__

    # ------------------------------------------------------------------
    # Lazy dep discovery
    # ------------------------------------------------------------------

    def _get_dep_graph(
        self,
    ) -> tuple[
        list[JITFunction],
        dict[int, list[Any]],
        dict[int, list[str]],
        dict[int, set[tuple[str, int]]],
        dict[tuple[int, str], list[tuple[str | None, str | None]] | None],
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
          does not appear as a key. In a diamond ``entry -> {A, B} -> shared``
          both ``A`` and ``B`` appear as callers of ``shared`` so dynamic-dim
          and dynvar propagation visits every branch.

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
        - ``dep_dyn_cache``: ``id(dep._func)`` → set of ``(param, dim)``
          pairs marked dynamic in that dep's body via ``bind_dynamic``.
          Cached here so ``_compute_per_func_dynamic_dims`` doesn't
          re-parse dep ASTs on every JIT invocation.
        - ``call_args_cache``: ``(id(caller_func), dep_name)`` → unified
          call-site arg list (see ``_extract_call_args_for_dep``) or
          ``None`` if the call site isn't found. Cached so propagation
          and dynvar back-fill don't re-walk caller ASTs on every call.
        """
        if self._dep_graph is None:
            deps_topo: list[JITFunction] = []
            seen: set[int] = set()
            callers_by_dep_id: dict[int, list[Any]] = {}
            callees_by_func_id: dict[int, list[str]] = {}
            dep_dyn_cache: dict[int, set[tuple[str, int]]] = {}
            call_args_cache: dict[tuple[int, str], list[tuple[str | None, str | None]] | None] = {}

            def visit(func: Any) -> None:
                direct = _discover_deps(func)
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
                    dep_dyn_cache[id(dep._func)] = _scan_dynamic_dims(dep._func, dep._param_names())
                    visit(dep._func)
                    deps_topo.append(dep)

            visit(self._func)
            self._dep_graph = (
                deps_topo,
                callers_by_dep_id,
                callees_by_func_id,
                dep_dyn_cache,
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
        dict[int, set[tuple[str, int]]],
    ]:
        """Bind *args/**kwargs to param names and classify into tensor/scalar metadata.

        Returns:
            (param_names, arguments, tensor_meta, scalar_values,
            scalar_dtypes, per_func_dyn).  ``per_func_dyn`` is the full
            per-function dynamic-dim map computed via the cached dep
            graph; the entry's set drives the cache key, and the same
            map is reused inside ``_compile`` to avoid re-walking the
            graph.
        """
        param_names = self._param_names()
        sig = inspect.signature(self._func)
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
        except TypeError as e:
            raise TypeError(f"@pl.jit function '{self.__name__}': {e}") from e

        arguments = dict(bound.arguments)

        tensor_meta: dict[str, TensorMeta] = {}
        scalar_values: dict[str, int | float | bool] = {}
        scalar_dtypes: dict[str, DataType] = {}

        for name, value in arguments.items():
            if _is_tensor(value):
                tensor_meta[name] = _extract_tensor_meta(value)
            elif isinstance(value, (int, float, bool)):
                scalar_values[name] = value

        deps, callers_by_id, _, dep_dyn_cache, call_args_cache = self._get_dep_graph()
        entry_dynamic_dims = _scan_dynamic_dims(self._func, param_names)
        per_func_dyn = _compute_per_func_dynamic_dims(
            self._func, deps, entry_dynamic_dims, callers_by_id, dep_dyn_cache, call_args_cache
        )

        return param_names, arguments, tensor_meta, scalar_values, scalar_dtypes, per_func_dyn

    # ------------------------------------------------------------------
    # Call
    # ------------------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Specialize, compile (or serve from cache), and execute on device.

        On the first call for a given shape/dtype combination the function is
        specialized into ``@pl.program`` source, parsed, and compiled via
        ``ir.compile()`` (passes + codegen).  The resulting ``CompiledProgram``
        is stored in the L1 in-memory cache so subsequent calls with the same
        specialization key skip compilation entirely.

        The compiled kernel is then executed on the NPU device with the given
        torch tensor arguments (Triton-like API).

        Args:
            *args: Positional arguments matching the decorated function's params.
            **kwargs: Keyword arguments.

        Returns:
            ``None`` for in-place calls (output tensors modified on device),
            or ``torch.Tensor`` / ``tuple[torch.Tensor, ...]`` for return-style
            calls.
        """
        import pypto.language as pl  # noqa: PLC0415

        # Extract RunConfig before binding — it is not a JIT function parameter
        # but is forwarded directly to CompiledProgram.__call__().
        run_config = kwargs.pop("config", None)

        param_names, arguments, tensor_meta, scalar_values, scalar_dtypes, per_func_dyn = self._bind_args(
            args, kwargs
        )

        # Build cache key. Platform is included so artifacts compiled for
        # different targets never collide in the same cache.
        platform = run_config.platform if run_config is not None else None
        entry_dyn = per_func_dyn[id(self._func)]
        key = make_cache_key(
            source_hash=self._get_source_hash(),
            param_names=param_names,
            tensor_shapes={n: m.shape for n, m in tensor_meta.items()},
            tensor_dtypes={n: m.dtype for n, m in tensor_meta.items()},
            dynamic_dims=entry_dyn,
            scalar_values=scalar_values,
            platform=platform,
        )

        # L1 cache lookup
        if key not in self._cache:
            self._cache[key] = self._compile(
                tensor_meta, scalar_values, scalar_dtypes, per_func_dyn, pl, platform=platform
            )

        # Execute the compiled kernel on device.
        # Use bound.arguments (in signature order) so keyword-style calls
        # like kernel(a=x, b=y) are routed correctly regardless of how the
        # caller passed them.
        compiled = self._cache[key]
        ordered_args = [arguments[n] for n in param_names if n in arguments]
        if run_config is not None:
            return compiled(*ordered_args, config=run_config)
        return compiled(*ordered_args)

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def _compile(
        self,
        tensor_meta: dict[str, TensorMeta],
        scalar_values: dict[str, int | float | bool],
        scalar_dtypes: dict[str, DataType],
        per_func_dyn: dict[int, set[tuple[str, int]]],
        pl: Any,
        platform: str | None = None,
    ) -> Any:
        """Specialize entry + deps into @pl.program source, parse, and compile.

        Runs the full compilation pipeline: pass pipeline + codegen via
        ``ir.compile()``.  Returns a ``CompiledProgram``
        containing the post-pass ``ir.Program`` and the generated output
        artifacts (orchestration C++, kernel MLIR).

        ``per_func_dyn`` is the precomputed dynamic-dim map from
        ``_bind_args``.  Reusing it here avoids walking the dep graph
        twice on every cache miss.
        """
        from pypto.ir.compile import compile as ir_compile  # noqa: PLC0415

        contexts = self._build_contexts(tensor_meta, scalar_values, scalar_dtypes, per_func_dyn)
        dynvar_bindings, dynvar_literals = _build_dynvar_bindings(contexts)
        deps, callers_by_id, _, _, call_args_cache = self._get_dep_graph()
        _backfill_dynvar_bindings(deps, callers_by_id, dynvar_bindings, dynvar_literals, call_args_cache)
        class_name = f"_jit_{self.__name__}"
        specializer = Specializer(class_name, contexts, dynvar_bindings, dynvar_literals)
        source = specializer.specialize()
        rename_map = specializer.rename_map
        try:
            parsed = pl.parse(source)
            skip_ptoas = not _ptoas_available()
            return ir_compile(parsed, skip_ptoas=skip_ptoas, platform=platform)
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
        per_func_dyn: dict[int, set[tuple[str, int]]],
        pl: Any,
    ) -> Any:
        """Specialize entry + deps and return the parsed ir.Program (pre-pass).

        This method is intended for testing only — it lets tests inspect and
        compare the specialized IR without running the full pass pipeline or
        requiring the Ascend toolchain.
        """
        contexts = self._build_contexts(tensor_meta, scalar_values, scalar_dtypes, per_func_dyn)
        dynvar_bindings, dynvar_literals = _build_dynvar_bindings(contexts)
        deps, callers_by_id, _, _, call_args_cache = self._get_dep_graph()
        _backfill_dynvar_bindings(deps, callers_by_id, dynvar_bindings, dynvar_literals, call_args_cache)
        class_name = f"_jit_{self.__name__}"
        specializer = Specializer(class_name, contexts, dynvar_bindings, dynvar_literals)
        source = specializer.specialize()
        rename_map = specializer.rename_map
        try:
            return pl.parse(source)
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
        per_func_dyn: dict[int, set[tuple[str, int]]],
    ) -> list[SpecializeContext]:
        """Build SpecializeContext list for entry + every transitive dep.

        Walks the JIT call graph from ``_get_dep_graph()``:

        - Tensor / scalar metadata is resolved top-down (caller-first), so
          each dep is built using its actual caller's resolved metadata.
          For nested deps the caller may itself be another dep, not the
          entry.
        - Dynamic dims are taken from ``per_func_dyn`` (computed once in
          ``_bind_args`` via ``_compute_per_func_dynamic_dims``) so we
          don't re-walk the dep graph here.
        - Each context's ``dep_names`` is the set of JIT deps that the
          context's function *directly* calls. The body transformer uses
          this to rewrite nested ``dep(args)`` calls into the
          ``self.dep(args)`` form required by multi-function ``@pl.program``.

        The returned list is in leaf-first order (deps before their
        callers) so the generated source defines callees before callers.
        """
        deps_topo, callers_by_id, callees_by_id, _, _ = self._get_dep_graph()

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
            dep_meta, dep_sv, dep_sd = _resolve_dep_call_metadata(dep, caller_func, c_meta, c_sv, c_sd)
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
                    dynamic_dims=per_func_dyn[id(dep._func)],
                    dep_names=callees_by_id[id(dep._func)],
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
            dynamic_dims=per_func_dyn[id(self._func)],
            dep_names=callees_by_id[id(self._func)],
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
            tensor_shapes={n: m.shape for n, m in tensor_meta.items()},
            tensor_dtypes={n: m.dtype for n, m in tensor_meta.items()},
            dynamic_dims=per_func_dyn[id(self._func)],
            scalar_values=scalar_values,
            platform=None,  # compile_for_test is platform-agnostic (testing only)
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


def _discover_deps(func: Any) -> list[JITFunction]:
    """Discover @pl.jit.incore JITFunctions called by func.

    Scans the function's AST for bare function calls, then resolves each name
    against both module globals and closure variables (for deps defined in an
    enclosing scope, e.g. inside a test method or a factory function).

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

    deps: list[JITFunction] = []
    seen: set[str] = set()
    for name in called_names:
        obj = all_vars.get(name)
        if (
            isinstance(obj, JITFunction)
            and obj._func_type in ("incore", "inline", "opaque")
            and name not in seen
        ):
            deps.append(obj)
            seen.add(name)
    return deps


# ---------------------------------------------------------------------------
# _JITDecorator — supports @jit, @jit.incore, @jit.incore(level=...)
# ---------------------------------------------------------------------------


class _SubFunctionDecorator:
    """Sub-decorator factory for ``@jit.<kind>`` (incore / inline / opaque).

    Every kind supports both ``@jit.kind`` (bare) and ``@jit.kind()`` (parens).
    Only ``incore`` honors a ``level=`` kwarg; passing it to other kinds raises.

    See `_JITDecorator` for the kind semantics:
      - ``incore``  → ``FunctionType.InCore`` (separate IR function, ``level`` selectable).
      - ``inline``  → ``FunctionType.Inline`` (spliced at every call site by the
        ``InlineFunctions`` IR pass).
      - ``opaque``  → ``FunctionType.Opaque`` (separate IR function; may wrap
        orchestration loops and ``pl.at`` scopes).
    """

    def __init__(self, func_type: str, *, allow_level: bool) -> None:
        self._func_type = func_type
        self._allow_level = allow_level

    def __call__(self, func: Any = None, *, level: Any = None) -> Any:
        if level is not None and not self._allow_level:
            raise TypeError(f"@pl.jit.{self._func_type} does not accept a level= argument")
        if func is None:
            return lambda f: JITFunction(f, func_type=self._func_type, level=level)
        return JITFunction(func, func_type=self._func_type, level=None)


class _JITDecorator:
    """The ``pl.jit`` object.

    Supports::

        @pl.jit                               # entry-point (Orchestration)
        @pl.jit.incore                        # InCore sub-function
        @pl.jit.incore(level=pl.Level.AIC)   # InCore with explicit level
        @pl.jit.inline                        # Inline sub-function (spliced at call site)
        @pl.jit.opaque                        # Opaque sub-function (separate IR function)
    """

    def __init__(self) -> None:
        self.incore = _SubFunctionDecorator("incore", allow_level=True)
        self.inline = _SubFunctionDecorator("inline", allow_level=False)
        self.opaque = _SubFunctionDecorator("opaque", allow_level=False)

    def __call__(self, func: Any) -> JITFunction:
        """Decorate an entry-point JIT function (Orchestration)."""
        return JITFunction(func, func_type="orchestration", level=None)


# Singleton decorator object exposed as ``pl.jit``
jit = _JITDecorator()


__all__ = ["JITFunction", "jit"]
