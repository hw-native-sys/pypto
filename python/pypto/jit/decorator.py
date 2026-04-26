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

    # Multi-function: @jit entry + one or more @jit.incore sub-functions
    @pl.jit.incore
    def sub_kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
        ...

    @pl.jit.incore(level=pl.Level.AIC)
    def aic_kernel(a: pl.Tensor, c: pl.Out[pl.Tensor]):
        ...

    @pl.jit
    def entry(a: pl.Tensor, c: pl.Out[pl.Tensor]):
        c = sub_kernel(a, c)   # dep discovered automatically
        return c

JITFunction.__call__ flow
-------------------------
1. Lazily discover @pl.jit.incore deps from entry function's globals (once).
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

    Only handles literal integer shapes and pl.XXX dtype attributes. Skips any
    assignment that cannot be statically resolved.
    """
    func_def = _get_func_def(func)
    result: dict[str, TensorMeta] = {}
    dtype_map = _get_pl_dtype_map()

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
        # Extract shape: first positional arg must be a list of int constants
        if not call.args or not isinstance(call.args[0], ast.List):
            continue
        shape_node = call.args[0]
        if not all(isinstance(e, ast.Constant) and isinstance(e.value, int) for e in shape_node.elts):
            continue
        shape = tuple(
            e.value for e in shape_node.elts if isinstance(e, ast.Constant) and isinstance(e.value, int)
        )
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


def _extract_call_args_for_dep(
    entry_func: Any, dep_name: str
) -> list[str | None] | list[tuple[str | None, str | None]] | None:
    """Find the argument names passed to dep_name in entry_func's body.

    Handles both positional and keyword calls:
    - Positional: ``dep(a, b)`` → ``["a", "b"]``
    - Keyword: ``dep(a=a, c=out)`` → ``{"a": "a", "c": "out"}`` stored as
      positional list via dep's parameter order (returned as-is from keyword
      keys/values here; caller zip handles ordering).

    Returns a list of argument name strings (None for non-name args like
    literals), or None if the call is not found.

    Only the first call site is examined.
    """
    func_def = _get_func_def(entry_func)
    for node in ast.walk(func_def):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == dep_name):
            continue
        # Positional args present: use them directly
        if node.args:
            return [arg.id if isinstance(arg, ast.Name) else None for arg in node.args]
        # Keyword-only call: map dep param name → entry arg name
        if node.keywords:
            # Return as a dict encoded in a special sentinel list is complex;
            # instead return a list of (kw.arg, entry_name) pairs via a dict.
            # We signal keyword mode by returning a dict-like list of 2-tuples.
            # Caller (_build_dep_context) handles both forms.
            return [
                (kw.arg, kw.value.id if isinstance(kw.value, ast.Name) else None)
                for kw in node.keywords
                if kw.arg is not None
            ]
        # Call with no args at all
        return []
    return None


def _propagate_dynamic_dims_from_deps(
    entry_func: Any,
    deps: list[Any],
    entry_dynamic_dims: set[tuple[str, int]],
) -> set[tuple[str, int]]:
    """Add to entry_dynamic_dims any dynamic dims that flow in from dep call sites.

    When a dep marks one of its params as dynamic (via bind_dynamic), and that
    dep param is mapped to an entry param at the call site, the corresponding
    entry param/dim pair should also be dynamic.

    Args:
        entry_func: The entry JITFunction's underlying function.
        deps: List of dep JITFunction objects.
        entry_dynamic_dims: Dynamic dims already found in the entry function.

    Returns:
        Augmented set of (param_name, dim_index) dynamic dim pairs for the entry.
    """
    result = set(entry_dynamic_dims)
    for dep in deps:
        dep_param_names = dep._param_names()
        dep_dynamic_dims = _scan_dynamic_dims(dep._func, dep_param_names)
        if not dep_dynamic_dims:
            continue
        call_args = _extract_call_args_for_dep(entry_func, dep.__name__)
        if call_args is None:
            continue
        if call_args and isinstance(call_args[0], tuple):
            # Keyword mode: list of (dep_param, entry_arg) pairs
            dep_param_to_entry: dict[str, str | None] = {
                dep_param: entry_arg for dep_param, entry_arg in call_args
            }
        else:
            # Positional mode
            dep_param_to_entry = {
                dep_param: entry_arg for dep_param, entry_arg in zip(dep_param_names, call_args)
            }
        for dep_param, dim_idx in dep_dynamic_dims:
            entry_arg = dep_param_to_entry.get(dep_param)
            if entry_arg is not None:
                result.add((entry_arg, dim_idx))
    return result


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


def _backfill_entry_dynvar_bindings(
    entry_func: Any,
    deps: list[Any],
    bindings: dict[str, str],
    literals: dict[str, str],
) -> None:
    """Add missing dynvar binding entries for the entry function's params.

    When dep functions use bind_dynamic but the entry function doesn't, the
    entry params that are passed to dynamic dep params won't have entries in
    ``bindings``.  This function adds them by reverse-mapping through the
    call site: for each dep, for each dep param that has a binding, find the
    entry param that maps to it and copy the binding.

    Mutates ``bindings`` and ``literals`` in place.
    """
    for dep in deps:
        dep_param_names = dep._param_names()
        call_args = _extract_call_args_for_dep(entry_func, dep.__name__)
        if call_args is None:
            continue
        if call_args and isinstance(call_args[0], tuple):
            dep_to_entry: dict[str, str | None] = {dep_param: entry_arg for dep_param, entry_arg in call_args}
        else:
            dep_to_entry = {dep_param: entry_arg for dep_param, entry_arg in zip(dep_param_names, call_args)}
        for dep_param, entry_arg in dep_to_entry.items():
            if entry_arg is None:
                continue
            # For each dim binding that dep_param has, copy to entry_arg
            prefix = f"{dep_param}__"
            for key, var_name in list(bindings.items()):
                if key.startswith(prefix):
                    dim_idx = key[len(prefix) :]
                    entry_key = f"{entry_arg}__{dim_idx}"
                    if entry_key not in bindings:
                        bindings[entry_key] = var_name
                        if var_name not in literals:
                            literals[var_name] = var_name


# ---------------------------------------------------------------------------


class JITFunction:
    """A JIT-compiled function with shape specialization and caching.

    Created by the ``@jit`` or ``@jit.incore`` decorators.

    Attributes:
        _func: Original Python function.
        _func_type: 'orchestration' | 'incore'.
        _level: pl.Level or None.
        _deps: Lazily-discovered list of @jit.incore sub-function dependencies.
        _deps_discovered: Whether dep discovery has been performed.
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
        self._deps: list[JITFunction] = []
        self._deps_discovered: bool = False
        self._cache: dict[CacheKey, Any] = {}  # CacheKey → CompiledProgram
        self._source_hash: str | None = None

        # Preserve function metadata
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__module__ = func.__module__

    # ------------------------------------------------------------------
    # Lazy dep discovery
    # ------------------------------------------------------------------

    def _get_deps(self) -> list[JITFunction]:
        """Return @pl.jit.incore deps called by this function (discovered lazily)."""
        if not self._deps_discovered:
            self._deps = _discover_deps(self._func)
            self._deps_discovered = True
        return self._deps

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
        set[tuple[str, int]],
    ]:
        """Bind *args/**kwargs to param names and classify into tensor/scalar metadata.

        Returns:
            (param_names, arguments, tensor_meta, scalar_values, scalar_dtypes, dynamic_dims)
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

        dynamic_dims = _scan_dynamic_dims(self._func, param_names)
        dynamic_dims = _propagate_dynamic_dims_from_deps(self._func, self._get_deps(), dynamic_dims)

        return param_names, arguments, tensor_meta, scalar_values, scalar_dtypes, dynamic_dims

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

        param_names, arguments, tensor_meta, scalar_values, scalar_dtypes, dynamic_dims = self._bind_args(
            args, kwargs
        )

        # Build cache key. Platform is included so artifacts compiled for
        # different targets never collide in the same cache.
        platform = run_config.platform if run_config is not None else None
        key = make_cache_key(
            source_hash=self._get_source_hash(),
            param_names=param_names,
            tensor_shapes={n: m.shape for n, m in tensor_meta.items()},
            tensor_dtypes={n: m.dtype for n, m in tensor_meta.items()},
            dynamic_dims=dynamic_dims,
            scalar_values=scalar_values,
            platform=platform,
        )

        # L1 cache lookup
        if key not in self._cache:
            self._cache[key] = self._compile(
                tensor_meta, scalar_values, scalar_dtypes, dynamic_dims, pl, platform=platform
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
        dynamic_dims: set[tuple[str, int]],
        pl: Any,
        platform: str | None = None,
    ) -> Any:
        """Specialize entry + deps into @pl.program source, parse, and compile.

        Runs the full compilation pipeline: pass pipeline + codegen via
        ``ir.compile()``.  Returns a ``CompiledProgram``
        containing the post-pass ``ir.Program`` and the generated output
        artifacts (orchestration C++, kernel MLIR).
        """
        from pypto.ir.compile import compile as ir_compile  # noqa: PLC0415

        contexts = self._build_contexts(tensor_meta, scalar_values, scalar_dtypes, dynamic_dims)
        dynvar_bindings, dynvar_literals = _build_dynvar_bindings(contexts)
        _backfill_entry_dynvar_bindings(self._func, self._get_deps(), dynvar_bindings, dynvar_literals)
        class_name = f"_jit_{self.__name__}_{self._get_source_hash()}"
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
        dynamic_dims: set[tuple[str, int]],
        pl: Any,
    ) -> Any:
        """Specialize entry + deps and return the parsed ir.Program (pre-pass).

        This method is intended for testing only — it lets tests inspect and
        compare the specialized IR without running the full pass pipeline or
        requiring the Ascend toolchain.
        """
        contexts = self._build_contexts(tensor_meta, scalar_values, scalar_dtypes, dynamic_dims)
        dynvar_bindings, dynvar_literals = _build_dynvar_bindings(contexts)
        _backfill_entry_dynvar_bindings(self._func, self._get_deps(), dynvar_bindings, dynvar_literals)
        class_name = f"_jit_{self.__name__}_{self._get_source_hash()}"
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
        dynamic_dims: set[tuple[str, int]],
    ) -> list[SpecializeContext]:
        """Build SpecializeContext list: deps first, entry last."""
        contexts: list[SpecializeContext] = []
        deps = self._get_deps()

        for dep in deps:
            dep_ctx = self._build_dep_context(dep, tensor_meta, scalar_values, scalar_dtypes)
            contexts.append(dep_ctx)

        # Propagate dynamic dims from deps back to the entry function.
        # If a dep marks one of its params as dynamic and that param maps
        # to an entry param at the call site, mark that entry param/dim too.
        entry_dynamic_dims = _propagate_dynamic_dims_from_deps(self._func, deps, dynamic_dims)

        dep_func_names = [dep.__name__ for dep in deps]
        entry_ctx = build_specialize_context(
            func=self._func,
            func_name=self.__name__,
            func_type=self._func_type,
            level=self._level,
            tensor_meta=tensor_meta,
            scalar_values=scalar_values,
            scalar_dtypes=scalar_dtypes,
            dynamic_dims=entry_dynamic_dims,
            dep_names=dep_func_names,
        )
        contexts.append(entry_ctx)
        return contexts

    def _build_dep_context(
        self,
        dep: JITFunction,
        entry_tensor_meta: dict[str, TensorMeta],
        entry_scalar_values: dict[str, int | float | bool],
        entry_scalar_dtypes: dict[str, DataType],
    ) -> SpecializeContext:
        """Build SpecializeContext for a single dep by mapping call-site arguments.

        Maps the entry function's argument values to the dep's parameter names
        using the call-site position: ``dep_func(entry_a, entry_b, ...)`` means
        dep's first param gets entry_a's TensorMeta, second gets entry_b's, etc.

        Falls back to name matching if call-site extraction fails.
        """
        dep_param_names = dep._param_names()

        # Try call-site positional mapping first
        call_args = _extract_call_args_for_dep(self._func, dep.__name__)

        # Also collect intermediate tensors created by pl.create_tensor in the entry
        intermediate_metas = _extract_create_tensor_metas(self._func)
        all_tensor_meta = {**intermediate_metas, **entry_tensor_meta}

        dep_tensor_meta: dict[str, TensorMeta] = {}
        dep_scalar_values: dict[str, int | float | bool] = {}
        dep_scalar_dtypes: dict[str, DataType] = {}

        if call_args is not None:
            # Detect keyword-mode: list of (dep_param, entry_arg) 2-tuples
            if call_args and isinstance(call_args[0], tuple):
                for dep_param, entry_arg in call_args:
                    if entry_arg is None or dep_param is None:
                        continue
                    if entry_arg in all_tensor_meta:
                        dep_tensor_meta[dep_param] = all_tensor_meta[entry_arg]
                    elif entry_arg in entry_scalar_values:
                        dep_scalar_values[dep_param] = entry_scalar_values[entry_arg]
                        if entry_arg in entry_scalar_dtypes:
                            dep_scalar_dtypes[dep_param] = entry_scalar_dtypes[entry_arg]
            else:
                # Positional mode: zip dep params with entry arg names
                for dep_param, entry_arg in zip(dep_param_names, call_args):
                    if entry_arg is None:
                        continue
                    if entry_arg in all_tensor_meta:
                        dep_tensor_meta[dep_param] = all_tensor_meta[entry_arg]
                    elif entry_arg in entry_scalar_values:
                        dep_scalar_values[dep_param] = entry_scalar_values[entry_arg]
                        if entry_arg in entry_scalar_dtypes:
                            dep_scalar_dtypes[dep_param] = entry_scalar_dtypes[entry_arg]
        else:
            # Fallback: name-based matching
            dep_tensor_meta = {n: all_tensor_meta[n] for n in dep_param_names if n in all_tensor_meta}
            dep_scalar_values = {
                n: entry_scalar_values[n] for n in dep_param_names if n in entry_scalar_values
            }
            dep_scalar_dtypes = {
                n: entry_scalar_dtypes[n] for n in dep_param_names if n in entry_scalar_dtypes
            }

        # Scan dep's own dynamic dims (from dep's source, using dep's param names)
        dep_dynamic_dims = _scan_dynamic_dims(dep._func, dep_param_names)

        return build_specialize_context(
            func=dep._func,
            func_name=dep.__name__,
            func_type=dep._func_type,
            level=dep._level,
            tensor_meta=dep_tensor_meta,
            scalar_values=dep_scalar_values,
            scalar_dtypes=dep_scalar_dtypes,
            dynamic_dims=dep_dynamic_dims,
            dep_names=[],
        )

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

        param_names, _, tensor_meta, scalar_values, scalar_dtypes, dynamic_dims = self._bind_args(
            args, kwargs
        )

        key = make_cache_key(
            source_hash=self._get_source_hash(),
            param_names=param_names,
            tensor_shapes={n: m.shape for n, m in tensor_meta.items()},
            tensor_dtypes={n: m.dtype for n, m in tensor_meta.items()},
            dynamic_dims=dynamic_dims,
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
                self._cache[key] = self._compile(tensor_meta, scalar_values, scalar_dtypes, dynamic_dims, pl)
            except Exception:
                pass

        # Return the post-pass ir.Program via the lightweight path
        # (no codegen) for structural equality comparison.
        pre_pass = self._compile_to_program(tensor_meta, scalar_values, scalar_dtypes, dynamic_dims, pl)
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
        if isinstance(obj, JITFunction) and obj._func_type == "incore" and name not in seen:
            deps.append(obj)
            seen.add(name)
    return deps


# ---------------------------------------------------------------------------
# _JITDecorator — supports @jit, @jit.incore, @jit.incore(level=...)
# ---------------------------------------------------------------------------


class _IncoreDecorator:
    """Handles ``@jit.incore`` and ``@jit.incore(level=...)`` sub-decorator."""

    def __call__(self, func: Any = None, *, level: Any = None) -> Any:
        """Support both ``@jit.incore`` (no args) and ``@jit.incore(level=...)``."""
        if func is None:
            # Called as @jit.incore(level=pl.Level.AIC)
            def _dec(f: Any) -> JITFunction:
                return JITFunction(f, func_type="incore", level=level)

            return _dec
        # Called as @jit.incore (no parentheses)
        return JITFunction(func, func_type="incore", level=None)


class _JITDecorator:
    """The ``pl.jit`` object.

    Supports::

        @pl.jit                               # entry-point (Orchestration)
        @pl.jit.incore                        # InCore sub-function
        @pl.jit.incore(level=pl.Level.AIC)   # InCore with explicit level
    """

    def __init__(self) -> None:
        self.incore = _IncoreDecorator()

    def __call__(self, func: Any) -> JITFunction:
        """Decorate an entry-point JIT function (Orchestration)."""
        return JITFunction(func, func_type="orchestration", level=None)


# Singleton decorator object exposed as ``pl.jit``
jit = _JITDecorator()


__all__ = ["JITFunction", "jit"]
