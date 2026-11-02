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

    @pl.jit.graph
    def layer(a: pl.Tensor, c: pl.InOut[pl.Tensor]):        # FunctionType.Graph
        with pl.at(level=pl.Level.CORE_GROUP):              # recorded once,
            ...                                             # replayed after

    @pl.jit
    def entry(a: pl.Tensor, c: pl.Out[pl.Tensor]):
        c = sub_kernel(a, c)   # dep discovered automatically (any of incore/inline/opaque)
        return c

JITFunction.__call__ flow
-------------------------
1. Capture namespaces and refresh deps when referenced JIT bindings change.
2. Classify args: tensor vs scalar.
3. Extract TensorMeta from torch.Tensor arguments.
4. Scan entry + dep ASTs for bind_dynamic declarations.
5. Resolve compile options; bypass caching for diagnostics or explicit output requests.
6. Build CacheKey including referenced constants (dynamic dims → None in shape tuple).
7. Cache hit  → execute cached CompiledProgram on device → return result.
8. Cache miss → specialize (entry + deps) → pl.parse() → ir.compile() → cache → execute → return.
"""

import ast
import copy
import functools
import inspect
import json
import os
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from pypto._cache_config import capture_cache_config, record_stats, time_stage
from pypto._external_source import external_source_digest
from pypto._identity import digest_record
from pypto._target import _platform_string
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import emit_source_loc_default
from pypto.compile_profiling import get_active_profiler
from pypto.ir.compile import _validate_pass_context_conflicts, make_default_output_dir
from pypto.ir.pass_manager import PassDumpLevel, coerce_dump_level
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir
from pypto.pypto_core import passes as _passes
from pypto.pypto_core.passes import runtime_kind_to_name

from . import _arguments, _ast_analysis, _call_binding, _tensor_meta
from ._source import cache_in_snapshot, capture_namespaces
from .cache import CacheKey, compute_source_hash, make_cache_key
from .specializer import (
    DynDim,
    SpecializeContext,
    Specializer,
    TensorMeta,
    _classify_params,
    build_specialize_context,
    free_name_source,
    func_name_lookup,
    tile_annotation_fingerprints,
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


def _ptoas_available() -> bool:
    """Return True if the ptoas binary is available on this machine."""
    return find_ptoas_binary() is not None


def _request_source_hash(source_hash: str, constexpr_records: Sequence[tuple[Any, ...]]) -> str:
    """Fold this request's constexpr bindings into the function's source hash.

    A ``pl.constexpr`` argument reaches the generated program exactly as a
    module-level constant does, so its identity is governed by the same rule:
    hash the text that gets emitted. Layering it onto ``source_hash`` rather
    than adding a ``CacheKey`` field keeps that one rule — and carries the
    identity into the persistent store for free, since the store's digest is
    derived from the same key.

    The records cover **every** function in the request, not just the entry.
    ``_get_source_hash`` can only render free *names*, so a value a dep call
    site reaches through anything else — ``helper(x, cfg.BLOCK)`` — is invisible
    to it: changing ``cfg.BLOCK`` would change the generated source while the
    key stood still, and the artifact built for the old value would be served.
    Keying the *resolved* text closes that for any expression form.

    Args:
        source_hash: The function's own hash, from ``JITFunction._get_source_hash``
        constexpr_records: ``(index, module, qualname, param, folded text)`` per
            resolved constexpr binding, in a deterministic order

    Returns:
        ``source_hash`` unchanged when there are no constexpr bindings, so a
        kernel without them keeps the identity it had before they existed
    """
    if not constexpr_records:
        return source_hash
    return compute_source_hash([source_hash, json.dumps(list(constexpr_records), separators=(",", ":"))])


class _Specialization(NamedTuple):
    """Metadata derived from one signature or sample-argument specialization."""

    param_names: list[str]
    arguments: dict[str, Any]
    tensor_meta: dict[str, TensorMeta]
    scalar_dtypes: dict[str, DataType]
    # Constexpr parameter name -> the generated-source text its value folds to.
    # Absent from ``arguments``: nothing is dispatched for a constexpr parameter.
    constexpr_values: dict[str, str]
    per_func_dyn: dict[int, dict[str, dict[int, DynDim]]]


# ---------------------------------------------------------------------------
# RunConfig -> pass-pipeline keyword forwarding
#
# The compile-side half has no counterpart here: ``RunConfig.compile_kwargs()``
# is the single mapping onto ``ir.compile()``'s parameters, and the JIT path
# calls it directly. ``lower()`` stops before codegen, so it needs its own
# narrower mapping onto ``_run_pass_pipeline``.
# ---------------------------------------------------------------------------


def _run_config_lower_kwargs(run_config: Any) -> dict[str, Any]:
    """Extract pass-only keyword arguments from a ``pypto.runtime.RunConfig``."""
    kwargs: dict[str, Any] = {
        "strategy": run_config.strategy,
        "diagnostic_phase": run_config.diagnostic_phase,
        "disabled_diagnostics": run_config.disabled_diagnostics,
        "analyze_auto_scopes_for_deps": run_config.analyze_auto_scopes_for_deps,
    }
    if run_config.memory_planner is not None:
        kwargs["memory_planner"] = run_config.memory_planner
    if run_config.enable_software_pipeline is not None:
        kwargs["enable_software_pipeline"] = run_config.enable_software_pipeline
    return kwargs


def _resolve_memory_planner(run_config: Any) -> _passes.MemoryPlanner:
    """Resolve the planner a compile would actually use, for the cache key.

    Mirrors ``ir.compile()``'s own precedence — explicit argument, then the
    active ``PassContext``, then ``PYPTO``. Reading the context matters: the
    planner is most often selected by wrapping a call in
    ``with PassContext([], memory_planner=...)``, which never reaches
    ``RunConfig`` at all. Keying only on the ``RunConfig`` field would let a
    PTOAS-wrapped call reuse a PYPTO-compiled artifact.
    """
    if run_config is not None and run_config.memory_planner is not None:
        return run_config.memory_planner
    ctx = _passes.PassContext.current()
    if ctx is not None:
        return ctx.get_memory_planner()
    return _passes.MemoryPlanner.PYPTO


def _resolve_enable_buffer_ir(software_pipeline: bool | None = None) -> bool:
    """Mirror compile's Tile IR selection for an explicit software pipeline."""
    ctx = _passes.PassContext.current()
    return (
        ctx.get_enable_buffer_ir()
        if ctx is not None
        else not _resolve_enable_software_pipeline(software_pipeline)
    )


def _resolve_enable_software_pipeline(explicit: bool | None = None) -> bool:
    """Resolve the explicit option, active context, then the default for caching."""
    if explicit is not None:
        return explicit
    ctx = _passes.PassContext.current()
    return ctx.get_enable_software_pipeline() if ctx is not None else False


def _resolve_enable_pypto_l0c_double_buffer() -> bool:
    """Resolve the stored legacy dbC flag for the compatibility cache key.

    Like ``_resolve_memory_planner``, this flag is most often set by wrapping a
    call in ``with PassContext([], enable_pypto_l0c_double_buffer=True)``, which
    ``ir.compile()`` inherits. ``RunConfig`` does not carry this PassContext-only
    flag, so the active context is the only source. Automatic dbC is disabled
    for PYPTO and DSA_RP; PTOAS retains it independently of this value.
    Existing cache-key handling is retained for compatibility.
    """
    ctx = _passes.PassContext.current()
    return ctx.get_enable_pypto_l0c_double_buffer() if ctx is not None else False


def _resolve_runtime() -> _passes.RuntimeKind:
    """Resolve the target Simpler runtime ABI for the cache key.

    Like ``_resolve_enable_pypto_l0c_double_buffer``, the runtime is selected by
    wrapping a call in ``with PassContext([], runtime=...)``, which
    ``ir.compile()`` inherits. ``RunConfig`` does not carry it, so the active
    context is the only source. Keying on it stops a ``host_build_graph`` call
    from reusing an artifact compiled for ``tensormap_and_ringbuffer``, whose
    ``kernel_config.py`` names a runtime no matching worker would bind.
    """
    ctx = _passes.PassContext.current()
    return ctx.get_runtime() if ctx is not None else _passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER


def _resolve_compile_request(run_config: Any) -> tuple[dict[str, Any], bool]:
    """Resolve compiler arguments and whether this call requires fresh compilation.

    RunConfig's compile mapping is the source for both codegen and cache keys.
    Diagnostics are requests to run the compiler, so decide bypass before any
    cache lookup (including source-key construction). With persistence disabled,
    tool discovery remains on the compile path; persistent hits additionally
    require a verified installation identity.
    """
    if run_config is None:
        from pypto.runtime import CompileOptions  # noqa: PLC0415

        kwargs = CompileOptions().as_compile_kwargs()
    else:
        kwargs = run_config.compile_kwargs()
    kwargs["dump_passes"] = coerce_dump_level(kwargs["dump_passes"])
    kwargs["emit_source_loc"] = emit_source_loc_default()
    outer = _validate_pass_context_conflicts(
        operation="compile",
        verification_level=kwargs.get("verification_level"),
        diagnostic_phase=kwargs.get("diagnostic_phase"),
        memory_planner=kwargs.get("memory_planner"),
        runtime=kwargs.get("runtime"),
        enable_software_pipeline=kwargs.get("enable_software_pipeline"),
    )
    bypass = (
        kwargs["dump_passes"] is not PassDumpLevel.NONE
        or kwargs["dump_ptoas_passes"]
        or kwargs["profiling"]
        or get_active_profiler() is not None
        or kwargs.get("output_dir") is not None
        or bool(os.environ.get("PYPTO_PROG_BUILD_DIR"))
        or os.environ.get("PYPTO_EMIT_DEBUG_RUNNER") is not None
        or os.environ.get("PYPTO_REBUILD_FROM_PTO") is not None
        or kwargs.get("verification_level") is not None
        or kwargs["diagnostic_phase"] is not None
        or kwargs["disabled_diagnostics"] is not None
    )
    if not bypass and outer is not None:
        # Match the pipeline defaults: ordinary planner/runtime contexts can
        # reuse artifacts, but instruments and custom checks must actually run.
        default_disabled = _passes.DiagnosticCheckSet()
        default_disabled.insert(_passes.DiagnosticCheck.UnusedControlFlowResult)
        bypass = (
            bool(outer.get_instruments())
            or outer.get_verification_level() != _passes.get_default_verification_level()
            or outer.get_diagnostic_phase() != _passes.get_default_diagnostic_phase()
            or outer.get_disabled_diagnostics() != default_disabled
        )
    return kwargs, bypass


# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=512)
def _persistent_dynamic_digest(static: str, values: tuple[tuple[int, str, str], ...]) -> str:
    return digest_record(("source", static, values))


class _DepGraph(NamedTuple):
    """Dependency graph published as one state and treated as read-only."""

    deps: "list[JITFunction]"
    callers: dict[int, list[tuple[Any, str]]]
    callees: dict[int, list[str]]
    call_args: dict[tuple[int, str], list[tuple[str | None, str | _call_binding.SlicedArg | None]] | None]


class _CachedLayouts(NamedTuple):
    """Resolved layouts and the annotation bindings used to derive them."""

    bindings: tuple[Any, ...]
    layouts: tuple[tuple[str, str, str], ...]


@dataclass
class _CachedDepGraph:
    """Read-only graph data with an atomically replaced layout cache."""

    graph: _DepGraph
    bindings: tuple[tuple[_call_binding.DepBinding, ...], ...]
    source_hash: str | None
    layout_dependencies: tuple[tuple[Any, tuple[str, ...]], ...]
    layouts: _CachedLayouts | None = None
    persistent_source_hash: str | None = None


@functools.lru_cache(maxsize=512)
def _layout_dependency_names(func: Any) -> tuple[str, ...]:
    """Capture roots read by postponed parameter annotations once per function."""
    names: set[str] = set()
    for name, param in inspect.signature(func).parameters.items():
        if name == "self" or not isinstance(param.annotation, str):
            continue
        try:
            annotation = ast.parse(param.annotation, mode="eval")
        except SyntaxError:
            continue
        names.update(node.id for node in ast.walk(annotation) if isinstance(node, ast.Name))
    return tuple(sorted(names))


@functools.lru_cache(maxsize=512)
def _python_source(func: Any) -> str:
    """Cache immutable Python source without caching mutable dependency state."""
    return inspect.getsource(func)


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
            the config is forwarded through ``_compile`` → ``ir.compile()``
            (see ``RunConfig.compile_kwargs``), which yields a
            ``DistributedCompiledProgram`` that ``__call__`` dispatches
            per-rank.
        _level: pl.Level or None.
        _auto_scope: Whether the compiler auto-inserts AUTO runtime scopes
            (SIMPLER_SCOPE) around the body and each for/if body. ``True`` by
            default; set ``False`` via ``@pl.jit(auto_scope=False)`` /
            ``@pl.jit.host(auto_scope=False)`` to place scopes by hand with
            ``with pl.scope()``. Also accepted on
            ``@pl.jit.inline(auto_scope=False)`` — after the ``InlineFunctions``
            pass splices the body, hand-placed scopes land in the caller.
            ``incore`` / ``opaque`` kinds reject it (they outline into
            separate kernels, so scopes never land in the caller).
        _dep_graph_state: Last resolved graph and its validating bindings,
            published together. Each call pins its own validated graph.
        _cache: L1 in-memory cache: CacheKey → CompiledProgram (post-pass ir.Program wrapped).
    """

    def __init__(
        self,
        func: Any,
        func_type: str | None = None,
        level: Any = None,
        auto_scope: bool = True,
        external_core_type: str | None = None,
        external_aic_source: str | None = None,
        external_aiv_source: str | None = None,
        external_dual_aiv_dispatch: bool = False,
        external_include_dirs: tuple[str, ...] = (),
    ) -> None:
        self._func = func
        self._func_type = func_type or "orchestration"
        self._level = level
        self._auto_scope = auto_scope
        # External C++ kernel backing (func_type == "extern"): resolved absolute
        # paths the specializer emits as @pl.function(external_source=...).
        self._external_core_type = external_core_type
        self._external_aic_source = external_aic_source
        self._external_aiv_source = external_aiv_source
        self._external_dual_aiv_dispatch = external_dual_aiv_dispatch
        self._external_include_dirs = external_include_dirs
        self._dep_graph_state: _CachedDepGraph | None = None
        self._cache: dict[CacheKey, Any] = {}  # CacheKey → CompiledProgram
        self._artifact_objects: dict[Any, Any] = {}
        self._cache_lock = threading.RLock()

        # Preserve function metadata
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__module__ = func.__module__

    @property
    def _diagnostic_filename(self) -> str:
        """Synthetic filename for the generated, specialized source.

        Statements that survive specialization are remapped to the user's real
        ``.py`` via the source map (see ``Specializer.source_map``); this
        ``<jit:name>`` marker is only the fallback identity for synthesized
        statements that have no original location. Naming the kernel here is far
        more navigable than an anonymous ``<string>``. See issue #1612.
        """
        return f"<jit:{self.__name__}>"

    # ------------------------------------------------------------------
    # Lazy dep discovery
    # ------------------------------------------------------------------

    @cache_in_snapshot
    def _dep_declared_layouts(self) -> tuple[tuple[str, str, str], ...]:
        """Layouts every reachable dep declares on its own parameters.

        ``_overlay_dep_declared_layouts`` folds these into the generated dep
        signatures, so they change the artifact — but they live outside the
        entry's ``tensor_meta``, and a postponed annotation
        (``pl.Tensor[..., L]`` with a module-level ``L``) keeps the source text,
        and therefore ``source_hash``, identical when ``L`` is rebound. Without
        them in the key, rebinding ``L`` would hand the second call the first
        one's artifact.

        Reused across calls while the graph and roots referenced by postponed
        annotations are unchanged. Retain the bindings themselves and compare
        identity, avoiding overloaded equality and recycled object IDs.

        Keyed by the *generated* name, not ``dep.__name__``: the triples are
        sorted, so position is not carried, and two same-named deps swapping
        layouts (``helper`` from two modules going ``NZ``/``ND`` -> ``ND``/``NZ``)
        would otherwise produce the same sorted set and hand the second call the
        first one's artifact. The generated name is the disambiguator the
        emitted signatures already carry.

        Returns:
            Sorted ``(generated dep name, parameter, layout)`` triples for the
            cache key.
        """
        state = self._get_dep_graph_state()
        bindings = tuple(
            func_name_lookup(func).get(name) for func, names in state.layout_dependencies for name in names
        )
        cached = state.layouts
        if cached is not None and all(a is b for a, b in zip(bindings, cached.bindings, strict=True)):
            return cached.layouts
        # Keyed per function, not per ``pl.constexpr`` variant: a parameter's
        # layout is a property of the declaration, so every variant of a dep
        # carries the same one and only needs to be counted once. The names
        # therefore agree with the generated program's for any single-binding
        # call graph, and stay a stable discriminator when a dep is emitted
        # more than once.
        gen_names = _allocate_generated_names([(id(d._func), d) for d in [self, *state.graph.deps]])
        layouts = tuple(
            sorted(
                (gen_names[id(dep._func)], param, str(layout))
                for dep in state.graph.deps
                # ``dep.__name__`` here is the diagnostic name only — a layout
                # error should name the user's own function.
                for param, layout in _arguments.param_layouts(dep._func, dep.__name__).items()
            )
        )
        state.layouts = _CachedLayouts(bindings, layouts)
        return layouts

    def _get_dep_graph(self) -> _DepGraph:
        """Return the transitive JIT dep graph rooted at this function.

        The graph is computed lazily and reused while its direct dependency
        bindings remain unchanged. Each request pins its validated graph;
        another request can publish a new graph without changing this one.
        Published graphs are never modified. Returns:

        - ``deps_topo``: every reachable dep in leaf-first topological order
          (deduplicated by underlying Python function identity). The entry
          function is NOT included.
        - ``callers_by_dep_id``: for each dep, the list of
          ``(caller_func, call_name)`` pairs whose bodies contain a call site
          to it — ``call_name`` being the name that caller's source calls it
          by, which differs from ``dep.__name__`` under an aliased import.
          Recorded in DFS-discovery order; deduplicated within each list. The
          entry has no caller and does not appear as a key.

          Tensor / scalar metadata for a shared dep is still resolved
          through the first-recorded caller — call sites in other branches
          must agree on shapes/dtypes (otherwise one specialization would
          have to differ from another, which the one-context-per-function
          design doesn't support).
        - ``callees_by_func_id``: for each function (entry + every reached
          dep), the *call names* of the JIT deps it directly calls. Used to
          set each context's ``dep_names`` so the body transformer rewrites
          nested dep calls into the ``self.<dep>(...)`` form required by
          multi-function ``@pl.program``.
        - ``call_args_cache``: ``(id(caller_func), call_name)`` → unified
          call-site arg list (see ``extract_call_args_for_dep``) or
          ``None`` if the call site isn't found. Cached so metadata
          resolution doesn't re-walk caller ASTs on every JIT call.
        """
        return self._get_dep_graph_state().graph

    @cache_in_snapshot
    def _get_dep_graph_state(self) -> _CachedDepGraph:
        """Pin a graph and its derived caches to the current request."""
        cached = self._dep_graph_state
        if cached is not None:
            bindings = tuple(
                tuple(_discover_dep_bindings(fn._func, fn._func_type)) for fn in [self, *cached.graph.deps]
            )
            if bindings == cached.bindings:
                return cached
        deps_topo: list[JITFunction] = []
        seen: set[int] = set()
        callers_by_dep_id: dict[int, list[Any]] = {}
        callees_by_func_id: dict[int, list[str]] = {}
        call_args_cache: dict[
            tuple[int, str], list[tuple[str | None, str | _call_binding.SlicedArg | None]] | None
        ] = {}

        def visit(func: Any, caller_func_type: str) -> None:
            direct = _discover_dep_bindings(func, caller_func_type)
            # Call names, not ``__name__``: these become ``ctx.dep_names``,
            # which the body transformer matches against the ``ast.Name``
            # the source actually calls.
            callees_by_func_id[id(func)] = [b.call_name for b in direct]
            for call_name, dep in direct:
                # Key everything off ``id(dep._func)`` (the underlying
                # Python function) — same key the downstream helpers
                # use, and stable across multiple wrapper objects for
                # the same source function.
                callers = callers_by_dep_id.setdefault(id(dep._func), [])
                if (func, call_name) not in callers:
                    callers.append((func, call_name))
                # Memoise per-(caller, call name) call-site args once.
                cache_key = (id(func), call_name)
                if cache_key not in call_args_cache:
                    call_args_cache[cache_key] = _call_binding.extract_call_args_for_dep(func, call_name)
                if id(dep._func) in seen:
                    continue
                # Mark before recursing — this also serves as a cycle
                # guard (a self-recursive JIT function is unsupported
                # but won't loop forever here).
                seen.add(id(dep._func))
                visit(dep._func, dep._func_type)
                deps_topo.append(dep)

        visit(self._func, self._func_type)
        graph = _DepGraph(
            deps_topo,
            callers_by_dep_id,
            callees_by_func_id,
            call_args_cache,
        )
        bindings = tuple(tuple(_discover_dep_bindings(fn._func, fn._func_type)) for fn in [self, *deps_topo])
        source_hash = (
            None
            if any(fn._func_type == "extern" for fn in [self, *deps_topo])
            else self._compute_static_source_hash(deps_topo)
        )
        layout_dependencies = tuple(
            (dep._func, names) for dep in deps_topo if (names := _layout_dependency_names(dep._func))
        )
        state = _CachedDepGraph(graph, bindings, source_hash, layout_dependencies)
        self._dep_graph_state = state
        return state

    def _get_deps(self) -> list["JITFunction"]:
        """Return all transitively-reachable JIT deps in leaf-first order."""
        return self._get_dep_graph()[0]

    # ------------------------------------------------------------------
    # Source hash (derived from the request's pinned dependency graph)
    # ------------------------------------------------------------------

    def _external_source_paths(self) -> list[str]:
        """Absolute paths of the C++ source(s) backing an external kernel dep."""
        return [p for p in (self._external_aic_source, self._external_aiv_source) if p is not None]

    @capture_namespaces()
    def _get_source_hash(self) -> str:
        """Hash source structure and the current values of referenced constants.

        Each referenced name contributes the exact text the specializer will fold
        it into, straight from ``free_name_source``. Reading the emitted form —
        rather than re-deciding here which types count — is what keeps the key in
        step with the folding rule: a constant that changes the generated source
        changes this hash by construction, and one that does not fold (an opaque
        object, a JIT dep, ``pl`` itself) contributes nothing because it leaves
        the source unchanged.

        Covers only what the *function* brings. A ``pl.constexpr`` argument
        folds by the same rule but is bound per call site, so it belongs to the
        request; :func:`_request_source_hash` layers it on top.
        """
        source_hash = self._get_static_source_hash()
        records: list[tuple[Any, ...]] = []
        for index, jit_func in enumerate([self, *self._get_deps()]):
            func = jit_func._func
            namespace = func_name_lookup(func)
            for name in _ast_analysis.constant_dependency_names(func):
                folded = free_name_source(name, namespace)
                if folded is None:
                    continue
                records.append((index, func.__module__, func.__qualname__, name, folded))
            if jit_func._func_type == "inline":
                # Tile annotations can retain factory values without closure
                # cells or name DynVars with no body-value rendering. Hash the
                # canonical annotation snapshot specialization uses.
                for name, source in tile_annotation_fingerprints(func).items():
                    records.append(
                        (index, func.__module__, func.__qualname__, f"<annotation:{name}>", source)
                    )
        return compute_source_hash([source_hash, json.dumps(records, separators=(",", ":"))])

    def _resolve_constexpr_bindings(self, entry_constexpr: Mapping[str, str]) -> _call_binding.VariantPlan:
        """Expand this request's call graph into one function per constexpr binding.

        Run before the cache key as well as during specialization: a dep's
        constant is part of what gets compiled, so it has to be part of what
        the key identifies, and the same walk backing both means the two cannot
        disagree.

        Args:
            entry_constexpr: The entry's own bindings, from argument binding

        Returns:
            The request's :class:`VariantPlan`
        """
        deps_topo, callers_by_id, _, _ = self._get_dep_graph()
        return _call_binding.expand_constexpr_variants(self, deps_topo, callers_by_id, entry_constexpr)

    def _constexpr_identity_records(self, plan: _call_binding.VariantPlan) -> list[tuple[Any, ...]]:
        """Flatten a variant plan into deterministic cache-identity records.

        Indexed by position in ``[self, *deps]`` and qualified by the defining
        module, mirroring ``_get_source_hash``'s records so two functions that
        share a parameter name stay distinguishable. A dep emitted once per
        binding contributes one record group per binding, so a program that
        compiles ``helper`` at 16 and at 32 cannot collide with one that
        compiles it twice at 16.

        Once anything is split, **which** specialization each call site reaches
        is also part of the identity, because the set of bindings does not
        determine it. Three calls at ``16``, ``32`` and ``cfg.N`` emit the same
        two bindings whichever value ``cfg.N`` holds; only the third call's
        target moves, and an attribute like ``cfg.N`` is invisible to
        ``_get_source_hash``, which can render free names only. Without the
        wiring the key stands still while the generated program changes, and
        the artifact built for the other target is served.

        The wiring is emitted only when some function really is split. With one
        binding apiece each call name has exactly one possible target, so the
        call graph in the source hash plus the bindings above already pin it,
        and staying silent keeps every existing key — and the artifacts behind
        it — valid.
        """
        position = {id(jit_func._func): index for index, jit_func in enumerate([self, *self._get_deps()])}
        split = len({key[0] for key in plan.order}) != len(plan.order)
        records: list[tuple[Any, ...]] = []
        for key in plan.order:
            func = plan.jit_func[key]._func
            index = position.get(id(func), -1)
            binding = _call_binding.freeze_binding(plan.binding[key])
            for name, text in binding:
                records.append((index, func.__module__, func.__qualname__, name, text))
            if not split:
                continue
            for (call_name, ordinal), target in sorted(plan.call_targets[key].items()):
                records.append(
                    (
                        index,
                        func.__module__,
                        func.__qualname__,
                        binding,
                        call_name,
                        ordinal,
                        position.get(target[0], -1),
                        target[1],
                    )
                )
        return records

    @cache_in_snapshot
    def _get_static_source_hash(self) -> str:
        """Hash source and compilation attributes from the request's graph."""
        deps = self._get_deps()
        state = self._get_dep_graph_state()
        if state.source_hash is not None:
            return state.source_hash
        return self._compute_static_source_hash(deps)

    def _compute_static_source_hash(self, deps: list["JITFunction"]) -> str:
        """Compute source structure on graph changes or external-source lookups."""
        sources: list[str] = []
        for jit_func in [self, *deps]:
            sources.append(
                json.dumps(
                    (jit_func.__name__, jit_func._func_type, str(jit_func._level), jit_func._auto_scope),
                    separators=(",", ":"),
                )
            )
            if jit_func._func_type == "extern":
                # External sources and quoted includes can change between
                # requests. Recompute their digest even when the graph is reused.
                sources.append(
                    external_source_digest(
                        jit_func._external_source_paths(),
                        metadata=(
                            _python_source(jit_func._func),
                            jit_func.__name__,
                            jit_func._external_core_type or "",
                            str(jit_func._external_dual_aiv_dispatch),
                            *(f"include_dir:{path}" for path in jit_func._external_include_dirs),
                        ),
                        include_dirs=jit_func._external_include_dirs,
                    )
                )
            else:
                sources.append(_python_source(jit_func._func))
        return compute_source_hash(sources)

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
        dict[str, DataType],
        dict[str, str],
        dict[int, dict[str, dict[int, DynDim]]],
    ]:
        """Bind *args/**kwargs to param names and classify into tensor/scalar metadata.

        A numeric argument to a ``pl.Scalar`` parameter is a **runtime value**:
        it is dispatched as-is and never folded into the artifact, so it
        contributes no metadata here and does not split the specialization
        cache. A ``pl.constexpr`` parameter is the opposite -- its argument is
        rendered to source text, leaves ``arguments`` so nothing is dispatched
        for it, and reaches ``_get_source_hash`` so each distinct value selects
        its own artifact.

        Tensor metas carry ``DynDim`` entries for every param dim that is
        either declared dynamic at this function (``bind_dynamic`` / annotation
        ``pl.dynamic()``) **or** cascaded up from a dep's declarations.
        Cascading happens during ``compute_per_func_dyndim_maps`` so the cache
        key reflects every dynamic dim reachable through the dep graph — two
        calls with different runtime extents reuse the same compilation.

        Returns the per-function DynDim map alongside the entry metadata so
        downstream specialization (``resolve_dep_call_metadata``) can pull
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
        per_func_dyn_maps = _call_binding.compute_per_func_dyndim_maps(
            self._func, param_names, deps, callers_by_id, call_args_cache
        )
        entry_dyn_map = per_func_dyn_maps[id(self._func)]
        # A layout has no runtime counterpart on a torch tensor, so it comes
        # from the annotation even on this path. Raw UINT8 carriers likewise
        # need the annotation to identify their packed FP4 format.
        param_layouts = _arguments.param_layouts(self._func, self.__name__)
        param_dtypes = _arguments.param_dtypes(self._func)
        tensor_meta: dict[str, TensorMeta] = {}
        scalar_dtypes: dict[str, DataType] = {}
        constexpr_names = _arguments.constexpr_params(self._func)
        constexpr_values: dict[str, str] = {}

        from pypto.language.typing.scalar import RUNTIME  # noqa: PLC0415

        for name, value in arguments.items():
            # ``pl.RUNTIME`` is a compile-time marker, not a value. This path
            # binds real arguments (dispatch, or sample-argument compile), where
            # it would otherwise reach the runtime and fail much later with an
            # opaque "must be real number". Note ``apply_defaults()`` above
            # materializes a ``= pl.RUNTIME`` signature default, so a plain
            # ``kernel(a, c)`` call reaches here too.
            if value is RUNTIME:
                raise TypeError(
                    f"@pl.jit function '{self.__name__}': parameter '{name}' received "
                    f"pl.RUNTIME, which is a compile-time marker rather than a value. This "
                    f"path binds real arguments, so pass '{name}' its actual value — a scalar "
                    f"parameter is a runtime value and does not compile a separate artifact "
                    f"per value. The marker is accepted only by annotation-driven signature "
                    f"mode ({self.__name__}.compile() with no tensor arguments), where it is "
                    f"now redundant."
                )
            if _arguments.is_tensor(value):
                tensor_meta[name] = _arguments.extract_tensor_meta(
                    value,
                    entry_dyn_map.get(name),
                    param_layouts.get(name),
                    param_dtypes.get(name),
                )

        # Drop constexpr bindings out of ``arguments``: they are resolved during
        # specialization and have no slot in the dispatch ABI, so the ordered
        # argument list this feeds must not carry them.
        for name in constexpr_names:
            constexpr_values[name] = _arguments.resolve_constexpr_value(
                self.__name__, name, arguments.pop(name)
            )

        return (
            param_names,
            arguments,
            tensor_meta,
            scalar_dtypes,
            constexpr_values,
            per_func_dyn_maps,
        )

    def _bind_args_from_signature(
        self, kwargs: dict[str, Any]
    ) -> tuple[
        list[str],
        dict[str, Any],
        dict[str, TensorMeta],
        dict[str, DataType],
        dict[str, str],
        dict[int, dict[str, dict[int, DynDim]]],
    ]:
        """Derive the same metadata as ``_bind_args``, but from the kernel's
        own parameter annotations — no tensor arguments required.

        Used by [`lower`][pypto.language.JITFunction.lower] and
        [`compile`][pypto.language.JITFunction.compile] in annotation-driven signature mode. Each tensor
        parameter's ``pl.Tensor[[...], dtype]`` annotation supplies the shape/dtype contract directly: static
        dims are annotation integers, while dynamic dims (``pl.dynamic`` / ``bind_dynamic``) are marked
        dynamic and given a placeholder extent. Dynamic dimensions lower to runtime ``pl.tensor.dim`` reads
        and, on the compiled path, collapse to ``None`` in the cache key.

        Scalar parameters need no value: each one survives into the generated
        program as a real ``pl.Scalar`` parameter whose value arrives at
        dispatch, exactly like a dynamic dim extent, and drops out of the cache
        key so one artifact serves every value. A value supplied anyway — a
        signature default, an explicit keyword, or ``pl.RUNTIME`` — is validated
        and then contributes nothing (see :func:`check_signature_scalar_value`).

        A ``pl.constexpr`` parameter is the one kind that *does* need a value
        here, because its value is the artifact: it comes from ``kwargs`` or a
        signature default, and a parameter with neither is an error.

        Raises:
            TypeError: if a tensor parameter has a bare ``pl.Tensor`` annotation
                (no shape to read), a scalar parameter is given a value that is
                neither a number nor ``pl.RUNTIME``, or a constexpr parameter
                has no value.
        """
        from pypto.language.typing.constexpr import ConstexprMarker  # noqa: PLC0415
        from pypto.language.typing.dynamic import DynVar  # noqa: PLC0415
        from pypto.language.typing.scalar import Scalar  # noqa: PLC0415
        from pypto.language.typing.tensor import Tensor  # noqa: PLC0415

        param_names = self._param_names()
        sig = inspect.signature(self._func)

        # Namespace for resolving string annotations (``from __future__ import
        # annotations``): the function's globals merged with its closure free-vars,
        # so an annotation referencing an enclosing scope (e.g. a ``pl.dynamic`` /
        # constant defined in an outer function) resolves. We ``eval`` each string
        # annotation directly rather than via ``typing.get_type_hints`` because the
        # latter runs ``_type_check`` on the result, which rejects our custom
        # ``Tensor`` / ``Scalar`` instance annotations on Python 3.10.
        ann_ns: Mapping[str, Any] | None = None
        if any(isinstance(p.annotation, str) for n, p in sig.parameters.items() if n != "self"):
            ann_ns = func_name_lookup(self._func)

        deps, callers_by_id, _, call_args_cache = self._get_dep_graph()
        per_func_dyn_maps = _call_binding.compute_per_func_dyndim_maps(
            self._func, param_names, deps, callers_by_id, call_args_cache
        )
        entry_dyn_map = per_func_dyn_maps[id(self._func)]

        tensor_meta: dict[str, TensorMeta] = {}
        scalar_dtypes: dict[str, DataType] = {}
        constexpr_values: dict[str, str] = {}

        for name in param_names:
            param = sig.parameters[name]
            annotation = param.annotation
            if isinstance(annotation, str) and ann_ns is not None:
                try:
                    # Trusted input: the kernel's own annotation source, evaluated
                    # in its own globals+closure namespace (same as Python would).
                    annotation = eval(annotation, dict(ann_ns))  # noqa: S307
                except Exception:  # noqa: BLE001 - leave as string; handled below as "cannot infer"
                    pass

            # A bare ``pl.Tensor`` is the Tensor *class* itself (no shape);
            # ``pl.Tensor[[...], dtype]`` is a Tensor *instance* carrying
            # shape/dtype. ``pl.Out[...]``/``pl.InOut[...]`` unwrap to their
            # inner type, so both directions flow through the instance branch.
            bare_msg = (
                f"@pl.jit function '{self.__name__}': cannot specialize from the signature "
                f"because parameter '{name}' has a bare 'pl.Tensor' annotation with no shape. "
                f"Give it a full 'pl.Tensor[[...], dtype]' annotation, or pass sample tensors "
                f"to lower(*sample_tensors) or compile(*sample_tensors)."
            )
            if isinstance(annotation, Tensor):
                if annotation.shape is None or annotation.dtype is None:
                    raise TypeError(bare_msg)
                tensor_meta[name] = _arguments.signature_tensor_meta(
                    annotation,
                    annotation.dtype,
                    entry_dyn_map.get(name, {}),
                    DynVar,
                    param_name=name,
                    func_name=self.__name__,
                )
                continue
            if isinstance(annotation, type) and issubclass(annotation, Tensor):
                raise TypeError(bare_msg)

            if isinstance(annotation, ConstexprMarker):
                constexpr_values[name] = _arguments.resolve_constexpr_value(
                    self.__name__,
                    name,
                    _arguments.signature_constexpr_value(self.__name__, name, param, kwargs),
                )
                continue

            # Scalar-like annotation: pl.Scalar[dtype] or a bare DataType.
            scalar_dtype = annotation.dtype if isinstance(annotation, Scalar) else None
            if scalar_dtype is None and isinstance(annotation, DataType):
                scalar_dtype = annotation
            if scalar_dtype is not None:
                # The specializer never substitutes a scalar parameter, so it
                # stays symbolic in the generated program and out of the cache
                # key — one artifact serves every runtime value. Any value the
                # caller supplied is only validated here.
                _arguments.check_signature_scalar_value(self.__name__, name, param, kwargs)
                scalar_dtypes[name] = scalar_dtype
                continue

            # Unknown / unannotated parameter — cannot infer without a value.
            raise TypeError(
                f"@pl.jit function '{self.__name__}': cannot infer parameter '{name}' from the "
                f"signature (annotation: {annotation!r}). Annotate it as a shaped 'pl.Tensor' / "
                f"'pl.Scalar[dtype]', or pass sample values to lower(*sample_args) or "
                f"compile(*sample_args)."
            )

        # Signature mode binds no runtime arguments at all: it has no tensor
        # samples, and a scalar's value now arrives at dispatch rather than at
        # compile time. Its callers (``compile`` / ``lower`` / ``warmup``)
        # discard the ordered-argument list this feeds.
        arguments: dict[str, Any] = {}
        return (
            param_names,
            arguments,
            tensor_meta,
            scalar_dtypes,
            constexpr_values,
            per_func_dyn_maps,
        )

    # ------------------------------------------------------------------
    # Call
    # ------------------------------------------------------------------

    def _resolve_specialization(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        allow_signature_mode: bool = False,
    ) -> tuple[_Specialization, Any | None]:
        """Bind signature or sample arguments and consume the ``RunConfig``.

        Shared by [`lower`][pypto.language.JITFunction.lower], ``__call__``, and
        [`compile`][pypto.language.JITFunction.compile].

        When ``allow_signature_mode`` is set and no positional args are given,
        the shape/dtype contract is read from the kernel's own annotations via
        ``_bind_args_from_signature``. ``__call__`` never enables this
        because on-device dispatch needs real tensors.

        Returns:
            The specialization metadata and the consumed ``RunConfig``.
        """
        # Extract RunConfig without mutating *kwargs* — although the caller's
        # ``**kwargs`` dict is normally owned by Python at this scope, building
        # a fresh dict is the same cost and removes the ambiguity for readers
        # who don't track the calling convention.
        run_config = kwargs.get("config")
        if "config" in kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k != "config"}

        # Annotation-driven signature mode (lower() and compile()) reads shapes
        # from annotations, but ONLY when no tensor values were supplied —
        # positionally OR by keyword. Keyword tensor samples (``lower(a=x)`` or
        # ``compile(a=x)``) must still bind through ``_bind_args``/``sig.bind``;
        # scalar/config kwargs do not block signature mode.
        signature_mode = (
            allow_signature_mode and not args and not any(_arguments.is_tensor(v) for v in kwargs.values())
        )
        bind = self._bind_args_from_signature(kwargs) if signature_mode else self._bind_args(args, kwargs)
        param_names, arguments, tensor_meta, scalar_dtypes, constexpr_values, per_func_dyn = bind

        return (
            _Specialization(
                param_names,
                arguments,
                tensor_meta,
                scalar_dtypes,
                constexpr_values,
                per_func_dyn,
            ),
            run_config,
        )

    @capture_namespaces()
    def _resolve_compiled(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        allow_signature_mode: bool = False,
    ) -> tuple[Any, list[Any], Any | None]:
        """Look up or build a specialized CompiledProgram.

        Shared by ``__call__`` (which then dispatches) and [`compile`][pypto.language.JITFunction.compile]
        (which then returns the CompiledProgram). Cache keys include all inputs
        that affect the generated artifact.

        Returns:
            ``(compiled, ordered_args, run_config)`` where ``ordered_args``
            follows the decorated function's declared parameter order.
        """
        import pypto.language as pl  # noqa: PLC0415

        specialization, run_config = self._resolve_specialization(
            args,
            kwargs,
            allow_signature_mode=allow_signature_mode,
        )

        compile_kwargs, bypass_cache = _resolve_compile_request(run_config)
        target_platform = _platform_string(compile_kwargs["backend_type"], compile_kwargs["execution_mode"])
        cache_config = capture_cache_config(getattr(run_config, "cache_config", None))
        record_stats(requests=1)
        if not cache_config.enabled:
            record_stats(disabled_requests=1)
        ordered_args = [
            specialization.arguments[n] for n in specialization.param_names if n in specialization.arguments
        ]

        def build(**overrides: Any) -> Any:
            record_stats(generation_builds=1)
            with time_stage("build_ns"):
                return self._compile(
                    specialization.tensor_meta,
                    specialization.scalar_dtypes,
                    specialization.constexpr_values,
                    specialization.per_func_dyn,
                    pl,
                    **(compile_kwargs | overrides),
                )

        if bypass_cache:
            record_stats(forced_rebuilds=1)
            return build(), ordered_args, run_config

        # Resolved before the key rather than during ``build()``: a dep's
        # compile-time value is part of what gets compiled, so the key has to
        # identify it too.
        constexpr_bindings = self._resolve_constexpr_bindings(specialization.constexpr_values)
        key = make_cache_key(
            source_hash=_request_source_hash(
                self._get_source_hash(), self._constexpr_identity_records(constexpr_bindings)
            ),
            param_names=specialization.param_names,
            tensor_shapes={n: m.static_shape() for n, m in specialization.tensor_meta.items()},
            tensor_dtypes={n: m.dtype for n, m in specialization.tensor_meta.items()},
            tensor_layouts={n: m.layout for n, m in specialization.tensor_meta.items()},
            dep_layouts=self._dep_declared_layouts(),
            dynamic_dims={
                (n, i) for n, m in specialization.tensor_meta.items() for i in m.dynamic_dim_indices()
            },
            platform=target_platform,
            strategy=compile_kwargs["strategy"],
            distributed_config=compile_kwargs.get("distributed_config"),
            analyze_auto_scopes_for_deps=compile_kwargs["analyze_auto_scopes_for_deps"],
            emit_source_loc=compile_kwargs["emit_source_loc"],
            memory_planner=compile_kwargs.get("memory_planner", _resolve_memory_planner(None)),
            enable_pypto_l0c_double_buffer=_resolve_enable_pypto_l0c_double_buffer(),
            enable_buffer_ir=_resolve_enable_buffer_ir(compile_kwargs.get("enable_software_pipeline")),
            enable_software_pipeline=_resolve_enable_software_pipeline(
                compile_kwargs.get("enable_software_pipeline")
            ),
            runtime=_resolve_runtime(),
        )

        with self._cache_lock:
            if cache_config.enabled:
                from ._persistent import resolve_persistent  # noqa: PLC0415

                compiled = resolve_persistent(
                    self,
                    key,
                    cache_config,
                    build,
                    self._persistent_source_digest,
                    platform=target_platform,
                    runtime_name=runtime_kind_to_name(_resolve_runtime()),
                    distributed=self._func_type == "host",
                )
            elif key in self._cache:
                record_stats(object_hits=1)
                compiled = self._cache[key]
            else:
                compiled = self._cache[key] = build()
        return compiled, ordered_args, run_config

    def _persistent_source_digest(self) -> str:
        """Memoize immutable graph content; refresh folded constants and externs."""
        state = self._get_dep_graph_state()
        functions = [self, *state.graph.deps]
        if state.persistent_source_hash is None:
            state.persistent_source_hash = digest_record(
                [
                    (
                        _python_source(fn._func),
                        fn._func.__module__,
                        fn._func.__qualname__,
                        fn._func.__code__.co_filename,
                        fn._func.__code__.co_firstlineno,
                        fn._func_type,
                        str(fn._level),
                        fn._auto_scope,
                        fn._external_core_type,
                        fn._external_dual_aiv_dispatch,
                        tuple(fn._external_source_paths()),
                        tuple(fn._external_include_dirs),
                    )
                    for fn in functions
                ]
            )
        values: list[tuple[Any, ...]] = []
        for index, fn in enumerate(functions):
            namespace = func_name_lookup(fn._func)
            for name in _ast_analysis.constant_dependency_names(fn._func):
                folded = free_name_source(name, namespace)
                if folded is not None:
                    values.append((index, name, folded))
            if fn._func_type == "inline":
                for name, source in tile_annotation_fingerprints(fn._func).items():
                    values.append((index, f"<annotation:{name}>", source))
            if fn._func_type == "extern":
                external = external_source_digest(
                    fn._external_source_paths(),
                    include_dirs=fn._external_include_dirs,
                    metadata=(*fn._external_source_paths(), *fn._external_include_dirs),
                )
                values.append((index, "<extern>", external))
        return _persistent_dynamic_digest(state.persistent_source_hash, tuple(values))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Specialize, compile (or serve from cache), and execute on device.

        A compatible compiled object is reused in process. When persistent
        caching is enabled, a disk hit restores generated code or complete
        binaries; a miss specializes into ``@pl.program`` and runs passes and
        codegen. Execution publishes missing binaries automatically. Diagnostic
        and explicit output requests bypass lookup and insertion on every call.

        The compiled kernel is then executed on the NPU device with the given
        torch tensor arguments (Triton-like API).

        A ``config=RunConfig(...)`` keyword argument is consumed here rather
        than passed to the decorated function: its compile-side fields
        (``strategy``, ``dump_passes``, diagnostics, ...) are forwarded to
        ``ir.compile()`` via ``RunConfig.compile_kwargs``, and its
        runtime fields drive on-device execution.  ``strategy`` also takes
        part in the cache key so artifacts compiled under different strategy
        values never share a cache entry.

        Args:
            *args: Positional arguments matching the decorated function's params.
            **kwargs: Keyword arguments.  A ``config`` keyword, if present, is
                a ``RunConfig`` and is consumed by
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
        and return the underlying ``CompiledProgram``.

        Same specialization / cache pipeline as ``__call__``, minus the
        on-device dispatch. Use this when you want to drive execution through
        the runtime worker API directly:

        - ``pypto.runtime.ChipWorker.run`` / ``register``
          for explicit L2 dispatch.
        - ``CompiledProgram.chip_callable`` / ``runtime_name`` / ``runtime_config``
          to drive a hand-constructed ``simpler.worker.Worker``.

        ``config=RunConfig(...)`` is still consumed (and its compile-side
        knobs forwarded to ``ir.compile()``) so the returned
        ``CompiledProgram`` honours the same options as a direct
        ``kernel(*args, config=...)`` call. Runtime-side fields on the
        ``RunConfig`` (``device_id``, DFX flags, ...) do not apply here —
        they affect dispatch, not the compiled artefact.

        Subsequent calls (either ``__call__`` or [`compile`][pypto.language.JITFunction.compile]) with the
        same specialization and compatible cache policy return the same
        ``CompiledProgram`` instance. With persistence enabled, a disk-restored
        object has ``program is None``; disable persistence when IR is required.
        Dump, compile-profiling, explicit output,
        and custom pass-diagnostic requests always compile afresh, preserving
        ordinary cached entries. Omitting ``config`` uses ``RunConfig`` defaults,
        including disabled dumps.

        **Compiling without tensors.** When called with **no tensor
        arguments** — neither positional nor keyword — the shape/dtype contract
        is read directly from the kernel's own parameter annotations, so no
        throwaway ``torch.empty(...)`` dummies are needed. (Passing tensors by
        keyword, e.g. ``compile(a=sample_a)``, still binds them normally.) This
        requires every tensor parameter to carry a full ``pl.Tensor[[...],
        dtype]`` annotation (a bare ``pl.Tensor`` has no shape to read and
        raises). Dynamic dims (``pl.dynamic`` / ``bind_dynamic``) need no value —
        the artifact is extent-independent. Scalar parameters need no value
        either: a ``pl.Scalar`` parameter is a **runtime value**, so it stays a
        real parameter in the artifact, its value arrives at dispatch, and —
        like a dynamic dim — it drops out of the cache key. A signature-mode
        call therefore always shares a cache entry with an equivalent
        ``compile(*sample_tensors)`` call. A value supplied anyway is inert: a
        signature default is accepted silently, an explicit keyword warns
        (that form used to specialize), and ``pl.RUNTIME`` is accepted here as
        the no-op it now is. The sample-argument path takes the scalar as an
        ordinary argument and rejects ``pl.RUNTIME``, which is a compile-time
        marker rather than a value.

        Example::

            M = pl.dynamic("M")

            @pl.jit
            def my_kernel(
                x: pl.Tensor[[M, 4096], pl.BF16],
                w: pl.Tensor[[4096, 4096], pl.BF16],
                out: pl.Out[pl.Tensor[[M, 4096], pl.BF16]],
                num_tokens: pl.Scalar[pl.INT32],
            ):
                ...

            from pypto.runtime import ChipWorker, ExecutionMode, RunConfig

            worker = ChipWorker(config=RunConfig(execution_mode=ExecutionMode.ONBOARD))

            # From sample tensors (shape/dtype read; contents ignored):
            compiled = my_kernel.compile(sample_x, sample_w, sample_out, 128)

            # Or straight from the (fully-annotated) signature — no tensors,
            # and no value for num_tokens: it is a runtime value, supplied on
            # each dispatch below. Calling my_kernel(...) eagerly with a
            # different num_tokens reuses this same artifact.
            compiled = my_kernel.compile()

            w_dev = worker.alloc_tensor(real_w.shape, real_w.dtype, init=real_w)
            h = worker.register(compiled)
            for batch in stream:
                h(batch.x, w_dev, batch.out, batch.num_tokens)

        Args:
            *args: Positional arguments matching the decorated function's
                params. Tensor values are inspected for shape/dtype only; their
                contents are not read. Omit **all** positional args to compile
                straight from the signature annotations instead.
            **kwargs: Keyword arguments. A ``config`` keyword, if present, is
                a ``RunConfig``. Signature mode needs no scalar values; one
                passed by name is validated and then ignored, and an explicit
                scalar keyword warns rather than specializing.

        Returns:
            The cached ``CompiledProgram`` for this specialization.
        """
        compiled, _ordered_args, _run_config = self._resolve_compiled(args, kwargs, allow_signature_mode=True)
        return compiled

    def warmup(self, *args: Any, **kwargs: Any) -> Any:
        """Compile and prepare all device binaries without executing the kernel.

        Uses the same arguments, configuration, specialization, and in-process
        cache as :meth:`compile`. Fully annotated tensors need no sample
        allocation, and scalar parameters need no value at all — they are
        runtime values, so one preparation serves every value.

        Unlike :meth:`compile`, this also assembles the kernel and orchestration
        binaries for every chip-level build before returning. It creates no
        runtime worker, initializes no NPU, and executes no kernel. The build
        host still needs the target compiler, SDK, and runtime dependencies.
        This method does not enable automatic persistent caching.

        Args:
            *args: Optional sample arguments accepted by :meth:`compile`.
                Tensor contents are not read.
            **kwargs: Kernel arguments and an optional ``config=RunConfig(...)``.
                Omit tensor arguments to use the complete tensor annotations.

        Returns:
            The same ``CompiledProgram`` or ``DistributedCompiledProgram``
            selected by :meth:`compile`, with all device binaries prepared.
            Execution remains a separate operation on the returned object.

        Raises:
            TypeError: The compiled result does not support device-free warmup.
            RuntimeError: A required binary cannot be prepared. Compiler and
                configuration errors propagate; a later warmup can retry.
        """
        from pypto.ir.compiled_program import CompiledProgram  # noqa: PLC0415
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415

        compiled = self.compile(*args, **kwargs)
        if isinstance(compiled, DistributedCompiledProgram):
            from pypto.runtime.distributed_runner import _assemble_chip_callables  # noqa: PLC0415

            _assemble_chip_callables(compiled)
        elif isinstance(compiled, CompiledProgram):
            if compiled.orchestration_names:
                for name in compiled.orchestration_names:
                    compiled[name].load()
            else:
                compiled.load()
        else:
            raise TypeError(
                f"@pl.jit function '{self.__name__}': device-free warmup is not supported "
                f"for compiled result {type(compiled).__name__}"
            )
        return compiled

    @capture_namespaces()
    def specialize(self, *args: Any, **kwargs: Any) -> _ir.Program:
        """Specialize this JIT function and return its **pre-pass** IR.

        The step [`lower`][pypto.language.JITFunction.lower] takes before it
        runs the pass pipeline: entry and every transitive dep are specialized
        into ``@pl.program`` source and parsed, and the parsed program is
        returned untransformed. No passes, no code generation, no ``ptoas``, no
        device, and the compiled-program cache is neither read nor written.

        Use this to hand a JIT kernel to a consumer that wants to drive the
        pass pipeline itself — most notably ``ir.compile(program,
        output_dir=...)``, which runs passes *and* code generation and so must
        be given the program before any pass has touched it. Passing
        ``lower()``'s result there would run the pipeline a second time.

        Two JIT kernels that specialize to the same program compare equal
        *after* passes, not before: the specializer renames SSA-rebound locals
        (``out`` becomes ``out_v1``), which canonicalization removes. Compare
        ``lower()`` output when asserting equivalence against a hand-written
        ``@pl.program``.

        Args:
            *args: Positional sample arguments matching the decorated function.
                Omit tensor samples to specialize from the annotations, which
                then must carry full shapes.
            **kwargs: Keyword sample arguments. Unlike ``lower()`` there is no
                ``config``: the pass pipeline never runs here, so a
                ``RunConfig`` would have nothing to configure.

        Returns:
            The parsed ``ir.Program``, before any pass has run.

        Raises:
            TypeError: ``config=`` was passed. Accepting it silently would let
                a caller believe a strategy or diagnostics setting shaped the
                returned IR, when no pass ran to read it.
        """
        import pypto.language as pl  # noqa: PLC0415

        if "config" in kwargs:
            raise TypeError(
                f"@pl.jit function '{self.__name__}': specialize() does not accept config=. "
                "No pass runs here, so a RunConfig would have nothing to configure. Use "
                "lower(config=...) for post-pass IR, or compile(config=...) to build an artifact."
            )
        specialization, _ = self._resolve_specialization(args, kwargs, allow_signature_mode=True)
        return self._compile_to_program(
            specialization.tensor_meta,
            specialization.scalar_dtypes,
            specialization.constexpr_values,
            specialization.per_func_dyn,
            pl,
        )

    @property
    def param_names(self) -> tuple[str, ...]:
        """Declared parameter names, in signature order (``self`` excluded)."""
        return tuple(self._param_names())

    @property
    def output_param_names(self) -> tuple[str, ...]:
        """Parameters the kernel writes — ``pl.Out[...]`` and ``pl.InOut[...]``.

        In declaration order, so the tuple stays aligned with the callee's
        return order. A caller that materialises tensors for this kernel uses
        it to decide which of them are results to validate.
        """
        out_params, inout_params, _, _, _ = _classify_params(_ast_analysis.get_func_def(self._func))
        written = set(out_params) | set(inout_params)
        return tuple(p for p in self._param_names() if p in written)

    @capture_namespaces()
    def lower(self, *args: Any, **kwargs: Any) -> _ir.Program:
        """Specialize this JIT function and return its post-pass IR.

        A ``config=RunConfig(...)`` keyword controls pass execution through its
        strategy, diagnostics, dependency-analysis, memory-planner, and platform
        fields. Runtime and artifact fields are ignored. This method does not
        run code generation, invoke ``ptoas``, execute on a device, write
        artifacts, or access the compiled-program cache.

        Args:
            *args: Positional sample arguments matching the decorated function.
                Omit tensor samples to specialize from fully shaped annotations.
            **kwargs: Keyword sample arguments and an optional ``config``. A
                scalar parameter needs no value in signature mode: it stays a
                symbolic ``pl.Scalar`` parameter in the returned program.

        Returns:
            The specialized ``ir.Program`` after configured passes.
        """
        import pypto.language as pl  # noqa: PLC0415
        from pypto.ir.compile import _run_pass_pipeline  # noqa: PLC0415

        specialization, run_config = self._resolve_specialization(
            args,
            kwargs,
            allow_signature_mode=True,
        )
        pre_pass, rename_map = self._compile_to_program_with_rename_map(
            specialization.tensor_meta,
            specialization.scalar_dtypes,
            specialization.constexpr_values,
            specialization.per_func_dyn,
            pl,
        )
        lower_kwargs = _run_config_lower_kwargs(run_config) if run_config is not None else {}
        platform = run_config.platform if run_config is not None else None
        try:
            return _run_pass_pipeline(
                pre_pass,
                operation="lower",
                platform=platform,
                inherit_outer_report_instruments=False,
                **lower_kwargs,
            ).transformed_program
        except Exception as exc:
            rewritten = _rewrite_jit_error(exc, rename_map)
            if rewritten is exc:
                raise
            raise rewritten from exc

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def _compile(
        self,
        tensor_meta: dict[str, TensorMeta],
        scalar_dtypes: dict[str, DataType],
        constexpr_values: dict[str, str],
        per_func_dyn: dict[int, dict[str, dict[int, DynDim]]],
        pl: Any,
        **ir_compile_kwargs: Any,
    ) -> Any:
        """Specialize entry + deps into @pl.program source, parse, and compile.

        Runs the full compilation pipeline: pass pipeline + codegen via
        ``ir.compile()``.  Returns a ``CompiledProgram``
        containing the post-pass ``ir.Program`` and the generated output
        artifacts (orchestration C++, kernel MLIR).

        ``per_func_dyn`` is the per-function effective DynDim map computed in
        ``_bind_args``; reused here so ``resolve_dep_call_metadata``
        doesn't re-walk the dep graph on every cache miss.

        ``ir_compile_kwargs`` are forwarded verbatim to ``ir.compile()`` —
        compile-side knobs (``platform``, ``strategy``, ``dump_passes``,
        ``output_dir``, ``profiling``, diagnostics, ...) that the JIT caller
        derives from a ``RunConfig`` via ``RunConfig.compile_kwargs``.
        If no output directory is supplied, allocate one for this compilation
        so a fresh diagnostic request cannot overwrite a cached artifact.
        """
        from pypto.ir.compile import compile as ir_compile  # noqa: PLC0415

        contexts = self._build_contexts(tensor_meta, scalar_dtypes, constexpr_values, per_func_dyn)
        class_name = f"_jit_{self.__name__}"
        specializer = Specializer(class_name, contexts)
        source = specializer.specialize()
        rename_map = specializer.rename_map
        try:
            parsed = pl.parse(source, filename=self._diagnostic_filename, source_map=specializer.source_map)
            skip_ptoas = not _ptoas_available()
            if ir_compile_kwargs.get("output_dir") is None:
                ir_compile_kwargs["output_dir"] = make_default_output_dir(parsed.name)
            return ir_compile(parsed, skip_ptoas=skip_ptoas, **ir_compile_kwargs)
        except Exception as exc:
            rewritten = _rewrite_jit_error(exc, rename_map)
            if rewritten is exc:
                raise
            raise rewritten from exc

    def _compile_to_program(
        self,
        tensor_meta: dict[str, TensorMeta],
        scalar_dtypes: dict[str, DataType],
        constexpr_values: dict[str, str],
        per_func_dyn: dict[int, dict[str, dict[int, DynDim]]],
        pl: Any,
    ) -> Any:
        """Specialize entry + deps and return the parsed pre-pass ``ir.Program``."""
        program, _rename_map = self._compile_to_program_with_rename_map(
            tensor_meta,
            scalar_dtypes,
            constexpr_values,
            per_func_dyn,
            pl,
        )
        return program

    def _compile_to_program_with_rename_map(
        self,
        tensor_meta: dict[str, TensorMeta],
        scalar_dtypes: dict[str, DataType],
        constexpr_values: dict[str, str],
        per_func_dyn: dict[int, dict[str, dict[int, DynDim]]],
        pl: Any,
    ) -> tuple[Any, dict[str, str]]:
        """Return the parsed pre-pass program and specializer rename map."""
        contexts = self._build_contexts(tensor_meta, scalar_dtypes, constexpr_values, per_func_dyn)
        class_name = f"_jit_{self.__name__}"
        specializer = Specializer(class_name, contexts)
        source = specializer.specialize()
        rename_map = specializer.rename_map
        try:
            program = pl.parse(source, filename=self._diagnostic_filename, source_map=specializer.source_map)
            return program, rename_map
        except Exception as exc:
            rewritten = _rewrite_jit_error(exc, rename_map)
            if rewritten is exc:
                raise
            raise rewritten from exc

    def _build_contexts(
        self,
        tensor_meta: dict[str, TensorMeta],
        scalar_dtypes: dict[str, DataType],
        constexpr_values: dict[str, str],
        per_func_dyn: dict[int, dict[str, dict[int, DynDim]]],
    ) -> list[SpecializeContext]:
        """Build SpecializeContext list for entry + every transitive dep.

        Walks the JIT call graph from ``_get_dep_graph()``:

        - Tensor / scalar metadata is resolved top-down (caller-first), so
          each dep is built using its actual caller's resolved metadata.
          For nested deps the caller may itself be another dep, not the
          entry. DynDim entries inside the metas propagate naturally as
          metas are forwarded into deps via ``resolve_dep_call_metadata``.
        - Each context's ``dep_names`` is the set of JIT deps that the
          context's function *directly* calls, named as its source calls
          them. The body transformer uses this — plus ``dep_func_names``,
          which maps those call names onto the generated function names they
          differ from under an aliased import — to rewrite nested
          ``dep(args)`` calls into the ``self.dep(args)`` form required by
          multi-function ``@pl.program``.

        The returned list is in leaf-first order (deps before their
        callers) so the generated source defines callees before callers.
        """
        _, callers_by_id, callees_by_id, _ = self._get_dep_graph()
        empty_dyn: dict[str, dict[int, DynDim]] = {}

        # Map each Python function id → its JIT ``_func_type`` so meta
        # resolution downstream can gate dep discovery on the caller's type
        # (a host orchestrator additionally admits ``orchestration`` deps).
        # Expand the graph into one generated function per compile-time binding.
        # A dep called with two different ``pl.constexpr`` values is two
        # functions here, not an error. The same walk backs the cache key, so
        # the two cannot disagree.
        plan = self._resolve_constexpr_bindings(constexpr_values)
        entry_key: _call_binding.VariantKey = (
            id(self._func),
            _call_binding.freeze_binding(constexpr_values),
        )

        func_type_by_id: dict[int, str] = {id(self._func): self._func_type}
        for key in plan.order:
            jit_func = plan.jit_func[key]
            func_type_by_id[id(jit_func._func)] = jit_func._func_type

        # One generated ``@pl.function`` name per variant, unique across the
        # program. Two names can collide for two reasons: two distinct deps
        # sharing a ``__name__`` (two modules each defining ``helper``), and
        # two bindings of one dep. Either way the later claimant is suffixed,
        # because emitting both as ``def helper`` made the parser reject the
        # program with a bare ``Duplicate function name "helper"``.
        # Entry first, so a clash never moves the name the user called.
        gen_names = _allocate_generated_names(plan.naming_order(entry_key))

        # Per variant, how its body's calls name the functions they reach.
        # ``dep_call_variants`` is keyed by ``(call name, ordinal)`` because one
        # body can now call one name at two constants and must reach a different
        # generated function at each site. ``dep_func_names`` keeps the
        # per-name view for the single-binding case, where every ordinal agrees.
        dep_func_names_by_variant: dict[_call_binding.VariantKey, dict[str, str]] = {}
        dep_call_variants_by_variant: dict[_call_binding.VariantKey, dict[tuple[str, int], str]] = {}
        for caller_key, targets in plan.call_targets.items():
            first_by_name: dict[str, str] = {}
            reached_by_name: dict[str, set[str]] = {}
            per_site: dict[tuple[str, int], str] = {}
            # Sorted by ``(call name, ordinal)``, so the first hit per name is
            # its ordinal 0 — the target ``dep_func_names`` should name.
            for (call_name, ordinal), dep_key in sorted(targets.items(), key=lambda item: item[0]):
                per_site[(call_name, ordinal)] = gen_names[dep_key]
                first_by_name.setdefault(call_name, gen_names[dep_key])
                reached_by_name.setdefault(call_name, set()).add(gen_names[dep_key])
            dep_func_names_by_variant[caller_key] = first_by_name
            # Left empty unless some call name really does reach more than one
            # generated function: the per-name map already covers the rest, and
            # an empty map is what tells the specializer it can skip numbering
            # this body's call sites.
            dep_call_variants_by_variant[caller_key] = (
                per_site if any(len(reached) > 1 for reached in reached_by_name.values()) else {}
            )

        # Walk caller-first (reverse of leaf-first order) so each variant's
        # caller metadata is already resolved when we get to it; collect
        # contexts caller-first, then reverse to restore leaf-first emit order.
        resolved: dict[
            _call_binding.VariantKey,
            tuple[
                dict[str, TensorMeta],
                dict[str, DataType],
            ],
        ] = {entry_key: (tensor_meta, scalar_dtypes)}

        contexts: list[SpecializeContext] = []
        for key in reversed(plan.order):
            dep = plan.jit_func[key]
            if key == entry_key:
                contexts.append(
                    build_specialize_context(
                        func=self._func,
                        func_name=gen_names[key],
                        func_type=self._func_type,
                        level=self._level,
                        tensor_meta=tensor_meta,
                        scalar_dtypes=scalar_dtypes,
                        constexpr_values=constexpr_values,
                        dep_names=callees_by_id[id(self._func)],
                        dep_func_names=dep_func_names_by_variant.get(key, {}),
                        dep_call_variants=dep_call_variants_by_variant.get(key, {}),
                        auto_scope=self._auto_scope,
                    )
                )
                continue
            # Each variant reads the call site that produced it, so its
            # metadata is the state live at *that* call rather than at the
            # first call sharing the name. Call sites that collapse onto one
            # variant still have to agree on shapes/dtypes, as before.
            origin = plan.origin.get(key)
            if origin is not None:
                caller_key, dep_call_name, call_node = origin
            else:
                # No call node to pin to (see the fallback in
                # ``expand_constexpr_variants``); resolve as before variants
                # existed, from the first recorded caller.
                caller_obj, dep_call_name = callers_by_id[id(dep._func)][0]
                caller_key = next((other for other in resolved if other[0] == id(caller_obj)), entry_key)
                call_node = None
            caller_func = plan.jit_func[caller_key]._func
            c_meta, c_sd = resolved[caller_key]
            caller_ftype = func_type_by_id.get(id(caller_func), "orchestration")
            dep_meta, dep_sd = _tensor_meta.resolve_dep_call_metadata(
                dep,
                caller_func,
                c_meta,
                c_sd,
                per_func_dyn.get(id(dep._func), empty_dyn),
                caller_func_type=caller_ftype,
                dep_call_name=dep_call_name,
                call_node=call_node,
                caller_constexpr=plan.binding[caller_key],
                discover_deps=_discover_dep_bindings,
            )
            resolved[key] = (dep_meta, dep_sd)
            contexts.append(
                build_specialize_context(
                    func=dep._func,
                    func_name=gen_names[key],
                    func_type=dep._func_type,
                    level=dep._level,
                    tensor_meta=dep_meta,
                    scalar_dtypes=dep_sd,
                    constexpr_values=plan.binding[key],
                    dep_names=callees_by_id[id(dep._func)],
                    dep_func_names=dep_func_names_by_variant.get(key, {}),
                    dep_call_variants=dep_call_variants_by_variant.get(key, {}),
                    auto_scope=dep._auto_scope,
                    external_core_type=dep._external_core_type,
                    external_aic_source=dep._external_aic_source,
                    external_aiv_source=dep._external_aiv_source,
                    external_dual_aiv_dispatch=dep._external_dual_aiv_dispatch,
                    external_include_dirs=dep._external_include_dirs,
                )
            )
        contexts.reverse()
        return contexts

    def __repr__(self) -> str:
        return f"JITFunction({self.__name__!r}, func_type={self._func_type!r})"


# ---------------------------------------------------------------------------
# Dep auto-discovery (defined after JITFunction to avoid forward reference)
# ---------------------------------------------------------------------------


def _discover_dep_bindings(
    func: Any, caller_func_type: str = "orchestration"
) -> list[_call_binding.DepBinding]:
    """Discover JIT dep functions called by ``func``, with their call names.

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
    func_def = _ast_analysis.get_func_def(func)

    called_names = _ast_analysis.collect_all_called_names(func_def)

    all_vars = func_name_lookup(func)

    allowed_dep_types: set[str] = {"incore", "inline", "opaque", "extern", "graph"}
    if caller_func_type == "host":
        allowed_dep_types.add("orchestration")

    deps: list[_call_binding.DepBinding] = []
    seen: set[str] = set()
    for name in called_names:
        obj = all_vars.get(name)
        if isinstance(obj, JITFunction) and obj._func_type in allowed_dep_types and name not in seen:
            deps.append(_call_binding.DepBinding(call_name=name, dep=obj))
            seen.add(name)
    return deps


# Cross-module private helper: the test suite (tests/ut/jit/test_decorator.py) reaches
# it by underscore import; pyright's unused analysis discounts cross-module refs.
def _discover_deps(  # pyright: ignore[reportUnusedFunction]
    func: Any, caller_func_type: str = "orchestration"
) -> list[JITFunction]:
    """Discover JIT dep functions called by ``func``, dropping their call names.

    Thin view over :func:`_discover_dep_bindings` for callers that only need
    the functions themselves. Anything that has to find the call site in the
    caller's source must use the bindings instead — see ``DepBinding``.
    """
    return [binding.dep for binding in _discover_dep_bindings(func, caller_func_type)]


def _generated_names_for(jit_func: JITFunction, base: str) -> tuple[str, ...]:
    """Every generated ``@pl.function`` name ``jit_func`` would occupy as ``base``.

    A single method for everything except an ``@pl.jit.extern`` mixed kernel,
    which the specializer renders as an AIC member, an AIV member, and a Group
    wrapper — three names derived from the same base.
    """
    if jit_func._func_type == "extern" and jit_func._external_core_type == "mixed":
        return (base, f"{base}_aic", f"{base}_aiv")
    return (base,)


def _allocate_generated_names(items: Sequence[tuple[Any, JITFunction]]) -> dict[Any, str]:
    """Map each caller-supplied key → the unique name its ``@pl.function`` gets.

    A generated ``@pl.program`` holds one method per generated function, so
    their names must be distinct — but two of them may legitimately want the
    same ``__name__``. That happens for two distinct deps (two modules each
    defining ``helper``, or two kernels built by the same factory) and for two
    ``pl.constexpr`` variants of a single dep, which are one Python function
    compiled against different constants. A clash is resolved by suffixing the
    later claimant ``__2``, ``__3``, … so every specialization survives instead
    of the parser rejecting the program with ``Duplicate function name
    "helper"``.

    ``items`` pairs each key with the function to name, entry first so a clash
    never moves the name the user called. The key is ``id(func)`` where one
    name per function is wanted and a ``(func, binding)`` variant key where one
    name per specialization is. Repeats of a key are ignored, and ``items``
    order is derived from source order, so the same call graph always yields
    the same names.
    """
    used: set[str] = set()
    names: dict[Any, str] = {}
    # Highest suffix already handed out per base name, so N functions sharing a
    # base cost O(N) probes overall rather than rescanning from 2 each time.
    next_suffix: dict[str, int] = {}
    for key, jit_func in items:
        if key in names:
            continue
        base = jit_func.__name__
        candidate = base
        suffix = next_suffix.get(base, 2)
        while not used.isdisjoint(_generated_names_for(jit_func, candidate)):
            candidate = f"{base}__{suffix}"
            suffix += 1
        next_suffix[base] = suffix
        names[key] = candidate
        used.update(_generated_names_for(jit_func, candidate))
    return names


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
      - ``graph``   → ``FunctionType.Graph`` (a recordable orchestration
        fragment; the host_build_graph runtime records its task topology on the
        first call and replays it after, so N calls cost one graph build rather
        than N). Requires ``RuntimeKind.HOST_BUILD_GRAPH`` — compile under a
        ``PassContext(runtime=...)``, or ``LegalizeGraphBoundary`` rejects it.
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

            def _deferred(f: Callable[..., Any]) -> JITFunction:
                return JITFunction(f, func_type=self._func_type, level=level, auto_scope=resolved_auto_scope)

            return _deferred
        return JITFunction(func, func_type=self._func_type, level=None, auto_scope=resolved_auto_scope)


def _resolve_extern_source(source: str | Any, func: Any) -> str:
    """Resolve an external-kernel source path to an absolute file string.

    A relative path is resolved against the directory of the file defining the
    decorated ``@pl.jit.extern`` stub (``inspect.getsourcefile``), matching the
    ``@pl.program`` external_source convention.
    """
    path = os.fspath(source)
    if not os.path.isabs(path):
        src_file = inspect.getsourcefile(func)
        if src_file is not None and not src_file.startswith("<"):
            path = os.path.join(os.path.dirname(src_file), path)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise ValueError(
            f"@pl.jit.extern source file not found: {path}. "
            "Provide an absolute path, or one relative to the file defining the kernel."
        )
    return path


def _resolve_extern_include_dirs(
    include_dirs: Sequence[str | os.PathLike[str]] | None,
    func: Any,
) -> tuple[str, ...]:
    """Resolve external-kernel include directories relative to the stub file."""
    if include_dirs is None:
        return ()
    if isinstance(include_dirs, (str, bytes, os.PathLike)):
        raise TypeError(
            f"@pl.jit.extern include_dirs must be a sequence of paths, got {type(include_dirs).__name__}"
        )

    src_file = inspect.getsourcefile(func)
    base_dir = None
    if src_file is not None and not src_file.startswith("<"):
        base_dir = os.path.dirname(src_file)

    resolved: list[str] = []
    for include_dir in include_dirs:
        path = os.fspath(include_dir)
        if not os.path.isabs(path) and base_dir is not None:
            path = os.path.join(base_dir, path)
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            raise ValueError(
                f"@pl.jit.extern include directory not found: {path}. "
                "Provide absolute paths, or paths relative to the file defining the kernel."
            )
        resolved.append(path)
    return tuple(resolved)


class _ExternKernelDecorator:
    """Sub-decorator for ``@pl.jit.extern`` — a hand-written C++ InCore kernel.

    The decorated function is a signature-only stub (``...`` body); its
    implementation is the referenced ``.cpp``. Forms::

        @pl.jit.extern(source="k_aiv.cpp", core_type="aiv")     # single core
        @pl.jit.extern(source="k_aic.cpp", core_type="aic")
        @pl.jit.extern(core_type="mixed",                       # AIC+AIV pair
                       aic_source="k.cpp", aiv_source="k.cpp")
        @pl.jit.extern(core_type="mixed",                       # AIC+2xAIV
                       aic_source="k.cpp", aiv_source="k.cpp",
                       dual_aiv_dispatch=True)
        @pl.jit.extern(source="k.cpp", core_type="aiv",        # extra headers
                       include_dirs=["include", "third_party/include"])

    A ``mixed`` kernel is dispatched as one ``MixedKernels`` submit: the
    specializer emits an AIC member, an AIV member, and a Group wrapper.
    Set ``dual_aiv_dispatch=True`` only when the external implementation is
    written for both AIV sub-lanes and uses the sub-block id to partition work.
    Source and include paths may be absolute or relative to the Python file
    defining the signature stub.
    """

    def __call__(
        self,
        func: Any = None,
        *,
        core_type: str = "aiv",
        source: str | Any = None,
        aic_source: str | Any = None,
        aiv_source: str | Any = None,
        dual_aiv_dispatch: bool = False,
        include_dirs: Sequence[str | os.PathLike[str]] | None = None,
    ) -> Any:
        if core_type not in ("aic", "aiv", "mixed"):
            raise ValueError(f"@pl.jit.extern core_type must be 'aic', 'aiv', or 'mixed', got {core_type!r}")
        if dual_aiv_dispatch and core_type != "mixed":
            raise ValueError("@pl.jit.extern dual_aiv_dispatch=True requires core_type='mixed'")

        def _make(f: Any) -> JITFunction:
            if core_type == "mixed":
                if aic_source is None or aiv_source is None:
                    raise ValueError(
                        "@pl.jit.extern(core_type='mixed') requires both aic_source= and aiv_source="
                    )
                ext_aic = _resolve_extern_source(aic_source, f)
                ext_aiv = _resolve_extern_source(aiv_source, f)
            else:
                single = source if source is not None else (aic_source or aiv_source)
                if single is None:
                    raise ValueError(f"@pl.jit.extern(core_type={core_type!r}) requires source=")
                resolved = _resolve_extern_source(single, f)
                ext_aic = resolved if core_type == "aic" else None
                ext_aiv = resolved if core_type == "aiv" else None
            ext_include_dirs = _resolve_extern_include_dirs(include_dirs, f)
            return JITFunction(
                f,
                func_type="extern",
                external_core_type=core_type,
                external_aic_source=ext_aic,
                external_aiv_source=ext_aiv,
                external_dual_aiv_dispatch=dual_aiv_dispatch,
                external_include_dirs=ext_include_dirs,
            )

        if func is None:
            return _make
        return _make(func)


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
        self.graph = _SubFunctionDecorator("graph", allow_level=False)
        self.extern = _ExternKernelDecorator()

    def __call__(self, func: Any = None, *, auto_scope: bool = True) -> Any:
        """Decorate an entry-point JIT function (Orchestration).

        Supports both the bare ``@pl.jit`` form and the parenthesized
        ``@pl.jit(auto_scope=False)`` form. Setting ``auto_scope=False`` opts
        out of compiler-inserted AUTO runtime scopes so the body can place
        them by hand with ``with pl.scope()``.
        """
        if func is None:

            def _deferred(f: Callable[..., Any]) -> JITFunction:
                return JITFunction(f, func_type="orchestration", level=None, auto_scope=auto_scope)

            return _deferred
        return JITFunction(func, func_type="orchestration", level=None, auto_scope=auto_scope)


# Singleton decorator object exposed as ``pl.jit``
jit = _JITDecorator()


__all__ = ["JITFunction", "jit"]
