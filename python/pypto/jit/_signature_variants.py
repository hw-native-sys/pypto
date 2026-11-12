# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Request-owned signature variants for call graphs containing external kernels.

Index each function's calls once, then walk each distinct caller signature once.
The metadata observer reads only each call's arguments from the live environment;
it neither copies that environment nor replays its prefix for every call site.
Provider names do not participate in binding or specialization decisions.
"""

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from pypto.pypto_core import DataType

from . import _arguments, _ast_analysis, _call_binding, _tensor_meta
from ._call_syntax import dependency_call
from ._specialize_types import DynDim, TensorMeta
from .specializer import SpecializeContext, build_specialize_context, func_name_lookup

if TYPE_CHECKING:
    from .decorator import JITFunction


_PeerScopeResolver = Callable[
    [_call_binding.VariantPlan, Mapping[_call_binding.VariantKey, str]],
    tuple[
        dict[_call_binding.VariantKey, Mapping[str, str | None]],
        set[_call_binding.VariantKey],
    ],
]


def _signature(
    names: Sequence[str], tensors: Mapping[str, TensorMeta], scalars: Mapping[str, DataType]
) -> tuple[Any, ...]:
    """Canonicalize equality relationships without folding dynamic sample bounds."""
    symbols: dict[str | int | tuple[int, int], int] = {}
    result: list[Any] = []
    for name in names:
        if name in tensors:
            meta = tensors[name]
            shape: list[Any] = []
            for axis in range(len(meta.shape)):
                dim = meta.actual_dim(axis)
                if isinstance(dim, int):
                    shape.append(("static", dim))
                else:
                    key = (
                        (id(dim) if dim.synthesized else dim.literal) if dim is not None else (id(meta), axis)
                    )
                    shape.append(("dynamic", symbols.setdefault(key, len(symbols))))
            # An omitted layout may inherit a downstream declaration; explicit
            # ND overrides it. Merging them makes specialization call-order dependent.
            result.append((name, tuple(shape), meta.dtype, meta.layout))
        elif name in scalars:
            result.append((name, scalars[name]))
        else:
            result.append((name, None))
    return tuple(result)


@dataclass
class _Variant:
    function: "JITFunction"
    tensors: dict[str, TensorMeta]
    scalars: dict[str, DataType]
    constants: dict[str, str]
    targets: dict[tuple[str, int], int]


class _DescriptorMetas:
    """Give unproven view extents their own request-local descriptor symbols."""

    def __init__(self) -> None:
        # Retain the source object along with its id so later temporary metadata
        # cannot reuse that id and accidentally acquire an equality constraint.
        self.values: dict[int, tuple[TensorMeta, TensorMeta]] = {}

    def for_call(self, meta: TensorMeta, owner: str) -> TensorMeta:
        if not meta.inexact_axes:
            return meta
        if id(meta) in self.values:
            return self.values[id(meta)][1]
        shape: list[int | DynDim] = []
        for axis, dimension in enumerate(meta.shape):
            if axis in meta.inexact_axes:
                extent = meta.actual_dim(axis)
                if extent is None:
                    extent = replace(
                        _tensor_meta.synthesized_dyn_dim(
                            owner, f"external_argument_{len(self.values)}", axis
                        ),
                        static_bound=dimension.static_bound if isinstance(dimension, DynDim) else dimension,
                    )
                shape.append(extent)
            else:
                shape.append(dimension)
        normalized = replace(meta, shape=tuple(shape), inexact_axes=frozenset(), runtime_dims={})
        self.values[id(meta)] = (meta, normalized)
        return normalized


def build_contexts(
    entry: "JITFunction",
    tensors: dict[str, TensorMeta],
    scalars: dict[str, DataType],
    constants: dict[str, str],
    dynamic_maps: dict[int, dict[str, dict[int, DynDim]]],
    *,
    discover: _call_binding.DiscoverDeps,
    allocate_names: Callable[[Sequence[tuple[Any, "JITFunction"]]], dict[Any, str]],
    resolve_peers: _PeerScopeResolver,
) -> list[SpecializeContext]:
    """Specialize external kernels and their ordinary helper callers together."""
    variants: list[_Variant] = []
    keys: dict[tuple[Any, ...], int] = {}
    order: list[int] = []
    active: set[int] = set()
    submitted: set[int] = set()
    # These caches belong to this request: globals and hook bindings can change
    # between requests, while every variant in this request sees the same index.
    indices: dict[int, dict[ast.stmt, list[tuple[str, int, ast.Call]]]] = {}
    dependencies: dict[int, dict[str, JITFunction]] = {}
    signatures: dict[int, tuple[list[str], list[str]]] = {}
    declared_scalars: dict[int, dict[str, DataType]] = {}
    descriptor_metas = _DescriptorMetas()

    def index(function: "JITFunction") -> dict[ast.stmt, list[tuple[str, int, ast.Call]]]:
        # Dependency discovery also depends on the wrapper's function type.
        identity = id(function)
        if identity in indices:
            return indices[identity]
        deps = {binding.call_name: binding.dep for binding in discover(function._func, function._func_type)}
        dependencies[identity] = deps
        calls: list[tuple[ast.stmt, str, ast.Call]] = []

        def walk(node: ast.AST, statement: ast.stmt | None = None) -> None:
            if isinstance(node, ast.stmt):
                statement = node
            view = dependency_call(node)
            if (
                isinstance(node, ast.Call)
                and view is not None
                and isinstance(view.func, ast.Name)
                and view.func.id in deps
            ):
                assert statement is not None
                if deps[view.func.id]._func_type == "extern" and (
                    not isinstance(statement, (ast.Expr, ast.Assign, ast.AnnAssign, ast.Return))
                    or statement.value is not node
                ):
                    raise ValueError(
                        "External calls must be standalone statements or the complete value of an "
                        "assignment/return; bind the call result before using it in a nested expression "
                        "or control-flow header"
                    )
                calls.append((statement, view.func.id, node))
            for child in ast.iter_child_nodes(node):
                walk(child, statement)

        walk(_ast_analysis.get_func_def(function._func))
        by_statement: dict[ast.stmt, list[tuple[str, int, ast.Call]]] = {}
        ordinals: dict[str, int] = {}
        for statement, name, call in sorted(calls, key=lambda item: (item[2].lineno, item[2].col_offset)):
            ordinal = ordinals.get(name, 0)
            ordinals[name] = ordinal + 1
            by_statement.setdefault(statement, []).append((name, ordinal, call))
        indices[identity] = by_statement
        return by_statement

    def visit(
        function: "JITFunction",
        tensor_meta: dict[str, TensorMeta],
        scalar_dtypes: dict[str, DataType],
        constexpr: dict[str, str],
    ) -> int:
        # Distinct wrappers may emit different core kinds for one callable.
        identity = id(function)
        if identity in active:
            raise TypeError(f"@pl.jit: recursive dependency {function.__name__!r} cannot be specialized")
        if identity not in declared_scalars:
            declared_scalars[identity] = _arguments.param_scalar_dtypes(function._func)
        # Literals carry no caller metadata; the helper's declaration still
        # types downstream uses. Explicit declarations also override inherited
        # types before signature lookup and recursive dependency resolution.
        scalar_dtypes = {**scalar_dtypes, **declared_scalars[identity]}
        if identity not in signatures:
            signatures[identity] = (_arguments.constexpr_params(function._func), function._param_names())
        signature = signatures[identity]
        key = (
            identity,
            _call_binding.freeze_binding(constexpr),
            _signature(signature[1], tensor_meta, scalar_dtypes),
        )
        if key in keys:
            return keys[key]
        variant_id = len(variants)
        keys[key] = variant_id
        variant = _Variant(function, tensor_meta, scalar_dtypes, constexpr, {})
        variants.append(variant)
        active.add(identity)
        calls = index(function)
        namespace = func_name_lookup(function._func)
        local_names = frozenset(_ast_analysis.body_local_names(function._func))

        def observe(
            statement: ast.stmt,
            live: Mapping[str, TensorMeta],
            resolve_expr: Callable[[ast.expr], TensorMeta | None],
            resolve_scalar: Callable[[ast.expr], DataType | None],
        ) -> None:
            for name, ordinal, call in calls.get(statement, ()):
                dep = dependencies[identity][name]
                dep_id = id(dep)
                if dep_id not in signatures:
                    signatures[dep_id] = (_arguments.constexpr_params(dep._func), dep._param_names())
                values = (
                    _call_binding.resolve_call_site_constexpr(
                        dep.__name__,
                        signatures[dep_id],
                        call,
                        function.__name__,
                        constexpr,
                        namespace,
                        local_names,
                    )
                    if signatures[dep_id][0]
                    else {}
                )
                dep_tensors, dep_scalars = _tensor_meta.resolve_dep_call_metadata(
                    dep,
                    function._func,
                    tensor_meta,
                    scalar_dtypes,
                    dynamic_maps.get(dep_id, {}),
                    caller_func_type=function._func_type,
                    dep_call_name=name,
                    call_node=_tensor_meta.CallSiteMetadata(call, live, resolve_expr, resolve_scalar),
                    caller_constexpr=constexpr,
                    discover_deps=discover,
                    caller_identity=identity,
                )
                # A callee reads its argument descriptor. Parent bounds retained
                # for local inference must not become constants in that callee.
                for parameter, meta in dep_tensors.items():
                    dep_tensors[parameter] = descriptor_metas.for_call(meta, function.__name__)
                target = visit(dep, dep_tensors, dep_scalars, values)
                variant.targets[(name, ordinal)] = target
                if dependency_call(call) is not call:
                    submitted.add(target)

        if calls:
            _tensor_meta.extract_local_tensor_metas(
                function._func,
                seed_meta=tensor_meta,
                caller_func_type=function._func_type,
                constexpr_values=constexpr,
                discover_deps=discover,
                scan=_tensor_meta.LocalMetaScan(before_stmt=observe, scalar_dtypes=scalar_dtypes),
            )
        active.remove(identity)
        order.append(variant_id)
        return variant_id

    entry_id = visit(entry, tensors, scalars, constants)
    names = allocate_names([(entry_id, entry), *((i, variants[i].function) for i in order if i != entry_id)])
    # Resolve peers over the signature-expanded graph: one wrapper can have
    # several generated functions, each belonging to its actual Group instance.
    variant_keys = {
        i: (i, _call_binding.freeze_binding(variant.constants)) for i, variant in enumerate(variants)
    }
    peer_plan = _call_binding.VariantPlan(
        order=[variant_keys[i] for i in order],
        jit_func={variant_keys[i]: variant.function for i, variant in enumerate(variants)},
        binding={variant_keys[i]: variant.constants for i, variant in enumerate(variants)},
        origin={},
        call_targets={
            variant_keys[i]: {site: variant_keys[target] for site, target in variant.targets.items()}
            for i, variant in enumerate(variants)
        },
        submitted={variant_keys[i] for i in submitted},
    )
    peer_names, ambiguous_peer_scopes = resolve_peers(
        peer_plan, {variant_keys[i]: name for i, name in names.items()}
    )
    contexts: list[SpecializeContext] = []
    for variant_id in order:
        variant = variants[variant_id]
        function = variant.function
        targets = {site: names[target] for site, target in variant.targets.items()}
        first: dict[str, str] = {}
        for (call_name, _), target in targets.items():
            first.setdefault(call_name, target)
        contexts.append(
            build_specialize_context(
                func=function._func,
                func_name=names[variant_id],
                func_type=function._func_type,
                level=function._level,
                tensor_meta=variant.tensors,
                scalar_dtypes=variant.scalars,
                constexpr_values=variant.constants,
                dep_names=list(first),
                dep_func_names=first,
                dep_call_variants=targets,
                peer_func_names=peer_names.get(variant_keys[variant_id], {}),
                peer_func_scope_ambiguous=variant_keys[variant_id] in ambiguous_peer_scopes,
                auto_scope=function._auto_scope,
                external_core_type=function._external_core_type,
                external_aic_source=function._external_aic_source,
                external_aiv_source=function._external_aiv_source,
                external_dual_aiv_dispatch=function._external_dual_aiv_dispatch,
                external_include_dirs=function._external_include_dirs,
                requires_return_contract=variant_id in submitted,
                return_metas=(
                    _tensor_meta.specialized_return_metas(
                        function, variant.tensors, variant.constants, discover
                    )
                    if variant_id in submitted and function._func_type != "extern"
                    else None
                ),
            )
        )
    return contexts
