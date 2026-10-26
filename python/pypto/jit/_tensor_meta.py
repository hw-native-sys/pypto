# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Source-ordered local tensor metadata inference for JIT specialization.

Branch joins, rebinding, and recursive dependency returns share one analysis.
The caller supplies dependency discovery so this module never imports the
decorator at runtime; recursive walks preserve that resolver and request snapshot."""

import ast
import copy
import functools
import math
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir

from . import _arguments, _ast_analysis, _call_binding
from .specializer import (
    DynDim,
    ShapeDim,
    TensorMeta,
    _BodyTransformer,
    _classify_params,
    func_name_lookup,
)

if TYPE_CHECKING:
    from .decorator import JITFunction


# Prefix for the dynamic symbols ``@pl.jit`` invents when a local tensor's
# extent is only known at runtime. Reserved: user DSL code must not declare
# ``pl.dynamic()`` symbols starting with it.
_SYNTHESIZED_DYN_PREFIX = "_jitdyn_"


def synthesized_dyn_dim(owner: str, base: str, dim_idx: int) -> DynDim:
    """Build the placeholder ``DynDim`` for an extent only known at runtime.

    A local tensor sized by a runtime expression — ``pld.world_size()``,
    ``pl.tensor.read(cfg, [0])``, arithmetic over a ``DynDim`` — has no static
    extent to record. Rather than dropping the whole meta (which surfaces as
    ``_build_params``' "missing inferred tensor metadata" the moment the local
    crosses a dep call boundary), stamp a fresh symbol on that dim. The
    generated program then declares it via ``pl.dynamic`` and the kernel binds
    it from the actual argument's descriptor, exactly as a hand-written
    ``@pl.program`` would.

    The symbol is qualified by the function that owns the local. Two functions
    each holding a runtime-sized local named ``tmp`` would otherwise land on one
    name for unrelated extents. That is not *wrong* — a dynamic symbol carries no
    cross-function equality constraint, and each function emits its own
    "read dim k of the declaring argument" binding — but one name on two
    unrelated tensors makes a dumped program much harder to read.

    Args:
        owner: Name of the function whose body declares the local
        base: Name of the local the meta describes, for a readable symbol
        dim_idx: Index of the unresolved dim within that local's shape

    Returns:
        A ``synthesized=True`` DynDim named ``_jitdyn_<owner>_<base>_d<dim_idx>``
    """
    name = f"{_SYNTHESIZED_DYN_PREFIX}{owner}_{base}_d{dim_idx}"
    # ``static_bound=1`` matches the placeholder extent ``signature_tensor_meta``
    # uses for annotation-declared dynamic dims: the specialized program is
    # extent-independent, so no concrete value is available or needed here.
    return DynDim(name=name, literal=name, static_bound=1, synthesized=True)


def _build_dynvar_anchor_index(
    seed_meta: dict[str, TensorMeta],
) -> dict[str, list[tuple[str, int]]]:
    """Inverse map ``DynVar name → list of (param, dim_idx) anchor sites``.

    Lets ``[M, HIDDEN]`` (where ``M`` is a DynVar bound to a seeded param's
    dim) resolve via ``extract_local_tensor_metas`` to the parent dim's
    ``DynDim``.
    """
    anchors: dict[str, list[tuple[str, int]]] = {}
    for pname, meta in seed_meta.items():
        for i, dim in enumerate(meta.shape):
            if isinstance(dim, DynDim):
                anchors.setdefault(dim.name, []).append((pname, i))
    return anchors


def scan_dep_io(
    func: Any, caller_func_type: str = "orchestration", *, discover_deps: _call_binding.DiscoverDeps
) -> dict[str, tuple[list[str], list[str]]]:
    """Return ``call_name → (param_names, output_param_names)`` for every @pl.jit
    dep called from ``func``'s body.

    Keyed by the name the body *calls* the dep by, so an aliased import
    (``from mod import kernel as kern``) still matches the ``ast.Name`` at the
    call site.

    Used by ``extract_local_tensor_metas`` to propagate metas through
    ``v1, ..., vk = dep(args)`` assignments (each ``vi`` inherits the meta of
    the caller arg bound to the i-th output-like parameter). A dep with no
    output-like params is still recorded: ``_dep_return_metas`` needs
    ``param_names`` to bind the call site's args when it descends into the
    callee's body.

    ``output_param_names`` covers both ``pl.Out[...]`` and ``pl.InOut[...]``
    params — a caller can capture either from ``v = dep(...)`` — and is kept in
    declaration order so it stays aligned with the callee's return order (the
    positional target<->param zip in ``_dep_out_metas``).

    ``caller_func_type`` mirrors ``_discover_deps``'s gating: a host
    orchestrator also admits ``orchestration`` deps (its chip orchestrators).
    """
    out: dict[str, tuple[list[str], list[str]]] = {}
    for call_name, dep in discover_deps(func, caller_func_type):
        try:
            out_params, inout_params, _, _, _ = _classify_params(_ast_analysis.get_func_def(dep._func))
        except OSError:
            continue
        param_names = dep._param_names()
        output_set = set(out_params) | set(inout_params)
        output_params = [p for p in param_names if p in output_set]
        out[call_name] = (param_names, output_params)
    return out


class _DepScan(NamedTuple):
    """Everything the local-meta walk needs to know about one caller's deps.

    ``io`` and ``funcs`` are both keyed by the name the caller's *source*
    calls the dep by (see :class:`DepBinding`). ``seen`` holds the ``id()`` of
    every Python function already on the extraction stack, so the recursive
    descent :func:`_dep_return_metas` performs cannot loop. ``constexpr`` and
    ``scope`` are the caller's own ``pl.constexpr`` binding and namespace, from
    which that descent resolves the binding each call site gives its callee.
    """

    io: dict[str, tuple[list[str], list[str]]]
    funcs: "dict[str, JITFunction]"
    seen: frozenset[int]
    constexpr: Mapping[str, str]
    scope: "_StaticScope"
    owner: str
    # Keep runtime JIT recognition at the decorator boundary, also for callee returns.
    discover_deps: _call_binding.DiscoverDeps


def _target_names(target: ast.expr) -> list[str]:
    """Return the bound names of ``v = ...`` / ``v1, ..., vk = ...``.

    Empty for any other target shape (a subscript, an attribute, a nested
    unpack) — those are not local tensor rebindings this extractor models.
    """
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Tuple) and all(isinstance(e, ast.Name) for e in target.elts):
        return [e.id for e in target.elts if isinstance(e, ast.Name)]
    return []


def _return_element_names(func_def: ast.FunctionDef) -> list[str]:
    """Return the names a function's first value-carrying ``return`` hands back.

    ``return a, b`` → ``["a", "b"]``; ``return a`` → ``["a"]``. An element that
    is not a bare ``Name`` (a call, a subscript, a literal) yields ``""`` so the
    positional alignment with the caller's targets survives — the empty name
    simply resolves to no meta.
    """
    for node in ast.walk(func_def):
        if not (isinstance(node, ast.Return) and node.value is not None):
            continue
        elts = node.value.elts if isinstance(node.value, ast.Tuple) else [node.value]
        return [e.id if isinstance(e, ast.Name) else "" for e in elts]
    return []


def _dep_out_metas(
    call: ast.Call,
    dep_name: str,
    target: ast.expr,
    deps: _DepScan,
    local: Mapping[str, TensorMeta],
) -> dict[str, TensorMeta]:
    """For ``v1, ..., vk = dep(args)`` where ``dep`` has ``k`` ``Out`` params,
    return each ``vi``'s meta from the caller arg passed to the matching ``Out``
    parameter.

    Mapping handles both positional and keyword args. No-op when the dep has
    no ``Out`` params or when target/arity don't match — ``_dep_return_metas``
    then covers the callee-allocated case.
    """
    dep_params, out_params = deps.io[dep_name]
    names = _target_names(target)
    if not names or not out_params or len(names) != len(out_params):
        return {}
    mapping: dict[str, str | None] = {}
    for i, arg in enumerate(call.args):
        if i < len(dep_params):
            mapping[dep_params[i]] = arg.id if isinstance(arg, ast.Name) else None
    for kw in call.keywords:
        if kw.arg is not None:
            mapping[kw.arg] = kw.value.id if isinstance(kw.value, ast.Name) else None
    output_metas: dict[str, TensorMeta] = {}
    for out_param in out_params:
        caller_arg = mapping.get(out_param)
        if caller_arg is not None and caller_arg in local:
            output_metas[out_param] = local[caller_arg]
    dep = deps.funcs.get(dep_name)
    if dep is not None:
        # A returned Out buffer must retain the callee's declared packed format.
        _validate_dep_packed_fp4_dtypes(dep, output_metas)
    return {
        vname: output_metas[out_param]
        for vname, out_param in zip(names, out_params, strict=True)
        if out_param in output_metas
    }


class _DepReturn(NamedTuple):
    """What descending into a callee's ``return`` statement established.

    ``metas`` are the targets the descent typed. ``stale`` are the targets it
    proved it *cannot* type: the callee returns a named local its own extractor
    declined, so whatever the caller knew about that name before the call no
    longer describes it. Both are empty when the descent did not happen at all,
    which leaves the caller's pre-call fallback in charge.
    """

    metas: dict[str, TensorMeta]
    stale: frozenset[str]


_DEP_RETURN_DECLINED = _DepReturn({}, frozenset())


def _dep_return_metas(
    call: ast.Call,
    dep_name: str,
    target: ast.expr,
    deps: _DepScan,
    local: Mapping[str, TensorMeta],
) -> _DepReturn:
    """For ``v1, ..., vk = dep(args)`` where ``dep`` returns tensors it created
    itself, resolve each ``vi`` from the callee's own ``return`` statement.

    ``_dep_out_metas`` covers the in-place convention, where every returned
    tensor is also an ``Out`` parameter the *caller* allocated, so its meta is
    already in the caller's pool. A helper may instead ``pl.create_tensor`` its
    results and hand them back — the classic ``a, b = make_pair(x)`` inline
    preparation step — and then the shape and dtype exist only inside the
    callee. Re-run the extractor over the callee's body, seeded with the params
    this call site binds, and read the returned names' metas off that.

    A returned element that is a *named* callee local the descent could not type
    comes back in ``stale`` rather than silently absent. The caller would
    otherwise keep the meta the target carried before the call — demonstrably
    the wrong tensor, since the callee rebinds it — and hand that shape to the
    next dep. An element that is not a bare ``Name`` (a call, a literal)
    establishes nothing either way and is simply left out.

    Returns :data:`_DEP_RETURN_DECLINED` when the callee's source is
    unavailable, its return arity doesn't match the target, or it is already on
    the extraction stack.
    """
    dep = deps.funcs.get(dep_name)
    names = _target_names(target)
    if dep is None or not names or id(dep._func) in deps.seen:
        return _DEP_RETURN_DECLINED
    try:
        ret_names = _return_element_names(_ast_analysis.get_func_def(dep._func))
    except OSError:
        return _DEP_RETURN_DECLINED
    if len(ret_names) != len(names):
        return _DEP_RETURN_DECLINED
    dep_params, _ = deps.io[dep_name]
    seed_meta: dict[str, TensorMeta] = {}
    for dep_param, caller_arg in _call_binding.build_param_mapping(
        dep_params, _call_binding.call_arg_refs(call)
    ).items():
        # A ``SlicedArg`` (``chip_orch(x[r], ...)``) seeds nothing: that
        # per-rank dispatch form returns through ``Out`` params, so the
        # descent below never needs the sliced param's meta.
        if not isinstance(caller_arg, str):
            continue
        if caller_arg in local:
            seed_meta[dep_param] = local[caller_arg]
    _validate_dep_packed_fp4_dtypes(dep, seed_meta)
    callee_metas = extract_local_tensor_metas(
        dep._func,
        seed_meta=seed_meta,
        caller_func_type=dep._func_type,
        dep_seen=deps.seen,
        constexpr_values=_call_site_constexpr(dep, call, deps),
        discover_deps=deps.discover_deps,
        visible_only=True,
    )
    inherited_dims = {id(dim) for meta in seed_meta.values() for dim in meta.shape if isinstance(dim, DynDim)}
    instance_dims: dict[int, DynDim] = {}

    def instantiate(meta: TensorMeta, result_index: int) -> TensorMeta:
        shape: list[ShapeDim] = []
        for axis, dim in enumerate(meta.shape):
            if isinstance(dim, DynDim) and dim.synthesized and id(dim) not in inherited_dims:
                # One helper source can produce independently sized results at
                # different caller sites. Preserve actual within-call aliases
                # and inherited dimensions; object IDs never enter IR names.
                key = id(dim)
                if key not in instance_dims:
                    result = f"return_l{call.lineno}_c{call.col_offset}_r{result_index}"
                    instance_dims[key] = synthesized_dyn_dim(deps.owner, result, axis)
                dim = instance_dims[key]
            shape.append(dim)
        return TensorMeta(shape=tuple(shape), dtype=meta.dtype, layout=meta.layout)

    resolved = {
        v: instantiate(callee_metas[r], index)
        for index, (v, r) in enumerate(zip(names, ret_names, strict=True))
        if r in callee_metas
    }
    stale = {v for v, r in zip(names, ret_names, strict=True) if r and r not in callee_metas}
    return _DepReturn(resolved, frozenset(stale))


def _call_site_constexpr(dep: "JITFunction", call: ast.Call, deps: _DepScan) -> dict[str, str]:
    """The ``pl.constexpr`` binding ``call`` gives ``dep``; empty when there is none.

    A call site that cannot bind is reported, by name, when the variant plan
    resolves it for compilation. This descent only reads metadata, so it binds
    nothing rather than raising first with less context.
    """
    constexpr_names = _arguments.constexpr_params(dep._func)
    if not constexpr_names:
        return {}
    try:
        return _call_binding.resolve_call_site_constexpr(
            dep.__name__,
            (constexpr_names, dep._param_names()),
            call,
            "<caller>",
            deps.constexpr,
            deps.scope.namespace,
            deps.scope.shadowed,
        )
    except TypeError:
        return {}


def _dep_out_metas_or_return(
    call: ast.Call,
    dep_name: str,
    target: ast.expr,
    deps: _DepScan,
    local: Mapping[str, TensorMeta],
) -> _DepReturn:
    """Resolve one ``v1, ..., vk = dep(args)`` target, ``Out`` convention first.

    The ``Out`` rule validates the caller's buffer against the callee's packed
    format without descending into its body. Where both rules apply they agree,
    so it wins and never marks a target stale.
    """
    out_metas = _dep_out_metas(call, dep_name, target, deps, local)
    if out_metas:
        return _DepReturn(out_metas, frozenset())
    return _dep_return_metas(call, dep_name, target, deps, local)


def _apply_dep_return(local: dict[str, TensorMeta], dep_return: _DepReturn) -> None:
    """Fold one dep-call target's resolution into the source-ordered pool.

    Resolved names land; names the descent proved stale are dropped, so the
    caller's pre-call fallback cannot hand a replaced tensor's shape onward.
    """
    local.update(dep_return.metas)
    for name in dep_return.stale:
        local.pop(name, None)


def _fold_int_arith(op: ast.operator, lhs: int, rhs: int) -> int | None:
    """Fold a binary arithmetic op over two Python ints, or return None.

    Used by ``extract_local_tensor_metas._resolve_shape_elt`` to keep the
    shape-element resolver under the per-function branch limit. Dynamic
    operands are handled by the caller; this helper only sees ``int·int``.
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


def _subscript_slice_meta(
    sub: ast.Subscript,
    local: Mapping[str, TensorMeta],
    resolve_int: Callable[[ast.expr], int | None],
) -> TensorMeta | None:
    """Infer the ``TensorMeta`` of ``var = src[a:b, i, ...]`` subscript-slice sugar.

    The documented equivalent of ``pl.slice`` (see ``extract_local_tensor_metas``
    form 2): dtype is inherited from ``src``; each slice dim resolves to
    ``stop - start`` (``start`` defaults to 0), and an open upper bound ``a:``
    resolves to ``parent_dim - start`` — mirroring the parser's
    ``_build_subscript_slice_args``. A dim falls back to the parent extent when
    its bounds aren't static — a ``DynDim`` parent flows through transparently
    that way; a scalar index drops its dim (numpy-style rank reduction); dims
    past the supplied indices are implicit ``:`` and keep the parent extent.
    Returns ``None`` (skipped, leaving the clear ``_build_params`` error) when
    ``src`` is unknown, a step slice is used, or the index count exceeds
    ``src``'s rank.
    """
    src = sub.value
    if not isinstance(src, ast.Name) or src.id not in local:
        return None
    src_meta = local[src.id]
    slc = sub.slice
    indices = list(slc.elts) if isinstance(slc, ast.Tuple) else [slc]
    if len(indices) > len(src_meta.shape):
        return None
    dims: list[ShapeDim] = []
    for dim_idx, idx in enumerate(indices):
        if not isinstance(idx, ast.Slice):
            continue  # scalar index → rank-reducing, dim dropped
        if idx.step is not None:
            return None
        start = 0 if idx.lower is None else resolve_int(idx.lower)
        parent = src_meta.shape[dim_idx]
        if idx.upper is None:
            # Open upper bound ``a:`` — the parser bounds it at ``parent - start``
            # (see ``_build_subscript_slice_args``), so mirror that here instead
            # of falling back to the full parent extent for a nonzero ``start``.
            extent = parent - start if isinstance(parent, int) and isinstance(start, int) else None
        else:
            stop = resolve_int(idx.upper)
            extent = stop - start if isinstance(start, int) and isinstance(stop, int) else None
        dims.append(extent if extent is not None else parent)
    dims.extend(src_meta.shape[len(indices) :])  # trailing implicit ``:``
    return TensorMeta(shape=tuple(dims), dtype=src_meta.dtype, layout=src_meta.layout)


def _call_operand(call: ast.Call, index: int | None, name: str) -> ast.expr | None:
    """The argument ``call`` binds to parameter ``name``.

    Args:
        call: The operator call
        index: The parameter's positional slot, or None when it is keyword-only
        name: The parameter's keyword name

    Returns:
        The argument expression, or None when the call does not supply it
    """
    if index is not None and len(call.args) > index:
        return call.args[index]
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def _tensor_or_input_operand(call: ast.Call) -> ast.expr | None:
    """The first positional operand of ``pl.slice``/``pl.reshape`` or their
    ``pl.tensor.*`` siblings — two real spellings, two real parameter names
    for the same slot (``tensor_ops.py``'s ``tensor`` vs ``unified_ops.py``'s
    ``input``). A positional call is unaffected either way; only an
    explicit-keyword one needs both names checked.
    """
    operand = _call_operand(call, 0, "tensor")
    if operand is None:
        operand = _call_operand(call, 0, "input")
    return operand


def _shape_attr_source(node: ast.expr) -> str | None:
    """The tensor name ``node`` reads the shape of (``src.shape`` → ``"src"``), or None."""
    if isinstance(node, ast.Attribute) and node.attr == "shape" and isinstance(node.value, ast.Name):
        return node.value.id
    return None


def _constant_shape_alias(value: ast.Subscript) -> tuple[str, int] | None:
    """Shape indexing syntax the body transformer can replace directly."""
    src = _shape_attr_source(value.value)
    axis = value.slice
    if (
        src is not None
        and isinstance(axis, ast.Constant)
        and isinstance(axis.value, int)
        and not isinstance(axis.value, bool)
    ):
        return src, axis.value
    return None


def _extract_dim_alias(
    value: ast.expr | None, resolve_axis: Callable[[ast.expr], int | None]
) -> tuple[str, int] | None:
    """Return the source and axis for ``pl.tensor.dim(source, axis)`` or ``source.shape[axis]``.

    The specializer folds ``source.shape[axis]`` to the same dim ``pl.tensor.dim``
    names — the static extent, or ``pl.tensor.dim(source, axis)`` when dynamic.
    """
    if isinstance(value, ast.Subscript):
        return _constant_shape_alias(value)
    if not isinstance(value, ast.Call):
        return None
    if _flatten_dotted_call(value.func) != ("pl", "tensor", "dim") or len(value.args) > 2:
        return None
    if any(isinstance(arg, ast.Starred) for arg in value.args):
        return None
    # Bind the real ``dim(tensor, axis)`` signature before reading either
    # operand. Invalid duplicates or extra keywords must not invent metadata.
    bound = dict(zip(("tensor", "axis"), value.args))
    for keyword in value.keywords:
        if keyword.arg is None or keyword.arg not in ("tensor", "axis") or keyword.arg in bound:
            return None
        bound[keyword.arg] = keyword.value
    src_arg, dim_arg = bound.get("tensor"), bound.get("axis")
    if isinstance(src_arg, ast.Name) and dim_arg is not None:
        axis = resolve_axis(dim_arg)
        if axis is not None:
            return src_arg.id, axis
    return None


def _flatten_dotted_call(fn: ast.expr) -> tuple[str, ...] | None:
    """Full dotted path of a plain attribute/name chain, e.g. ``pld.tensor.all_to_all_v``
    -> ``("pld", "tensor", "all_to_all_v")``, or None if ``fn`` isn't a pure chain of
    ``Attribute`` nodes rooted at a ``Name`` (so this walker cannot reason about it at all —
    for example the callee of a higher-order call, or an attribute on a subscript)."""
    segments: list[str] = []
    node = fn
    while isinstance(node, ast.Attribute):
        segments.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    segments.append(node.id)
    segments.reverse()
    return tuple(segments)


# Qualified (3+-segment) call dispatch: full dotted path -> a handler that
# computes the result's TensorMeta, or None for an op documented as
# returning its own rebind target's type unchanged (#2638).
_QualifiedCallHandlers = dict[tuple[str, ...], Callable[[ast.Call, str | None], TensorMeta | None] | None]


def _qualified_call_meta(
    fn: ast.expr,
    value: ast.Call,
    named_target: str | None,
    qualified_call_handlers: _QualifiedCallHandlers,
) -> tuple[TensorMeta | None, bool]:
    """Metadata effect of a qualified call: ``(meta, preserve_existing)``.

    Unlisted paths return ``(None, False)`` — the caller then falls through to
    ``local.pop()``, exactly like any other unrecognized call.
    """
    path = _flatten_dotted_call(fn)
    if path is None or path not in qualified_call_handlers:
        return None, False
    handler = qualified_call_handlers[path]
    if handler is not None:
        return handler(value, named_target), False
    return None, True


def _same_operand_meta(
    local: Mapping[str, TensorMeta], operand_index: int, operand_name: str
) -> Callable[[ast.Call, str | None], TensorMeta | None]:
    """Handler factory for a collective documented as returning a specific
    argument's own type unchanged (e.g. ``broadcast``'s ``target``,
    ``barrier``'s ``signal``). Reads that argument's tracked metadata
    directly rather than assuming the call's LHS name is the same name as
    the operand: ``old = pld.tensor.broadcast(window, signal, root=0)`` must
    get ``window``'s metadata, not whatever ``old`` already held — a blind
    ``preserve_existing`` would silently serve stale metadata when the two
    names differ (#2638 review). When the two names *are* the same (the
    documented self-rebind idiom), this reads ``local[operand.id]`` before
    the assignment takes effect, which is exactly what preserving would
    have returned anyway — so the self-rebind case is unaffected.
    """

    def handler(call: ast.Call, target: str | None = None) -> TensorMeta | None:
        operand = _call_operand(call, operand_index, operand_name)
        if not isinstance(operand, ast.Name) or operand.id not in local:
            return None
        return local[operand.id]

    return handler


def _alias_dim(alias: tuple[str, int] | None, local: Mapping[str, TensorMeta]) -> ShapeDim | None:
    """Dim ``k`` of tensor ``P`` for a ``(P, k)`` dim alias, or None when ``P`` is untracked."""
    if alias is None:
        return None
    src, axis = alias
    src_meta = local.get(src)
    if src_meta is None:
        return None
    rank = len(src_meta.shape)
    if axis < 0:
        axis += rank
    return src_meta.shape[axis] if 0 <= axis < rank else None


def _assignment_parts(stmt: ast.stmt) -> tuple[list[ast.expr], ast.expr | None] | None:
    """Return all targets and the value for a supported assignment."""
    if isinstance(stmt, ast.Assign):
        return stmt.targets, stmt.value
    if isinstance(stmt, ast.AnnAssign):
        return [stmt.target], stmt.value
    return None


_ScalarExpr = tuple[type[ast.operator | ast.unaryop], tuple["_DimValue", ...]]
_DimValue = ShapeDim | _ScalarExpr | None
_BoundShapeDims = dict[int, tuple[_ScalarExpr, DynDim]]


def _bind_scalar_shape_dim(
    value: _DimValue, bound_dim: DynDim | None, bound_dims: _BoundShapeDims | None
) -> ShapeDim | None:
    """Give a scalar expression an axis symbol, shared only within one result shape."""
    if not isinstance(value, tuple):
        return value
    if bound_dim is None or bound_dims is None:
        return bound_dim
    # Keep the snapshot alive so a later inline expression cannot reuse its id.
    return bound_dims.setdefault(id(value), (value, bound_dim))[1]


def _declared_dim_values(
    func_def: ast.FunctionDef, seed_meta: Mapping[str, TensorMeta]
) -> dict[ast.expr, DynDim]:
    """Recognize body DynVar declarations by their assigned seed-anchor name."""
    anchors = {dim.name: dim for meta in seed_meta.values() for dim in meta.shape if isinstance(dim, DynDim)}
    declarations: dict[ast.expr, DynDim] = {}
    for stmt in ast.walk(func_def):
        if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call):
            continue
        if _flatten_dotted_call(stmt.value.func) not in (("pl", "dynamic"), ("dynamic",)):
            continue
        for target in stmt.targets:
            if isinstance(target, ast.Name) and target.id in anchors:
                declarations[stmt.value] = anchors[target.id]
    return declarations


def _merge_tensor_metas(lhs: TensorMeta | None, rhs: TensorMeta | None) -> TensorMeta | None:
    """Merge equal tensor types without losing known layout or dynamic-axis identity."""
    if lhs is None or rhs is None or lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
        return None
    # An omitted annotation layout spells ND, not an arbitrary layout wildcard.
    lhs_layout = _ir.TensorLayout.ND if lhs.layout is None else lhs.layout
    rhs_layout = _ir.TensorLayout.ND if rhs.layout is None else rhs.layout
    if lhs_layout != rhs_layout:
        return None
    if any(
        isinstance(old, DynDim) and old.synthesized and old is not new
        for old, new in zip(lhs.shape, rhs.shape, strict=True)
    ):
        return None
    # Keep explicit ND so a later helper annotation cannot overwrite it as unspecified.
    return rhs if lhs.layout is None and rhs.layout is not None else lhs


def _resolve_dim_binop(elt: ast.BinOp, resolve: Callable[[ast.expr], _DimValue]) -> _DimValue:
    lhs = resolve(elt.left)
    rhs = resolve(elt.right)
    if isinstance(lhs, int) and isinstance(rhs, int):
        return _fold_int_arith(elt.op, lhs, rhs)
    # Keep the operation and operand snapshots, not a sample dimension value.
    # Unknown or invalid subexpressions still invalidate the whole expression.
    if (
        lhs is not None
        and rhs is not None
        and isinstance(elt.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv))
        and not (isinstance(elt.op, ast.FloorDiv) and rhs == 0)
    ):
        return type(elt.op), (lhs, rhs)
    return None


def _stmt_calls_dep(stmt: ast.stmt, dep_name: str | None, stop_at_call: ast.Call | None = None) -> bool:
    """Check expression fields on ``stmt`` without searching nested bodies.

    ``stop_at_call`` narrows the match from "any call to ``dep_name``" to one
    specific call node. A dep reached with two different ``pl.constexpr``
    bindings is compiled once per binding, and each of those compilations must
    read the metadata live at *its own* call site — stopping at the first call
    to the name would hand every later binding the first site's tensors.
    """
    if stop_at_call is None and dep_name is None:
        return False

    def matches(node: ast.AST) -> bool:
        if stop_at_call is not None:
            return node is stop_at_call
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == dep_name

    # Nested bodies are skipped here and walked by the caller one statement at a
    # time, so the metadata a scope defines before the call is still collected.
    nested_stmt_fields = {"body", "orelse", "finalbody", "handlers"}
    for field_name, value in ast.iter_fields(stmt):
        if field_name in nested_stmt_fields:
            continue
        items = cast("list[Any]", value) if isinstance(value, list) else [value]
        for item in items:
            if not isinstance(item, ast.AST):
                continue
            if any(matches(node) for node in ast.walk(item)):
                return True
    return False


def _update_local_tensor_meta(
    stmt: ast.stmt,
    local: dict[str, TensorMeta],
    dim_values: dict[str, _DimValue],
    deps: _DepScan,
    resolve_int: Callable[[ast.expr], int | None],
    pl_attr_handlers: dict[str, Callable[[ast.Call, str | None], TensorMeta | None]],
    qualified_call_handlers: _QualifiedCallHandlers,
    resolve_dim: Callable[[ast.expr], _DimValue],
) -> None:
    """Apply one assignment's metadata effects to the source-ordered state."""
    parts = _assignment_parts(stmt)
    if parts is None:
        _forget_rebound_dims(stmt, dim_values)
        return
    targets, value = parts
    named_target = next((t.id for t in targets if isinstance(t, ast.Name)), None)
    has_named_target = named_target is not None
    meta: TensorMeta | None = None
    preserve_existing = False
    dep_returns: list[_DepReturn] = [_DEP_RETURN_DECLINED for _ in targets]

    # Python evaluates the RHS once before assigning any target. Infer all RHS
    # effects from the same pre-assignment state so a self-referential chained
    # assignment cannot affect the metadata applied to later targets.
    if isinstance(value, ast.Subscript) and has_named_target:
        meta = _subscript_slice_meta(value, local, resolve_int)
    elif isinstance(value, ast.Name) and has_named_target:
        meta = local.get(value.id)
    elif isinstance(value, ast.Call):
        fn = value.func
        if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and has_named_target:
            handler = pl_attr_handlers.get(fn.attr)
            if handler is not None:
                # The target name seeds any dynamic symbol the handler has to
                # synthesize for a runtime-sized extent.
                meta = handler(value, named_target)
            else:
                # Keep the pre-existing behavior for pl operations whose
                # result metadata this extractor does not model (for example,
                # same-shaped pl.assemble rebindings).
                preserve_existing = True
        elif isinstance(fn, ast.Attribute) and has_named_target:
            # A qualified (3+-segment) spelling, e.g. pld.tensor.all_to_all_v(...)
            # or pl.tensor.slice(...) — invisible to the one-level branch above
            # since fn.value is itself an Attribute, not a Name (#2638).
            meta, preserve_existing = _qualified_call_meta(fn, value, named_target, qualified_call_handlers)
        elif isinstance(fn, ast.Name) and fn.id in deps.io:
            # The in-place ``Out``-param convention first; a callee that
            # allocates its own results falls through to its return statement.
            if len(targets) > 1 and all(isinstance(target, ast.Name) for target in targets):
                names = [target.id for target in targets if isinstance(target, ast.Name)]
                result = _dep_out_metas_or_return(value, fn.id, targets[0], deps, local)
                shared_meta = result.metas.get(names[0])
                shared_stale = names[0] in result.stale
                # A chain evaluates one call. Relabel its single result rather
                # than inventing fresh derived dimensions for every alias.
                dep_returns = [
                    _DepReturn(
                        {} if shared_meta is None else {name: shared_meta},
                        frozenset({name}) if shared_stale else frozenset(),
                    )
                    for name in names
                ]
            else:
                dep_returns = [
                    _dep_out_metas_or_return(value, fn.id, target, deps, local) for target in targets
                ]
            # Preserve the existing dependency-result behavior when neither
            # rule resolves the callee's results: an already-known target keeps
            # its metadata until a later supported rebinding can refine it.
            # This is how bare inline helpers propagate same-shaped results
            # today. A target the return descent proved stale is exempt — see
            # ``_dep_return_metas``.
            preserve_existing = True

    _update_dim_values(targets, value, dim_values, local, resolve_dim)
    for target, dep_return in zip(targets, dep_returns, strict=True):
        _apply_dep_return(local, dep_return)
        named = target if isinstance(target, ast.Name) else None

        if named is None:
            continue
        if meta is not None:
            local[named.id] = meta
        elif value is not None and not preserve_existing:
            # Any unsupported rebinding shadows an older parameter or local tensor
            # rather than leaving stale metadata visible.
            local.pop(named.id, None)


def _update_dim_values(
    targets: list[ast.expr],
    value: ast.expr | None,
    dim_values: dict[str, _DimValue],
    local: Mapping[str, TensorMeta],
    resolve_dim: Callable[[ast.expr], _DimValue],
) -> None:
    """Snapshot dimension/integer arithmetic before rebinding any target.

    ``d = pl.tensor.dim(src, k)`` and ``d = src.shape[k]`` bind ``d`` to dim
    ``k`` of ``src``; ``M, N = src.shape`` binds each name to its position. The
    dim is read from ``local`` as it stands before the assignment, so a later
    rebinding of ``src`` cannot change what ``d`` already holds. Arithmetic
    aliases retain an abstract derived value, not a lazily evaluated expression.
    Unknown assignments leave a tombstone, preventing fallback to an older
    DynVar anchor. A starred unpack shifts positions, so it binds none.
    """
    if value is None:
        return
    dim = resolve_dim(value)
    unpack_src = _shape_attr_source(value)
    unpack_meta = None if unpack_src is None else local.get(unpack_src)
    for target in targets:
        if isinstance(target, ast.Name):
            dim_values[target.id] = dim
            continue
        if not isinstance(target, (ast.Tuple, ast.List)):
            continue
        starred = any(isinstance(elt, ast.Starred) for elt in target.elts)
        for axis, elt in enumerate(target.elts):
            if (
                unpack_meta is not None
                and not starred
                and isinstance(elt, ast.Name)
                and axis < len(unpack_meta.shape)
            ):
                dim_values[elt.id] = unpack_meta.shape[axis]
                continue
            _forget_names(elt, dim_values)


def _forget_rebound_dims(stmt: ast.stmt, dim_values: dict[str, _DimValue]) -> None:
    """A ``for`` or ``with ... as`` target rebinds its names, so the dims they held are gone."""
    if isinstance(stmt, ast.For):
        _forget_names(stmt.target, dim_values)
    elif isinstance(stmt, ast.With):
        for item in stmt.items:
            if item.optional_vars is not None:
                _forget_names(item.optional_vars, dim_values)
    elif isinstance(stmt, ast.Delete):
        for target in stmt.targets:
            _forget_names(target, dim_values)
    elif isinstance(stmt, ast.AugAssign):
        # The parser rejects augmented assignments. Do not infer their result
        # or leave a previous dimension binding visible to later reshapes.
        _forget_names(stmt.target, dim_values)


def _forget_names(target: ast.expr, dim_values: dict[str, _DimValue]) -> None:
    for node in ast.walk(target):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            dim_values[node.id] = None


def _stmt_binding_names(stmt: ast.stmt) -> set[str]:
    """Names whose bindings one statement can update, excluding nested bodies."""
    parts = _assignment_parts(stmt)
    if parts is not None:
        targets, value = parts
        if value is None:
            return set()
    elif isinstance(stmt, (ast.For, ast.AugAssign)):
        targets = [stmt.target]
    elif isinstance(stmt, ast.Delete):
        targets = stmt.targets
    elif isinstance(stmt, ast.With):
        targets = [item.optional_vars for item in stmt.items if item.optional_vars is not None]
    else:
        return set()
    return {
        node.id
        for target in targets
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
    }


def _body_binding_names(stmts: list[ast.stmt]) -> set[str]:
    """Find possible scalar writes before analyzing a loop iteration."""
    return {
        node.id
        for stmt in stmts
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
    }


def _walk_local_tensor_meta_stmts(
    stmts: list[ast.stmt],
    stop_at_dep: str | None,
    local: dict[str, TensorMeta],
    dim_values: dict[str, _DimValue],
    collected: dict[str, TensorMeta],
    update: Callable[[ast.stmt], None],
    stop_at_call: ast.Call | None = None,
) -> bool:
    """Walk source order with plain binding tables and branch undo journals.

    Each arm rewinds only names it wrote, never copies the whole environment.
    Scalar writes become unknown at joins; tensor metadata survives only when
    both paths agree, including synthesized-dimension identity. Nested joins
    revisit their changed names; unrelated bindings remain untouched. For N AST nodes
    and nesting depth D, joins and loop-write scans take O(N * D) worst-case
    work; branch entry never copies the complete binding tables.
    """
    dim_undo: list[tuple[str, _DimValue]] = []
    tensor_undo: list[tuple[str, TensorMeta | None]] = []

    def forget_dims(names: set[str]) -> None:
        for name in names:
            dim_undo.append((name, dim_values.get(name)))
            dim_values[name] = None

    def rewind(dim_mark: int, tensor_mark: int) -> tuple[dict[str, _DimValue], dict[str, TensorMeta | None]]:
        dims = {name: dim_values.get(name) for name, _ in dim_undo[dim_mark:]}
        tensors = {name: local.get(name) for name, _ in tensor_undo[tensor_mark:]}
        for name, value in reversed(dim_undo[dim_mark:]):
            dim_values[name] = value
        for name, meta in reversed(tensor_undo[tensor_mark:]):
            if meta is None:
                local.pop(name, None)
            else:
                local[name] = meta
        del dim_undo[dim_mark:]
        del tensor_undo[tensor_mark:]
        return dims, tensors

    def walk_if(stmt: ast.If) -> bool:
        dim_mark, tensor_mark = len(dim_undo), len(tensor_undo)
        if walk(stmt.body):
            return True
        true_dims, true_tensors = rewind(dim_mark, tensor_mark)
        if walk(stmt.orelse):
            return True
        false_dims, false_tensors = rewind(dim_mark, tensor_mark)
        forget_dims(true_dims.keys() | false_dims.keys())
        for name in true_tensors.keys() | false_tensors.keys():
            lhs = true_tensors.get(name, local.get(name))
            rhs = false_tensors.get(name, local.get(name))
            tensor_undo.append((name, local.get(name)))
            merged = _merge_tensor_metas(lhs, rhs)
            if merged is not None:
                local[name] = merged
            else:
                local.pop(name, None)
        return False

    def walk(body: list[ast.stmt]) -> bool:
        for stmt in body:
            if _stmt_calls_dep(stmt, stop_at_dep, stop_at_call):
                return True
            names = _stmt_binding_names(stmt)
            for name in names:
                dim_undo.append((name, dim_values.get(name)))
                tensor_undo.append((name, local.get(name)))
            update(stmt)
            for name in names:
                if name in local:
                    collected[name] = local[name]
                else:
                    collected.pop(name, None)

            if isinstance(stmt, ast.If):
                if walk_if(stmt):
                    return True
                continue

            if isinstance(stmt, (ast.For, ast.While)):
                written = _body_binding_names(stmt.body)
                if isinstance(stmt, ast.For):
                    written.update(_stmt_binding_names(stmt))
                # A value from outside a loop cannot describe every iteration
                # when any path writes it. Definite body assignments can still
                # establish useful dimensions later in this iteration.
                forget_dims(written)
                if walk(stmt.body):
                    return True
                forget_dims(written)
                if walk(stmt.orelse):
                    return True
                forget_dims(_body_binding_names(stmt.orelse))
                continue

            for attr in ("body", "orelse", "finalbody"):
                nested = getattr(stmt, attr, None)
                if isinstance(nested, list) and walk(cast("list[ast.stmt]", nested)):
                    return True
        return False

    return walk(stmts)


@dataclass(frozen=True)
class _StaticScope:
    """Resolves an operand to the compile-time value it names in a function.

    Resolution mirrors the specializer's folding: the same namespace, the same
    shadowing, the same ``pl.constexpr`` binding. It may accept a spelling the
    generated program then rejects (``Cfg.DT`` on a module-level class); the
    parser reports that one by name, which is still clearer than a dropped meta.

    Attributes:
        namespace: The function's globals plus closure cells
        shadowed: Its parameters and assignment targets, which hide a same-named
            global here exactly as they do in the specializer's free-name folding
        constexpr: Its ``pl.constexpr`` binding, each value parsed back to an
            expression
    """

    namespace: Mapping[str, Any]
    shadowed: frozenset[str]
    constexpr: Mapping[str, ast.expr]

    @classmethod
    def of(cls, func: Any, constexpr_values: Mapping[str, str] | None) -> "_StaticScope":
        """The scope of ``func``, under its ``pl.constexpr`` binding (name → folded source text)."""
        return cls(
            namespace=func_name_lookup(func),
            shadowed=frozenset(_ast_analysis.body_local_names(func)),
            constexpr={
                name: ast.parse(text, mode="eval").body for name, text in (constexpr_values or {}).items()
            },
        )

    def fold(self, node: ast.expr | None) -> ast.expr | None:
        """The folded value's expression for a ``pl.constexpr`` parameter, else ``node``."""
        if isinstance(node, ast.Name) and node.id in self.constexpr:
            return self.constexpr[node.id]
        return node

    def value(self, node: ast.expr) -> Any:
        """The value a constant name or attribute chain is bound to, else None.

        Covers a module-level or closure constant and an attribute chain rooted
        at a module or class (``pl.FP32``, ``pl.TensorLayout.NZ``). Attributes
        are read only off modules and classes, so resolving an operand never
        runs an instance property.
        """
        if isinstance(node, ast.Name):
            return None if node.id in self.shadowed else self.namespace.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self.value(node.value)
            if isinstance(base, (types.ModuleType, type)):
                return getattr(base, node.attr, None)
        return None

    def int_shape(self, node: ast.expr) -> tuple[int, ...] | None:
        """A shape constant: a name bound to a list or tuple of ints."""
        value = self.value(node) if isinstance(node, ast.Name) else None
        if not isinstance(value, (list, tuple)):
            return None
        seq = cast("Sequence[Any]", value)
        return tuple(seq) if all(isinstance(d, int) and not isinstance(d, bool) for d in seq) else None


def _resolve_dim_axis(node: ast.expr, scope: _StaticScope) -> int | None:
    """Fold a dimension axis from static values, never from runtime scalar aliases."""
    folded = scope.fold(node)
    if isinstance(folded, ast.Constant):
        return folded.value if isinstance(folded.value, int) and not isinstance(folded.value, bool) else None
    if isinstance(folded, (ast.Name, ast.Attribute)):
        value = scope.value(folded)
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    if isinstance(folded, ast.UnaryOp):
        value = _resolve_dim_axis(folded.operand, scope)
        if value is not None:
            if isinstance(folded.op, ast.USub):
                return -value
            if isinstance(folded.op, ast.UAdd):
                return value
    if isinstance(folded, ast.BinOp):
        lhs = _resolve_dim_axis(folded.left, scope)
        rhs = _resolve_dim_axis(folded.right, scope)
        if lhs is not None and rhs is not None:
            return _fold_int_arith(folded.op, lhs, rhs)
    return None


def _preview_static_unary_plus(
    func_def: ast.FunctionDef,
    seed_meta: Mapping[str, TensorMeta],
    scope: _StaticScope,
    constexpr_values: Mapping[str, str] | None,
) -> dict[tuple[int, int], int]:
    """Preview the real body rewrite once, recording static unary-plus source points.

    Runtime names left by alpha-renaming must never resolve through globals.
    Memoizing the pure constant fold keeps nested unary expressions linear.
    """
    if not any(isinstance(node, ast.UAdd) for node in ast.walk(func_def)):
        return {}
    values: dict[tuple[int, int], int] = {}

    @functools.cache
    def static_int(node: ast.expr) -> int | None:
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, int) and not isinstance(node.value, bool) else None
        if isinstance(node, ast.UnaryOp):
            value = static_int(node.operand)
            if value is not None:
                if isinstance(node.op, ast.USub):
                    return -value
                if isinstance(node.op, ast.UAdd):
                    return value
        if isinstance(node, ast.BinOp):
            lhs, rhs = static_int(node.left), static_int(node.right)
            if lhs is not None and rhs is not None:
                return _fold_int_arith(node.op, lhs, rhs)
        return None

    class Preview(_BodyTransformer):
        def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.UnaryOp:
            position = (node.lineno, node.col_offset)
            transformed = cast("ast.UnaryOp", self.generic_visit(node))
            if isinstance(transformed.op, ast.UAdd):
                value = static_int(transformed.operand)
                if value is not None:
                    values[position] = value
            return transformed

    transformer = Preview(
        tensor_meta=dict(seed_meta),
        dep_names=set(),
        param_names=[
            arg.arg
            for arg in func_def.args.args
            if arg.arg != "self" and arg.arg not in (constexpr_values or {})
        ],
        initial_used_names={
            node.id
            for node in ast.walk(func_def)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        },
        py_globals=scope.namespace,
        constexpr_values=constexpr_values,
    )
    try:
        for stmt in copy.deepcopy(func_def.body):
            transformer.visit(stmt)
    except (IndexError, KeyError, TypeError, ValueError):
        # Invalid source still belongs to the normal specializer/parser path.
        return {}
    return values


def _matching_static_unary_plus(value: int | None, operand: _DimValue) -> int | None:
    """Accept the preview only when the current binding independently agrees."""
    return value if value is not None and isinstance(operand, int) and operand == value else None


def _drop_dims_is_noop(drop_dims_node: ast.expr | None, scope: _StaticScope) -> bool:
    """True when ``drop_dims`` is a no-op: omitted, ``None``, an empty
    list/tuple literal, or a ``Name`` bound to one of those. ``scope.fold()``
    resolves a ``pl.constexpr`` parameter binding first (its own, narrower
    mechanism — a name outside ``constexpr`` passes through unchanged); what
    remains is then resolved as a module-level or closure constant
    (e.g. ``EMPTY = []`` or ``NONE_DIMS = None``) if it's still a ``Name`` —
    read directly off ``scope.namespace``/``scope.shadowed`` rather than
    through ``scope.value()``, whose bare ``None`` return is ambiguous
    between "genuinely bound to None" and "a local parameter, unresolvable
    here" (e.g. ``def body(src, drop_dims): ... drop_dims=drop_dims``) — the
    latter must never be assumed empty, or a real non-empty runtime value
    would advertise the wrong, pre-drop rank.
    """
    node = scope.fold(drop_dims_node)
    if isinstance(node, ast.Name):
        if node.id in scope.shadowed or node.id not in scope.namespace:
            return False
        resolved = scope.namespace[node.id]
        return resolved is None or (isinstance(resolved, (list, tuple)) and not resolved)
    return (
        node is None
        or (isinstance(node, ast.Constant) and node.value is None)
        or (isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) == 0)
    )


def extract_local_tensor_metas(
    func: Any,
    seed_meta: dict[str, TensorMeta] | None = None,
    caller_func_type: str = "orchestration",
    stop_at_dep: str | None = None,
    dep_seen: frozenset[int] = frozenset(),
    stop_at_call: ast.Call | None = None,
    constexpr_values: Mapping[str, str] | None = None,
    visible_only: bool = False,
    *,
    discover_deps: _call_binding.DiscoverDeps,
) -> dict[str, TensorMeta]:
    """Infer ``TensorMeta`` for the local tensor variables in ``func``'s body.

    Walks the body in source order, tracking the three ways a local tensor can
    be produced inside a JIT function:

    1. ``var = pl.create_tensor([shape], dtype=pl.XXX)`` — each operand by
       position or keyword. Shape from a list or tuple literal (literal ints,
       ``Name`` refs to int globals, ``M`` from ``M, N = src.shape`` /
       ``src.shape[k]``, and simple int arithmetic over those), or whole from
       ``src.shape`` or a module-level shape constant. Dtype from ``pl.XXX``, a
       module-level or closure constant bound to one, or ``src.dtype``; a
       non-ND ``layout`` the same way.
       A shape element that resolves through a dynamic alias — either
       ``tokens = pl.tensor.dim(P, k)`` for a seeded param ``P`` whose dim
       ``k`` is ``DynDim``-bound, or a direct reference to a DynVar
       declared in the seed metas — stamps the matching ``DynDim`` onto the
       local's shape so the dynamic chain keeps flowing through subsequent
       deps.
    2. ``var = pl.slice(src, [shape], [...])`` — dtype inherited from ``src`` (a
       parameter or earlier local); each shape dim that is a static int is used
       as-is, and a non-static dim (e.g. a runtime ``valid_len``) falls back to
       ``src``'s corresponding dim, since a slice is bounded above by its
       parent. ``src`` dims that are themselves ``DynDim`` flow through
       transparently. The subscript-slice sugar ``var = src[a:b, i, ...]`` (an
       ``ast.Subscript``) is the documented equivalent and is tracked the same
       way: each slice dim resolves to ``stop - start`` (parent-dim fallback
       when not static), a scalar index drops its dim, and trailing implicit
       ``:`` dims keep the parent extent.
    3. ``v1, ..., vk = jit_dep(args)`` where ``jit_dep`` is an
       ``@pl.jit.incore`` / ``inline`` / ``opaque`` callee with ``k``
       ``pl.Out[...]`` parameters — each ``vi`` inherits the meta of the caller
       argument bound to the i-th ``Out`` parameter (the in-place-output
       convention every such kernel follows, and the same heuristic
       ``_infer_return_type`` uses on the callee side).
    4. ``v1, ..., vk = jit_dep(args)`` where ``jit_dep`` instead allocates its
       own results (``a = pl.create_tensor(...); ...; return a, b``) — the
       extractor descends into the callee's body, seeded with the params this
       call site binds, and each ``vi`` takes the meta of the i-th returned
       name (see :func:`_dep_return_metas`). ``dep_seen`` carries the ``id()``
       of every function already on that stack so the descent cannot loop.

    ``seed_meta`` pre-populates the table with the caller's parameter metas
    (including any ``DynDim`` entries those carry) so a ``pl.slice`` of a
    parameter, a dep call passing a parameter through, or a local
    ``pl.create_tensor`` sized off a dynamic dim of a parameter all resolve.

    A *scalar parameter* is a runtime value and therefore never resolves a
    shape dim statically; a dim sized off one becomes a synthesized ``DynDim``
    like any other runtime extent. Size a local off a module-level or closure
    constant, a ``pl.constexpr`` parameter, or a static dim of another tensor
    when the shape has to be static.

    A ``pl.create_tensor`` / ``pld.window`` dim that no static rule resolves —
    a runtime extent such as ``pld.world_size()``, ``pl.tensor.read(cfg, [0])``,
    or arithmetic over a ``DynDim`` — becomes a synthesized ``DynDim`` (see
    :func:`synthesized_dyn_dim`) rather than voiding the whole meta. Reshape
    dimensions composed of known tensor dimensions and integer arithmetic use
    the same descriptor-bound symbols, preserving their expressions in IR.
    Other unresolved reshape dimensions are still skipped, and the clear
    ``ValueError`` in ``Specializer._build_params`` fires for that variable.
    When ``stop_at_dep`` is provided, extraction stops
    immediately before the first source-ordered call to that dependency. This
    produces the point-in-time metadata visible to that call and ignores later
    rebindings. ``stop_at_call`` selects one specific call node instead, which
    is what a dep reached with several ``pl.constexpr`` bindings needs: each
    binding is a separate compilation and must see the tensors live at its own
    call site, not at the first call to the same name.

    ``constexpr_values`` is ``func``'s own ``pl.constexpr`` binding (parameter
    name → folded source text). A shape or dtype operand naming one resolves to
    that value, as the specializer folds it, instead of to a runtime extent.

    The default result collects observed metadata, including branch locals.
    Dependency consumers request ``visible_only`` so only bindings valid at
    their selected call or at the end of a callee can type a dependency.
    """
    func_def = _ast_analysis.get_func_def(func)
    scope = _StaticScope.of(func, constexpr_values)
    dim_values: dict[str, _DimValue] = {name: None for name in scope.shadowed}
    local = dict(seed_meta or {})
    collected = dict(local)
    dynvar_anchors = _build_dynvar_anchor_index(seed_meta or {})
    declared_dims = _declared_dim_values(func_def, seed_meta or {})
    static_unary_plus = _preview_static_unary_plus(func_def, seed_meta or {}, scope, constexpr_values)

    def _resolve_dim_value(elt: ast.expr) -> _DimValue:
        """Resolve a scalar to a snapshot of known dimension/integer arithmetic.

        Dynamic resolution paths (added on top of the original static integer
        resolver):

        - ``Name`` bound to dim ``k`` of ``P`` (``pl.tensor.dim(P, k)``,
          ``P.shape[k]``, or ``M, N = P.shape``) → that dim as it was when the
          name was bound, a ``DynDim`` flowing through as-is.
        - ``Name`` that's a DynVar declared on a seeded param → returns the
          DynDim of the (first) anchor site.
        - ``P.shape[k]`` / ``pl.tensor.dim(P, k)`` written inline → dim ``k``
          of ``P``, the same as through a named alias.

        Falls back to integer resolution for literal ints, int globals,
        ``pl.constexpr`` parameters, and arithmetic over those.
        A derived value never replaces an unknown operand or uses a sample
        bound. It acquires a descriptor-bound symbol only when used in a shape.
        """
        if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
            return elt.value
        if isinstance(elt, (ast.Subscript, ast.Call)):
            return declared_dims.get(elt) or _alias_dim(
                _extract_dim_alias(elt, lambda axis: _resolve_dim_axis(axis, scope)), local
            )
        if isinstance(elt, ast.Name):
            if elt.id in scope.constexpr:
                return _resolve_dim_value(scope.constexpr[elt.id])
            # A name bound to a dim: pl.tensor.dim(P, k) / P.shape[k] / k-th of P.shape
            if elt.id in dim_values:
                return dim_values[elt.id]
            # Direct DynVar reference (e.g. M used as a shape entry).
            anchors = dynvar_anchors.get(elt.id)
            dim = _alias_dim(anchors[0] if anchors else None, local)
            if isinstance(dim, DynDim):
                return dim
            # Static int via globals (module-level or closure constants).
            value = scope.value(elt)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            return None
        if isinstance(elt, ast.BinOp):
            return _resolve_dim_binop(elt, _resolve_dim_value)
        if isinstance(elt, ast.UnaryOp):
            if isinstance(elt.op, ast.UAdd):
                return _matching_static_unary_plus(
                    static_unary_plus.get((elt.lineno, elt.col_offset)), _resolve_dim_value(elt.operand)
                )
            v = _resolve_dim_value(elt.operand)
            if isinstance(v, (DynDim, tuple)) and isinstance(elt.op, ast.USub):
                return type(elt.op), (v,)
            if not isinstance(v, int):
                return None
            if isinstance(elt.op, ast.USub):
                return -v
            return None
        return None

    def _resolve_shape_elt(
        elt: ast.expr, bound_dim: DynDim | None = None, bound_dims: _BoundShapeDims | None = None
    ) -> ShapeDim | None:
        return _bind_scalar_shape_dim(_resolve_dim_value(elt), bound_dim, bound_dims)

    def _resolve_int(elt: ast.expr) -> int | None:
        """Integer-only wrapper retained for _slice_meta's existing call site."""
        v = _resolve_shape_elt(elt)
        return v if isinstance(v, int) else None

    def _resolve_whole_shape(node: ast.expr | None) -> tuple[ShapeDim, ...] | None:
        """Resolve a shape operand that names a whole shape instead of listing it.

        Covers ``src.shape`` of a tracked tensor and a module-level or closure
        constant holding a list or tuple of ints — both of which the specializer
        folds into a literal shape in the generated source.
        """
        if node is None:
            return None
        src = _shape_attr_source(node)
        if src is not None:
            return local[src].shape if src in local else None
        return scope.int_shape(node)

    def _resolve_shape(
        node: ast.expr | None, dyn_base: str | None = None, *, dimension_arithmetic: bool = False
    ) -> tuple[ShapeDim, ...] | None:
        """Resolve a shape operand, optionally synthesizing runtime-only dims.

        A list or tuple literal resolves element by element; ``src.shape`` or a
        shape constant resolves whole (see :func:`_resolve_whole_shape`).

        ``dyn_base`` is the name of the local being assigned. When given, a dim
        that no static rule resolves becomes a synthesized ``DynDim`` instead of
        failing the whole shape (see :func:`synthesized_dyn_dim`). Callers that
        pass ``None`` keep the strict all-or-nothing behaviour.
        ``dimension_arithmetic`` limits synthesis to arithmetic whose operands
        are known dimensions or integers; arbitrary runtime values are rejected.
        """
        node = scope.fold(node)
        whole = _resolve_whole_shape(node)
        if whole is not None:
            return whole
        if not isinstance(node, (ast.List, ast.Tuple)):
            return None
        dims: list[ShapeDim] = []
        bound_dims: _BoundShapeDims = {}
        for i, elt in enumerate(node.elts):
            bound_dim = (
                synthesized_dyn_dim(func_def.name, dyn_base, i)
                if dyn_base is not None and dimension_arithmetic
                else None
            )
            v = _resolve_shape_elt(elt, bound_dim, bound_dims)
            if v is None:
                if dyn_base is None or dimension_arithmetic:
                    return None
                v = synthesized_dyn_dim(func_def.name, dyn_base, i)
            dims.append(v)
        return tuple(dims)

    def _resolve_dtype(node: ast.expr | None) -> DataType | None:
        """Resolve a ``dtype`` operand to the ``DataType`` it evaluates to.

        Resolves by value, not by spelling, so the inferred meta agrees with
        what the specializer folds into the generated source: ``WDT`` bound to
        ``pl.FP32`` at module or closure scope is ``FP32`` exactly as the
        literal ``pl.FP32`` is, a ``pl.constexpr`` parameter is its folded
        value, and ``src.dtype`` is the dtype of the tensor ``src`` holds at
        this point of the walk.
        """
        node = scope.fold(node)
        if node is None:
            return None
        if isinstance(node, ast.Attribute) and node.attr == "dtype" and isinstance(node.value, ast.Name):
            src_meta = local.get(node.value.id)
            if src_meta is not None:
                return src_meta.dtype
        value = scope.value(node)
        return value if isinstance(value, DataType) else None

    def _create_tensor_meta(call: ast.Call, target: str | None = None) -> TensorMeta | None:
        # pl.create_tensor(shape, dtype, layout, ...) — each by position or keyword.
        # A runtime-sized extent (``pl.create_tensor([n, 128], ...)`` where ``n``
        # is read from a tensor) synthesizes a dynamic dim rather than dropping
        # the meta -- the allocation itself is runtime-sized either way.
        shape = _resolve_shape(_call_operand(call, 0, "shape"), target)
        dtype_val = _resolve_dtype(_call_operand(call, 1, "dtype"))
        # An omitted layout is ND. One that does not resolve declines the meta:
        # claiming ND for it would silently mis-declare every dep it reaches.
        layout_node = scope.fold(_call_operand(call, 2, "layout"))
        layout = _ir.TensorLayout.ND if layout_node is None else scope.value(layout_node)
        if shape is None or dtype_val is None or not isinstance(layout, _ir.TensorLayout):
            return None
        # ND is recorded as no layout — what an omitted annotation slot means.
        return TensorMeta(
            shape=shape, dtype=dtype_val, layout=None if layout == _ir.TensorLayout.ND else layout
        )

    def _window_meta(call: ast.Call, target: str | None = None) -> TensorMeta | None:
        # pld.window(buffer, shape, *, dtype) — a distributed window view over a
        # window buffer. Lets a host orchestrator's per-rank window locals
        # propagate their meta into the ``pld.DistributedTensor`` parameters of
        # the chip orchestrator it calls. A runtime-sized dim
        # (``[pld.world_size(), 1]``) synthesizes a dynamic dim, the same way
        # create_tensor does.
        shape = _resolve_shape(_call_operand(call, 1, "shape"), target)
        dtype_val = _resolve_dtype(_call_operand(call, None, "dtype"))
        if shape is None or dtype_val is None:
            return None
        return TensorMeta(shape=shape, dtype=dtype_val)

    def _reshape_meta(call: ast.Call, target: str | None = None) -> TensorMeta | None:
        # pl.reshape(input, shape) / pl.tensor.reshape(tensor, shape) — dtype
        # inherited from source tensor.
        src = _tensor_or_input_operand(call)
        if not isinstance(src, ast.Name) or src.id not in local:
            return None
        src_meta = local[src.id]
        shape_node = _call_operand(call, 1, "shape")
        if shape_node is None:
            return None
        # Type descriptor-derived extents without evaluating dynamic arithmetic.
        # The original reshape expression stays in IR and its validation is
        # unchanged; dependency parameters bind these symbols from the result's
        # real descriptor. Unknown scalars must not acquire a placeholder here.
        # Rebinding the same Python name must not equate unrelated result
        # descriptors. Source coordinates also distinguish same-line calls and
        # stay deterministic across extractions and sample bounds.
        result_name = f"{target}_l{call.lineno}_c{call.col_offset}" if target is not None else None
        shape = _resolve_shape(shape_node, result_name, dimension_arithmetic=True)
        if shape is None:
            return None
        # Other non-ND reshapes reinterpret layout coordinates, so claiming ND
        # would mis-declare the result. The whole static NZ buffer is the one
        # exception: its rank-1 physical view needs no coordinate rewrite.
        # Match BlockNzTensorViews::IsWholeTensorFlatten; symbolic bounds do
        # not prove the runtime element count. The reshape result is ND.
        if src_meta.layout not in (None, _ir.TensorLayout.ND):
            whole_nz_flatten = (
                src_meta.layout == _ir.TensorLayout.NZ
                and len(shape) == 1
                and type(shape[0]) is int
                and all(type(dim) is int for dim in src_meta.shape)
                and math.prod(cast(tuple[int, ...], src_meta.shape)) == shape[0]
            )
            if not whole_nz_flatten:
                return None
        # Unlike an omitted parameter layout, a reshape result is known ND.
        # Keep it explicit so a dependency's annotation cannot overwrite it.
        return TensorMeta(shape=shape, dtype=src_meta.dtype, layout=_ir.TensorLayout.ND)

    def _slice_meta(call: ast.Call, target: str | None = None) -> TensorMeta | None:
        # pl.slice(input, shape, offset, ...) / pl.tensor.slice(tensor, shape,
        # offset, ...) — shape/offset/drop_dims/etc. share the same names in
        # both real spellings; only the first param's name differs.
        #
        # drop_dims rank-reduces the result; this handler doesn't model that,
        # so decline rather than advertise the pre-drop shape at the wrong
        # rank — UNLESS drop_dims is a no-op (see _drop_dims_is_noop).
        # Pre-existing gap for the 2-segment spelling too, not introduced by
        # qualified dispatch — just newly reachable through it.
        if not _drop_dims_is_noop(_call_operand(call, 4, "drop_dims"), scope):
            return None
        src = _tensor_or_input_operand(call)
        if not isinstance(src, ast.Name) or src.id not in local:
            return None
        src_meta = local[src.id]
        shape_node = scope.fold(_call_operand(call, 1, "shape"))
        whole = _resolve_whole_shape(shape_node)
        if whole is not None:
            if len(whole) != len(src_meta.shape):
                return None
            return TensorMeta(shape=whole, dtype=src_meta.dtype, layout=src_meta.layout)
        if not isinstance(shape_node, (ast.List, ast.Tuple)) or len(shape_node.elts) != len(src_meta.shape):
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
        return TensorMeta(shape=tuple(dims), dtype=src_meta.dtype, layout=src_meta.layout)

    deps = _DepScan(
        io=scan_dep_io(func, caller_func_type, discover_deps=discover_deps),
        funcs={b.call_name: b.dep for b in discover_deps(func, caller_func_type)},
        seen=dep_seen | {id(func)},
        constexpr=dict(constexpr_values or {}),
        scope=scope,
        owner=func_def.name,
        discover_deps=discover_deps,
    )

    # Dispatch table: pl.<attr>(...) → meta extraction function.
    # Replaces sequential if-chains, reducing branch and statement counts.
    _pl_attr_handlers = {
        "create_tensor": _create_tensor_meta,
        "slice": _slice_meta,
        "window": _window_meta,
        "reshape": _reshape_meta,
    }

    # 3-segment spellings this walker also understands, keyed by the FULL
    # dotted path so a match can never collide across namespaces the way
    # attr-name-only dispatch could. The pl.tensor.* / pld.tensor.window
    # entries are DERIVED from _pl_attr_handlers above rather than
    # hand-duplicated, so a new entry added there is automatically covered
    # under its 3-segment spelling too — the two tables can't drift apart
    # for that half. The managed collectives have no 2-segment sugar to
    # derive from (3-segment-only), so they're listed explicitly, each with
    # _same_operand_meta reading the real operand its own docstring names as
    # the rebind target — not a bare `preserve_existing`, which would trust
    # the LHS name instead of the operand when the two differ (review
    # finding: `old = pld.tensor.broadcast(window, ...)` must get `window`'s
    # metadata, not `old`'s stale one). None stays available in the type for
    # a future op with no derivable operand at all; nothing currently uses
    # it. Anything NOT listed here safely falls through to local.pop(),
    # exactly like today's behavior for any other unrecognized call: a new
    # op starts safe by default and must be added here deliberately, never
    # silently assumed same-shape (#2638).
    _qualified_call_handlers: _QualifiedCallHandlers = {
        **{("pl", "tensor", attr): handler for attr, handler in _pl_attr_handlers.items()},
        ("pld", "tensor", "window"): _pl_attr_handlers["window"],
        ("pld", "tensor", "all_to_all_v"): _same_operand_meta(local, 1, "target"),
        ("pld", "tensor", "all_to_all"): _same_operand_meta(local, 1, "target"),
        ("pld", "tensor", "allreduce"): _same_operand_meta(local, 0, "target"),
        ("pld", "tensor", "reduce_scatter"): _same_operand_meta(local, 0, "target"),
        ("pld", "tensor", "broadcast"): _same_operand_meta(local, 0, "target"),
        ("pld", "tensor", "allgather"): _same_operand_meta(local, 1, "target"),
        ("pld", "tensor", "barrier"): _same_operand_meta(local, 0, "signal"),
    }

    _walk_local_tensor_meta_stmts(
        func_def.body,
        stop_at_dep,
        local,
        dim_values,
        collected,
        functools.partial(
            _update_local_tensor_meta,
            local=local,
            dim_values=dim_values,
            deps=deps,
            resolve_int=_resolve_int,
            pl_attr_handlers=_pl_attr_handlers,
            qualified_call_handlers=_qualified_call_handlers,
            resolve_dim=_resolve_dim_value,
        ),
        stop_at_call,
    )
    return local if visible_only or stop_at_dep is not None or stop_at_call is not None else collected


def resolve_dep_call_metadata(
    dep: "JITFunction",
    caller_func: Any,
    caller_tensor_meta: dict[str, TensorMeta],
    caller_scalar_dtypes: dict[str, DataType],
    dep_dyn_map: dict[str, dict[int, DynDim]],
    caller_func_type: str = "orchestration",
    dep_call_name: str | None = None,
    call_node: ast.Call | None = None,
    caller_constexpr: Mapping[str, str] | None = None,
    *,
    discover_deps: _call_binding.DiscoverDeps,
) -> tuple[
    dict[str, TensorMeta],
    dict[str, DataType],
]:
    """Map ``dep``'s parameter names to TensorMeta / scalar dtypes using
    ``caller_func``'s call-site arguments.

    The caller may be the entry function or another dep (transitive case);
    in either case we look up the (first) ``dep(...)`` call site in
    ``caller_func``'s body and apply the positional-or-keyword mapping.
    Intermediate tensors produced in the caller — ``pl.create_tensor``,
    ``pl.slice`` views, and the return values of other ``@pl.jit`` deps — are
    folded into the metadata pool (see ``extract_local_tensor_metas``).
    Falls back to name-based matching when call-site extraction fails.

    ``caller_func_type`` is forwarded to ``extract_local_tensor_metas``
    so a host orchestrator's body can also recognise chip-orchestrator deps
    when walking ``v = chip_orch(...)`` return-capture assignments.

    ``dep_call_name`` is the name ``caller_func``'s source calls the dep by;
    it differs from ``dep.__name__`` under an aliased import or any other
    rebinding. Defaults to ``dep.__name__`` for callers that resolve a dep
    reached under its own name.

    ``call_node`` pins resolution to one specific call site instead of the
    first one bearing that name. A dep called with two different
    ``pl.constexpr`` bindings is compiled once per binding, and each of those
    compilations reads the arguments and the point-in-time metadata of the call
    site that produced it.

    ``caller_constexpr`` is the caller's own ``pl.constexpr`` binding, so a
    local the caller sizes off one of its compile-time parameters reaches the
    dep with that static extent.
    """
    dep_param_names = dep._param_names()
    call_name = dep_call_name or dep.__name__
    call_args = (
        _call_binding.call_arg_refs(call_node)
        if call_node is not None
        else _call_binding.extract_call_args_for_dep(caller_func, call_name)
    )
    intermediate_metas = extract_local_tensor_metas(
        caller_func,
        seed_meta=caller_tensor_meta,
        caller_func_type=caller_func_type,
        stop_at_dep=call_name if call_args is not None and call_node is None else None,
        stop_at_call=call_node,
        constexpr_values=caller_constexpr,
        discover_deps=discover_deps,
        visible_only=True,
    )
    # The extractor starts from caller_tensor_meta, then applies source-ordered
    # rebindings. Its result is therefore the authoritative state at the call.
    all_tensor_meta = intermediate_metas

    dep_tensor_meta: dict[str, TensorMeta] = {}
    dep_scalar_dtypes: dict[str, DataType] = {}

    if call_args is not None:
        for dep_param, caller_arg in _call_binding.build_param_mapping(dep_param_names, call_args).items():
            if caller_arg is None:
                continue
            if isinstance(caller_arg, _call_binding.SlicedArg):
                # Per-rank dispatch ``chip_orch(x[r], ...)``: the dep parameter
                # inherits the base tensor's meta with ``drop`` leading dims
                # removed (the subscripted dims selected by integer indices).
                base_meta = all_tensor_meta.get(caller_arg.base)
                if base_meta is not None and caller_arg.drop < len(base_meta.shape):
                    dep_tensor_meta[dep_param] = TensorMeta(
                        shape=base_meta.shape[caller_arg.drop :],
                        dtype=base_meta.dtype,
                        # Layout describes the trailing dims, so dropping
                        # leading ones leaves it intact.
                        layout=base_meta.layout,
                    )
                continue
            if caller_arg in all_tensor_meta:
                dep_tensor_meta[dep_param] = all_tensor_meta[caller_arg]
            elif caller_arg in caller_scalar_dtypes:
                # A scalar arg is a runtime value, so only its declared type
                # crosses the call boundary.
                dep_scalar_dtypes[dep_param] = caller_scalar_dtypes[caller_arg]
    else:
        # Fallback: name-based matching against the caller's metadata pool.
        dep_tensor_meta = {n: all_tensor_meta[n] for n in dep_param_names if n in all_tensor_meta}
        dep_scalar_dtypes = {n: caller_scalar_dtypes[n] for n in dep_param_names if n in caller_scalar_dtypes}

    _validate_dep_packed_fp4_dtypes(dep, dep_tensor_meta)
    _overlay_dep_declared_dyn_dims(dep_dyn_map, dep_tensor_meta)
    _overlay_dep_declared_layouts(dep, dep_tensor_meta)

    return dep_tensor_meta, dep_scalar_dtypes


def _validate_dep_packed_fp4_dtypes(dep: "JITFunction", dep_tensor_meta: dict[str, TensorMeta]) -> None:
    """Require dependency arguments to preserve their declared packed FP4 format."""
    for dep_param, dtype in _arguments.param_dtypes(dep._func).items():
        meta = dep_tensor_meta.get(dep_param)
        if meta is None or meta.dtype == dtype:
            continue
        dtype_name = str(dtype).upper()
        context = f"@pl.jit function {dep.__name__!r}: parameter {dep_param!r} annotated pl.{dtype_name}"
        raise TypeError(
            f"{context} but got tensor metadata dtype {meta.dtype}; "
            "pass the declared FP4 format (cast explicitly to change formats)"
        )


def _overlay_dep_declared_dyn_dims(
    dep_dyn_map: dict[str, dict[int, DynDim]], dep_tensor_meta: dict[str, TensorMeta]
) -> None:
    """Stamp the DynDims a dep declares itself onto its parameter metas, in place.

    ``dep_dyn_map`` is pre-computed by ``compute_per_func_dyndim_maps``. A
    DynDim the caller actually derived takes precedence — its symbol is the one
    bound at the call site — so only plain int dims and synthesized placeholders
    are overwritten. The replacement keeps whatever ``static_bound`` the dim
    already carried (the int extent, or a placeholder's ``1``) so cache keys stay
    coherent.

    Replacing a placeholder matters beyond naming: the dep's body may reference
    its declared symbol (``for i in pl.range(NR)``). Keeping the invented name
    on the parameter would leave that reference unbound in the generated
    program, so the declaration wins whenever we only had a placeholder.

    Args:
        dep_dyn_map: Per-parameter ``dim_idx -> DynDim`` the dep declares
        dep_tensor_meta: Per-parameter meta to update in place
    """
    for dep_param, dim_to_dyn in dep_dyn_map.items():
        meta = dep_tensor_meta.get(dep_param)
        if meta is None:
            continue
        new_shape: list[ShapeDim] = list(meta.shape)
        changed = False
        for i, dyn in dim_to_dyn.items():
            if i >= len(new_shape):
                continue
            existing = new_shape[i]
            if isinstance(existing, DynDim) and not existing.synthesized:
                continue
            static_bound = existing.static_bound if isinstance(existing, DynDim) else existing
            new_shape[i] = DynDim(name=dyn.name, literal=dyn.literal, static_bound=static_bound)
            changed = True
        if changed:
            dep_tensor_meta[dep_param] = TensorMeta(
                shape=tuple(new_shape), dtype=meta.dtype, layout=meta.layout
            )


def _overlay_dep_declared_layouts(dep: "JITFunction", dep_tensor_meta: dict[str, TensorMeta]) -> None:
    """Fill in layouts a dep declares itself, in place.

    An omitted caller layout leaves ``meta.layout`` as None, so without this a
    dep declaring ``pl.Tensor[[...], pl.NZ]`` under a caller that declares none
    compiles as ND, silently. Known caller layouts, including an operation's
    inferred ND result, win on conflict; only None is filled.

    Args:
        dep: The dep whose own annotations to read
        dep_tensor_meta: Per-parameter meta to update in place
    """
    for dep_param, dep_layout in _arguments.param_layouts(dep._func, dep.__name__).items():
        meta = dep_tensor_meta.get(dep_param)
        if meta is None or meta.layout is not None:
            continue
        dep_tensor_meta[dep_param] = TensorMeta(shape=meta.shape, dtype=meta.dtype, layout=dep_layout)
