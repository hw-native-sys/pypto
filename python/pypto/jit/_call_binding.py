# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""JIT call-site argument binding and constexpr specialization variants.

This module maps existing dependency objects without owning their caches or
performing runtime JITFunction recognition. Dependency discovery stays at the
decorator boundary; type-only references do not introduce a runtime import cycle."""

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from . import _arguments, _ast_analysis
from .specializer import (
    DynDim,
    free_name_source,
    func_name_lookup,
)

if TYPE_CHECKING:
    from .decorator import JITFunction


def compute_per_func_dyndim_maps(
    entry_func: Any,
    entry_param_names: list[str],
    deps: list[Any],
    callers_by_dep_id: dict[int, list[tuple[Any, str]]],
    call_args_cache: "dict[tuple[int, str], list[tuple[str | None, str | SlicedArg | None]] | None]",
) -> dict[int, dict[str, dict[int, DynDim]]]:
    """Per JIT function in the dep graph, return ``param → dim_idx → DynDim``.

    Each function's map starts from its own declarations
    (``build_dyndim_map_for_func``) and is augmented leaf-first with
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
            for p, dims in _ast_analysis.build_dyndim_map_for_func(
                entry_func, tuple(entry_param_names)
            ).items()
        }
    }
    for dep in deps:
        out[id(dep._func)] = {
            p: dict(dims)
            for p, dims in _ast_analysis.build_dyndim_map_for_func(
                dep._func, tuple(dep._param_names())
            ).items()
        }

    # Leaf-first cascade: a dep's dynamic dim flows up to every recorded caller's arg.
    for dep in deps:
        dep_map = out[id(dep._func)]
        if not dep_map:
            continue
        for caller_func, call_name in callers_by_dep_id.get(id(dep._func), ()):
            call_args = call_args_cache.get((id(caller_func), call_name))
            if call_args is None:
                continue
            param_mapping = build_param_mapping(dep._param_names(), call_args)
            caller_map = out.get(id(caller_func))
            if caller_map is None:
                continue
            for dep_param, dim_to_dyn in dep_map.items():
                caller_arg = param_mapping.get(dep_param)
                # A per-rank sliced arg (x[r]) is keyed by a SlicedArg, not a
                # caller variable name; DynDim does not flow through the dropped
                # leading dim, so skip it here.
                if caller_arg is None or isinstance(caller_arg, SlicedArg):
                    continue
                target = caller_map.setdefault(caller_arg, {})
                for i, dyn in dim_to_dyn.items():
                    target.setdefault(i, dyn)
    return out


class DepBinding(NamedTuple):
    """A JIT dep together with the name its caller's source calls it by.

    The two differ whenever the caller reaches the dep through a rebinding —
    ``from mod import kernel as kern``, or a module-level ``kern = kernel``.
    Anything that has to *find the call site* (argument extraction, ``Out``
    param propagation, the ``self.<dep>(...)`` rewrite) must key on
    ``call_name``; anything that names the *function* (the generated
    ``@pl.function``, diagnostics, the source hash) keys on ``dep.__name__``.
    """

    call_name: str
    dep: "JITFunction"


# Resolve bindings in the caller-owned namespace snapshot, including recursive walks.
DiscoverDeps = Callable[[Any, str], list[DepBinding]]


class SlicedArg(NamedTuple):
    """A call-site argument of the form ``base[i]`` / ``base[i, j]`` that drops
    one or more leading dims of a Name (the per-rank ``chip_orch(x[r], ...)``
    dispatch pattern). ``drop`` counts the integer (non-slice) index elements;
    the dep parameter inherits ``base``'s meta with those leading dims removed.
    """

    base: str
    drop: int


def arg_ref(arg: ast.expr) -> str | SlicedArg | None:
    """Caller-side reference for a call argument.

    - ``ast.Name`` → the variable name (``str``).
    - ``ast.Subscript`` of a Name with integer indices (``x[r]``, ``x[r, 0]``)
      → a ``SlicedArg`` recording the base name and how many leading
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
            return SlicedArg(arg.value.id, drop)
    return None


def extract_call_args_for_dep(
    entry_func: Any, dep_name: str
) -> list[tuple[str | None, str | SlicedArg | None]] | None:
    """Find the arguments passed to ``dep_name`` in ``entry_func``'s body.

    Returns a unified list of ``(param_name, arg_ref)`` pairs:

    - ``param_name`` is ``None`` for a positional argument (the consumer
      pairs it with the dep's parameter list by index) and the keyword
      name for a keyword argument.
    - ``arg_ref`` is the caller-side reference: a variable name (``str``), a
      ``SlicedArg`` for a per-rank subscript (``x[r]``), or ``None`` for
      other non-``Name`` expressions (literals, attribute access, …).

    Mixed calls like ``dep(a, out=out)`` are preserved correctly. Returns
    ``None`` if no call site to ``dep_name`` is found. Only the first call
    site is examined.
    """
    func_def = _ast_analysis.get_func_def(entry_func)
    calls = [
        node
        for node in ast.walk(func_def)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == dep_name
    ]
    if not calls:
        return None
    return call_arg_refs(min(calls, key=lambda call: (call.lineno, call.col_offset)))


def _dep_call_nodes(caller_func: Any, dep_call_name: str) -> list[ast.Call]:
    """Every call to ``dep_call_name`` in ``caller_func``'s body, in source order."""
    func_def = _ast_analysis.get_func_def(caller_func)
    calls = [
        node
        for node in ast.walk(func_def)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == dep_call_name
    ]
    return sorted(calls, key=lambda call: (call.lineno, call.col_offset))


def _fold_call_site_constant(
    arg: ast.expr,
    caller_constexpr: Mapping[str, str],
    caller_globals: Mapping[str, Any],
    caller_locals: frozenset[str],
) -> str | None:
    """Source text a dep's constexpr argument folds to, or None when it does not.

    A bare name is either one of the caller's own constexpr parameters, which
    forwards, or a module-level/closure constant. Anything else is evaluated in
    the caller's namespace and rendered by the same writer the entry path uses,
    so every documented value form reaches a dep the same way it reaches an
    entry: ``helper(x, -1)``, ``helper(x, [16, 32])``, ``helper(x, pl.FP32)``,
    ``helper(x, pl.Mem.Vec)``.

    ``caller_locals`` are the caller's parameters and assignment targets. A
    local shadows a same-named global, so it is removed from the namespace
    before anything is resolved: without that, a caller with a runtime scalar
    ``n`` and an unrelated module-level ``n = 16`` binds the dep's compile-time
    parameter to 16 instead of rejecting the runtime value, and the dep is
    compiled against a constant the caller never passed.

    Returns None when there is no compile-time value — a runtime scalar, a
    tensor, an expression over either — so the caller can report it against the
    parameter it was bound to.
    """
    from .specializer import constant_source  # noqa: PLC0415

    if isinstance(arg, ast.Name):
        forwarded = caller_constexpr.get(arg.id)
        if forwarded is not None:
            return forwarded
        if arg.id in caller_locals:
            return None
        return free_name_source(arg.id, caller_globals)
    # Trusted input: the caller's own source, evaluated in its own globals with
    # builtins stripped and its locals removed, so only constant construction
    # can run and any run-time name is absent and raises — the rejection we
    # want. Forwarded constexpr parameters are re-bound because they *do* have a
    # compile-time value, unlike the runtime locals they sit beside.
    namespace = {k: v for k, v in caller_globals.items() if k not in caller_locals}
    namespace["__builtins__"] = {}
    for forwarded_name, text in caller_constexpr.items():
        try:
            namespace[forwarded_name] = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            continue
    try:
        value = eval(ast.unparse(arg), namespace)  # noqa: S307
    except Exception:  # noqa: BLE001 - any failure means "no compile-time value"
        return None
    return constant_source(value)


def resolve_call_site_constexpr(
    dep_name: str,
    signature: tuple[list[str], list[str]],
    call: ast.Call,
    caller_name: str,
    caller_binding: Mapping[str, str],
    caller_globals: Mapping[str, Any],
    caller_locals: frozenset[str],
) -> dict[str, str]:
    """Resolve one call site's ``pl.constexpr`` arguments into folded source text.

    Args:
        dep_name: The dependency's name, for diagnostics
        signature: ``(constexpr parameter names, all parameter names)`` for the
            dependency. Passed in rather than re-derived because the caller
            resolves every call site of a dep and both lists cost a full
            ``inspect.signature`` with annotation resolution.
        call: The call node whose arguments to bind
        caller_name: The calling function's name, for diagnostics
        caller_binding: The caller's own resolved constexpr values, so a
            forwarded parameter (``helper(x, BLOCK)``) carries through
        caller_globals: The caller's global/closure namespace
        caller_locals: The caller's parameters and assignment targets, which
            shadow same-named globals

    Returns:
        Folded source text per constexpr parameter name

    Raises:
        TypeError: if a constexpr parameter is unbound at this call site or is
            bound to something with no compile-time value.
    """
    constexpr_names, dep_param_names = signature
    bound: dict[str, ast.expr] = {}
    for index, arg in enumerate(call.args):
        if index < len(dep_param_names):
            bound[dep_param_names[index]] = arg
    for keyword in call.keywords:
        if keyword.arg is not None:
            bound[keyword.arg] = keyword.value

    resolved: dict[str, str] = {}
    for name in constexpr_names:
        arg = bound.get(name)
        if arg is None:
            raise TypeError(
                f"@pl.jit function '{dep_name}': constexpr parameter '{name}' is not "
                f"bound at the call site in '{caller_name}'. A 'pl.constexpr' parameter is "
                f"resolved at compile time, so every call must pass it — a signature "
                f"default is not read for a dep call."
            )
        folded = _fold_call_site_constant(arg, caller_binding, caller_globals, caller_locals)
        if folded is None:
            raise TypeError(
                f"@pl.jit function '{dep_name}': constexpr parameter '{name}' is bound "
                f"to '{ast.unparse(arg)}' at the call site in '{caller_name}', which has "
                f"no compile-time value. Pass a literal, one of the caller's own "
                f"'pl.constexpr' parameters, or a module-level/closure constant."
            )
        resolved[name] = folded
    return resolved


# A resolved compile-time binding: sorted ``(param, folded source text)`` pairs.
# Hashable, so it can stand in the identity of one generated function.
_ConstexprBinding = tuple[tuple[str, str], ...]
# One generated ``@pl.function``: a Python function plus the compile-time
# binding it is compiled against. Everything downstream used to key on
# ``id(func)`` alone, which is why a dep reached with two different constants
# had to be refused -- there was one slot to put it in.
VariantKey = tuple[int, _ConstexprBinding]


def freeze_binding(values: Mapping[str, str]) -> _ConstexprBinding:
    """Normalise a resolved binding into its hashable identity form."""
    return tuple(sorted(values.items()))


@dataclass
class VariantPlan:
    """Every generated function for one request, and how call sites reach them.

    A ``pl.constexpr`` argument is folded into the callee's body, so two call
    sites passing different constants need two different generated functions.
    This plan is the expansion of the (constexpr-independent, cached) dep graph
    into those per-binding functions, computed fresh per request because the
    entry's own binding seeds it.

    Attributes:
        order: Every variant in leaf-first order, entry last — the order the
            generated ``@pl.program`` must define them in.
        jit_func: The ``JITFunction`` each variant specializes.
        binding: Folded constexpr text per parameter, per variant.
        origin: ``(caller variant, call name, call node)`` for the first call
            site that produced each dep variant. Metadata is resolved from that
            site, so each binding reads the tensors live at its own call rather
            than at the first call sharing its name.
        call_targets: Per variant, ``(call name, ordinal) -> callee variant``.
            The ordinal is the call's position among same-named calls in source
            order, which is how the specializer tells two otherwise identical
            call sites apart.
    """

    order: list[VariantKey]
    jit_func: "dict[VariantKey, JITFunction]"
    binding: dict[VariantKey, dict[str, str]]
    origin: dict[VariantKey, tuple[VariantKey, str, ast.Call]]
    call_targets: dict[VariantKey, dict[tuple[str, int], VariantKey]]

    def naming_order(self, entry_key: VariantKey) -> "list[tuple[VariantKey, JITFunction]]":
        """The order generated names are handed out in, entry first.

        Functions keep the leaf-first order names were allocated in before
        variants existed, so a call graph with one binding per dep gets exactly
        the names it got then. Within one function, variants are ordered by the
        call site that produced them, so the first site in the source keeps the
        unsuffixed name and later ones take ``__2``, ``__3``, … — reading the
        generated program in source order then matches reading the original.
        """
        groups: dict[int, list[VariantKey]] = {}
        func_order: list[int] = []
        for key in self.order:
            if key == entry_key:
                continue
            func_id = key[0]
            if func_id not in groups:
                groups[func_id] = []
                func_order.append(func_id)
            groups[func_id].append(key)
        result: list[tuple[VariantKey, JITFunction]] = [(entry_key, self.jit_func[entry_key])]
        for func_id in func_order:
            # ``order`` is the reverse of the discovery walk, so one function's
            # variants sit in reverse source order; flip them back.
            result.extend((key, self.jit_func[key]) for key in reversed(groups[func_id]))
        return result


def expand_constexpr_variants(
    entry: "JITFunction",
    deps_topo: "Sequence[JITFunction]",
    callers_by_id: Mapping[int, Sequence[tuple[Any, str]]],
    entry_constexpr: Mapping[str, str],
) -> VariantPlan:
    """Expand the dep graph into one generated function per compile-time binding.

    Walks caller-first (``reversed(deps_topo)``), so by the time a dep is
    reached every caller variant that can reach it already exists and its own
    forwarded values are resolved. Each call site is folded independently; call
    sites that agree on every value collapse onto one variant, which is what
    keeps the ordinary single-binding program emitting exactly one function per
    dep.

    Args:
        entry: The entry function, whose binding seeds the walk
        deps_topo: Every reachable dep, leaf-first
        callers_by_id: ``id(dep func) -> [(caller func, call name), ...]``
        entry_constexpr: The entry's own resolved binding

    Returns:
        The full :class:`VariantPlan` for this request

    Raises:
        TypeError: if a constexpr parameter is unbound at a call site or bound
            to something with no compile-time value.
    """
    entry_key: VariantKey = (id(entry._func), freeze_binding(entry_constexpr))
    plan = VariantPlan(
        order=[entry_key],
        jit_func={entry_key: entry},
        binding={entry_key: dict(entry_constexpr)},
        origin={},
        call_targets={entry_key: {}},
    )
    variants_by_func: dict[int, list[VariantKey]] = {id(entry._func): [entry_key]}

    for dep in reversed(deps_topo):
        # Both lists cost a full ``inspect.signature`` with annotation
        # resolution, so they are derived once per dep rather than per call
        # site. A dep declaring no constexpr parameter cannot split at all, and
        # skips call-site folding entirely.
        signature = (_arguments.constexpr_params(dep._func), dep._param_names())
        splits = bool(signature[0])
        for caller_func, call_name in callers_by_id.get(id(dep._func), ()):
            caller_globals = func_name_lookup(caller_func)
            caller_locals = frozenset(_ast_analysis.body_local_names(caller_func))
            caller_name = getattr(caller_func, "__name__", "<caller>")
            call_nodes = _dep_call_nodes(caller_func, call_name)
            for caller_key in variants_by_func.get(id(caller_func), ()):
                caller_binding = plan.binding[caller_key]
                for ordinal, call in enumerate(call_nodes):
                    values = (
                        resolve_call_site_constexpr(
                            dep.__name__,
                            signature,
                            call,
                            caller_name,
                            caller_binding,
                            caller_globals,
                            caller_locals,
                        )
                        if splits
                        else {}
                    )
                    dep_key: VariantKey = (id(dep._func), freeze_binding(values))
                    if dep_key not in plan.binding:
                        plan.order.append(dep_key)
                        plan.jit_func[dep_key] = dep
                        plan.binding[dep_key] = values
                        plan.call_targets[dep_key] = {}
                        plan.origin[dep_key] = (caller_key, call_name, call)
                        variants_by_func.setdefault(id(dep._func), []).append(dep_key)
                    plan.call_targets[caller_key][(call_name, ordinal)] = dep_key
        if id(dep._func) not in variants_by_func:
            # Reached by dep discovery but with no ``name(...)`` call node to
            # fold against. It still has to be emitted -- dropping it would
            # leave the caller calling an undefined function -- so it gets the
            # empty binding, exactly what it received before variants existed.
            fallback: VariantKey = (id(dep._func), ())
            plan.order.append(fallback)
            plan.jit_func[fallback] = dep
            plan.binding[fallback] = {}
            plan.call_targets[fallback] = {}
            variants_by_func[id(dep._func)] = [fallback]

    # Collected caller-first; the generated program must define callees first.
    plan.order.reverse()
    return plan


def call_arg_refs(node: ast.Call) -> list[tuple[str | None, str | SlicedArg | None]]:
    """Unify one call node's positional and keyword args into ``(param, ref)`` pairs.

    ``param`` is ``None`` for a positional argument (paired with the callee's
    parameter list by index in :func:`build_param_mapping`) and the keyword
    name otherwise. ``**kwargs`` splats are skipped — they carry no name to
    bind against.
    """
    result: list[tuple[str | None, str | SlicedArg | None]] = [(None, arg_ref(arg)) for arg in node.args]
    result.extend((kw.arg, arg_ref(kw.value)) for kw in node.keywords if kw.arg is not None)
    return result


def build_param_mapping(
    dep_param_names: list[str],
    call_args: list[tuple[str | None, str | SlicedArg | None]],
) -> dict[str, str | SlicedArg | None]:
    """Map dep parameter name → caller argument ref from call-site args.

    ``call_args`` is the unified form returned by
    ``extract_call_args_for_dep``: a list of ``(param_name, arg_ref)``
    pairs where ``param_name is None`` marks a positional argument (paired
    with ``dep_param_names`` by index) and a string is a keyword name. The
    ``arg_ref`` may be a name (``str``), a ``SlicedArg``, or ``None``.
    Mixed positional + keyword call sites collapse to the same dict.
    """
    mapping: dict[str, str | SlicedArg | None] = {}
    pos_idx = 0
    for param_name, arg_name in call_args:
        if param_name is None:
            if pos_idx < len(dep_param_names):
                mapping[dep_param_names[pos_idx]] = arg_name
            pos_idx += 1
        else:
            mapping[param_name] = arg_name
    return mapping
