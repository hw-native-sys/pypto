# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime scope context managers and submit primitive for the PyPTO Language DSL."""

from enum import Enum
from typing import Any


class ScopeMode(Enum):
    """Dependency-tracking mode of a runtime scope (``PTO2_SCOPE``).

    - ``AUTO``: OverlapMap auto-dependency tracking is on (``PTO2_SCOPE()``).
    - ``MANUAL``: auto tracking is off; the user declares every edge via
      ``pl.submit(..., deps=[...])`` (``PTO2_SCOPE(PTO2ScopeMode::MANUAL)``).
    """

    AUTO = 0
    MANUAL = 1


class scope:
    """Context manager marking a runtime scope (``PTO2_SCOPE``) region.

    A runtime scope is a resource-management + dependency-tracking boundary in
    the simpler runtime: it bounds OverlapMap auto-dependency tracking and gives
    a per-scope HeapRing level (nested scopes reclaim memory independently). The
    simpler runtime provides an implicit top-level scope, so writing scopes is
    a **tuning / control** mechanism, never a correctness requirement.

    By default the compiler inserts AUTO scopes for you (function body + each
    ``for`` / ``if`` body). To place scopes yourself, set
    ``@pl.function(auto_scope=False)`` and use this context manager (and the
    ``pl.range(..., scope=...)`` sugar). See :class:`ScopeMode` for AUTO vs
    MANUAL.

    Usage::

        with pl.scope():                          # AUTO
            out = self.kernel(a, b, out)

        with pl.scope(mode=pl.ScopeMode.MANUAL):  # MANUAL — user owns deps
            out, tid = pl.submit(self.stage1, x, out)

    Rules:
      - Must appear inside an Orchestration function (not InCore).
      - ``mode=AUTO`` is only allowed under ``@pl.function(auto_scope=False)``
        (in the default ``auto_scope=True`` the compiler owns AUTO placement).
      - ``mode=MANUAL`` is allowed in either mode (it is a dependency-semantics
        choice, not ring tuning).
      - AUTO scope may not nest inside a MANUAL scope (runtime forbids).
    """

    def __init__(self, mode: "ScopeMode" = ScopeMode.AUTO):
        self.mode = mode

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class manual_scope:
    """Alias for ``pl.scope(mode=pl.ScopeMode.MANUAL)``.

    Turns OverlapMap auto dep-tracking off for a region.

    Inside this block, the simpler runtime skips OverlapMap lookup and insert
    for every kernel submit, so the user takes full responsibility for
    declaring task-to-task ordering edges. ``manual_scope`` is the
    coarsest-grained of the runtime's auto-dep-tracking opt-outs; finer
    granularities are available and compose with auto scope:

      - ``with pl.manual_scope():`` — this construct; whole-region opt-out.
      - ``pl.create_tensor(..., manual_dep=True)`` — opt out a single tensor
        for its entire lifetime (any task referencing it skips OverlapMap).
      - ``pl.no_dep(arg)`` at a kernel-call arg position — opt out a single
        tensor for a single task only.

    Manual dependency edges (``pl.submit(..., deps=[...])``) are **orthogonal**
    to all of the above: the runtime adds them on top of whatever auto-tracked
    deps remain (final fanin = auto ∪ explicit), so ``deps=`` works in auto
    scope too. Use ``manual_scope`` only when you want full ownership of the
    dep graph; otherwise stay in auto scope and use ``pl.submit(..., deps=)``
    as a precision tool that patches the edges auto cannot infer.

    Usage::

        # Full-manual region.
        with pl.manual_scope():
            scratch, tid = pl.submit(self.stage1, x, scratch)
            out, _       = pl.submit(self.stage2, scratch, out, deps=[tid])

        # Auto scope with explicit-edge patching (no manual_scope needed).
        a, a_tid = pl.submit(self.k1, x)
        b, _     = pl.submit(self.k2, x, deps=[a_tid])

    Restrictions:
      - Must appear inside an Orchestration function (not InCore).
      - Cannot be nested inside another ``manual_scope`` (runtime forbids).
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def submit(*args: Any, **kwargs: Any) -> Any:
    """Submit a kernel and capture its producer TaskId.

    ``pl.submit`` is a **parser construct**, not a runtime function — the
    DSL parser intercepts ``result, tid = pl.submit(self.kernel, *args,
    deps=[...])`` syntactically and never actually calls this body. It is
    defined only so the name resolves (for imports / linters).

    Surface form (must be unpacked as a 2-tuple)::

        out, tid    = pl.submit(self.stage1, x, scratch, deps=[prev_tid])
        (a, b), tid = pl.submit(self.multi_out_kernel, x)
        out, tid    = pl.submit(self.stage1, x, scratch, deps=[prev_tid], dumps=[x])
        out, tid    = pl.submit(self.stage1, x, scratch, allow_early_resolve=True)

    The kernel-side ``ir.Submit`` natively returns
    ``Tuple[<kernel return>, TASK_ID]``; element 0 is the tensor result(s),
    element 1 is the producer TaskId (``Scalar[TASK_ID]``). The optional
    ``deps=[...]`` kwarg lists TaskId scalars / arrays this submit must
    wait on. The single-LHS form ``res = pl.submit(...)`` is also accepted
    and binds the whole flat tuple to one variable.

    ``pl.submit`` and its ``deps=`` kwarg work in **both auto and manual
    scope**: ``Arg::set_dependencies`` is orthogonal to OverlapMap
    auto-tracking (final fanin = auto ∪ explicit). In auto scope, use
    ``deps=[...]`` as a precision tool to patch edges the runtime cannot
    infer; in ``pl.manual_scope()``, use it to declare every edge.

    The optional ``dumps=[...]`` kwarg is the submit-side selective tensor
    dump surface (symmetric with ``deps=``): it lists tensor arguments of
    this submit to mark for dump (simpler#844), so an enabled dump pipeline
    filters down to just those bindings. Each entry must be a tensor passed
    positionally to the submitted kernel. ``dumps=`` is the explicit dump
    surface on a submit; the declarative ``pl.dump_tag(t)`` statement feeds the
    same ``dump_vars`` set. These marks only take effect under partial dump
    (``RunConfig.enable_dump_tensor == 1``); they are a no-op when dump is off
    (``0``) and irrelevant under full dump (``2``), which captures every
    binding regardless.

    The optional ``allow_early_resolve=True`` kwarg (default ``False``) opts
    this task in as a speculative early-dispatch producer (simpler#1065): the
    scheduler may pre-stage this task's consumers onto idle cores before it
    completes, releasing them with a doorbell the instant it finishes. It is a
    producer-side hint — a consumer only pre-stages once *all* of its producers
    are flagged (or already complete). It lowers to
    ``Arg::set_allow_early_resolve(true)`` in orchestration codegen and is a
    pure scheduling optimisation (no effect on results). Pays off on critical
    paths built from many short tasks; harmless otherwise.

    The return annotation is ``Any`` (not ``NoReturn``) because the parser
    intercepts the call and binds a 2-tuple to the LHS — downstream code
    that does ``out, tid = pl.submit(...)`` would not type-check under
    ``NoReturn``.
    """
    raise RuntimeError(
        "pl.submit is a DSL parser construct and cannot be called directly; "
        "use it as `result, task_id = pl.submit(self.kernel, *args, deps=[...])` "
        "inside a @pl.function body."
    )


def spmd_submit(*args: Any, **kwargs: Any) -> Any:
    """Launch a kernel as an SPMD task and capture its producer TaskId.

    ``pl.spmd_submit`` is a **parser construct**, not a runtime function — the
    DSL parser intercepts ``result, tid = pl.spmd_submit(self.kernel, *args,
    core_num=N, sync_start=..., deps=[...])`` syntactically and never actually
    calls this body. It is defined only so the name resolves (imports / linters).

    It is the SPMD sibling of :func:`submit`: a single orchestration task that
    the runtime fans out across ``core_num`` logical blocks (each kernel reads
    its block index via ``pl.tile.get_block_idx()``). Like :func:`submit` it
    returns one producer TaskId, so the whole dispatch can be named as a
    dependency of later tasks.

    Surface form (must be unpacked as a 2-tuple)::

        out, tid = pl.spmd_submit(self.incore_kernel, x, y, core_num=8)
        out, tid = pl.spmd_submit(self.kernel, x, core_num=8, sync_start=True,
                                  deps=[prev_tid])

    ``core_num`` is a **required keyword argument** (a positive integer
    expression) — the positional slots are the kernel's own arguments.
    ``sync_start`` (default ``False``) requires all blocks to launch atomically.
    ``deps=[...]`` and ``allow_early_resolve=True`` work exactly as on
    :func:`submit` (note: a ``sync_start`` task cannot itself be block-by-block
    pre-staged, but it can still be flagged to let its consumers pre-stage). The
    callee may be an InCore / AIC / AIV kernel or a co-scheduled Group.

    Like :func:`submit`, it works in both auto and manual scope; its primary use
    is explicit dependency wiring inside ``pl.manual_scope()``.
    """
    raise RuntimeError(
        "pl.spmd_submit is a DSL parser construct and cannot be called directly; "
        "use it as `result, task_id = pl.spmd_submit(self.kernel, *args, core_num=N, deps=[...])` "
        "inside a @pl.function body."
    )


__all__ = ["ScopeMode", "manual_scope", "scope", "submit", "spmd_submit"]
