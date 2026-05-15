# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime scope context managers and submit primitive for the PyPTO Language DSL."""

from typing import Any, NoReturn


class manual_scope:
    """Context manager that turns OverlapMap auto dep-tracking off for a region.

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


def submit(*args: Any, **kwargs: Any) -> NoReturn:
    """Submit a kernel and capture its producer TaskId.

    ``pl.submit`` is a **parser construct**, not a runtime function — the
    DSL parser intercepts ``result, tid = pl.submit(self.kernel, *args,
    deps=[...])`` syntactically and never actually calls this body. It is
    defined only so the name resolves (for imports / linters).

    Surface form (must be unpacked as a 2-tuple)::

        out, tid    = pl.submit(self.stage1, x, scratch, deps=[prev_tid])
        (a, b), tid = pl.submit(self.multi_out_kernel, x)

    The kernel ``Call`` natively returns ``Tuple[<kernel return>, TASK_ID]``;
    element 0 is the tensor result(s), element 1 is the producer TaskId
    (``Scalar[TASK_ID]``). The optional ``deps=[...]`` kwarg lists TaskId
    scalars / arrays this submit must wait on.

    ``pl.submit`` and its ``deps=`` kwarg work in **both auto and manual
    scope**: ``Arg::set_dependencies`` is orthogonal to OverlapMap
    auto-tracking (final fanin = auto ∪ explicit). In auto scope, use
    ``deps=[...]`` as a precision tool to patch edges the runtime cannot
    infer; in ``pl.manual_scope()``, use it to declare every edge.
    """
    raise RuntimeError(
        "pl.submit is a DSL parser construct and cannot be called directly; "
        "use it as `result, task_id = pl.submit(self.kernel, *args, deps=[...])` "
        "inside a @pl.function body."
    )


__all__ = ["manual_scope", "submit"]
