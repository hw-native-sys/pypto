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
    """Context manager for a manual-dependency runtime scope.

    Inside this block, the simpler runtime skips OverlapMap dependency
    tracking and TensorMap insert. The user declares every required
    task-to-task ordering edge explicitly:

      - A plain ``out = self.kernel(...)`` call is fire-and-forget — no
        task id, and ``deps=`` is rejected.
      - ``out, tid = pl.submit(self.kernel, ...)`` submits the kernel,
        binds the result tensor(s) to ``out`` and the producer TaskId to
        ``tid``, and accepts an optional ``deps=[...]`` kwarg.
      - ``pl.no_dep(arg)`` suppresses the auto-derived edge for a single
        argument (auto-scope primitive; no effect inside manual_scope).

    Usage::

        with pl.manual_scope():
            scratch, tid = pl.submit(self.stage1, x, scratch)
            out, _       = pl.submit(self.stage2, scratch, out, deps=[tid])

    Restrictions:
      - Must appear inside an Orchestration function (not InCore).
      - Cannot be nested inside another ``manual_scope`` (runtime forbids).
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def submit(*args: Any, **kwargs: Any) -> NoReturn:
    """Submit a kernel inside a ``manual_scope`` and capture its TaskId.

    ``pl.submit`` is a **parser construct**, not a runtime function — the
    DSL parser intercepts ``result, tid = pl.submit(self.kernel, *args,
    deps=[...])`` syntactically and never actually calls this body. It is
    defined only so the name resolves (for imports / linters).

    Surface form (must be unpacked as a 2-tuple)::

        out, tid       = pl.submit(self.stage1, x, scratch, deps=[prev_tid])
        (a, b), tid    = pl.submit(self.multi_out_kernel, x)

    The kernel ``Call`` natively returns ``Tuple[<kernel return>, TASK_ID]``;
    element 0 is the tensor result(s), element 1 is the producer TaskId
    (``Scalar[TASK_ID]``). The optional ``deps=[...]`` kwarg lists TaskId
    scalars / arrays this submit must wait on.
    """
    raise RuntimeError(
        "pl.submit is a DSL parser construct and cannot be called directly; "
        "use it as `result, task_id = pl.submit(self.kernel, *args, deps=[...])` "
        "inside a @pl.function body."
    )


__all__ = ["manual_scope", "submit"]
