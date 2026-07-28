# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""User-owned buffers for manual memory-reuse control.

``pl.Buffer()`` declares a buffer the kernel author owns. Tiles bound to it
share its storage; the compiler's opportunistic reuse (``MemoryReuse``) never
packs anything else into it. Use it to stop the packer from coalescing tiles
whose lifetimes happen not to overlap but which you want to stay independent —
coalescing them creates a false dependency that serializes the two.
"""


class Buffer:
    """A user-owned buffer.

    Declare one, then reference it in the ``pl.Tile[...]`` annotation's trailing
    slot. Two tiles referencing the same buffer share one allocation; a tile with
    a buffer to itself is never packed together with anything else::

        scratch = pl.Buffer()   # takes its name from the variable

        t0: pl.Tile[[64, 64], pl.FP32, scratch, pl.Mem.Vec] = pl.load(x, [0, 0], [64, 64])

    Prefer this declare-then-reference form: a misspelled reference is a Python
    ``NameError``, whereas a misspelled string in the inline
    ``pl.Buffer("scratch")`` form silently declares a second buffer. The inline
    form remains valid — it is what the IR printer emits, so a dumped program can
    be reparsed without a surrounding Python scope.

    Neither a size nor a memory space is written here — both are derived by the
    ``InitMemRef`` pass from the tiles actually bound to the buffer (size = the
    largest bound tile; memory space = the one they all share).

    A memory space is required alongside the buffer, as for any ``MemRef``. All
    tiles bound to one buffer must agree on it; the compiler checks that::

        stage_in, scratch = pl.Buffer(), pl.Buffer()

        # `a` gets a buffer of its own — the packer will not reuse it.
        a: pl.Tile[[64, 64], pl.FP32, stage_in, pl.Mem.Vec] = pl.load(x, [0, 0], [64, 64])
        # `b` and `c` explicitly share one buffer; their lifetimes must not
        # overlap, which the compiler checks.
        b: pl.Tile[[64, 64], pl.FP32, scratch, pl.Mem.Vec] = pl.exp(a)
        c: pl.Tile[[64, 64], pl.FP32, scratch, pl.Mem.Vec] = pl.exp(b)

    Buffers are function-scoped and do not clone per pipeline stage, so a binding
    inside a ``pl.pipeline(stage=2)`` body is rejected: the cloned stages would
    make the tile co-live with itself on one allocation. Declaring slots and asking
    the compiler to multi-buffer are alternatives, not layers — to manage a level
    yourself, drive it with ``pl.range`` and declare one buffer per slot::

        ping, pong = pl.Buffer(), pl.Buffer()

        # Author-managed ping-pong: two slots over a 2x-unrolled body.
        for i, (acc,) in pl.range(0, N, 2 * STEP, init_values=[out]):
            p0: pl.Tile[[64, 64], pl.FP32, ping, pl.Mem.Vec] = pl.load(x, [i, 0], [64, 64])
            p1: pl.Tile[[64, 64], pl.FP32, pong, pl.Mem.Vec] = pl.load(x, [i + STEP, 0], [64, 64])

    Note:
        Both spellings are resolved by the parser (``parser/type_resolver.py``)
        from the annotation's AST — a reference by reading the ``Buffer`` instance
        out of the enclosing scope, the inline form from its string literal. This
        class exists so the annotation is valid Python that type-checkers accept.
    """

    def __init__(self, name: str | None = None) -> None:
        """Declare a user-owned buffer.

        Args:
            name: Buffer name. Optional when the buffer is declared as a variable
                and referenced by name in the annotation — it then takes that
                variable's name, so the buffer is named once rather than twice.
                Required for the inline ``pl.Tile[..., pl.Buffer("scratch"), ...]``
                form, where there is no variable to take a name from.
        """
        self.name = name

    def __repr__(self) -> str:
        return "Buffer()" if self.name is None else f"Buffer({self.name!r})"


__all__ = ["Buffer"]
