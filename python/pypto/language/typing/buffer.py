# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""User-owned buffers for manual memory-reuse control.

``pl.Buffer("name")`` names a buffer the kernel author owns. Tiles bound to it
share its storage; the compiler's opportunistic reuse (``MemoryReuse``) never
packs anything else into it. Use it to stop the packer from coalescing tiles
whose lifetimes happen not to overlap but which you want to stay independent —
coalescing them creates a false dependency that serializes the two.
"""


class Buffer:
    """A user-owned buffer, identified by name within a function.

    Bind a tile to a buffer by naming it in the ``pl.Tile[...]`` annotation's
    trailing slot. Two tiles naming the same buffer share one allocation; a tile
    naming its own buffer is never packed together with anything else.

    Neither a size nor a memory space is written here — both are derived by the
    ``InitMemRef`` pass from the tiles actually bound to the buffer (size = the
    largest bound tile; memory space = the one they all share).

    A memory space is required alongside the buffer, as for any ``MemRef``::

        # `a` gets a buffer of its own — the packer will not reuse it.
        a: pl.Tile[[64, 64], pl.FP32, pl.Buffer("stage_in"), pl.Mem.Vec] = pl.load(x, [0, 0], [64, 64])
        # `b` and `c` explicitly share one buffer; their lifetimes must not
        # overlap, which the compiler checks.
        b: pl.Tile[[64, 64], pl.FP32, pl.Buffer("scratch"), pl.Mem.Vec] = pl.exp(a)
        c: pl.Tile[[64, 64], pl.FP32, pl.Buffer("scratch"), pl.Mem.Vec] = pl.exp(b)

    Buffers are function-scoped and do not clone per pipeline stage, so a binding
    inside a ``pl.pipeline(stage=2)`` body is rejected: the cloned stages would
    make the tile co-live with itself on one allocation. Naming slots and asking
    the compiler to multi-buffer are alternatives, not layers — to manage a level
    yourself, drive it with ``pl.range`` and name one buffer per slot::

        # Author-managed ping-pong: two slots over a 2x-unrolled body.
        for i, (acc,) in pl.range(0, N, 2 * STEP, init_values=[out]):
            ping: pl.Tile[[64, 64], pl.FP32, pl.Buffer("ping"), pl.Mem.Vec] = pl.load(x, [i, 0], [64, 64])
            pong: pl.Tile[[64, 64], pl.FP32, pl.Buffer("pong"), pl.Mem.Vec] = pl.load(
                x, [i + STEP, 0], [64, 64]
            )

    Note:
        ``pl.Buffer(...)`` inside a ``@pl.program`` body is resolved by the
        parser (``parser/type_resolver.py``); this class exists so the annotation
        is valid Python that type-checkers accept.
    """

    def __init__(self, name: str) -> None:
        """Name a user-owned buffer.

        Args:
            name: Buffer name. Buffers are identified by name within a function —
                two annotations naming the same buffer bind to one allocation.
                The parser is what validates it (a non-empty string literal); this
                constructor only exists so the annotation type-checks.
        """
        self.name = name

    def __repr__(self) -> str:
        return f"Buffer({self.name!r})"


__all__ = ["Buffer"]
