# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for user-managed (pinned) tile buffers — the custom-buffer feature.

A user pins a tile to an explicit buffer by annotating its TileType with a
``pl.MemRef(addr, size, id)``. The feature spans three passes:

* InitMemRef honors the pin (interns the base by id, keeps the user's address)
  and enforces the whole-space-manual contract (every tile in a pinned space
  must itself be pinned).
* MemoryReuse never coalesces a pinned buffer, and rejects two distinct pinned
  buffers whose addresses AND lifetimes both overlap.
* AllocateMemoryAddr honors the user's address verbatim (no auto re-layout).

Tiles sharing one ``id`` share one physical buffer; distinct ids get separate
buffers that MemoryReuse never merges (preventing over-reuse).

Annotations must be written with literal memory spaces / integers because the
parser resolves type annotations statically from the AST (a module-level alias
like ``VEC`` is not recognised inside an annotation). ``SIZE`` below is only used
in runtime assertions, never inside an annotation. 16384 == 64 * 64 * 4 (FP32).
"""

from typing import cast

import pypto.language as pl
import pytest

from pypto import ir, passes

SIZE = 16384  # 64 * 64 * 4 bytes (FP32)


def _reuse(program: ir.Program) -> ir.Program:
    """init_mem_ref + memory_reuse."""
    return passes.memory_reuse()(passes.init_mem_ref()(program))


def _allocate(program: ir.Program) -> ir.Program:
    """init_mem_ref + memory_reuse + allocate_memory_addr."""
    return passes.allocate_memory_addr()(_reuse(program))


def _tile_memrefs(program: ir.Program) -> dict[str, ir.MemRef]:
    """Map each top-level tile var name to its (non-None) MemRef (after passes)."""
    func = next(iter(program.functions.values()))
    result: dict[str, ir.MemRef] = {}
    for stmt in cast(ir.SeqStmts, func.body).stmts:
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.var.type, ir.TileType):
            memref = stmt.var.type.memref
            if memref is not None:
                result[stmt.var.name_hint] = memref
    return result


def _addr(memref: ir.MemRef) -> int | None:
    bo = memref.byte_offset_
    return bo.value if isinstance(bo, ir.ConstInt) else None


def test_pinned_distinct_buffers_not_merged():
    """Distinct ids → distinct buffers MemoryReuse must not merge.

    Without pins this producer-consumer chain collapses into ONE buffer
    (see test_memory_reuse.TestBasic.test_sequential). Pinning ``a``/``c`` to
    buffer 0 and ``b`` to buffer 1 keeps ``b`` on its own buffer.
    """

    @pl.program
    class Prog:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(0, 16384, 0), pl.MemorySpace.Vec] = pl.load(
                input_a, [0, 0], [64, 64]
            )
            b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(16384, 16384, 1), pl.MemorySpace.Vec] = pl.add(a, a)
            c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(0, 16384, 0), pl.MemorySpace.Vec] = pl.add(b, b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(c, [0, 0], output)
            return result

    mrs = _tile_memrefs(_reuse(Prog))
    # a and c share buffer id 0; b is on its own buffer id 1 — never merged.
    assert ir.MemRef.same_allocation(mrs["a"], mrs["c"]), "same id must share one buffer"
    assert not ir.MemRef.same_allocation(mrs["a"], mrs["b"]), "distinct ids must NOT be merged"
    assert not ir.MemRef.same_allocation(mrs["b"], mrs["c"]), "distinct ids must NOT be merged"


def test_pinned_addresses_honored_verbatim():
    """AllocateMemoryAddr must emit exactly the user-assigned addresses."""

    @pl.program
    class Prog:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(0, 16384, 0), pl.MemorySpace.Vec] = pl.load(
                input_a, [0, 0], [64, 64]
            )
            b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(16384, 16384, 1), pl.MemorySpace.Vec] = pl.add(a, a)
            c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(0, 16384, 0), pl.MemorySpace.Vec] = pl.add(b, b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(c, [0, 0], output)
            return result

    mrs = _tile_memrefs(_allocate(Prog))
    assert _addr(mrs["a"]) == 0
    assert _addr(mrs["b"]) == SIZE
    assert _addr(mrs["c"]) == 0  # reuses buffer 0 (a is dead — legal manual reuse)


def test_lifetime_disjoint_same_address_distinct_ids_ok():
    """Distinct ids at the SAME address with disjoint lifetimes is legal manual reuse."""

    @pl.program
    class Prog:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(0, 16384, 0), pl.MemorySpace.Vec] = pl.load(
                input_a, [0, 0], [64, 64]
            )
            b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(16384, 16384, 1), pl.MemorySpace.Vec] = pl.add(a, a)
            # c at addr 0 but a DIFFERENT id than a; a is already dead → allowed.
            c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(0, 16384, 2), pl.MemorySpace.Vec] = pl.add(b, b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(c, [0, 0], output)
            return result

    mrs = _tile_memrefs(_allocate(Prog))
    assert _addr(mrs["a"]) == 0
    assert _addr(mrs["c"]) == 0
    assert not ir.MemRef.same_allocation(mrs["a"], mrs["c"]), "different ids stay distinct buffers"


def test_overlapping_buffers_overlapping_lifetimes_error():
    """Two distinct pinned buffers overlapping in BOTH address and lifetime → error."""

    @pl.program
    class Prog:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            input_b: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            # a [0, 16384) and b [8192, 24576) overlap; both live until c reads them.
            a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(0, 16384, 0), pl.MemorySpace.Vec] = pl.load(
                input_a, [0, 0], [64, 64]
            )
            b: pl.Tile[[64, 64], pl.FP32, pl.MemRef(8192, 16384, 1), pl.MemorySpace.Vec] = pl.load(
                input_b, [0, 0], [64, 64]
            )
            c: pl.Tile[[64, 64], pl.FP32, pl.MemRef(32768, 16384, 2), pl.MemorySpace.Vec] = pl.add(a, b)
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(c, [0, 0], output)
            return result

    with pytest.raises(ValueError, match="overlap"):
        _reuse(Prog)


def test_partial_manual_space_error():
    """A pinned space with an unannotated fresh tile → error (not a pure tile kernel)."""

    @pl.program
    class Prog:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(0, 16384, 0), pl.MemorySpace.Vec] = pl.load(
                input_a, [0, 0], [64, 64]
            )
            b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(a, a)  # no buffer pin → error
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(b, [0, 0], output)
            return result

    with pytest.raises(ValueError, match="user-managed buffer mode"):
        passes.init_mem_ref()(Prog)


def test_pinned_buffer_too_small_error():
    """A pin whose declared size is smaller than the tile footprint → error."""

    @pl.program
    class Prog:
        @pl.function
        def main(
            self,
            input_a: pl.Tensor[[64, 64], pl.FP32],
            output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            a: pl.Tile[[64, 64], pl.FP32, pl.MemRef(0, 1024, 0), pl.MemorySpace.Vec] = pl.load(
                input_a, [0, 0], [64, 64]
            )
            result: pl.Tensor[[64, 64], pl.FP32] = pl.store(a, [0, 0], output)
            return result

    with pytest.raises(ValueError, match="too small"):
        passes.init_mem_ref()(Prog)


def test_l0_spaces_pinnable_in_explicit_matmul():
    """L0 matmul operands/accumulator (Left/Right/Acc) are pinnable too.

    When a matmul is written explicitly (load -> move-to-Left/Right -> matmul),
    its L0 tiles are user-authored, so they pin on the same terms as Vec/Mat.
    This is what lets a fully-explicit attention kernel pin EVERY tile. Address
    handling is backend-agnostic, so this runs under whatever backend is active.
    """

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def mm(
            self,
            q: pl.Tensor[[16, 128], pl.BF16],
            k: pl.Tensor[[128, 128], pl.BF16],
            out: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            q_mat: pl.Tile[[16, 128], pl.BF16, pl.MemRef(0, 4096, 0), pl.MemorySpace.Mat] = pl.tile.load(
                q, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
            )
            k_mat: pl.Tile[[128, 128], pl.BF16, pl.MemRef(4096, 32768, 1), pl.MemorySpace.Mat] = pl.tile.load(
                k, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
            )
            k_t = pl.tile.transpose_view(k_mat)
            q_left: pl.Tile[[16, 128], pl.BF16, pl.MemRef(0, 4096, 2), pl.MemorySpace.Left] = pl.tile.move(
                q_mat, target_memory=pl.MemorySpace.Left
            )
            k_right: pl.Tile[[128, 128], pl.BF16, pl.MemRef(0, 32768, 3), pl.MemorySpace.Right] = (
                pl.tile.move(k_t, target_memory=pl.MemorySpace.Right)
            )
            scores: pl.Tile[[16, 128], pl.FP32, pl.MemRef(0, 8192, 4), pl.MemorySpace.Acc] = pl.tile.matmul(
                q_left, k_right
            )
            out = pl.tile.store(scores, [0, 0], out)
            return out

    mrs = _tile_memrefs(_allocate(Prog))
    # Every L0 tile is honored at its user-assigned address, on a pinned base.
    for name in ("q_left", "k_right", "scores"):
        assert mrs[name].base_.name_hint.startswith("__pinned__"), f"{name} must be user-pinned"
        assert _addr(mrs[name]) == 0  # each L0 space is independent → addr 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
