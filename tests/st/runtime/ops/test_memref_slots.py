# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime st: a declared allocation's slots, selected by a *runtime* index.

``pl.MemRef("name", slots=2)`` reserves two equally-sized slots of one
allocation and ``[i % 2]`` picks one per iteration. The index is an ordinary
index expression, so the address is computed per iteration -- PTOAS takes
``alloc_tile addr`` by value, so that is a legal target::

    %0 = arith.remsi %i, %c2            // i % 2
    %1 = arith.muli  %0, %c16384        // * slot size
    %2 = arith.addi  %c16384, %1        // + allocation base
    %lo = pto.alloc_tile addr = ...

Both cases hold the two slots **co-live**: each iteration writes ``[i % 2]`` and
``[(i + 1) % 2]`` and only then consumes them. That is what makes the golden
comparison a check of the addressing rather than a smoke test -- if the two slot
addresses collapsed onto one (an index folded to a constant, a dropped dynamic
offset, a slot sized to the whole allocation so the two ranges overlap), the
second write would clobber the first and the result would be visibly wrong. A
unit test can assert the offset expression survives the pipeline; only the
device shows the address it computes is the right one.

Two cases, one per memory class, because the address reaches a different ISA
operand in each: ``pl.Mem.Acc`` (L0C, matmul output) and ``pl.Mem.Vec`` (UB).

Note the inline ``pl.MemRef("name", slots=2)`` spelling rather than a Python
variable bound to a declaration. Both are valid, but ``@pl.jit`` re-parses a
generated source in a fresh module namespace, so a declaration held in a Python
variable is not in scope there; the named form is self-contained (it is also the
form the IR printer emits).
"""

import pytest

torch = pytest.importorskip("torch")

import pypto.language as pl  # noqa: E402

# Acc case: [M, K] @ [K, N] in NT column tiles. One fp32 slot is M * TN * 4 = 16 KB,
# so the pair fits L0C with room to spare.
M, K, N = 64, 64, 256
TN = 64
NT = N // TN

# Vec case: NT row tiles of [TR, TC] fp32 -- 16 KB per slot.
TR, TC = 64, 64
ROWS = TR * NT


@pl.jit
def matmul_slot_pingpong(
    a: pl.Tensor[[M, K], pl.FP32],
    a2: pl.Tensor[[M, K], pl.FP32],
    b: pl.Tensor[[K, N], pl.FP32],
    out1: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    out2: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    """Two matmuls per iteration into the two L0C slots, co-live until drained."""
    for _ in pl.spmd(1, name_hint="memref_slots_acc"):
        la: pl.Tile[[M, K], pl.FP32, pl.Mem.Mat] = pl.load(a, [0, 0], [M, K], target_memory=pl.Mem.Mat)
        la2: pl.Tile[[M, K], pl.FP32, pl.Mem.Mat] = pl.load(a2, [0, 0], [M, K], target_memory=pl.Mem.Mat)
        for i in pl.range(NT):
            lb: pl.Tile[[K, TN], pl.FP32, pl.Mem.Mat] = pl.load(
                b, [0, i * TN], [K, TN], target_memory=pl.Mem.Mat
            )
            # The slot rotates with `i`, so neither address is a constant.
            cur: pl.Tile[[M, TN], pl.FP32, pl.MemRef("l0c", slots=2)[i % 2], pl.Mem.Acc] = pl.tile.matmul(
                la, lb
            )
            other: pl.Tile[
                [M, TN], pl.FP32, pl.MemRef("l0c", slots=2)[(i + 1) % 2], pl.Mem.Acc
            ] = pl.tile.matmul(la2, lb)
            # Both accumulators are still live here -- collapse the two addresses
            # and one result overwrites the other.
            pl.store(cur, [0, i * TN], out1)
            pl.store(other, [0, i * TN], out2)


@pl.jit
def vector_slot_pingpong(
    a: pl.Tensor[[ROWS, TC], pl.FP32],
    b: pl.Tensor[[ROWS, TC], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, TC], pl.FP32]],
):
    """Two UB slots co-live per iteration, summed -- `a + b`, which `b + b` cannot fake."""
    for _ in pl.spmd(1, name_hint="memref_slots_vec"):
        for i in pl.range(NT):
            lo: pl.Tile[[TR, TC], pl.FP32, pl.MemRef("ub", slots=2)[i % 2], pl.Mem.Vec] = pl.load(
                a, [i * TR, 0], [TR, TC], target_memory=pl.Mem.Vec
            )
            hi: pl.Tile[[TR, TC], pl.FP32, pl.MemRef("ub", slots=2)[(i + 1) % 2], pl.Mem.Vec] = pl.load(
                b, [i * TR, 0], [TR, TC], target_memory=pl.Mem.Vec
            )
            # If both slots resolved to one address, `lo` is gone by now.
            s: pl.Tile[[TR, TC], pl.FP32, pl.Mem.Vec] = pl.add(lo, hi)
            pl.store(s, [i * TR, 0], out)


@pytest.mark.platforms("a2a3", "a2a3sim")
class TestMemRefSlots:
    """Runtime-indexed slots of one declared allocation, on device."""

    def test_acc_slots_rotate_at_runtime(self, test_config):
        """L0C ping-pong: each slot keeps its own matmul, so the two outputs differ."""
        matmul_slot_pingpong._cache.clear()
        torch.manual_seed(0)
        a = torch.randn(M, K, dtype=torch.float32)
        a2 = torch.randn(M, K, dtype=torch.float32)
        b = torch.randn(K, N, dtype=torch.float32)
        out1 = torch.zeros((M, N), dtype=torch.float32)
        out2 = torch.zeros((M, N), dtype=torch.float32)

        matmul_slot_pingpong(a, a2, b, out1, out2, config=test_config)

        torch.testing.assert_close(out1, a @ b, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(out2, a2 @ b, rtol=1e-3, atol=1e-3)

    def test_vec_slots_rotate_at_runtime(self, test_config):
        """UB ping-pong: the sum is `a + b`, not `b + b`."""
        vector_slot_pingpong._cache.clear()
        torch.manual_seed(0)
        a = torch.randn(ROWS, TC, dtype=torch.float32)
        b = torch.randn(ROWS, TC, dtype=torch.float32)
        out = torch.zeros((ROWS, TC), dtype=torch.float32)

        vector_slot_pingpong(a, b, out, config=test_config)

        torch.testing.assert_close(out, a + b, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
