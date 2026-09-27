# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Round-trip tests for the FIXPIPE ``pre_quant`` / ``pre_relu`` kwargs.

``pre_quant`` is the project's first ``set_attr<double>`` operator kwarg, so the
real-valued kwarg path through printer and parser has no other coverage. A scale
that survives print -> parse only approximately is worse than one that fails:
the emitted instruction still assembles, and the kernel quietly computes with a
different multiplier. These tests therefore assert the recovered value is
*exactly* equal, including for a scale with no finite binary expansion.

The FP32 narrowing that ``EncodeFixpipePreQuant`` performs happens at codegen,
not here — the IR keeps the author's ``double`` so a reprint is byte-stable.
"""

import math

import pypto.language as pl
import pytest
from pypto import ir

M, N, K = 128, 128, 64


def _reparse(prog):
    """print -> parse, asserting the reprint is a fixpoint. Returns the reparsed program."""
    text = ir.python_print(prog)
    reparsed = pl.parse_program(text)
    reprinted = ir.python_print(reparsed)
    assert reprinted == text, (
        f"round-trip is not a fixpoint\n--- first ---\n{text}\n--- second ---\n{reprinted}"
    )
    return reparsed


def _assemble_program(scale, relu):
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            k: pl.Tensor[[M, K], pl.INT8],
            q: pl.Tensor[[N, K], pl.INT8],
            out: pl.Out[pl.Tensor[[M, N], pl.FP16]],
        ) -> pl.Tensor[[M, N], pl.FP16]:
            k_mat = pl.tile.load(k, [0, 0], [M, K], target_memory=pl.Mem.Mat)
            q_mat = pl.tile.load(q, [0, 0], [N, K], target_memory=pl.Mem.Mat)
            acc = pl.tile.matmul(
                pl.tile.move(k_mat, target_memory=pl.Mem.Left),
                pl.tile.move(pl.tile.transpose_view(q_mat), target_memory=pl.Mem.Right),
            )
            s_mat = pl.tile.create([M, N], pl.FP16, target_memory=pl.Mem.Mat)
            s_mat = pl.tile.assemble(s_mat, acc, [0, 0], pre_quant=scale, pre_relu=relu)
            vec = pl.tile.move(s_mat, target_memory=pl.Mem.Vec)
            return pl.tile.store(vec, [0, 0], out)

    return Prog


def _store_program(scale, relu):
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            k: pl.Tensor[[M, K], pl.INT8],
            q: pl.Tensor[[N, K], pl.INT8],
            out: pl.Out[pl.Tensor[[M, N], pl.FP16]],
        ) -> pl.Tensor[[M, N], pl.FP16]:
            k_mat = pl.tile.load(k, [0, 0], [M, K], target_memory=pl.Mem.Mat)
            q_mat = pl.tile.load(q, [0, 0], [N, K], target_memory=pl.Mem.Mat)
            acc = pl.tile.matmul(
                pl.tile.move(k_mat, target_memory=pl.Mem.Left),
                pl.tile.move(pl.tile.transpose_view(q_mat), target_memory=pl.Mem.Right),
            )
            return pl.tile.store(acc, [0, 0], out, pre_quant=scale, pre_relu=relu)

    return Prog


def _epilogue_kwargs(prog, op_name):
    """Collect ``(pre_quant, pre_relu)`` from every ``op_name`` call in the program."""
    found: list[tuple[float | None, bool]] = []
    target = ir.get_op(op_name).name

    class _Collect(ir.IRVisitor):
        def visit_call(self, op):
            if op.op.name == target:
                kwargs = dict(op.kwargs)
                found.append((kwargs.get("pre_quant"), bool(kwargs.get("pre_relu", False))))
            super().visit_call(op)

    _Collect().visit_program(prog)
    return found


@pytest.mark.parametrize(
    "scale",
    [
        # Exactly representable: the indexer's 2**-10, whose FP32 word (0x3A800000)
        # is the constant CANN writes for the same kernel.
        1.0 / 1024,
        # No finite binary expansion, so a decimal round-trip that drops digits
        # changes the value rather than the spelling.
        0.1,
        # Negative: legal, and the one case where fusing ReLU *before* the scale
        # would differ, so the value's sign has to survive intact.
        -2.5,
    ],
    ids=["pow2", "inexact", "negative"],
)
@pytest.mark.parametrize("builder", [_assemble_program, _store_program], ids=["assemble", "store"])
def test_pre_quant_survives_print_parse_exactly(builder, scale):
    prog = builder(scale, True)
    op_name = "tile.assemble" if builder is _assemble_program else "tile.store"

    before = _epilogue_kwargs(prog, op_name)
    assert before == [(scale, True)], before

    after = _epilogue_kwargs(_reparse(prog), op_name)
    assert after == before, f"pre_quant changed across a round-trip: {before} -> {after}"
    # `==` on floats is the point here, but spell the intent out: a scale that
    # comes back "close enough" is a silently different kernel.
    assert after[0][0] is not None and math.isclose(after[0][0], scale, rel_tol=0.0, abs_tol=0.0)


def test_absent_epilogue_emits_no_kwargs():
    """``pre_relu=False`` and no scale must leave the call bare, so ordinary tile
    IR keeps printing and comparing exactly as before this feature existed.

    Written on a same-dtype Vec assemble and a plain store rather than on the
    programs above: without a scale, an INT32-into-FP16 assemble is no longer a
    legal op at all, so the epilogue-free spelling has to be exercised on IR that
    stands on its own.
    """

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[32, 16], pl.FP32]],
        ) -> pl.Tensor[[32, 16], pl.FP32]:
            src = pl.tile.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
            dst = pl.tile.create([32, 16], pl.FP32, target_memory=pl.Mem.Vec)
            dst = pl.tile.assemble(dst, src, [0, 0])
            return pl.tile.store(dst, [0, 0], out)

    assert _epilogue_kwargs(Prog, "tile.assemble") == [(None, False)]
    assert _epilogue_kwargs(Prog, "tile.store") == [(None, False)]
    printed = ir.python_print(Prog)
    assert "pre_quant" not in printed
    assert "pre_relu" not in printed
    _reparse(Prog)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
