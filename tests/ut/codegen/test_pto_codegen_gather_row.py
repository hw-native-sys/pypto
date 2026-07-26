# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTO codegen tests for tile.gather_row (kernel-driven paged gather into L1).

A transposing per-row gather must place the GM row [r=1, c] as the L1 column
[c, 1]. pto.tload itself does NOT transpose, so the source must be presented as a
DN-strided view: codegen builds a ``pto.make_tensor_view ... {layout = #pto.layout<dn>}``
of the GM source (shape/strides swapped, same base ptr) and partitions the row as
a column, so the tload runs DN2NZ — the actual transpose. A straight ND2NZ tload
scrambles the fractal layout (wrong results / AICore 507018 at scale).

The non-transposing path keeps the canonical ND source view (no DN make_tensor_view).
"""

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

MM = 32
ROWS = 128
HEAD_DIM = 128
NSRC = 256


def _build_program(*, transpose: bool):
    """gather into L1 then consume as a matmul B-operand (keeps the kernel InCore)."""
    acc_shape = [HEAD_DIM, ROWS] if transpose else [ROWS, HEAD_DIM]
    a_shape = [MM, HEAD_DIM] if transpose else [MM, ROWS]
    out_shape = [MM, ROWS] if transpose else [MM, HEAD_DIM]

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[NSRC, HEAD_DIM], pl.BF16],
            a: pl.Tensor[a_shape, pl.BF16],
        ) -> pl.Tensor[out_shape, pl.FP32]:
            kv = pl.create_l1(acc_shape, pl.BF16, transpose=transpose)
            for r in pl.range(ROWS):
                if transpose:
                    kv = pl.gather_row(kv, src, [0, r], [r, 0], [1, HEAD_DIM], transpose=True)
                else:
                    kv = pl.gather_row(kv, src, [r, 0], [r, 0], [1, HEAD_DIM])
            return pl.matmul(a, kv, out_dtype=pl.FP32)

        @pl.function
        def main(
            self,
            src: pl.Tensor[[NSRC, HEAD_DIM], pl.BF16],
            a: pl.Tensor[a_shape, pl.BF16],
        ) -> pl.Tensor[out_shape, pl.FP32]:
            r = self.kernel(src, a)
            return r

    return Program


def _codegen_incore(program) -> str:
    """Run the Default pipeline + PTO codegen, returning the InCore kernel's MLIR."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    optimized = pm.run_passes(program)
    gen = codegen.PTOCodegen()
    out = []
    for func in optimized.functions.values():
        single = ir.Program([func], func.name, optimized.span)
        try:
            out.append(gen.generate(single))
        except Exception as exc:
            # Skip only the orchestration `main` (PTO targets InCore functions);
            # a genuine InCore codegen failure must surface, not be swallowed.
            if "InCore-variant" not in str(exc):
                raise
    return "\n".join(out)


def test_gather_row_transpose_emits_dn_source_view():
    """transpose=True feeds tload a DN-strided source view so it runs DN2NZ (the transpose)."""
    mlir = _codegen_incore(_build_program(transpose=True))
    assert "pto.gather_row" not in mlir  # lowered to subview + tload, not a single op
    # The transposing source view: a DN make_tensor_view of the GM source.
    assert "make_tensor_view" in mlir
    assert "layout = #pto.layout<dn>" in mlir
    # The row is read as a [c, 1] DN column partition (1x... source partition).
    assert "pto.tload" in mlir
    assert "pto.subview" in mlir


def test_gather_row_no_transpose_keeps_nd_source_view():
    """The non-transposing path uses the canonical ND source view (no DN make_tensor_view)."""
    mlir = _codegen_incore(_build_program(transpose=False))
    assert "pto.tload" in mlir
    assert "pto.subview" in mlir
    # No DN-strided source view is built for the straight ND2NZ row load.
    assert "layout = #pto.layout<dn>" not in mlir


STATIC_ROWS = ROWS // 2


def _build_dynamic_valid_program():
    """Gather the whole ROWS-row window in one call, transferring only ``n`` rows.

    ``shapes`` stays the static [ROWS, HEAD_DIM] window (it sizes pto.subview);
    only the transfer length varies, and here it is read from a GM scalar at
    runtime — the case the operand exists for.
    """

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[NSRC, HEAD_DIM], pl.BF16],
            a: pl.Tensor[[MM, ROWS], pl.BF16],
            n: pl.Tensor[[1], pl.INT32],
        ) -> pl.Tensor[[MM, HEAD_DIM], pl.FP32]:
            rows = pl.cast(pl.read(n, [0]), pl.INDEX)
            kv = pl.create_l1([ROWS, HEAD_DIM], pl.BF16)
            kv = pl.gather_row(kv, src, [0, 0], [0, 0], [ROWS, HEAD_DIM], valid_shape=[rows, HEAD_DIM])
            return pl.matmul(a, kv, out_dtype=pl.FP32)

        @pl.function
        def main(
            self,
            src: pl.Tensor[[NSRC, HEAD_DIM], pl.BF16],
            a: pl.Tensor[[MM, ROWS], pl.BF16],
            n: pl.Tensor[[1], pl.INT32],
        ) -> pl.Tensor[[MM, HEAD_DIM], pl.FP32]:
            r = self.kernel(src, a, n)
            return r

    return Program


def _build_static_valid_program():
    """Same kernel shape, but with a compile-time constant transfer extent.

    A separate builder rather than a flag on the dynamic one because the two
    kernels differ in signature — the dynamic case needs the ``n`` scalar operand
    to read the extent from, and threading an unused one through here would
    obscure that this path has no runtime input at all.
    """

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[NSRC, HEAD_DIM], pl.BF16],
            a: pl.Tensor[[MM, ROWS], pl.BF16],
        ) -> pl.Tensor[[MM, HEAD_DIM], pl.FP32]:
            kv = pl.create_l1([ROWS, HEAD_DIM], pl.BF16)
            kv = pl.gather_row(kv, src, [0, 0], [0, 0], [ROWS, HEAD_DIM], valid_shape=[STATIC_ROWS, HEAD_DIM])
            return pl.matmul(a, kv, out_dtype=pl.FP32)

        @pl.function
        def main(
            self,
            src: pl.Tensor[[NSRC, HEAD_DIM], pl.BF16],
            a: pl.Tensor[[MM, ROWS], pl.BF16],
        ) -> pl.Tensor[[MM, HEAD_DIM], pl.FP32]:
            r = self.kernel(src, a)
            return r

    return Program


def _subview_result_type(mlir: str) -> str:
    """The gather_row subview's *result* tile_buf type (the text after ``->``).

    Asserting on the whole module would be meaningless for v_row/v_col: a plain
    (non-subview) tile type always renders as ``v_row=?, v_col=?`` regardless of
    its IR valid_shape, so the parent operand type on the very same line reads
    ``?`` in both the static and dynamic cases.
    """
    lines = [ln for ln in mlir.splitlines() if "gather_row_view = pto.subview" in ln]
    assert len(lines) == 1, f"expected exactly one gather_row subview, got {len(lines)}"
    return lines[0].split("->", 1)[1].strip()


def test_gather_row_dynamic_valid_shape_emits_dynamic_subview_and_partition():
    """A runtime row count narrows the transfer without making the window dynamic.

    ptoas types pto.subview's `sizes` as a static I64ArrayAttr but `valid_row` /
    `valid_col` as Optional<Index> SSA operands, so the dynamic extent must land
    on the valid side and on the GM partition, while `sizes` keeps the static
    window (and with it the tile allocation and NZ box alignment).
    """
    mlir = _codegen_incore(_build_dynamic_valid_program())
    subview = [ln for ln in mlir.splitlines() if "gather_row_view = pto.subview" in ln][0]
    # Static window survives in `sizes`; the L1 tile allocation is unaffected.
    assert f"sizes [{ROWS}, {HEAD_DIM}]" in subview
    # The row extent is an SSA operand, not a folded constant.
    assert f"valid [%2, %c{HEAD_DIM}_index]" in subview
    # Per-dim result valid: dynamic row, static col. SubViewOp::verify's
    # expectedValidDim marks only the non-constant operand kDynamic and rejects a
    # result type that disagrees per dim, so these must NOT both be `?`.
    result_type = _subview_result_type(mlir)
    assert f"v_row=?, v_col={HEAD_DIM}" in result_type
    # GM side carries the same dynamic extent.
    assert f"!pto.partition_tensor_view<?x{HEAD_DIM}xbf16>" in mlir
    assert "pto.tload" in mlir


def test_gather_row_static_valid_shape_stays_static():
    """A constant valid_shape keeps both the subview result type and the partition static."""
    mlir = _codegen_incore(_build_static_valid_program())
    subview = [ln for ln in mlir.splitlines() if "gather_row_view = pto.subview" in ln][0]
    assert f"sizes [{ROWS}, {HEAD_DIM}]" in subview
    assert f"valid [%c{STATIC_ROWS}_index, %c{HEAD_DIM}_index]" in subview
    result_type = _subview_result_type(mlir)
    assert f"v_row={STATIC_ROWS}, v_col={HEAD_DIM}" in result_type
    assert "v_row=?" not in result_type
    assert f"!pto.partition_tensor_view<{STATIC_ROWS}x{HEAD_DIM}xbf16>" in mlir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
