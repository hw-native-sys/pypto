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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
