# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ``FoldFixpipeAccEpilogue``.

The pass collapses a vector dequant/ReLU epilogue into ``pre_quant`` /
``pre_relu`` kwargs on the cube's ``Acc -> GM`` writeback. What makes that worth
a pass rather than a micro-optimization is the *structural* cost of the vector
form: it splits a pure-cube kernel into AIC+AIV halves with a C2V/V2C round-trip
and a 128 KiB pipe ring, which at these shapes does not even fit Vec space.

Every decline below leaves the IR byte-identical, asserted as
``assert_structural_equal(After, Before)`` -- an optimization that declines must
be a no-op, not a partial rewrite.
"""

import pypto.language as pl
import pytest
from pypto import backend, ir
from pypto.backend import BackendType
from pypto.pypto_core import passes as _passes

M = 128
N = 128
K = 64
SCALE = 1.0 / 1024


def _run(program, backend_type=BackendType.Ascend910B):
    """Run the pass alone, with the backend configured so the handler tables answer."""
    backend.reset_for_testing()
    backend.set_backend_type(backend_type)
    with ir.PassContext([], ir.VerificationLevel.NONE):
        return _passes.fold_fixpipe_acc_epilogue()(program)


class TestFolds:
    """Chains the fix-pipe can reproduce exactly."""

    def test_dequant_then_relu_folds_for_a_positive_scale(self):
        """The shape users write: dequantize, activate, narrow, store.

        ``maximum(acc * s, 0)`` is not the hardware's own order -- it applies the
        ReLU to the accumulator *before* the multiply -- but the two are the same
        function for every ``s >= 0``, which is what makes this foldable.
        """

        @pl.program
        class Before:
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
                f = pl.tile.cast(acc, pl.FP32, mode="rint")
                s = pl.tile.muls(f, SCALE)
                r = pl.tile.maximums(s, 0.0)
                h = pl.tile.cast(r, pl.FP16, mode="rint")
                t = pl.tile.store(h, [0, 0], out)
                return t

        @pl.program
        class Expected:
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
                t = pl.tile.store(acc, [0, 0], out, pre_quant=SCALE, pre_relu=True)
                return t

        ir.assert_structural_equal(_run(Before), Expected)

    def test_relu_then_dequant_folds_for_a_negative_scale(self):
        """``maximum(acc, 0) * s`` *is* the hardware's order, so no sign guard
        applies -- this folds where the other spelling would not."""

        @pl.program
        class Before:
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
                r = pl.tile.maximums(acc, 0.0)
                f = pl.tile.cast(r, pl.FP32, mode="rint")
                s = pl.tile.muls(f, -SCALE)
                h = pl.tile.cast(s, pl.FP16, mode="rint")
                t = pl.tile.store(h, [0, 0], out)
                return t

        @pl.program
        class Expected:
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
                t = pl.tile.store(acc, [0, 0], out, pre_quant=-SCALE, pre_relu=True)
                return t

        ir.assert_structural_equal(_run(Before), Expected)

    def test_scale_without_activation_folds(self):
        """A plain dequantization: `pre_quant` alone, no `pre_relu`."""

        @pl.program
        class Before:
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
                f = pl.tile.cast(acc, pl.FP32, mode="rint")
                s = pl.tile.muls(f, SCALE)
                h = pl.tile.cast(s, pl.FP16, mode="rint")
                t = pl.tile.store(h, [0, 0], out)
                return t

        @pl.program
        class Expected:
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
                t = pl.tile.store(acc, [0, 0], out, pre_quant=SCALE)
                return t

        ir.assert_structural_equal(_run(Before), Expected)


class TestDeclines:
    """Shapes the fix-pipe cannot reproduce. Each must leave the IR untouched."""

    def test_negative_scale_after_the_relu_is_declined(self):
        """The discriminating case.

        ``maximum(acc * s, 0)`` with ``s < 0`` keeps the originally-negative
        entries; the hardware's ``maximum(acc, 0) * s`` keeps the positive ones
        and negates them. They disagree at every element, so folding here would
        silently change the kernel -- exactly the defect the a2a3 st caught in
        the hand-written form of this feature.
        """

        @pl.program
        class Before:
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
                f = pl.tile.cast(acc, pl.FP32, mode="rint")
                s = pl.tile.muls(f, -SCALE)
                r = pl.tile.maximums(s, 0.0)
                h = pl.tile.cast(r, pl.FP16, mode="rint")
                t = pl.tile.store(h, [0, 0], out)
                return t

        ir.assert_structural_equal(_run(Before), Before)

    def test_frontend_default_round_mode_is_declined(self):
        """``mode="round"`` breaks ties away from zero; the writeback breaks them
        to even. Folding would change results at ties, so the vector ``pto.tcvt``
        stays -- the same rule the unscaled ``CastFoldableToFixpipeMat`` applies."""

        @pl.program
        class Before:
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
                f = pl.tile.cast(acc, pl.FP32, mode="rint")
                s = pl.tile.muls(f, SCALE)
                h = pl.tile.cast(s, pl.FP16)  # frontend default: "round"
                t = pl.tile.store(h, [0, 0], out)
                return t

        ir.assert_structural_equal(_run(Before), Before)

    def test_a_second_reader_of_an_intermediate_is_declined(self):
        """Folding deletes the chain, so a value someone else still reads must
        keep its producer."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                k: pl.Tensor[[M, K], pl.INT8],
                q: pl.Tensor[[N, K], pl.INT8],
                out: pl.Out[pl.Tensor[[M, N], pl.FP16]],
                aux: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP16]:
                k_mat = pl.tile.load(k, [0, 0], [M, K], target_memory=pl.Mem.Mat)
                q_mat = pl.tile.load(q, [0, 0], [N, K], target_memory=pl.Mem.Mat)
                acc = pl.tile.matmul(
                    pl.tile.move(k_mat, target_memory=pl.Mem.Left),
                    pl.tile.move(pl.tile.transpose_view(q_mat), target_memory=pl.Mem.Right),
                )
                f = pl.tile.cast(acc, pl.FP32, mode="rint")
                s = pl.tile.muls(f, SCALE)
                h = pl.tile.cast(s, pl.FP16, mode="rint")
                # `s` is read a second time, so it cannot be folded away.
                pl.tile.store(s, [0, 0], aux)
                t = pl.tile.store(h, [0, 0], out)
                return t

        ir.assert_structural_equal(_run(Before), Before)

    def test_unsupported_dtype_pair_is_declined(self):
        """A2/A3 has no scale-bearing ``f32 -> f16`` writeback, so an FP32
        accumulator scaled into an FP16 tensor must keep the vector multiply."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[M, K], pl.FP16],
                b: pl.Tensor[[K, N], pl.FP16],
                out: pl.Out[pl.Tensor[[M, N], pl.FP16]],
            ) -> pl.Tensor[[M, N], pl.FP16]:
                a_mat = pl.tile.load(a, [0, 0], [M, K], target_memory=pl.Mem.Mat)
                b_mat = pl.tile.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat)
                acc = pl.tile.matmul(
                    pl.tile.move(a_mat, target_memory=pl.Mem.Left),
                    pl.tile.move(b_mat, target_memory=pl.Mem.Right),
                )
                s = pl.tile.muls(acc, SCALE)
                h = pl.tile.cast(s, pl.FP16, mode="rint")
                t = pl.tile.store(h, [0, 0], out)
                return t

        ir.assert_structural_equal(_run(Before), Before)

    def test_an_epilogue_already_on_the_store_is_left_alone(self):
        """A hand-written epilogue would have to be *composed* with the folded
        one; that is not this pass's job, so it declines rather than guess."""

        @pl.program
        class Before:
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
                r = pl.tile.maximums(acc, 0.0)
                h = pl.tile.cast(r, pl.FP16, mode="rint")
                t = pl.tile.store(h, [0, 0], out, pre_quant=SCALE)
                return t

        ir.assert_structural_equal(_run(Before), Before)

    def test_a_bare_matmul_store_is_untouched(self):
        """No epilogue at all: nothing to fold, and the pass must not invent one."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                k: pl.Tensor[[M, K], pl.INT8],
                q: pl.Tensor[[N, K], pl.INT8],
                out: pl.Out[pl.Tensor[[M, N], pl.INT32]],
            ) -> pl.Tensor[[M, N], pl.INT32]:
                k_mat = pl.tile.load(k, [0, 0], [M, K], target_memory=pl.Mem.Mat)
                q_mat = pl.tile.load(q, [0, 0], [N, K], target_memory=pl.Mem.Mat)
                acc = pl.tile.matmul(
                    pl.tile.move(k_mat, target_memory=pl.Mem.Left),
                    pl.tile.move(pl.tile.transpose_view(q_mat), target_memory=pl.Mem.Right),
                )
                t = pl.tile.store(acc, [0, 0], out)
                return t

        ir.assert_structural_equal(_run(Before), Before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
