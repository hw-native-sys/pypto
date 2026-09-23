# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for the FIXPIPE epilogue (``pre_quant`` / ``pre_relu``) dtype contract.

The cube's fix-pipe can multiply an accumulator by a scalar while it drains L0C,
which is what makes a *scale-bearing* conversion reachable without vector work:
``INT32 -> FP16`` (dequantization), ``INT32 -> INT8`` (requantization),
``FP32 -> INT8`` (quantization). Which pairs exist is per-backend and lives in
``BackendHandler::SupportsFixpipePreQuant``; ``FixpipeEpilogueValid`` enforces it
for the Acc->Mat writeback (``tile.assemble``) and ``AccToGmStoreValid`` for the
Acc->GM one (``tile.store``).

**Why these are errors and not perf hints.** There is no reliable backstop
underneath. ptoas verifies the quantized dtype pair on a2a3 only; on a5 it
accepts pairs pto-isa has no mode for, and pto-isa answers those from
``GetScalarPreQuantMode`` with ``QuantMode_t::NoQuant`` — which *drops the scale*
rather than failing. A pair these verifiers let through would compile, run, and
return unscaled numbers.

The A2/A3 table asserted below was measured against ptoas v0.61 by assembling
each pair; PTOAS 0.65 adds the typed ``TINSERT`` operands that make the
``INT32 -> FP16`` Mat form unambiguous on both A2/A3 and A5.
"""

import pypto
import pypto.language as pl
import pytest
from pypto import backend
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

M, N, K, H = 128, 128, 64, 32
SCALE = 1.0 / 1024


@pytest.fixture(autouse=True)
def _reset_backend():
    yield
    backend.reset_for_testing()


def _run(prog, backend_type=BackendType.Ascend910B):
    backend.reset_for_testing()
    backend.set_backend_type(backend_type)
    return PassManager.get_strategy(OptimizationStrategy.Default).run_passes(prog)


def _int_matmul_to_mat(mat_dtype, *, pre_quant: float | None = SCALE, pre_relu=False):
    """INT8 x INT8 -> INT32 Acc, drained into a ``mat_dtype`` Mat scratch that a
    second matmul then reads — the lightning-indexer shape from issue #2765."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            k: pl.Tensor[[M, K], pl.INT8],
            q: pl.Tensor[[N, K], pl.INT8],
            w: pl.Tensor[[N, H], mat_dtype],
            out: pl.Out[pl.Tensor[[M, H], pl.FP32]],
        ) -> pl.Tensor[[M, H], pl.FP32]:
            k_mat = pl.tile.load(k, [0, 0], [M, K], target_memory=pl.Mem.Mat)
            q_mat = pl.tile.load(q, [0, 0], [N, K], target_memory=pl.Mem.Mat)
            acc = pl.tile.matmul(
                pl.tile.move(k_mat, target_memory=pl.Mem.Left),
                pl.tile.move(pl.tile.transpose_view(q_mat), target_memory=pl.Mem.Right),
            )
            s_mat = pl.tile.create([M, N], mat_dtype, target_memory=pl.Mem.Mat)
            s_mat = pl.tile.assemble(s_mat, acc, [0, 0], pre_quant=pre_quant, pre_relu=pre_relu)
            w_mat = pl.tile.load(w, [0, 0], [N, H], target_memory=pl.Mem.Mat)
            y = pl.tile.matmul(
                pl.tile.move(s_mat, target_memory=pl.Mem.Left),
                pl.tile.move(w_mat, target_memory=pl.Mem.Right),
            )
            return pl.tile.store(y, [0, 0], out)

    return Prog


def _float_matmul_to_mat(mat_dtype, *, pre_quant: float | None = SCALE):
    """FP16 x FP16 -> FP32 Acc drained into a ``mat_dtype`` Mat scratch."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[M, K], pl.FP16],
            b: pl.Tensor[[K, N], pl.FP16],
            w: pl.Tensor[[N, H], mat_dtype],
            out: pl.Out[pl.Tensor[[M, H], pl.FP32]],
        ) -> pl.Tensor[[M, H], pl.FP32]:
            a_mat = pl.tile.load(a, [0, 0], [M, K], target_memory=pl.Mem.Mat)
            b_mat = pl.tile.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat)
            acc = pl.tile.matmul(
                pl.tile.move(a_mat, target_memory=pl.Mem.Left),
                pl.tile.move(b_mat, target_memory=pl.Mem.Right),
            )
            s_mat = pl.tile.create([M, N], mat_dtype, target_memory=pl.Mem.Mat)
            s_mat = pl.tile.assemble(s_mat, acc, [0, 0], pre_quant=pre_quant)
            w_mat = pl.tile.load(w, [0, 0], [N, H], target_memory=pl.Mem.Mat)
            y = pl.tile.matmul(
                pl.tile.move(s_mat, target_memory=pl.Mem.Left),
                pl.tile.move(w_mat, target_memory=pl.Mem.Right),
            )
            return pl.tile.store(y, [0, 0], out)

    return Prog


def _int_matmul_to_gm(out_dtype, *, pre_quant: float | None = SCALE, pre_relu=False):
    """INT8 x INT8 -> INT32 Acc stored straight to an ``out_dtype`` tensor."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            k: pl.Tensor[[M, K], pl.INT8],
            q: pl.Tensor[[N, K], pl.INT8],
            out: pl.Out[pl.Tensor[[M, N], out_dtype]],
        ) -> pl.Tensor[[M, N], out_dtype]:
            k_mat = pl.tile.load(k, [0, 0], [M, K], target_memory=pl.Mem.Mat)
            q_mat = pl.tile.load(q, [0, 0], [N, K], target_memory=pl.Mem.Mat)
            acc = pl.tile.matmul(
                pl.tile.move(k_mat, target_memory=pl.Mem.Left),
                pl.tile.move(pl.tile.transpose_view(q_mat), target_memory=pl.Mem.Right),
            )
            return pl.tile.store(acc, [0, 0], out, pre_quant=pre_quant, pre_relu=pre_relu)

    return Prog


class TestAccToMatPreQuant:
    """``tile.assemble`` — the Acc->Mat (``pto.tinsert``) writeback."""

    @pytest.mark.parametrize("backend_type", [BackendType.Ascend910B, BackendType.Ascend950])
    def test_int32_to_fp16_scale_is_supported(self, backend_type):
        """PTOAS 0.65 emits the typed scaled ``TINSERT`` on both backends."""
        _run(_int_matmul_to_mat(pl.FP16, pre_relu=True), backend_type)

    @pytest.mark.parametrize("backend_type", [BackendType.Ascend910B, BackendType.Ascend950])
    @pytest.mark.parametrize("mat_dtype", [pl.BF16, pl.INT8, pl.INT16])
    def test_unsupported_int32_targets_are_rejected(self, mat_dtype, backend_type):
        with pytest.raises(pypto.Error, match="FixpipeEpilogueValid") as excinfo:
            _run(_int_matmul_to_mat(mat_dtype), backend_type)
        message = str(excinfo.value)
        assert "cannot be assembled" in message, message
        assert "Supported Mat targets from a 'int32' accumulator here are fp16" in message, message

    @pytest.mark.parametrize("backend_type", [BackendType.Ascend910B, BackendType.Ascend950])
    def test_unsupported_fp32_source_is_rejected(self, backend_type):
        with pytest.raises(pypto.Error, match="FixpipeEpilogueValid") as excinfo:
            _run(_float_matmul_to_mat(pl.FP16), backend_type)
        message = str(excinfo.value)
        assert "cannot be assembled" in message, message
        assert "Supported Mat targets from a 'fp32' accumulator here are none" in message, message

    def test_pre_relu_alone_needs_a_converting_writeback(self):
        """Acc -> Mat has two lowerings and only one is the fix-pipe.

        A same-dtype assemble is an MTE1 ``pto.subview`` + ``pto.tmov`` with no
        fix-pipe in the path, so it can carry no activation — unlike the
        Acc -> GM store, where every write is a fix-pipe drain and ``pre_relu``
        alone is fine. Until the emitter and this verifier shared one predicate
        (``CubeMatWritebackUsesFixpipe``) they disagreed here, and this program
        passed verification only to trip the emitter's own internal check.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[M, K], pl.FP16],
                b: pl.Tensor[[K, N], pl.FP16],
                out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
            ) -> pl.Tensor[[M, N], pl.FP32]:
                a_mat = pl.tile.load(a, [0, 0], [M, K], target_memory=pl.Mem.Mat)
                b_mat = pl.tile.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat)
                acc = pl.tile.matmul(
                    pl.tile.move(a_mat, target_memory=pl.Mem.Left),
                    pl.tile.move(b_mat, target_memory=pl.Mem.Right),
                )
                # Same dtype as the accumulator: no conversion, so no fix-pipe.
                s_mat = pl.tile.create([M, N], pl.FP32, target_memory=pl.Mem.Mat)
                s_mat = pl.tile.assemble(s_mat, acc, [0, 0], pre_relu=True)
                vec = pl.tile.move(s_mat, target_memory=pl.Mem.Vec)
                return pl.tile.store(vec, [0, 0], out)

        with pytest.raises(pypto.Error, match="FixpipeEpilogueValid") as excinfo:
            _run(Prog)
        message = str(excinfo.value)
        assert "is a plain on-chip move, not a fix-pipe writeback" in message, message
        assert "assemble into an FP16 or BF16 Mat tile" in message, message

    def test_epilogue_on_a_non_acc_to_mat_move_is_rejected(self):
        """The epilogue is part of the cube writeback; a Vec-to-Vec assemble has
        no fix-pipe to configure, so asking for one is a mistake rather than a
        no-op that quietly drops the scale."""

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
                dst = pl.tile.assemble(dst, src, [0, 0], pre_relu=True)
                return pl.tile.store(dst, [0, 0], out)

        with pytest.raises(pypto.Error, match="FixpipeEpilogueValid") as excinfo:
            _run(Prog)
        assert "has no meaning on any other move" in str(excinfo.value)


class TestAccToGmPreQuant:
    """``tile.store`` — the Acc->GM (``pto.tstore``) writeback."""

    def test_int32_to_fp16_dequant_with_relu_is_accepted(self):
        _run(_int_matmul_to_gm(pl.FP16, pre_relu=True))

    def test_int32_to_int8_requant_is_accepted_though_int8_is_not_a_plain_destination(self):
        """INT8 is *not* in ``SupportsAccToGmDtype`` — a plain Acc->GM store
        cannot reach it. A requantizing one can, which is why the pre_quant path
        replaces that whitelist check instead of being layered on top of it."""
        _run(_int_matmul_to_gm(pl.INT8))

    def test_fp32_accumulator_to_fp16_is_rejected_on_a2a3(self):
        """The A2/A3 quantized tstore reaches only i8/ui8 from an FP32
        accumulator, so this asks for a mode that does not exist."""

        @pl.program
        class Prog:
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
                return pl.tile.store(acc, [0, 0], out, pre_quant=SCALE)

        with pytest.raises(pypto.Error, match="AccToGmStoreValid") as excinfo:
            _run(Prog)
        message = str(excinfo.value)
        assert "no scale-bearing fp32 -> fp16 mode" in message, message
        assert "Supported destinations from a 'fp32' accumulator here are int8/uint8" in message, message

    @pytest.mark.parametrize("kwarg", ["pre_quant", "pre_relu"])
    def test_epilogue_on_a_vec_source_store_is_rejected(self, kwarg):
        """A store from the vector unit is an MTE3 DMA with no fix-pipe in the
        path. ptoas does reject ``reluPreMode`` there, but only against a
        generated ``.pto`` line — and a ``pre_quant`` on a Vec source it accepts
        and then ignores, which is the silent case this guard exists for."""
        scale = SCALE if kwarg == "pre_quant" else None
        relu = kwarg == "pre_relu"

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t = pl.tile.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Vec)
                return pl.tile.store(t, [0, 0], out, pre_quant=scale, pre_relu=relu)

        with pytest.raises(pypto.Error, match="AccToGmStoreValid") as excinfo:
            _run(Prog)
        assert "source tile is Vec-resident, not Acc" in str(excinfo.value)

    def test_plain_store_still_rejects_an_unscaled_scaled_conversion(self):
        """Regression guard: adding the pre_quant branch must not weaken the
        existing unscaled rule. Without a scale, ``i32 -> f16`` is still the
        dequantization the fix-pipe cannot do."""
        with pytest.raises(pypto.Error, match="AccToGmStoreValid") as excinfo:
            _run(_int_matmul_to_gm(pl.FP16, pre_quant=None))
        assert "needs a scale this store cannot carry" in str(excinfo.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
