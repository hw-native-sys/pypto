# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Before/Expected tests for the SkewCrossCorePipeline pass.

The pass runs immediately before LowerPipelineLoops and rewrites mixed cube/vector
``pl.pipeline`` loops whose body has both a cross-core ``tile.tpush_*`` and
``tile.tpop_*``:
  - single round-trip, PRODUCE-first (one tpush before its tpop, the tpush's
    backward slice does not feed the body) -> SKEW (producer one iteration ahead:
    produce(start) prologue + Sequential steady ``pl.range(start+step, start+trip*step)``
    whose loop var k indexes the produce and pairs produce(k)/consume(k-step) +
    consume(last) epilogue). Core-agnostic: holds for a cube ``tpush_to_aiv`` loop
    AND a vector ``tpush_to_aic`` loop.
  - CONSUME-first, multi-round-trip, or otherwise non-skewable -> demote to a plain
    Sequential loop (body unchanged).
Non-cross-core pipeline loops are left intact (for LowerPipelineLoops to unroll).
"""

import pypto.language as pl
import pytest
from pypto import ir, passes


def _skew(program: ir.Program) -> ir.Program:
    return passes.skew_cross_core_pipeline()(program)


class TestSkewCrossCorePipeline:
    """Producer-role single-round-trip loops SKEW; consumer-role / multi-round-trip
    loops DEMOTE to Sequential; non-cross-core loops are left for the unroll pass."""

    def test_single_roundtrip_producer_skews(self):
        """AIC-style (cube) produce-first body — one ``tpush_to_aiv`` then one
        ``tpop_from_aiv`` — skews: produce(0) prologue, a Sequential steady
        ``pl.range(1, 4)`` pairing produce(i)/consume(i-1), consume(3) epilogue."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                for i in pl.pipeline(0, 4, 1, stage=2):
                    qa: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [i * 16, 0], [16, 64])
                    rs: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa, qa)
                    pl.tile.tpush_to_aiv(rs, split=0)
                    e: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e, e)
                    pl.tile.store(oi, [i * 16, 0], out)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                # prologue: produce(0)
                qa0: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(
                    q, [pl.const(0, pl.INDEX) * pl.const(16, pl.INDEX), 0], [16, 64]
                )
                rs0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa0, qa0)
                pl.tile.tpush_to_aiv(rs0, split=0)
                # steady: produce(i) ; consume(i-1)
                for i in pl.range(1, 4, 1):
                    qa1: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [i * 16, 0], [16, 64])
                    rs1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa1, qa1)
                    pl.tile.tpush_to_aiv(rs1, split=0)
                    e0: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e0, e0)
                    pl.tile.store(oi0, [(i - 1) * 16, 0], out)
                # epilogue: consume(3)
                e1: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                oi1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e1, e1)
                pl.tile.store(oi1, [pl.const(3, pl.INDEX) * pl.const(16, pl.INDEX), 0], out)

        ir.assert_structural_equal(_skew(Before), Expected)

    def test_single_roundtrip_producer_skews_vector_direction(self):
        """Core-agnostic: a produce-first loop on the VECTOR side (``tpush_to_aic``
        then ``tpop_from_aic``) skews identically to the cube side. The skew keys on
        produce-first vs consume-first, not on cube-vs-vector."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                for i in pl.pipeline(0, 4, 1, stage=2):
                    qa: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [i * 16, 0], [16, 64])
                    rs: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa, qa)
                    pl.tile.tpush_to_aic(rs, split=0)  # PRODUCE first (V->C)
                    e: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aic(split=0)
                    oi: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e, e)
                    pl.tile.store(oi, [i * 16, 0], out)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                qa0: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(
                    q, [pl.const(0, pl.INDEX) * pl.const(16, pl.INDEX), 0], [16, 64]
                )
                rs0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa0, qa0)
                pl.tile.tpush_to_aic(rs0, split=0)
                for i in pl.range(1, 4, 1):
                    qa1: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [i * 16, 0], [16, 64])
                    rs1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa1, qa1)
                    pl.tile.tpush_to_aic(rs1, split=0)
                    e0: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aic(split=0)
                    oi0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e0, e0)
                    pl.tile.store(oi0, [(i - 1) * 16, 0], out)
                e1: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aic(split=0)
                oi1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e1, e1)
                pl.tile.store(oi1, [pl.const(3, pl.INDEX) * pl.const(16, pl.INDEX), 0], out)

        ir.assert_structural_equal(_skew(Before), Expected)

    def test_recomputable_scalar_carry_skews(self):
        """Producer loop whose produce half defines an ADDRESS SCALAR (``off``) that
        the consume half re-uses (K-load and V-load share the offset, like fa_fused's
        ``cache_row``). The only genuine cross-core carry is the tile through the FIFO;
        ``off`` is a pure function of the loop var, so the pass recomputes it in the
        consume half (``off`` at i for produce, ``off`` at i-1 for consume) and SKEWS
        rather than demoting."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, kv: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                for i in pl.pipeline(0, 4, 1, stage=2):
                    off: pl.Scalar[pl.INDEX] = i * 16
                    ka: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(kv, [off, 0], [16, 64])  # K-load (produce)
                    rs: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(ka, ka)
                    pl.tile.tpush_to_aiv(rs, split=0)
                    e: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    va: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(kv, [off, 0], [16, 64])  # V-load REUSES off
                    oi: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e, va)
                    pl.tile.store(oi, [off, 0], out)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, kv: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                # prologue: produce(0) — off recomputed at 0
                off0: pl.Scalar[pl.INDEX] = pl.const(0, pl.INDEX) * pl.const(16, pl.INDEX)
                ka0: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(kv, [off0, 0], [16, 64])
                rs0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(ka0, ka0)
                pl.tile.tpush_to_aiv(rs0, split=0)
                # steady: produce(i) with off=i ; consume(i-1) with off recomputed at i-1
                for i in pl.range(1, 4, 1):
                    offp: pl.Scalar[pl.INDEX] = i * 16
                    ka1: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(kv, [offp, 0], [16, 64])
                    rs1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(ka1, ka1)
                    pl.tile.tpush_to_aiv(rs1, split=0)
                    offc: pl.Scalar[pl.INDEX] = (i - 1) * 16
                    e0: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    va0: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(kv, [offc, 0], [16, 64])
                    oi0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e0, va0)
                    pl.tile.store(oi0, [offc, 0], out)
                # epilogue: consume(3) — off recomputed at 3
                off3: pl.Scalar[pl.INDEX] = pl.const(3, pl.INDEX) * pl.const(16, pl.INDEX)
                e1: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                va1: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(kv, [off3, 0], [16, 64])
                oi1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e1, va1)
                pl.tile.store(oi1, [off3, 0], out)

        ir.assert_structural_equal(_skew(Before), Expected)

    def test_consumer_multi_roundtrip_demotes_to_sequential(self):
        """AIV->AIC->AIV (consume-first, two ``tpop_from_aic``): the lead tpop feeds
        the body and there are two pops on one FIFO. Demote to a single Sequential
        loop with the body unchanged (FIFO order pop/push/pop preserved)."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                for i in pl.pipeline(0, 4, 1, stage=2):
                    s0: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aic(split=0)
                    c0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(s0, s0)
                    pl.tile.tpush_to_aic(c0, split=0)
                    c1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(c0, c0)
                    s1: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aic(split=0)
                    o0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(s1, c1)
                    pl.tile.store(o0, [i * 16, 0], out)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                for i in pl.range(0, 4, 1):
                    s0: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aic(split=0)
                    c0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(s0, s0)
                    pl.tile.tpush_to_aic(c0, split=0)
                    c1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(c0, c0)
                    s1: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aic(split=0)
                    o0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(s1, c1)
                    pl.tile.store(o0, [i * 16, 0], out)

        ir.assert_structural_equal(_skew(Before), Expected)

    def test_producer_multi_roundtrip_demotes_to_sequential(self):
        """AIC->AIV->AIC->AIV (two ``tpush_to_aiv`` on one FIFO): skewing only the
        lead push would reorder the in-order FIFO (silent wrong-data). Demote to a
        single Sequential loop with push/pop order preserved exactly."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                for i in pl.pipeline(0, 4, 1, stage=2):
                    qa: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [i * 16, 0], [16, 64])
                    p0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa, qa)
                    pl.tile.tpush_to_aiv(p0, split=0)
                    e0: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    p1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e0, e0)
                    pl.tile.tpush_to_aiv(p1, split=0)
                    e1: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    o0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e1, e1)
                    pl.tile.store(o0, [i * 16, 0], out)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                for i in pl.range(0, 4, 1):
                    qa: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [i * 16, 0], [16, 64])
                    p0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa, qa)
                    pl.tile.tpush_to_aiv(p0, split=0)
                    e0: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    p1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e0, e0)
                    pl.tile.tpush_to_aiv(p1, split=0)
                    e1: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    o0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e1, e1)
                    pl.tile.store(o0, [i * 16, 0], out)

        ir.assert_structural_equal(_skew(Before), Expected)

    def test_non_cross_core_pipeline_left_for_unroll(self):
        """A pipeline body with NO cross-core ops is left intact (still
        ``pl.pipeline(stage=2)``, not skewed) for LowerPipelineLoops to replicate."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                for i in pl.pipeline(0, 4, 1, stage=2):
                    qa: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [i * 16, 0], [16, 64])
                    oi: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa, qa)
                    pl.tile.store(oi, [i * 16, 0], out)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
                for i in pl.pipeline(0, 4, 1, stage=2):
                    qa: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [i * 16, 0], [16, 64])
                    oi: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa, qa)
                    pl.tile.store(oi, [i * 16, 0], out)

        ir.assert_structural_equal(_skew(Before), Expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
