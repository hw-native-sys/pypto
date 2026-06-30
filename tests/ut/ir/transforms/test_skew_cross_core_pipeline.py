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

import re
from typing import cast

import pypto.language as pl
import pytest
from pypto import ir, passes


def _strip_membership(program: ir.Program) -> ir.Program:
    """Re-emit @p program with the transient ``pipeline_membership`` attr removed.

    SkewCrossCorePipeline stamps ``pipeline_membership`` on its produce/consume
    clones (one stage per clone) so MemoryReuse keeps the per-stage Mat-L1 load
    buffers private — see ``test_membership_tags_distinct_stage_per_clone``. That is
    a downstream-only hint, not part of the loop *shape* the before/after tests
    assert, so strip it before the structural compare. The tiles in these tests
    carry no other attrs, so dropping the whole ``attrs={...}`` dict is exact."""
    text = ir.python_print(program)
    text = re.sub(r',\s*attrs=\{"pipeline_membership":\s*"[^"]*"\}', "", text)
    return cast(ir.Program, pl.loads(text))


def _skew(program: ir.Program) -> ir.Program:
    return _strip_membership(passes.skew_cross_core_pipeline()(program))


class TestSkewCrossCorePipeline:
    """Producer-role single-round-trip loops SKEW; consumer-role / multi-round-trip
    loops DEMOTE to Sequential; non-cross-core loops are left for the unroll pass."""

    def test_single_roundtrip_producer_skews(self):
        """AIC-style (cube) produce-first body — one ``tpush_to_aiv`` then one
        ``tpop_from_aiv`` — skews at the cross-core default depth-2: a 2-produce
        prologue (produce 0,1), a Sequential steady ``pl.range(2, 4, 2)`` pairing
        produce(i),produce(i+1) with consume(i-2),consume(i-1), and a 2-consume
        epilogue (consume 2,3) — i.e. tpush,tpush then tpop,tpop per body."""

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
                # prologue: produce(0), produce(1)
                qa0: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(
                    q, [pl.const(0, pl.INDEX) * pl.const(16, pl.INDEX), 0], [16, 64]
                )
                rs0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa0, qa0)
                pl.tile.tpush_to_aiv(rs0, split=0)
                qa1: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(
                    q, [pl.const(1, pl.INDEX) * pl.const(16, pl.INDEX), 0], [16, 64]
                )
                rs1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa1, qa1)
                pl.tile.tpush_to_aiv(rs1, split=0)
                # steady: produce(i), produce(i+1) ; consume(i-2), consume(i-1)
                for i in pl.range(2, 4, 2):
                    qa2: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [i * 16, 0], [16, 64])
                    rs2: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa2, qa2)
                    pl.tile.tpush_to_aiv(rs2, split=0)
                    qa3: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [(i + 1) * 16, 0], [16, 64])
                    rs3: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa3, qa3)
                    pl.tile.tpush_to_aiv(rs3, split=0)
                    e0: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e0, e0)
                    pl.tile.store(oi0, [(i - 2) * 16, 0], out)
                    e1: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e1, e1)
                    pl.tile.store(oi1, [(i - 1) * 16, 0], out)
                # epilogue: consume(2), consume(3)
                e2: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                oi2: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e2, e2)
                pl.tile.store(oi2, [pl.const(2, pl.INDEX) * pl.const(16, pl.INDEX), 0], out)
                e3: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                oi3: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e3, e3)
                pl.tile.store(oi3, [pl.const(3, pl.INDEX) * pl.const(16, pl.INDEX), 0], out)

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
                # prologue: produce(0), produce(1)
                qa0: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(
                    q, [pl.const(0, pl.INDEX) * pl.const(16, pl.INDEX), 0], [16, 64]
                )
                rs0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa0, qa0)
                pl.tile.tpush_to_aic(rs0, split=0)
                qa1: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(
                    q, [pl.const(1, pl.INDEX) * pl.const(16, pl.INDEX), 0], [16, 64]
                )
                rs1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa1, qa1)
                pl.tile.tpush_to_aic(rs1, split=0)
                # steady: produce(i), produce(i+1) ; consume(i-2), consume(i-1)
                for i in pl.range(2, 4, 2):
                    qa2: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [i * 16, 0], [16, 64])
                    rs2: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa2, qa2)
                    pl.tile.tpush_to_aic(rs2, split=0)
                    qa3: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [(i + 1) * 16, 0], [16, 64])
                    rs3: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa3, qa3)
                    pl.tile.tpush_to_aic(rs3, split=0)
                    e0: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aic(split=0)
                    oi0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e0, e0)
                    pl.tile.store(oi0, [(i - 2) * 16, 0], out)
                    e1: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aic(split=0)
                    oi1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e1, e1)
                    pl.tile.store(oi1, [(i - 1) * 16, 0], out)
                # epilogue: consume(2), consume(3)
                e2: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aic(split=0)
                oi2: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e2, e2)
                pl.tile.store(oi2, [pl.const(2, pl.INDEX) * pl.const(16, pl.INDEX), 0], out)
                e3: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aic(split=0)
                oi3: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e3, e3)
                pl.tile.store(oi3, [pl.const(3, pl.INDEX) * pl.const(16, pl.INDEX), 0], out)

        ir.assert_structural_equal(_skew(Before), Expected)

    def test_recomputable_scalar_carry_skews(self):
        """Producer loop whose produce half defines an ADDRESS SCALAR (``off``) that
        the consume half re-uses (K-load and V-load share the offset, like fa_fused's
        ``cache_row``). The only genuine cross-core carry is the tile through the FIFO;
        ``off`` is a pure function of the loop var, so the pass recomputes it in EACH
        consume clone (at the default depth-2: produce ``off`` at i and i+1, consume
        ``off`` recomputed at i-2 and i-1) and SKEWS rather than demoting."""

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
                # prologue: produce(0), produce(1) — off recomputed at 0 and 1
                off0: pl.Scalar[pl.INDEX] = pl.const(0, pl.INDEX) * pl.const(16, pl.INDEX)
                ka0: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(kv, [off0, 0], [16, 64])
                rs0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(ka0, ka0)
                pl.tile.tpush_to_aiv(rs0, split=0)
                off1: pl.Scalar[pl.INDEX] = pl.const(1, pl.INDEX) * pl.const(16, pl.INDEX)
                ka1: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(kv, [off1, 0], [16, 64])
                rs1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(ka1, ka1)
                pl.tile.tpush_to_aiv(rs1, split=0)
                # steady: produce(i),(i+1) with off=i,i+1 ; consume(i-2),(i-1) with off recomputed
                for i in pl.range(2, 4, 2):
                    offp0: pl.Scalar[pl.INDEX] = i * 16
                    ka2: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(kv, [offp0, 0], [16, 64])
                    rs2: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(ka2, ka2)
                    pl.tile.tpush_to_aiv(rs2, split=0)
                    offp1: pl.Scalar[pl.INDEX] = (i + 1) * 16
                    ka3: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(kv, [offp1, 0], [16, 64])
                    rs3: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(ka3, ka3)
                    pl.tile.tpush_to_aiv(rs3, split=0)
                    offc0: pl.Scalar[pl.INDEX] = (i - 2) * 16
                    e0: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    va0: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(kv, [offc0, 0], [16, 64])
                    oi0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e0, va0)
                    pl.tile.store(oi0, [offc0, 0], out)
                    offc1: pl.Scalar[pl.INDEX] = (i - 1) * 16
                    e1: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    va1: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(kv, [offc1, 0], [16, 64])
                    oi1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e1, va1)
                    pl.tile.store(oi1, [offc1, 0], out)
                # epilogue: consume(2), consume(3) — off recomputed at 2 and 3
                off2: pl.Scalar[pl.INDEX] = pl.const(2, pl.INDEX) * pl.const(16, pl.INDEX)
                e2: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                va2: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(kv, [off2, 0], [16, 64])
                oi2: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e2, va2)
                pl.tile.store(oi2, [off2, 0], out)
                off3: pl.Scalar[pl.INDEX] = pl.const(3, pl.INDEX) * pl.const(16, pl.INDEX)
                e3: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                va3: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(kv, [off3, 0], [16, 64])
                oi3: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e3, va3)
                pl.tile.store(oi3, [off3, 0], out)

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

    def test_odd_trip_falls_back_to_depth1(self):
        """Default depth-2 needs ``trip % 2 == 0`` and ``trip >= 4``. A trip of 3
        (odd) is infeasible at depth-2, so the pass falls back to the largest
        feasible depth (depth-1): produce(0) prologue, steady ``pl.range(1, 3)``
        pairing produce(i)/consume(i-1), consume(2) epilogue."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[128, 64], pl.FP32], out: pl.Tensor[[128, 64], pl.FP32]):
                for i in pl.pipeline(0, 3, 1, stage=2):
                    qa: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [i * 16, 0], [16, 64])
                    rs: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa, qa)
                    pl.tile.tpush_to_aiv(rs, split=0)
                    e: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e, e)
                    pl.tile.store(oi, [i * 16, 0], out)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[128, 64], pl.FP32], out: pl.Tensor[[128, 64], pl.FP32]):
                # prologue: produce(0)
                qa0: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(
                    q, [pl.const(0, pl.INDEX) * pl.const(16, pl.INDEX), 0], [16, 64]
                )
                rs0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa0, qa0)
                pl.tile.tpush_to_aiv(rs0, split=0)
                # steady: produce(i) ; consume(i-1)
                for i in pl.range(1, 3, 1):
                    qa1: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [i * 16, 0], [16, 64])
                    rs1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa1, qa1)
                    pl.tile.tpush_to_aiv(rs1, split=0)
                    e0: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e0, e0)
                    pl.tile.store(oi0, [(i - 1) * 16, 0], out)
                # epilogue: consume(2)
                e1: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                oi1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e1, e1)
                pl.tile.store(oi1, [pl.const(2, pl.INDEX) * pl.const(16, pl.INDEX), 0], out)

        ir.assert_structural_equal(_skew(Before), Expected)

    def test_stage4_skews_depth3(self):
        """``stage=4`` requests depth-3 (producer 3 ahead). With trip=6 (``6 % 3 ==
        0``, ``6 >= 6``) the steady body emits 3 produces then 3 consumes: a 3-produce
        prologue (0,1,2), steady ``pl.range(3, 6, 3)`` pairing produce(i..i+2) with
        consume(i-3..i-1), and a 3-consume epilogue (3,4,5)."""

        @pl.program
        class Before:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[128, 64], pl.FP32], out: pl.Tensor[[128, 64], pl.FP32]):
                for i in pl.pipeline(0, 6, 1, stage=4):
                    qa: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [i * 16, 0], [16, 64])
                    rs: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa, qa)
                    pl.tile.tpush_to_aiv(rs, split=0)
                    e: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e, e)
                    pl.tile.store(oi, [i * 16, 0], out)

        @pl.program
        class Expected:
            @pl.function(strict_ssa=True)
            def main(self, q: pl.Tensor[[128, 64], pl.FP32], out: pl.Tensor[[128, 64], pl.FP32]):
                # prologue: produce(0), produce(1), produce(2)
                qa0: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(
                    q, [pl.const(0, pl.INDEX) * pl.const(16, pl.INDEX), 0], [16, 64]
                )
                rs0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa0, qa0)
                pl.tile.tpush_to_aiv(rs0, split=0)
                qa1: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(
                    q, [pl.const(1, pl.INDEX) * pl.const(16, pl.INDEX), 0], [16, 64]
                )
                rs1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa1, qa1)
                pl.tile.tpush_to_aiv(rs1, split=0)
                qa2: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(
                    q, [pl.const(2, pl.INDEX) * pl.const(16, pl.INDEX), 0], [16, 64]
                )
                rs2: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa2, qa2)
                pl.tile.tpush_to_aiv(rs2, split=0)
                # steady: produce(i),(i+1),(i+2) ; consume(i-3),(i-2),(i-1)
                for i in pl.range(3, 6, 3):
                    qa3: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [i * 16, 0], [16, 64])
                    rs3: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa3, qa3)
                    pl.tile.tpush_to_aiv(rs3, split=0)
                    qa4: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [(i + 1) * 16, 0], [16, 64])
                    rs4: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa4, qa4)
                    pl.tile.tpush_to_aiv(rs4, split=0)
                    qa5: pl.Tile[[16, 64], pl.FP32] = pl.tile.load(q, [(i + 2) * 16, 0], [16, 64])
                    rs5: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(qa5, qa5)
                    pl.tile.tpush_to_aiv(rs5, split=0)
                    e0: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi0: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e0, e0)
                    pl.tile.store(oi0, [(i - 3) * 16, 0], out)
                    e1: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi1: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e1, e1)
                    pl.tile.store(oi1, [(i - 2) * 16, 0], out)
                    e2: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                    oi2: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e2, e2)
                    pl.tile.store(oi2, [(i - 1) * 16, 0], out)
                # epilogue: consume(3), consume(4), consume(5)
                e3: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                oi3: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e3, e3)
                pl.tile.store(oi3, [pl.const(3, pl.INDEX) * pl.const(16, pl.INDEX), 0], out)
                e4: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                oi4: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e4, e4)
                pl.tile.store(oi4, [pl.const(4, pl.INDEX) * pl.const(16, pl.INDEX), 0], out)
                e5: pl.Tile[[16, 64], pl.FP32] = pl.tile.tpop_from_aiv(split=0)
                oi5: pl.Tile[[16, 64], pl.FP32] = pl.tile.add(e5, e5)
                pl.tile.store(oi5, [pl.const(5, pl.INDEX) * pl.const(16, pl.INDEX), 0], out)

        ir.assert_structural_equal(_skew(Before), Expected)

    def test_membership_tags_distinct_stage_per_clone(self):
        """The skew stamps ``pipeline_membership`` on its produce/consume clones —
        one shared group, a distinct stage per clone (depth-2 → stages 0 and 1).
        MemoryReuse reads this to keep each stage's Mat-L1 load buffer private (the
        buffer separation that fixes the fa_fused_aic over-reuse). This asserts the
        raw (un-stripped) tags; the shape tests above strip them before comparing."""

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

        text = ir.python_print(passes.skew_cross_core_pipeline()(Before))
        tags = re.findall(r'"pipeline_membership":\s*"(\d+):(\d+)"', text)
        assert tags, "skew must stamp pipeline_membership on its produce/consume clones"
        groups = {g for g, _ in tags}
        stages = {s for _, s in tags}
        assert len(groups) == 1, f"all clones of one skewed loop share a group, got {groups}"
        assert stages == {"0", "1"}, f"depth-2 → exactly stages 0 and 1, got {stages}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
