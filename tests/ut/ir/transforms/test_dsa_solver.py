# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Unit tests for the IR-free DSA solver (`passes.dsa`, RFC pypto#1980).

These exercise the FirstFitByLifetimeSolver on synthetic instances — no IR is
built. Every test independently re-validates the solver's output with
`dsa.validate`, so a placement bug fails the test even if the solver's own
status says feasible.
"""
import pytest
from pypto.pypto_core import passes

dsa = passes.dsa


def _solve(problem):
    """Solve + independently validate; return the result (asserts validity)."""
    result = dsa.solve_first_fit(problem)
    assert result.solution is not None
    errors = dsa.validate(problem, result.solution)
    assert errors == [], f"validator rejected the solver's own solution: {errors}"
    return result


def test_capabilities_are_core_only():
    caps = dsa.first_fit_capabilities()
    # v1 first-fit is the portable core: no overlay, single-hull intervals.
    assert caps.cost_model is False
    assert caps.multi_interval is False
    assert caps.colocations is True
    assert caps.separations is True
    assert caps.multi_pool is True


def test_lifetime_overlap_forces_distinct_addresses():
    # Two co-live buffers in the same pool must not overlap in address.
    p = dsa.DsaProblem()
    p.add_buffer(id=0, size=100, align=1, pool=0, start=0, end=5)
    p.add_buffer(id=1, size=100, align=1, pool=0, start=3, end=8)  # overlaps [0,5]
    res = _solve(p)
    o0, o1 = res.solution.offset_of(0), res.solution.offset_of(1)
    assert abs(o0 - o1) >= 100  # disjoint address ranges
    assert res.objective.peak == 200


def test_disjoint_lifetimes_reuse_the_same_address():
    # B's lifetime is entirely after A's -> B reuses A's freed slot.
    p = dsa.DsaProblem()
    p.add_buffer(id=0, size=100, align=1, pool=0, start=0, end=1)
    p.add_buffer(id=1, size=100, align=1, pool=0, start=2, end=3)  # disjoint from A
    res = _solve(p)
    assert res.solution.offset_of(0) == res.solution.offset_of(1) == 0
    assert res.objective.peak == 100  # not 200 — the slot is reused


def test_fragmentation_freed_region_is_subdivided_issue_1908():
    # #1908 shape: a big producer frees its region, then two smaller lifetime-
    # disjoint consumers (co-live with each other) must subdivide that freed
    # region. Buffer-granularity first-fit packs all three into the producer's
    # 64 KB — a group-bump (one max-sized slot per group, no reclaim) could not.
    kb = 1024
    p = dsa.DsaProblem()
    p.add_buffer(id=0, size=64 * kb, align=1, pool=0, start=0, end=2)  # producer
    p.add_buffer(id=1, size=32 * kb, align=1, pool=0, start=4, end=6)  # consumer 1
    p.add_buffer(id=2, size=32 * kb, align=1, pool=0, start=4, end=6)  # consumer 2
    p.set_pool_cap(pool=0, cap=64 * kb)  # only fits if the freed 64 KB is reused
    res = _solve(p)
    assert res.status == dsa.SolveStatus.kFeasible
    assert res.objective.peak == 64 * kb
    # The two consumers tile the producer's freed region, disjoint from each other.
    o1, o2 = res.solution.offset_of(1), res.solution.offset_of(2)
    assert {o1, o2} == {0, 32 * kb}


def test_colocation_shares_offset():
    # Must-alias: colocated buffers get the same offset even though co-live.
    p = dsa.DsaProblem()
    p.add_buffer(id=0, size=128, align=1, pool=0, start=0, end=5)
    p.add_buffer(id=1, size=128, align=1, pool=0, start=0, end=5)
    p.add_colocation(a=0, b=1)
    res = _solve(p)
    assert res.solution.offset_of(0) == res.solution.offset_of(1)
    assert res.objective.peak == 128  # one slot, not two


def test_separation_keeps_apart_despite_disjoint_lifetimes():
    # Separated buffers must not share address even though their lifetimes are
    # disjoint (the hazard-guard / pipeline-clone case).
    p = dsa.DsaProblem()
    p.add_buffer(id=0, size=100, align=1, pool=0, start=0, end=1)
    p.add_buffer(id=1, size=100, align=1, pool=0, start=2, end=3)  # disjoint from 0
    p.add_separation(a=0, b=1)
    res = _solve(p)
    o0, o1 = res.solution.offset_of(0), res.solution.offset_of(1)
    assert abs(o0 - o1) >= 100  # kept apart despite reuse being possible
    assert res.objective.peak == 200


def test_alignment_is_respected():
    p = dsa.DsaProblem()
    p.add_buffer(id=0, size=48, align=1, pool=0, start=0, end=5)
    p.add_buffer(id=1, size=48, align=32, pool=0, start=0, end=5)  # co-live, 32-aligned
    res = _solve(p)
    assert res.solution.offset_of(1) % 32 == 0


def test_pools_are_independent():
    # Buffers in different pools never conflict, even with identical lifetimes.
    p = dsa.DsaProblem()
    p.add_buffer(id=0, size=100, align=1, pool=0, start=0, end=5)
    p.add_buffer(id=1, size=100, align=1, pool=1, start=0, end=5)
    res = _solve(p)
    assert res.solution.offset_of(0) == 0
    assert res.solution.offset_of(1) == 0  # pool 1 starts fresh
    assert res.objective.peak_by_pool[0] == 100
    assert res.objective.peak_by_pool[1] == 100


def test_reserved_base_is_the_floor():
    p = dsa.DsaProblem()
    p.add_buffer(id=0, size=100, align=1, pool=0, start=0, end=5)
    p.set_reserved_base(pool=0, base=256)
    res = _solve(p)
    assert res.solution.offset_of(0) == 256


def test_over_cap_reports_best_effort_not_error():
    # A buffer larger than the cap still gets placed, but status flags the overflow.
    p = dsa.DsaProblem()
    p.add_buffer(id=0, size=200, align=1, pool=0, start=0, end=5)
    p.set_pool_cap(pool=0, cap=100)
    res = dsa.solve_first_fit(p)
    assert res.status == dsa.SolveStatus.kBestEffortNoFit
    assert res.solution is not None  # best-effort, not an error
    assert any("cap" in d for d in res.diagnostics)


def test_decreasing_size_order_packs_tightly():
    # Three co-live buffers of different sizes stack without gaps.
    p = dsa.DsaProblem()
    p.add_buffer(id=0, size=40, align=1, pool=0, start=0, end=9)
    p.add_buffer(id=1, size=20, align=1, pool=0, start=0, end=9)
    p.add_buffer(id=2, size=10, align=1, pool=0, start=0, end=9)
    res = _solve(p)
    # All co-live => total footprint is the exact sum, no fragmentation.
    assert res.objective.peak == 70


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
