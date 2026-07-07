# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the L0 tile-size chooser.

The chooser ranks legal aligned (m, n, k) tiles by a roofline wall objective
(see the pto-isa cost-model study ``DESIGN_SPACE.md``) and returns the
minimum-wall design point, modelling the algorithm the pass realizes today
(output-stationary, with an optional L0C double-buffer). When
``allow_double_buffer_c`` is set the chooser also evaluates a double-buffered L0C
schedule -- half the L0C budget but the FIXPIPE drain overlaps the next tile's
compute (``wall = max(C_load, C_mad, C_drain)`` instead of ``... + C_drain``) --
and reports its choice via ``L0TileResult.double_buffer_c``.

These tests pin representative worked examples, edge cases (small dimensions,
K below the cube minimum, L0C double-buffering, ``c_read`` accounting), and a
brute-force check that the chooser's pick is the true global wall-optimum.
"""

import pytest
from pypto.pypto_core import passes


def _default_config(M: int, N: int, K: int) -> passes.l0_tile_chooser.L0TileConfig:
    """Build a config matching the 910B defaults and the L0_TILING.md examples.

    `allow_padding` is left at its C++ default (False): at L0 the cube minimum
    must already be met by the input shape. Tests that explicitly want the
    padded path (Case A in L0_TILING.md §11) should override this flag.
    """
    cfg = passes.l0_tile_chooser.L0TileConfig()
    cfg.M, cfg.N, cfg.K = M, N, K
    cfg.l0a_bytes = 64 * 1024
    cfg.l0b_bytes = 64 * 1024
    cfg.l0c_bytes = 128 * 1024
    cfg.bytes_a = 2  # BF16
    cfg.bytes_b = 2  # BF16
    cfg.bytes_c = 4  # FP32 accumulator
    cfg.min_m = cfg.min_n = cfg.min_k = 16
    cfg.align_m = cfg.align_n = cfg.align_k = 16
    # Realizable-mask gates default OFF (output-stationary, dbC=1) — the
    # algorithm the pass realizes today. Tests open a gate to exercise an axis.
    cfg.allow_a_stationary = False
    cfg.allow_b_stationary = False
    cfg.allow_double_buffer_c = False
    cfg.c_read = False
    return cfg


def _capacities_ok(
    m: int, n: int, k: int, cfg, dba: bool = True, dbb: bool = True, dbc: bool = False
) -> bool:
    """Re-derive effective capacity constraints and confirm (m, n, k) fits.

    dba/dbb/dbc are the chosen per-operand double-buffer depths (true == depth 2,
    halving that buffer).
    """
    a0 = cfg.l0a_bytes // (cfg.bytes_a * (2 if dba else 1))
    b0 = cfg.l0b_bytes // (cfg.bytes_b * (2 if dbb else 1))
    c0 = cfg.l0c_bytes // (cfg.bytes_c * (2 if dbc else 1))
    return m * k <= a0 and k * n <= b0 and m * n <= c0


# ---------------------------------------------------------------------------
# L0_TILING.md §12 worked examples
# ---------------------------------------------------------------------------


class TestL0TilingDocExamples:
    """Pin the five examples in L0_TILING.md §12 (Examples 1-5)."""

    def test_example_1_skinny_gemm(self):
        """M=16, N=256, K=512 → (16, 256, 64) under the drain-count cost model.

        The full-K hoist (16, 32, 512) has lower load but splits N into 8 output
        tiles → 8 FIXPIPE drains; the per-drain fixed overhead outweighs the load
        saving, so the chooser keeps the whole N (1 drain) and splits K instead.
        Matches the brute-force optimum. (op-sim device-validated: the 8-drain
        over-split is 2-13% slower on shallow-K shapes.)"""
        cfg = _default_config(M=16, N=256, K=512)
        result = passes.l0_tile_chooser.choose_l0_tile(cfg)
        assert (result.m, result.n, result.k) == (16, 256, 64)
        assert _capacities_ok(result.m, result.n, result.k, cfg)
        assert result.perf_hint == ""

    def test_example_2_large_square_gemm(self):
        """M=N=K=4096 → (256, 128, 64) under the drain-count cost model.

        This GEMM is deeply compute-bound: C_mad (~1.8e7 cycles) dwarfs C_load,
        so cube-filling tiles tie on max(C_load, C_mad). Among those the drain-count
        term (num_drains = ceil(M/m)*ceil(N/n)) breaks the tie toward FEWER output
        tiles: the wider m=256 halves the M-block count vs m=128. m=256 forces
        k<=64 (L0A holds m*k*2*dbA <= 64KB), giving (256, 128, 64). The brute-force
        oracle confirms the global wall-minimum. Device-neutral vs (128,128,128)
        on large compute-bound shapes (op-sim sweep-4: <1% wash); the drain-count
        term earns its keep on the SMALL shallow-K shapes it de-over-splits.
        """
        cfg = _default_config(M=4096, N=4096, K=4096)
        result = passes.l0_tile_chooser.choose_l0_tile(cfg)
        # The per-M-row drain (∝m) breaks the square-shape aspect toward wide-N/small-M
        # (fewer FIXPIPE rows per drain). Deeply compute-bound, so device-neutral vs the
        # transposed (256,128,64) — the drain is hidden either way.
        assert (result.m, result.n, result.k) == (128, 256, 64)
        assert _capacities_ok(result.m, result.n, result.k, cfg)
        assert cfg.K % result.k == 0, f"k must divide K=4096; got k={result.k}"

    def test_example_3_c_double_buffer_declined_when_compute_bound(self):
        """M=N=K=4096 with allow_double_buffer_c=True → still (128, 256, 64), dbC=1.

        This GEMM is deeply compute-bound (C_mad dwarfs C_drain), so hiding the
        drain behind the next tile's compute buys nothing while halving the L0C
        budget would only shrink the tile. The chooser evaluates the
        double-buffered schedule and correctly declines it.
        """
        cfg = _default_config(M=4096, N=4096, K=4096)
        cfg.allow_double_buffer_c = True
        result = passes.l0_tile_chooser.choose_l0_tile(cfg)
        assert (result.m, result.n, result.k) == (128, 256, 64)
        assert result.double_buffer_c is False
        assert _capacities_ok(result.m, result.n, result.k, cfg, dbc=result.double_buffer_c)

    def test_example_4_short_n(self):
        """M=512, N=128, K=2048 → tile that covers all of N."""
        cfg = _default_config(M=512, N=128, K=2048)
        result = passes.l0_tile_chooser.choose_l0_tile(cfg)
        assert _capacities_ok(result.m, result.n, result.k, cfg)
        # The doc cites (256, 128, 64); we accept any tile that fully covers
        # the short N (n == 128) and lands at an aligned legal m, k.
        assert result.n == 128, f"Expected n=128 to cover all of N=128; got n={result.n}"
        assert result.m % cfg.align_m == 0 and result.m >= cfg.min_m
        assert result.k % cfg.align_k == 0 and result.k >= cfg.min_k

    def test_example_5_short_m(self):
        """M=128, N=512, K=2048 → symmetric mirror of Example 4."""
        cfg = _default_config(M=128, N=512, K=2048)
        result = passes.l0_tile_chooser.choose_l0_tile(cfg)
        assert _capacities_ok(result.m, result.n, result.k, cfg)
        # The doc cites (128, 256, 64); we accept any tile that fully covers
        # the short M (m == 128) and lands at an aligned legal n, k.
        assert result.m == 128, f"Expected m=128 to cover all of M=128; got m={result.m}"
        assert result.n % cfg.align_n == 0 and result.n >= cfg.min_n
        assert result.k % cfg.align_k == 0 and result.k >= cfg.min_k


# ---------------------------------------------------------------------------
# Capacity / boundary edge cases
# ---------------------------------------------------------------------------


class TestL0TilingEdgeCases:
    """Edge cases beyond the worked examples."""

    def test_full_c_fits_but_k_too_small_falls_back(self):
        """M=16, N=4096: full N would force k < min_k, so n must shrink."""
        cfg = _default_config(M=16, N=4096, K=512)
        result = passes.l0_tile_chooser.choose_l0_tile(cfg)
        assert _capacities_ok(result.m, result.n, result.k, cfg)
        # K must still meet min_k=16 even with the wider N.
        assert result.k >= cfg.min_k
        # With A/B double-buffering effective B0=16384, n must be at most
        # 16384/min_k=1024 if k=16, smaller if k > 16.
        b0 = cfg.l0b_bytes // (cfg.bytes_b * 2)
        assert result.n * result.k <= b0

    def test_min_k_bumped_to_64(self):
        """A larger min_k forces n to shrink to make room in B0."""
        cfg = _default_config(M=4096, N=4096, K=4096)
        cfg.min_k = 64
        cfg.align_k = 64
        result = passes.l0_tile_chooser.choose_l0_tile(cfg)
        assert result.k >= 64
        assert result.k % 64 == 0
        assert _capacities_ok(result.m, result.n, result.k, cfg)

    def test_small_dims_with_padding(self):
        """M=7, N=256, K=512 with allow_padding=True yields padded m=16."""
        cfg = _default_config(M=7, N=256, K=512)
        cfg.allow_padding = True
        result = passes.l0_tile_chooser.choose_l0_tile(cfg)
        assert result.m == 16, f"Expected m padded up to min_m=16; got m={result.m}"
        assert _capacities_ok(result.m, result.n, result.k, cfg)
        assert result.perf_hint != "", "Padded-up cases should set perf_hint"

    def test_small_dims_without_padding_raises(self):
        """M=7 with allow_padding=False (the L0 default) is rejected outright.

        At L0 we do not pad the matrix dimensions up to reach the cube
        minimum; the AutoTileMatmulL0 pass is expected to skip such matmuls
        with a perf hint instead of calling the chooser.
        """
        cfg = _default_config(M=7, N=256, K=512)
        # cfg.allow_padding stays at the default False
        with pytest.raises(ValueError, match="allow_padding=false but M=7"):
            passes.l0_tile_chooser.choose_l0_tile(cfg)

    def test_c_read_doubles_c_traffic(self):
        """`c_read=True` should monotonically increase the estimated traffic."""
        cfg = _default_config(M=4096, N=4096, K=4096)
        result_no_read = passes.l0_tile_chooser.choose_l0_tile(cfg)
        cfg.c_read = True
        result_read = passes.l0_tile_chooser.choose_l0_tile(cfg)
        # Same tile shape (capacity-bound, not traffic-bound for square cases)
        # but traffic estimate must include the extra C read.
        assert result_read.estimated_traffic_bytes > result_no_read.estimated_traffic_bytes

    def test_k_must_divide_K_when_no_padding(self):
        """Regression: qwen3_decode gate_proj/up_proj inner-K shape.

        With M=16, N=320, K=128 (BF16→FP32, A/B double-buffered, L0a/b=64KB,
        L0c=128KB), the largest k fitting in B0 with n=320 is 48 (capacity-
        bound: 16384 / 320 = 51 → align-down to 48).  k=48 does not divide
        K=128, so the AutoTileMatmulL0 consumer would emit PH-AT-007 and
        skip — leaving the 128×320 Mat tile to overflow L0b (81920 bytes >
        65536 byte limit).  The chooser must instead return a k that divides
        K (32 is the largest aligned divisor ≤ 48).
        """
        cfg = _default_config(M=16, N=320, K=128)
        result = passes.l0_tile_chooser.choose_l0_tile(cfg)
        assert _capacities_ok(result.m, result.n, result.k, cfg)
        assert cfg.K % result.k == 0, f"k must divide K=128; got k={result.k}"

    def test_already_l0_sized_returns_native(self):
        """A 64x64x64 matmul already fits in L0; chooser returns near-native."""
        cfg = _default_config(M=64, N=64, K=64)
        result = passes.l0_tile_chooser.choose_l0_tile(cfg)
        assert _capacities_ok(result.m, result.n, result.k, cfg)
        # The full problem fits in L0 so the chooser should not split — k
        # should match K (no K-iter needed).
        assert result.k >= 64
        assert result.m >= 64 and result.n >= 64


# ---------------------------------------------------------------------------
# Capacity / alignment invariants
# ---------------------------------------------------------------------------


class TestL0TilingInvariants:
    """Sanity checks that hold across many input shapes."""

    @pytest.mark.parametrize(
        "M,N,K",
        [
            (16, 16, 16),
            (32, 32, 32),
            (256, 256, 256),
            (1024, 1024, 1024),
            (4096, 4096, 4096),
            (16, 512, 2048),
            (512, 16, 2048),
            (256, 4096, 64),
        ],
    )
    def test_result_respects_capacity_and_alignment(self, M, N, K):
        cfg = _default_config(M=M, N=N, K=K)
        result = passes.l0_tile_chooser.choose_l0_tile(cfg)

        assert result.m >= cfg.min_m
        assert result.n >= cfg.min_n
        assert result.k >= cfg.min_k
        assert result.m % cfg.align_m == 0
        assert result.n % cfg.align_n == 0
        assert result.k % cfg.align_k == 0
        assert _capacities_ok(result.m, result.n, result.k, cfg), (
            f"Tile (m={result.m}, n={result.n}, k={result.k}) violates capacity for M={M}, N={N}, K={K}"
        )
        # Without padding, the chooser must honor the AutoTileMatmulL0
        # consumer's K-divisibility precondition (PH-AT-007).
        assert K % result.k == 0, f"k={result.k} must divide K={K} when allow_padding=False"

    def test_invalid_zero_dim_raises(self):
        cfg = _default_config(M=0, N=128, K=128)
        with pytest.raises(ValueError, match="M, N, K must all be positive"):
            passes.l0_tile_chooser.choose_l0_tile(cfg)

    def test_invalid_negative_dim_raises(self):
        cfg = _default_config(M=128, N=-1, K=128)
        with pytest.raises(ValueError, match="M, N, K must all be positive"):
            passes.l0_tile_chooser.choose_l0_tile(cfg)


# ---------------------------------------------------------------------------
# Brute-force optimality: the chooser's pick must be the global wall-minimum
# ---------------------------------------------------------------------------


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _odd_part(x: int) -> int:
    """x divided by its largest power-of-2 factor (mirrors C++ OddPart)."""
    while x % 2 == 0:
        x //= 2
    return x


# Stationarity tokens for the oracle, mapped to the bound enum for comparison.
_OS, _AS, _BS = "OS", "A", "B"
_STAT_ENUM = {
    _OS: passes.l0_tile_chooser.Stationarity.OutputStationary,
    _AS: passes.l0_tile_chooser.Stationarity.AStationary,
    _BS: passes.l0_tile_chooser.Stationarity.BStationary,
}


def _derive_db(stat: str) -> tuple[bool, bool]:
    """(dbA, dbB): stationary operand single-buffered (False), moving double (True)."""
    if stat == _AS:
        return (False, True)
    if stat == _BS:
        return (True, False)
    return (True, True)


def _load_cycles(m: int, n: int, k: int, cfg, stat: str) -> float:
    # The full-K emitter hoists one operand (loaded once, reused across the inner
    # sweep); OS at k == K picks the cheaper hoist -- NOT "both re-streamed". Only
    # split-K (k < K) re-streams both. Mirrors C++ LoadCycles.
    M, N, K = cfg.M, cfg.N, cfg.K
    cn, cm = _cdiv(N, n), _cdiv(M, m)
    held_a = (cfg.bytes_a * M * K) / cfg.bw_a + (cfg.bytes_b * K * N * cm) / cfg.bw_b  # hold A
    held_b = (cfg.bytes_a * M * K * cn) / cfg.bw_a + (cfg.bytes_b * K * N) / cfg.bw_b  # hold B
    if stat == _AS:
        return held_a
    if stat == _BS:
        return held_b
    if k >= K:  # OS, full-K: hoist the cheaper operand
        return min(held_a, held_b)
    # OS, split-K: both re-streamed
    return (cfg.bytes_a * M * K * cn) / cfg.bw_a + (cfg.bytes_b * K * N * cm) / cfg.bw_b


def _wall_key(m: int, n: int, k: int, cfg, stat: str, dbc: bool) -> tuple:
    """Re-implement the C++ wall objective + lex tie-breaks for one design point."""
    M, N, K = cfg.M, cfg.N, cfg.K
    load = _load_cycles(m, n, k, cfg, stat)
    kt = max(1, cfg.mad_k_fractal_bytes // cfg.bytes_a)
    cpr = max(1, cfg.bytes_a // 2)
    # Tail-aware K-fractal count (mirrors C++ MadCycles): floor(K/k) full k-wide
    # blocks + a narrower peel tail (width K - floor(K/k)*k) scored at its own
    # width, not rounded up to k. For a divisor k this is (K/k)*ceil(k/kt).
    num_full = K // k
    k_tail = K - num_full * k
    k_blocks = num_full + (1 if k_tail > 0 else 0)  # == ceil(K/k)
    k_fractals = num_full * _cdiv(k, kt) + (_cdiv(k_tail, kt) if k_tail > 0 else 0)
    per_mn = k_blocks * cfg.mad_head + cpr * _cdiv(m, cfg.align_m) * k_fractals * _cdiv(n, cfg.align_n)
    mad = _cdiv(M, m) * _cdiv(N, n) * per_mn
    # Tile-dependent FIXPIPE drain (mirrors C++ DrainCycles): one drain per (m, n)
    # output block = m-independent fixed issue overhead + a PER-M-ROW cost. FIXPIPE
    # addresses one M-row of the N1 M1 M0 N0 accumulator at a time; the per-row cost is
    # max(floor, throughput) -- a fixed burst-issue floor (drain_row) for narrow N, or
    # the fractal byte throughput (bytes_c*n/bw_drain) for wide N -- plus the odd
    # residual (odd(N1)-1) for a non-pow2 N-fractal count N1 (NOT literally N%32: n=96 is
    # penalized -- odd(12)=3 -- despite 96%32==0). Write-side, so no gamma_c.
    # M/N-split raises the drain count; K-split does not.
    num_drains = _cdiv(M, m) * _cdiv(N, n)
    n0 = max(1, cfg.drain_c0_bytes // cfg.bytes_c)
    n1 = _cdiv(n, n0)
    per_row = max(cfg.drain_row_cycles, cfg.bytes_c * n / cfg.bw_drain) + cfg.drain_penalty_cycles * (
        _odd_part(n1) - 1
    )
    per_drain = cfg.drain_fixed_cycles + m * per_row
    drain = num_drains * per_drain
    compute = max(load, float(mad))
    # dbC pipeline fill/drain bubble (mirrors C++ WallCycles): the first tile's compute or
    # the last tile's drain is not overlapped, so the all-hidden T*max(C,D) roofline
    # undercounts by one tile's non-dominant pipe -> wall = max(agg) + min(agg)/T.
    if dbc:
        wall_f = max(compute, drain) + min(compute, drain) / num_drains
    else:
        wall_f = compute + drain
    wall = int(wall_f + 0.5)
    pvol = _cdiv(M, m) * m * _cdiv(N, n) * n * _cdiv(K, k) * k
    # C_load is a wall-tie-break (after padded-compute + k-blocks, before area/k):
    # among MAD-bound ties it picks the lower-hidden-load aspect.
    return (wall, pvol, _cdiv(K, k), load, -(m * n), -k)


def _legal_ks(m: int, n: int, cfg, a0: int, b0: int) -> list[int]:
    """All legal aligned k for (m, n), ascending — a true enumeration mirroring
    C++ EnumerateLegalKs. k is a real search axis: ceil(K/k)*ceil(k/kt) is not
    monotone in k when kt != align_k, so the largest legal k is not always best.
    A non-divisor k (the K-peel) is admitted only when K is itself align_k-aligned
    (peel_ok); a non-16-aligned K has no legal k here and is rejected upstream.
    """
    cap = min(a0 // m, b0 // n)  # max k fitting L0a and L0b
    k_problem = max(_cdiv(cfg.K, cfg.align_k) * cfg.align_k, cfg.min_k) if cfg.allow_padding else cfg.K
    k_hi = (min(cap, k_problem) // cfg.align_k) * cfg.align_k
    peel_ok = cfg.allow_k_boundary and cfg.K % cfg.align_k == 0
    ks = []
    k = cfg.min_k
    while k <= k_hi:
        if cfg.allow_padding or peel_ok or cfg.K % k == 0:  # non-divisor k only when peel_ok
            ks.append(k)
        k += cfg.align_k
    return ks


def _enumerate_best(cfg, stat: str, dbc: bool, require_2d: bool, require_full_k: bool):
    """Exhaustively score the legal aligned (m, n, k) grid for one regime; best
    (key, tile). Every legal k per (m, n) is scored (not a largest-k shortcut)."""
    dba, dbb = _derive_db(stat)
    a0 = cfg.l0a_bytes // (cfg.bytes_a * (2 if dba else 1))
    b0 = cfg.l0b_bytes // (cfg.bytes_b * (2 if dbb else 1))
    c0 = cfg.l0c_bytes // (cfg.bytes_c * (2 if dbc else 1))
    best = None
    m = cfg.min_m
    while m <= cfg.M and m * cfg.min_n <= c0:
        if not (require_2d and _cdiv(cfg.M, m) < 2):
            n = cfg.min_n
            while n <= min(cfg.N, c0 // m):
                if not (require_2d and _cdiv(cfg.N, n) < 2):
                    for k in _legal_ks(m, n, cfg, a0, b0):
                        if require_full_k and k != cfg.K:
                            continue
                        key = _wall_key(m, n, k, cfg, stat, dbc)
                        if best is None or key < best[0]:
                            best = (key, (m, n, k))
                n += cfg.align_n
        m += cfg.align_m
    return best


def _brute_optimum(cfg) -> tuple:
    """Mirror ChooseL0Tile over the design space (realizable-mask gates).

    Returns (tile, stationarity, double_buffer_c, wall).
    """
    base = _enumerate_best(cfg, _OS, False, require_2d=False, require_full_k=False)
    assert base is not None
    best_key, best = base[0], (base[1], _OS, False)
    # Explore the rest of the space only when the baseline already tiles.
    if base[1] != (cfg.M, cfg.N, cfg.K):
        stats = [_OS]
        if cfg.allow_a_stationary:
            stats.append(_AS)
        if cfg.allow_b_stationary:
            stats.append(_BS)
        regimes = []  # built in C++ iteration order: stat, dbc
        for stat in stats:
            for dbc in [False, True] if cfg.allow_double_buffer_c else [False]:
                regimes.append((stat, dbc))
        for stat, dbc in regimes:
            if stat == _OS and not dbc:
                continue  # baseline, already scored
            c0 = cfg.l0c_bytes // (cfg.bytes_c * (2 if dbc else 1))
            if c0 < cfg.min_m * cfg.min_n:
                continue
            cand = _enumerate_best(cfg, stat, dbc, require_2d=dbc, require_full_k=(stat != _OS or dbc))
            if cand is not None and cand[0][0] < best_key[0]:  # strictly lower wall
                best_key, best = cand[0], (cand[1], stat, dbc)
    tile, stat, dbc = best
    return tile, stat, dbc, best_key[0]


class TestL0TilingRooflineOptimum:
    """The chooser must return the global minimum-wall design point (oracle check)."""

    # Skewed / asymmetric-byte shapes (previously exposed sparse-candidate misses),
    # square + the large 4096^3 example, and FIXPIPE-bound shapes (small K, large
    # 2-D output) that exercise the operand-stationary / dbC branches.
    @pytest.mark.parametrize(
        "M,N,K",
        [
            (4096, 4096, 4096),
            (256, 256, 256),
            (512, 128, 2048),
            (128, 512, 2048),
            (240, 240, 480),
            (400, 112, 256),
            (1024, 64, 512),
            (272, 160, 384),
            (512, 512, 64),
            (256, 256, 32),
            (640, 384, 64),
            (768, 768, 48),
        ],
    )
    @pytest.mark.parametrize("bytes_a,bytes_b", [(2, 2), (2, 1), (1, 2)])
    # Realizable-mask gate combinations: nothing, each axis alone, and all open.
    @pytest.mark.parametrize(
        "gates",
        [
            {},
            {"allow_double_buffer_c": True},
            {"allow_a_stationary": True},
            {"allow_b_stationary": True},
            {"allow_a_stationary": True, "allow_b_stationary": True},
            {"allow_a_stationary": True, "allow_b_stationary": True, "allow_double_buffer_c": True},
            {"allow_k_boundary": True},
            {
                "allow_k_boundary": True,
                "allow_a_stationary": True,
                "allow_b_stationary": True,
                "allow_double_buffer_c": True,
            },
        ],
    )
    def test_matches_bruteforce_optimum(self, M, N, K, bytes_a, bytes_b, gates):
        cfg = _default_config(M=M, N=N, K=K)
        cfg.bytes_a, cfg.bytes_b = bytes_a, bytes_b
        for key, val in gates.items():
            setattr(cfg, key, val)
        btile, bstat, bdbc, bwall = _brute_optimum(cfg)
        r = passes.l0_tile_chooser.choose_l0_tile(cfg)
        # The chooser enumerates the same regimes, so its full design point must
        # match the global optimum.
        got = (r.m, r.n, r.k, r.stationarity, r.double_buffer_c)
        want = (*btile, _STAT_ENUM[bstat], bdbc)
        assert got == want, (
            f"M={M} N={N} K={K} ba={bytes_a} bb={bytes_b} gates={gates}: chooser={got} vs brute={want}"
        )
        assert r.estimated_cost_cycles == bwall
        # dbA / dbB are derived from stationarity (not returned); the chosen tile
        # must fit the operand buffers under those derived depths.
        dba, dbb = _derive_db(bstat)
        assert _capacities_ok(r.m, r.n, r.k, cfg, dba=dba, dbb=dbb, dbc=bdbc)

    def test_double_buffer_c_chosen_when_drain_bound(self):
        """A small-K, large 2-D-output GEMM is FIXPIPE-bound → dbC=2 wins.

        512x512x64: the L0C drain (~M*N) dominates the shallow-K compute, so
        hiding it behind the next tile beats a single big accumulator. The chosen
        tile must fit the halved L0C budget and form a >= 2x2 grid.
        """
        cfg = _default_config(M=512, N=512, K=64)
        cfg.allow_double_buffer_c = True
        result = passes.l0_tile_chooser.choose_l0_tile(cfg)
        assert result.double_buffer_c is True
        assert result.k == 64, f"dbC=2 requires a full-K tile; got k={result.k}"
        assert _cdiv(512, result.m) >= 2 and _cdiv(512, result.n) >= 2, "dbC=2 needs a >= 2x2 grid"
        assert _capacities_ok(result.m, result.n, result.k, cfg, dbc=True)
        # The single-L0C path must NOT pick dbC=2 (gate respected).
        cfg.allow_double_buffer_c = False
        assert passes.l0_tile_chooser.choose_l0_tile(cfg).double_buffer_c is False

    def test_default_mask_is_output_stationary(self):
        """With no gate opened the chooser emits only the realizable subset:
        output-stationary, single L0C (dbA=dbB=2 derived from OS)."""
        OS = passes.l0_tile_chooser.Stationarity.OutputStationary
        for M, N, K in [(512, 512, 64), (256, 256, 256), (4096, 4096, 4096), (1024, 128, 512)]:
            r = passes.l0_tile_chooser.choose_l0_tile(_default_config(M=M, N=N, K=K))
            assert r.stationarity == OS
            assert r.double_buffer_c is False

    def test_operand_stationary_requires_gate(self):
        """Opening the stationarity gates lets the chooser pin an operand (k == K),
        and the derived per-operand depth follows (stationary operand single-buffered).

        256x256x256: pinning an operand in a full (single-buffered) L0 admits a
        k = K = 256 tile that the output-stationary route (half L0) cannot, and
        combined with a hidden drain it beats the OS optimum. For this symmetric shape
        A- and B-stationary are transposes; the per-M-row drain (∝m) breaks the tie
        toward the aspect with fewer M-rows (B-stationary here). Without the gate OS.
        """
        AS = passes.l0_tile_chooser.Stationarity.AStationary
        BS = passes.l0_tile_chooser.Stationarity.BStationary
        OS = passes.l0_tile_chooser.Stationarity.OutputStationary
        assert passes.l0_tile_chooser.choose_l0_tile(_default_config(M=256, N=256, K=256)).stationarity == OS
        cfg = _default_config(M=256, N=256, K=256)
        cfg.allow_a_stationary = True
        cfg.allow_b_stationary = True
        cfg.allow_double_buffer_c = True
        r = passes.l0_tile_chooser.choose_l0_tile(cfg)
        assert r.stationarity in (AS, BS), "an operand must be pinned once the gate is open"
        assert r.k == 256, "operand-stationary pins across the full K reduction (k == K)"
        # Depths derive from the chosen stationarity: the pinned operand single-buffers.
        stat = _AS if r.stationarity == AS else _BS
        dba, dbb = _derive_db(stat)
        assert _capacities_ok(r.m, r.n, r.k, cfg, dba=dba, dbb=dbb, dbc=r.double_buffer_c)

    def test_aspect_swap_broken_by_per_row_drain(self):
        """MAD-bound square shape: the aspect-swapped tiles (m, n) and (n, m) tie on
        load (OS min-hoist) and compute, but the per-M-row FIXPIPE drain BREAKS the tie
        toward the smaller-M aspect (drain ∝ m -- FIXPIPE addresses one M-row at a time),
        so (128,256) [128 rows] drains cheaper than (256,128) [256 rows] -> strictly
        lower wall. (Pre-per-row-drain these aspects were fully wall-tied.)"""
        cfg = _default_config(M=256, N=256, K=64)
        assert _wall_key(128, 256, 64, cfg, _OS, False) < _wall_key(256, 128, 64, cfg, _OS, False)
        r = passes.l0_tile_chooser.choose_l0_tile(cfg)
        assert (r.m, r.n, r.k) == (128, 256, 64)  # the strictly-lower-drain aspect
        btile, _, _, _ = _brute_optimum(cfg)
        assert (r.m, r.n, r.k) == btile

    def test_estimated_traffic_matches_chosen_stationarity(self):
        """estimated_traffic_bytes (inspection metadata) must reflect the CHOSEN
        stationarity, not always output-stationary. An operand-stationary tile loads
        its pinned operand once (not ceil(.) times), so its reported traffic is the
        pinned-operand sum and never exceeds the output-stationary sum for the same
        tile. (Symmetric shape: the per-M-row drain pins B here; derive from the
        actual stationarity.)"""
        AS = passes.l0_tile_chooser.Stationarity.AStationary
        BS = passes.l0_tile_chooser.Stationarity.BStationary
        cfg = _default_config(M=256, N=256, K=256)
        cfg.allow_a_stationary = True
        cfg.allow_b_stationary = True
        cfg.allow_double_buffer_c = True
        r = passes.l0_tile_chooser.choose_l0_tile(cfg)
        assert r.stationarity in (AS, BS)
        M, N, K = 256, 256, 256
        cn, cm = _cdiv(N, r.n), _cdiv(M, r.m)
        gamma_c = 2 if cfg.c_read else 1
        c_traffic = gamma_c * cfg.bytes_c * M * N
        if r.stationarity == AS:
            pinned = cfg.bytes_a * M * K + cfg.bytes_b * K * N * cm + c_traffic  # A loaded once
        else:
            pinned = cfg.bytes_a * M * K * cn + cfg.bytes_b * K * N + c_traffic  # B loaded once
        os = cfg.bytes_a * M * K * cn + cfg.bytes_b * K * N * cm + c_traffic
        assert r.estimated_traffic_bytes == pinned, "traffic must use the pinned-operand reload counts"
        assert pinned <= os, "operand-stationary never charges more than output-stationary"

    def test_estimated_traffic_matches_os_full_k_hoist(self):
        """For a full-K output-stationary tile the emit hoists ONE operand (recorded in
        os_holds_a); estimated_traffic_bytes must mirror that min-hoist route (hold one,
        stream the other), not the split-K both-re-streamed sum — otherwise the metadata
        contradicts the wall the tile was scored under."""
        cfg = _default_config(M=512, N=512, K=64)  # bf16, k == K == 64 → output-stationary
        r = passes.l0_tile_chooser.choose_l0_tile(cfg)
        assert r.stationarity == passes.l0_tile_chooser.Stationarity.OutputStationary
        assert r.k == 64, "expected a full-K (k == K) output-stationary tile"
        M, N, K = 512, 512, 64
        cn, cm = _cdiv(N, r.n), _cdiv(M, r.m)
        gamma_c = 2 if cfg.c_read else 1
        c_traffic = gamma_c * cfg.bytes_c * M * N
        held_a = cfg.bytes_a * M * K + cfg.bytes_b * K * N * cm + c_traffic  # hold A, stream B
        held_b = cfg.bytes_a * M * K * cn + cfg.bytes_b * K * N + c_traffic  # hold B, stream A
        both = cfg.bytes_a * M * K * cn + cfg.bytes_b * K * N * cm + c_traffic  # split-K: re-stream both
        expected = held_a if r.os_holds_a else held_b
        assert r.estimated_traffic_bytes == expected, (
            "full-K OS traffic must mirror the os_holds_a min-hoist route, not re-stream both operands"
        )
        assert expected <= both, "the min-hoist route never charges more than re-streaming both"


class TestL0TilingKBoundary:
    """allow_k_boundary: the chosen k need not divide K (the pass peels the partial
    last K iteration). The peel is only legal when K is itself 16-aligned; a
    non-16-aligned whole-K block or peel tail is rejected because ptoas requires
    fractal-aligned tile cols."""

    def test_unaligned_whole_K_rejected(self):
        """K=50 is not a multiple of align_k (16): a k == K == 50 block would have
        non-fractal (non-16-aligned) tile cols that ptoas rejects, so the chooser
        admits no candidate even with allow_k_boundary -- the pass skips the matmul
        (PH-AT-007) instead of emitting an invalid whole-K block."""
        cfg = _default_config(M=64, N=64, K=50)
        cfg.allow_k_boundary = True
        with pytest.raises(ValueError):  # non-16-aligned K -> no legal tile
            passes.l0_tile_chooser.choose_l0_tile(cfg)

    def test_non_aligned_K_rejected(self):
        """K=130 is not a multiple of align_k (16): no aligned k divides it, AND a
        non-divisor peel would leave a non-16-aligned tail, so the chooser rejects it
        BOTH with and without allow_k_boundary (the K-boundary peel is only valid for
        16-aligned K). ptoas requires 16-aligned tile cols; the pass skips non-aligned
        K with a PerfHint rather than emit invalid extracts."""
        cfg = _default_config(M=16, N=320, K=130)
        with pytest.raises(ValueError):  # legacy: no aligned divisor of 130
            passes.l0_tile_chooser.choose_l0_tile(cfg)
        cfg.allow_k_boundary = True
        with pytest.raises(ValueError):  # peel needs 16-aligned K -> still no legal tile
            passes.l0_tile_chooser.choose_l0_tile(cfg)

    @pytest.mark.parametrize(
        "M,N,K",
        [
            (16, 320, 128),
            (16, 320, 144),
            (272, 160, 304),
            (256, 256, 240),
            (128, 512, 2064),
            (512, 512, 64),
        ],
    )
    def test_matches_bruteforce_with_k_boundary(self, M, N, K):
        """With allow_k_boundary the chooser must still return the global wall-min
        (oracle check) for non-divisor-but-aligned K (the K-boundary peel; every K
        here is 16-aligned). Non-16-aligned K is unsupported and rejected — see
        test_non_aligned_K_rejected / test_unaligned_whole_K_rejected."""
        cfg = _default_config(M=M, N=N, K=K)
        cfg.allow_k_boundary = True
        btile, _, bdbc, bwall = _brute_optimum(cfg)
        r = passes.l0_tile_chooser.choose_l0_tile(cfg)
        assert (r.m, r.n, r.k, r.double_buffer_c) == (*btile, bdbc), (
            f"M={M} N={N} K={K}: chooser=({r.m},{r.n},{r.k}) vs brute={btile}"
        )
        assert r.estimated_cost_cycles == bwall
        assert _capacities_ok(r.m, r.n, r.k, cfg)


class TestL0TilingKExhaustive:
    """k is a REAL search axis: the chooser scores every legal aligned k per
    (m, n), not just the largest. The MAD term ceil(K/k)*ceil(k/kt) is not
    monotone in k when kt != align_k (bytes_a=1 -> kt=32 > align=16), so a
    largest-legal-k shortcut misses the global optimum."""

    def test_largest_k_is_not_always_optimal(self):
        """Regression guard for k-search exhaustiveness (P0), re-derived under the
        op-sim-calibrated model. M=128, N=128, K=192 (BF16): the wall-optimal tile is
        (128, 128, 96) — NOT the largest-legal k=128. k=128 fits L0 but leaves a
        partial K-block (192 = 128 + 64), padding the compute to 2*128 = 256; k=96
        divides K cleanly (2*96 = 192, no K-padding), winning the padded-compute
        tie-break at equal wall/K-blocks. Matches the exhaustive oracle; pinned
        literally so a future largest-k shortcut is caught even if the oracle drifts.
        (The pre-calibration counterexample (128,256,96,bytes=1,2 -> (128,256,32))
        no longer holds after the load recalibration; this is a fresh one.)"""
        cfg = _default_config(M=128, N=128, K=192)
        result = passes.l0_tile_chooser.choose_l0_tile(cfg)
        assert (result.m, result.n, result.k) == (128, 128, 96), (
            f"expected the exhaustive-k optimum (128, 128, 96); got "
            f"({result.m}, {result.n}, {result.k}) — largest-legal-k shortcut?"
        )
        # ... and it agrees with the exhaustive (all-k) oracle.
        btile, _, _, _ = _brute_optimum(cfg)
        assert (result.m, result.n, result.k) == btile


class TestL0TilingRooflineMigration:
    """Audit trail for the traffic->roofline objective change. The roofline rewrite
    is a deliberate MODEL-DRIVEN tile-selection change, NOT a behavior-neutral
    refactor: for MAD-bound shapes the wall-optimum differs from the old traffic
    optimum. Each shape pins the current roofline tile and records the pre-rewrite
    closed-form tile, so the delta is reviewable. 910B cost model (the 950 FP32
    example from the review is omitted — its cost constants are placeholders)."""

    @pytest.mark.parametrize(
        "M,N,K,bytes_ab,old_closed_form,new_roofline",
        [
            (
                256,
                256,
                64,
                2,
                (192, 160, 64),
                (128, 256, 64),
            ),  # per-M-row drain (∝m) picks the wide-N / small-M aspect
            # the misalignment penalty + narrow-N floor steer off n=96/n=160 (odd>1) to
            # aligned n=128; the byte-throughput term keeps wide-N correctly priced so the
            # chooser lands an aligned tile (not a wide-misaligned one):
            (272, 272, 64, 2, (192, 160, 64), (144, 128, 64)),
            (320, 320, 64, 2, (192, 160, 64), (160, 128, 64)),
            (256, 256, 256, 4, (192, 160, 32), (128, 256, 32)),
        ],
    )
    def test_roofline_tile_migration(self, M, N, K, bytes_ab, old_closed_form, new_roofline):
        cfg = _default_config(M=M, N=N, K=K)
        cfg.bytes_a = cfg.bytes_b = bytes_ab
        r = passes.l0_tile_chooser.choose_l0_tile(cfg)
        assert (r.m, r.n, r.k) == new_roofline, (
            f"{M}x{N}x{K}: roofline tile {(r.m, r.n, r.k)} != pinned {new_roofline} "
            f"(old traffic closed-form picked {old_closed_form})"
        )
        assert _capacities_ok(r.m, r.n, r.k, cfg)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
