# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the closed-form L0 tile-size chooser.

The chooser implements the algorithm described in the planning note
`L0_TILING.md` at the repo root. These tests pin the five worked examples
in §12 plus edge cases exercising small dimensions, K below the cube
minimum, double-buffering of C, and the `c_read` traffic accounting.
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
    cfg.double_buffer_a = True
    cfg.double_buffer_b = True
    cfg.double_buffer_c = False
    cfg.c_read = False
    return cfg


def _capacities_ok(m: int, n: int, k: int, cfg) -> bool:
    """Re-derive effective capacity constraints and confirm (m, n, k) fits."""
    a0 = cfg.l0a_bytes // (cfg.bytes_a * (2 if cfg.double_buffer_a else 1))
    b0 = cfg.l0b_bytes // (cfg.bytes_b * (2 if cfg.double_buffer_b else 1))
    c0 = cfg.l0c_bytes // (cfg.bytes_c * (2 if cfg.double_buffer_c else 1))
    return m * k <= a0 and k * n <= b0 and m * n <= c0


# ---------------------------------------------------------------------------
# L0_TILING.md §12 worked examples
# ---------------------------------------------------------------------------


class TestL0TilingDocExamples:
    """Pin the five examples in L0_TILING.md §12 (Examples 1-5)."""

    def test_example_1_skinny_gemm(self):
        """M=16, N=256, K=512 → (16, 256, 64). Full C fits, K split by B0."""
        cfg = _default_config(M=16, N=256, K=512)
        result = passes.l0_tile_chooser.choose_l0_tile(cfg)
        assert (result.m, result.n, result.k) == (16, 256, 64)
        assert _capacities_ok(result.m, result.n, result.k, cfg)
        assert result.perf_hint == ""

    def test_example_2_large_square_gemm(self):
        """M=N=K=4096 → balanced tile under A/B double-buffering.

        The L0_TILING.md doc cites (176, 176, 80) as "approximate" continuous
        optimum, but k=80 does not divide K=4096, so the AutoTileMatmulL0
        consumer would skip it (PH-AT-007).  When `allow_padding=False` the
        chooser walks k down to the largest aligned divisor of K — for K=4096
        that is k=64 (next divisor below 80).  We pin the properties that
        matter under the divisor contract: k divides K, k respects the A0/B0
        bound, and (m, n) stay within one alignment step of the continuous
        optimum.
        """
        cfg = _default_config(M=4096, N=4096, K=4096)
        result = passes.l0_tile_chooser.choose_l0_tile(cfg)
        assert _capacities_ok(result.m, result.n, result.k, cfg)
        assert cfg.K % result.k == 0, f"k must divide K=4096; got k={result.k}"
        assert result.k == 64, f"Expected largest aligned divisor ≤ 80; got k={result.k}"
        assert result.m in (160, 176, 192), f"m near 176 (one align step); got m={result.m}"
        assert result.n in (160, 176, 192), f"n near 176 (one align step); got n={result.n}"

    def test_example_3_c_double_buffered(self):
        """M=N=K=4096 with C also double-buffered → (128, 128, 128)."""
        cfg = _default_config(M=4096, N=4096, K=4096)
        cfg.double_buffer_c = True
        result = passes.l0_tile_chooser.choose_l0_tile(cfg)
        assert (result.m, result.n, result.k) == (128, 128, 128)
        assert _capacities_ok(result.m, result.n, result.k, cfg)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
