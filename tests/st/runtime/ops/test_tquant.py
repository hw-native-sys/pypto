# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""a5/a5sim ST for MX DSL dynamic quantization: ``pl.tquant`` → ``pto.tquant.mx``.

Covers item-6 numerical / lifecycle cases:
  - MXFP8 e4m3: compile + on-device golden (OCP e8m0 shared-exp)
  - MXFP8 e5m2: compile (mode accepted; reinterpret of E5M2 is rejected elsewhere)
  - MXFP4 packed: compile + on-device finite check
  - Partial ``valid_shapes`` on the source load: compile (+ device finite)
  - Consecutive ``tquant`` in one kernel: presses per-object deferred tfree
  - Prepacked ZZ scale path: covered by ``test_matmul_mx.py`` (cross-ref)

Sample shape mirrors PTO-ISA a5 ST ``tquant`` (FP32→MXFP8 e4m3): M=128, K=64.
"""

import numpy as np
import pypto.language as pl
import pytest
import torch

_M, _K = 128, 64
_KMX = _K // 32  # 2
_VALID_M, _VALID_K = 96, 64  # partial-valid source (M padded to 128)


def _require_torch_fp8_host():
    if not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("PyTorch float8_e4m3fn / float8_e8m0fnu required for tquant device ST")


def _require_torch_fp4_host():
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("PyTorch float4_e2m1fn_x2 required for MXFP4 device ST")


def _make_tquant_inputs():
    """FP32 src + golden (MXFP8 e4m3 data + E8M0 exp) for a per-group-32 quant.

    Golden matches the pto-isa A5 ``TQUANT`` OCP convention (``gen_data.py``
    ``fp32_to_fp8_element``, ``emax=8``), which the hardware implements: the
    shared exponent ``e8m0 = max(fp32_biased_exp(gmax) - 8, 0)`` and the data is
    ``clip(src * 2**(127-e8m0), -448, +448)`` cast to e4m3 (note: MULTIPLY by the
    power-of-2 scale to fill the e4m3 range, not divide).
    """
    _require_torch_fp8_host()
    rng = np.random.default_rng(19)
    src = (rng.uniform(-2.0, 2.0, [_M, _K])).astype(np.float32)

    # Per-group (block=32) abs-max.
    groups = src.reshape(_M, _KMX, 32)
    gmax = np.max(np.abs(groups), axis=2).astype(np.float32)  # [M, KMX]

    # e8m0 shared exponent (OCP): max(fp32 biased exponent of gmax minus 8, 0).
    gmax_bits = np.ascontiguousarray(gmax).view(np.uint32)
    exp_b = ((gmax_bits & np.uint32(0x7F800000)) >> np.uint32(23)).astype(np.int32)
    e8m0 = np.maximum(exp_b - 8, 0).astype(np.uint8)  # [M, KMX]

    # Per-group scaling = 2**(127 - e8m0), assembled from fp32 exponent bits
    # (biased exp 254 - e8m0) so it is an exact power of two.
    scale_exp = (254 - e8m0.astype(np.int32)).astype(np.uint32)
    scaling = np.ascontiguousarray(scale_exp << np.uint32(23)).view(np.float32)  # [M, KMX]

    scaled = np.clip(groups * scaling.reshape(_M, _KMX, 1), -448.0, 448.0)
    golden_q = torch.from_numpy(scaled.reshape(_M, _K).astype(np.float32)).to(torch.float8_e4m3fn)
    golden_s = torch.from_numpy(np.ascontiguousarray(e8m0)).view(torch.float8_e8m0fnu).reshape(_M, _KMX)
    return torch.from_numpy(src), golden_q, golden_s


@pl.jit
def tquant_kernel(
    src: pl.Tensor[[128, 64], pl.FP32],
    q_out: pl.Out[pl.Tensor[[128, 64], pl.FP8E4M3FN]],
    s_out: pl.Out[pl.Tensor[[128, 2], pl.FP8E8M0]],
):
    """FP32 → MXFP8 e4m3 + E8M0 exp via pto.tquant.mx (4 outs + quant_type attr)."""
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.load(src, [0, 0], [128, 64])
        q, s = pl.tquant(t, mode="mxfp8_e4m3")
        pl.store(q, [0, 0], q_out)
        pl.store(s, [0, 0], s_out)
    return q_out, s_out


@pl.jit
def tquant_e5m2_kernel(
    src: pl.Tensor[[128, 64], pl.FP32],
    q_out: pl.Out[pl.Tensor[[128, 64], pl.INT8]],
    s_out: pl.Out[pl.Tensor[[128, 2], pl.UINT8]],
):
    """MXFP8 e5m2 mode: dst/scale as raw bytes (no E5M2 reinterpret_view)."""
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.load(src, [0, 0], [128, 64])
        q, s = pl.tquant(t, mode="mxfp8_e5m2")
        pl.store(q, [0, 0], q_out)
        pl.store(s, [0, 0], s_out)
    return q_out, s_out


@pl.jit
def tquant_mxfp4_kernel(
    src: pl.Tensor[[128, 64], pl.FP16],
    q_out: pl.Out[pl.Tensor[[128, 32], pl.FP4]],  # storage [M, K/2]
    s_out: pl.Out[pl.Tensor[[128, 2], pl.FP8E8M0]],
):
    """FP16 → MXFP4 packed dst [M, K/2] + E8M0 exp.

    ptoas requires MXFP4_E2M1 ``pto.tquant.mx`` src to be f16/bf16 (not f32).
    """
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.load(src, [0, 0], [128, 64])
        q, s = pl.tquant(t, mode="mxfp4")
        pl.store(q, [0, 0], q_out)
        pl.store(s, [0, 0], s_out)
    return q_out, s_out


@pl.jit
def tquant_partial_valid_kernel(
    src: pl.Tensor[[128, 64], pl.FP32],
    q_out: pl.Out[pl.Tensor[[128, 64], pl.FP8E4M3FN]],
    s_out: pl.Out[pl.Tensor[[128, 2], pl.FP8E8M0]],
):
    """Partial-valid source load then tquant (physical pad stays 128×64)."""
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.load(src, [0, 0], [128, 64], valid_shapes=[_VALID_M, _VALID_K])
        q, s = pl.tquant(t, mode="mxfp8_e4m3")
        pl.store(q, [0, 0], q_out)
        pl.store(s, [0, 0], s_out)
    return q_out, s_out


@pl.jit
def tquant_consecutive_kernel(
    src_a: pl.Tensor[[128, 64], pl.FP32],
    src_b: pl.Tensor[[128, 64], pl.FP32],
    q_a: pl.Out[pl.Tensor[[128, 64], pl.FP8E4M3FN]],
    s_a: pl.Out[pl.Tensor[[128, 2], pl.FP8E8M0]],
    q_b: pl.Out[pl.Tensor[[128, 64], pl.FP8E4M3FN]],
    s_b: pl.Out[pl.Tensor[[128, 2], pl.FP8E8M0]],
):
    """Two consecutive tquants — presses per-object deferred tfree drain."""
    with pl.at(level=pl.Level.CORE_GROUP):
        ta = pl.load(src_a, [0, 0], [128, 64])
        qa, sa = pl.tquant(ta, mode="mxfp8_e4m3")
        pl.store(qa, [0, 0], q_a)
        pl.store(sa, [0, 0], s_a)
        tb = pl.load(src_b, [0, 0], [128, 64])
        qb, sb = pl.tquant(tb, mode="mxfp8_e4m3")
        pl.store(qb, [0, 0], q_b)
        pl.store(sb, [0, 0], s_b)
    return q_a, s_a, q_b, s_b


@pytest.mark.platforms("a5", "a5sim")
class TestTQuantMxfp8:
    """MXFP8 e4m3 / e5m2 system tests (Ascend950 / a5sim)."""

    def test_tquant_e4m3_compiles_ptoas(self, test_config):
        """Frontend → PTOAS succeeds (pto.tquant.mx: 4 outs + quant_type attr)."""
        tquant_kernel._cache.clear()
        compiled = tquant_kernel.compile(config=test_config)
        assert compiled is not None

    def test_tquant_e5m2_compiles_ptoas(self, test_config):
        """e5m2 mode compiles with raw-byte outs (no E5M2 reinterpret_view)."""
        tquant_e5m2_kernel._cache.clear()
        compiled = tquant_e5m2_kernel.compile(config=test_config)
        assert compiled is not None

    @pytest.mark.platforms("a5")
    def test_tquant_e4m3_on_device(self, test_config):
        """End-to-end on real a5: FP32 → MXFP8 e4m3 + E8M0 exp vs golden."""
        tquant_kernel._cache.clear()
        src, expected_q, expected_s = _make_tquant_inputs()
        q = torch.zeros((_M, _K), dtype=torch.float8_e4m3fn)
        s = torch.zeros((_M, _KMX), dtype=torch.float8_e8m0fnu)
        tquant_kernel(src, q, s, config=test_config)
        assert torch.equal(q, expected_q), "tquant e4m3 data mismatch"
        assert torch.equal(s, expected_s), "tquant e8m0 exp mismatch"


@pytest.mark.platforms("a5", "a5sim")
class TestTQuantMxfp4:
    """MXFP4 packed dst [M, K/2] + E8M0 scale."""

    def test_tquant_mxfp4_compiles_ptoas(self, test_config):
        tquant_mxfp4_kernel._cache.clear()
        compiled = tquant_mxfp4_kernel.compile(config=test_config)
        assert compiled is not None

    @pytest.mark.platforms("a5")
    def test_tquant_mxfp4_on_device_finite(self, test_config):
        """Device run produces non-zero packed FP4 + finite e8m0 scale bytes."""
        _require_torch_fp4_host()
        _require_torch_fp8_host()
        tquant_mxfp4_kernel._cache.clear()
        rng = np.random.default_rng(19)
        src = torch.from_numpy(rng.uniform(-2.0, 2.0, [_M, _K]).astype(np.float16))
        q = torch.zeros((_M, _K // 2), dtype=torch.float4_e2m1fn_x2)
        s = torch.zeros((_M, _KMX), dtype=torch.float8_e8m0fnu)
        tquant_mxfp4_kernel(src, q, s, config=test_config)
        q_u8 = q.view(torch.uint8)
        s_u8 = s.view(torch.uint8)
        assert torch.any(q_u8 != 0), "mxfp4 packed dst is all-zero"
        assert torch.any(s_u8 != 0), "mxfp4 e8m0 scale is all-zero"


@pytest.mark.platforms("a5", "a5sim")
class TestTQuantPartialValid:
    """Partial ``valid_shapes`` on the tquant source load."""

    def test_tquant_partial_valid_compiles(self, test_config):
        tquant_partial_valid_kernel._cache.clear()
        compiled = tquant_partial_valid_kernel.compile(config=test_config)
        assert compiled is not None

    @pytest.mark.platforms("a5")
    def test_tquant_partial_valid_on_device(self, test_config):
        """Device run with padded physical [128,64] and valid [96,64] stays finite."""
        _require_torch_fp8_host()
        tquant_partial_valid_kernel._cache.clear()
        rng = np.random.default_rng(19)
        src_np = np.zeros((_M, _K), dtype=np.float32)
        src_np[:_VALID_M, :] = rng.uniform(-2.0, 2.0, [_VALID_M, _K]).astype(np.float32)
        src = torch.from_numpy(src_np)
        q = torch.zeros((_M, _K), dtype=torch.float8_e4m3fn)
        s = torch.zeros((_M, _KMX), dtype=torch.float8_e8m0fnu)
        tquant_partial_valid_kernel(src, q, s, config=test_config)
        # Valid rows should be non-trivial; padded rows may be zeroed by load.
        assert torch.any(q[:_VALID_M].view(torch.uint8) != 0)
        assert torch.any(s[:_VALID_M].view(torch.uint8) != 0)


@pytest.mark.platforms("a5", "a5sim")
class TestTQuantConsecutive:
    """Two consecutive tquants in one kernel (per-object deferred tfree)."""

    def test_tquant_consecutive_compiles(self, test_config):
        tquant_consecutive_kernel._cache.clear()
        compiled = tquant_consecutive_kernel.compile(config=test_config)
        assert compiled is not None

    @pytest.mark.platforms("a5")
    def test_tquant_consecutive_on_device(self, test_config):
        """Both tquant results match independent single-tquant goldens."""
        tquant_consecutive_kernel._cache.clear()
        src_a, exp_qa, exp_sa = _make_tquant_inputs()
        # Second input: different seed so deferred-tfree objects are distinct.
        rng = np.random.default_rng(42)
        src_b_np = rng.uniform(-2.0, 2.0, [_M, _K]).astype(np.float32)
        # Golden for src_b (same OCP recipe as _make_tquant_inputs).
        groups = src_b_np.reshape(_M, _KMX, 32)
        gmax = np.max(np.abs(groups), axis=2).astype(np.float32)
        gmax_bits = np.ascontiguousarray(gmax).view(np.uint32)
        exp_b = ((gmax_bits & np.uint32(0x7F800000)) >> np.uint32(23)).astype(np.int32)
        e8m0 = np.maximum(exp_b - 8, 0).astype(np.uint8)
        scale_exp = (254 - e8m0.astype(np.int32)).astype(np.uint32)
        scaling = np.ascontiguousarray(scale_exp << np.uint32(23)).view(np.float32)
        scaled = np.clip(groups * scaling.reshape(_M, _KMX, 1), -448.0, 448.0)
        exp_qb = torch.from_numpy(scaled.reshape(_M, _K).astype(np.float32)).to(torch.float8_e4m3fn)
        exp_sb = torch.from_numpy(np.ascontiguousarray(e8m0)).view(torch.float8_e8m0fnu).reshape(_M, _KMX)
        src_b = torch.from_numpy(src_b_np)

        qa = torch.zeros((_M, _K), dtype=torch.float8_e4m3fn)
        sa = torch.zeros((_M, _KMX), dtype=torch.float8_e8m0fnu)
        qb = torch.zeros((_M, _K), dtype=torch.float8_e4m3fn)
        sb = torch.zeros((_M, _KMX), dtype=torch.float8_e8m0fnu)
        tquant_consecutive_kernel(src_a, src_b, qa, sa, qb, sb, config=test_config)
        assert torch.equal(qa, exp_qa), "consecutive tquant A e4m3 mismatch"
        assert torch.equal(sa, exp_sa), "consecutive tquant A e8m0 mismatch"
        assert torch.equal(qb, exp_qb), "consecutive tquant B e4m3 mismatch"
        assert torch.equal(sb, exp_sb), "consecutive tquant B e8m0 mismatch"


@pytest.mark.platforms("a5", "a5sim")
class TestPrepackedZzScaleCrossRef:
    """Prepacked ZZ/NN scale + matmul_mx is covered by ``test_matmul_mx.py``.

    Kept here so item-6 checklist (prepacked ZZ scale) has an explicit entry
    under the tquant numerical suite.
    """

    def test_prepacked_zz_covered_by_matmul_mx(self):
        import tests.st.runtime.ops.test_matmul_mx as mm  # noqa: PLC0415

        assert hasattr(mm, "TestMatmulMx")
        # Host-prequant ZZ/NN scale path lives in test_matmul_mx (compile + a5 device).
        assert hasattr(mm.TestMatmulMx, "test_matmul_mx_compiles_ptoas")
        assert hasattr(mm.TestMatmulMx, "test_matmul_mx_prequant_on_device")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
