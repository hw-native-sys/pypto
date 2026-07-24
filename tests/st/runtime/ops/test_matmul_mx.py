# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""a5/a5sim ST for MX DSL: load scales → tget_scale_addr → matmul_mx.

Sample = PTO-ISA a5 ST ``tmatmul_mx`` case1 / PTOAS ``fp8_e5m2_128x64x64``:
  M=128, K=64, N=64; A/B = FP8E5M2; scale = FP8E8M0; C = FP32
  scales: A_s [M, K/32]=[128, 2], B_s [K/32, N]=[2, 64]
  GM scale layouts: mx_a_zz / mx_b_nn (ISA ``Layout::MX_A_ZZ`` / ``MX_B_NN``)

Host inputs + golden are ported from:
  pto-isa/tests/npu/a5/src/st/testcase/tmatmul_mx/gen_data.py
  (case1: seed=19, randint A/B in [-10,10), scale uint8 in [127,130), zz/nn pack)
"""

import math

import numpy as np
import pypto.language as pl
import pytest
import torch
from pypto.backend import BackendType, reset_for_testing, set_backend_type

# PTO-ISA tmatmul_mx case1 / PTOAS fp8_e5m2_128x64x64
_M, _K, _N = 128, 64, 64
_KMX = _K // 32  # 2
# PTOAS cases.py eps / ISA ResultCmp(…, 0.001f)
_EPS = 1e-3


def _require_torch_fp8_host():
    """Host tensors for device ST need torch float8 (incl. E8M0, PyTorch 2.7+).

    CI images currently ship torch 2.6 (has e5m2, not e8m0fnu). Compile-only
    coverage uses signature-mode ``compile()`` and does not call this helper.
    """
    if not hasattr(torch, "float8_e5m2") or not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("PyTorch float8_e5m2 / float8_e8m0fnu required for MX device ST")


# --- Ported from pto-isa tmatmul_mx/gen_data.py (convert_*_scale_format) ---


def _convert_x1_scale_format(x1_mx_gm: np.ndarray, block_size: int = 16, c0_size_mx: int = 2) -> np.ndarray:
    """Pack A-side E8M0 scales to MX_A_ZZ GM layout (ISA gen_data.convert_x1_scale_format)."""
    m, k = x1_mx_gm.shape
    pad_m = (block_size - m % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx

    if pad_m > 0 or pad_k > 0:
        padded = np.pad(x1_mx_gm, ((0, pad_m), (0, pad_k)), mode="constant", constant_values=0)
    else:
        padded = x1_mx_gm

    m_padded = m + pad_m
    k_padded = k + pad_k

    x1_scale_gm = padded.reshape(
        (int(m_padded / block_size), block_size, int(k_padded / c0_size_mx), c0_size_mx)
    )
    x1_scale_gm = x1_scale_gm.transpose(0, 2, 1, 3)
    x1_scale_gm = x1_scale_gm.reshape(
        x1_scale_gm.shape[0] * x1_scale_gm.shape[1],
        x1_scale_gm.shape[2] * x1_scale_gm.shape[3],
    )
    return x1_scale_gm


def _convert_x2_scale_format(x2_mx_gm: np.ndarray, block_size: int = 16, c0_size_mx: int = 2) -> np.ndarray:
    """Pack B-side E8M0 scales to MX_B_NN GM layout (ISA gen_data.convert_x2_scale_format)."""
    k, n = x2_mx_gm.shape
    pad_n = (block_size - n % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx

    if pad_n > 0 or pad_k > 0:
        padded = np.pad(x2_mx_gm, ((0, pad_k), (0, pad_n)), mode="constant", constant_values=0)
    else:
        padded = x2_mx_gm

    k_padded, n_padded = padded.shape

    x2_scale_gm = padded.reshape((int(k_padded / c0_size_mx), c0_size_mx, int(n_padded / 16), 16)).transpose(
        2, 0, 3, 1
    )
    x2_scale_gm = x2_scale_gm.reshape(
        x2_scale_gm.shape[1] * x2_scale_gm.shape[3],
        x2_scale_gm.shape[0] * x2_scale_gm.shape[2],
    )
    return x2_scale_gm


def _u8_e8m0_to_torch(buf: np.ndarray, logical_shape: tuple[int, int]) -> torch.Tensor:
    """View ZZ/NN-packed uint8 E8M0 bytes as torch.float8_e8m0fnu with logical shape."""
    flat = np.ascontiguousarray(buf, dtype=np.uint8).reshape(-1)
    assert flat.size == logical_shape[0] * logical_shape[1]
    return torch.from_numpy(flat.copy()).view(torch.float8_e8m0fnu).reshape(logical_shape)


def _make_case1_inputs():
    """Build host tensors + golden for ISA ``TMATMULMXTest.case1``.

    Logic mirrors ``gen_golden_data`` for
    ``TmatmulmxParams(fp8_e5m2, fp8_e5m2, float32, 128, 64, 64, False)``
    with default ``scale_a_format='zz'``, ``scale_b_format='nn'``.
    """
    _require_torch_fp8_host()
    # gen_data.py: np.random.seed(19) at module import; case1 is first emitted case.
    np.random.seed(19)

    m, k, n = _M, _K, _N
    k_aligned = (k + 63) // 64 * 64  # align_to_multiple(k, 64)
    assert k_aligned == k  # case1 K already 64-aligned

    # ISA: np.random.randint(-10, 10, …).astype(fp8_e5m2) — ints in range are exact in e5m2.
    x1_i = np.random.randint(-10, 10, [m, k])
    x2_i = np.random.randint(-10, 10, [k, n])
    a = torch.from_numpy(x1_i.astype(np.float32)).to(torch.float8_e5m2)
    b = torch.from_numpy(x2_i.astype(np.float32)).to(torch.float8_e5m2)

    x1_mx_gm = np.random.randint(127, 130, [m, math.ceil(k / 32)]).astype(np.uint8)
    x2_mx_gm = np.random.randint(127, 130, [math.ceil(k / 32), n]).astype(np.uint8)

    # Golden: dequant by E8M0 then matmul (ISA float64 path).
    x1_mx = 2.0 ** (x1_mx_gm.astype(np.float64) - 127)
    x2_mx = 2.0 ** (x2_mx_gm.astype(np.float64) - 127)
    x1_full = np.zeros([m, k_aligned], dtype=np.float64)
    x2_full = np.zeros([k_aligned, n], dtype=np.float64)
    x1_f = a.to(torch.float64).numpy()
    x2_f = b.to(torch.float64).numpy()
    for i in range(x1_f.shape[1]):
        x1_full[:, i] = x1_f[:, i] * x1_mx[:, i // 32]
        x2_full[i, :] = x2_f[i, :] * x2_mx[i // 32, :]
    golden = np.matmul(x1_full[:, :k], x2_full[:k, :]).astype(np.float32)

    # GM scales must be ZZ / NN packed (TLoad MX_A_ZZ / MX_B_NN → ZZ2ZZ / NN2NN).
    a_s = _u8_e8m0_to_torch(_convert_x1_scale_format(x1_mx_gm), (m, _KMX))
    b_s = _u8_e8m0_to_torch(_convert_x2_scale_format(x2_mx_gm), (_KMX, n))
    c = torch.zeros((m, n), dtype=torch.float32)
    expected = torch.from_numpy(golden)
    return a, a_s, b, b_s, c, expected


@pl.jit
def matmul_mx_prequant(
    a: pl.Tensor[[128, 64], pl.FP8E5M2],
    a_s: pl.Tensor[[128, 2], pl.FP8E8M0],
    b: pl.Tensor[[64, 64], pl.FP8E5M2],
    b_s: pl.Tensor[[2, 64], pl.FP8E8M0],
    c: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        ta = pl.load(a, [0, 0], [128, 64], target_memory=pl.Mem.Mat)
        tas = pl.load(a_s, [0, 0], [128, 2], target_memory=pl.Mem.Mat, mx_layout="mx_a_zz")
        tb = pl.load(b, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
        tbs = pl.load(b_s, [0, 0], [2, 64], target_memory=pl.Mem.Mat, mx_layout="mx_b_nn")
        la = pl.move(ta, target_memory=pl.Mem.Left)
        las = pl.move(tas, target_memory=pl.Mem.LeftScale)
        rb = pl.move(tb, target_memory=pl.Mem.Right)
        rbs = pl.move(tbs, target_memory=pl.Mem.RightScale)
        las = pl.tget_scale_addr(las, la)
        rbs = pl.tget_scale_addr(rbs, rb)
        tile_c = pl.matmul_mx(la, las, rb, rbs)
        pl.store(tile_c, [0, 0], c)
    return c


@pytest.mark.platforms("a5", "a5sim")
class TestMatmulMx:
    """MX matmul system tests (Ascend950 / a5sim), ISA case1 sample."""

    def test_matmul_mx_compiles_ptoas(self, test_config):
        """Frontend → PTOAS EmitC succeeds for the MX GEMM chain.

        Uses signature-mode ``compile()`` so CI (torch 2.6, no float8_e8m0fnu)
        still exercises the full codegen/PTOAS path without host FP8 tensors.
        """
        matmul_mx_prequant._cache.clear()
        reset_for_testing()
        set_backend_type(BackendType.Ascend950)
        compiled = matmul_mx_prequant.compile()
        assert compiled is not None
        assert test_config is not None

    @pytest.mark.platforms("a5")
    def test_matmul_mx_prequant_on_device(self, test_config):
        """End-to-end on real a5 (ISA case1; host-prequant FP8+E8M0).

        a5sim is excluded: CPU TLoad stubs only accept ND/DN/NZ, not MX_A_ZZ/MX_B_NN.
        Requires torch.float8_e8m0fnu (PyTorch 2.7+) for host scale tensors.
        """
        matmul_mx_prequant._cache.clear()
        a, a_s, b, b_s, c, expected = _make_case1_inputs()
        matmul_mx_prequant(a, a_s, b, b_s, c, config=test_config)
        assert torch.allclose(c, expected, rtol=_EPS, atol=_EPS), (
            f"matmul_mx case1 failed: max diff = {(c - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
