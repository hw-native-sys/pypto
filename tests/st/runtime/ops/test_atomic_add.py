# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for atomic-add accumulation.

Covers both surface forms that emit an atomic-add store:

  * ``pl.store(tile, offsets, tensor, atomic=pl.AtomicType.Add)``
      Atomically accumulates a tile into a tensor at ``offsets``. The
      destination tensor is expected to already hold the baseline value
      onto which the tile is added.

  * ``pl.assemble(tensor, tile, offsets, atomic=pl.AtomicType.Add)``
      Tensor-level atomic accumulation. Used canonically by Split-K
      matmul, where each parallel core atomic-adds its partial product
      into a shared output (see ``examples/kernels/10_split_k.py``).

Codegen-level coverage already exists in
``tests/ut/codegen/test_pto_codegen_ops.py`` and ``tests/ut/jit/test_split_k.py``;
this module exercises the end-to-end execution path on device/simulator.
"""

import pypto.language as pl
import pytest
import torch

# ---------------------------------------------------------------------------
# Kernels: pl.store(..., atomic=AtomicType.Add)
# ---------------------------------------------------------------------------


@pl.jit
def atomic_add_store_fp32(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    """``out += x`` via a single atomic-add store of the loaded tile."""
    with pl.at(level=pl.Level.CORE_GROUP):
        x_tile = pl.load(x, [0, 0], [16, 16])
        pl.store(x_tile, [0, 0], out, atomic=pl.AtomicType.Add)
    return out


@pl.jit
def atomic_add_store_int32(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    """INT32 variant of :func:`atomic_add_store_fp32` (atomic-add accumulation)."""
    with pl.at(level=pl.Level.CORE_GROUP):
        x_tile = pl.load(x, [0, 0], [16, 16])
        pl.store(x_tile, [0, 0], out, atomic=pl.AtomicType.Add)
    return out


# ---------------------------------------------------------------------------
# Kernel: pl.assemble(..., atomic=AtomicType.Add) -- Split-K matmul
# ---------------------------------------------------------------------------

_SPLIT_K_M = 64
_SPLIT_K_N = 64
_SPLIT_K_K = 512
_SPLIT_K_SPLITS = 4
_SPLIT_K_KS = _SPLIT_K_K // _SPLIT_K_SPLITS


@pl.jit
def matmul_split_k_atomic(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    """Split-K matmul: K split across ``_SPLIT_K_SPLITS`` parallel cores.

    Each core computes an ``[M, KS] @ [KS, N]`` partial and atomic-adds the
    result into the shared output ``c``. ``c`` is zero-initialised inside
    the kernel so the accumulation starts from a clean buffer.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="zero_init"):
        c = pl.assemble(c, pl.full([_SPLIT_K_M, _SPLIT_K_N], dtype=pl.FP32, value=0.0), [0, 0])
    for ks in pl.parallel(0, _SPLIT_K_SPLITS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="split_k"):
            k0 = ks * _SPLIT_K_KS
            a_k = a[:, k0 : k0 + _SPLIT_K_KS]
            b_k = b[k0 : k0 + _SPLIT_K_KS, :]
            partial = pl.matmul(a_k, b_k, out_dtype=pl.FP32)
            c = pl.assemble(c, partial, [0, 0], atomic=pl.AtomicType.Add)
    return c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAtomicAddStore:
    """``pl.store(..., atomic=AtomicType.Add)`` accumulates a tile onto a baseline."""

    def test_atomic_add_store_fp32(self, test_config):
        """FP32: ``out`` starts at 1.0 everywhere; kernel atomic-adds ``x`` onto it."""
        atomic_add_store_fp32._cache.clear()
        torch.manual_seed(0)
        x = torch.randn(16, 16, dtype=torch.float32)
        baseline = 1.0
        out = torch.full((16, 16), baseline, dtype=torch.float32)
        atomic_add_store_fp32(x, out, config=test_config)
        expected = baseline + x
        assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5), (
            f"FP32 atomic-add store mismatch: max diff = {(out - expected).abs().max().item()}"
        )

    def test_atomic_add_store_int32(self, test_config):
        """INT32: ``out`` starts at 5 everywhere; kernel atomic-adds ``x`` onto it."""
        atomic_add_store_int32._cache.clear()
        torch.manual_seed(0)
        x = torch.randint(-100, 100, (16, 16), dtype=torch.int32)
        baseline = 5
        out = torch.full((16, 16), baseline, dtype=torch.int32)
        atomic_add_store_int32(x, out, config=test_config)
        expected = baseline + x
        assert torch.equal(out, expected), (
            f"INT32 atomic-add store mismatch: max abs diff = {(out - expected).abs().max().item()}"
        )


class TestAtomicAddAssemble:
    """``pl.assemble(..., atomic=AtomicType.Add)`` atomically accumulates into a shared tensor."""

    def test_split_k_matmul_atomic_add_fp32(self, test_config):
        """Split-K matmul: ``SPLIT`` parallel cores atomic-add their partials into ``c``."""
        matmul_split_k_atomic._cache.clear()
        torch.manual_seed(0)
        a = torch.randn(_SPLIT_K_M, _SPLIT_K_K, dtype=torch.float32)
        b = torch.randn(_SPLIT_K_K, _SPLIT_K_N, dtype=torch.float32)
        c = torch.zeros((_SPLIT_K_M, _SPLIT_K_N), dtype=torch.float32)
        matmul_split_k_atomic(a, b, c, config=test_config)
        expected = a @ b
        # Atomic-add accumulation order across cores is non-deterministic at
        # ULP level for floating-point, so allow a small tolerance.
        assert torch.allclose(c, expected, rtol=1e-3, atol=1e-3), (
            f"Split-K atomic-add mismatch: max diff = {(c - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
