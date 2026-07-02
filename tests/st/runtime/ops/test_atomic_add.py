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


@pl.jit
def atomic_add_store_bf16(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    """BF16 variant of :func:`atomic_add_store_fp32` — a VECTOR-unit UB->GM atomic-add.

    A plain loaded Vec tile is atomic-added into a bf16 GM tensor. On A2/A3 this
    lowers to set_atomic_bf16() on the MTE3 store pipe. bf16 atomic-add is not
    supported on A5, so this kernel targets the Ascend910B profile.
    """
    with pl.at(level=pl.Level.CORE_GROUP):
        x_tile = pl.load(x, [0, 0], [16, 16])
        pl.store(x_tile, [0, 0], out, atomic=pl.AtomicType.Add)
    return out


@pl.jit
def atomic_add_store_fp16(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    """FP16 VECTOR-unit UB->GM atomic-add (set_atomic_f16)."""
    with pl.at(level=pl.Level.CORE_GROUP):
        x_tile = pl.load(x, [0, 0], [16, 16])
        pl.store(x_tile, [0, 0], out, atomic=pl.AtomicType.Add)
    return out


@pl.jit
def atomic_add_store_int16(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    """INT16 VECTOR-unit UB->GM atomic-add (set_atomic_s16)."""
    with pl.at(level=pl.Level.CORE_GROUP):
        x_tile = pl.load(x, [0, 0], [16, 16])
        pl.store(x_tile, [0, 0], out, atomic=pl.AtomicType.Add)
    return out


@pl.jit
def atomic_add_store_int8(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    """INT8 VECTOR-unit UB->GM atomic-add (set_atomic_s8).

    Uses a 32-col tile: for int8 the tile row byte size (cols * 1) must be
    32-byte aligned, so 16 cols (16 bytes) is rejected by ptoas.
    """
    with pl.at(level=pl.Level.CORE_GROUP):
        x_tile = pl.load(x, [0, 0], [16, 32])
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


@pl.jit
def matmul_split_k_atomic_bf16(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    """BF16 split-K matmul — CUBE fix-pipe atomic-add straight into bf16 GM.

    Written exactly like :func:`matmul_split_k_atomic` but with a bf16 output: each
    core's fp32 matmul accumulator is atomic-added directly into the shared bf16
    output ``c``, letting the fix-pipe down-convert (fp32 Acc -> bf16 GM). This is
    the direct-bf16-accumulation form enabled on A2/A3 (set_atomic_bf16), avoiding
    the fp32-scratch-then-cast workaround.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="zero_init"):
        c = pl.assemble(c, pl.full([_SPLIT_K_M, _SPLIT_K_N], dtype=pl.BF16, value=0.0), [0, 0])
    for ks in pl.parallel(0, _SPLIT_K_SPLITS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="split_k"):
            k0 = ks * _SPLIT_K_KS
            a_k = a[:, k0 : k0 + _SPLIT_K_KS]
            b_k = b[k0 : k0 + _SPLIT_K_KS, :]
            partial = pl.matmul(a_k, b_k, out_dtype=pl.FP32)
            c = pl.assemble(c, partial, [0, 0], atomic=pl.AtomicType.Add)
    return c


@pl.jit
def matmul_split_k_atomic_int32(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    """INT32 split-K matmul — CUBE int32 atomic-add (int8 x int8 -> int32 Acc).

    Each core's int8 x int8 matmul yields an int32 accumulator (matmul defaults
    non-float inputs to int32) that is atomic-added directly into the shared int32
    output ``c`` (pto-isa set_atomic_s32). Integer accumulation is exact.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="zero_init"):
        c = pl.assemble(c, pl.full([_SPLIT_K_M, _SPLIT_K_N], dtype=pl.INT32, value=0), [0, 0])
    for ks in pl.parallel(0, _SPLIT_K_SPLITS):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="split_k"):
            k0 = ks * _SPLIT_K_KS
            a_k = a[:, k0 : k0 + _SPLIT_K_KS]
            b_k = b[k0 : k0 + _SPLIT_K_KS, :]
            partial = pl.matmul(a_k, b_k, out_dtype=pl.INT32)
            c = pl.assemble(c, partial, [0, 0], atomic=pl.AtomicType.Add)
    return c


@pl.jit
def matmul_split_k_atomic_fp16(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    """FP16 split-K matmul — CUBE fix-pipe atomic-add straight into fp16 GM.

    Each core's fp32 matmul accumulator is atomic-added directly into the shared
    fp16 output ``c`` (set_atomic_f16); half is a legal Acc->GM destination dtype.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="zero_init"):
        c = pl.assemble(c, pl.full([_SPLIT_K_M, _SPLIT_K_N], dtype=pl.FP16, value=0.0), [0, 0])
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

    @pytest.mark.platforms("a2a3", "a2a3sim")
    def test_atomic_add_store_bf16(self, test_config):
        """BF16 (A2/A3, VECTOR path): ``out`` starts at 1.0; kernel atomic-adds ``x``.

        bf16 atomic-add is A2/A3-only; on A5 it is rejected in codegen, so this
        test is restricted to the a2a3 platforms.

        Uses a loose tolerance because bf16 has ~8 mantissa bits — the single
        accumulation ``1.0 + x`` is exact up to bf16 rounding of the operands.
        """
        atomic_add_store_bf16._cache.clear()
        torch.manual_seed(0)
        x = torch.randn(16, 16, dtype=torch.bfloat16)
        baseline = 1.0
        out = torch.full((16, 16), baseline, dtype=torch.bfloat16)
        atomic_add_store_bf16(x, out, config=test_config)
        expected = (baseline + x.float()).bfloat16()
        diff = (out.float() - expected.float()).abs().max().item()
        assert torch.allclose(out.float(), expected.float(), rtol=2e-2, atol=2e-2), (
            f"BF16 atomic-add store mismatch: max diff = {diff}"
        )

    def test_atomic_add_store_fp16(self, test_config):
        """FP16 (VECTOR path): ``out`` starts at 1.0; kernel atomic-adds ``x`` (set_atomic_f16)."""
        atomic_add_store_fp16._cache.clear()
        torch.manual_seed(0)
        x = torch.randn(16, 16, dtype=torch.float16)
        baseline = 1.0
        out = torch.full((16, 16), baseline, dtype=torch.float16)
        atomic_add_store_fp16(x, out, config=test_config)
        expected = (baseline + x.float()).half()
        diff = (out.float() - expected.float()).abs().max().item()
        assert torch.allclose(out.float(), expected.float(), rtol=5e-3, atol=5e-3), (
            f"FP16 atomic-add store mismatch: max diff = {diff}"
        )

    def test_atomic_add_store_int16(self, test_config):
        """INT16 (VECTOR path): ``out`` starts at 5; kernel atomic-adds ``x`` (set_atomic_s16); exact."""
        atomic_add_store_int16._cache.clear()
        torch.manual_seed(0)
        x = torch.randint(-100, 100, (16, 16), dtype=torch.int16)
        baseline = 5
        out = torch.full((16, 16), baseline, dtype=torch.int16)
        atomic_add_store_int16(x, out, config=test_config)
        expected = baseline + x
        assert torch.equal(out, expected), (
            f"INT16 atomic-add store mismatch: max abs diff = {(out - expected).abs().max().item()}"
        )

    def test_atomic_add_store_int8(self, test_config):
        """INT8 (VECTOR path): ``out`` starts at 1; kernel atomic-adds ``x`` (set_atomic_s8); exact.

        Values are kept small so the int8 accumulation cannot overflow.
        """
        atomic_add_store_int8._cache.clear()
        torch.manual_seed(0)
        x = torch.randint(-20, 20, (16, 32), dtype=torch.int8)
        baseline = 1
        out = torch.full((16, 32), baseline, dtype=torch.int8)
        atomic_add_store_int8(x, out, config=test_config)
        expected = baseline + x
        assert torch.equal(out, expected), (
            f"INT8 atomic-add store mismatch: max abs diff = {(out - expected).abs().max().item()}"
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

    @pytest.mark.platforms("a2a3", "a2a3sim")
    def test_split_k_matmul_atomic_add_bf16(self, test_config):
        """BF16 (A2/A3, CUBE path): parallel cores atomic-add bf16 partials into ``c``.

        bf16 atomic-add is A2/A3-only (rejected in codegen on A5), so this test is
        restricted to the a2a3 platforms.

        Each core's fp32 accumulator is cast to bf16 and atomic-added directly into
        the shared bf16 output (set_atomic_bf16). Inputs are scaled down so the
        accumulated magnitude stays O(1), keeping bf16 rounding within tolerance.
        """
        matmul_split_k_atomic_bf16._cache.clear()
        torch.manual_seed(0)
        # Scale so partial/accumulated magnitudes are small — bf16 has ~2-3
        # decimal digits, so large sums would exceed a sane tolerance.
        a = (torch.randn(_SPLIT_K_M, _SPLIT_K_K) * 0.05).bfloat16()
        b = (torch.randn(_SPLIT_K_K, _SPLIT_K_N) * 0.05).bfloat16()
        c = torch.zeros((_SPLIT_K_M, _SPLIT_K_N), dtype=torch.bfloat16)
        matmul_split_k_atomic_bf16(a, b, c, config=test_config)
        expected = a.float() @ b.float()
        assert torch.allclose(c.float(), expected, rtol=5e-2, atol=5e-2), (
            f"BF16 split-K atomic-add mismatch: max diff = {(c.float() - expected).abs().max().item()}"
        )

    def test_split_k_matmul_atomic_add_int32(self, test_config):
        """INT32 (CUBE): int8 x int8 -> int32 partials atomic-added into ``c``; exact.

        Integer atomic accumulation is exact regardless of core-ordering, so this
        asserts bit-exact equality against the reference int32 matmul.
        """
        matmul_split_k_atomic_int32._cache.clear()
        torch.manual_seed(0)
        a = torch.randint(-4, 4, (_SPLIT_K_M, _SPLIT_K_K), dtype=torch.int8)
        b = torch.randint(-4, 4, (_SPLIT_K_K, _SPLIT_K_N), dtype=torch.int8)
        c = torch.zeros((_SPLIT_K_M, _SPLIT_K_N), dtype=torch.int32)
        matmul_split_k_atomic_int32(a, b, c, config=test_config)
        expected = a.to(torch.int32) @ b.to(torch.int32)
        assert torch.equal(c, expected), (
            f"INT32 split-K atomic-add mismatch: max abs diff = {(c - expected).abs().max().item()}"
        )

    def test_split_k_matmul_atomic_add_fp16(self, test_config):
        """FP16 (CUBE path): fp32 partials atomic-added directly into fp16 ``c`` (set_atomic_f16).

        Inputs are scaled down so the fp16 accumulated magnitude stays O(1).
        """
        matmul_split_k_atomic_fp16._cache.clear()
        torch.manual_seed(0)
        a = (torch.randn(_SPLIT_K_M, _SPLIT_K_K) * 0.1).half()
        b = (torch.randn(_SPLIT_K_K, _SPLIT_K_N) * 0.1).half()
        c = torch.zeros((_SPLIT_K_M, _SPLIT_K_N), dtype=torch.float16)
        matmul_split_k_atomic_fp16(a, b, c, config=test_config)
        expected = a.float() @ b.float()
        assert torch.allclose(c.float(), expected, rtol=2e-2, atol=2e-2), (
            f"FP16 split-K atomic-add mismatch: max diff = {(c.float() - expected).abs().max().item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
