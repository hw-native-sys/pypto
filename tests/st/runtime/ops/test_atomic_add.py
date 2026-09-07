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
      into a shared output (see ``examples/advanced/01_split_k.py``).

Codegen-level coverage already exists in
``tests/ut/codegen/test_pto_codegen_ops.py`` and ``tests/ut/jit/test_split_k.py``;
this module exercises the end-to-end execution path on device/simulator.
"""

import pypto.language as pl
import pytest
import torch
from harness import st

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
# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
#
# Every tensor a case declares is staged to the device, outputs included, which
# is what these kernels need: an atomic-add accumulates onto the destination it
# is handed, so ``out`` arrives holding the test's baseline rather than zeros.
#
# The integer cases pass rtol=atol=0 to keep the bit-exact assertion they had
# before the migration; integer atomic accumulation is exact regardless of the
# order the cores land in. The bf16/fp16 goldens are stored at the output's own
# dtype, so a fp32 reference is rounded once (<=0.4% relative) before the
# comparison — far inside the tolerances these cases already carried.


def _store_case(kernel, name, x, baseline, **kwargs):
    """``out`` starts at *baseline* everywhere; the kernel atomic-adds ``x`` onto it."""
    out = torch.full(tuple(x.shape), baseline, dtype=x.dtype)
    return st.case(kernel, x, out, name=name, golden=lambda _: baseline + x.float(), **kwargs)


def _split_k_case(kernel, name, a, b, out_dtype, **kwargs):
    """Split-K matmul: SPLIT parallel cores atomic-add their partials into ``c``."""
    c = torch.zeros((_SPLIT_K_M, _SPLIT_K_N), dtype=out_dtype)
    return st.case(kernel, a, b, c, name=name, golden=lambda _: a.float() @ b.float(), **kwargs)


def _seeded(fn, *args, **kwargs):
    """Draw *fn* from a freshly seeded generator, as each original test did."""
    torch.manual_seed(0)
    return fn(*args, **kwargs)


_EXACT = {"rtol": 0.0, "atol": 0.0}


@st.cases(
    _store_case(atomic_add_store_fp32, "atomic_add_store_fp32", _seeded(torch.randn, 16, 16), 1.0),
    _store_case(
        atomic_add_store_int32,
        "atomic_add_store_int32",
        _seeded(torch.randint, -100, 100, (16, 16), dtype=torch.int32),
        5,
        **_EXACT,
    ),
    # bf16 atomic-add is A2/A3-only; codegen rejects it on A5.
    pytest.param(
        _store_case(
            atomic_add_store_bf16,
            "atomic_add_store_bf16",
            _seeded(torch.randn, 16, 16, dtype=torch.bfloat16),
            1.0,
            rtol=2e-2,
            atol=2e-2,
        ),
        marks=pytest.mark.platforms("a2a3", "a2a3sim", reason="bf16 atomic-add is A2/A3-only"),
    ),
    _store_case(
        atomic_add_store_fp16,
        "atomic_add_store_fp16",
        _seeded(torch.randn, 16, 16, dtype=torch.float16),
        1.0,
        rtol=5e-3,
        atol=5e-3,
    ),
    _store_case(
        atomic_add_store_int16,
        "atomic_add_store_int16",
        _seeded(torch.randint, -100, 100, (16, 16), dtype=torch.int16),
        5,
        **_EXACT,
    ),
    # int8 needs a 32-col tile: the row byte size (cols * 1) must be 32-byte
    # aligned, so 16 cols is rejected by ptoas. Values stay small so the int8
    # accumulation cannot overflow.
    _store_case(
        atomic_add_store_int8,
        "atomic_add_store_int8",
        _seeded(torch.randint, -20, 20, (16, 32), dtype=torch.int8),
        1,
        **_EXACT,
    ),
)
def test_atomic_add_store(case_run):
    """``pl.store(..., atomic=AtomicType.Add)`` accumulates a tile onto a baseline."""
    case_run.assert_passed()


@st.cases(
    # Atomic-add accumulation order across cores is non-deterministic at ULP
    # level for floating point, hence the loosened tolerance.
    _split_k_case(
        matmul_split_k_atomic,
        "split_k_atomic_fp32",
        _seeded(torch.randn, _SPLIT_K_M, _SPLIT_K_K),
        torch.randn(_SPLIT_K_K, _SPLIT_K_N),
        torch.float32,
        rtol=1e-3,
        atol=1e-3,
    ),
    # Inputs are scaled down so the accumulated magnitude stays O(1) — bf16 has
    # ~2-3 decimal digits, so large sums would exceed a sane tolerance.
    pytest.param(
        _split_k_case(
            matmul_split_k_atomic_bf16,
            "split_k_atomic_bf16",
            (_seeded(torch.randn, _SPLIT_K_M, _SPLIT_K_K) * 0.05).bfloat16(),
            (torch.randn(_SPLIT_K_K, _SPLIT_K_N) * 0.05).bfloat16(),
            torch.bfloat16,
            rtol=5e-2,
            atol=5e-2,
        ),
        marks=pytest.mark.platforms("a2a3", "a2a3sim", reason="bf16 atomic-add is A2/A3-only"),
    ),
    _split_k_case(
        matmul_split_k_atomic_int32,
        "split_k_atomic_int32",
        _seeded(torch.randint, -4, 4, (_SPLIT_K_M, _SPLIT_K_K), dtype=torch.int8),
        torch.randint(-4, 4, (_SPLIT_K_K, _SPLIT_K_N), dtype=torch.int8),
        torch.int32,
        **_EXACT,
    ),
    _split_k_case(
        matmul_split_k_atomic_fp16,
        "split_k_atomic_fp16",
        (_seeded(torch.randn, _SPLIT_K_M, _SPLIT_K_K) * 0.1).half(),
        (torch.randn(_SPLIT_K_K, _SPLIT_K_N) * 0.1).half(),
        torch.float16,
        rtol=2e-2,
        atol=2e-2,
    ),
)
def test_split_k_matmul_atomic_add(case_run):
    """``pl.assemble(..., atomic=AtomicType.Add)`` accumulates into a shared tensor."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
