# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end split-K matmul: a large-K reduction parallelised across cores.

Each core multiplies an [M, KS] x [KS, N] K-slice and atomically adds its
partial product into one global-memory output via
``pl.assemble(..., atomic=pl.AtomicType.Add)``. The output is zero-initialised
in-kernel before the parallel loop. This test compiles the pattern through the
full pass pipeline and verifies the per-core kernel emits an atomic-add store.

Mirrors ``examples/kernels/10_split_k.py``.
"""

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir
from pypto.backend import BackendType
from pypto.debug import torch_codegen
from pypto.jit.decorator import jit

# Module-level constants — the JIT specializer inlines module-level ints.
_M = 64
_N = 64
_K = 512
_SPLIT = 4  # K reduction spread across 4 cores
_KS = _K // _SPLIT  # per-core K-slice width

# Down-projection-pattern constants (mirrors qwen3_decode's down_projection
# kernel: split-K matmul into an fp32 accumulator, then residual + bf16 cast).
_DM = 16
_DN = 64
_DK = 512
_DSPLIT = 4
_DKS = _DK // _DSPLIT


@pytest.fixture(autouse=True)
def _setup_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _split_k_program():
    @jit
    def matmul_split_k(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="zero_init"):
            c = pl.assemble(c, pl.full([_M, _N], dtype=pl.FP32, value=0.0), [0, 0])
        for ks in pl.parallel(0, _SPLIT):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="split_k"):
                k0 = ks * _KS
                a_k = a[:, k0 : k0 + _KS]
                b_k = b[k0 : k0 + _KS, :]
                partial = pl.matmul(a_k, b_k, out_dtype=pl.FP32)
                c = pl.assemble(c, partial, [0, 0], atomic=pl.AtomicType.Add)
        return c

    return matmul_split_k


def test_split_k_matmul_compiles():
    """The split-K matmul compiles through the full pipeline into an entry + per-core kernel."""
    torch = pytest.importorskip("torch")
    post = _split_k_program().compile_for_test(torch.randn(_M, _K), torch.randn(_K, _N), torch.empty(_M, _N))
    func_types = {f.func_type for f in post.functions.values()}
    assert ir.FunctionType.Orchestration in func_types, f"expected an Orchestration entry, got {func_types}"
    assert any(ir.is_incore_type(f.func_type) for f in post.functions.values()), (
        f"expected an InCore-variant per-core kernel, got {func_types}"
    )


def test_split_k_matmul_emits_atomic_add_store():
    """The per-core kernel accumulates its partial product with an atomic-add store."""
    torch = pytest.importorskip("torch")
    post = _split_k_program().compile_for_test(torch.randn(_M, _K), torch.randn(_K, _N), torch.empty(_M, _N))
    incore = next(f for f in post.functions.values() if ir.is_incore_type(f.func_type))
    mlir = codegen.PTOCodegen().generate(ir.Program([incore], incore.name, post.span))

    tstore_lines = [line.strip() for line in mlir.splitlines() if "pto.tstore" in line]
    assert tstore_lines, f"no pto.tstore emitted by the split-K kernel:\n{mlir}"
    atomic_lines = [line for line in tstore_lines if "{atomicType = #pto<atomic_type atomic_add>}" in line]
    assert atomic_lines, f"split-K partial product must be stored with atomic-add, got:\n{tstore_lines}"
    # The partial is a fp32 matmul accumulator stored straight to GM — the cube
    # (loc=acc) fix-pipe atomic-add store on the AIC kernel.
    assert all("loc=acc" in line for line in atomic_lines), (
        f"split-K atomic store must be a cube accumulator (loc=acc) store, got:\n{atomic_lines}"
    )
    assert "pto.tmatmul" in mlir, f"expected a matmul in the per-core kernel:\n{mlir}"


def _split_k_bf16_program():
    """Split-K matmul accumulating directly into a bf16 output (no fp32 scratch).

    Written exactly like the fp32 ``_split_k_program`` but with a bf16 output ``c``:
    each core's fp32 matmul accumulator is atomic-added straight into the bf16 GM
    target, letting the fix-pipe down-convert (fp32 Acc -> bf16 GM). This is the
    direct bf16 atomic-add form enabled on A2/A3, replacing the
    ``down_proj_split_k`` fp32-accumulator-then-cast workaround.
    """

    @jit
    def matmul_split_k_bf16(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="zero_init"):
            c = pl.assemble(c, pl.full([_M, _N], dtype=pl.BF16, value=0.0), [0, 0])
        for ks in pl.parallel(0, _SPLIT):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="split_k"):
                k0 = ks * _KS
                a_k = a[:, k0 : k0 + _KS]
                b_k = b[k0 : k0 + _KS, :]
                partial = pl.matmul(a_k, b_k, out_dtype=pl.FP32)
                c = pl.assemble(c, partial, [0, 0], atomic=pl.AtomicType.Add)
        return c

    return matmul_split_k_bf16


def test_split_k_bf16_direct_emits_atomic_add_store():
    """Direct-bf16 split-K emits a CUBE-unit bf16 atomic-add store (fix-pipe down-convert).

    The fp32 matmul accumulator is atomic-added straight into the bf16 GM output.
    The atomic-add ``pto.tstore`` lands on the cube (AIC) kernel with a
    ``loc=acc, dtype=f32`` source tile and a bf16 destination view — the fix-pipe
    Acc->GM path lowered via set_atomic_bf16. This is the true cube-unit bf16
    atomic-add, and it lets bf16 split-K be written exactly like the fp32 form
    (previously this required an fp32 scratch + explicit cast).
    """
    torch = pytest.importorskip("torch")
    post = _split_k_bf16_program().compile_for_test(
        torch.randn(_M, _K, dtype=torch.bfloat16),
        torch.randn(_K, _N, dtype=torch.bfloat16),
        torch.empty(_M, _N, dtype=torch.bfloat16),
    )
    incore = [f for f in post.functions.values() if ir.is_incore_type(f.func_type)]
    assert incore, "expected at least one InCore kernel"
    mlir = "\n".join(codegen.PTOCodegen().generate(ir.Program([f], f.name, post.span)) for f in incore)

    tstore_lines = [line.strip() for line in mlir.splitlines() if "pto.tstore" in line]
    assert tstore_lines, f"no pto.tstore emitted by the bf16 split-K kernels:\n{mlir}"
    # The split-K partial is an accumulator (loc=acc) atomic-added into the bf16 GM
    # target — the cube fix-pipe path. Its destination partition view is bf16.
    atomic_acc_stores = [
        line
        for line in tstore_lines
        if "{atomicType = #pto<atomic_type atomic_add>}" in line and "loc=acc" in line
    ]
    assert atomic_acc_stores, (
        f"bf16 split-K must lower to a cube (loc=acc) atomic-add store, got:\n{tstore_lines}"
    )
    assert all("xbf16>" in line for line in atomic_acc_stores), (
        f"the cube atomic-add store must target a bf16 GM view, got:\n{atomic_acc_stores}"
    )
    assert "pto.tmatmul" in mlir, f"expected a cube matmul across the InCore kernels:\n{mlir}"


def _split_k_fp16_program():
    """Split-K matmul accumulating directly into a fp16 output (fp32 Acc -> fp16 GM).

    Like the bf16 variant but with a fp16 output: each core's fp32 matmul
    accumulator is atomic-added straight into the fp16 GM target via the fix-pipe
    (half is in the Acc->GM whitelist), lowered through set_atomic_f16.
    """

    @jit
    def matmul_split_k_fp16(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="zero_init"):
            c = pl.assemble(c, pl.full([_M, _N], dtype=pl.FP16, value=0.0), [0, 0])
        for ks in pl.parallel(0, _SPLIT):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="split_k"):
                k0 = ks * _KS
                a_k = a[:, k0 : k0 + _KS]
                b_k = b[k0 : k0 + _KS, :]
                partial = pl.matmul(a_k, b_k, out_dtype=pl.FP32)
                c = pl.assemble(c, partial, [0, 0], atomic=pl.AtomicType.Add)
        return c

    return matmul_split_k_fp16


def test_split_k_fp16_direct_emits_atomic_add_store():
    """Direct-fp16 split-K emits a CUBE-unit fp16 atomic-add store (set_atomic_f16).

    An fp32 matmul accumulator atomic-added straight into a fp16 GM output — the
    cube fix-pipe path (half is a legal Acc->GM destination dtype).
    """
    torch = pytest.importorskip("torch")
    post = _split_k_fp16_program().compile_for_test(
        torch.randn(_M, _K, dtype=torch.float16),
        torch.randn(_K, _N, dtype=torch.float16),
        torch.empty(_M, _N, dtype=torch.float16),
    )
    incore = [f for f in post.functions.values() if ir.is_incore_type(f.func_type)]
    assert incore, "expected at least one InCore kernel"
    mlir = "\n".join(codegen.PTOCodegen().generate(ir.Program([f], f.name, post.span)) for f in incore)

    tstore_lines = [line.strip() for line in mlir.splitlines() if "pto.tstore" in line]
    atomic_acc = [
        line
        for line in tstore_lines
        if "{atomicType = #pto<atomic_type atomic_add>}" in line and "loc=acc" in line
    ]
    assert atomic_acc, f"fp16 split-K must lower to a cube (loc=acc) atomic-add store, got:\n{tstore_lines}"
    assert all("xf16>" in line for line in atomic_acc), (
        f"the cube atomic-add store must target a fp16 GM view, got:\n{atomic_acc}"
    )
    assert "pto.tmatmul" in mlir, f"expected a cube matmul across the InCore kernels:\n{mlir}"


def _split_k_int32_program():
    """Split-K matmul accumulating int32 partials (int8 x int8 -> int32 Acc).

    Each core's int8 matmul produces an int32 accumulator that is atomic-added
    directly into the int32 GM output — the cube (loc=acc) int32 atomic-add
    (pto-isa set_atomic_s32) path.
    """

    @jit
    def matmul_split_k_int32(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="zero_init"):
            c = pl.assemble(c, pl.full([_M, _N], dtype=pl.INT32, value=0), [0, 0])
        for ks in pl.parallel(0, _SPLIT):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="split_k"):
                k0 = ks * _KS
                a_k = a[:, k0 : k0 + _KS]
                b_k = b[k0 : k0 + _KS, :]
                partial = pl.matmul(a_k, b_k, out_dtype=pl.INT32)
                c = pl.assemble(c, partial, [0, 0], atomic=pl.AtomicType.Add)
        return c

    return matmul_split_k_int32


def test_split_k_int32_emits_atomic_add_store():
    """int8-matmul split-K emits a CUBE-unit int32 atomic-add store (set_atomic_s32).

    An int8 x int8 matmul yields an int32 accumulator (matmul.cpp defaults
    non-float inputs to int32), atomic-added straight into the int32 GM output.
    The atomic-add ``pto.tstore`` lands on the cube (AIC) kernel with a
    ``loc=acc, dtype=i32`` source tile.
    """
    torch = pytest.importorskip("torch")
    post = _split_k_int32_program().compile_for_test(
        torch.randint(-4, 4, (_M, _K), dtype=torch.int8),
        torch.randint(-4, 4, (_K, _N), dtype=torch.int8),
        torch.zeros(_M, _N, dtype=torch.int32),
    )
    incore = [f for f in post.functions.values() if ir.is_incore_type(f.func_type)]
    assert incore, "expected at least one InCore kernel"
    mlir = "\n".join(codegen.PTOCodegen().generate(ir.Program([f], f.name, post.span)) for f in incore)

    tstore_lines = [line.strip() for line in mlir.splitlines() if "pto.tstore" in line]
    atomic_acc = [
        line
        for line in tstore_lines
        if "{atomicType = #pto<atomic_type atomic_add>}" in line and "loc=acc" in line
    ]
    assert atomic_acc, f"int32 split-K must lower to a cube (loc=acc) atomic-add store, got:\n{tstore_lines}"
    assert all("dtype=i32" in line for line in atomic_acc), (
        f"the cube atomic-add store must be int32, got:\n{atomic_acc}"
    )
    assert "pto.tmatmul" in mlir, f"expected a cube matmul across the InCore kernels:\n{mlir}"


def test_split_k_matmul_numerically_correct():
    """Executing the split-K matmul (via torch_codegen) matches torch.matmul.

    Drives the lowered IR through torch_codegen — which honours the atomic-add
    store as an accumulate — so the per-core partial products sum to the full
    product. This validates split-K end to end without the device toolchain.
    """
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    a = torch.randn(_M, _K, dtype=torch.float32)
    b = torch.randn(_K, _N, dtype=torch.float32)
    c = torch.zeros(_M, _N, dtype=torch.float32)

    post = _split_k_program().compile_for_test(a, b, c)
    code = torch_codegen(post)
    ns: dict = {}
    exec(code, ns)  # noqa: S102 — executing generated reference code is the point

    out = c.clone()
    ns["matmul_split_k"](a, b, out)
    expected = torch.matmul(a, b)
    assert torch.allclose(out, expected, rtol=1e-3, atol=1e-3), (
        f"split-K result mismatch: max abs diff {(expected - out).abs().max().item():.3e}"
    )


def _down_proj_split_k_program():
    @jit
    def down_proj_split_k(mlp: pl.Tensor, w_down: pl.Tensor, resid: pl.Tensor, out: pl.Out[pl.Tensor]):
        # fp32 GM accumulator for the split-K partials — atomic-add needs an
        # fp32 target, and `out` is bf16. Zero-initialised before the loop.
        acc = pl.create_tensor([_DM, _DN], dtype=pl.FP32)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="dp_zero_init"):
            acc = pl.assemble(acc, pl.full([_DM, _DN], dtype=pl.FP32, value=0.0), [0, 0])
        for ks in pl.parallel(0, _DSPLIT):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="dp_split_k"):
                k0 = ks * _DKS
                mlp_k = mlp[:, k0 : k0 + _DKS]
                w_k = w_down[k0 : k0 + _DKS, :]
                part = pl.matmul(mlp_k, w_k, out_dtype=pl.FP32)
                acc = pl.assemble(acc, part, [0, 0], atomic=pl.AtomicType.Add)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="dp_residual"):
            out = pl.assemble(out, pl.cast(pl.add(acc, resid), target_type=pl.BF16), [0, 0])
        return out

    return down_proj_split_k


def test_split_k_down_projection_pattern_numerically_correct():
    """A qwen3-style down projection (split-K matmul + residual + bf16 cast) is correct.

    This is the kernel shape rewritten in ``qwen3_decode_split_k.py``: a split-K
    reduction accumulating into an fp32 global-memory tensor, finalised by adding
    the residual and casting to bf16.
    """
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    mlp = torch.randn(_DM, _DK, dtype=torch.float32)
    w_down = torch.randn(_DK, _DN, dtype=torch.float32)
    resid = torch.randn(_DM, _DN, dtype=torch.float32)
    out = torch.zeros(_DM, _DN, dtype=torch.bfloat16)

    post = _down_proj_split_k_program().compile_for_test(mlp, w_down, resid, out)
    code = torch_codegen(post)
    ns: dict = {}
    exec(code, ns)  # noqa: S102 — executing generated reference code is the point

    actual = out.clone()
    ns["down_proj_split_k"](mlp, w_down, resid, actual)
    expected = (torch.matmul(mlp, w_down) + resid).bfloat16()
    assert torch.allclose(actual.float(), expected.float(), rtol=2e-2, atol=2e-2), (
        f"down-proj split-K mismatch: max abs diff "
        f"{(actual.float() - expected.float()).abs().max().item():.3e}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
