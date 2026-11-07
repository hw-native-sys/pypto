# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 runtime tests for host-prequantized MX matmul.

A5 MX matmul takes FP8E4M3FN or FP8E5M2 on each operand independently, so all
four FP8 pairs run here against an FP32 torch golden, in the base, accumulating
and bias forms. Native packed-FP4 matmul is intentionally outside the supported
surface; the FP4xFP8 case uses packed FP4E2M1X2 carrier dimensions and
explicitly casts its lhs to FP8E4M3FN.

The same four pairs also run on the **Tensor** surface, in
``test_matmul_mx_tensor_a5.py``. That lives in its own module because CI runs it
in its own ``onboard-tests-a5`` step; both modules import their operands from
``_matmul_mx_data`` so the two surfaces are measured on the same bytes.

Exactness (``rtol=atol=0``) is deliberate and constrains the inputs: every
decoded product is a multiple of 2^-4 and every dot product stays below 2^15, so
the whole FP32 accumulation is exact and any numerical difference is a real bug
rather than rounding. Operand values beyond the other format's range (E5M2
reaches 57344, E4M3 stops at 448) and subnormals of both formats would break
that budget if they shared a dot product with ordinary values, so each lives in
its own K group paired with a compensating E8M0 scale -- which also makes the
scales vary per group, as the ISA contract requires.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig
from runtime.ops._matmul_mx_data import (
    _FP8_DTYPES,
    _MX_FP8_PAIR_VALUES,
    _ORDINARY_SCALE_CODES,
    _PL_DTYPES,
    _RHS_VALUES,
    A_COLS,
    A_OFF,
    A_ROWS,
    A_SCALE_OFF,
    B_COLS,
    B_OFF,
    B_ROWS,
    B_SCALE_OFF,
    K_GROUPS,
    MX_GROUP_SIZE,
    REQUIRED_TORCH_DTYPES,
    K,
    M,
    N,
    _decode_fp4_data,
    _exact_fp8,
    _make_inputs,
    _matmul_mx_golden,
    _pack_a_scale,
    _pack_b_scale,
    _tile_choices,
    _unpack_a_scale,
    _unpack_b_scale,
)

if not all(hasattr(torch, name) for name in REQUIRED_TORCH_DTYPES):
    pytest.skip("torch MXFP4/MXFP8/E8M0 dtypes required", allow_module_level=True)

_MX_FP8_PAIRS = [pytest.param(lhs, rhs, id=pair_id) for lhs, rhs, pair_id in _MX_FP8_PAIR_VALUES]


class MatmulMxTestCase(PTOTestCase):
    """One host-prequantized MX FP8 pair, covering base, acc and bias outputs."""

    __test__ = False

    def __init__(self, lhs_dtype: DataType, rhs_dtype: DataType):
        if lhs_dtype not in _FP8_DTYPES or rhs_dtype not in _FP8_DTYPES:
            raise ValueError(
                f"A5 MX matmul pairs FP8E4M3FN/FP8E5M2 on each side; got {lhs_dtype}, {rhs_dtype}"
            )
        super().__init__(RunConfig(rtol=0.0, atol=0.0), platform="a5")
        self._lhs_dtype = lhs_dtype
        self._rhs_dtype = rhs_dtype

    def get_name(self) -> str:
        return f"matmul_mx_{self._lhs_dtype.value}_x_{self._rhs_dtype.value}_base_acc_bias"

    def _make_inputs(self) -> tuple[torch.Tensor, ...]:
        return _make_inputs(self._lhs_dtype, self._rhs_dtype)

    def define_tensors(self) -> list[TensorSpec]:
        a, a_scale, b, b_scale, bias = self._make_inputs()
        return [
            TensorSpec("a", [A_ROWS, A_COLS], self._lhs_dtype, init_value=a),
            TensorSpec("a_scale", list(a_scale.shape), DataType.FP8E8M0, init_value=a_scale),
            TensorSpec("b", [B_ROWS, B_COLS], self._rhs_dtype, init_value=b),
            TensorSpec("b_scale", list(b_scale.shape), DataType.FP8E8M0, init_value=b_scale),
            TensorSpec("bias", [1, N], DataType.FP32, init_value=bias),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
            TensorSpec("out_acc", [M, N], DataType.FP32, is_output=True),
            TensorSpec("out_bias", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return _mx_matmul_entry(self._lhs_dtype, self._rhs_dtype).specialize()

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        base = _matmul_mx_golden(
            tensors["a"][A_OFF[0] : A_OFF[0] + M, A_OFF[1] : A_OFF[1] + K],
            _unpack_a_scale(tensors["a_scale"].view(torch.uint8))[
                A_SCALE_OFF[0] : A_SCALE_OFF[0] + M, A_SCALE_OFF[1] : A_SCALE_OFF[1] + K_GROUPS
            ],
            tensors["b"][B_OFF[0] : B_OFF[0] + K, B_OFF[1] : B_OFF[1] + N],
            _unpack_b_scale(tensors["b_scale"].view(torch.uint8))[
                B_SCALE_OFF[0] : B_SCALE_OFF[0] + K_GROUPS, B_SCALE_OFF[1] : B_SCALE_OFF[1] + N
            ],
            k_valid=K,
        )
        tensors["out"][:] = base
        tensors["out_acc"][:] = 2 * base
        tensors["out_bias"][:] = base + tensors["bias"].to(torch.float64).to(torch.float32)


def _mx_matmul_entry(lhs_dtype: DataType, rhs_dtype: DataType):
    """Build the jitted entry for one FP8 pair.

    The operand dtypes appear in the kernel's annotations, so one entry is
    specialized per pair rather than branching inside a single kernel.
    """
    lhs_pl = _PL_DTYPES[lhs_dtype]
    rhs_pl = _PL_DTYPES[rhs_dtype]

    @pl.jit.incore
    def mx_matmul_kernel(
        a: pl.Tensor[[A_ROWS, A_COLS], lhs_pl],
        a_scale: pl.Tensor[[A_ROWS, A_COLS // 32], pl.FP8E8M0, pl.MX_A_ZZ],
        b: pl.Tensor[[B_ROWS, B_COLS], rhs_pl],
        b_scale: pl.Tensor[[B_ROWS // 32, B_COLS], pl.FP8E8M0, pl.MX_B_NN],
        bias: pl.Tensor[[1, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        out_bias: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> tuple[
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
    ]:
        lhs = pl.load(a, [A_OFF[0], A_OFF[1]], [M, K])
        lhs_scale = pl.load(a_scale, [A_SCALE_OFF[0], A_SCALE_OFF[1]], [M, K_GROUPS])
        rhs = pl.load(b, [B_OFF[0], B_OFF[1]], [K, N])
        rhs_scale = pl.load(b_scale, [B_SCALE_OFF[0], B_SCALE_OFF[1]], [K_GROUPS, N])
        bias_tile = pl.load(bias, [0, 0], [1, N])
        base = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)
        out = pl.store(base, [0, 0], out)
        out_acc = pl.store(pl.matmul_mx_acc(base, lhs, lhs_scale, rhs, rhs_scale), [0, 0], out_acc)
        out_bias = pl.store(pl.matmul_mx_bias(lhs, lhs_scale, rhs, rhs_scale, bias_tile), [0, 0], out_bias)
        return out, out_acc, out_bias

    @pl.jit
    def mx_matmul(
        a: pl.Tensor[[A_ROWS, A_COLS], lhs_pl],
        a_scale: pl.Tensor[[A_ROWS, A_COLS // 32], pl.FP8E8M0, pl.MX_A_ZZ],
        b: pl.Tensor[[B_ROWS, B_COLS], rhs_pl],
        b_scale: pl.Tensor[[B_ROWS // 32, B_COLS], pl.FP8E8M0, pl.MX_B_NN],
        bias: pl.Tensor[[1, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        out_bias: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> tuple[
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
        pl.Tensor[[M, N], pl.FP32],
    ]:
        return mx_matmul_kernel(a, a_scale, b, b_scale, bias, out, out_acc, out_bias)

    return mx_matmul


# ---------------------------------------------------------------------------
# Packed FP4 lhs, cast to FP8 before the matmul
# ---------------------------------------------------------------------------
# Native FP4 x FP4 is unsupported: PTOAS emits inconsistent packed operand and
# accumulator dimensions for it. The supported form casts the packed lhs to FP8
# first, so physical K, valid K and the ceil(K/32) group count are all measured
# on the post-cast FP8 tile rather than the x2 carrier.


def _fp4_matmul_entry():
    """Jitted entry for a packed-FP4 lhs explicitly cast to FP8E4M3FN."""

    @pl.jit.incore
    def mxfp4_fp8_kernel(
        a: pl.Tensor[[A_ROWS, A_COLS // 2], pl.FP4E2M1X2],
        a_scale: pl.Tensor[[A_ROWS, A_COLS // 32], pl.FP8E8M0, pl.MX_A_ZZ],
        b: pl.Tensor[[B_ROWS, B_COLS], pl.FP8E4M3FN],
        b_scale: pl.Tensor[[B_ROWS // 32, B_COLS], pl.FP8E8M0, pl.MX_B_NN],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> tuple[pl.Tensor[[M, N], pl.FP32], pl.Tensor[[M, N], pl.FP32]]:
        # Two FP4 values per carrier byte, so the load window is half as wide
        # along K; the cast restores the logical K the matmul reduces over.
        lhs = pl.cast(pl.load(a, [A_OFF[0], A_OFF[1] // 2], [M, K // 2]), pl.FP8E4M3FN)
        lhs_scale = pl.load(a_scale, [A_SCALE_OFF[0], A_SCALE_OFF[1]], [M, K_GROUPS])
        rhs = pl.load(b, [B_OFF[0], B_OFF[1]], [K, N])
        rhs_scale = pl.load(b_scale, [B_SCALE_OFF[0], B_SCALE_OFF[1]], [K_GROUPS, N])
        base = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)
        out = pl.store(base, [0, 0], out)
        out_acc = pl.store(pl.matmul_mx_acc(base, lhs, lhs_scale, rhs, rhs_scale), [0, 0], out_acc)
        return out, out_acc

    @pl.jit
    def mxfp4_fp8_matmul(
        a: pl.Tensor[[A_ROWS, A_COLS // 2], pl.FP4E2M1X2],
        a_scale: pl.Tensor[[A_ROWS, A_COLS // 32], pl.FP8E8M0, pl.MX_A_ZZ],
        b: pl.Tensor[[B_ROWS, B_COLS], pl.FP8E4M3FN],
        b_scale: pl.Tensor[[B_ROWS // 32, B_COLS], pl.FP8E8M0, pl.MX_B_NN],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> tuple[pl.Tensor[[M, N], pl.FP32], pl.Tensor[[M, N], pl.FP32]]:
        return mxfp4_fp8_kernel(a, a_scale, b, b_scale, out, out_acc)

    return mxfp4_fp8_matmul


class MatmulMxFp4TestCase(PTOTestCase):
    """Packed FP4 lhs x FP8 rhs, with the explicit lhs cast."""

    __test__ = False

    def __init__(self):
        super().__init__(RunConfig(rtol=0.0, atol=0.0), platform="a5")

    def get_name(self) -> str:
        return "matmul_mx_fp4e2m1x2_x_fp8e4m3fn_base_acc"

    def define_tensors(self) -> list[TensorSpec]:
        generator = torch.Generator().manual_seed(101)
        # Every FP4 code appears, so a decode error cannot hide; all decoded
        # values are multiples of 0.5 and at most 6, keeping the sum exact.
        codes = torch.randint(0, 16, (A_ROWS, A_COLS), generator=generator).to(torch.uint8)
        codes.reshape(-1)[:16] = torch.arange(16, dtype=torch.uint8)
        packed = ((codes[:, 1::2] & 0x0F) << 4) | (codes[:, 0::2] & 0x0F)
        a = packed.contiguous().view(torch.float4_e2m1fn_x2)
        b = _tile_choices(_exact_fp8(_RHS_VALUES, DataType.FP8E4M3FN), (B_ROWS, B_COLS), generator)
        a_scale_codes = torch.randint(
            _ORDINARY_SCALE_CODES[0],
            _ORDINARY_SCALE_CODES[1] + 1,
            (A_ROWS, A_COLS // MX_GROUP_SIZE),
            generator=generator,
        ).to(torch.uint8)
        b_scale_codes = torch.randint(
            _ORDINARY_SCALE_CODES[0],
            _ORDINARY_SCALE_CODES[1] + 1,
            (B_ROWS // MX_GROUP_SIZE, B_COLS),
            generator=generator,
        ).to(torch.uint8)
        assert torch.unique(a_scale_codes).numel() > 1
        assert torch.unique(b_scale_codes).numel() > 1
        return [
            TensorSpec("a", [A_ROWS, A_COLS // 2], DataType.FP4E2M1X2, init_value=a),
            TensorSpec(
                "a_scale",
                [A_ROWS, A_COLS // MX_GROUP_SIZE],
                DataType.FP8E8M0,
                init_value=_pack_a_scale(a_scale_codes).view(torch.float8_e8m0fnu),
            ),
            TensorSpec("b", [B_ROWS, B_COLS], DataType.FP8E4M3FN, init_value=b),
            TensorSpec(
                "b_scale",
                [B_ROWS // MX_GROUP_SIZE, B_COLS],
                DataType.FP8E8M0,
                init_value=_pack_b_scale(b_scale_codes).view(torch.float8_e8m0fnu),
            ),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
            TensorSpec("out_acc", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return _fp4_matmul_entry().specialize()

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        a_logical = _decode_fp4_data(tensors["a"], A_ROWS, A_COLS)
        base = _matmul_mx_golden(
            a_logical[A_OFF[0] : A_OFF[0] + M, A_OFF[1] : A_OFF[1] + K],
            _unpack_a_scale(tensors["a_scale"].view(torch.uint8))[
                A_SCALE_OFF[0] : A_SCALE_OFF[0] + M, A_SCALE_OFF[1] : A_SCALE_OFF[1] + K_GROUPS
            ],
            tensors["b"][B_OFF[0] : B_OFF[0] + K, B_OFF[1] : B_OFF[1] + N],
            _unpack_b_scale(tensors["b_scale"].view(torch.uint8))[
                B_SCALE_OFF[0] : B_SCALE_OFF[0] + K_GROUPS, B_SCALE_OFF[1] : B_SCALE_OFF[1] + N
            ],
            k_valid=K,
        )
        tensors["out"][:] = base
        tensors["out_acc"][:] = 2 * base


@pytest.mark.platforms("a5")
class TestMatmulMx:
    """Numerical execution coverage for the four A5 MX FP8 pairs."""

    @pytest.mark.parametrize(("lhs_dtype", "rhs_dtype"), _MX_FP8_PAIRS)
    def test_matmul_mx_base_acc_and_bias(self, test_runner, lhs_dtype, rhs_dtype):
        result = test_runner.run(MatmulMxTestCase(lhs_dtype, rhs_dtype))
        assert result.passed, f"Test failed: {result.error}"

    def test_mxfp4_lhs_cast_to_fp8(self, test_runner):
        result = test_runner.run(MatmulMxFp4TestCase())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
