# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 device coverage for ordinary matmul over mixed FP8 operand pairs.

PTOAS v0.72 accepts any (E4M3FN | E5M2) operand pair with an f32 destination on
A5 -- including the two mixed orders -- for ``pto.tmatmul``, ``.acc`` and
``.bias`` (``verifyMatmulTypeTriple``; pto-isa ``npu/a5/TMatmul.hpp``). These
cases run all three Tile forms and both Tensor forms on device.

Two operand tables divide the work, because one table cannot do both jobs
without making the reference ill-conditioned:

- a **dyadic** table (multiples of 2**-3, magnitudes up to 6) drives the matmul
  cases. Every product is a multiple of 2**-6 and a K-long sum stays well under
  2**24 of them, so the FP32 reference is exact whatever order the Cube
  accumulates in -- every case here asserts rtol=atol=0.
- an **extremes** table (each format's smallest subnormal, a subnormal that uses
  the mantissa bits, the smallest normal, 1.0, and the format maximum, both
  signs) drives a separate ``A @ I`` case. The identity rhs makes each output
  element one decoded operand value, so no summation error can arise. That is
  what pins the decode: the two formats disagree on the value of nearly every
  byte pattern, so reading an E5M2 byte as E4M3FN (or the reverse) cannot pass.

Mixing the two tables into one sum would be a worse test, not a better one: a
K-long sum of terms spanning 2**-32 to 2**8 cancels catastrophically, and the
tiny residual left over is an artifact of accumulation order rather than a
property of the instruction. (Measured on device while writing these: products
of subnormal FP8 operands are *not* flushed -- a uniform 2**-16 x 2**-16 matmul
is bit-exact -- so the residual really is cancellation, not underflow.)

Physical K is a multiple of 32 throughout: a 1-byte Left/Right fractal packs 32
bytes along K and PTOAS rejects an FP8 operand whose physical K is not. Valid K
carries no such restriction, which the tail case exercises.

E5M2 host tensors reach the device through the runtime's UINT8 byte carrier
(``pypto.runtime.tensor_arg.runtime_host_tensor``); ``test_fp8e5m2_transfer.py``
covers that path on its own.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

if not hasattr(torch, "float8_e5m2"):
    pytest.skip("torch.float8_e5m2 required", allow_module_level=True)

M, K, N = 64, 64, 64
VALID_K = 40

# Multiples of 2**-3, exact in both FP8 formats. Products are multiples of 2**-6
# bounded by 36, so a K=64 sum needs at most 64*36*64 = 147456 < 2**24 units of
# 2**-6: exactly representable in FP32, hence independent of summation order.
_DYADIC_MAGNITUDES = [0.0, 0.25, 0.375, 1.0, 1.5, 3.0, 6.0]

# The byte patterns on which the two formats disagree. Deliberately **no zero**:
# the golden validator compares with ``torch.isclose``, and ``isclose(-0.0,
# +0.0)`` is True at any tolerance, so a zero entry would be the one value whose
# comparison proves nothing. Every entry here is finite and non-zero, which makes
# rtol=atol=0 exact equality for the whole table.
_EXTREME_MAGNITUDES = {
    DataType.FP8E4M3FN: [2.0**-9, 3 * 2.0**-9, 2.0**-8, 2.0**-6, 1.0, 448.0],
    DataType.FP8E5M2: [2.0**-16, 3 * 2.0**-16, 2.0**-15, 2.0**-14, 1.0, 57344.0],
}

_SMALLEST_NORMAL = {DataType.FP8E4M3FN: 2.0**-6, DataType.FP8E5M2: 2.0**-14}

# Both tables keep the FP32 reference exact, so every case demands equality.
#
# The dyadic table does carry a zero, so a matmul case whose sum happens to be
# exactly zero is compared sign-blind (see above). That costs nothing: the matmul
# cases assert the value of a sum, never the sign of a zero. The identity case,
# which is the one that claims per-byte decode fidelity, uses the zero-free
# extremes table precisely so its claim holds without that caveat.
_RUN_CONFIG = RunConfig(rtol=0.0, atol=0.0)


def _signed(magnitudes: list[float]) -> torch.Tensor:
    return torch.tensor([s * m for m in magnitudes for s in (1.0, -1.0)], dtype=torch.float32)


def _fill_from_table(values: torch.Tensor, rows: int, cols: int, dtype: DataType, seed: int):
    """Tile `values` over a [rows, cols] operand, visiting every table entry.

    Stride 5 is coprime with both table sizes (14 dyadic, 12 extreme entries), so
    the walk covers the whole table. An even stride would hold the sign parity
    fixed and silently yield a single-signed operand.
    """
    assert values.numel() % 5 != 0, "stride 5 must be coprime with the table size"
    index = (torch.arange(rows * cols) * 5 + seed) % values.numel()
    exact = values[index].reshape(rows, cols)
    data = exact.to(dtype.torch_dtype)
    # The table must survive the round trip, or the case would be about rounding
    # rather than about the operand pair.
    torch.testing.assert_close(data.float(), exact, rtol=0, atol=0)
    assert (data.float() > 0).any() and (data.float() < 0).any(), "operand must cover both signs"
    return data


def _dyadic_operand(rows: int, cols: int, dtype: DataType, seed: int) -> torch.Tensor:
    """Well-conditioned operand for the matmul cases."""
    return _fill_from_table(_signed(_DYADIC_MAGNITUDES), rows, cols, dtype, seed)


def _extreme_operand(rows: int, cols: int, dtype: DataType, seed: int) -> torch.Tensor:
    """Operand spanning the format's full range, for the identity-decode case."""
    data = _fill_from_table(_signed(_EXTREME_MAGNITUDES[dtype]), rows, cols, dtype, seed)
    decoded = data.float()
    nonzero = decoded.abs().masked_fill(decoded == 0, float("inf"))
    assert (nonzero < _SMALLEST_NORMAL[dtype]).any(), "operand must cover subnormals"
    return data


def _pl_dtype(dtype: DataType):
    return pl.FP8E4M3FN if dtype == DataType.FP8E4M3FN else pl.FP8E5M2


def _make_tile_entry(lhs_dtype: DataType, rhs_dtype: DataType):
    """`tile.matmul` / `_acc` / `_bias` over one operand pair, in one kernel.

    Built here rather than written out once per pair because only the two operand
    dtypes vary; they are closure constants the DSL parser resolves while tracing
    the annotations.
    """
    lhs_pl = _pl_dtype(lhs_dtype)
    rhs_pl = _pl_dtype(rhs_dtype)

    @pl.jit.incore
    def tile_kernel(
        a: pl.Tensor[[M, K], lhs_pl],
        b: pl.Tensor[[K, N], rhs_pl],
        bias: pl.Tensor[[1, N], pl.FP32],
        out_base: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        out_bias: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ):
        lhs = pl.load(a, [0, 0], [M, K], target_memory=pl.MemorySpace.Mat)
        rhs = pl.load(b, [0, 0], [K, N], target_memory=pl.MemorySpace.Mat)
        bias_tile = pl.load(bias, [0, 0], [1, N], target_memory=pl.MemorySpace.Mat)
        base = pl.tile.matmul(lhs, rhs)
        out_base = pl.store(base, [0, 0], out_base)
        # acc seeds from the FP32 product, so the expected result is exactly 2x base.
        accumulated = pl.tile.matmul_acc(base, lhs, rhs)
        out_acc = pl.store(accumulated, [0, 0], out_acc)
        biased = pl.tile.matmul_bias(lhs, rhs, bias_tile)
        out_bias = pl.store(biased, [0, 0], out_bias)
        return out_base, out_acc, out_bias

    @pl.jit
    def tile_entry(
        a: pl.Tensor[[M, K], lhs_pl],
        b: pl.Tensor[[K, N], rhs_pl],
        bias: pl.Tensor[[1, N], pl.FP32],
        out_base: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        out_bias: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ):
        return tile_kernel(a, b, bias, out_base, out_acc, out_bias)

    return tile_entry


def _make_tensor_entry(lhs_dtype: DataType, rhs_dtype: DataType):
    """`pl.matmul` / `pl.matmul_acc` on the Tensor surface over one operand pair."""
    lhs_pl = _pl_dtype(lhs_dtype)
    rhs_pl = _pl_dtype(rhs_dtype)

    @pl.jit
    def tensor_entry(
        a: pl.Tensor[[M, K], lhs_pl],
        b: pl.Tensor[[K, N], rhs_pl],
        out_base: pl.Out[pl.Tensor[[M, N], pl.FP32]],
        out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            # out_dtype is omitted on purpose: an FP8 operand pair defaults to the
            # FP32 accumulator, the only dtype the unscaled writeback can produce.
            base = pl.matmul(a, b)
            out_base = pl.assemble(out_base, base, [0, 0])
            accumulated = pl.matmul_acc(base, a, b)
            out_acc = pl.assemble(out_acc, accumulated, [0, 0])
        return out_base, out_acc

    return tensor_entry


@pl.jit
def valid_k_tail_matmul(
    a: pl.Tensor[[M, K], pl.FP8E4M3FN],
    b: pl.Tensor[[K, N], pl.FP8E5M2],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    """Valid K ends mid-box: only *physical* K must be a multiple of 32."""
    with pl.at(level=pl.Level.CORE_GROUP):
        av = pl.slice(a, [M, K], [0, 0], valid_shape=[M, VALID_K], clamp=True)
        bv = pl.slice(b, [K, N], [0, 0], valid_shape=[VALID_K, N], clamp=True)
        out = pl.assemble(out, pl.matmul(av, bv, out_dtype=pl.FP32), [0, 0])
    return out


class MixedFp8TileMatmulTestCase(PTOTestCase):
    """All three Tile matmul forms over one FP8 operand pair."""

    __test__ = False

    def __init__(self, lhs_dtype: DataType, rhs_dtype: DataType):
        super().__init__(_RUN_CONFIG, platform="a5")
        self._lhs_dtype = lhs_dtype
        self._rhs_dtype = rhs_dtype

    def get_name(self) -> str:
        return f"matmul_fp8_{self._lhs_dtype.value}_x_{self._rhs_dtype.value}_tile"

    def define_tensors(self) -> list[TensorSpec]:
        a = _dyadic_operand(M, K, self._lhs_dtype, seed=3)
        b = _dyadic_operand(K, N, self._rhs_dtype, seed=11)
        # Small integers: exact in FP32, and on the same 2**-6 grid as the products.
        bias = (torch.arange(N, dtype=torch.float32) % 5 - 2).reshape(1, N)
        return [
            TensorSpec("a", [M, K], self._lhs_dtype, init_value=a),
            TensorSpec("b", [K, N], self._rhs_dtype, init_value=b),
            TensorSpec("bias", [1, N], DataType.FP32, init_value=bias),
            TensorSpec("out_base", [M, N], DataType.FP32, is_output=True),
            TensorSpec("out_acc", [M, N], DataType.FP32, is_output=True),
            TensorSpec("out_bias", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return _make_tile_entry(self._lhs_dtype, self._rhs_dtype).specialize()

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        product = tensors["a"].float() @ tensors["b"].float()
        tensors["out_base"][:] = product
        tensors["out_acc"][:] = 2 * product
        tensors["out_bias"][:] = product + tensors["bias"].float()


class MixedFp8TensorMatmulTestCase(PTOTestCase):
    """`pl.matmul` and `pl.matmul_acc` over one FP8 operand pair."""

    __test__ = False

    def __init__(self, lhs_dtype: DataType, rhs_dtype: DataType):
        super().__init__(_RUN_CONFIG, platform="a5")
        self._lhs_dtype = lhs_dtype
        self._rhs_dtype = rhs_dtype

    def get_name(self) -> str:
        return f"matmul_fp8_{self._lhs_dtype.value}_x_{self._rhs_dtype.value}_tensor"

    def define_tensors(self) -> list[TensorSpec]:
        a = _dyadic_operand(M, K, self._lhs_dtype, seed=5)
        b = _dyadic_operand(K, N, self._rhs_dtype, seed=13)
        return [
            TensorSpec("a", [M, K], self._lhs_dtype, init_value=a),
            TensorSpec("b", [K, N], self._rhs_dtype, init_value=b),
            TensorSpec("out_base", [M, N], DataType.FP32, is_output=True),
            TensorSpec("out_acc", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return _make_tensor_entry(self._lhs_dtype, self._rhs_dtype).specialize()

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        product = tensors["a"].float() @ tensors["b"].float()
        tensors["out_base"][:] = product
        tensors["out_acc"][:] = 2 * product


class MixedFp8IdentityDecodeTestCase(MixedFp8TensorMatmulTestCase):
    """``A @ I`` over the format's full range: each output element is one operand.

    The identity rhs removes summation entirely, which is what lets this case
    carry subnormals and the format maximum and still demand bit equality. Any
    mismatch is a decode error on one operand byte, not an accumulation artifact.
    """

    __test__ = False

    def get_name(self) -> str:
        return f"matmul_fp8_{self._lhs_dtype.value}_x_{self._rhs_dtype.value}_identity"

    def define_tensors(self) -> list[TensorSpec]:
        a = _extreme_operand(M, K, self._lhs_dtype, seed=19)
        # 1.0 is exact in both FP8 formats, so the identity itself carries no error.
        identity = torch.eye(K, N, dtype=torch.float32).to(self._rhs_dtype.torch_dtype)
        return [
            TensorSpec("a", [M, K], self._lhs_dtype, init_value=a),
            TensorSpec("b", [K, N], self._rhs_dtype, init_value=identity),
            TensorSpec("out_base", [M, N], DataType.FP32, is_output=True),
            TensorSpec("out_acc", [M, N], DataType.FP32, is_output=True),
        ]


class MixedFp8ValidKTailTestCase(PTOTestCase):
    """A contraction that stops 40 of 64 physical K columns in."""

    __test__ = False

    def __init__(self):
        super().__init__(_RUN_CONFIG, platform="a5")

    def get_name(self) -> str:
        return "matmul_fp8_e4m3_x_e5m2_valid_k_tail"

    def define_tensors(self) -> list[TensorSpec]:
        a = _dyadic_operand(M, K, DataType.FP8E4M3FN, seed=7)
        b = _dyadic_operand(K, N, DataType.FP8E5M2, seed=17)
        return [
            TensorSpec("a", [M, K], DataType.FP8E4M3FN, init_value=a),
            TensorSpec("b", [K, N], DataType.FP8E5M2, init_value=b),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        return valid_k_tail_matmul.specialize()

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        # PTO contracts over the lhs valid K, so the reference truncates both
        # operands to that rectangle; the padding columns must not contribute.
        lhs = tensors["a"].float()[:, :VALID_K]
        rhs = tensors["b"].float()[:VALID_K, :]
        tensors["out"][:] = lhs @ rhs


_MIXED_PAIRS = [
    pytest.param(DataType.FP8E4M3FN, DataType.FP8E5M2, id="e4m3-x-e5m2"),
    pytest.param(DataType.FP8E5M2, DataType.FP8E4M3FN, id="e5m2-x-e4m3"),
    # Same-dtype E5M2 smoke: PTOAS accepts it and PyPTO never exercised it.
    pytest.param(DataType.FP8E5M2, DataType.FP8E5M2, id="e5m2-x-e5m2"),
]


@pytest.mark.platforms("a5")
class TestMatmulFp8Mixed:
    """Numerical execution coverage for the A5 FP8 matmul operand pairs."""

    @pytest.mark.parametrize(("lhs_dtype", "rhs_dtype"), _MIXED_PAIRS)
    def test_tile_matmul_base_acc_bias(self, test_runner, lhs_dtype, rhs_dtype):
        result = test_runner.run(MixedFp8TileMatmulTestCase(lhs_dtype, rhs_dtype))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize(("lhs_dtype", "rhs_dtype"), _MIXED_PAIRS)
    def test_tensor_matmul_base_and_acc(self, test_runner, lhs_dtype, rhs_dtype):
        result = test_runner.run(MixedFp8TensorMatmulTestCase(lhs_dtype, rhs_dtype))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize(("lhs_dtype", "rhs_dtype"), _MIXED_PAIRS)
    def test_identity_preserves_every_operand_value(self, test_runner, lhs_dtype, rhs_dtype):
        result = test_runner.run(MixedFp8IdentityDecodeTestCase(lhs_dtype, rhs_dtype))
        assert result.passed, f"Test failed: {result.error}"

    def test_tile_matmul_valid_k_tail(self, test_runner):
        result = test_runner.run(MixedFp8ValidKTailTestCase())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
