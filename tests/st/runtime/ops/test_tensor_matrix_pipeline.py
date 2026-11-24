# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""On-device coverage for matrix regions that exceed the on-chip capacity."""

import pypto.language as pl
import pytest
import torch
from harness import st

M = 771
N = 1025
K = 1057


@pl.jit
def _cast_then_matmul(
    a: pl.Tensor[[M, K], pl.FP32],
    b: pl.Tensor[[K, N], pl.BF16],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        scaled = pl.cast(pl.mul(a, 0.5), pl.BF16)
        out[:] = pl.matmul(scaled, b, out_dtype=pl.FP32)
    return out


@pl.jit
def _chained_matmuls(
    a: pl.Tensor[[M, K], pl.BF16],
    b: pl.Tensor[[K, N], pl.BF16],
    c: pl.Tensor[[N, 257], pl.BF16],
    out: pl.Out[pl.Tensor[[M, 257], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        first = pl.matmul(a, b, out_dtype=pl.BF16)
        out[:] = pl.matmul(first, c, out_dtype=pl.FP32)
    return out


@pl.jit
def _accumulate_from_gm(
    a: pl.Tensor[[M, K], pl.BF16],
    b: pl.Tensor[[K, N], pl.BF16],
    seed: pl.Tensor[[M, N], pl.FP32],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        out[:] = pl.matmul_acc(seed, a, b)
    return out


@pl.jit
def _accumulate_products(
    a: pl.Tensor[[M, K], pl.BF16],
    b: pl.Tensor[[K, N], pl.BF16],
    c: pl.Tensor[[M, K], pl.BF16],
    d: pl.Tensor[[K, N], pl.BF16],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        first = pl.matmul(a, b, out_dtype=pl.FP32)
        out[:] = pl.matmul_acc(first, c, d)
    return out


@pl.jit
def _attention(
    query: pl.Tensor[[513, 256], pl.BF16],
    key: pl.Tensor[[2049, 256], pl.BF16],
    value: pl.Tensor[[2049, 128], pl.BF16],
    out: pl.Out[pl.Tensor[[513, 128], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        scores = pl.matmul(query, key, b_trans=True, out_dtype=pl.FP32)
        top = pl.row_max(scores)
        weight = pl.exp(pl.sub(scores, top))
        total = pl.row_sum(weight)
        probability = pl.cast(pl.div(weight, total), pl.BF16)
        out[:] = pl.matmul(probability, value, out_dtype=pl.FP32)
    return out


@pl.jit
def _wide_softmax(
    a: pl.Tensor[[513, 4097], pl.FP32],
    out: pl.Out[pl.Tensor[[513, 4097], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        top = pl.row_max(a)
        weight = pl.exp(pl.sub(a, top))
        total = pl.row_sum(weight)
        out[:] = pl.div(weight, total)
    return out


NORM_COLUMNS = 1537
NORM_RECIPROCAL = 1.0 / NORM_COLUMNS


@pl.jit
def _layer_norm(
    a: pl.Tensor[[257, NORM_COLUMNS], pl.FP32],
    out: pl.Out[pl.Tensor[[257, NORM_COLUMNS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        mean = pl.mul(pl.row_sum(a), NORM_RECIPROCAL)
        centered = pl.sub(a, mean)
        variance = pl.mul(pl.row_sum(pl.mul(centered, centered)), NORM_RECIPROCAL)
        inverse_stddev = pl.rsqrt(pl.add(variance, 1e-5))
        out[:] = pl.mul(centered, inverse_stddev)
    return out


@pl.jit
def _singleton_row_sum(
    wide: pl.Tensor[[1, 65536], pl.FP32],
    value: pl.Tensor[[1, 1], pl.FP32],
    out: pl.Out[pl.Tensor[[1, 1], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        # Only the neighboring operation needs a wide window.
        _unused = pl.add(wide, 1.0)
        out[:] = pl.row_sum(value)
    return out


@pl.jit
def _singleton_col_sum(
    tall: pl.Tensor[[32768, 8], pl.FP32],
    value: pl.Tensor[[1, 8], pl.FP32],
    out: pl.Out[pl.Tensor[[1, 8], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        _unused = pl.add(tall, 1.0)
        out[:] = pl.col_sum(value)
    return out


def _pipeline_cases():
    generator = torch.Generator().manual_seed(53)

    def rand(shape, dtype=torch.bfloat16, scale=0.05):
        return (torch.randn(shape, generator=generator) * scale).to(dtype)

    def dot(a, b):
        return a.float() @ b.float()

    def attention_golden(tensors):
        scores = dot(tensors["query"], tensors["key"].T)
        probability = torch.softmax(scores, dim=-1).bfloat16()
        return dot(probability, tensors["value"])

    a = rand((M, K))
    b = rand((K, N), scale=0.5)
    return (
        st.case(
            _cast_then_matmul,
            a.float(),
            b,
            torch.zeros(M, N),
            name="cast_then_matmul",
            golden=lambda t: dot((t["a"] * 0.5).bfloat16(), t["b"]),
            rtol=0.006,
            atol=0.0015,
        ),
        st.case(
            _chained_matmuls,
            a,
            b,
            rand((N, 257)),
            torch.zeros(M, 257),
            name="chained_matmuls",
            golden=lambda t: dot(dot(t["a"], t["b"]).bfloat16(), t["c"]),
            rtol=0.006,
            atol=0.003,
        ),
        st.case(
            _accumulate_from_gm,
            a,
            b,
            rand((M, N), torch.float32),
            torch.zeros(M, N),
            name="accumulate_from_gm",
            golden=lambda t: t["seed"] + dot(t["a"], t["b"]),
            rtol=0.006,
            atol=0.0015,
        ),
        st.case(
            _accumulate_products,
            a,
            b,
            rand((M, K)),
            rand((K, N), scale=0.5),
            torch.zeros(M, N),
            name="accumulate_products",
            golden=lambda t: dot(t["a"], t["b"]) + dot(t["c"], t["d"]),
            rtol=0.006,
            atol=0.0015,
        ),
        st.case(
            _attention,
            rand((513, 256)),
            rand((2049, 256), scale=0.25),
            rand((2049, 128), scale=0.125),
            torch.zeros(513, 128),
            name="attention_softmax",
            golden=attention_golden,
            rtol=0.006,
            atol=0.002,
        ),
        st.case(
            _wide_softmax,
            rand((513, 4097), torch.float32, scale=1.0),
            torch.zeros(513, 4097),
            name="wide_softmax",
            golden=lambda t: torch.softmax(t["a"], dim=-1),
            rtol=0.006,
            atol=2e-7,
        ),
        st.case(
            _layer_norm,
            rand((257, NORM_COLUMNS), torch.float32, scale=1.0),
            torch.zeros(257, NORM_COLUMNS),
            name="layer_norm_odd_width",
            golden=lambda t: torch.nn.functional.layer_norm(t["a"], (NORM_COLUMNS,)),
            rtol=0.006,
            atol=0.0015,
        ),
        st.case(
            _singleton_row_sum,
            torch.zeros(1, 65536),
            torch.full((1, 1), 7.0),
            torch.zeros(1, 1),
            name="singleton_row_sum_with_wide_neighbor",
            golden=lambda t: t["value"],
            rtol=0.0,
            atol=0.0,
        ),
        st.case(
            _singleton_col_sum,
            torch.zeros(32768, 8),
            torch.arange(1, 9, dtype=torch.float32).reshape(1, 8),
            torch.zeros(1, 8),
            name="singleton_col_sum_with_tall_neighbor",
            golden=lambda t: t["value"],
            rtol=0.0,
            atol=0.0,
        ),
    )


def _partial_k_matmul_case(n, dtype, torch_dtype):
    m, k = 1025, 1027

    @pl.jit
    def kernel(
        a: pl.Tensor[[m, k], dtype],
        b: pl.Tensor[[n, k], dtype],
        out: pl.Out[pl.Tensor[[m, n], dtype]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            out[:] = pl.matmul(a, b, b_trans=True, out_dtype=dtype)
        return out

    generator = torch.Generator().manual_seed(23)
    a = (torch.randn(m, k, generator=generator) * 0.03).to(torch_dtype)
    b = (torch.randn(n, k, generator=generator) * 0.5).to(torch_dtype)
    return st.case(
        kernel,
        a,
        b,
        torch.zeros(m, n, dtype=torch_dtype),
        name=f"partial_k_matmul_n{n}_{torch_dtype}",
        golden=lambda t: (t["a"].float() @ t["b"].float().T).to(torch_dtype),
        rtol=0.006,
        atol=0.0015,
    )


@pytest.mark.platforms("a2a3", "a2a3sim")
@st.cases(*_pipeline_cases())
def test_large_matrix_pipeline(case_run):
    case_run.assert_passed()


@pytest.mark.platforms("a2a3", "a2a3sim")
@st.cases(
    _partial_k_matmul_case(1543, pl.FP16, torch.float16),
    _partial_k_matmul_case(1542, pl.FP16, torch.float16),
    _partial_k_matmul_case(1543, pl.BF16, torch.bfloat16),
)
def test_large_matmul_partial_k(case_run):
    case_run.assert_passed()


def _sequential_col_sum_case(rows, columns):
    @pl.jit
    def kernel(
        a: pl.Tensor[[rows, columns], pl.FP32],
        out: pl.Out[pl.Tensor[[1, columns], pl.FP32]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            out[:] = pl.col_sum(a)
        return out

    # Sequential FP32 addition loses the unit between two large cancelling
    # terms. Regrouped partial sums can recover it, exposing an order change
    # which a random-input tolerance check would usually accept.
    values = torch.tensor([2.0**24, 1.0, -(2.0**24)], dtype=torch.float32)
    indices = (torch.arange(rows).reshape(-1, 1) + torch.arange(columns)) % 3
    data = values[indices]

    def sequential(tensors):
        result = tensors["a"][0].clone()
        for row in tensors["a"][1:]:
            result.add_(row)
        return result.unsqueeze(0)

    return st.case(
        kernel,
        data,
        torch.zeros(1, columns),
        name=f"sequential_col_sum_{rows}x{columns}",
        golden=sequential,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.platforms("a2a3", "a2a3sim")
@st.cases(
    _sequential_col_sum_case(49153, 8),
    _sequential_col_sum_case(65537, 9),
    _sequential_col_sum_case(131089, 1),
)
def test_large_col_sum_preserves_sequential_order(case_run):
    case_run.assert_passed()


def _authored_scalar_column_case(dtype, torch_dtype, rows):
    @pl.jit
    def kernel(
        a: pl.Tensor[[1, 1], dtype],
        out: pl.Out[pl.Tensor[[1, 1], dtype]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            value = pl.load(a, [0, 0], [rows, 1], valid_shape=[1, 1])
            out = pl.store(value, [0, 0], out)
        return out

    return st.case(
        kernel,
        torch.tensor([[2.0]], dtype=torch_dtype),
        torch.zeros((1, 1), dtype=torch_dtype),
        name=f"authored_scalar_column_{dtype}",
        golden=lambda tensors: tensors["a"],
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.platforms("a2a3", "a2a3sim")
@st.cases(
    _authored_scalar_column_case(pl.FP16, torch.float16, 16),
    _authored_scalar_column_case(pl.FP32, torch.float32, 8),
)
def test_authored_scalar_column_layout(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
