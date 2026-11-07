# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit C2V epilogues through JIT specialization, compilation and device execution."""

import pypto.language as pl
import pytest
import torch
from harness import st
from harness.core.harness import DataType as TensorDType
from harness.core.harness import TensorSpec


def _group(dtype, relu):
    slot_bytes = 64 * 128 * dtype.get_bit() // 8
    buffer_bytes = slot_bytes * 2

    @pl.jit.inline(auto_scope=False)
    def initialize():
        peer = pl.import_peer_buffer(name="explicit_scores", peer_func="consumer")
        pl.aic_initialize_pipe(
            c2v_consumer_buf=peer,
            dir_mask=1,
            slot_size=slot_bytes,
            slot_num=2,
            c2v_dtype=dtype,
            pre_relu=relu,
        )

    def shared(
        a: pl.Tensor[[64, 128], pl.BF16],
        b: pl.Tensor[[128, 128], pl.BF16],
        out: pl.Out[pl.Tensor[[64, 128], dtype]],
        raw: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        cube_role: pl.constexpr,
    ):
        if cube_role:
            initialize()
            for iteration in pl.range(3):
                a_mat = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
                b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
                left = pl.move(a_mat, target_memory=pl.Mem.Left)
                right = pl.move(b_mat, target_memory=pl.Mem.Right)
                acc = pl.matmul(left, right)
                pl.tpush_to_aiv(acc, split=0)
                pl.store(acc, [0, 0], raw)
        else:
            buf = pl.reserve_buffer(name="explicit_scores", size=buffer_bytes)
            pl.aiv_initialize_pipe(
                c2v_consumer_buf=buf,
                dir_mask=1,
                slot_size=slot_bytes,
                slot_num=2,
                c2v_dtype=dtype,
                pre_relu=relu,
            )
            for iteration in pl.range(3):
                value = pl.tpop_from_aic(shape=[64, 128], dtype=dtype, split=0)
                pl.store(value, [0, 0], out)
                pl.tfree_to_aic(value)
        return out, raw

    producer = pl.jit.incore(level=pl.Level.AIC)(shared)
    consumer = pl.jit.incore(level=pl.Level.AIV)(shared)

    @pl.jit.inline(auto_scope=False)
    def launch(
        a: pl.Tensor[[64, 128], pl.BF16],
        b: pl.Tensor[[128, 128], pl.BF16],
        out: pl.Tensor[[64, 128], dtype],
        raw: pl.Tensor[[64, 128], pl.FP32],
    ):
        produced_out, produced_raw = producer(a, b, out, raw, True)
        consumed_out, consumed_raw = consumer(a, b, produced_out, produced_raw, False)
        return consumed_out, consumed_raw

    @pl.jit.incore(level=pl.Level.CORE_GROUP)
    def group(
        a: pl.Tensor[[64, 128], pl.BF16],
        b: pl.Tensor[[128, 128], pl.BF16],
        out: pl.Out[pl.Tensor[[64, 128], dtype]],
        raw: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
    ):
        next_out, next_raw = launch(a, b, out, raw)
        return next_out, next_raw

    return group


def _program(dtype, relu):
    group = _group(dtype, relu)

    @pl.jit
    def main(
        a: pl.Tensor[[64, 128], pl.BF16],
        b: pl.Tensor[[128, 128], pl.BF16],
        out: pl.Out[pl.Tensor[[64, 128], dtype]],
        raw: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
    ) -> tuple[pl.Tensor[[64, 128], dtype], pl.Tensor[[64, 128], pl.FP32]]:
        with pl.spmd(1, sync_start=True):
            group(a, b, out, raw)
        return out, raw

    return main


def _inputs():
    # Independent FP32 golden: observe conversion ties and keep the original Acc.
    a = torch.zeros(64, 128, dtype=torch.bfloat16)
    a[:, :2] = 1
    b = torch.zeros(128, 128, dtype=torch.bfloat16)
    b[0] = torch.tensor([1.0, 1.0078125, -1.0, -1.0078125] * 32).to(torch.bfloat16)
    b[1] = torch.tensor([2**-8, 2**-11, -(2**-8), -(2**-11)] * 32).to(torch.bfloat16)
    raw = a.float() @ b.float()
    return a, b, raw


def _case(dtype, torch_dtype, relu):
    a, b, raw = _inputs()
    activated = torch.maximum(raw, torch.zeros_like(raw)) if relu else raw
    spec_dtype = {pl.FP32: TensorDType.FP32, pl.FP16: TensorDType.FP16, pl.BF16: TensorDType.BF16}[dtype]
    return st.case(
        _program(dtype, relu),
        tensors=[
            TensorSpec("a", [64, 128], TensorDType.BF16, init_value=a),
            TensorSpec("b", [128, 128], TensorDType.BF16, init_value=b),
            TensorSpec("out", [64, 128], spec_dtype, is_output=True),
            TensorSpec("raw", [64, 128], TensorDType.FP32, is_output=True),
        ],
        golden=lambda _: {"out": activated.to(torch_dtype), "raw": raw},
        name=f"explicit_{dtype}_relu{relu}",
        rtol=0,
        atol=0,
    )


def _two_groups_case():
    # Both factories wrap shared as producer/consumer. JIT must uniquify the
    # second group's peer_func references along with its ordinary calls.
    bf16_group = _group(pl.BF16, True)
    fp16_group = _group(pl.FP16, False)

    @pl.jit
    def main(
        a: pl.Tensor[[64, 128], pl.BF16],
        b: pl.Tensor[[128, 128], pl.BF16],
        bf16_out: pl.Out[pl.Tensor[[64, 128], pl.BF16]],
        bf16_raw: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        fp16_out: pl.Out[pl.Tensor[[64, 128], pl.FP16]],
        fp16_raw: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
    ):
        with pl.spmd(1, sync_start=True):
            bf16_group(a, b, bf16_out, bf16_raw)
        with pl.spmd(1, sync_start=True):
            fp16_group(a, b, fp16_out, fp16_raw)
        return bf16_out, bf16_raw, fp16_out, fp16_raw

    a, b, raw = _inputs()
    return st.case(
        main,
        tensors=[
            TensorSpec("a", [64, 128], TensorDType.BF16, init_value=a),
            TensorSpec("b", [128, 128], TensorDType.BF16, init_value=b),
            TensorSpec("bf16_out", [64, 128], TensorDType.BF16, is_output=True),
            TensorSpec("bf16_raw", [64, 128], TensorDType.FP32, is_output=True),
            TensorSpec("fp16_out", [64, 128], TensorDType.FP16, is_output=True),
            TensorSpec("fp16_raw", [64, 128], TensorDType.FP32, is_output=True),
        ],
        golden=lambda _: {
            "bf16_out": raw.clamp_min(0).to(torch.bfloat16),
            "bf16_raw": raw,
            "fp16_out": raw.to(torch.float16),
            "fp16_raw": raw,
        },
        name="explicit_two_groups",
        rtol=0,
        atol=0,
    )


@pytest.mark.platforms("a5")
@st.cases(
    *(
        _case(dtype, torch_dtype, relu)
        for dtype, torch_dtype in (
            (pl.FP32, torch.float32),
            (pl.FP16, torch.float16),
            (pl.BF16, torch.bfloat16),
        )
        for relu in (False, True)
    ),
    _two_groups_case(),
)
def test_explicit_c2v_epilogue(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
