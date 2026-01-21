#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
""" """
import os
import torch
import torch_npu
import pypto
import numpy as np
from numpy.testing import assert_allclose


def rms_norm_golden(hidden_states, gamma, eps):
    x_dtype = hidden_states.dtype
    mean_coff = 1.0 / hidden_states.shape[-1]
    x_f32 = hidden_states
    square = x_f32 * x_f32
    square = square.sum(dim=-1, keepdim=True)
    mean_res = square * mean_coff
    reduce_sum = mean_res + eps
    reduce_sqrt = torch.sqrt(reduce_sum)
    res_div = x_f32 / reduce_sqrt
    res = res_div * gamma.to(torch.float32)
    if x_dtype != torch.float32:
        res = res.to(x_dtype)
    return res


@pypto.block
def rms_norm_block(
    x: pypto.Tensor, x_gamma: pypto.Tensor, y: pypto.Tensor, block_idx, epsilon
):
    # 每个核上行方向可以处理的块数
    row_size = cut[0] / tile[0]
    # 当前位于哪一行
    row = pypto.block.get_block_idx() / row_size
    # 当前位于哪一列
    col = pypto.block.get_block_idx() % row_size
    x_tmp = pypto.block.load(
        x, row * tile[0] + block_idx, col * tile[1], tile[0], tile[1]
    )
    x_sq = pypto.block.mul(x_tmp, x_tmp)
    sum_x_sq = pypto.block.sum(x_sq, -1, True)
    mean_x_sq = pypto.block.div(sum_x_sq, x_tmp.size)
    mean_x_sq_eps = pypto.block.add(mean_x_sq, epsilon)
    sqrt_mean = pypto.block.sqrt(mean_x_sq_eps)
    div_tmp = pypto.block.div(x_tmp, sqrt_mean)
    out_tmp = pypto.block.mul(div_tmp, x_gamma)
    pypto.block.store(
        out_tmp, block_idx + row * tile[0], 0, y.shape[0], y.shape[1], y
    )


def up_align(n, align):
    return (n + align - 1) // align


cut = (128, 5120)
tile = (8, 5120)


@pypto.jit
def rms_norm(
    x: pypto.Tensor,
    x_gamma: pypto.Tensor,
    hidden_states_out: pypto.Tensor,
    epsilon: float = 1e-5,
):
    pypto.set_vec_tile_shapes(*tile)
    # MPMD编程
    x_gamma_2d = pypto.reshape(x_gamma, [1, x_gamma.shape[0]], inplace=True)
    # block编程
    for block_idx in pypto.loop(up_align(x.shape[0], cut[0])):
        total_block_num = up_align(x.shape[0], tile[0]) * up_align(x.shape[1], tile[1])
        rms_norm_block(
            total_block_num, x, x_gamma_2d, hidden_states_out, block_idx, epsilon
        )


def test_rms_norm_main():
    bs = 32
    h_num = 5120

    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    eps = 1e-5

    for i in range(0, 2):
        if i == 1:
            bs = 1026
        # 准备测试数据
        hidden_states_tensor = torch.rand(
            (bs, h_num), dtype=torch.bfloat16, device=f"npu:{device_id}"
        )
        weight_tensor = torch.rand(
            (h_num), dtype=torch.bfloat16, device=f"npu:{device_id}"
        )

        output_hidden_states = torch.empty(
            (bs, h_num), dtype=torch.bfloat16, device=f"npu:{device_id}"
        )

        inputs = {
            hidden_states_tensor: [0],
            weight_tensor: [],
        }
        outputs = {
            output_hidden_states: [0],
        }

        pto_inputs = [
            pypto.from_torch(tensor, dynamic_axis=axis)
            for tensor, axis in inputs.items()
        ]
        pto_outputs = [
            pypto.from_torch(tensor, dynamic_axis=axis)
            for tensor, axis in outputs.items()
        ]
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            rms_norm(*pto_inputs, *pto_outputs, eps)
        g.replay()

        golden_hidden_states = rms_norm_golden(hidden_states_tensor, weight_tensor, eps)
        assert_allclose(
            np.array(output_hidden_states.cpu().flatten().tolist()),
            np.array(golden_hidden_states.flatten().tolist()),
            rtol=8e-3,
            atol=8e-3,
        )


def main():
    test_rms_norm_main()


if __name__ == "__main__":
    main()
