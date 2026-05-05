# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Qwen3-32B single-layer decode forward — JIT entry point.

Composes the per-scope kernels from ``kernels/`` via plain Python imports.
Each kernel marked ``@pl.jit.inline`` is auto-discovered as a dep and
spliced into this entry by the ``InlineFunctions`` IR pass.

Run with::

    python -m examples.models.qwen3_jit.qwen3_decode

(For numerical validation against a PyTorch golden, see
``qwen3_32b_decode.py`` from the upstream pypto-lib repository.)
"""

import pypto.language as pl

from .config import (
    BATCH,
    CACHE_ROWS,
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE,
    KV_HIDDEN,
    MAX_SEQ,
    Q_HEAD_PAD,
    TOTAL_Q_GROUPS,
)
from .kernels.attention import rope_kv_cache_update
from .kernels.mlp import mlp_block
from .kernels.projection import (
    down_projection_residual,
    k_projection,
    out_projection_residual,
    q_projection,
    v_projection,
)
from .kernels.rmsnorm import input_rmsnorm, post_rmsnorm


@pl.jit
def qwen3_decode(  # noqa: PLR0913 — model signature is intrinsic
    hidden_states: pl.Tensor,
    input_rms_weight: pl.Tensor,
    wq: pl.Tensor,
    wk: pl.Tensor,
    wv: pl.Tensor,
    seq_lens: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    k_cache: pl.Tensor,
    v_cache: pl.Tensor,
    wo: pl.Tensor,
    post_rms_weight: pl.Tensor,
    w_gate: pl.Tensor,
    w_up: pl.Tensor,
    w_down: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    # ── Scope 1: input RMSNorm + Q/K/V projection ──
    normed_states = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    q_proj = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    k_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    v_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)

    normed_states = input_rmsnorm(hidden_states, input_rms_weight, normed_states)
    q_proj = q_projection(normed_states, wq, q_proj)
    k_proj = k_projection(normed_states, wk, k_proj)
    v_proj = v_projection(normed_states, wv, v_proj)

    # ── Scope 2: RoPE + KV cache update ──
    # rope_kv_cache_update writes to k_cache/v_cache and pre-pads Q for
    # downstream grouped-query attention.
    all_q_padded = pl.create_tensor([BATCH * TOTAL_Q_GROUPS * Q_HEAD_PAD, HEAD_DIM], dtype=pl.BF16)
    k_cache = rope_kv_cache_update(
        q_proj,
        k_proj,
        v_proj,
        seq_lens,
        rope_cos,
        rope_sin,
        k_cache,
        v_cache,
        all_q_padded,
    )

    # The full grouped-query attention (5 sub-stages × 2 group-pairs) follows
    # the same composition pattern as Scope 1/Scope 3 kernels — see
    # ``kernels/attention.py`` for the pattern doc. For now, attn_out is
    # created here so Scope 3 has something to consume.
    attn_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)

    # ── Scope 3: output projection + residual + post-RMSNorm + MLP + residual ──
    resid1_tile = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    post_norm_tile = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)

    resid1_tile = out_projection_residual(attn_out, hidden_states, wo, resid1_tile)
    post_norm_tile = post_rmsnorm(resid1_tile, post_rms_weight, post_norm_tile)

    mlp_tile = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.BF16)
    mlp_tile = mlp_block(post_norm_tile, w_gate, w_up, mlp_tile)
    out = down_projection_residual(mlp_tile, resid1_tile, w_down, out)

    return out


if __name__ == "__main__":
    # Minimal smoke test: build TensorMeta-shaped inputs and call
    # ``compile_for_test`` to run the full pass pipeline (no device execution).
    import torch

    def randn(shape, dtype):
        return torch.empty(shape, dtype=dtype).normal_()

    args = [
        randn([BATCH, HIDDEN], torch.bfloat16),  # hidden_states
        randn([1, HIDDEN], torch.float32),  # input_rms_weight
        randn([HIDDEN, HIDDEN], torch.bfloat16),  # wq
        randn([HIDDEN, KV_HIDDEN], torch.bfloat16),  # wk
        randn([HIDDEN, KV_HIDDEN], torch.bfloat16),  # wv
        torch.randint(1, MAX_SEQ + 1, (BATCH,), dtype=torch.int32),
        randn([MAX_SEQ, HEAD_DIM], torch.float32),  # rope_cos
        randn([MAX_SEQ, HEAD_DIM], torch.float32),  # rope_sin
        randn([CACHE_ROWS, HEAD_DIM], torch.bfloat16),  # k_cache
        randn([CACHE_ROWS, HEAD_DIM], torch.bfloat16),  # v_cache
        randn([HIDDEN, HIDDEN], torch.bfloat16),  # wo
        randn([1, HIDDEN], torch.float32),  # post_rms_weight
        randn([HIDDEN, INTERMEDIATE], torch.bfloat16),  # w_gate
        randn([HIDDEN, INTERMEDIATE], torch.bfloat16),  # w_up
        randn([INTERMEDIATE, HIDDEN], torch.bfloat16),  # w_down
        torch.empty([BATCH, HIDDEN], dtype=torch.bfloat16),  # out
    ]
    post_pass = qwen3_decode.compile_for_test(*args)
    print(f"Compiled program has {len(post_pass.functions)} function(s):")
    for fn in post_pass.functions.values():
        print(f"  {fn.name}: {fn.func_type}")
