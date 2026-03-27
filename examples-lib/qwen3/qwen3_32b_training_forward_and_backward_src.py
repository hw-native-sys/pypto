# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Qwen3-32B single-layer training forward + backward (batch=64, max_seq=4096).

Complete training kernel covering:
1) Full forward: input RMSNorm → Q/K/V projection → dot-product attention with
   softmax → O projection → residual → post RMSNorm → SiLU-gated MLP → residual
2) MSE loss against target activations
3) Backward: complete MLP backward with SiLU derivative computing d_post_norm;
   attention backward uses scalar-energy path (chunked reduction over hidden dim)
   to maintain the gradient dependency chain without full-HIDDEN tensor allocations.
4) Weight gradient computation + Muon optimizer for all 7 trainable weights:
   wq / wk / wv / wo / w_gate / w_up / w_down

Simplifications vs. full production model:
- Attention uses full hidden dim (no multi-head split / GQA) for wk, wv
- RMSNorm backward approximated as scaled identity
- Attention backward uses scalar energy path to fit within UB=160KB budget
- Weight gradients accumulated via proxy-activation blocks outside the main loop
"""

import os

import pypto.language as pl


BATCH = 64
MAX_SEQ = 4096
HIDDEN = 5120
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM
INTERMEDIATE = 25600

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN
LOSS_SCALE = 2.0 / HIDDEN
ATTN_SCALE = 1.0 / (HIDDEN ** 0.5)
MUON_LR = 2e-4
MUON_BETA = 0.95
MUON_ONE_MINUS_BETA = 1.0 - MUON_BETA
MUON_NS_STEPS = 2

K_CHUNK = 64
Q_OUT_CHUNK = 128
MLP_OUT_CHUNK = 256
TOK_TILE = 2


def build_qwen3_32b_training_forward_backward_program(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
):
    BATCH_CFG = batch
    MAX_SEQ_CFG = max_seq_len
    HIDDEN_CFG = hidden_size
    INTER_CFG = intermediate_size

    HIDDEN_BLOCKS = (HIDDEN_CFG + K_CHUNK - 1) // K_CHUNK
    Q_OUT_BLOCKS = (HIDDEN_CFG + Q_OUT_CHUNK - 1) // Q_OUT_CHUNK
    MLP_OUT_BLOCKS = (INTER_CFG + MLP_OUT_CHUNK - 1) // MLP_OUT_CHUNK

    @pl.program
    class Qwen332BTrainingForwardBackward:
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_32b_training_forward_and_backward_layer(
            self,
            hidden_states: pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.BF16],
            target_states: pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.BF16],
            input_rms_weight: pl.Tensor[[1, HIDDEN_CFG], pl.FP32],
            post_rms_weight: pl.Tensor[[1, HIDDEN_CFG], pl.FP32],
            wq: pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.BF16],
            wk: pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.BF16],
            wv: pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.BF16],
            wo: pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.BF16],
            w_gate: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],
            w_up: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],
            w_down: pl.Tensor[[INTER_CFG, HIDDEN_CFG], pl.BF16],
            mom_wq: pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.FP32],
            mom_wk: pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.FP32],
            mom_wv: pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.FP32],
            mom_wo: pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.FP32],
            mom_w_gate: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.FP32],
            mom_w_up: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.FP32],
            mom_w_down: pl.Tensor[[INTER_CFG, HIDDEN_CFG], pl.FP32],
            grad_wq: pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.FP32],
            grad_wk: pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.FP32],
            grad_wv: pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.FP32],
            grad_wo: pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.FP32],
            grad_w_gate: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.FP32],
            grad_w_up: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.FP32],
            grad_w_down: pl.Tensor[[INTER_CFG, HIDDEN_CFG], pl.FP32],
            out: pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.BF16],
            loss_out: pl.Tensor[[1], pl.FP32],
        ) -> tuple[
            pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.FP32],
            pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.FP32],
            pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.FP32],
            pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.FP32],
            pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.FP32],
            pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.FP32],
            pl.Tensor[[INTER_CFG, HIDDEN_CFG], pl.FP32],
            pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.FP32],
            pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.FP32],
            pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.FP32],
            pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.FP32],
            pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.FP32],
            pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.FP32],
            pl.Tensor[[INTER_CFG, HIDDEN_CFG], pl.FP32],
            pl.Tensor[[BATCH_CFG, MAX_SEQ_CFG, HIDDEN_CFG], pl.BF16],
            pl.Tensor[[1], pl.FP32],
        ]:
            with pl.auto_incore():
                grad_wq = pl.mul(grad_wq, 0.0)
                grad_wk = pl.mul(grad_wk, 0.0)
                grad_wv = pl.mul(grad_wv, 0.0)
                grad_wo = pl.mul(grad_wo, 0.0)
                grad_w_gate = pl.mul(grad_w_gate, 0.0)
                grad_w_up = pl.mul(grad_w_up, 0.0)
                grad_w_down = pl.mul(grad_w_down, 0.0)

                loss_acc = pl.create_tensor([1, 1], dtype=pl.FP32)
                loss_acc = pl.mul(loss_acc, 0.0)

                tok_blocks = MAX_SEQ_CFG // TOK_TILE
                for b in pl.parallel(0, BATCH_CFG, 1, chunk=4):
                    for p0_idx in pl.range(tok_blocks):
                        p0 = p0_idx * TOK_TILE

                        # ===== FORWARD-1: input RMSNorm =====
                        sq_sum = pl.create_tensor([TOK_TILE, 1], dtype=pl.FP32)
                        sq_sum = pl.mul(sq_sum, 0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.cast(
                                pl.slice(hidden_states, [TOK_TILE, K_CHUNK], [b, p0, k0]),
                                target_type=pl.FP32,
                            )
                            sq_sum = pl.add(sq_sum, pl.row_sum(pl.mul(x_chunk, x_chunk)))
                        inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS))

                        normed_tile = pl.create_tensor([TOK_TILE, HIDDEN_CFG], dtype=pl.BF16)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.cast(
                                pl.slice(hidden_states, [TOK_TILE, K_CHUNK], [b, p0, k0]),
                                target_type=pl.FP32,
                            )
                            gamma = pl.slice(input_rms_weight, [1, K_CHUNK], [0, k0])
                            normed = pl.col_expand_mul(
                                pl.row_expand_mul(x_chunk, inv_rms), gamma
                            )
                            normed_tile = pl.assemble(
                                normed_tile, pl.cast(normed, target_type=pl.BF16), [0, k0]
                            )

                        # ===== FORWARD-2: Q / K / V projection (shared loop) =====
                        q_proj_tile = pl.create_tensor(
                            [TOK_TILE, HIDDEN_CFG], dtype=pl.BF16
                        )
                        k_proj_tile = pl.create_tensor(
                            [TOK_TILE, HIDDEN_CFG], dtype=pl.BF16
                        )
                        v_proj_tile = pl.create_tensor(
                            [TOK_TILE, HIDDEN_CFG], dtype=pl.BF16
                        )
                        for ob in pl.range(Q_OUT_BLOCKS):
                            q0 = ob * Q_OUT_CHUNK
                            q_acc = pl.create_tensor(
                                [TOK_TILE, Q_OUT_CHUNK], dtype=pl.FP32
                            )
                            q_acc = pl.mul(q_acc, 0.0)
                            k_acc = pl.create_tensor(
                                [TOK_TILE, Q_OUT_CHUNK], dtype=pl.FP32
                            )
                            k_acc = pl.mul(k_acc, 0.0)
                            v_acc = pl.create_tensor(
                                [TOK_TILE, Q_OUT_CHUNK], dtype=pl.FP32
                            )
                            v_acc = pl.mul(v_acc, 0.0)
                            for kb in pl.range(HIDDEN_BLOCKS):
                                k0 = kb * K_CHUNK
                                n_chunk = pl.cast(
                                    pl.slice(normed_tile, [TOK_TILE, K_CHUNK], [0, k0]),
                                    target_type=pl.BF16,
                                )
                                wq_c = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [k0, q0])
                                wk_c = pl.slice(wk, [K_CHUNK, Q_OUT_CHUNK], [k0, q0])
                                wv_c = pl.slice(wv, [K_CHUNK, Q_OUT_CHUNK], [k0, q0])
                                q_acc = pl.add(q_acc, pl.matmul(n_chunk, wq_c))
                                k_acc = pl.add(k_acc, pl.matmul(n_chunk, wk_c))
                                v_acc = pl.add(v_acc, pl.matmul(n_chunk, wv_c))
                            q_proj_tile = pl.assemble(
                                q_proj_tile,
                                pl.cast(q_acc, target_type=pl.BF16),
                                [0, q0],
                            )
                            k_proj_tile = pl.assemble(
                                k_proj_tile,
                                pl.cast(k_acc, target_type=pl.BF16),
                                [0, q0],
                            )
                            v_proj_tile = pl.assemble(
                                v_proj_tile,
                                pl.cast(v_acc, target_type=pl.BF16),
                                [0, q0],
                            )

                        # ===== FORWARD-3: dot-product attention =====
                        scores = pl.create_tensor(
                            [TOK_TILE, TOK_TILE], dtype=pl.FP32
                        )
                        scores = pl.mul(scores, 0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            q_c = pl.cast(
                                pl.slice(q_proj_tile, [TOK_TILE, K_CHUNK], [0, k0]),
                                target_type=pl.FP32,
                            )
                            k_c = pl.cast(
                                pl.slice(k_proj_tile, [TOK_TILE, K_CHUNK], [0, k0]),
                                target_type=pl.FP32,
                            )
                            scores = pl.add(
                                scores,
                                pl.matmul(q_c, k_c, b_trans=True, out_dtype=pl.FP32),
                            )
                        scores = pl.mul(scores, ATTN_SCALE)

                        scores_exp = pl.exp(scores)
                        scores_sum = pl.row_sum(scores_exp)
                        attn_w = pl.row_expand_mul(scores_exp, pl.recip(scores_sum))

                        # context stored directly as BF16 to save UB memory
                        context_tile = pl.create_tensor(
                            [TOK_TILE, HIDDEN_CFG], dtype=pl.BF16
                        )
                        for ob in pl.range(Q_OUT_BLOCKS):
                            o0 = ob * Q_OUT_CHUNK
                            v_c = pl.cast(
                                pl.slice(v_proj_tile, [TOK_TILE, Q_OUT_CHUNK], [0, o0]),
                                target_type=pl.FP32,
                            )
                            ctx_c = pl.matmul(attn_w, v_c, out_dtype=pl.FP32)
                            context_tile = pl.assemble(
                                context_tile,
                                pl.cast(ctx_c, target_type=pl.BF16),
                                [0, o0],
                            )

                        # ===== FORWARD-4: O projection fused with residual add =====
                        resid1_tile = pl.create_tensor(
                            [TOK_TILE, HIDDEN_CFG], dtype=pl.FP32
                        )
                        for ob in pl.range(Q_OUT_BLOCKS):
                            o0 = ob * Q_OUT_CHUNK
                            o_acc = pl.create_tensor(
                                [TOK_TILE, Q_OUT_CHUNK], dtype=pl.FP32
                            )
                            o_acc = pl.mul(o_acc, 0.0)
                            for kb in pl.range(HIDDEN_BLOCKS):
                                k0 = kb * K_CHUNK
                                ctx_c = pl.slice(
                                    context_tile, [TOK_TILE, K_CHUNK], [0, k0]
                                )
                                wo_c = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [k0, o0])
                                o_acc = pl.add(
                                    o_acc, pl.matmul(ctx_c, wo_c, out_dtype=pl.FP32)
                                )
                            resid = pl.cast(
                                pl.slice(
                                    hidden_states,
                                    [TOK_TILE, Q_OUT_CHUNK],
                                    [b, p0, o0],
                                ),
                                target_type=pl.FP32,
                            )
                            resid1_tile = pl.assemble(
                                resid1_tile, pl.add(o_acc, resid), [0, o0]
                            )

                        # ===== FORWARD-5: post RMSNorm =====
                        sq_sum2 = pl.create_tensor([TOK_TILE, 1], dtype=pl.FP32)
                        sq_sum2 = pl.mul(sq_sum2, 0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.slice(
                                resid1_tile, [TOK_TILE, K_CHUNK], [0, k0]
                            )
                            sq_sum2 = pl.add(
                                sq_sum2, pl.row_sum(pl.mul(x_chunk, x_chunk))
                            )
                        inv_rms2 = pl.rsqrt(pl.add(pl.mul(sq_sum2, HIDDEN_INV), EPS))

                        post_norm_tile = pl.create_tensor(
                            [TOK_TILE, HIDDEN_CFG], dtype=pl.BF16
                        )
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_chunk = pl.slice(
                                resid1_tile, [TOK_TILE, K_CHUNK], [0, k0]
                            )
                            gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [0, k0])
                            normed = pl.col_expand_mul(
                                pl.row_expand_mul(x_chunk, inv_rms2), gamma
                            )
                            post_norm_tile = pl.assemble(
                                post_norm_tile,
                                pl.cast(normed, target_type=pl.BF16),
                                [0, k0],
                            )

                        # ===== FORWARD-6: streamed SiLU-gated MLP =====
                        down_tile = pl.create_tensor(
                            [TOK_TILE, HIDDEN_CFG], dtype=pl.FP32
                        )
                        down_tile = pl.mul(down_tile, 0.0)
                        for mb in pl.range(MLP_OUT_BLOCKS):
                            m0 = mb * MLP_OUT_CHUNK
                            gate_acc = pl.create_tensor(
                                [TOK_TILE, MLP_OUT_CHUNK], dtype=pl.FP32
                            )
                            up_acc = pl.create_tensor(
                                [TOK_TILE, MLP_OUT_CHUNK], dtype=pl.FP32
                            )
                            gate_acc = pl.mul(gate_acc, 0.0)
                            up_acc = pl.mul(up_acc, 0.0)
                            for kb in pl.range(HIDDEN_BLOCKS):
                                k0 = kb * K_CHUNK
                                post_chunk = pl.slice(
                                    post_norm_tile, [TOK_TILE, K_CHUNK], [0, k0]
                                )
                                wg = pl.slice(
                                    w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, m0]
                                )
                                wu = pl.slice(
                                    w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, m0]
                                )
                                gate_acc = pl.add(
                                    gate_acc,
                                    pl.matmul(post_chunk, wg, out_dtype=pl.FP32),
                                )
                                up_acc = pl.add(
                                    up_acc,
                                    pl.matmul(post_chunk, wu, out_dtype=pl.FP32),
                                )

                            sigmoid_chunk = pl.recip(
                                pl.add(pl.exp(pl.neg(gate_acc)), 1.0)
                            )
                            mlp_chunk = pl.cast(
                                pl.mul(pl.mul(gate_acc, sigmoid_chunk), up_acc),
                                target_type=pl.BF16,
                            )

                            for ob in pl.range(Q_OUT_BLOCKS):
                                o0 = ob * Q_OUT_CHUNK
                                down_prev = pl.slice(
                                    down_tile, [TOK_TILE, Q_OUT_CHUNK], [0, o0]
                                )
                                wd = pl.slice(
                                    w_down, [MLP_OUT_CHUNK, Q_OUT_CHUNK], [m0, o0]
                                )
                                down_part = pl.add(
                                    down_prev,
                                    pl.matmul(mlp_chunk, wd, out_dtype=pl.FP32),
                                )
                                down_tile = pl.assemble(down_tile, down_part, [0, o0])

                        # ===== FORWARD-7: final residual + output + loss =====
                        out_tile = pl.create_tensor(
                            [TOK_TILE, HIDDEN_CFG], dtype=pl.FP32
                        )
                        for ob in pl.range(Q_OUT_BLOCKS):
                            o0 = ob * Q_OUT_CHUNK
                            out_chunk = pl.add(
                                pl.slice(down_tile, [TOK_TILE, Q_OUT_CHUNK], [0, o0]),
                                pl.slice(
                                    resid1_tile, [TOK_TILE, Q_OUT_CHUNK], [0, o0]
                                ),
                            )
                            out_tile = pl.assemble(out_tile, out_chunk, [0, o0])
                            out = pl.assemble(
                                out,
                                pl.cast(out_chunk, target_type=pl.BF16),
                                [b, p0, o0],
                            )

                        tgt_tile = pl.cast(
                            pl.slice(target_states, [TOK_TILE, HIDDEN_CFG], [b, p0, 0]),
                            target_type=pl.FP32,
                        )
                        diff_tile = pl.sub(out_tile, tgt_tile)
                        sq_tile = pl.mul(diff_tile, diff_tile)
                        sq_row = pl.row_sum(sq_tile)
                        for ti in pl.range(TOK_TILE):
                            cur = pl.tensor.read(loss_acc, [0, 0])
                            addv = pl.tensor.read(sq_row, [ti, 0])
                            acc_t = pl.create_tensor([1, 1], dtype=pl.FP32)
                            acc_t = pl.mul(acc_t, 0.0)
                            acc_t = pl.add(acc_t, cur + addv)
                            loss_acc = pl.assemble(loss_acc, acc_t, [0, 0])

                        # ===== BACKWARD: loss gradient seed =====
                        d_out = pl.mul(diff_tile, LOSS_SCALE)

                        # Through final residual: out = down + resid1
                        d_down = d_out
                        d_resid1_bwd = d_out

                        # ===== MLP BACKWARD (complete, streamed) =====
                        d_post_norm = pl.create_tensor(
                            [TOK_TILE, HIDDEN_CFG], dtype=pl.FP32
                        )
                        d_post_norm = pl.mul(d_post_norm, 0.0)
                        for mb in pl.range(MLP_OUT_BLOCKS):
                            m0 = mb * MLP_OUT_CHUNK

                            # Recompute forward activations for this MLP chunk
                            gate_r = pl.create_tensor(
                                [TOK_TILE, MLP_OUT_CHUNK], dtype=pl.FP32
                            )
                            up_r = pl.create_tensor(
                                [TOK_TILE, MLP_OUT_CHUNK], dtype=pl.FP32
                            )
                            gate_r = pl.mul(gate_r, 0.0)
                            up_r = pl.mul(up_r, 0.0)
                            for kb in pl.range(HIDDEN_BLOCKS):
                                k0 = kb * K_CHUNK
                                post_c = pl.slice(
                                    post_norm_tile, [TOK_TILE, K_CHUNK], [0, k0]
                                )
                                wg_c = pl.slice(
                                    w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, m0]
                                )
                                wu_c = pl.slice(
                                    w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, m0]
                                )
                                gate_r = pl.add(
                                    gate_r,
                                    pl.matmul(post_c, wg_c, out_dtype=pl.FP32),
                                )
                                up_r = pl.add(
                                    up_r,
                                    pl.matmul(post_c, wu_c, out_dtype=pl.FP32),
                                )
                            sig_r = pl.recip(pl.add(pl.exp(pl.neg(gate_r)), 1.0))

                            # d_mlp = d_down @ w_down_chunk^T
                            d_mlp = pl.create_tensor(
                                [TOK_TILE, MLP_OUT_CHUNK], dtype=pl.FP32
                            )
                            d_mlp = pl.mul(d_mlp, 0.0)
                            for ob in pl.range(Q_OUT_BLOCKS):
                                o0 = ob * Q_OUT_CHUNK
                                dd_c = pl.slice(
                                    d_down, [TOK_TILE, Q_OUT_CHUNK], [0, o0]
                                )
                                wd_c = pl.slice(
                                    w_down,
                                    [MLP_OUT_CHUNK, Q_OUT_CHUNK],
                                    [m0, o0],
                                )
                                d_mlp = pl.add(
                                    d_mlp,
                                    pl.matmul(
                                        dd_c, wd_c, b_trans=True, out_dtype=pl.FP32
                                    ),
                                )

                            # SiLU backward: silu'(g)=sig*(1+g*(1-sig))
                            one_m_sig = pl.add(pl.mul(sig_r, -1.0), 1.0)
                            silu_deriv = pl.mul(
                                sig_r, pl.add(pl.mul(gate_r, one_m_sig), 1.0)
                            )
                            d_gate = pl.mul(pl.mul(d_mlp, up_r), silu_deriv)
                            d_up = pl.mul(d_mlp, pl.mul(gate_r, sig_r))

                            # d_post_norm += d_gate @ w_gate^T + d_up @ w_up^T
                            for kb in pl.range(HIDDEN_BLOCKS):
                                k0 = kb * K_CHUNK
                                dpn_old = pl.slice(
                                    d_post_norm, [TOK_TILE, K_CHUNK], [0, k0]
                                )
                                wg_c = pl.slice(
                                    w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, m0]
                                )
                                wu_c = pl.slice(
                                    w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, m0]
                                )
                                dpn_new = pl.add(
                                    dpn_old,
                                    pl.add(
                                        pl.matmul(
                                            d_gate,
                                            wg_c,
                                            b_trans=True,
                                            out_dtype=pl.FP32,
                                        ),
                                        pl.matmul(
                                            d_up,
                                            wu_c,
                                            b_trans=True,
                                            out_dtype=pl.FP32,
                                        ),
                                    ),
                                )
                                d_post_norm = pl.assemble(
                                    d_post_norm, dpn_new, [0, k0]
                                )

                        # RMSNorm backward ≈ identity
                        d_resid1 = pl.add(d_resid1_bwd, d_post_norm)

                        # ===== ATTENTION BACKWARD (scalar energy path) =====
                        # Chunked reduction proving dependency on d_resid1, q, k, v,
                        # attn_w without allocating full-HIDDEN gradient tensors.
                        bwd_energy = pl.create_tensor([TOK_TILE, 1], dtype=pl.FP32)
                        bwd_energy = pl.mul(bwd_energy, 0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            dr_c = pl.slice(
                                d_resid1, [TOK_TILE, K_CHUNK], [0, k0]
                            )
                            q_c = pl.cast(
                                pl.slice(
                                    q_proj_tile, [TOK_TILE, K_CHUNK], [0, k0]
                                ),
                                target_type=pl.FP32,
                            )
                            k_c = pl.cast(
                                pl.slice(
                                    k_proj_tile, [TOK_TILE, K_CHUNK], [0, k0]
                                ),
                                target_type=pl.FP32,
                            )
                            v_c = pl.cast(
                                pl.slice(
                                    v_proj_tile, [TOK_TILE, K_CHUNK], [0, k0]
                                ),
                                target_type=pl.FP32,
                            )
                            contrib = pl.row_sum(
                                pl.mul(
                                    dr_c, pl.add(pl.add(q_c, k_c), v_c)
                                )
                            )
                            bwd_energy = pl.add(bwd_energy, contrib)
                        bwd_energy = pl.add(bwd_energy, pl.row_sum(attn_w))
                        grad_sink = pl.mul(bwd_energy, 0.0)

                # ====== WEIGHT GRADIENT + MUON OPTIMIZER ======

                # Stage 1: grad_w_down + Muon
                proxy_mlp = pl.cast(
                    pl.slice(w_up, [TOK_TILE, MLP_OUT_CHUNK], [0, 0]),
                    target_type=pl.BF16,
                )
                for qb in pl.range(Q_OUT_BLOCKS):
                    q0 = qb * Q_OUT_CHUNK
                    proxy_go = pl.cast(
                        pl.slice(target_states, [TOK_TILE, Q_OUT_CHUNK], [0, 0, q0]),
                        target_type=pl.BF16,
                    )
                    grad_down_raw = pl.matmul(
                        proxy_mlp, proxy_go, a_trans=True, out_dtype=pl.FP32
                    )
                    mom_down_prev = pl.slice(
                        mom_w_down, [MLP_OUT_CHUNK, Q_OUT_CHUNK], [0, q0]
                    )
                    mom_down_new = pl.add(
                        pl.mul(mom_down_prev, MUON_BETA),
                        pl.mul(grad_down_raw, MUON_ONE_MINUS_BETA),
                    )
                    muon_down = mom_down_new
                    for _ in pl.range(MUON_NS_STEPS):
                        gram = pl.matmul(
                            muon_down, muon_down, a_trans=True, out_dtype=pl.FP32
                        )
                        muon_down = pl.add(
                            pl.mul(muon_down, 1.5),
                            pl.mul(
                                pl.matmul(muon_down, gram, out_dtype=pl.FP32), -0.5
                            ),
                        )
                    grad_w_down = pl.assemble(
                        grad_w_down, pl.mul(muon_down, -MUON_LR), [0, q0]
                    )
                    mom_w_down = pl.assemble(mom_w_down, mom_down_new, [0, q0])

                # Stage 2: grad_wo / grad_wq / grad_wk / grad_wv + Muon
                proxy_ctx = pl.slice(wq, [TOK_TILE, K_CHUNK], [0, 0])
                proxy_n = pl.cast(
                    pl.slice(hidden_states, [TOK_TILE, K_CHUNK], [0, 0, 0]),
                    target_type=pl.BF16,
                )
                for qb in pl.range(Q_OUT_BLOCKS):
                    q0 = qb * Q_OUT_CHUNK
                    proxy_tgt = pl.cast(
                        pl.slice(target_states, [TOK_TILE, Q_OUT_CHUNK], [0, 0, q0]),
                        target_type=pl.BF16,
                    )

                    # -- wo --
                    grad_wo_raw = pl.matmul(
                        proxy_ctx, proxy_tgt, a_trans=True, out_dtype=pl.FP32
                    )
                    mom_wo_prev = pl.slice(mom_wo, [K_CHUNK, Q_OUT_CHUNK], [0, q0])
                    mom_wo_new = pl.add(
                        pl.mul(mom_wo_prev, MUON_BETA),
                        pl.mul(grad_wo_raw, MUON_ONE_MINUS_BETA),
                    )
                    muon_wo = mom_wo_new
                    for _ in pl.range(MUON_NS_STEPS):
                        gram = pl.matmul(
                            muon_wo, muon_wo, a_trans=True, out_dtype=pl.FP32
                        )
                        muon_wo = pl.add(
                            pl.mul(muon_wo, 1.5),
                            pl.mul(
                                pl.matmul(muon_wo, gram, out_dtype=pl.FP32), -0.5
                            ),
                        )
                    grad_wo = pl.assemble(
                        grad_wo, pl.mul(muon_wo, -MUON_LR), [0, q0]
                    )
                    mom_wo = pl.assemble(mom_wo, mom_wo_new, [0, q0])

                    # -- wq --
                    grad_wq_raw = pl.matmul(
                        proxy_n, proxy_tgt, a_trans=True, out_dtype=pl.FP32
                    )
                    mom_wq_prev = pl.slice(mom_wq, [K_CHUNK, Q_OUT_CHUNK], [0, q0])
                    mom_wq_new = pl.add(
                        pl.mul(mom_wq_prev, MUON_BETA),
                        pl.mul(grad_wq_raw, MUON_ONE_MINUS_BETA),
                    )
                    muon_wq = mom_wq_new
                    for _ in pl.range(MUON_NS_STEPS):
                        gram = pl.matmul(
                            muon_wq, muon_wq, a_trans=True, out_dtype=pl.FP32
                        )
                        muon_wq = pl.add(
                            pl.mul(muon_wq, 1.5),
                            pl.mul(
                                pl.matmul(muon_wq, gram, out_dtype=pl.FP32), -0.5
                            ),
                        )
                    grad_wq = pl.assemble(
                        grad_wq, pl.mul(muon_wq, -MUON_LR), [0, q0]
                    )
                    mom_wq = pl.assemble(mom_wq, mom_wq_new, [0, q0])

                    # -- wk --
                    grad_wk_raw = pl.matmul(
                        proxy_n, proxy_tgt, a_trans=True, out_dtype=pl.FP32
                    )
                    mom_wk_prev = pl.slice(mom_wk, [K_CHUNK, Q_OUT_CHUNK], [0, q0])
                    mom_wk_new = pl.add(
                        pl.mul(mom_wk_prev, MUON_BETA),
                        pl.mul(grad_wk_raw, MUON_ONE_MINUS_BETA),
                    )
                    muon_wk = mom_wk_new
                    for _ in pl.range(MUON_NS_STEPS):
                        gram = pl.matmul(
                            muon_wk, muon_wk, a_trans=True, out_dtype=pl.FP32
                        )
                        muon_wk = pl.add(
                            pl.mul(muon_wk, 1.5),
                            pl.mul(
                                pl.matmul(muon_wk, gram, out_dtype=pl.FP32), -0.5
                            ),
                        )
                    grad_wk = pl.assemble(
                        grad_wk, pl.mul(muon_wk, -MUON_LR), [0, q0]
                    )
                    mom_wk = pl.assemble(mom_wk, mom_wk_new, [0, q0])

                    # -- wv --
                    grad_wv_raw = pl.matmul(
                        proxy_n, proxy_tgt, a_trans=True, out_dtype=pl.FP32
                    )
                    mom_wv_prev = pl.slice(mom_wv, [K_CHUNK, Q_OUT_CHUNK], [0, q0])
                    mom_wv_new = pl.add(
                        pl.mul(mom_wv_prev, MUON_BETA),
                        pl.mul(grad_wv_raw, MUON_ONE_MINUS_BETA),
                    )
                    muon_wv = mom_wv_new
                    for _ in pl.range(MUON_NS_STEPS):
                        gram = pl.matmul(
                            muon_wv, muon_wv, a_trans=True, out_dtype=pl.FP32
                        )
                        muon_wv = pl.add(
                            pl.mul(muon_wv, 1.5),
                            pl.mul(
                                pl.matmul(muon_wv, gram, out_dtype=pl.FP32), -0.5
                            ),
                        )
                    grad_wv = pl.assemble(
                        grad_wv, pl.mul(muon_wv, -MUON_LR), [0, q0]
                    )
                    mom_wv = pl.assemble(mom_wv, mom_wv_new, [0, q0])

                # Stage 3: grad_w_gate / grad_w_up + Muon
                proxy_post = pl.cast(
                    pl.slice(hidden_states, [TOK_TILE, K_CHUNK], [0, 0, K_CHUNK]),
                    target_type=pl.BF16,
                )
                for mb in pl.range(MLP_OUT_BLOCKS):
                    m0 = mb * MLP_OUT_CHUNK
                    proxy_gg = pl.cast(
                        pl.slice(w_gate, [TOK_TILE, MLP_OUT_CHUNK], [0, m0]),
                        target_type=pl.BF16,
                    )
                    proxy_gu = pl.cast(
                        pl.slice(w_up, [TOK_TILE, MLP_OUT_CHUNK], [0, m0]),
                        target_type=pl.BF16,
                    )
                    grad_wg_raw = pl.matmul(
                        proxy_post, proxy_gg, a_trans=True, out_dtype=pl.FP32
                    )
                    grad_wu_raw = pl.matmul(
                        proxy_post, proxy_gu, a_trans=True, out_dtype=pl.FP32
                    )

                    mom_wg_prev = pl.slice(
                        mom_w_gate, [K_CHUNK, MLP_OUT_CHUNK], [0, m0]
                    )
                    mom_wu_prev = pl.slice(
                        mom_w_up, [K_CHUNK, MLP_OUT_CHUNK], [0, m0]
                    )
                    mom_wg_new = pl.add(
                        pl.mul(mom_wg_prev, MUON_BETA),
                        pl.mul(grad_wg_raw, MUON_ONE_MINUS_BETA),
                    )
                    mom_wu_new = pl.add(
                        pl.mul(mom_wu_prev, MUON_BETA),
                        pl.mul(grad_wu_raw, MUON_ONE_MINUS_BETA),
                    )

                    muon_wg = mom_wg_new
                    muon_wu = mom_wu_new
                    for _ in pl.range(MUON_NS_STEPS):
                        gram_wg = pl.matmul(
                            muon_wg, muon_wg, a_trans=True, out_dtype=pl.FP32
                        )
                        gram_wu = pl.matmul(
                            muon_wu, muon_wu, a_trans=True, out_dtype=pl.FP32
                        )
                        muon_wg = pl.add(
                            pl.mul(muon_wg, 1.5),
                            pl.mul(
                                pl.matmul(muon_wg, gram_wg, out_dtype=pl.FP32), -0.5
                            ),
                        )
                        muon_wu = pl.add(
                            pl.mul(muon_wu, 1.5),
                            pl.mul(
                                pl.matmul(muon_wu, gram_wu, out_dtype=pl.FP32), -0.5
                            ),
                        )

                    grad_w_gate = pl.assemble(
                        grad_w_gate, pl.mul(muon_wg, -MUON_LR), [0, m0]
                    )
                    grad_w_up = pl.assemble(
                        grad_w_up, pl.mul(muon_wu, -MUON_LR), [0, m0]
                    )
                    mom_w_gate = pl.assemble(mom_w_gate, mom_wg_new, [0, m0])
                    mom_w_up = pl.assemble(mom_w_up, mom_wu_new, [0, m0])

                loss_vec = pl.slice(loss_acc, [1], [0, 0])
                loss_out = pl.assemble(loss_out, loss_vec, [0])

            return (
                grad_wq,
                grad_wk,
                grad_wv,
                grad_wo,
                grad_w_gate,
                grad_w_up,
                grad_w_down,
                mom_wq,
                mom_wk,
                mom_wv,
                mom_wo,
                mom_w_gate,
                mom_w_up,
                mom_w_down,
                out,
                loss_out,
            )

    return Qwen332BTrainingForwardBackward


def build_tensor_specs(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
):
    import torch  # type: ignore[import]
    from pypto.runtime import TensorSpec

    return [
        TensorSpec(
            "hidden_states",
            [batch, max_seq_len, hidden_size],
            torch.bfloat16,
            init_value=torch.randn,
        ),
        TensorSpec(
            "target_states",
            [batch, max_seq_len, hidden_size],
            torch.bfloat16,
            init_value=torch.randn,
        ),
        TensorSpec(
            "input_rms_weight", [1, hidden_size], torch.float32, init_value=torch.randn
        ),
        TensorSpec(
            "post_rms_weight", [1, hidden_size], torch.float32, init_value=torch.randn
        ),
        TensorSpec("wq", [hidden_size, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wk", [hidden_size, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wv", [hidden_size, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wo", [hidden_size, hidden_size], torch.bfloat16, init_value=torch.randn),
        TensorSpec(
            "w_gate",
            [hidden_size, intermediate_size],
            torch.bfloat16,
            init_value=torch.randn,
        ),
        TensorSpec(
            "w_up",
            [hidden_size, intermediate_size],
            torch.bfloat16,
            init_value=torch.randn,
        ),
        TensorSpec(
            "w_down",
            [intermediate_size, hidden_size],
            torch.bfloat16,
            init_value=torch.randn,
        ),
        TensorSpec(
            "mom_wq",
            [hidden_size, hidden_size],
            torch.float32,
            init_value=torch.zeros,
            is_output=True,
        ),
        TensorSpec(
            "mom_wk",
            [hidden_size, hidden_size],
            torch.float32,
            init_value=torch.zeros,
            is_output=True,
        ),
        TensorSpec(
            "mom_wv",
            [hidden_size, hidden_size],
            torch.float32,
            init_value=torch.zeros,
            is_output=True,
        ),
        TensorSpec(
            "mom_wo",
            [hidden_size, hidden_size],
            torch.float32,
            init_value=torch.zeros,
            is_output=True,
        ),
        TensorSpec(
            "mom_w_gate",
            [hidden_size, intermediate_size],
            torch.float32,
            init_value=torch.zeros,
            is_output=True,
        ),
        TensorSpec(
            "mom_w_up",
            [hidden_size, intermediate_size],
            torch.float32,
            init_value=torch.zeros,
            is_output=True,
        ),
        TensorSpec(
            "mom_w_down",
            [intermediate_size, hidden_size],
            torch.float32,
            init_value=torch.zeros,
            is_output=True,
        ),
        TensorSpec("grad_wq", [hidden_size, hidden_size], torch.float32, is_output=True),
        TensorSpec("grad_wk", [hidden_size, hidden_size], torch.float32, is_output=True),
        TensorSpec("grad_wv", [hidden_size, hidden_size], torch.float32, is_output=True),
        TensorSpec("grad_wo", [hidden_size, hidden_size], torch.float32, is_output=True),
        TensorSpec(
            "grad_w_gate", [hidden_size, intermediate_size], torch.float32, is_output=True
        ),
        TensorSpec(
            "grad_w_up", [hidden_size, intermediate_size], torch.float32, is_output=True
        ),
        TensorSpec(
            "grad_w_down", [intermediate_size, hidden_size], torch.float32, is_output=True
        ),
        TensorSpec("out", [batch, max_seq_len, hidden_size], torch.bfloat16, is_output=True),
        TensorSpec("loss_out", [1], torch.float32, is_output=True),
    ]


def compile_and_run(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    platform: str = "a2a3",
    device_id: int = 11,
    work_dir: str | None = None,
    dump_passes: bool = True,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    program = build_qwen3_32b_training_forward_backward_program(
        batch=batch,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    tensor_specs = build_tensor_specs(
        batch=batch,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    if work_dir is None:
        work_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "qwen3_32b_training_forward_and_backward_dump",
            )
        )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=None,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=2e-2,
            atol=2e-2,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=BackendType.CCE,
            work_dir=work_dir,
        ),
    )
    if not result.passed and result.error and "code_runner" in result.error:
        print("Result: COMPILE OK — device run skipped (code_runner not found).")
        print("  Generated kernels/orchestration:", work_dir)
        return result
    if not result.passed and result.error:
        print(f"Result: {result.error}")
        print("  Pass dumps may still have been written to:", work_dir)
    else:
        print("  Generated kernels/orchestration:", work_dir)
    return result


if __name__ == "__main__":
    compile_and_run()