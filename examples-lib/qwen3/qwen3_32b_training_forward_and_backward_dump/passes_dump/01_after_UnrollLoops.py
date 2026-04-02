# pypto.program: Qwen332BTrainingForwardBackward
import pypto.language as pl

@pl.program
class Qwen332BTrainingForwardBackward:
    @pl.function
    def qwen3_32b_training_forward_and_backward_layer(self, hidden_states: pl.Tensor[[1, 4, 80], pl.BF16], target_states: pl.Tensor[[1, 4, 80], pl.BF16], input_rms_weight: pl.Tensor[[1, 80], pl.FP32], post_rms_weight: pl.Tensor[[1, 80], pl.FP32], wq: pl.Tensor[[80, 80], pl.BF16], wk: pl.Tensor[[80, 80], pl.BF16], wv: pl.Tensor[[80, 80], pl.BF16], wo: pl.Tensor[[80, 80], pl.BF16], w_gate: pl.Tensor[[80, 400], pl.BF16], w_up: pl.Tensor[[80, 400], pl.BF16], w_down: pl.Tensor[[400, 80], pl.BF16], mom_wq: pl.Tensor[[80, 80], pl.FP32], mom_wk: pl.Tensor[[80, 80], pl.FP32], mom_wv: pl.Tensor[[80, 80], pl.FP32], mom_wo: pl.Tensor[[80, 80], pl.FP32], mom_w_gate: pl.Tensor[[80, 400], pl.FP32], mom_w_up: pl.Tensor[[80, 400], pl.FP32], mom_w_down: pl.Tensor[[400, 80], pl.FP32], grad_wq: pl.Tensor[[80, 80], pl.FP32], grad_wk: pl.Tensor[[80, 80], pl.FP32], grad_wv: pl.Tensor[[80, 80], pl.FP32], grad_wo: pl.Tensor[[80, 80], pl.FP32], grad_w_gate: pl.Tensor[[80, 400], pl.FP32], grad_w_up: pl.Tensor[[80, 400], pl.FP32], grad_w_down: pl.Tensor[[400, 80], pl.FP32], out: pl.Tensor[[1, 4, 80], pl.BF16], loss_out: pl.Tensor[[1], pl.FP32]) -> tuple[pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[400, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[400, 80], pl.FP32], pl.Tensor[[1, 4, 80], pl.BF16], pl.Tensor[[1], pl.FP32]]:
        muon_scratch: pl.Tensor[[16, 16], pl.BF16] = pl.tensor.create([16, 16], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        proxy_scratch: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.create([2, 16], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        btrans_scratch: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.create([2, 4], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        with pl.auto_incore():
            grad_wq: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.muls(grad_wq, 0.0)
            grad_wk: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.muls(grad_wk, 0.0)
            grad_wv: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.muls(grad_wv, 0.0)
            grad_wo: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.muls(grad_wo, 0.0)
            grad_w_gate: pl.Tensor[[80, 400], pl.FP32] = pl.tensor.muls(grad_w_gate, 0.0)
            grad_w_up: pl.Tensor[[80, 400], pl.FP32] = pl.tensor.muls(grad_w_up, 0.0)
            grad_w_down: pl.Tensor[[400, 80], pl.FP32] = pl.tensor.muls(grad_w_down, 0.0)
            loss_acc: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.create([2, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            loss_acc: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.muls(loss_acc, 0.0)
            tok_blocks: pl.Scalar[pl.INDEX] = 2
            for b in pl.parallel(1, chunk=4):
                for p0_idx in pl.range(tok_blocks):
                    p0: pl.Scalar[pl.INDEX] = p0_idx * 2
                    sq_sum: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.create([2, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    sq_sum: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.muls(sq_sum, 0.0)
                    for kb in pl.range(20):
                        k0: pl.Scalar[pl.INDEX] = kb * 4
                        x_chunk: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.reshape(pl.tensor.cast(pl.tensor.slice(hidden_states, [1, 2, 4], [b, p0, k0]), target_type=pl.FP32, mode='round'), [2, 4])
                        sq_sum: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.add(sq_sum, pl.tensor.row_sum(pl.tensor.mul(x_chunk, x_chunk)))
                    inv_rms: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.rsqrt(pl.tensor.adds(pl.tensor.muls(sq_sum, 0.0125), 1e-06))
                    normed_tile: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.create([2, 80], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                    for kb_1 in pl.range(20):
                        k0: pl.Scalar[pl.INDEX] = kb_1 * 4
                        x_chunk: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.reshape(pl.tensor.cast(pl.tensor.slice(hidden_states, [1, 2, 4], [b, p0, k0]), target_type=pl.FP32, mode='round'), [2, 4])
                        gamma: pl.Tensor[[1, 4], pl.FP32] = pl.tensor.slice(input_rms_weight, [1, 4], [0, k0])
                        normed: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(x_chunk, inv_rms), gamma)
                        normed_tile: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.assemble(normed_tile, pl.tensor.cast(normed, target_type=pl.BF16, mode='round'), [0, k0])
                    q_proj_tile: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.create([2, 80], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                    k_proj_tile: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.create([2, 80], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                    v_proj_tile: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.create([2, 80], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                    for ob in pl.range(10):
                        q0: pl.Scalar[pl.INDEX] = ob * 8
                        q_acc: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.create([2, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        q_acc: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.muls(q_acc, 0.0)
                        k_acc: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.create([2, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        k_acc: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.muls(k_acc, 0.0)
                        v_acc: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.create([2, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        v_acc: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.muls(v_acc, 0.0)
                        for kb_2 in pl.range(20):
                            k0: pl.Scalar[pl.INDEX] = kb_2 * 4
                            n_chunk: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.cast(pl.tensor.slice(normed_tile, [2, 4], [0, k0]), target_type=pl.BF16, mode='round')
                            wq_c: pl.Tensor[[4, 8], pl.BF16] = pl.tensor.slice(wq, [4, 8], [k0, q0])
                            wk_c: pl.Tensor[[4, 8], pl.BF16] = pl.tensor.slice(wk, [4, 8], [k0, q0])
                            wv_c: pl.Tensor[[4, 8], pl.BF16] = pl.tensor.slice(wv, [4, 8], [k0, q0])
                            q_acc: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.add(q_acc, pl.tensor.matmul(n_chunk, wq_c, a_trans=False, b_trans=False, c_matrix_nz=False))
                            k_acc: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.add(k_acc, pl.tensor.matmul(n_chunk, wk_c, a_trans=False, b_trans=False, c_matrix_nz=False))
                            v_acc: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.add(v_acc, pl.tensor.matmul(n_chunk, wv_c, a_trans=False, b_trans=False, c_matrix_nz=False))
                        q_proj_tile: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.assemble(q_proj_tile, pl.tensor.cast(q_acc, target_type=pl.BF16, mode='round'), [0, q0])
                        k_proj_tile: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.assemble(k_proj_tile, pl.tensor.cast(k_acc, target_type=pl.BF16, mode='round'), [0, q0])
                        v_proj_tile: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.assemble(v_proj_tile, pl.tensor.cast(v_acc, target_type=pl.BF16, mode='round'), [0, q0])
                    scores: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.create([2, 2], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    scores: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.muls(scores, 0.0)
                    for kb_3 in pl.range(20):
                        k0: pl.Scalar[pl.INDEX] = kb_3 * 4
                        q_c: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.cast(pl.tensor.slice(q_proj_tile, [2, 4], [0, k0]), target_type=pl.FP32, mode='round')
                        k_c: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.cast(pl.tensor.slice(k_proj_tile, [2, 4], [0, k0]), target_type=pl.FP32, mode='round')
                        btrans_scratch: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.assemble(btrans_scratch, k_c, [0, 0])
                        k_c_t: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.slice(btrans_scratch, [2, 4], [0, 0])
                        scores: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.add(scores, pl.tensor.matmul(q_c, k_c_t, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32))
                    scores: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.muls(scores, 0.111803)
                    scores_exp: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.exp(scores)
                    scores_sum: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.row_sum(scores_exp)
                    attn_w: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.row_expand_mul(scores_exp, pl.tensor.recip(scores_sum))
                    context_tile: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.create([2, 80], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                    for ob_1 in pl.range(10):
                        o0: pl.Scalar[pl.INDEX] = ob_1 * 8
                        v_c: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.cast(pl.tensor.slice(v_proj_tile, [2, 8], [0, o0]), target_type=pl.FP32, mode='round')
                        ctx_acc: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.create([2, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        ctx_acc: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.muls(ctx_acc, 0.0)
                        ctx_acc: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.add(ctx_acc, pl.tensor.matmul(attn_w, v_c, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                        context_tile: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.assemble(context_tile, pl.tensor.cast(ctx_acc, target_type=pl.BF16, mode='round'), [0, o0])
                    resid1_tile: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.create([2, 80], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    for ob_2 in pl.range(10):
                        o0: pl.Scalar[pl.INDEX] = ob_2 * 8
                        o_acc: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.create([2, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        o_acc: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.muls(o_acc, 0.0)
                        for kb_4 in pl.range(20):
                            k0: pl.Scalar[pl.INDEX] = kb_4 * 4
                            ctx_chunk: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.slice(context_tile, [2, 4], [0, k0])
                            wo_c: pl.Tensor[[4, 8], pl.BF16] = pl.tensor.slice(wo, [4, 8], [k0, o0])
                            o_acc: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.add(o_acc, pl.tensor.matmul(ctx_chunk, wo_c, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                        resid: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.reshape(pl.tensor.cast(pl.tensor.slice(hidden_states, [1, 2, 8], [b, p0, o0]), target_type=pl.FP32, mode='round'), [2, 8])
                        resid1_tile: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.assemble(resid1_tile, pl.tensor.add(o_acc, resid), [0, o0])
                    sq_sum2: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.create([2, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    sq_sum2: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.muls(sq_sum2, 0.0)
                    for kb_5 in pl.range(20):
                        k0: pl.Scalar[pl.INDEX] = kb_5 * 4
                        x_chunk: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.slice(resid1_tile, [2, 4], [0, k0])
                        sq_sum2: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.add(sq_sum2, pl.tensor.row_sum(pl.tensor.mul(x_chunk, x_chunk)))
                    inv_rms2: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.rsqrt(pl.tensor.adds(pl.tensor.muls(sq_sum2, 0.0125), 1e-06))
                    post_norm_tile: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.create([2, 80], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                    for kb_6 in pl.range(20):
                        k0: pl.Scalar[pl.INDEX] = kb_6 * 4
                        x_chunk: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.slice(resid1_tile, [2, 4], [0, k0])
                        gamma: pl.Tensor[[1, 4], pl.FP32] = pl.tensor.slice(post_rms_weight, [1, 4], [0, k0])
                        normed: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(x_chunk, inv_rms2), gamma)
                        post_norm_tile: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.assemble(post_norm_tile, pl.tensor.cast(normed, target_type=pl.BF16, mode='round'), [0, k0])
                    down_tile: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.create([2, 80], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    down_tile: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.muls(down_tile, 0.0)
                    for mb in pl.range(25):
                        m0: pl.Scalar[pl.INDEX] = mb * 16
                        gate_acc: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.create([2, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        up_acc: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.create([2, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        gate_acc: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.muls(gate_acc, 0.0)
                        up_acc: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.muls(up_acc, 0.0)
                        for kb_7 in pl.range(20):
                            k0: pl.Scalar[pl.INDEX] = kb_7 * 4
                            post_chunk: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.slice(post_norm_tile, [2, 4], [0, k0])
                            wg: pl.Tensor[[4, 16], pl.BF16] = pl.tensor.slice(w_gate, [4, 16], [k0, m0])
                            wu: pl.Tensor[[4, 16], pl.BF16] = pl.tensor.slice(w_up, [4, 16], [k0, m0])
                            gate_acc: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.add(gate_acc, pl.tensor.matmul(post_chunk, wg, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                            up_acc: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.add(up_acc, pl.tensor.matmul(post_chunk, wu, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                        sigmoid_chunk: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.recip(pl.tensor.adds(pl.tensor.exp(pl.tensor.neg(gate_acc)), 1.0))
                        mlp_chunk: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.cast(pl.tensor.mul(pl.tensor.mul(gate_acc, sigmoid_chunk), up_acc), target_type=pl.BF16, mode='round')
                        for ob_3 in pl.range(10):
                            o0: pl.Scalar[pl.INDEX] = ob_3 * 8
                            down_prev: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.slice(down_tile, [2, 8], [0, o0])
                            wd: pl.Tensor[[16, 8], pl.BF16] = pl.tensor.slice(w_down, [16, 8], [m0, o0])
                            down_part: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.add(down_prev, pl.tensor.matmul(mlp_chunk, wd, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                            down_tile: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.assemble(down_tile, down_part, [0, o0])
                    out_tile: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.create([2, 80], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    for ob_4 in pl.range(10):
                        o0: pl.Scalar[pl.INDEX] = ob_4 * 8
                        out_chunk: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.add(pl.tensor.slice(down_tile, [2, 8], [0, o0]), pl.tensor.slice(resid1_tile, [2, 8], [0, o0]))
                        out_tile: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.assemble(out_tile, out_chunk, [0, o0])
                        out: pl.Tensor[[1, 4, 80], pl.BF16] = pl.tensor.assemble(out, pl.tensor.reshape(pl.tensor.cast(out_chunk, target_type=pl.BF16, mode='round'), [1, 2, 8]), [b, p0, o0])
                    tgt_tile: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.reshape(pl.tensor.cast(pl.tensor.slice(target_states, [1, 2, 80], [b, p0, 0]), target_type=pl.FP32, mode='round'), [2, 80])
                    diff_tile: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.sub(out_tile, tgt_tile)
                    sq_tile: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.mul(diff_tile, diff_tile)
                    sq_row: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.row_sum(sq_tile)
                    loss_prev: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.slice(loss_acc, [2, 1], [0, 0])
                    loss_acc: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.assemble(loss_acc, pl.tensor.add(loss_prev, sq_row), [0, 0])
                    d_out: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.muls(diff_tile, 0.025)
                    d_down: pl.Tensor[[2, 80], pl.FP32] = d_out
                    d_resid1_bwd: pl.Tensor[[2, 80], pl.FP32] = d_out
                    d_post_norm: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.create([2, 80], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    d_post_norm: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.muls(d_post_norm, 0.0)
                    for mb_1 in pl.range(25):
                        m0: pl.Scalar[pl.INDEX] = mb_1 * 16
                        gate_r: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.create([2, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        up_r: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.create([2, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        gate_r: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.muls(gate_r, 0.0)
                        up_r: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.muls(up_r, 0.0)
                        for kb_8 in pl.range(20):
                            k0: pl.Scalar[pl.INDEX] = kb_8 * 4
                            post_c: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.slice(post_norm_tile, [2, 4], [0, k0])
                            wg_c: pl.Tensor[[4, 16], pl.BF16] = pl.tensor.slice(w_gate, [4, 16], [k0, m0])
                            wu_c: pl.Tensor[[4, 16], pl.BF16] = pl.tensor.slice(w_up, [4, 16], [k0, m0])
                            gate_r: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.add(gate_r, pl.tensor.matmul(post_c, wg_c, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                            up_r: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.add(up_r, pl.tensor.matmul(post_c, wu_c, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                        sig_r: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.recip(pl.tensor.adds(pl.tensor.exp(pl.tensor.neg(gate_r)), 1.0))
                        d_mlp: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.create([2, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        d_mlp: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.muls(d_mlp, 0.0)
                        for ob_5 in pl.range(10):
                            o0: pl.Scalar[pl.INDEX] = ob_5 * 8
                            dd_c: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.cast(pl.tensor.slice(d_down, [2, 8], [0, o0]), target_type=pl.BF16, mode='round')
                            wd_c: pl.Tensor[[16, 8], pl.BF16] = pl.tensor.slice(w_down, [16, 8], [m0, o0])
                            d_mlp: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.add(d_mlp, pl.tensor.matmul(dd_c, wd_c, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32))
                        one_m_sig: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.adds(pl.tensor.muls(sig_r, -1.0), 1.0)
                        silu_deriv: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.mul(sig_r, pl.tensor.adds(pl.tensor.mul(gate_r, one_m_sig), 1.0))
                        d_gate_bf16: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.cast(pl.tensor.mul(pl.tensor.mul(d_mlp, up_r), silu_deriv), target_type=pl.BF16, mode='round')
                        d_up_bf16: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.cast(pl.tensor.mul(d_mlp, pl.tensor.mul(gate_r, sig_r)), target_type=pl.BF16, mode='round')
                        for kb_9 in pl.range(20):
                            k0: pl.Scalar[pl.INDEX] = kb_9 * 4
                            dpn_old: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.slice(d_post_norm, [2, 4], [0, k0])
                            wg_c: pl.Tensor[[4, 16], pl.BF16] = pl.tensor.slice(w_gate, [4, 16], [k0, m0])
                            wu_c: pl.Tensor[[4, 16], pl.BF16] = pl.tensor.slice(w_up, [4, 16], [k0, m0])
                            dpn_tmp: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.add(dpn_old, pl.tensor.matmul(d_gate_bf16, wg_c, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32))
                            dpn_new: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.add(dpn_tmp, pl.tensor.matmul(d_up_bf16, wu_c, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32))
                            d_post_norm: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.assemble(d_post_norm, dpn_new, [0, k0])
                    d_resid1: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.add(d_resid1_bwd, d_post_norm)
                    bwd_energy: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.create([2, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    bwd_energy: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.muls(bwd_energy, 0.0)
                    for kb_10 in pl.range(20):
                        k0: pl.Scalar[pl.INDEX] = kb_10 * 4
                        dr_c: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.slice(d_resid1, [2, 4], [0, k0])
                        q_c: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.cast(pl.tensor.slice(q_proj_tile, [2, 4], [0, k0]), target_type=pl.FP32, mode='round')
                        k_c: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.cast(pl.tensor.slice(k_proj_tile, [2, 4], [0, k0]), target_type=pl.FP32, mode='round')
                        v_bwd: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.cast(pl.tensor.slice(v_proj_tile, [2, 4], [0, k0]), target_type=pl.FP32, mode='round')
                        contrib: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.row_sum(pl.tensor.mul(dr_c, pl.tensor.add(pl.tensor.add(q_c, k_c), v_bwd)))
                        bwd_energy: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.add(bwd_energy, contrib)
                    bwd_energy: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.add(bwd_energy, pl.tensor.row_sum(attn_w))
                    grad_sink: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.muls(bwd_energy, 0.0)
            proxy_mlp: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.cast(pl.tensor.slice(w_up, [2, 16], [0, 0]), target_type=pl.BF16, mode='round')
            proxy_scratch: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.assemble(proxy_scratch, proxy_mlp, [0, 0])
            proxy_mlp_t: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.slice(proxy_scratch, [2, 16], [0, 0])
            for qb in pl.range(10):
                q0: pl.Scalar[pl.INDEX] = qb * 8
                proxy_go: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.reshape(pl.tensor.cast(pl.tensor.slice(target_states, [1, 2, 8], [0, 0, q0]), target_type=pl.BF16, mode='round'), [2, 8])
                proxy_scratch: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.assemble(proxy_scratch, proxy_go, [0, 0])
                proxy_go_t: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.slice(proxy_scratch, [2, 8], [0, 0])
                grad_down_raw: pl.Tensor[[16, 8], pl.FP32] = pl.tensor.matmul(proxy_mlp_t, proxy_go_t, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                mom_down_prev: pl.Tensor[[16, 8], pl.FP32] = pl.tensor.slice(mom_w_down, [16, 8], [0, q0])
                mom_down_new: pl.Tensor[[16, 8], pl.FP32] = pl.tensor.add(pl.tensor.muls(mom_down_prev, 0.95), pl.tensor.muls(grad_down_raw, 0.05))
                muon_down: pl.Tensor[[16, 8], pl.FP32] = mom_down_new
                muon_scratch: pl.Tensor[[16, 16], pl.BF16] = pl.tensor.assemble(muon_scratch, pl.tensor.cast(muon_down, target_type=pl.BF16, mode='round'), [0, 0])
                for _ in pl.range(2):
                    gram: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.create([8, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    gram: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.muls(gram, 0.0)
                    for tb in pl.range(8):
                        t0: pl.Scalar[pl.INDEX] = tb * 2
                        m_blk: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.slice(muon_scratch, [2, 8], [t0, 0])
                        m_blk_t: pl.Tensor[[8, 2], pl.BF16] = pl.tensor.transpose(m_blk, 0, 1)
                        gram: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.add(gram, pl.tensor.matmul(m_blk_t, m_blk, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                    muon_down: pl.Tensor[[16, 8], pl.FP32] = pl.tensor.add(pl.tensor.muls(muon_down, 1.5), pl.tensor.muls(pl.tensor.matmul(muon_down, gram, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32), -0.5))
                grad_w_down: pl.Tensor[[400, 80], pl.FP32] = pl.tensor.assemble(grad_w_down, pl.tensor.muls(muon_down, -0.0002), [0, q0])
                mom_w_down: pl.Tensor[[400, 80], pl.FP32] = pl.tensor.assemble(mom_w_down, mom_down_new, [0, q0])
            proxy_ctx: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.cast(pl.tensor.slice(wq, [2, 4], [0, 0]), target_type=pl.BF16, mode='round')
            proxy_n: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.reshape(pl.tensor.cast(pl.tensor.slice(hidden_states, [1, 2, 4], [0, 0, 0]), target_type=pl.BF16, mode='round'), [2, 4])
            proxy_scratch: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.assemble(proxy_scratch, proxy_ctx, [0, 0])
            proxy_ctx_t: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.slice(proxy_scratch, [2, 4], [0, 0])
            proxy_scratch: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.assemble(proxy_scratch, proxy_n, [0, 0])
            proxy_n_t: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.slice(proxy_scratch, [2, 4], [0, 0])
            for qb_1 in pl.range(10):
                q0: pl.Scalar[pl.INDEX] = qb_1 * 8
                proxy_tgt: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.reshape(pl.tensor.cast(pl.tensor.slice(target_states, [1, 2, 8], [0, 0, q0]), target_type=pl.BF16, mode='round'), [2, 8])
                proxy_scratch: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.assemble(proxy_scratch, proxy_tgt, [0, 0])
                proxy_tgt_t: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.slice(proxy_scratch, [2, 8], [0, 0])
                grad_wo_raw: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.matmul(proxy_ctx_t, proxy_tgt_t, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                mom_wo_prev: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.slice(mom_wo, [4, 8], [0, q0])
                mom_wo_new: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.add(pl.tensor.muls(mom_wo_prev, 0.95), pl.tensor.muls(grad_wo_raw, 0.05))
                muon_wo: pl.Tensor[[4, 8], pl.FP32] = mom_wo_new
                muon_scratch: pl.Tensor[[16, 16], pl.BF16] = pl.tensor.assemble(muon_scratch, pl.tensor.cast(muon_wo, target_type=pl.BF16, mode='round'), [0, 0])
                for __1 in pl.range(2):
                    gram: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.create([8, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    gram: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.muls(gram, 0.0)
                    for tb_1 in pl.range(2):
                        t0: pl.Scalar[pl.INDEX] = tb_1 * 2
                        m_blk: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.slice(muon_scratch, [2, 8], [t0, 0])
                        m_blk_t: pl.Tensor[[8, 2], pl.BF16] = pl.tensor.transpose(m_blk, 0, 1)
                        gram: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.add(gram, pl.tensor.matmul(m_blk_t, m_blk, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                    muon_wo: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.add(pl.tensor.muls(muon_wo, 1.5), pl.tensor.muls(pl.tensor.matmul(muon_wo, gram, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32), -0.5))
                grad_wo: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.assemble(grad_wo, pl.tensor.muls(muon_wo, -0.0002), [0, q0])
                mom_wo: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.assemble(mom_wo, mom_wo_new, [0, q0])
                grad_wq_raw: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.matmul(proxy_n_t, proxy_tgt_t, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                mom_wq_prev: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.slice(mom_wq, [4, 8], [0, q0])
                mom_wq_new: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.add(pl.tensor.muls(mom_wq_prev, 0.95), pl.tensor.muls(grad_wq_raw, 0.05))
                muon_wq: pl.Tensor[[4, 8], pl.FP32] = mom_wq_new
                muon_scratch: pl.Tensor[[16, 16], pl.BF16] = pl.tensor.assemble(muon_scratch, pl.tensor.cast(muon_wq, target_type=pl.BF16, mode='round'), [0, 0])
                for __2 in pl.range(2):
                    gram: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.create([8, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    gram: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.muls(gram, 0.0)
                    for tb_2 in pl.range(2):
                        t0: pl.Scalar[pl.INDEX] = tb_2 * 2
                        m_blk: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.slice(muon_scratch, [2, 8], [t0, 0])
                        m_blk_t: pl.Tensor[[8, 2], pl.BF16] = pl.tensor.transpose(m_blk, 0, 1)
                        gram: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.add(gram, pl.tensor.matmul(m_blk_t, m_blk, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                    muon_wq: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.add(pl.tensor.muls(muon_wq, 1.5), pl.tensor.muls(pl.tensor.matmul(muon_wq, gram, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32), -0.5))
                grad_wq: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.assemble(grad_wq, pl.tensor.muls(muon_wq, -0.0002), [0, q0])
                mom_wq: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.assemble(mom_wq, mom_wq_new, [0, q0])
                grad_wk_raw: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.matmul(proxy_n_t, proxy_tgt_t, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                mom_wk_prev: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.slice(mom_wk, [4, 8], [0, q0])
                mom_wk_new: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.add(pl.tensor.muls(mom_wk_prev, 0.95), pl.tensor.muls(grad_wk_raw, 0.05))
                muon_wk: pl.Tensor[[4, 8], pl.FP32] = mom_wk_new
                muon_scratch: pl.Tensor[[16, 16], pl.BF16] = pl.tensor.assemble(muon_scratch, pl.tensor.cast(muon_wk, target_type=pl.BF16, mode='round'), [0, 0])
                for __3 in pl.range(2):
                    gram: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.create([8, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    gram: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.muls(gram, 0.0)
                    for tb_3 in pl.range(2):
                        t0: pl.Scalar[pl.INDEX] = tb_3 * 2
                        m_blk: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.slice(muon_scratch, [2, 8], [t0, 0])
                        m_blk_t: pl.Tensor[[8, 2], pl.BF16] = pl.tensor.transpose(m_blk, 0, 1)
                        gram: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.add(gram, pl.tensor.matmul(m_blk_t, m_blk, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                    muon_wk: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.add(pl.tensor.muls(muon_wk, 1.5), pl.tensor.muls(pl.tensor.matmul(muon_wk, gram, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32), -0.5))
                grad_wk: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.assemble(grad_wk, pl.tensor.muls(muon_wk, -0.0002), [0, q0])
                mom_wk: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.assemble(mom_wk, mom_wk_new, [0, q0])
                grad_wv_raw: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.matmul(proxy_n_t, proxy_tgt_t, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                mom_wv_prev: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.slice(mom_wv, [4, 8], [0, q0])
                mom_wv_new: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.add(pl.tensor.muls(mom_wv_prev, 0.95), pl.tensor.muls(grad_wv_raw, 0.05))
                muon_wv: pl.Tensor[[4, 8], pl.FP32] = mom_wv_new
                muon_scratch: pl.Tensor[[16, 16], pl.BF16] = pl.tensor.assemble(muon_scratch, pl.tensor.cast(muon_wv, target_type=pl.BF16, mode='round'), [0, 0])
                for __4 in pl.range(2):
                    gram: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.create([8, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    gram: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.muls(gram, 0.0)
                    for tb_4 in pl.range(2):
                        t0: pl.Scalar[pl.INDEX] = tb_4 * 2
                        m_blk: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.slice(muon_scratch, [2, 8], [t0, 0])
                        m_blk_t: pl.Tensor[[8, 2], pl.BF16] = pl.tensor.transpose(m_blk, 0, 1)
                        gram: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.add(gram, pl.tensor.matmul(m_blk_t, m_blk, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                    muon_wv: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.add(pl.tensor.muls(muon_wv, 1.5), pl.tensor.muls(pl.tensor.matmul(muon_wv, gram, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32), -0.5))
                grad_wv: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.assemble(grad_wv, pl.tensor.muls(muon_wv, -0.0002), [0, q0])
                mom_wv: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.assemble(mom_wv, mom_wv_new, [0, q0])
            proxy_post: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.reshape(pl.tensor.cast(pl.tensor.slice(hidden_states, [1, 2, 4], [0, 0, 4]), target_type=pl.BF16, mode='round'), [2, 4])
            proxy_scratch: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.assemble(proxy_scratch, proxy_post, [0, 0])
            proxy_post_t: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.slice(proxy_scratch, [2, 4], [0, 0])
            for mb_2 in pl.range(25):
                m0: pl.Scalar[pl.INDEX] = mb_2 * 16
                proxy_gg: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.cast(pl.tensor.slice(w_gate, [2, 16], [0, m0]), target_type=pl.BF16, mode='round')
                proxy_gu: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.cast(pl.tensor.slice(w_up, [2, 16], [0, m0]), target_type=pl.BF16, mode='round')
                proxy_scratch: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.assemble(proxy_scratch, proxy_gg, [0, 0])
                proxy_gg_t: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.slice(proxy_scratch, [2, 16], [0, 0])
                proxy_scratch: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.assemble(proxy_scratch, proxy_gu, [0, 0])
                proxy_gu_t: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.slice(proxy_scratch, [2, 16], [0, 0])
                grad_wg_raw: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.matmul(proxy_post_t, proxy_gg_t, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                grad_wu_raw: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.matmul(proxy_post_t, proxy_gu_t, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                mom_wg_prev: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.slice(mom_w_gate, [4, 16], [0, m0])
                mom_wu_prev: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.slice(mom_w_up, [4, 16], [0, m0])
                mom_wg_new: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.add(pl.tensor.muls(mom_wg_prev, 0.95), pl.tensor.muls(grad_wg_raw, 0.05))
                mom_wu_new: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.add(pl.tensor.muls(mom_wu_prev, 0.95), pl.tensor.muls(grad_wu_raw, 0.05))
                muon_wg: pl.Tensor[[4, 16], pl.FP32] = mom_wg_new
                muon_scratch: pl.Tensor[[16, 16], pl.BF16] = pl.tensor.assemble(muon_scratch, pl.tensor.cast(muon_wg, target_type=pl.BF16, mode='round'), [0, 0])
                for __5 in pl.range(2):
                    ns_acc_g: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.create([4, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    ns_acc_g: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.muls(ns_acc_g, 0.0)
                    muon_wg_bf: pl.Tensor[[4, 16], pl.BF16] = pl.tensor.cast(muon_wg, target_type=pl.BF16, mode='round')
                    for tb_5 in pl.range(2):
                        t0: pl.Scalar[pl.INDEX] = tb_5 * 2
                        m_blk_g: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.slice(muon_scratch, [2, 16], [t0, 0])
                        m_blk_gt: pl.Tensor[[16, 2], pl.BF16] = pl.tensor.transpose(m_blk_g, 0, 1)
                        tmp_g: pl.Tensor[[4, 2], pl.FP32] = pl.tensor.matmul(muon_wg_bf, m_blk_gt, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        tmp_g_bf: pl.Tensor[[4, 2], pl.BF16] = pl.tensor.cast(tmp_g, target_type=pl.BF16, mode='round')
                        ns_acc_g: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.add(ns_acc_g, pl.tensor.matmul(tmp_g_bf, m_blk_g, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                    muon_wg: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.add(pl.tensor.muls(muon_wg, 1.5), pl.tensor.muls(ns_acc_g, -0.5))
                muon_wu: pl.Tensor[[4, 16], pl.FP32] = mom_wu_new
                muon_scratch: pl.Tensor[[16, 16], pl.BF16] = pl.tensor.assemble(muon_scratch, pl.tensor.cast(muon_wu, target_type=pl.BF16, mode='round'), [0, 0])
                for __6 in pl.range(2):
                    ns_acc_u: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.create([4, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    ns_acc_u: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.muls(ns_acc_u, 0.0)
                    muon_wu_bf: pl.Tensor[[4, 16], pl.BF16] = pl.tensor.cast(muon_wu, target_type=pl.BF16, mode='round')
                    for tb_6 in pl.range(2):
                        t0: pl.Scalar[pl.INDEX] = tb_6 * 2
                        m_blk_u: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.slice(muon_scratch, [2, 16], [t0, 0])
                        m_blk_ut: pl.Tensor[[16, 2], pl.BF16] = pl.tensor.transpose(m_blk_u, 0, 1)
                        tmp_u: pl.Tensor[[4, 2], pl.FP32] = pl.tensor.matmul(muon_wu_bf, m_blk_ut, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        tmp_u_bf: pl.Tensor[[4, 2], pl.BF16] = pl.tensor.cast(tmp_u, target_type=pl.BF16, mode='round')
                        ns_acc_u: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.add(ns_acc_u, pl.tensor.matmul(tmp_u_bf, m_blk_u, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32))
                    muon_wu: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.add(pl.tensor.muls(muon_wu, 1.5), pl.tensor.muls(ns_acc_u, -0.5))
                grad_w_gate: pl.Tensor[[80, 400], pl.FP32] = pl.tensor.assemble(grad_w_gate, pl.tensor.muls(muon_wg, -0.0002), [0, m0])
                grad_w_up: pl.Tensor[[80, 400], pl.FP32] = pl.tensor.assemble(grad_w_up, pl.tensor.muls(muon_wu, -0.0002), [0, m0])
                mom_w_gate: pl.Tensor[[80, 400], pl.FP32] = pl.tensor.assemble(mom_w_gate, mom_wg_new, [0, m0])
                mom_w_up: pl.Tensor[[80, 400], pl.FP32] = pl.tensor.assemble(mom_w_up, mom_wu_new, [0, m0])
            loss_vec: pl.Tensor[[1], pl.FP32] = pl.tensor.slice(loss_acc, [1], [0, 0])
            loss_out: pl.Tensor[[1], pl.FP32] = pl.tensor.assemble(loss_out, loss_vec, [0])
        return grad_wq, grad_wk, grad_wv, grad_wo, grad_w_gate, grad_w_up, grad_w_down, mom_wq, mom_wk, mom_wv, mom_wo, mom_w_gate, mom_w_up, mom_w_down, out, loss_out