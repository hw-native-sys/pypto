# pypto.program: Qwen332BTrainingForwardBackward
import pypto.language as pl

@pl.program
class Qwen332BTrainingForwardBackward:
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_32b_training_forward_and_backward_layer_incore_0(self, grad_w_down__ssa_v0: pl.Tensor[[400, 80], pl.FP32], grad_w_gate__ssa_v0: pl.Tensor[[80, 400], pl.FP32], grad_w_up__ssa_v0: pl.Tensor[[80, 400], pl.FP32], grad_wk__ssa_v0: pl.Tensor[[80, 80], pl.FP32], grad_wo__ssa_v0: pl.Tensor[[80, 80], pl.FP32], grad_wq__ssa_v0: pl.Tensor[[80, 80], pl.FP32], grad_wv__ssa_v0: pl.Tensor[[80, 80], pl.FP32]) -> tuple[pl.Tensor[[400, 80], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32]]:
        grad_wq__ssa_v1: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.muls(grad_wq__ssa_v0, 0.0)
        grad_wk__ssa_v1: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.muls(grad_wk__ssa_v0, 0.0)
        grad_wv__ssa_v1: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.muls(grad_wv__ssa_v0, 0.0)
        grad_wo__ssa_v1: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.muls(grad_wo__ssa_v0, 0.0)
        grad_w_gate__ssa_v1: pl.Tensor[[80, 400], pl.FP32] = pl.tensor.muls(grad_w_gate__ssa_v0, 0.0)
        grad_w_up__ssa_v1: pl.Tensor[[80, 400], pl.FP32] = pl.tensor.muls(grad_w_up__ssa_v0, 0.0)
        grad_w_down__ssa_v1: pl.Tensor[[400, 80], pl.FP32] = pl.tensor.muls(grad_w_down__ssa_v0, 0.0)
        return grad_w_down__ssa_v1, grad_w_gate__ssa_v1, grad_w_up__ssa_v1, grad_wk__ssa_v1, grad_wo__ssa_v1, grad_wq__ssa_v1, grad_wv__ssa_v1
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_32b_training_forward_and_backward_layer_incore_1(self, loss_acc__ssa_v0: pl.Tensor[[2, 1], pl.FP32]) -> pl.Tensor[[2, 1], pl.FP32]:
        loss_acc__ssa_v1: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.muls(loss_acc__ssa_v0, 0.0)
        return loss_acc__ssa_v1
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_32b_training_forward_and_backward_layer_incore_2(self, btrans_scratch__ssa_v0: pl.Tensor[[2, 4], pl.FP32], hidden_states__ssa_v0: pl.Tensor[[1, 4, 80], pl.BF16], input_rms_weight__ssa_v0: pl.Tensor[[1, 80], pl.FP32], loss_acc__ssa_v1: pl.Tensor[[2, 1], pl.FP32], out__ssa_v0: pl.Tensor[[1, 4, 80], pl.BF16], post_rms_weight__ssa_v0: pl.Tensor[[1, 80], pl.FP32], target_states__ssa_v0: pl.Tensor[[1, 4, 80], pl.BF16], tok_blocks__ssa_v0: pl.Scalar[pl.INDEX], w_down__ssa_v0: pl.Tensor[[400, 80], pl.BF16], w_gate__ssa_v0: pl.Tensor[[80, 400], pl.BF16], w_up__ssa_v0: pl.Tensor[[80, 400], pl.BF16], wk__ssa_v0: pl.Tensor[[80, 80], pl.BF16], wo__ssa_v0: pl.Tensor[[80, 80], pl.BF16], wq__ssa_v0: pl.Tensor[[80, 80], pl.BF16], wv__ssa_v0: pl.Tensor[[80, 80], pl.BF16]) -> tuple[pl.Tensor[[2, 1], pl.FP32], pl.Tensor[[1, 4, 80], pl.BF16]]:
        for b__cr_idx_v0, (btrans_scratch__cr_iter_v1, loss_acc__cr_iter_v2, out__cr_iter_v1) in pl.parallel(1, init_values=(btrans_scratch__ssa_v0, loss_acc__ssa_v1, out__ssa_v0)):
            for p0_idx__idx_v0, (btrans_scratch__iter_v3, loss_acc__iter_v4, out__iter_v3) in pl.range(tok_blocks__ssa_v0, init_values=(btrans_scratch__cr_iter_v1, loss_acc__cr_iter_v2, out__cr_iter_v1)):
                p0__ssa_v0: pl.Scalar[pl.INDEX] = p0_idx__idx_v0 * 2
                sq_sum__ssa_v0: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.create([2, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                sq_sum__ssa_v1: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.muls(sq_sum__ssa_v0, 0.0)
                for kb__idx_v0, (sq_sum__iter_v2,) in pl.range(20, init_values=(sq_sum__ssa_v1,)):
                    k0__ssa_v0: pl.Scalar[pl.INDEX] = kb__idx_v0 * 4
                    t__tmp_v0: pl.Tensor[[1, 2, 4], pl.BF16] = pl.tensor.slice(hidden_states__ssa_v0, [1, 2, 4], [0 + b__cr_idx_v0 * 1, p0__ssa_v0, k0__ssa_v0])
                    t__tmp_v1: pl.Tensor[[1, 2, 4], pl.FP32] = pl.tensor.cast(t__tmp_v0, target_type=pl.FP32, mode='round')
                    x_chunk__ssa_v0: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.reshape(t__tmp_v1, [2, 4])
                    t__tmp_v2: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.mul(x_chunk__ssa_v0, x_chunk__ssa_v0)
                    t__tmp_v3: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.row_sum(t__tmp_v2)
                    sq_sum__ssa_v4: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.add(sq_sum__iter_v2, t__tmp_v3)
                    sq_sum__rv_v3: pl.Tensor[[2, 1], pl.FP32] = pl.yield_(sq_sum__ssa_v4)
                t__tmp_v4: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.muls(sq_sum__rv_v3, 0.0125)
                t__tmp_v5: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.adds(t__tmp_v4, 1e-06)
                inv_rms__ssa_v0: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.rsqrt(t__tmp_v5)
                normed_tile__ssa_v0: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.create([2, 80], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                for kb__idx_v0_1, (normed_tile__iter_v1,) in pl.range(20, init_values=(normed_tile__ssa_v0,)):
                    k0__ssa_v1: pl.Scalar[pl.INDEX] = kb__idx_v0_1 * 4
                    t__tmp_v6: pl.Tensor[[1, 2, 4], pl.BF16] = pl.tensor.slice(hidden_states__ssa_v0, [1, 2, 4], [0 + b__cr_idx_v0 * 1, p0__ssa_v0, k0__ssa_v1])
                    t__tmp_v7: pl.Tensor[[1, 2, 4], pl.FP32] = pl.tensor.cast(t__tmp_v6, target_type=pl.FP32, mode='round')
                    x_chunk__ssa_v1: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.reshape(t__tmp_v7, [2, 4])
                    gamma__ssa_v0: pl.Tensor[[1, 4], pl.FP32] = pl.tensor.slice(input_rms_weight__ssa_v0, [1, 4], [0, k0__ssa_v1])
                    t__tmp_v8: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.row_expand_mul(x_chunk__ssa_v1, inv_rms__ssa_v0)
                    normed__ssa_v0: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.col_expand_mul(t__tmp_v8, gamma__ssa_v0)
                    t__tmp_v9: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.cast(normed__ssa_v0, target_type=pl.BF16, mode='round')
                    normed_tile__ssa_v3: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.assemble(normed_tile__iter_v1, t__tmp_v9, [0, k0__ssa_v1])
                    normed_tile__rv_v2: pl.Tensor[[2, 80], pl.BF16] = pl.yield_(normed_tile__ssa_v3)
                q_proj_tile__ssa_v0: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.create([2, 80], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                k_proj_tile__ssa_v0: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.create([2, 80], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                v_proj_tile__ssa_v0: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.create([2, 80], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                for ob__idx_v0, (k_proj_tile__iter_v1, q_proj_tile__iter_v1, v_proj_tile__iter_v1) in pl.range(10, init_values=(k_proj_tile__ssa_v0, q_proj_tile__ssa_v0, v_proj_tile__ssa_v0)):
                    q0__ssa_v0: pl.Scalar[pl.INDEX] = ob__idx_v0 * 8
                    q_acc__ssa_v0: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.create([2, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    q_acc__ssa_v1: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.muls(q_acc__ssa_v0, 0.0)
                    k_acc__ssa_v0: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.create([2, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    k_acc__ssa_v1: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.muls(k_acc__ssa_v0, 0.0)
                    v_acc__ssa_v0: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.create([2, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    v_acc__ssa_v1: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.muls(v_acc__ssa_v0, 0.0)
                    for kb__idx_v0_2, (k_acc__iter_v2, q_acc__iter_v2, v_acc__iter_v2) in pl.range(20, init_values=(k_acc__ssa_v1, q_acc__ssa_v1, v_acc__ssa_v1)):
                        k0__ssa_v2: pl.Scalar[pl.INDEX] = kb__idx_v0_2 * 4
                        t__tmp_v10: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.slice(normed_tile__rv_v2, [2, 4], [0, k0__ssa_v2])
                        n_chunk__ssa_v0: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.cast(t__tmp_v10, target_type=pl.BF16, mode='round')
                        wq_c__ssa_v0: pl.Tensor[[4, 8], pl.BF16] = pl.tensor.slice(wq__ssa_v0, [4, 8], [k0__ssa_v2, q0__ssa_v0])
                        wk_c__ssa_v0: pl.Tensor[[4, 8], pl.BF16] = pl.tensor.slice(wk__ssa_v0, [4, 8], [k0__ssa_v2, q0__ssa_v0])
                        wv_c__ssa_v0: pl.Tensor[[4, 8], pl.BF16] = pl.tensor.slice(wv__ssa_v0, [4, 8], [k0__ssa_v2, q0__ssa_v0])
                        t__tmp_v11: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.matmul(n_chunk__ssa_v0, wq_c__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False)
                        q_acc__ssa_v4: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.add(q_acc__iter_v2, t__tmp_v11)
                        t__tmp_v12: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.matmul(n_chunk__ssa_v0, wk_c__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False)
                        k_acc__ssa_v4: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.add(k_acc__iter_v2, t__tmp_v12)
                        t__tmp_v13: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.matmul(n_chunk__ssa_v0, wv_c__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False)
                        v_acc__ssa_v4: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.add(v_acc__iter_v2, t__tmp_v13)
                        k_acc__rv_v3, q_acc__rv_v3, v_acc__rv_v3 = pl.yield_(k_acc__ssa_v4, q_acc__ssa_v4, v_acc__ssa_v4)
                    t__tmp_v14: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.cast(q_acc__rv_v3, target_type=pl.BF16, mode='round')
                    q_proj_tile__ssa_v3: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.assemble(q_proj_tile__iter_v1, t__tmp_v14, [0, q0__ssa_v0])
                    t__tmp_v15: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.cast(k_acc__rv_v3, target_type=pl.BF16, mode='round')
                    k_proj_tile__ssa_v3: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.assemble(k_proj_tile__iter_v1, t__tmp_v15, [0, q0__ssa_v0])
                    t__tmp_v16: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.cast(v_acc__rv_v3, target_type=pl.BF16, mode='round')
                    v_proj_tile__ssa_v3: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.assemble(v_proj_tile__iter_v1, t__tmp_v16, [0, q0__ssa_v0])
                    k_proj_tile__rv_v2, q_proj_tile__rv_v2, v_proj_tile__rv_v2 = pl.yield_(k_proj_tile__ssa_v3, q_proj_tile__ssa_v3, v_proj_tile__ssa_v3)
                scores__ssa_v0: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.create([2, 2], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                scores__ssa_v1: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.muls(scores__ssa_v0, 0.0)
                for kb__idx_v0_3, (btrans_scratch__iter_v5, scores__iter_v2) in pl.range(20, init_values=(btrans_scratch__iter_v3, scores__ssa_v1)):
                    k0__ssa_v3: pl.Scalar[pl.INDEX] = kb__idx_v0_3 * 4
                    t__tmp_v17: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.slice(q_proj_tile__rv_v2, [2, 4], [0, k0__ssa_v3])
                    q_c__ssa_v0: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.cast(t__tmp_v17, target_type=pl.FP32, mode='round')
                    t__tmp_v18: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.slice(k_proj_tile__rv_v2, [2, 4], [0, k0__ssa_v3])
                    k_c__ssa_v0: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.cast(t__tmp_v18, target_type=pl.FP32, mode='round')
                    btrans_scratch__ssa_v7: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.assemble(btrans_scratch__iter_v5, k_c__ssa_v0, [0, 0])
                    k_c_t__ssa_v0: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.slice(btrans_scratch__ssa_v7, [2, 4], [0, 0])
                    t__tmp_v19: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.matmul(q_c__ssa_v0, k_c_t__ssa_v0, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                    scores__ssa_v4: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.add(scores__iter_v2, t__tmp_v19)
                    btrans_scratch__rv_v6, scores__rv_v3 = pl.yield_(btrans_scratch__ssa_v7, scores__ssa_v4)
                scores__ssa_v5: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.muls(scores__rv_v3, 0.111803)
                scores_exp__ssa_v0: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.exp(scores__ssa_v5)
                scores_sum__ssa_v0: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.row_sum(scores_exp__ssa_v0)
                t__tmp_v20: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.recip(scores_sum__ssa_v0)
                attn_w__ssa_v0: pl.Tensor[[2, 2], pl.FP32] = pl.tensor.row_expand_mul(scores_exp__ssa_v0, t__tmp_v20)
                context_tile__ssa_v0: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.create([2, 80], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                for ob__idx_v0_1, (context_tile__iter_v1,) in pl.range(10, init_values=(context_tile__ssa_v0,)):
                    o0__ssa_v0: pl.Scalar[pl.INDEX] = ob__idx_v0_1 * 8
                    t__tmp_v21: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.slice(v_proj_tile__rv_v2, [2, 8], [0, o0__ssa_v0])
                    v_c__ssa_v0: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.cast(t__tmp_v21, target_type=pl.FP32, mode='round')
                    ctx_acc__ssa_v0: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.create([2, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    ctx_acc__ssa_v1: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.muls(ctx_acc__ssa_v0, 0.0)
                    t__tmp_v22: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.matmul(attn_w__ssa_v0, v_c__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    ctx_acc__ssa_v2: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.add(ctx_acc__ssa_v1, t__tmp_v22)
                    t__tmp_v23: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.cast(ctx_acc__ssa_v2, target_type=pl.BF16, mode='round')
                    context_tile__ssa_v3: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.assemble(context_tile__iter_v1, t__tmp_v23, [0, o0__ssa_v0])
                    context_tile__rv_v2: pl.Tensor[[2, 80], pl.BF16] = pl.yield_(context_tile__ssa_v3)
                resid1_tile__ssa_v0: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.create([2, 80], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for ob__idx_v0_2, (resid1_tile__iter_v1,) in pl.range(10, init_values=(resid1_tile__ssa_v0,)):
                    o0__ssa_v1: pl.Scalar[pl.INDEX] = ob__idx_v0_2 * 8
                    o_acc__ssa_v0: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.create([2, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    o_acc__ssa_v1: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.muls(o_acc__ssa_v0, 0.0)
                    for kb__idx_v0_4, (o_acc__iter_v2,) in pl.range(20, init_values=(o_acc__ssa_v1,)):
                        k0__ssa_v4: pl.Scalar[pl.INDEX] = kb__idx_v0_4 * 4
                        ctx_chunk__ssa_v0: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.slice(context_tile__rv_v2, [2, 4], [0, k0__ssa_v4])
                        wo_c__ssa_v0: pl.Tensor[[4, 8], pl.BF16] = pl.tensor.slice(wo__ssa_v0, [4, 8], [k0__ssa_v4, o0__ssa_v1])
                        t__tmp_v24: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.matmul(ctx_chunk__ssa_v0, wo_c__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        o_acc__ssa_v4: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.add(o_acc__iter_v2, t__tmp_v24)
                        o_acc__rv_v3: pl.Tensor[[2, 8], pl.FP32] = pl.yield_(o_acc__ssa_v4)
                    t__tmp_v25: pl.Tensor[[1, 2, 8], pl.BF16] = pl.tensor.slice(hidden_states__ssa_v0, [1, 2, 8], [0 + b__cr_idx_v0 * 1, p0__ssa_v0, o0__ssa_v1])
                    t__tmp_v26: pl.Tensor[[1, 2, 8], pl.FP32] = pl.tensor.cast(t__tmp_v25, target_type=pl.FP32, mode='round')
                    resid__ssa_v0: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.reshape(t__tmp_v26, [2, 8])
                    t__tmp_v27: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.add(o_acc__rv_v3, resid__ssa_v0)
                    resid1_tile__ssa_v3: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.assemble(resid1_tile__iter_v1, t__tmp_v27, [0, o0__ssa_v1])
                    resid1_tile__rv_v2: pl.Tensor[[2, 80], pl.FP32] = pl.yield_(resid1_tile__ssa_v3)
                sq_sum2__ssa_v0: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.create([2, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                sq_sum2__ssa_v1: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.muls(sq_sum2__ssa_v0, 0.0)
                for kb__idx_v0_5, (sq_sum2__iter_v2,) in pl.range(20, init_values=(sq_sum2__ssa_v1,)):
                    k0__ssa_v5: pl.Scalar[pl.INDEX] = kb__idx_v0_5 * 4
                    x_chunk__ssa_v2: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.slice(resid1_tile__rv_v2, [2, 4], [0, k0__ssa_v5])
                    t__tmp_v28: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.mul(x_chunk__ssa_v2, x_chunk__ssa_v2)
                    t__tmp_v29: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.row_sum(t__tmp_v28)
                    sq_sum2__ssa_v4: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.add(sq_sum2__iter_v2, t__tmp_v29)
                    sq_sum2__rv_v3: pl.Tensor[[2, 1], pl.FP32] = pl.yield_(sq_sum2__ssa_v4)
                t__tmp_v30: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.muls(sq_sum2__rv_v3, 0.0125)
                t__tmp_v31: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.adds(t__tmp_v30, 1e-06)
                inv_rms2__ssa_v0: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.rsqrt(t__tmp_v31)
                post_norm_tile__ssa_v0: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.create([2, 80], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                for kb__idx_v0_6, (post_norm_tile__iter_v1,) in pl.range(20, init_values=(post_norm_tile__ssa_v0,)):
                    k0__ssa_v6: pl.Scalar[pl.INDEX] = kb__idx_v0_6 * 4
                    x_chunk__ssa_v3: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.slice(resid1_tile__rv_v2, [2, 4], [0, k0__ssa_v6])
                    gamma__ssa_v1: pl.Tensor[[1, 4], pl.FP32] = pl.tensor.slice(post_rms_weight__ssa_v0, [1, 4], [0, k0__ssa_v6])
                    t__tmp_v32: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.row_expand_mul(x_chunk__ssa_v3, inv_rms2__ssa_v0)
                    normed__ssa_v1: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.col_expand_mul(t__tmp_v32, gamma__ssa_v1)
                    t__tmp_v33: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.cast(normed__ssa_v1, target_type=pl.BF16, mode='round')
                    post_norm_tile__ssa_v3: pl.Tensor[[2, 80], pl.BF16] = pl.tensor.assemble(post_norm_tile__iter_v1, t__tmp_v33, [0, k0__ssa_v6])
                    post_norm_tile__rv_v2: pl.Tensor[[2, 80], pl.BF16] = pl.yield_(post_norm_tile__ssa_v3)
                down_tile__ssa_v0: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.create([2, 80], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                down_tile__ssa_v1: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.muls(down_tile__ssa_v0, 0.0)
                for mb__idx_v0, (down_tile__iter_v2,) in pl.range(25, init_values=(down_tile__ssa_v1,)):
                    m0__ssa_v0: pl.Scalar[pl.INDEX] = mb__idx_v0 * 16
                    gate_acc__ssa_v0: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.create([2, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    up_acc__ssa_v0: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.create([2, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    gate_acc__ssa_v1: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.muls(gate_acc__ssa_v0, 0.0)
                    up_acc__ssa_v1: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.muls(up_acc__ssa_v0, 0.0)
                    for kb__idx_v0_7, (gate_acc__iter_v2, up_acc__iter_v2) in pl.range(20, init_values=(gate_acc__ssa_v1, up_acc__ssa_v1)):
                        k0__ssa_v7: pl.Scalar[pl.INDEX] = kb__idx_v0_7 * 4
                        post_chunk__ssa_v0: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.slice(post_norm_tile__rv_v2, [2, 4], [0, k0__ssa_v7])
                        wg__ssa_v0: pl.Tensor[[4, 16], pl.BF16] = pl.tensor.slice(w_gate__ssa_v0, [4, 16], [k0__ssa_v7, m0__ssa_v0])
                        wu__ssa_v0: pl.Tensor[[4, 16], pl.BF16] = pl.tensor.slice(w_up__ssa_v0, [4, 16], [k0__ssa_v7, m0__ssa_v0])
                        t__tmp_v34: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.matmul(post_chunk__ssa_v0, wg__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        gate_acc__ssa_v4: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.add(gate_acc__iter_v2, t__tmp_v34)
                        t__tmp_v35: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.matmul(post_chunk__ssa_v0, wu__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        up_acc__ssa_v4: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.add(up_acc__iter_v2, t__tmp_v35)
                        gate_acc__rv_v3, up_acc__rv_v3 = pl.yield_(gate_acc__ssa_v4, up_acc__ssa_v4)
                    t__tmp_v36: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.neg(gate_acc__rv_v3)
                    t__tmp_v37: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.exp(t__tmp_v36)
                    t__tmp_v38: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.adds(t__tmp_v37, 1.0)
                    sigmoid_chunk__ssa_v0: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.recip(t__tmp_v38)
                    t__tmp_v39: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.mul(gate_acc__rv_v3, sigmoid_chunk__ssa_v0)
                    t__tmp_v40: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.mul(t__tmp_v39, up_acc__rv_v3)
                    mlp_chunk__ssa_v0: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.cast(t__tmp_v40, target_type=pl.BF16, mode='round')
                    for ob__idx_v0_3, (down_tile__iter_v4,) in pl.range(10, init_values=(down_tile__iter_v2,)):
                        o0__ssa_v2: pl.Scalar[pl.INDEX] = ob__idx_v0_3 * 8
                        down_prev__ssa_v0: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.slice(down_tile__iter_v4, [2, 8], [0, o0__ssa_v2])
                        wd__ssa_v0: pl.Tensor[[16, 8], pl.BF16] = pl.tensor.slice(w_down__ssa_v0, [16, 8], [m0__ssa_v0, o0__ssa_v2])
                        t__tmp_v41: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.matmul(mlp_chunk__ssa_v0, wd__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        down_part__ssa_v0: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.add(down_prev__ssa_v0, t__tmp_v41)
                        down_tile__ssa_v6: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.assemble(down_tile__iter_v4, down_part__ssa_v0, [0, o0__ssa_v2])
                        down_tile__rv_v5: pl.Tensor[[2, 80], pl.FP32] = pl.yield_(down_tile__ssa_v6)
                    down_tile__rv_v3: pl.Tensor[[2, 80], pl.FP32] = pl.yield_(down_tile__rv_v5)
                out_tile__ssa_v0: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.create([2, 80], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for ob__idx_v0_4, (out__iter_v5, out_tile__iter_v1) in pl.range(10, init_values=(out__iter_v3, out_tile__ssa_v0)):
                    o0__ssa_v3: pl.Scalar[pl.INDEX] = ob__idx_v0_4 * 8
                    t__tmp_v42: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.slice(down_tile__rv_v3, [2, 8], [0, o0__ssa_v3])
                    t__tmp_v43: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.slice(resid1_tile__rv_v2, [2, 8], [0, o0__ssa_v3])
                    out_chunk__ssa_v0: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.add(t__tmp_v42, t__tmp_v43)
                    out_tile__ssa_v3: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.assemble(out_tile__iter_v1, out_chunk__ssa_v0, [0, o0__ssa_v3])
                    t__tmp_v44: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.cast(out_chunk__ssa_v0, target_type=pl.BF16, mode='round')
                    t__tmp_v45: pl.Tensor[[1, 2, 8], pl.BF16] = pl.tensor.reshape(t__tmp_v44, [1, 2, 8])
                    out__ssa_v7: pl.Tensor[[1, 4, 80], pl.BF16] = pl.tensor.assemble(out__iter_v5, t__tmp_v45, [0 + b__cr_idx_v0 * 1, p0__ssa_v0, o0__ssa_v3])
                    out__rv_v6, out_tile__rv_v2 = pl.yield_(out__ssa_v7, out_tile__ssa_v3)
                t__tmp_v46: pl.Tensor[[1, 2, 80], pl.BF16] = pl.tensor.slice(target_states__ssa_v0, [1, 2, 80], [0 + b__cr_idx_v0 * 1, p0__ssa_v0, 0])
                t__tmp_v47: pl.Tensor[[1, 2, 80], pl.FP32] = pl.tensor.cast(t__tmp_v46, target_type=pl.FP32, mode='round')
                tgt_tile__ssa_v0: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.reshape(t__tmp_v47, [2, 80])
                diff_tile__ssa_v0: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.sub(out_tile__rv_v2, tgt_tile__ssa_v0)
                sq_tile__ssa_v0: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.mul(diff_tile__ssa_v0, diff_tile__ssa_v0)
                sq_row__ssa_v0: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.row_sum(sq_tile__ssa_v0)
                loss_prev__ssa_v0: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.slice(loss_acc__iter_v4, [2, 1], [0, 0])
                t__tmp_v48: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.add(loss_prev__ssa_v0, sq_row__ssa_v0)
                loss_acc__ssa_v6: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.assemble(loss_acc__iter_v4, t__tmp_v48, [0, 0])
                d_out__ssa_v0: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.muls(diff_tile__ssa_v0, 0.025)
                d_down__ssa_v0: pl.Tensor[[2, 80], pl.FP32] = d_out__ssa_v0
                d_resid1_bwd__ssa_v0: pl.Tensor[[2, 80], pl.FP32] = d_out__ssa_v0
                d_post_norm__ssa_v0: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.create([2, 80], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                d_post_norm__ssa_v1: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.muls(d_post_norm__ssa_v0, 0.0)
                for mb__idx_v0_1, (d_post_norm__iter_v2,) in pl.range(25, init_values=(d_post_norm__ssa_v1,)):
                    m0__ssa_v1: pl.Scalar[pl.INDEX] = mb__idx_v0_1 * 16
                    gate_r__ssa_v0: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.create([2, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    up_r__ssa_v0: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.create([2, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    gate_r__ssa_v1: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.muls(gate_r__ssa_v0, 0.0)
                    up_r__ssa_v1: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.muls(up_r__ssa_v0, 0.0)
                    for kb__idx_v0_8, (gate_r__iter_v2, up_r__iter_v2) in pl.range(20, init_values=(gate_r__ssa_v1, up_r__ssa_v1)):
                        k0__ssa_v8: pl.Scalar[pl.INDEX] = kb__idx_v0_8 * 4
                        post_c__ssa_v0: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.slice(post_norm_tile__rv_v2, [2, 4], [0, k0__ssa_v8])
                        wg_c__ssa_v0: pl.Tensor[[4, 16], pl.BF16] = pl.tensor.slice(w_gate__ssa_v0, [4, 16], [k0__ssa_v8, m0__ssa_v1])
                        wu_c__ssa_v0: pl.Tensor[[4, 16], pl.BF16] = pl.tensor.slice(w_up__ssa_v0, [4, 16], [k0__ssa_v8, m0__ssa_v1])
                        t__tmp_v49: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.matmul(post_c__ssa_v0, wg_c__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        gate_r__ssa_v4: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.add(gate_r__iter_v2, t__tmp_v49)
                        t__tmp_v50: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.matmul(post_c__ssa_v0, wu_c__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        up_r__ssa_v4: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.add(up_r__iter_v2, t__tmp_v50)
                        gate_r__rv_v3, up_r__rv_v3 = pl.yield_(gate_r__ssa_v4, up_r__ssa_v4)
                    t__tmp_v51: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.neg(gate_r__rv_v3)
                    t__tmp_v52: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.exp(t__tmp_v51)
                    t__tmp_v53: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.adds(t__tmp_v52, 1.0)
                    sig_r__ssa_v0: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.recip(t__tmp_v53)
                    d_mlp__ssa_v0: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.create([2, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    d_mlp__ssa_v1: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.muls(d_mlp__ssa_v0, 0.0)
                    for ob__idx_v0_5, (d_mlp__iter_v2,) in pl.range(10, init_values=(d_mlp__ssa_v1,)):
                        o0__ssa_v4: pl.Scalar[pl.INDEX] = ob__idx_v0_5 * 8
                        t__tmp_v54: pl.Tensor[[2, 8], pl.FP32] = pl.tensor.slice(d_down__ssa_v0, [2, 8], [0, o0__ssa_v4])
                        dd_c__ssa_v0: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.cast(t__tmp_v54, target_type=pl.BF16, mode='round')
                        wd_c__ssa_v0: pl.Tensor[[16, 8], pl.BF16] = pl.tensor.slice(w_down__ssa_v0, [16, 8], [m0__ssa_v1, o0__ssa_v4])
                        t__tmp_v55: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.matmul(dd_c__ssa_v0, wd_c__ssa_v0, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                        d_mlp__ssa_v4: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.add(d_mlp__iter_v2, t__tmp_v55)
                        d_mlp__rv_v3: pl.Tensor[[2, 16], pl.FP32] = pl.yield_(d_mlp__ssa_v4)
                    t__tmp_v56: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.muls(sig_r__ssa_v0, -1.0)
                    one_m_sig__ssa_v0: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.adds(t__tmp_v56, 1.0)
                    t__tmp_v57: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.mul(gate_r__rv_v3, one_m_sig__ssa_v0)
                    t__tmp_v58: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.adds(t__tmp_v57, 1.0)
                    silu_deriv__ssa_v0: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.mul(sig_r__ssa_v0, t__tmp_v58)
                    t__tmp_v59: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.mul(d_mlp__rv_v3, up_r__rv_v3)
                    t__tmp_v60: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.mul(t__tmp_v59, silu_deriv__ssa_v0)
                    d_gate_bf16__ssa_v0: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.cast(t__tmp_v60, target_type=pl.BF16, mode='round')
                    t__tmp_v61: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.mul(gate_r__rv_v3, sig_r__ssa_v0)
                    t__tmp_v62: pl.Tensor[[2, 16], pl.FP32] = pl.tensor.mul(d_mlp__rv_v3, t__tmp_v61)
                    d_up_bf16__ssa_v0: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.cast(t__tmp_v62, target_type=pl.BF16, mode='round')
                    for kb__idx_v0_9, (d_post_norm__iter_v4,) in pl.range(20, init_values=(d_post_norm__iter_v2,)):
                        k0__ssa_v9: pl.Scalar[pl.INDEX] = kb__idx_v0_9 * 4
                        dpn_old__ssa_v0: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.slice(d_post_norm__iter_v4, [2, 4], [0, k0__ssa_v9])
                        wg_c__ssa_v1: pl.Tensor[[4, 16], pl.BF16] = pl.tensor.slice(w_gate__ssa_v0, [4, 16], [k0__ssa_v9, m0__ssa_v1])
                        wu_c__ssa_v1: pl.Tensor[[4, 16], pl.BF16] = pl.tensor.slice(w_up__ssa_v0, [4, 16], [k0__ssa_v9, m0__ssa_v1])
                        t__tmp_v63: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.matmul(d_gate_bf16__ssa_v0, wg_c__ssa_v1, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                        dpn_tmp__ssa_v0: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.add(dpn_old__ssa_v0, t__tmp_v63)
                        t__tmp_v64: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.matmul(d_up_bf16__ssa_v0, wu_c__ssa_v1, a_trans=False, b_trans=True, c_matrix_nz=False, out_dtype=pl.FP32)
                        dpn_new__ssa_v0: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.add(dpn_tmp__ssa_v0, t__tmp_v64)
                        d_post_norm__ssa_v6: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.assemble(d_post_norm__iter_v4, dpn_new__ssa_v0, [0, k0__ssa_v9])
                        d_post_norm__rv_v5: pl.Tensor[[2, 80], pl.FP32] = pl.yield_(d_post_norm__ssa_v6)
                    d_post_norm__rv_v3: pl.Tensor[[2, 80], pl.FP32] = pl.yield_(d_post_norm__rv_v5)
                d_resid1__ssa_v0: pl.Tensor[[2, 80], pl.FP32] = pl.tensor.add(d_resid1_bwd__ssa_v0, d_post_norm__rv_v3)
                bwd_energy__ssa_v0: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.create([2, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                bwd_energy__ssa_v1: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.muls(bwd_energy__ssa_v0, 0.0)
                for kb__idx_v0_10, (bwd_energy__iter_v2,) in pl.range(20, init_values=(bwd_energy__ssa_v1,)):
                    k0__ssa_v10: pl.Scalar[pl.INDEX] = kb__idx_v0_10 * 4
                    dr_c__ssa_v0: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.slice(d_resid1__ssa_v0, [2, 4], [0, k0__ssa_v10])
                    t__tmp_v65: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.slice(q_proj_tile__rv_v2, [2, 4], [0, k0__ssa_v10])
                    q_c__ssa_v1: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.cast(t__tmp_v65, target_type=pl.FP32, mode='round')
                    t__tmp_v66: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.slice(k_proj_tile__rv_v2, [2, 4], [0, k0__ssa_v10])
                    k_c__ssa_v1: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.cast(t__tmp_v66, target_type=pl.FP32, mode='round')
                    t__tmp_v67: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.slice(v_proj_tile__rv_v2, [2, 4], [0, k0__ssa_v10])
                    v_bwd__ssa_v0: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.cast(t__tmp_v67, target_type=pl.FP32, mode='round')
                    t__tmp_v68: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.add(q_c__ssa_v1, k_c__ssa_v1)
                    t__tmp_v69: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.add(t__tmp_v68, v_bwd__ssa_v0)
                    t__tmp_v70: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.mul(dr_c__ssa_v0, t__tmp_v69)
                    contrib__ssa_v0: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.row_sum(t__tmp_v70)
                    bwd_energy__ssa_v4: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.add(bwd_energy__iter_v2, contrib__ssa_v0)
                    bwd_energy__rv_v3: pl.Tensor[[2, 1], pl.FP32] = pl.yield_(bwd_energy__ssa_v4)
                t__tmp_v71: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.row_sum(attn_w__ssa_v0)
                bwd_energy__ssa_v5: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.add(bwd_energy__rv_v3, t__tmp_v71)
                grad_sink__ssa_v0: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.muls(bwd_energy__ssa_v5, 0.0)
                btrans_scratch__rv_v4, loss_acc__rv_v5, out__rv_v4 = pl.yield_(btrans_scratch__rv_v6, loss_acc__ssa_v6, out__rv_v6)
            btrans_scratch__cr_rv_v1, loss_acc__cr_rv_v2, out__cr_rv_v1 = pl.yield_(btrans_scratch__rv_v4, loss_acc__rv_v5, out__rv_v4)
        return loss_acc__cr_rv_v2, out__cr_rv_v1
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_32b_training_forward_and_backward_layer_incore_3(self, t__tmp_v72: pl.Tensor[[2, 16], pl.BF16]) -> pl.Tensor[[2, 16], pl.BF16]:
        proxy_mlp__ssa_v0: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.cast(t__tmp_v72, target_type=pl.BF16, mode='round')
        return proxy_mlp__ssa_v0
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_32b_training_forward_and_backward_layer_incore_4(self, grad_w_down__ssa_v1: pl.Tensor[[400, 80], pl.FP32], mom_w_down__ssa_v0: pl.Tensor[[400, 80], pl.FP32], muon_scratch__ssa_v0: pl.Tensor[[16, 16], pl.BF16], proxy_mlp_t__ssa_v0: pl.Tensor[[2, 16], pl.BF16], proxy_scratch__ssa_v1: pl.Tensor[[2, 16], pl.BF16], target_states__ssa_v0: pl.Tensor[[1, 4, 80], pl.BF16]) -> tuple[pl.Tensor[[400, 80], pl.FP32], pl.Tensor[[400, 80], pl.FP32], pl.Tensor[[16, 16], pl.BF16], pl.Tensor[[2, 16], pl.BF16]]:
        for qb__idx_v0, (grad_w_down__iter_v2, mom_w_down__iter_v1, muon_scratch__iter_v1, proxy_scratch__iter_v2) in pl.range(10, init_values=(grad_w_down__ssa_v1, mom_w_down__ssa_v0, muon_scratch__ssa_v0, proxy_scratch__ssa_v1)):
            q0__ssa_v1: pl.Scalar[pl.INDEX] = qb__idx_v0 * 8
            t__tmp_v73: pl.Tensor[[1, 2, 8], pl.BF16] = pl.tensor.slice(target_states__ssa_v0, [1, 2, 8], [0, 0, q0__ssa_v1])
            t__tmp_v74: pl.Tensor[[1, 2, 8], pl.BF16] = pl.tensor.cast(t__tmp_v73, target_type=pl.BF16, mode='round')
            proxy_go__ssa_v0: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.reshape(t__tmp_v74, [2, 8])
            proxy_scratch__ssa_v4: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.assemble(proxy_scratch__iter_v2, proxy_go__ssa_v0, [0, 0])
            proxy_go_t__ssa_v0: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.slice(proxy_scratch__ssa_v4, [2, 8], [0, 0])
            grad_down_raw__ssa_v0: pl.Tensor[[16, 8], pl.FP32] = pl.tensor.matmul(proxy_mlp_t__ssa_v0, proxy_go_t__ssa_v0, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
            mom_down_prev__ssa_v0: pl.Tensor[[16, 8], pl.FP32] = pl.tensor.slice(mom_w_down__iter_v1, [16, 8], [0, q0__ssa_v1])
            t__tmp_v75: pl.Tensor[[16, 8], pl.FP32] = pl.tensor.muls(mom_down_prev__ssa_v0, 0.95)
            t__tmp_v76: pl.Tensor[[16, 8], pl.FP32] = pl.tensor.muls(grad_down_raw__ssa_v0, 0.05)
            mom_down_new__ssa_v0: pl.Tensor[[16, 8], pl.FP32] = pl.tensor.add(t__tmp_v75, t__tmp_v76)
            muon_down__ssa_v0: pl.Tensor[[16, 8], pl.FP32] = mom_down_new__ssa_v0
            t__tmp_v77: pl.Tensor[[16, 8], pl.BF16] = pl.tensor.cast(muon_down__ssa_v0, target_type=pl.BF16, mode='round')
            muon_scratch__ssa_v3: pl.Tensor[[16, 16], pl.BF16] = pl.tensor.assemble(muon_scratch__iter_v1, t__tmp_v77, [0, 0])
            for ___idx_v0, (muon_down__iter_v1,) in pl.range(2, init_values=(muon_down__ssa_v0,)):
                gram__ssa_v0: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.create([8, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                gram__ssa_v1: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.muls(gram__ssa_v0, 0.0)
                for tb__idx_v0, (gram__iter_v2,) in pl.range(8, init_values=(gram__ssa_v1,)):
                    t0__ssa_v0: pl.Scalar[pl.INDEX] = tb__idx_v0 * 2
                    m_blk__ssa_v0: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.slice(muon_scratch__ssa_v3, [2, 8], [t0__ssa_v0, 0])
                    m_blk_t__ssa_v0: pl.Tensor[[8, 2], pl.BF16] = pl.tensor.transpose(m_blk__ssa_v0, 0, 1)
                    t__tmp_v78: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.matmul(m_blk_t__ssa_v0, m_blk__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    gram__ssa_v4: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.add(gram__iter_v2, t__tmp_v78)
                    gram__rv_v3: pl.Tensor[[8, 8], pl.FP32] = pl.yield_(gram__ssa_v4)
                t__tmp_v79: pl.Tensor[[16, 8], pl.FP32] = pl.tensor.muls(muon_down__iter_v1, 1.5)
                t__tmp_v80: pl.Tensor[[16, 8], pl.FP32] = pl.tensor.matmul(muon_down__iter_v1, gram__rv_v3, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                t__tmp_v81: pl.Tensor[[16, 8], pl.FP32] = pl.tensor.muls(t__tmp_v80, -0.5)
                muon_down__ssa_v3: pl.Tensor[[16, 8], pl.FP32] = pl.tensor.add(t__tmp_v79, t__tmp_v81)
                muon_down__rv_v2: pl.Tensor[[16, 8], pl.FP32] = pl.yield_(muon_down__ssa_v3)
            t__tmp_v82: pl.Tensor[[16, 8], pl.FP32] = pl.tensor.muls(muon_down__rv_v2, -0.0002)
            grad_w_down__ssa_v4: pl.Tensor[[400, 80], pl.FP32] = pl.tensor.assemble(grad_w_down__iter_v2, t__tmp_v82, [0, q0__ssa_v1])
            mom_w_down__ssa_v3: pl.Tensor[[400, 80], pl.FP32] = pl.tensor.assemble(mom_w_down__iter_v1, mom_down_new__ssa_v0, [0, q0__ssa_v1])
            grad_w_down__rv_v3, mom_w_down__rv_v2, muon_scratch__rv_v2, proxy_scratch__rv_v3 = pl.yield_(grad_w_down__ssa_v4, mom_w_down__ssa_v3, muon_scratch__ssa_v3, proxy_scratch__ssa_v4)
        return grad_w_down__rv_v3, mom_w_down__rv_v2, muon_scratch__rv_v2, proxy_scratch__rv_v3
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_32b_training_forward_and_backward_layer_incore_5(self, t__tmp_v83: pl.Tensor[[2, 4], pl.BF16]) -> pl.Tensor[[2, 4], pl.BF16]:
        proxy_ctx__ssa_v0: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.cast(t__tmp_v83, target_type=pl.BF16, mode='round')
        return proxy_ctx__ssa_v0
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_32b_training_forward_and_backward_layer_incore_6(self, t__tmp_v84: pl.Tensor[[1, 2, 4], pl.BF16]) -> pl.Tensor[[1, 2, 4], pl.BF16]:
        t__tmp_v85: pl.Tensor[[1, 2, 4], pl.BF16] = pl.tensor.cast(t__tmp_v84, target_type=pl.BF16, mode='round')
        return t__tmp_v85
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_32b_training_forward_and_backward_layer_incore_7(self, grad_wk__ssa_v1: pl.Tensor[[80, 80], pl.FP32], grad_wo__ssa_v1: pl.Tensor[[80, 80], pl.FP32], grad_wq__ssa_v1: pl.Tensor[[80, 80], pl.FP32], grad_wv__ssa_v1: pl.Tensor[[80, 80], pl.FP32], mom_wk__ssa_v0: pl.Tensor[[80, 80], pl.FP32], mom_wo__ssa_v0: pl.Tensor[[80, 80], pl.FP32], mom_wq__ssa_v0: pl.Tensor[[80, 80], pl.FP32], mom_wv__ssa_v0: pl.Tensor[[80, 80], pl.FP32], muon_scratch__rv_v2: pl.Tensor[[16, 16], pl.BF16], proxy_ctx_t__ssa_v0: pl.Tensor[[2, 4], pl.BF16], proxy_n_t__ssa_v0: pl.Tensor[[2, 4], pl.BF16], proxy_scratch__ssa_v6: pl.Tensor[[2, 16], pl.BF16], target_states__ssa_v0: pl.Tensor[[1, 4, 80], pl.BF16]) -> tuple[pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[16, 16], pl.BF16], pl.Tensor[[2, 16], pl.BF16]]:
        for qb__idx_v0, (grad_wk__iter_v2, grad_wo__iter_v2, grad_wq__iter_v2, grad_wv__iter_v2, mom_wk__iter_v1, mom_wo__iter_v1, mom_wq__iter_v1, mom_wv__iter_v1, muon_scratch__iter_v4, proxy_scratch__iter_v7) in pl.range(10, init_values=(grad_wk__ssa_v1, grad_wo__ssa_v1, grad_wq__ssa_v1, grad_wv__ssa_v1, mom_wk__ssa_v0, mom_wo__ssa_v0, mom_wq__ssa_v0, mom_wv__ssa_v0, muon_scratch__rv_v2, proxy_scratch__ssa_v6)):
            q0__ssa_v2: pl.Scalar[pl.INDEX] = qb__idx_v0 * 8
            t__tmp_v86: pl.Tensor[[1, 2, 8], pl.BF16] = pl.tensor.slice(target_states__ssa_v0, [1, 2, 8], [0, 0, q0__ssa_v2])
            t__tmp_v87: pl.Tensor[[1, 2, 8], pl.BF16] = pl.tensor.cast(t__tmp_v86, target_type=pl.BF16, mode='round')
            proxy_tgt__ssa_v0: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.reshape(t__tmp_v87, [2, 8])
            proxy_scratch__ssa_v9: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.assemble(proxy_scratch__iter_v7, proxy_tgt__ssa_v0, [0, 0])
            proxy_tgt_t__ssa_v0: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.slice(proxy_scratch__ssa_v9, [2, 8], [0, 0])
            grad_wo_raw__ssa_v0: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.matmul(proxy_ctx_t__ssa_v0, proxy_tgt_t__ssa_v0, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
            mom_wo_prev__ssa_v0: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.slice(mom_wo__iter_v1, [4, 8], [0, q0__ssa_v2])
            t__tmp_v88: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.muls(mom_wo_prev__ssa_v0, 0.95)
            t__tmp_v89: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.muls(grad_wo_raw__ssa_v0, 0.05)
            mom_wo_new__ssa_v0: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.add(t__tmp_v88, t__tmp_v89)
            muon_wo__ssa_v0: pl.Tensor[[4, 8], pl.FP32] = mom_wo_new__ssa_v0
            t__tmp_v90: pl.Tensor[[4, 8], pl.BF16] = pl.tensor.cast(muon_wo__ssa_v0, target_type=pl.BF16, mode='round')
            muon_scratch__ssa_v6: pl.Tensor[[16, 16], pl.BF16] = pl.tensor.assemble(muon_scratch__iter_v4, t__tmp_v90, [0, 0])
            for ___idx_v0, (muon_wo__iter_v1,) in pl.range(2, init_values=(muon_wo__ssa_v0,)):
                gram__ssa_v5: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.create([8, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                gram__ssa_v6: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.muls(gram__ssa_v5, 0.0)
                for tb__idx_v0, (gram__iter_v7,) in pl.range(2, init_values=(gram__ssa_v6,)):
                    t0__ssa_v1: pl.Scalar[pl.INDEX] = tb__idx_v0 * 2
                    m_blk__ssa_v1: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.slice(muon_scratch__ssa_v6, [2, 8], [t0__ssa_v1, 0])
                    m_blk_t__ssa_v1: pl.Tensor[[8, 2], pl.BF16] = pl.tensor.transpose(m_blk__ssa_v1, 0, 1)
                    t__tmp_v91: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.matmul(m_blk_t__ssa_v1, m_blk__ssa_v1, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    gram__ssa_v9: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.add(gram__iter_v7, t__tmp_v91)
                    gram__rv_v8: pl.Tensor[[8, 8], pl.FP32] = pl.yield_(gram__ssa_v9)
                t__tmp_v92: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.muls(muon_wo__iter_v1, 1.5)
                t__tmp_v93: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.matmul(muon_wo__iter_v1, gram__rv_v8, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                t__tmp_v94: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.muls(t__tmp_v93, -0.5)
                muon_wo__ssa_v3: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.add(t__tmp_v92, t__tmp_v94)
                muon_wo__rv_v2: pl.Tensor[[4, 8], pl.FP32] = pl.yield_(muon_wo__ssa_v3)
            t__tmp_v95: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.muls(muon_wo__rv_v2, -0.0002)
            grad_wo__ssa_v4: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.assemble(grad_wo__iter_v2, t__tmp_v95, [0, q0__ssa_v2])
            mom_wo__ssa_v3: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.assemble(mom_wo__iter_v1, mom_wo_new__ssa_v0, [0, q0__ssa_v2])
            grad_wq_raw__ssa_v0: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.matmul(proxy_n_t__ssa_v0, proxy_tgt_t__ssa_v0, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
            mom_wq_prev__ssa_v0: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.slice(mom_wq__iter_v1, [4, 8], [0, q0__ssa_v2])
            t__tmp_v96: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.muls(mom_wq_prev__ssa_v0, 0.95)
            t__tmp_v97: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.muls(grad_wq_raw__ssa_v0, 0.05)
            mom_wq_new__ssa_v0: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.add(t__tmp_v96, t__tmp_v97)
            muon_wq__ssa_v0: pl.Tensor[[4, 8], pl.FP32] = mom_wq_new__ssa_v0
            t__tmp_v98: pl.Tensor[[4, 8], pl.BF16] = pl.tensor.cast(muon_wq__ssa_v0, target_type=pl.BF16, mode='round')
            muon_scratch__ssa_v7: pl.Tensor[[16, 16], pl.BF16] = pl.tensor.assemble(muon_scratch__ssa_v6, t__tmp_v98, [0, 0])
            for ___idx_v0_1, (muon_wq__iter_v1,) in pl.range(2, init_values=(muon_wq__ssa_v0,)):
                gram__ssa_v10: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.create([8, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                gram__ssa_v11: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.muls(gram__ssa_v10, 0.0)
                for tb__idx_v0_1, (gram__iter_v12,) in pl.range(2, init_values=(gram__ssa_v11,)):
                    t0__ssa_v2: pl.Scalar[pl.INDEX] = tb__idx_v0_1 * 2
                    m_blk__ssa_v2: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.slice(muon_scratch__ssa_v7, [2, 8], [t0__ssa_v2, 0])
                    m_blk_t__ssa_v2: pl.Tensor[[8, 2], pl.BF16] = pl.tensor.transpose(m_blk__ssa_v2, 0, 1)
                    t__tmp_v99: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.matmul(m_blk_t__ssa_v2, m_blk__ssa_v2, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    gram__ssa_v14: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.add(gram__iter_v12, t__tmp_v99)
                    gram__rv_v13: pl.Tensor[[8, 8], pl.FP32] = pl.yield_(gram__ssa_v14)
                t__tmp_v100: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.muls(muon_wq__iter_v1, 1.5)
                t__tmp_v101: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.matmul(muon_wq__iter_v1, gram__rv_v13, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                t__tmp_v102: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.muls(t__tmp_v101, -0.5)
                muon_wq__ssa_v3: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.add(t__tmp_v100, t__tmp_v102)
                muon_wq__rv_v2: pl.Tensor[[4, 8], pl.FP32] = pl.yield_(muon_wq__ssa_v3)
            t__tmp_v103: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.muls(muon_wq__rv_v2, -0.0002)
            grad_wq__ssa_v4: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.assemble(grad_wq__iter_v2, t__tmp_v103, [0, q0__ssa_v2])
            mom_wq__ssa_v3: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.assemble(mom_wq__iter_v1, mom_wq_new__ssa_v0, [0, q0__ssa_v2])
            grad_wk_raw__ssa_v0: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.matmul(proxy_n_t__ssa_v0, proxy_tgt_t__ssa_v0, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
            mom_wk_prev__ssa_v0: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.slice(mom_wk__iter_v1, [4, 8], [0, q0__ssa_v2])
            t__tmp_v104: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.muls(mom_wk_prev__ssa_v0, 0.95)
            t__tmp_v105: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.muls(grad_wk_raw__ssa_v0, 0.05)
            mom_wk_new__ssa_v0: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.add(t__tmp_v104, t__tmp_v105)
            muon_wk__ssa_v0: pl.Tensor[[4, 8], pl.FP32] = mom_wk_new__ssa_v0
            t__tmp_v106: pl.Tensor[[4, 8], pl.BF16] = pl.tensor.cast(muon_wk__ssa_v0, target_type=pl.BF16, mode='round')
            muon_scratch__ssa_v8: pl.Tensor[[16, 16], pl.BF16] = pl.tensor.assemble(muon_scratch__ssa_v7, t__tmp_v106, [0, 0])
            for ___idx_v0_2, (muon_wk__iter_v1,) in pl.range(2, init_values=(muon_wk__ssa_v0,)):
                gram__ssa_v15: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.create([8, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                gram__ssa_v16: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.muls(gram__ssa_v15, 0.0)
                for tb__idx_v0_2, (gram__iter_v17,) in pl.range(2, init_values=(gram__ssa_v16,)):
                    t0__ssa_v3: pl.Scalar[pl.INDEX] = tb__idx_v0_2 * 2
                    m_blk__ssa_v3: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.slice(muon_scratch__ssa_v8, [2, 8], [t0__ssa_v3, 0])
                    m_blk_t__ssa_v3: pl.Tensor[[8, 2], pl.BF16] = pl.tensor.transpose(m_blk__ssa_v3, 0, 1)
                    t__tmp_v107: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.matmul(m_blk_t__ssa_v3, m_blk__ssa_v3, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    gram__ssa_v19: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.add(gram__iter_v17, t__tmp_v107)
                    gram__rv_v18: pl.Tensor[[8, 8], pl.FP32] = pl.yield_(gram__ssa_v19)
                t__tmp_v108: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.muls(muon_wk__iter_v1, 1.5)
                t__tmp_v109: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.matmul(muon_wk__iter_v1, gram__rv_v18, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                t__tmp_v110: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.muls(t__tmp_v109, -0.5)
                muon_wk__ssa_v3: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.add(t__tmp_v108, t__tmp_v110)
                muon_wk__rv_v2: pl.Tensor[[4, 8], pl.FP32] = pl.yield_(muon_wk__ssa_v3)
            t__tmp_v111: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.muls(muon_wk__rv_v2, -0.0002)
            grad_wk__ssa_v4: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.assemble(grad_wk__iter_v2, t__tmp_v111, [0, q0__ssa_v2])
            mom_wk__ssa_v3: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.assemble(mom_wk__iter_v1, mom_wk_new__ssa_v0, [0, q0__ssa_v2])
            grad_wv_raw__ssa_v0: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.matmul(proxy_n_t__ssa_v0, proxy_tgt_t__ssa_v0, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
            mom_wv_prev__ssa_v0: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.slice(mom_wv__iter_v1, [4, 8], [0, q0__ssa_v2])
            t__tmp_v112: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.muls(mom_wv_prev__ssa_v0, 0.95)
            t__tmp_v113: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.muls(grad_wv_raw__ssa_v0, 0.05)
            mom_wv_new__ssa_v0: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.add(t__tmp_v112, t__tmp_v113)
            muon_wv__ssa_v0: pl.Tensor[[4, 8], pl.FP32] = mom_wv_new__ssa_v0
            t__tmp_v114: pl.Tensor[[4, 8], pl.BF16] = pl.tensor.cast(muon_wv__ssa_v0, target_type=pl.BF16, mode='round')
            muon_scratch__ssa_v9: pl.Tensor[[16, 16], pl.BF16] = pl.tensor.assemble(muon_scratch__ssa_v8, t__tmp_v114, [0, 0])
            for ___idx_v0_3, (muon_wv__iter_v1,) in pl.range(2, init_values=(muon_wv__ssa_v0,)):
                gram__ssa_v20: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.create([8, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                gram__ssa_v21: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.muls(gram__ssa_v20, 0.0)
                for tb__idx_v0_3, (gram__iter_v22,) in pl.range(2, init_values=(gram__ssa_v21,)):
                    t0__ssa_v4: pl.Scalar[pl.INDEX] = tb__idx_v0_3 * 2
                    m_blk__ssa_v4: pl.Tensor[[2, 8], pl.BF16] = pl.tensor.slice(muon_scratch__ssa_v9, [2, 8], [t0__ssa_v4, 0])
                    m_blk_t__ssa_v4: pl.Tensor[[8, 2], pl.BF16] = pl.tensor.transpose(m_blk__ssa_v4, 0, 1)
                    t__tmp_v115: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.matmul(m_blk_t__ssa_v4, m_blk__ssa_v4, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    gram__ssa_v24: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.add(gram__iter_v22, t__tmp_v115)
                    gram__rv_v23: pl.Tensor[[8, 8], pl.FP32] = pl.yield_(gram__ssa_v24)
                t__tmp_v116: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.muls(muon_wv__iter_v1, 1.5)
                t__tmp_v117: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.matmul(muon_wv__iter_v1, gram__rv_v23, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                t__tmp_v118: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.muls(t__tmp_v117, -0.5)
                muon_wv__ssa_v3: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.add(t__tmp_v116, t__tmp_v118)
                muon_wv__rv_v2: pl.Tensor[[4, 8], pl.FP32] = pl.yield_(muon_wv__ssa_v3)
            t__tmp_v119: pl.Tensor[[4, 8], pl.FP32] = pl.tensor.muls(muon_wv__rv_v2, -0.0002)
            grad_wv__ssa_v4: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.assemble(grad_wv__iter_v2, t__tmp_v119, [0, q0__ssa_v2])
            mom_wv__ssa_v3: pl.Tensor[[80, 80], pl.FP32] = pl.tensor.assemble(mom_wv__iter_v1, mom_wv_new__ssa_v0, [0, q0__ssa_v2])
            grad_wk__rv_v3, grad_wo__rv_v3, grad_wq__rv_v3, grad_wv__rv_v3, mom_wk__rv_v2, mom_wo__rv_v2, mom_wq__rv_v2, mom_wv__rv_v2, muon_scratch__rv_v5, proxy_scratch__rv_v8 = pl.yield_(grad_wk__ssa_v4, grad_wo__ssa_v4, grad_wq__ssa_v4, grad_wv__ssa_v4, mom_wk__ssa_v3, mom_wo__ssa_v3, mom_wq__ssa_v3, mom_wv__ssa_v3, muon_scratch__ssa_v9, proxy_scratch__ssa_v9)
        return grad_wk__rv_v3, grad_wo__rv_v3, grad_wq__rv_v3, grad_wv__rv_v3, mom_wk__rv_v2, mom_wo__rv_v2, mom_wq__rv_v2, mom_wv__rv_v2, muon_scratch__rv_v5, proxy_scratch__rv_v8
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_32b_training_forward_and_backward_layer_incore_8(self, t__tmp_v120: pl.Tensor[[1, 2, 4], pl.BF16]) -> pl.Tensor[[1, 2, 4], pl.BF16]:
        t__tmp_v121: pl.Tensor[[1, 2, 4], pl.BF16] = pl.tensor.cast(t__tmp_v120, target_type=pl.BF16, mode='round')
        return t__tmp_v121
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_32b_training_forward_and_backward_layer_incore_9(self, grad_w_gate__ssa_v1: pl.Tensor[[80, 400], pl.FP32], grad_w_up__ssa_v1: pl.Tensor[[80, 400], pl.FP32], mom_w_gate__ssa_v0: pl.Tensor[[80, 400], pl.FP32], mom_w_up__ssa_v0: pl.Tensor[[80, 400], pl.FP32], muon_scratch__rv_v5: pl.Tensor[[16, 16], pl.BF16], proxy_post_t__ssa_v0: pl.Tensor[[2, 4], pl.BF16], proxy_scratch__ssa_v10: pl.Tensor[[2, 16], pl.BF16], w_gate__ssa_v0: pl.Tensor[[80, 400], pl.BF16], w_up__ssa_v0: pl.Tensor[[80, 400], pl.BF16]) -> tuple[pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[80, 400], pl.FP32]]:
        for mb__idx_v0, (grad_w_gate__iter_v2, grad_w_up__iter_v2, mom_w_gate__iter_v1, mom_w_up__iter_v1, muon_scratch__iter_v10, proxy_scratch__iter_v11) in pl.range(25, init_values=(grad_w_gate__ssa_v1, grad_w_up__ssa_v1, mom_w_gate__ssa_v0, mom_w_up__ssa_v0, muon_scratch__rv_v5, proxy_scratch__ssa_v10)):
            m0__ssa_v2: pl.Scalar[pl.INDEX] = mb__idx_v0 * 16
            t__tmp_v122: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.slice(w_gate__ssa_v0, [2, 16], [0, m0__ssa_v2])
            proxy_gg__ssa_v0: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.cast(t__tmp_v122, target_type=pl.BF16, mode='round')
            t__tmp_v123: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.slice(w_up__ssa_v0, [2, 16], [0, m0__ssa_v2])
            proxy_gu__ssa_v0: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.cast(t__tmp_v123, target_type=pl.BF16, mode='round')
            proxy_scratch__ssa_v13: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.assemble(proxy_scratch__iter_v11, proxy_gg__ssa_v0, [0, 0])
            proxy_gg_t__ssa_v0: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.slice(proxy_scratch__ssa_v13, [2, 16], [0, 0])
            proxy_scratch__ssa_v14: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.assemble(proxy_scratch__ssa_v13, proxy_gu__ssa_v0, [0, 0])
            proxy_gu_t__ssa_v0: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.slice(proxy_scratch__ssa_v14, [2, 16], [0, 0])
            grad_wg_raw__ssa_v0: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.matmul(proxy_post_t__ssa_v0, proxy_gg_t__ssa_v0, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
            grad_wu_raw__ssa_v0: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.matmul(proxy_post_t__ssa_v0, proxy_gu_t__ssa_v0, a_trans=True, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
            mom_wg_prev__ssa_v0: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.slice(mom_w_gate__iter_v1, [4, 16], [0, m0__ssa_v2])
            mom_wu_prev__ssa_v0: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.slice(mom_w_up__iter_v1, [4, 16], [0, m0__ssa_v2])
            t__tmp_v124: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.muls(mom_wg_prev__ssa_v0, 0.95)
            t__tmp_v125: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.muls(grad_wg_raw__ssa_v0, 0.05)
            mom_wg_new__ssa_v0: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.add(t__tmp_v124, t__tmp_v125)
            t__tmp_v126: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.muls(mom_wu_prev__ssa_v0, 0.95)
            t__tmp_v127: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.muls(grad_wu_raw__ssa_v0, 0.05)
            mom_wu_new__ssa_v0: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.add(t__tmp_v126, t__tmp_v127)
            muon_wg__ssa_v0: pl.Tensor[[4, 16], pl.FP32] = mom_wg_new__ssa_v0
            t__tmp_v128: pl.Tensor[[4, 16], pl.BF16] = pl.tensor.cast(muon_wg__ssa_v0, target_type=pl.BF16, mode='round')
            muon_scratch__ssa_v12: pl.Tensor[[16, 16], pl.BF16] = pl.tensor.assemble(muon_scratch__iter_v10, t__tmp_v128, [0, 0])
            for ___idx_v0, (muon_wg__iter_v1,) in pl.range(2, init_values=(muon_wg__ssa_v0,)):
                ns_acc_g__ssa_v0: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.create([4, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                ns_acc_g__ssa_v1: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.muls(ns_acc_g__ssa_v0, 0.0)
                muon_wg_bf__ssa_v0: pl.Tensor[[4, 16], pl.BF16] = pl.tensor.cast(muon_wg__iter_v1, target_type=pl.BF16, mode='round')
                for tb__idx_v0, (ns_acc_g__iter_v2,) in pl.range(2, init_values=(ns_acc_g__ssa_v1,)):
                    t0__ssa_v5: pl.Scalar[pl.INDEX] = tb__idx_v0 * 2
                    m_blk_g__ssa_v0: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.slice(muon_scratch__ssa_v12, [2, 16], [t0__ssa_v5, 0])
                    m_blk_gt__ssa_v0: pl.Tensor[[16, 2], pl.BF16] = pl.tensor.transpose(m_blk_g__ssa_v0, 0, 1)
                    tmp_g__ssa_v0: pl.Tensor[[4, 2], pl.FP32] = pl.tensor.matmul(muon_wg_bf__ssa_v0, m_blk_gt__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    tmp_g_bf__ssa_v0: pl.Tensor[[4, 2], pl.BF16] = pl.tensor.cast(tmp_g__ssa_v0, target_type=pl.BF16, mode='round')
                    t__tmp_v129: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.matmul(tmp_g_bf__ssa_v0, m_blk_g__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    ns_acc_g__ssa_v4: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.add(ns_acc_g__iter_v2, t__tmp_v129)
                    ns_acc_g__rv_v3: pl.Tensor[[4, 16], pl.FP32] = pl.yield_(ns_acc_g__ssa_v4)
                t__tmp_v130: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.muls(muon_wg__iter_v1, 1.5)
                t__tmp_v131: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.muls(ns_acc_g__rv_v3, -0.5)
                muon_wg__ssa_v3: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.add(t__tmp_v130, t__tmp_v131)
                muon_wg__rv_v2: pl.Tensor[[4, 16], pl.FP32] = pl.yield_(muon_wg__ssa_v3)
            muon_wu__ssa_v0: pl.Tensor[[4, 16], pl.FP32] = mom_wu_new__ssa_v0
            t__tmp_v132: pl.Tensor[[4, 16], pl.BF16] = pl.tensor.cast(muon_wu__ssa_v0, target_type=pl.BF16, mode='round')
            muon_scratch__ssa_v13: pl.Tensor[[16, 16], pl.BF16] = pl.tensor.assemble(muon_scratch__ssa_v12, t__tmp_v132, [0, 0])
            for ___idx_v0_1, (muon_wu__iter_v1,) in pl.range(2, init_values=(muon_wu__ssa_v0,)):
                ns_acc_u__ssa_v0: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.create([4, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                ns_acc_u__ssa_v1: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.muls(ns_acc_u__ssa_v0, 0.0)
                muon_wu_bf__ssa_v0: pl.Tensor[[4, 16], pl.BF16] = pl.tensor.cast(muon_wu__iter_v1, target_type=pl.BF16, mode='round')
                for tb__idx_v0_1, (ns_acc_u__iter_v2,) in pl.range(2, init_values=(ns_acc_u__ssa_v1,)):
                    t0__ssa_v6: pl.Scalar[pl.INDEX] = tb__idx_v0_1 * 2
                    m_blk_u__ssa_v0: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.slice(muon_scratch__ssa_v13, [2, 16], [t0__ssa_v6, 0])
                    m_blk_ut__ssa_v0: pl.Tensor[[16, 2], pl.BF16] = pl.tensor.transpose(m_blk_u__ssa_v0, 0, 1)
                    tmp_u__ssa_v0: pl.Tensor[[4, 2], pl.FP32] = pl.tensor.matmul(muon_wu_bf__ssa_v0, m_blk_ut__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    tmp_u_bf__ssa_v0: pl.Tensor[[4, 2], pl.BF16] = pl.tensor.cast(tmp_u__ssa_v0, target_type=pl.BF16, mode='round')
                    t__tmp_v133: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.matmul(tmp_u_bf__ssa_v0, m_blk_u__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                    ns_acc_u__ssa_v4: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.add(ns_acc_u__iter_v2, t__tmp_v133)
                    ns_acc_u__rv_v3: pl.Tensor[[4, 16], pl.FP32] = pl.yield_(ns_acc_u__ssa_v4)
                t__tmp_v134: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.muls(muon_wu__iter_v1, 1.5)
                t__tmp_v135: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.muls(ns_acc_u__rv_v3, -0.5)
                muon_wu__ssa_v3: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.add(t__tmp_v134, t__tmp_v135)
                muon_wu__rv_v2: pl.Tensor[[4, 16], pl.FP32] = pl.yield_(muon_wu__ssa_v3)
            t__tmp_v136: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.muls(muon_wg__rv_v2, -0.0002)
            grad_w_gate__ssa_v4: pl.Tensor[[80, 400], pl.FP32] = pl.tensor.assemble(grad_w_gate__iter_v2, t__tmp_v136, [0, m0__ssa_v2])
            t__tmp_v137: pl.Tensor[[4, 16], pl.FP32] = pl.tensor.muls(muon_wu__rv_v2, -0.0002)
            grad_w_up__ssa_v4: pl.Tensor[[80, 400], pl.FP32] = pl.tensor.assemble(grad_w_up__iter_v2, t__tmp_v137, [0, m0__ssa_v2])
            mom_w_gate__ssa_v3: pl.Tensor[[80, 400], pl.FP32] = pl.tensor.assemble(mom_w_gate__iter_v1, mom_wg_new__ssa_v0, [0, m0__ssa_v2])
            mom_w_up__ssa_v3: pl.Tensor[[80, 400], pl.FP32] = pl.tensor.assemble(mom_w_up__iter_v1, mom_wu_new__ssa_v0, [0, m0__ssa_v2])
            grad_w_gate__rv_v3, grad_w_up__rv_v3, mom_w_gate__rv_v2, mom_w_up__rv_v2, muon_scratch__rv_v11, proxy_scratch__rv_v12 = pl.yield_(grad_w_gate__ssa_v4, grad_w_up__ssa_v4, mom_w_gate__ssa_v3, mom_w_up__ssa_v3, muon_scratch__ssa_v13, proxy_scratch__ssa_v14)
        return grad_w_gate__rv_v3, grad_w_up__rv_v3, mom_w_gate__rv_v2, mom_w_up__rv_v2
    @pl.function(type=pl.FunctionType.Orchestration)
    def qwen3_32b_training_forward_and_backward_layer(self, hidden_states__ssa_v0: pl.Tensor[[1, 4, 80], pl.BF16], target_states__ssa_v0: pl.Tensor[[1, 4, 80], pl.BF16], input_rms_weight__ssa_v0: pl.Tensor[[1, 80], pl.FP32], post_rms_weight__ssa_v0: pl.Tensor[[1, 80], pl.FP32], wq__ssa_v0: pl.Tensor[[80, 80], pl.BF16], wk__ssa_v0: pl.Tensor[[80, 80], pl.BF16], wv__ssa_v0: pl.Tensor[[80, 80], pl.BF16], wo__ssa_v0: pl.Tensor[[80, 80], pl.BF16], w_gate__ssa_v0: pl.Tensor[[80, 400], pl.BF16], w_up__ssa_v0: pl.Tensor[[80, 400], pl.BF16], w_down__ssa_v0: pl.Tensor[[400, 80], pl.BF16], mom_wq__ssa_v0: pl.Tensor[[80, 80], pl.FP32], mom_wk__ssa_v0: pl.Tensor[[80, 80], pl.FP32], mom_wv__ssa_v0: pl.Tensor[[80, 80], pl.FP32], mom_wo__ssa_v0: pl.Tensor[[80, 80], pl.FP32], mom_w_gate__ssa_v0: pl.Tensor[[80, 400], pl.FP32], mom_w_up__ssa_v0: pl.Tensor[[80, 400], pl.FP32], mom_w_down__ssa_v0: pl.Tensor[[400, 80], pl.FP32], grad_wq__ssa_v0: pl.Tensor[[80, 80], pl.FP32], grad_wk__ssa_v0: pl.Tensor[[80, 80], pl.FP32], grad_wv__ssa_v0: pl.Tensor[[80, 80], pl.FP32], grad_wo__ssa_v0: pl.Tensor[[80, 80], pl.FP32], grad_w_gate__ssa_v0: pl.Tensor[[80, 400], pl.FP32], grad_w_up__ssa_v0: pl.Tensor[[80, 400], pl.FP32], grad_w_down__ssa_v0: pl.Tensor[[400, 80], pl.FP32], out__ssa_v0: pl.Tensor[[1, 4, 80], pl.BF16], loss_out__ssa_v0: pl.Tensor[[1], pl.FP32]) -> tuple[pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[400, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[400, 80], pl.FP32], pl.Tensor[[1, 4, 80], pl.BF16], pl.Tensor[[1], pl.FP32]]:
        muon_scratch__ssa_v0: pl.Tensor[[16, 16], pl.BF16] = pl.tensor.create([16, 16], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        proxy_scratch__ssa_v0: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.create([2, 16], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        btrans_scratch__ssa_v0: pl.Tensor[[2, 4], pl.FP32] = pl.tensor.create([2, 4], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        ret__tmp_v0: pl.Tuple[pl.Tensor[[400, 80], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32]] = self.qwen3_32b_training_forward_and_backward_layer_incore_0(grad_w_down__ssa_v0, grad_w_gate__ssa_v0, grad_w_up__ssa_v0, grad_wk__ssa_v0, grad_wo__ssa_v0, grad_wq__ssa_v0, grad_wv__ssa_v0)
        grad_w_down__ssa_v1: pl.Tensor[[400, 80], pl.FP32] = ret__tmp_v0[0]
        grad_w_gate__ssa_v1: pl.Tensor[[80, 400], pl.FP32] = ret__tmp_v0[1]
        grad_w_up__ssa_v1: pl.Tensor[[80, 400], pl.FP32] = ret__tmp_v0[2]
        grad_wk__ssa_v1: pl.Tensor[[80, 80], pl.FP32] = ret__tmp_v0[3]
        grad_wo__ssa_v1: pl.Tensor[[80, 80], pl.FP32] = ret__tmp_v0[4]
        grad_wq__ssa_v1: pl.Tensor[[80, 80], pl.FP32] = ret__tmp_v0[5]
        grad_wv__ssa_v1: pl.Tensor[[80, 80], pl.FP32] = ret__tmp_v0[6]
        loss_acc__ssa_v0: pl.Tensor[[2, 1], pl.FP32] = pl.tensor.create([2, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        loss_acc__ssa_v1: pl.Tensor[[2, 1], pl.FP32] = self.qwen3_32b_training_forward_and_backward_layer_incore_1(loss_acc__ssa_v0)
        tok_blocks__ssa_v0: pl.Scalar[pl.INDEX] = 2
        ret__tmp_v0_1: pl.Tuple[pl.Tensor[[2, 1], pl.FP32], pl.Tensor[[1, 4, 80], pl.BF16]] = self.qwen3_32b_training_forward_and_backward_layer_incore_2(btrans_scratch__ssa_v0, hidden_states__ssa_v0, input_rms_weight__ssa_v0, loss_acc__ssa_v1, out__ssa_v0, post_rms_weight__ssa_v0, target_states__ssa_v0, tok_blocks__ssa_v0, w_down__ssa_v0, w_gate__ssa_v0, w_up__ssa_v0, wk__ssa_v0, wo__ssa_v0, wq__ssa_v0, wv__ssa_v0)
        loss_acc__cr_rv_v2: pl.Tensor[[2, 1], pl.FP32] = ret__tmp_v0_1[0]
        out__cr_rv_v1: pl.Tensor[[1, 4, 80], pl.BF16] = ret__tmp_v0_1[1]
        t__tmp_v72: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.slice(w_up__ssa_v0, [2, 16], [0, 0])
        proxy_mlp__ssa_v0: pl.Tensor[[2, 16], pl.BF16] = self.qwen3_32b_training_forward_and_backward_layer_incore_3(t__tmp_v72)
        proxy_scratch__ssa_v1: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.assemble(proxy_scratch__ssa_v0, proxy_mlp__ssa_v0, [0, 0])
        proxy_mlp_t__ssa_v0: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.slice(proxy_scratch__ssa_v1, [2, 16], [0, 0])
        ret__tmp_v0_2: pl.Tuple[pl.Tensor[[400, 80], pl.FP32], pl.Tensor[[400, 80], pl.FP32], pl.Tensor[[16, 16], pl.BF16], pl.Tensor[[2, 16], pl.BF16]] = self.qwen3_32b_training_forward_and_backward_layer_incore_4(grad_w_down__ssa_v1, mom_w_down__ssa_v0, muon_scratch__ssa_v0, proxy_mlp_t__ssa_v0, proxy_scratch__ssa_v1, target_states__ssa_v0)
        grad_w_down__rv_v3: pl.Tensor[[400, 80], pl.FP32] = ret__tmp_v0_2[0]
        mom_w_down__rv_v2: pl.Tensor[[400, 80], pl.FP32] = ret__tmp_v0_2[1]
        muon_scratch__rv_v2: pl.Tensor[[16, 16], pl.BF16] = ret__tmp_v0_2[2]
        proxy_scratch__rv_v3: pl.Tensor[[2, 16], pl.BF16] = ret__tmp_v0_2[3]
        t__tmp_v83: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.slice(wq__ssa_v0, [2, 4], [0, 0])
        proxy_ctx__ssa_v0: pl.Tensor[[2, 4], pl.BF16] = self.qwen3_32b_training_forward_and_backward_layer_incore_5(t__tmp_v83)
        t__tmp_v84: pl.Tensor[[1, 2, 4], pl.BF16] = pl.tensor.slice(hidden_states__ssa_v0, [1, 2, 4], [0, 0, 0])
        t__tmp_v85: pl.Tensor[[1, 2, 4], pl.BF16] = self.qwen3_32b_training_forward_and_backward_layer_incore_6(t__tmp_v84)
        proxy_n__ssa_v0: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.reshape(t__tmp_v85, [2, 4])
        proxy_scratch__ssa_v5: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.assemble(proxy_scratch__rv_v3, proxy_ctx__ssa_v0, [0, 0])
        proxy_ctx_t__ssa_v0: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.slice(proxy_scratch__ssa_v5, [2, 4], [0, 0])
        proxy_scratch__ssa_v6: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.assemble(proxy_scratch__ssa_v5, proxy_n__ssa_v0, [0, 0])
        proxy_n_t__ssa_v0: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.slice(proxy_scratch__ssa_v6, [2, 4], [0, 0])
        ret__tmp_v0_3: pl.Tuple[pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[80, 80], pl.FP32], pl.Tensor[[16, 16], pl.BF16], pl.Tensor[[2, 16], pl.BF16]] = self.qwen3_32b_training_forward_and_backward_layer_incore_7(grad_wk__ssa_v1, grad_wo__ssa_v1, grad_wq__ssa_v1, grad_wv__ssa_v1, mom_wk__ssa_v0, mom_wo__ssa_v0, mom_wq__ssa_v0, mom_wv__ssa_v0, muon_scratch__rv_v2, proxy_ctx_t__ssa_v0, proxy_n_t__ssa_v0, proxy_scratch__ssa_v6, target_states__ssa_v0)
        grad_wk__rv_v3: pl.Tensor[[80, 80], pl.FP32] = ret__tmp_v0_3[0]
        grad_wo__rv_v3: pl.Tensor[[80, 80], pl.FP32] = ret__tmp_v0_3[1]
        grad_wq__rv_v3: pl.Tensor[[80, 80], pl.FP32] = ret__tmp_v0_3[2]
        grad_wv__rv_v3: pl.Tensor[[80, 80], pl.FP32] = ret__tmp_v0_3[3]
        mom_wk__rv_v2: pl.Tensor[[80, 80], pl.FP32] = ret__tmp_v0_3[4]
        mom_wo__rv_v2: pl.Tensor[[80, 80], pl.FP32] = ret__tmp_v0_3[5]
        mom_wq__rv_v2: pl.Tensor[[80, 80], pl.FP32] = ret__tmp_v0_3[6]
        mom_wv__rv_v2: pl.Tensor[[80, 80], pl.FP32] = ret__tmp_v0_3[7]
        muon_scratch__rv_v5: pl.Tensor[[16, 16], pl.BF16] = ret__tmp_v0_3[8]
        proxy_scratch__rv_v8: pl.Tensor[[2, 16], pl.BF16] = ret__tmp_v0_3[9]
        t__tmp_v120: pl.Tensor[[1, 2, 4], pl.BF16] = pl.tensor.slice(hidden_states__ssa_v0, [1, 2, 4], [0, 0, 4])
        t__tmp_v121: pl.Tensor[[1, 2, 4], pl.BF16] = self.qwen3_32b_training_forward_and_backward_layer_incore_8(t__tmp_v120)
        proxy_post__ssa_v0: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.reshape(t__tmp_v121, [2, 4])
        proxy_scratch__ssa_v10: pl.Tensor[[2, 16], pl.BF16] = pl.tensor.assemble(proxy_scratch__rv_v8, proxy_post__ssa_v0, [0, 0])
        proxy_post_t__ssa_v0: pl.Tensor[[2, 4], pl.BF16] = pl.tensor.slice(proxy_scratch__ssa_v10, [2, 4], [0, 0])
        ret__tmp_v0_4: pl.Tuple[pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[80, 400], pl.FP32], pl.Tensor[[80, 400], pl.FP32]] = self.qwen3_32b_training_forward_and_backward_layer_incore_9(grad_w_gate__ssa_v1, grad_w_up__ssa_v1, mom_w_gate__ssa_v0, mom_w_up__ssa_v0, muon_scratch__rv_v5, proxy_post_t__ssa_v0, proxy_scratch__ssa_v10, w_gate__ssa_v0, w_up__ssa_v0)
        grad_w_gate__rv_v3: pl.Tensor[[80, 400], pl.FP32] = ret__tmp_v0_4[0]
        grad_w_up__rv_v3: pl.Tensor[[80, 400], pl.FP32] = ret__tmp_v0_4[1]
        mom_w_gate__rv_v2: pl.Tensor[[80, 400], pl.FP32] = ret__tmp_v0_4[2]
        mom_w_up__rv_v2: pl.Tensor[[80, 400], pl.FP32] = ret__tmp_v0_4[3]
        loss_vec__ssa_v0: pl.Tensor[[1], pl.FP32] = pl.tensor.slice(loss_acc__cr_rv_v2, [1], [0, 0])
        loss_out__ssa_v1: pl.Tensor[[1], pl.FP32] = pl.tensor.assemble(loss_out__ssa_v0, loss_vec__ssa_v0, [0])
        return grad_wq__rv_v3, grad_wk__rv_v3, grad_wv__rv_v3, grad_wo__rv_v3, grad_w_gate__rv_v3, grad_w_up__rv_v3, grad_w_down__rv_v3, mom_wq__rv_v2, mom_wk__rv_v2, mom_wv__rv_v2, mom_wo__rv_v2, mom_w_gate__rv_v2, mom_w_up__rv_v2, mom_w_down__rv_v2, out__cr_rv_v1, loss_out__ssa_v1