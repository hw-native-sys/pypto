# pypto.program: Qwen3SingleLayerPrefill
import pypto.language as pl

valid_len = pl.dynamic("valid_len")
valid_tok = pl.dynamic("valid_tok")

@pl.program
class Qwen3SingleLayerPrefill:
    @pl.function
    def qwen3_prefill_layer(self, hidden_states__ssa_v0: pl.Tensor[[16, 4096, 5120], pl.BF16], seq_lens__ssa_v0: pl.Tensor[[16], pl.INT32], rope_cos__ssa_v0: pl.Tensor[[4096, 128], pl.FP32], rope_sin__ssa_v0: pl.Tensor[[4096, 128], pl.FP32], k_cache__ssa_v0: pl.Tensor[[524288, 128], pl.BF16], v_cache__ssa_v0: pl.Tensor[[524288, 128], pl.BF16], input_rms_weight__ssa_v0: pl.Tensor[[1, 5120], pl.FP32], wq__ssa_v0: pl.Tensor[[5120, 5120], pl.BF16], wk__ssa_v0: pl.Tensor[[5120, 1024], pl.BF16], wv__ssa_v0: pl.Tensor[[5120, 1024], pl.BF16], wo__ssa_v0: pl.Tensor[[5120, 5120], pl.BF16], post_rms_weight__ssa_v0: pl.Tensor[[1, 5120], pl.FP32], w_gate__ssa_v0: pl.Tensor[[5120, 25600], pl.BF16], w_up__ssa_v0: pl.Tensor[[5120, 25600], pl.BF16], w_down__ssa_v0: pl.Tensor[[25600, 5120], pl.BF16], out__ssa_v0: pl.Tensor[[16, 4096, 5120], pl.BF16]) -> pl.Tensor[[16, 4096, 5120], pl.BF16]:
        for b__idx_v0, (k_cache__iter_v1, out__iter_v1, v_cache__iter_v1) in pl.parallel(16, init_values=(k_cache__ssa_v0, out__ssa_v0, v_cache__ssa_v0)):
            seq_len_b__ssa_v0: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens__ssa_v0, [b__idx_v0])
            tok_blocks__ssa_v0: pl.Scalar[pl.INDEX] = (pl.cast(seq_len_b__ssa_v0, pl.INDEX) + 4 - 1) // 4
            for p0_idx__idx_v0, (k_cache__iter_v3, out__iter_v3, v_cache__iter_v3) in pl.range(tok_blocks__ssa_v0, init_values=(k_cache__iter_v1, out__iter_v1, v_cache__iter_v1)):
                p0__ssa_v0: pl.Scalar[pl.INDEX] = p0_idx__idx_v0 * 4
                valid_tok__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(4, pl.cast(seq_len_b__ssa_v0, pl.INDEX) - p0__ssa_v0)
                with pl.auto_incore():
                    sq_sum__ssa_v0: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.create([4, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    sq_sum__ssa_v1: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.muls(sq_sum__ssa_v0, 0.0)
                    for kb__idx_v0, (sq_sum__iter_v2,) in pl.range(20, init_values=(sq_sum__ssa_v1,)):
                        k0__ssa_v0: pl.Scalar[pl.INDEX] = kb__idx_v0 * 256
                        t__tmp_v0: pl.Tensor[[1, 4, 256], pl.BF16, pl.TensorView(valid_shape=[1, valid_tok, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(hidden_states__ssa_v0, [1, 4, 256], [b__idx_v0, p0__ssa_v0, k0__ssa_v0], [1, valid_tok__ssa_v0, 256])
                        t__tmp_v1: pl.Tensor[[1, 4, 256], pl.FP32] = pl.tensor.cast(t__tmp_v0, target_type=pl.FP32, mode='round')
                        x_chunk__ssa_v0: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.reshape(t__tmp_v1, [4, 256])
                        t__tmp_v2: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.mul(x_chunk__ssa_v0, x_chunk__ssa_v0)
                        t__tmp_v3: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.row_sum(t__tmp_v2)
                        sq_sum__ssa_v4: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.add(sq_sum__iter_v2, t__tmp_v3)
                        sq_sum__rv_v3: pl.Tensor[[4, 1], pl.FP32] = pl.yield_(sq_sum__ssa_v4)
                    t__tmp_v4: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.muls(sq_sum__rv_v3, 0.000195313)
                    t__tmp_v5: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.adds(t__tmp_v4, 1e-06)
                    inv_rms__ssa_v0: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.rsqrt(t__tmp_v5)
                    q_proj_tile__ssa_v0: pl.Tensor[[4, 5120], pl.BF16] = pl.tensor.create([4, 5120], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                    k_proj_tile__ssa_v0: pl.Tensor[[4, 1024], pl.BF16] = pl.tensor.create([4, 1024], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                    v_proj_tile__ssa_v0: pl.Tensor[[4, 1024], pl.BF16] = pl.tensor.create([4, 1024], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                    for ob__idx_v0, (q_proj_tile__iter_v1,) in pl.parallel(80, init_values=(q_proj_tile__ssa_v0,), chunk=8):
                        q0__ssa_v0: pl.Scalar[pl.INDEX] = ob__idx_v0 * 64
                        q_acc__ssa_v0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.create([4, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        q_acc__ssa_v1: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.muls(q_acc__ssa_v0, 0.0)
                        for kb__idx_v0_1, (q_acc__iter_v2,) in pl.range(20, init_values=(q_acc__ssa_v1,)):
                            k0__ssa_v1: pl.Scalar[pl.INDEX] = kb__idx_v0_1 * 256
                            t__tmp_v6: pl.Tensor[[1, 4, 256], pl.BF16, pl.TensorView(valid_shape=[1, valid_tok, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(hidden_states__ssa_v0, [1, 4, 256], [b__idx_v0, p0__ssa_v0, k0__ssa_v1], [1, valid_tok__ssa_v0, 256])
                            t__tmp_v7: pl.Tensor[[1, 4, 256], pl.FP32] = pl.tensor.cast(t__tmp_v6, target_type=pl.FP32, mode='round')
                            x_chunk__ssa_v1: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.reshape(t__tmp_v7, [4, 256])
                            gamma__ssa_v0: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(input_rms_weight__ssa_v0, [1, 256], [0, k0__ssa_v1])
                            t__tmp_v8: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.row_expand_mul(x_chunk__ssa_v1, inv_rms__ssa_v0)
                            normed__ssa_v0: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.col_expand_mul(t__tmp_v8, gamma__ssa_v0)
                            wq_chunk__ssa_v0: pl.Tensor[[256, 64], pl.BF16] = pl.tensor.slice(wq__ssa_v0, [256, 64], [k0__ssa_v1, q0__ssa_v0])
                            t__tmp_v9: pl.Tensor[[4, 256], pl.BF16] = pl.tensor.cast(normed__ssa_v0, target_type=pl.BF16, mode='round')
                            t__tmp_v10: pl.Tensor[[4, 64], pl.BF16] = pl.tensor.matmul(t__tmp_v9, wq_chunk__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False)
                            q_acc__ssa_v4: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(q_acc__iter_v2, t__tmp_v10)
                            q_acc__rv_v3: pl.Tensor[[4, 64], pl.FP32] = pl.yield_(q_acc__ssa_v4)
                        t__tmp_v11: pl.Tensor[[4, 64], pl.BF16] = pl.tensor.cast(q_acc__rv_v3, target_type=pl.BF16, mode='round')
                        q_proj_tile__ssa_v3: pl.Tensor[[4, 5120], pl.BF16] = pl.tensor.assemble(q_proj_tile__iter_v1, t__tmp_v11, [0, q0__ssa_v0])
                        q_proj_tile__rv_v2: pl.Tensor[[4, 5120], pl.BF16] = pl.yield_(q_proj_tile__ssa_v3)
                    for ob__idx_v0_1, (k_proj_tile__iter_v1, v_proj_tile__iter_v1) in pl.parallel(32, init_values=(k_proj_tile__ssa_v0, v_proj_tile__ssa_v0), chunk=8):
                        kv0__ssa_v0: pl.Scalar[pl.INDEX] = ob__idx_v0_1 * 32
                        k_acc__ssa_v0: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.create([4, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        v_acc__ssa_v0: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.create([4, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        k_acc__ssa_v1: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.muls(k_acc__ssa_v0, 0.0)
                        v_acc__ssa_v1: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.muls(v_acc__ssa_v0, 0.0)
                        for kb__idx_v0_2, (k_acc__iter_v2, v_acc__iter_v2) in pl.range(20, init_values=(k_acc__ssa_v1, v_acc__ssa_v1)):
                            k0__ssa_v2: pl.Scalar[pl.INDEX] = kb__idx_v0_2 * 256
                            t__tmp_v12: pl.Tensor[[1, 4, 256], pl.BF16, pl.TensorView(valid_shape=[1, valid_tok, 256], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(hidden_states__ssa_v0, [1, 4, 256], [b__idx_v0, p0__ssa_v0, k0__ssa_v2], [1, valid_tok__ssa_v0, 256])
                            t__tmp_v13: pl.Tensor[[1, 4, 256], pl.FP32] = pl.tensor.cast(t__tmp_v12, target_type=pl.FP32, mode='round')
                            x_chunk__ssa_v2: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.reshape(t__tmp_v13, [4, 256])
                            gamma__ssa_v1: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(input_rms_weight__ssa_v0, [1, 256], [0, k0__ssa_v2])
                            t__tmp_v14: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.row_expand_mul(x_chunk__ssa_v2, inv_rms__ssa_v0)
                            normed__ssa_v1: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.col_expand_mul(t__tmp_v14, gamma__ssa_v1)
                            normed_bf16__ssa_v0: pl.Tensor[[4, 256], pl.BF16] = pl.tensor.cast(normed__ssa_v1, target_type=pl.BF16, mode='round')
                            wk_chunk__ssa_v0: pl.Tensor[[256, 32], pl.BF16] = pl.tensor.slice(wk__ssa_v0, [256, 32], [k0__ssa_v2, kv0__ssa_v0])
                            wv_chunk__ssa_v0: pl.Tensor[[256, 32], pl.BF16] = pl.tensor.slice(wv__ssa_v0, [256, 32], [k0__ssa_v2, kv0__ssa_v0])
                            t__tmp_v15: pl.Tensor[[4, 32], pl.BF16] = pl.tensor.matmul(normed_bf16__ssa_v0, wk_chunk__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False)
                            k_acc__ssa_v4: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.add(k_acc__iter_v2, t__tmp_v15)
                            t__tmp_v16: pl.Tensor[[4, 32], pl.BF16] = pl.tensor.matmul(normed_bf16__ssa_v0, wv_chunk__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False)
                            v_acc__ssa_v4: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.add(v_acc__iter_v2, t__tmp_v16)
                            k_acc__rv_v3, v_acc__rv_v3 = pl.yield_(k_acc__ssa_v4, v_acc__ssa_v4)
                        t__tmp_v17: pl.Tensor[[4, 32], pl.BF16] = pl.tensor.cast(k_acc__rv_v3, target_type=pl.BF16, mode='round')
                        k_proj_tile__ssa_v3: pl.Tensor[[4, 1024], pl.BF16] = pl.tensor.assemble(k_proj_tile__iter_v1, t__tmp_v17, [0, kv0__ssa_v0])
                        t__tmp_v18: pl.Tensor[[4, 32], pl.BF16] = pl.tensor.cast(v_acc__rv_v3, target_type=pl.BF16, mode='round')
                        v_proj_tile__ssa_v3: pl.Tensor[[4, 1024], pl.BF16] = pl.tensor.assemble(v_proj_tile__iter_v1, t__tmp_v18, [0, kv0__ssa_v0])
                        k_proj_tile__rv_v2, v_proj_tile__rv_v2 = pl.yield_(k_proj_tile__ssa_v3, v_proj_tile__ssa_v3)
                attn_tile__ssa_v0: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.create([4, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                attn_tile__ssa_v1: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.muls(attn_tile__ssa_v0, 0.0)
                for ti__idx_v0, (attn_tile__iter_v2, k_cache__iter_v5, v_cache__iter_v5) in pl.range(valid_tok__ssa_v0, init_values=(attn_tile__ssa_v1, k_cache__iter_v3, v_cache__iter_v3)):
                    pos__ssa_v0: pl.Scalar[pl.INDEX] = p0__ssa_v0 + ti__idx_v0
                    ctx_len__ssa_v0: pl.Scalar[pl.INDEX] = pos__ssa_v0 + 1
                    ctx_blocks__ssa_v0: pl.Scalar[pl.INDEX] = (ctx_len__ssa_v0 + 120 - 1) // 120
                    cos_row__ssa_v0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.slice(rope_cos__ssa_v0, [1, 128], [pos__ssa_v0, 0])
                    sin_row__ssa_v0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.slice(rope_sin__ssa_v0, [1, 128], [pos__ssa_v0, 0])
                    cos_lo__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(cos_row__ssa_v0, [1, 64], [0, 0])
                    cos_hi__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(cos_row__ssa_v0, [1, 64], [0, 64])
                    sin_lo__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(sin_row__ssa_v0, [1, 64], [0, 0])
                    sin_hi__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(sin_row__ssa_v0, [1, 64], [0, 64])
                    with pl.auto_incore():
                        attn_row__ssa_v0: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.create([1, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        attn_row__ssa_v1: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.muls(attn_row__ssa_v0, 0.0)
                        for kvh__idx_v0, (k_cache__iter_v7, v_cache__iter_v7) in pl.parallel(8, init_values=(k_cache__iter_v5, v_cache__iter_v5), chunk=4):
                            kv_col__ssa_v0: pl.Scalar[pl.INDEX] = kvh__idx_v0 * 128
                            t__tmp_v19: pl.Tensor[[1, 128], pl.BF16] = pl.tensor.slice(k_proj_tile__rv_v2, [1, 128], [ti__idx_v0, kv_col__ssa_v0])
                            k_row__ssa_v0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.cast(t__tmp_v19, target_type=pl.FP32, mode='round')
                            k_lo__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(k_row__ssa_v0, [1, 64], [0, 0])
                            k_hi__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(k_row__ssa_v0, [1, 64], [0, 64])
                            k_rot__ssa_v0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                            t__tmp_v20: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.col_expand_mul(k_lo__ssa_v0, cos_lo__ssa_v0)
                            t__tmp_v21: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.col_expand_mul(k_hi__ssa_v0, sin_lo__ssa_v0)
                            t__tmp_v22: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.sub(t__tmp_v20, t__tmp_v21)
                            k_rot__ssa_v1: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(k_rot__ssa_v0, t__tmp_v22, [0, 0])
                            t__tmp_v23: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.col_expand_mul(k_hi__ssa_v0, cos_hi__ssa_v0)
                            t__tmp_v24: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.col_expand_mul(k_lo__ssa_v0, sin_hi__ssa_v0)
                            t__tmp_v25: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(t__tmp_v23, t__tmp_v24)
                            k_rot__ssa_v2: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(k_rot__ssa_v1, t__tmp_v25, [0, 64])
                            cache_row__ssa_v0: pl.Scalar[pl.INDEX] = b__idx_v0 * 8 * 4096 + kvh__idx_v0 * 4096 + pos__ssa_v0
                            t__tmp_v26: pl.Tensor[[1, 128], pl.BF16] = pl.tensor.cast(k_rot__ssa_v2, target_type=pl.BF16, mode='round')
                            k_cache__ssa_v9: pl.Tensor[[524288, 128], pl.BF16] = pl.tensor.assemble(k_cache__iter_v7, t__tmp_v26, [cache_row__ssa_v0, 0])
                            t__tmp_v27: pl.Tensor[[1, 128], pl.BF16] = pl.tensor.slice(v_proj_tile__rv_v2, [1, 128], [ti__idx_v0, kv_col__ssa_v0])
                            v_cache__ssa_v9: pl.Tensor[[524288, 128], pl.BF16] = pl.tensor.assemble(v_cache__iter_v7, t__tmp_v27, [cache_row__ssa_v0, 0])
                            k_cache__rv_v8, v_cache__rv_v8 = pl.yield_(k_cache__ssa_v9, v_cache__ssa_v9)
                        for h__idx_v0, (attn_row__iter_v2,) in pl.parallel(64, init_values=(attn_row__ssa_v1,), chunk=8):
                            kvh__ssa_v1: pl.Scalar[pl.INDEX] = h__idx_v0 // 8
                            q_col__ssa_v0: pl.Scalar[pl.INDEX] = h__idx_v0 * 128
                            t__tmp_v28: pl.Tensor[[1, 128], pl.BF16] = pl.tensor.slice(q_proj_tile__rv_v2, [1, 128], [ti__idx_v0, q_col__ssa_v0])
                            q_row__ssa_v0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.cast(t__tmp_v28, target_type=pl.FP32, mode='round')
                            q_lo__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(q_row__ssa_v0, [1, 64], [0, 0])
                            q_hi__ssa_v0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(q_row__ssa_v0, [1, 64], [0, 64])
                            q_rot__ssa_v0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                            t__tmp_v29: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.col_expand_mul(q_lo__ssa_v0, cos_lo__ssa_v0)
                            t__tmp_v30: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.col_expand_mul(q_hi__ssa_v0, sin_lo__ssa_v0)
                            t__tmp_v31: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.sub(t__tmp_v29, t__tmp_v30)
                            q_rot__ssa_v1: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(q_rot__ssa_v0, t__tmp_v31, [0, 0])
                            t__tmp_v32: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.col_expand_mul(q_hi__ssa_v0, cos_hi__ssa_v0)
                            t__tmp_v33: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.col_expand_mul(q_lo__ssa_v0, sin_hi__ssa_v0)
                            t__tmp_v34: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.add(t__tmp_v32, t__tmp_v33)
                            q_rot__ssa_v2: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(q_rot__ssa_v1, t__tmp_v34, [0, 64])
                            q_rot_bf16__ssa_v0: pl.Tensor[[1, 128], pl.BF16] = pl.tensor.cast(q_rot__ssa_v2, target_type=pl.BF16, mode='round')
                            oi__ssa_v0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                            li__ssa_v0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                            mi__ssa_v0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                            oi__ssa_v1: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.muls(oi__ssa_v0, 0.0)
                            li__ssa_v1: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.muls(li__ssa_v0, 0.0)
                            mi__ssa_v1: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.muls(mi__ssa_v0, 0.0)
                            for sb__idx_v0, (li__iter_v2, mi__iter_v2, oi__iter_v2) in pl.range(ctx_blocks__ssa_v0, init_values=(li__ssa_v1, mi__ssa_v1, oi__ssa_v1)):
                                s0__ssa_v0: pl.Scalar[pl.INDEX] = sb__idx_v0 * 120
                                valid_len__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(120, ctx_len__ssa_v0 - s0__ssa_v0)
                                cache_row0__ssa_v0: pl.Scalar[pl.INDEX] = b__idx_v0 * 8 * 4096 + kvh__ssa_v1 * 4096 + s0__ssa_v0
                                k_tile__ssa_v0: pl.Tensor[[120, 128], pl.BF16, pl.TensorView(valid_shape=[valid_len, 128], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(k_cache__rv_v8, [120, 128], [cache_row0__ssa_v0, 0], [valid_len__ssa_v0, 128])
                                v_tile__ssa_v0: pl.Tensor[[120, 128], pl.BF16, pl.TensorView(valid_shape=[valid_len, 128], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(v_cache__rv_v8, [120, 128], [cache_row0__ssa_v0, 0], [valid_len__ssa_v0, 128])
                                t__tmp_v35: pl.Tensor[[1, 120], pl.BF16] = pl.tensor.matmul(q_rot_bf16__ssa_v0, k_tile__ssa_v0, a_trans=False, b_trans=True, c_matrix_nz=False)
                                scores__ssa_v0: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.muls(t__tmp_v35, 0.0883883)
                                scores_valid__ssa_v0: pl.Tensor[[1, 120], pl.FP32, pl.TensorView(valid_shape=[1, valid_len], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(scores__ssa_v0, [1, 120], [0, 0], [1, valid_len__ssa_v0])
                                t__tmp_v36: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.row_max(scores_valid__ssa_v0)
                                cur_mi__ssa_v0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.cast(t__tmp_v36, target_type=pl.FP32, mode='round')
                                t__tmp_v37: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.row_expand_sub(scores_valid__ssa_v0, cur_mi__ssa_v0)
                                exp_scores__ssa_v0: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.exp(t__tmp_v37)
                                t__tmp_v38: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.row_sum(exp_scores__ssa_v0)
                                cur_li__ssa_v0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.cast(t__tmp_v38, target_type=pl.FP32, mode='round')
                                exp_pad__ssa_v0: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.create([1, 120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                                exp_pad__ssa_v1: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.muls(exp_pad__ssa_v0, 0.0)
                                exp_pad__ssa_v2: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.assemble(exp_pad__ssa_v1, exp_scores__ssa_v0, [0, 0])
                                t__tmp_v39: pl.Tensor[[1, 120], pl.BF16] = pl.tensor.cast(exp_pad__ssa_v2, target_type=pl.BF16, mode='round')
                                oi_tmp__ssa_v0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.matmul(t__tmp_v39, v_tile__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                                if sb__idx_v0 == 0:
                                    oi__ssa_v4: pl.Tensor[[1, 128], pl.FP32] = oi_tmp__ssa_v0
                                    li__ssa_v4: pl.Tensor[[1, 1], pl.FP32] = cur_li__ssa_v0
                                    mi__ssa_v4: pl.Tensor[[1, 1], pl.FP32] = cur_mi__ssa_v0
                                    li__phi_v6, mi__phi_v6, oi__phi_v6 = pl.yield_(li__ssa_v4, mi__ssa_v4, oi__ssa_v4)
                                else:
                                    mi_new__ssa_v0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.maximum(mi__iter_v2, cur_mi__ssa_v0)
                                    t__tmp_v40: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.sub(mi__iter_v2, mi_new__ssa_v0)
                                    alpha__ssa_v0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.exp(t__tmp_v40)
                                    t__tmp_v41: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.sub(cur_mi__ssa_v0, mi_new__ssa_v0)
                                    beta__ssa_v0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.exp(t__tmp_v41)
                                    t__tmp_v42: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(alpha__ssa_v0, li__iter_v2)
                                    t__tmp_v43: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(beta__ssa_v0, cur_li__ssa_v0)
                                    li__ssa_v5: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.add(t__tmp_v42, t__tmp_v43)
                                    t__tmp_v44: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.row_expand_mul(oi__iter_v2, alpha__ssa_v0)
                                    t__tmp_v45: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.row_expand_mul(oi_tmp__ssa_v0, beta__ssa_v0)
                                    oi__ssa_v5: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.add(t__tmp_v44, t__tmp_v45)
                                    mi__ssa_v5: pl.Tensor[[1, 1], pl.FP32] = mi_new__ssa_v0
                                    li__phi_v6, mi__phi_v6, oi__phi_v6 = pl.yield_(li__ssa_v5, mi__ssa_v5, oi__ssa_v5)
                                li__rv_v3, mi__rv_v3, oi__rv_v3 = pl.yield_(li__phi_v6, mi__phi_v6, oi__phi_v6)
                            ctx__ssa_v0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.row_expand_div(oi__rv_v3, li__rv_v3)
                            attn_row__ssa_v4: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.assemble(attn_row__iter_v2, ctx__ssa_v0, [0, q_col__ssa_v0])
                            attn_row__rv_v3: pl.Tensor[[1, 5120], pl.FP32] = pl.yield_(attn_row__ssa_v4)
                        attn_tile__ssa_v4: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.assemble(attn_tile__iter_v2, attn_row__rv_v3, [ti__idx_v0, 0])
                    attn_tile__rv_v3, k_cache__rv_v6, v_cache__rv_v6 = pl.yield_(attn_tile__ssa_v4, k_cache__rv_v8, v_cache__rv_v8)
                with pl.auto_incore():
                    resid1_tile__ssa_v0: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.create([4, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    for ob__idx_v0_2, (resid1_tile__iter_v1,) in pl.parallel(80, init_values=(resid1_tile__ssa_v0,), chunk=8):
                        o0__ssa_v0: pl.Scalar[pl.INDEX] = ob__idx_v0_2 * 64
                        o_acc__ssa_v0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.create([4, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        o_acc__ssa_v1: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.muls(o_acc__ssa_v0, 0.0)
                        for kb__idx_v0_3, (o_acc__iter_v2,) in pl.range(20, init_values=(o_acc__ssa_v1,)):
                            k0__ssa_v3: pl.Scalar[pl.INDEX] = kb__idx_v0_3 * 256
                            t__tmp_v46: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.slice(attn_tile__rv_v3, [4, 256], [0, k0__ssa_v3])
                            a_chunk__ssa_v0: pl.Tensor[[4, 256], pl.BF16] = pl.tensor.cast(t__tmp_v46, target_type=pl.BF16, mode='round')
                            w_chunk__ssa_v0: pl.Tensor[[256, 64], pl.BF16] = pl.tensor.slice(wo__ssa_v0, [256, 64], [k0__ssa_v3, o0__ssa_v0])
                            t__tmp_v47: pl.Tensor[[4, 64], pl.BF16] = pl.tensor.matmul(a_chunk__ssa_v0, w_chunk__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False)
                            o_acc__ssa_v4: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(o_acc__iter_v2, t__tmp_v47)
                            o_acc__rv_v3: pl.Tensor[[4, 64], pl.FP32] = pl.yield_(o_acc__ssa_v4)
                        t__tmp_v48: pl.Tensor[[1, 4, 64], pl.BF16, pl.TensorView(valid_shape=[1, valid_tok, 64], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(hidden_states__ssa_v0, [1, 4, 64], [b__idx_v0, p0__ssa_v0, o0__ssa_v0], [1, valid_tok__ssa_v0, 64])
                        t__tmp_v49: pl.Tensor[[1, 4, 64], pl.FP32] = pl.tensor.cast(t__tmp_v48, target_type=pl.FP32, mode='round')
                        resid__ssa_v0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.reshape(t__tmp_v49, [4, 64])
                        t__tmp_v50: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(o_acc__rv_v3, resid__ssa_v0)
                        resid1_tile__ssa_v3: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.assemble(resid1_tile__iter_v1, t__tmp_v50, [0, o0__ssa_v0])
                        resid1_tile__rv_v2: pl.Tensor[[4, 5120], pl.FP32] = pl.yield_(resid1_tile__ssa_v3)
                    sq_sum__ssa_v5: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.create([4, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    sq_sum__ssa_v6: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.muls(sq_sum__ssa_v5, 0.0)
                    for kb__idx_v0_4, (sq_sum__iter_v7,) in pl.range(20, init_values=(sq_sum__ssa_v6,)):
                        k0__ssa_v4: pl.Scalar[pl.INDEX] = kb__idx_v0_4 * 256
                        x_chunk__ssa_v3: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.slice(resid1_tile__rv_v2, [4, 256], [0, k0__ssa_v4])
                        t__tmp_v51: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.mul(x_chunk__ssa_v3, x_chunk__ssa_v3)
                        t__tmp_v52: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.row_sum(t__tmp_v51)
                        sq_sum__ssa_v9: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.add(sq_sum__iter_v7, t__tmp_v52)
                        sq_sum__rv_v8: pl.Tensor[[4, 1], pl.FP32] = pl.yield_(sq_sum__ssa_v9)
                    t__tmp_v53: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.muls(sq_sum__rv_v8, 0.000195313)
                    t__tmp_v54: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.adds(t__tmp_v53, 1e-06)
                    inv_rms__ssa_v1: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.rsqrt(t__tmp_v54)
                    post_norm_tile__ssa_v0: pl.Tensor[[4, 5120], pl.BF16] = pl.tensor.create([4, 5120], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                    down_proj_tile__ssa_v0: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.create([4, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    down_proj_tile__ssa_v1: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.muls(down_proj_tile__ssa_v0, 0.0)
                    for kb__idx_v0_5, (post_norm_tile__iter_v1,) in pl.range(20, init_values=(post_norm_tile__ssa_v0,)):
                        k0__ssa_v5: pl.Scalar[pl.INDEX] = kb__idx_v0_5 * 256
                        x_chunk__ssa_v4: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.slice(resid1_tile__rv_v2, [4, 256], [0, k0__ssa_v5])
                        gamma__ssa_v2: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(post_rms_weight__ssa_v0, [1, 256], [0, k0__ssa_v5])
                        t__tmp_v55: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.row_expand_mul(x_chunk__ssa_v4, inv_rms__ssa_v1)
                        normed__ssa_v2: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.col_expand_mul(t__tmp_v55, gamma__ssa_v2)
                        t__tmp_v56: pl.Tensor[[4, 256], pl.BF16] = pl.tensor.cast(normed__ssa_v2, target_type=pl.BF16, mode='round')
                        post_norm_tile__ssa_v3: pl.Tensor[[4, 5120], pl.BF16] = pl.tensor.assemble(post_norm_tile__iter_v1, t__tmp_v56, [0, k0__ssa_v5])
                        post_norm_tile__rv_v2: pl.Tensor[[4, 5120], pl.BF16] = pl.yield_(post_norm_tile__ssa_v3)
                    for ob__idx_v0_3, (down_proj_tile__iter_v2, out__iter_v5) in pl.range(400, init_values=(down_proj_tile__ssa_v1, out__iter_v3)):
                        o0__ssa_v1: pl.Scalar[pl.INDEX] = ob__idx_v0_3 * 64
                        gate_acc__ssa_v0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.create([4, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        up_acc__ssa_v0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.create([4, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        gate_acc__ssa_v1: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.muls(gate_acc__ssa_v0, 0.0)
                        up_acc__ssa_v1: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.muls(up_acc__ssa_v0, 0.0)
                        for kb__idx_v0_6, (gate_acc__iter_v2, up_acc__iter_v2) in pl.range(20, init_values=(gate_acc__ssa_v1, up_acc__ssa_v1)):
                            k0__ssa_v6: pl.Scalar[pl.INDEX] = kb__idx_v0_6 * 256
                            post_chunk__ssa_v0: pl.Tensor[[4, 256], pl.BF16] = pl.tensor.slice(post_norm_tile__rv_v2, [4, 256], [0, k0__ssa_v6])
                            wg__ssa_v0: pl.Tensor[[256, 64], pl.BF16] = pl.tensor.slice(w_gate__ssa_v0, [256, 64], [k0__ssa_v6, o0__ssa_v1])
                            wu__ssa_v0: pl.Tensor[[256, 64], pl.BF16] = pl.tensor.slice(w_up__ssa_v0, [256, 64], [k0__ssa_v6, o0__ssa_v1])
                            t__tmp_v57: pl.Tensor[[4, 64], pl.BF16] = pl.tensor.matmul(post_chunk__ssa_v0, wg__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False)
                            gate_acc__ssa_v4: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(gate_acc__iter_v2, t__tmp_v57)
                            t__tmp_v58: pl.Tensor[[4, 64], pl.BF16] = pl.tensor.matmul(post_chunk__ssa_v0, wu__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False)
                            up_acc__ssa_v4: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(up_acc__iter_v2, t__tmp_v58)
                            gate_acc__rv_v3, up_acc__rv_v3 = pl.yield_(gate_acc__ssa_v4, up_acc__ssa_v4)
                        t__tmp_v59: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.neg(gate_acc__rv_v3)
                        t__tmp_v60: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.exp(t__tmp_v59)
                        t__tmp_v61: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.adds(t__tmp_v60, 1.0)
                        sigmoid__ssa_v0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.recip(t__tmp_v61)
                        t__tmp_v62: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.mul(gate_acc__rv_v3, sigmoid__ssa_v0)
                        mlp_chunk__ssa_v0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.mul(t__tmp_v62, up_acc__rv_v3)
                        mlp_chunk_bf16__ssa_v0: pl.Tensor[[4, 64], pl.BF16] = pl.tensor.cast(mlp_chunk__ssa_v0, target_type=pl.BF16, mode='round')
                        for dob__idx_v0, (down_proj_tile__iter_v4, out__iter_v7) in pl.parallel(80, init_values=(down_proj_tile__iter_v2, out__iter_v5), chunk=8):
                            d0__ssa_v0: pl.Scalar[pl.INDEX] = dob__idx_v0 * 64
                            down_prev__ssa_v0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.slice(down_proj_tile__iter_v4, [4, 64], [0, d0__ssa_v0])
                            w_down_chunk__ssa_v0: pl.Tensor[[64, 64], pl.BF16] = pl.tensor.slice(w_down__ssa_v0, [64, 64], [o0__ssa_v1, d0__ssa_v0])
                            t__tmp_v63: pl.Tensor[[4, 64], pl.BF16] = pl.tensor.matmul(mlp_chunk_bf16__ssa_v0, w_down_chunk__ssa_v0, a_trans=False, b_trans=False, c_matrix_nz=False)
                            down_next__ssa_v0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(down_prev__ssa_v0, t__tmp_v63)
                            down_proj_tile__ssa_v6: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.assemble(down_proj_tile__iter_v4, down_next__ssa_v0, [0, d0__ssa_v0])
                            if ob__idx_v0_3 == 399:
                                t__tmp_v64: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.slice(down_proj_tile__ssa_v6, [4, 64], [0, d0__ssa_v0])
                                t__tmp_v65: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.slice(resid1_tile__rv_v2, [4, 64], [0, d0__ssa_v0])
                                down_acc__ssa_v0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(t__tmp_v64, t__tmp_v65)
                                t__tmp_v66: pl.Tensor[[4, 64], pl.BF16] = pl.tensor.cast(down_acc__ssa_v0, target_type=pl.BF16, mode='round')
                                out__ssa_v9: pl.Tensor[[16, 4096, 5120], pl.BF16] = pl.tensor.assemble(out__iter_v7, t__tmp_v66, [b__idx_v0, p0__ssa_v0, d0__ssa_v0])
                                out__phi_v10: pl.Tensor[[16, 4096, 5120], pl.BF16] = pl.yield_(out__ssa_v9)
                            else:
                                out__phi_v10: pl.Tensor[[16, 4096, 5120], pl.BF16] = pl.yield_(out__iter_v7)
                            down_proj_tile__rv_v5, out__rv_v8 = pl.yield_(down_proj_tile__ssa_v6, out__phi_v10)
                        down_proj_tile__rv_v3, out__rv_v6 = pl.yield_(down_proj_tile__rv_v5, out__rv_v8)
                k_cache__rv_v4, out__rv_v4, v_cache__rv_v4 = pl.yield_(k_cache__rv_v6, out__rv_v6, v_cache__rv_v6)
            k_cache__rv_v2, out__rv_v2, v_cache__rv_v2 = pl.yield_(k_cache__rv_v4, out__rv_v4, v_cache__rv_v4)
        return out__rv_v2