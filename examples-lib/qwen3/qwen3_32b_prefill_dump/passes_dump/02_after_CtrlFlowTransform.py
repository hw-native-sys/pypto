# pypto.program: Qwen3SingleLayerPrefill
import pypto.language as pl

valid_len = pl.dynamic("valid_len")

@pl.program
class Qwen3SingleLayerPrefill:
    @pl.function
    def qwen3_prefill_layer(self, hidden_states: pl.Tensor[[16, 4096, 5120], pl.BF16], seq_lens: pl.Tensor[[16], pl.INT32], rope_cos: pl.Tensor[[4096, 128], pl.FP32], rope_sin: pl.Tensor[[4096, 128], pl.FP32], k_cache: pl.Tensor[[524288, 128], pl.BF16], v_cache: pl.Tensor[[524288, 128], pl.BF16], input_rms_weight: pl.Tensor[[1, 5120], pl.FP32], wq: pl.Tensor[[5120, 5120], pl.BF16], wk: pl.Tensor[[5120, 1024], pl.BF16], wv: pl.Tensor[[5120, 1024], pl.BF16], wo: pl.Tensor[[5120, 5120], pl.BF16], post_rms_weight: pl.Tensor[[1, 5120], pl.FP32], w_gate: pl.Tensor[[5120, 25600], pl.BF16], w_up: pl.Tensor[[5120, 25600], pl.BF16], w_down: pl.Tensor[[25600, 5120], pl.BF16], out: pl.Tensor[[16, 4096, 5120], pl.BF16]) -> pl.Tensor[[16, 4096, 5120], pl.BF16]:
        for b in pl.parallel(16):
            seq_len_b: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens, [b])
            tok_blocks: pl.Scalar[pl.INDEX] = (pl.cast(seq_len_b, pl.INDEX) + 4 - 1) // 4
            for p0_idx in pl.range(tok_blocks):
                p0: pl.Scalar[pl.INDEX] = p0_idx * 4
                valid_tok: pl.Scalar[pl.INDEX] = pl.min(4, pl.cast(seq_len_b, pl.INDEX) - p0)
                with pl.auto_incore():
                    sq_sum: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.create([4, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    sq_sum: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.muls(sq_sum, 0.0)
                    for kb in pl.range(20):
                        k0: pl.Scalar[pl.INDEX] = kb * 256
                        x_chunk: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.reshape(pl.tensor.cast(pl.tensor.slice(hidden_states, [1, 4, 256], [b, p0, k0], [1, valid_tok, 256]), target_type=pl.FP32, mode='round'), [4, 256])
                        sq_sum: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.add(sq_sum, pl.tensor.row_sum(pl.tensor.mul(x_chunk, x_chunk)))
                    inv_rms: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.rsqrt(pl.tensor.adds(pl.tensor.muls(sq_sum, 0.000195313), 1e-06))
                    q_proj_tile: pl.Tensor[[4, 5120], pl.BF16] = pl.tensor.create([4, 5120], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                    k_proj_tile: pl.Tensor[[4, 1024], pl.BF16] = pl.tensor.create([4, 1024], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                    v_proj_tile: pl.Tensor[[4, 1024], pl.BF16] = pl.tensor.create([4, 1024], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                    for ob in pl.parallel(80, chunk=8):
                        q0: pl.Scalar[pl.INDEX] = ob * 64
                        q_acc: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.create([4, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        q_acc: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.muls(q_acc, 0.0)
                        for kb_1 in pl.range(20):
                            k0: pl.Scalar[pl.INDEX] = kb_1 * 256
                            x_chunk: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.reshape(pl.tensor.cast(pl.tensor.slice(hidden_states, [1, 4, 256], [b, p0, k0], [1, valid_tok, 256]), target_type=pl.FP32, mode='round'), [4, 256])
                            gamma: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(input_rms_weight, [1, 256], [0, k0])
                            normed: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(x_chunk, inv_rms), gamma)
                            wq_chunk: pl.Tensor[[256, 64], pl.BF16] = pl.tensor.slice(wq, [256, 64], [k0, q0])
                            q_acc: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(q_acc, pl.tensor.matmul(pl.tensor.cast(normed, target_type=pl.BF16, mode='round'), wq_chunk, a_trans=False, b_trans=False, c_matrix_nz=False))
                        q_proj_tile: pl.Tensor[[4, 5120], pl.BF16] = pl.tensor.assemble(q_proj_tile, pl.tensor.cast(q_acc, target_type=pl.BF16, mode='round'), [0, q0])
                    for ob_1 in pl.parallel(32, chunk=8):
                        kv0: pl.Scalar[pl.INDEX] = ob_1 * 32
                        k_acc: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.create([4, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        v_acc: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.create([4, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        k_acc: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.muls(k_acc, 0.0)
                        v_acc: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.muls(v_acc, 0.0)
                        for kb_2 in pl.range(20):
                            k0: pl.Scalar[pl.INDEX] = kb_2 * 256
                            x_chunk: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.reshape(pl.tensor.cast(pl.tensor.slice(hidden_states, [1, 4, 256], [b, p0, k0], [1, valid_tok, 256]), target_type=pl.FP32, mode='round'), [4, 256])
                            gamma: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(input_rms_weight, [1, 256], [0, k0])
                            normed: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(x_chunk, inv_rms), gamma)
                            normed_bf16: pl.Tensor[[4, 256], pl.BF16] = pl.tensor.cast(normed, target_type=pl.BF16, mode='round')
                            wk_chunk: pl.Tensor[[256, 32], pl.BF16] = pl.tensor.slice(wk, [256, 32], [k0, kv0])
                            wv_chunk: pl.Tensor[[256, 32], pl.BF16] = pl.tensor.slice(wv, [256, 32], [k0, kv0])
                            k_acc: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.add(k_acc, pl.tensor.matmul(normed_bf16, wk_chunk, a_trans=False, b_trans=False, c_matrix_nz=False))
                            v_acc: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.add(v_acc, pl.tensor.matmul(normed_bf16, wv_chunk, a_trans=False, b_trans=False, c_matrix_nz=False))
                        k_proj_tile: pl.Tensor[[4, 1024], pl.BF16] = pl.tensor.assemble(k_proj_tile, pl.tensor.cast(k_acc, target_type=pl.BF16, mode='round'), [0, kv0])
                        v_proj_tile: pl.Tensor[[4, 1024], pl.BF16] = pl.tensor.assemble(v_proj_tile, pl.tensor.cast(v_acc, target_type=pl.BF16, mode='round'), [0, kv0])
                attn_tile: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.create([4, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                attn_tile: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.muls(attn_tile, 0.0)
                for ti in pl.range(valid_tok):
                    pos: pl.Scalar[pl.INDEX] = p0 + ti
                    ctx_len: pl.Scalar[pl.INDEX] = pos + 1
                    ctx_blocks: pl.Scalar[pl.INDEX] = (ctx_len + 120 - 1) // 120
                    cos_row: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.slice(rope_cos, [1, 128], [pos, 0])
                    sin_row: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.slice(rope_sin, [1, 128], [pos, 0])
                    cos_lo: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(cos_row, [1, 64], [0, 0])
                    cos_hi: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(cos_row, [1, 64], [0, 64])
                    sin_lo: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(sin_row, [1, 64], [0, 0])
                    sin_hi: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(sin_row, [1, 64], [0, 64])
                    with pl.auto_incore():
                        attn_row: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.create([1, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        attn_row: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.muls(attn_row, 0.0)
                        for kvh in pl.parallel(8, chunk=4):
                            kv_col: pl.Scalar[pl.INDEX] = kvh * 128
                            k_row: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.cast(pl.tensor.slice(k_proj_tile, [1, 128], [ti, kv_col]), target_type=pl.FP32, mode='round')
                            k_lo: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(k_row, [1, 64], [0, 0])
                            k_hi: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(k_row, [1, 64], [0, 64])
                            k_rot: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                            k_rot: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(k_rot, pl.tensor.sub(pl.tensor.col_expand_mul(k_lo, cos_lo), pl.tensor.col_expand_mul(k_hi, sin_lo)), [0, 0])
                            k_rot: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(k_rot, pl.tensor.add(pl.tensor.col_expand_mul(k_hi, cos_hi), pl.tensor.col_expand_mul(k_lo, sin_hi)), [0, 64])
                            cache_row: pl.Scalar[pl.INDEX] = b * 8 * 4096 + kvh * 4096 + pos
                            k_cache: pl.Tensor[[524288, 128], pl.BF16] = pl.tensor.assemble(k_cache, pl.tensor.cast(k_rot, target_type=pl.BF16, mode='round'), [cache_row, 0])
                            v_cache: pl.Tensor[[524288, 128], pl.BF16] = pl.tensor.assemble(v_cache, pl.tensor.slice(v_proj_tile, [1, 128], [ti, kv_col]), [cache_row, 0])
                        for h in pl.parallel(64, chunk=8):
                            kvh: pl.Scalar[pl.INDEX] = h // 8
                            q_col: pl.Scalar[pl.INDEX] = h * 128
                            q_row: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.cast(pl.tensor.slice(q_proj_tile, [1, 128], [ti, q_col]), target_type=pl.FP32, mode='round')
                            q_lo: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(q_row, [1, 64], [0, 0])
                            q_hi: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(q_row, [1, 64], [0, 64])
                            q_rot: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                            q_rot: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(q_rot, pl.tensor.sub(pl.tensor.col_expand_mul(q_lo, cos_lo), pl.tensor.col_expand_mul(q_hi, sin_lo)), [0, 0])
                            q_rot: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(q_rot, pl.tensor.add(pl.tensor.col_expand_mul(q_hi, cos_hi), pl.tensor.col_expand_mul(q_lo, sin_hi)), [0, 64])
                            q_rot_bf16: pl.Tensor[[1, 128], pl.BF16] = pl.tensor.cast(q_rot, target_type=pl.BF16, mode='round')
                            oi: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                            li: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                            mi: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                            oi: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.muls(oi, 0.0)
                            li: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.muls(li, 0.0)
                            mi: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.muls(mi, 0.0)
                            for sb in pl.range(ctx_blocks):
                                s0: pl.Scalar[pl.INDEX] = sb * 120
                                valid_len: pl.Scalar[pl.INDEX] = pl.min(120, ctx_len - s0)
                                cache_row0: pl.Scalar[pl.INDEX] = b * 8 * 4096 + kvh * 4096 + s0
                                k_tile: pl.Tensor[[120, 128], pl.BF16, pl.TensorView(valid_shape=[valid_len, 128], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(k_cache, [120, 128], [cache_row0, 0], [valid_len, 128])
                                v_tile: pl.Tensor[[120, 128], pl.BF16, pl.TensorView(valid_shape=[valid_len, 128], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(v_cache, [120, 128], [cache_row0, 0], [valid_len, 128])
                                scores: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.muls(pl.tensor.matmul(q_rot_bf16, k_tile, a_trans=False, b_trans=True, c_matrix_nz=False), 0.0883883)
                                scores_valid: pl.Tensor[[1, 120], pl.FP32, pl.TensorView(valid_shape=[1, valid_len], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(scores, [1, 120], [0, 0], [1, valid_len])
                                cur_mi: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.cast(pl.tensor.row_max(scores_valid), target_type=pl.FP32, mode='round')
                                exp_scores: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.exp(pl.tensor.row_expand_sub(scores_valid, cur_mi))
                                cur_li: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.cast(pl.tensor.row_sum(exp_scores), target_type=pl.FP32, mode='round')
                                exp_pad: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.create([1, 120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                                exp_pad: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.muls(exp_pad, 0.0)
                                exp_pad: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.assemble(exp_pad, exp_scores, [0, 0])
                                oi_tmp: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.matmul(pl.tensor.cast(exp_pad, target_type=pl.BF16, mode='round'), v_tile, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                                if sb == 0:
                                    oi: pl.Tensor[[1, 128], pl.FP32] = oi_tmp
                                    li: pl.Tensor[[1, 1], pl.FP32] = cur_li
                                    mi: pl.Tensor[[1, 1], pl.FP32] = cur_mi
                                else:
                                    mi_new: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.maximum(mi, cur_mi)
                                    alpha: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(mi, mi_new))
                                    beta: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(cur_mi, mi_new))
                                    li: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.add(pl.tensor.mul(alpha, li), pl.tensor.mul(beta, cur_li))
                                    oi: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.add(pl.tensor.row_expand_mul(oi, alpha), pl.tensor.row_expand_mul(oi_tmp, beta))
                                    mi: pl.Tensor[[1, 1], pl.FP32] = mi_new
                            ctx: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.row_expand_div(oi, li)
                            attn_row: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.assemble(attn_row, ctx, [0, q_col])
                        attn_tile: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.assemble(attn_tile, attn_row, [ti, 0])
                with pl.auto_incore():
                    resid1_tile: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.create([4, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    for ob_2 in pl.parallel(80, chunk=8):
                        o0: pl.Scalar[pl.INDEX] = ob_2 * 64
                        o_acc: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.create([4, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        o_acc: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.muls(o_acc, 0.0)
                        for kb_3 in pl.range(20):
                            k0: pl.Scalar[pl.INDEX] = kb_3 * 256
                            a_chunk: pl.Tensor[[4, 256], pl.BF16] = pl.tensor.cast(pl.tensor.slice(attn_tile, [4, 256], [0, k0]), target_type=pl.BF16, mode='round')
                            w_chunk: pl.Tensor[[256, 64], pl.BF16] = pl.tensor.slice(wo, [256, 64], [k0, o0])
                            o_acc: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(o_acc, pl.tensor.matmul(a_chunk, w_chunk, a_trans=False, b_trans=False, c_matrix_nz=False))
                        resid: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.reshape(pl.tensor.cast(pl.tensor.slice(hidden_states, [1, 4, 64], [b, p0, o0], [1, valid_tok, 64]), target_type=pl.FP32, mode='round'), [4, 64])
                        resid1_tile: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.assemble(resid1_tile, pl.tensor.add(o_acc, resid), [0, o0])
                    sq_sum: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.create([4, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    sq_sum: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.muls(sq_sum, 0.0)
                    for kb_4 in pl.range(20):
                        k0: pl.Scalar[pl.INDEX] = kb_4 * 256
                        x_chunk: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.slice(resid1_tile, [4, 256], [0, k0])
                        sq_sum: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.add(sq_sum, pl.tensor.row_sum(pl.tensor.mul(x_chunk, x_chunk)))
                    inv_rms: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.rsqrt(pl.tensor.adds(pl.tensor.muls(sq_sum, 0.000195313), 1e-06))
                    post_norm_tile: pl.Tensor[[4, 5120], pl.BF16] = pl.tensor.create([4, 5120], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                    down_proj_tile: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.create([4, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    down_proj_tile: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.muls(down_proj_tile, 0.0)
                    for kb_5 in pl.range(20):
                        k0: pl.Scalar[pl.INDEX] = kb_5 * 256
                        x_chunk: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.slice(resid1_tile, [4, 256], [0, k0])
                        gamma: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(post_rms_weight, [1, 256], [0, k0])
                        normed: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(x_chunk, inv_rms), gamma)
                        post_norm_tile: pl.Tensor[[4, 5120], pl.BF16] = pl.tensor.assemble(post_norm_tile, pl.tensor.cast(normed, target_type=pl.BF16, mode='round'), [0, k0])
                    for ob_3 in pl.range(400):
                        o0: pl.Scalar[pl.INDEX] = ob_3 * 64
                        gate_acc: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.create([4, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        up_acc: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.create([4, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        gate_acc: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.muls(gate_acc, 0.0)
                        up_acc: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.muls(up_acc, 0.0)
                        for kb_6 in pl.range(20):
                            k0: pl.Scalar[pl.INDEX] = kb_6 * 256
                            post_chunk: pl.Tensor[[4, 256], pl.BF16] = pl.tensor.slice(post_norm_tile, [4, 256], [0, k0])
                            wg: pl.Tensor[[256, 64], pl.BF16] = pl.tensor.slice(w_gate, [256, 64], [k0, o0])
                            wu: pl.Tensor[[256, 64], pl.BF16] = pl.tensor.slice(w_up, [256, 64], [k0, o0])
                            gate_acc: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(gate_acc, pl.tensor.matmul(post_chunk, wg, a_trans=False, b_trans=False, c_matrix_nz=False))
                            up_acc: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(up_acc, pl.tensor.matmul(post_chunk, wu, a_trans=False, b_trans=False, c_matrix_nz=False))
                        sigmoid: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.recip(pl.tensor.adds(pl.tensor.exp(pl.tensor.neg(gate_acc)), 1.0))
                        mlp_chunk: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.mul(pl.tensor.mul(gate_acc, sigmoid), up_acc)
                        mlp_chunk_bf16: pl.Tensor[[4, 64], pl.BF16] = pl.tensor.cast(mlp_chunk, target_type=pl.BF16, mode='round')
                        for dob in pl.parallel(80, chunk=8):
                            d0: pl.Scalar[pl.INDEX] = dob * 64
                            down_prev: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.slice(down_proj_tile, [4, 64], [0, d0])
                            w_down_chunk: pl.Tensor[[64, 64], pl.BF16] = pl.tensor.slice(w_down, [64, 64], [o0, d0])
                            down_next: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(down_prev, pl.tensor.matmul(mlp_chunk_bf16, w_down_chunk, a_trans=False, b_trans=False, c_matrix_nz=False))
                            down_proj_tile: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.assemble(down_proj_tile, down_next, [0, d0])
                            if ob_3 == 399:
                                down_acc: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(pl.tensor.slice(down_proj_tile, [4, 64], [0, d0]), pl.tensor.slice(resid1_tile, [4, 64], [0, d0]))
                                out: pl.Tensor[[16, 4096, 5120], pl.BF16] = pl.tensor.assemble(out, pl.tensor.cast(down_acc, target_type=pl.BF16, mode='round'), [b, p0, d0])
        return out