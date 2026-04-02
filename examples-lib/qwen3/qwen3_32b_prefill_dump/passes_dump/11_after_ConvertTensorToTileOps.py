# pypto.program: Qwen3SingleLayerPrefill
import pypto.language as pl

valid_len__ssa_v0 = pl.dynamic("valid_len__ssa_v0")
valid_tok__ssa_v0 = pl.dynamic("valid_tok__ssa_v0")
valid_tok__ssa_v0_1 = pl.dynamic("valid_tok__ssa_v0_1")
valid_tok__ssa_v0_2 = pl.dynamic("valid_tok__ssa_v0_2")
valid_tok__ssa_v0_3 = pl.dynamic("valid_tok__ssa_v0_3")

@pl.program
class Qwen3SingleLayerPrefill:
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_prefill_layer_incore_0(self, b__idx_v0: pl.Scalar[pl.INDEX], hidden_states__ssa_v0: pl.Tensor[[16, 4096, 5120], pl.BF16], p0__ssa_v0: pl.Scalar[pl.INDEX], sq_sum__ssa_v0: pl.Tensor[[4, 1], pl.FP32], valid_tok__ssa_v0: pl.Scalar[pl.INDEX], ret0__out: pl.Out[pl.Tensor[[4, 1], pl.FP32]]) -> pl.Tensor[[4, 1], pl.FP32]:
        sq_sum__tile: pl.Tile[[4, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.load(sq_sum__ssa_v0, [0, 0], [4, 1], [4, 1], target_memory=pl.Mem.Vec, transpose=False)
        sq_sum__tile_1: pl.Tile[[4, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(sq_sum__tile, 0.0)
        for kb__idx_v0, (sq_sum__iter_v2,) in pl.range(20, init_values=(sq_sum__tile_1,)):
            k0__ssa_v0: pl.Scalar[pl.INDEX] = kb__idx_v0 * 256
            t__tile: pl.Tile[[1, 4, 256], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_tok__ssa_v0, 256])] = pl.tile.load(hidden_states__ssa_v0, [b__idx_v0, p0__ssa_v0, k0__ssa_v0], [1, 4, 256], [1, valid_tok__ssa_v0, 256], target_memory=pl.Mem.Vec, transpose=False)
            t__tile_1: pl.Tile[[1, 4, 256], pl.FP32, pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode='round')
            x_chunk__tile: pl.Tile[[4, 256], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_1, [4, 256])
            t__tile_2: pl.Tile[[4, 256], pl.FP32, pl.Mem.Vec] = pl.tile.mul(x_chunk__tile, x_chunk__tile)
            tmp_tile: pl.Tile[[4, 256], pl.FP32, pl.Mem.Vec] = pl.tile.create([4, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_3: pl.Tile[[4, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.row_sum(t__tile_2, tmp_tile)
            sq_sum__tile_2: pl.Tile[[4, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.add(sq_sum__iter_v2, t__tile_3)
            sq_sum__rv_v3: pl.Tile[[4, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.yield_(sq_sum__tile_2)
        t__tile_4: pl.Tile[[4, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(sq_sum__rv_v3, 0.000195313)
        t__tile_5: pl.Tile[[4, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.adds(t__tile_4, 1e-06)
        inv_rms__tile: pl.Tile[[4, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.rsqrt(t__tile_5)
        ret0__store: pl.Tensor[[4, 1], pl.FP32] = pl.tile.store(inv_rms__tile, [0, 0], ret0__out)
        return ret0__store
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_prefill_layer_incore_1(self, b__idx_v0: pl.Scalar[pl.INDEX], hidden_states__ssa_v0: pl.Tensor[[16, 4096, 5120], pl.BF16], input_rms_weight__ssa_v0: pl.Tensor[[1, 5120], pl.FP32], inv_rms__ssa_v0: pl.Tensor[[4, 1], pl.FP32], ob__co_idx_v0: pl.Scalar[pl.INDEX], p0__ssa_v0: pl.Scalar[pl.INDEX], q_proj_tile__co_l0_iter_v1: pl.Out[pl.Tensor[[4, 5120], pl.BF16]], valid_tok__ssa_v0_1: pl.Scalar[pl.INDEX], wq__ssa_v0: pl.Tensor[[5120, 5120], pl.BF16]) -> pl.Tensor[[4, 5120], pl.BF16]:
        inv_rms__tile: pl.Tile[[4, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.load(inv_rms__ssa_v0, [0, 0], [4, 1], [4, 1], target_memory=pl.Mem.Vec, transpose=False)
        for ob__ci_idx_v0, (q_proj_tile__co_l1_iter_v1,) in pl.parallel(8, init_values=(q_proj_tile__co_l0_iter_v1,)):
            q0__ssa_v0: pl.Scalar[pl.INDEX] = (0 + (ob__co_idx_v0 * 8 + ob__ci_idx_v0) * 1) * 64
            q_acc__tile: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([4, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            q_acc__tile_1: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(q_acc__tile, 0.0)
            for kb__idx_v0, (q_acc__iter_v2,) in pl.range(20, init_values=(q_acc__tile_1,)):
                k0__ssa_v1: pl.Scalar[pl.INDEX] = kb__idx_v0 * 256
                t__tile: pl.Tile[[1, 4, 256], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_tok__ssa_v0_1, 256])] = pl.tile.load(hidden_states__ssa_v0, [b__idx_v0, p0__ssa_v0, k0__ssa_v1], [1, 4, 256], [1, valid_tok__ssa_v0_1, 256], target_memory=pl.Mem.Vec, transpose=False)
                t__tile_1: pl.Tile[[1, 4, 256], pl.FP32, pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode='round')
                x_chunk__tile: pl.Tile[[4, 256], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_1, [4, 256])
                gamma__tile: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(input_rms_weight__ssa_v0, [0, k0__ssa_v1], [1, 256], [1, 256], target_memory=pl.Mem.Vec, transpose=False)
                t__tile_2: pl.Tile[[4, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(x_chunk__tile, inv_rms__tile)
                normed__tile: pl.Tile[[4, 256], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_2, gamma__tile)
                wq_chunk__tile: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(wq__ssa_v0, [k0__ssa_v1, q0__ssa_v0], [256, 64], [256, 64], target_memory=pl.Mem.Mat, transpose=False)
                t__tile_3: pl.Tile[[4, 256], pl.BF16, pl.Mem.Vec] = pl.tile.cast(normed__tile, target_type=pl.BF16, mode='round')
                t__tile_4: pl.Tile[[4, 64], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(t__tile_3, wq_chunk__tile)
                q_acc__tile_2: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(q_acc__iter_v2, t__tile_4)
                q_acc__rv_v3: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(q_acc__tile_2)
            t__tile_5: pl.Tile[[4, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(q_acc__rv_v3, target_type=pl.BF16, mode='round')
            q_proj_tile__tile: pl.Tensor[[4, 5120], pl.BF16] = pl.tile.store(t__tile_5, [0, q0__ssa_v0], q_proj_tile__co_l1_iter_v1)
            q_proj_tile__co_l1_rv_v1: pl.Tensor[[4, 5120], pl.BF16] = pl.yield_(q_proj_tile__tile)
        return q_proj_tile__co_l1_rv_v1
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_prefill_layer_incore_10(self, b__idx_v0: pl.Scalar[pl.INDEX], dob__co_idx_v0: pl.Scalar[pl.INDEX], down_proj_tile__co_l0_iter_v4: pl.InOut[pl.Tensor[[4, 5120], pl.FP32]], mlp_chunk_bf16__ssa_v0: pl.Tensor[[4, 64], pl.BF16], o0__ssa_v1: pl.Scalar[pl.INDEX], ob__idx_v0: pl.Scalar[pl.INDEX], out__co_l0_iter_v7: pl.Out[pl.Tensor[[16, 4096, 5120], pl.BF16]], p0__ssa_v0: pl.Scalar[pl.INDEX], resid1_tile__co_l0_rv_v1: pl.Tensor[[4, 5120], pl.FP32], w_down__ssa_v0: pl.Tensor[[25600, 5120], pl.BF16]) -> tuple[pl.Tensor[[4, 5120], pl.FP32], pl.Tensor[[16, 4096, 5120], pl.BF16]]:
        for dob__ci_idx_v0, (down_proj_tile__co_l1_iter_v4, out__co_l1_iter_v7) in pl.parallel(8, init_values=(down_proj_tile__co_l0_iter_v4, out__co_l0_iter_v7)):
            d0__ssa_v0: pl.Scalar[pl.INDEX] = (0 + (dob__co_idx_v0 * 8 + dob__ci_idx_v0) * 1) * 64
            down_prev__tile: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(down_proj_tile__co_l1_iter_v4, [0, d0__ssa_v0], [4, 64], [4, 64], target_memory=pl.Mem.Vec, transpose=False)
            w_down_chunk__tile: pl.Tile[[64, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(w_down__ssa_v0, [o0__ssa_v1, d0__ssa_v0], [64, 64], [64, 64], target_memory=pl.Mem.Mat, transpose=False)
            lhs_mat: pl.Tile[[4, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(mlp_chunk_bf16__ssa_v0, [0, 0], [4, 64], [4, 64], target_memory=pl.Mem.Mat, transpose=False)
            t__tile: pl.Tile[[4, 64], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(lhs_mat, w_down_chunk__tile)
            down_next__tile: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(down_prev__tile, t__tile)
            down_proj_tile__tile: pl.Tensor[[4, 5120], pl.FP32] = pl.tile.store(down_next__tile, [0, d0__ssa_v0], down_proj_tile__co_l1_iter_v4)
            if ob__idx_v0 == 399:
                t__tile_1: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(down_proj_tile__tile, [0, d0__ssa_v0], [4, 64], [4, 64], target_memory=pl.Mem.Vec, transpose=False)
                t__tile_2: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(resid1_tile__co_l0_rv_v1, [0, d0__ssa_v0], [4, 64], [4, 64], target_memory=pl.Mem.Vec, transpose=False)
                down_acc__tile: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_1, t__tile_2)
                t__tile_3: pl.Tile[[4, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(down_acc__tile, target_type=pl.BF16, mode='round')
                out__tile: pl.Tensor[[16, 4096, 5120], pl.BF16] = pl.tile.store(t__tile_3, [b__idx_v0, p0__ssa_v0, d0__ssa_v0], out__co_l1_iter_v7)
                out__phi_v10: pl.Tensor[[16, 4096, 5120], pl.BF16] = pl.yield_(out__tile)
            else:
                out__phi_v10: pl.Tensor[[16, 4096, 5120], pl.BF16] = pl.yield_(out__co_l1_iter_v7)
            down_proj_tile__co_l1_rv_v4, out__co_l1_rv_v7 = pl.yield_(down_proj_tile__tile, out__phi_v10)
        return down_proj_tile__co_l1_rv_v4, out__co_l1_rv_v7
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_prefill_layer_incore_2(self, b__idx_v0: pl.Scalar[pl.INDEX], hidden_states__ssa_v0: pl.Tensor[[16, 4096, 5120], pl.BF16], input_rms_weight__ssa_v0: pl.Tensor[[1, 5120], pl.FP32], inv_rms__ssa_v0: pl.Tensor[[4, 1], pl.FP32], k_proj_tile__co_l0_iter_v1: pl.Out[pl.Tensor[[4, 1024], pl.BF16]], ob__co_idx_v0: pl.Scalar[pl.INDEX], p0__ssa_v0: pl.Scalar[pl.INDEX], v_proj_tile__co_l0_iter_v1: pl.Out[pl.Tensor[[4, 1024], pl.BF16]], valid_tok__ssa_v0_2: pl.Scalar[pl.INDEX], wk__ssa_v0: pl.Tensor[[5120, 1024], pl.BF16], wv__ssa_v0: pl.Tensor[[5120, 1024], pl.BF16]) -> tuple[pl.Tensor[[4, 1024], pl.BF16], pl.Tensor[[4, 1024], pl.BF16]]:
        inv_rms__tile: pl.Tile[[4, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.load(inv_rms__ssa_v0, [0, 0], [4, 1], [4, 1], target_memory=pl.Mem.Vec, transpose=False)
        for ob__ci_idx_v0, (k_proj_tile__co_l1_iter_v1, v_proj_tile__co_l1_iter_v1) in pl.parallel(8, init_values=(k_proj_tile__co_l0_iter_v1, v_proj_tile__co_l0_iter_v1)):
            kv0__ssa_v0: pl.Scalar[pl.INDEX] = (0 + (ob__co_idx_v0 * 8 + ob__ci_idx_v0) * 1) * 32
            k_acc__tile: pl.Tile[[4, 32], pl.FP32, pl.Mem.Vec] = pl.tile.create([4, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            v_acc__tile: pl.Tile[[4, 32], pl.FP32, pl.Mem.Vec] = pl.tile.create([4, 32], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            k_acc__tile_1: pl.Tile[[4, 32], pl.FP32, pl.Mem.Vec] = pl.tile.muls(k_acc__tile, 0.0)
            v_acc__tile_1: pl.Tile[[4, 32], pl.FP32, pl.Mem.Vec] = pl.tile.muls(v_acc__tile, 0.0)
            for kb__idx_v0, (k_acc__iter_v2, v_acc__iter_v2) in pl.range(20, init_values=(k_acc__tile_1, v_acc__tile_1)):
                k0__ssa_v2: pl.Scalar[pl.INDEX] = kb__idx_v0 * 256
                t__tile: pl.Tile[[1, 4, 256], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_tok__ssa_v0_2, 256])] = pl.tile.load(hidden_states__ssa_v0, [b__idx_v0, p0__ssa_v0, k0__ssa_v2], [1, 4, 256], [1, valid_tok__ssa_v0_2, 256], target_memory=pl.Mem.Vec, transpose=False)
                t__tile_1: pl.Tile[[1, 4, 256], pl.FP32, pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode='round')
                x_chunk__tile: pl.Tile[[4, 256], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_1, [4, 256])
                gamma__tile: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(input_rms_weight__ssa_v0, [0, k0__ssa_v2], [1, 256], [1, 256], target_memory=pl.Mem.Vec, transpose=False)
                t__tile_2: pl.Tile[[4, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(x_chunk__tile, inv_rms__tile)
                normed__tile: pl.Tile[[4, 256], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile_2, gamma__tile)
                normed_bf16__tile: pl.Tile[[4, 256], pl.BF16, pl.Mem.Vec] = pl.tile.cast(normed__tile, target_type=pl.BF16, mode='round')
                wk_chunk__tile: pl.Tile[[256, 32], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(wk__ssa_v0, [k0__ssa_v2, kv0__ssa_v0], [256, 32], [256, 32], target_memory=pl.Mem.Mat, transpose=False)
                wv_chunk__tile: pl.Tile[[256, 32], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(wv__ssa_v0, [k0__ssa_v2, kv0__ssa_v0], [256, 32], [256, 32], target_memory=pl.Mem.Mat, transpose=False)
                t__tile_3: pl.Tile[[4, 32], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(normed_bf16__tile, wk_chunk__tile)
                k_acc__tile_2: pl.Tile[[4, 32], pl.FP32, pl.Mem.Vec] = pl.tile.add(k_acc__iter_v2, t__tile_3)
                t__tile_4: pl.Tile[[4, 32], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(normed_bf16__tile, wv_chunk__tile)
                v_acc__tile_2: pl.Tile[[4, 32], pl.FP32, pl.Mem.Vec] = pl.tile.add(v_acc__iter_v2, t__tile_4)
                k_acc__rv_v3, v_acc__rv_v3 = pl.yield_(k_acc__tile_2, v_acc__tile_2)
            t__tile_5: pl.Tile[[4, 32], pl.BF16, pl.Mem.Vec] = pl.tile.cast(k_acc__rv_v3, target_type=pl.BF16, mode='round')
            k_proj_tile__tile: pl.Tensor[[4, 1024], pl.BF16] = pl.tile.store(t__tile_5, [0, kv0__ssa_v0], k_proj_tile__co_l1_iter_v1)
            t__tile_6: pl.Tile[[4, 32], pl.BF16, pl.Mem.Vec] = pl.tile.cast(v_acc__rv_v3, target_type=pl.BF16, mode='round')
            v_proj_tile__tile: pl.Tensor[[4, 1024], pl.BF16] = pl.tile.store(t__tile_6, [0, kv0__ssa_v0], v_proj_tile__co_l1_iter_v1)
            k_proj_tile__co_l1_rv_v1, v_proj_tile__co_l1_rv_v1 = pl.yield_(k_proj_tile__tile, v_proj_tile__tile)
        return k_proj_tile__co_l1_rv_v1, v_proj_tile__co_l1_rv_v1
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_prefill_layer_incore_3(self, attn_row__ssa_v0: pl.Tensor[[1, 5120], pl.FP32], ret0__out: pl.Out[pl.Tensor[[1, 5120], pl.FP32]]) -> pl.Tensor[[1, 5120], pl.FP32]:
        attn_row__tile: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec] = pl.tile.load(attn_row__ssa_v0, [0, 0], [1, 5120], [1, 5120], target_memory=pl.Mem.Vec, transpose=False)
        attn_row__tile_1: pl.Tile[[1, 5120], pl.FP32, pl.Mem.Vec] = pl.tile.muls(attn_row__tile, 0.0)
        ret0__store: pl.Tensor[[1, 5120], pl.FP32] = pl.tile.store(attn_row__tile_1, [0, 0], ret0__out)
        return ret0__store
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_prefill_layer_incore_4(self, b__idx_v0: pl.Scalar[pl.INDEX], cos_hi__ssa_v0: pl.Tensor[[1, 64], pl.FP32], cos_lo__ssa_v0: pl.Tensor[[1, 64], pl.FP32], k_cache__co_l0_iter_v7: pl.Out[pl.Tensor[[524288, 128], pl.BF16]], k_proj_tile__co_l0_rv_v1: pl.Tensor[[4, 1024], pl.BF16], kvh__co_idx_v0: pl.Scalar[pl.INDEX], pos__ssa_v0: pl.Scalar[pl.INDEX], sin_hi__ssa_v0: pl.Tensor[[1, 64], pl.FP32], sin_lo__ssa_v0: pl.Tensor[[1, 64], pl.FP32], ti__idx_v0: pl.Scalar[pl.INDEX], v_cache__co_l0_iter_v7: pl.Out[pl.Tensor[[524288, 128], pl.BF16]], v_proj_tile__co_l0_rv_v1: pl.Tensor[[4, 1024], pl.BF16]) -> tuple[pl.Tensor[[524288, 128], pl.BF16], pl.Tensor[[524288, 128], pl.BF16]]:
        cos_hi__tile: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(cos_hi__ssa_v0, [0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec, transpose=False)
        cos_lo__tile: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(cos_lo__ssa_v0, [0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec, transpose=False)
        sin_hi__tile: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(sin_hi__ssa_v0, [0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec, transpose=False)
        sin_lo__tile: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(sin_lo__ssa_v0, [0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec, transpose=False)
        for kvh__ci_idx_v0, (k_cache__co_l1_iter_v7, v_cache__co_l1_iter_v7) in pl.parallel(4, init_values=(k_cache__co_l0_iter_v7, v_cache__co_l0_iter_v7)):
            kv_col__ssa_v0: pl.Scalar[pl.INDEX] = (0 + (kvh__co_idx_v0 * 4 + kvh__ci_idx_v0) * 1) * 128
            t__tile: pl.Tile[[1, 128], pl.BF16, pl.Mem.Vec] = pl.tile.load(k_proj_tile__co_l0_rv_v1, [ti__idx_v0, kv_col__ssa_v0], [1, 128], [1, 128], target_memory=pl.Mem.Vec, transpose=False)
            k_row__tile: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode='round')
            k_lo__tile: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(k_row__tile, [1, 64], [0, 0])
            k_hi__tile: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(k_row__tile, [1, 64], [0, 64])
            k_rot__tile: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_1: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(k_lo__tile, cos_lo__tile)
            t__tile_2: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(k_hi__tile, sin_lo__tile)
            t__tile_3: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(t__tile_1, t__tile_2)
            k_rot__tile_1: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(k_rot__tile, t__tile_3, [0, 0])
            t__tile_4: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(k_hi__tile, cos_hi__tile)
            t__tile_5: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(k_lo__tile, sin_hi__tile)
            t__tile_6: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_4, t__tile_5)
            k_rot__tile_2: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(k_rot__tile_1, t__tile_6, [0, 64])
            cache_row__ssa_v0: pl.Scalar[pl.INDEX] = b__idx_v0 * 8 * 4096 + (0 + (kvh__co_idx_v0 * 4 + kvh__ci_idx_v0) * 1) * 4096 + pos__ssa_v0
            t__tile_7: pl.Tile[[1, 128], pl.BF16, pl.Mem.Vec] = pl.tile.cast(k_rot__tile_2, target_type=pl.BF16, mode='round')
            k_cache__tile: pl.Tensor[[524288, 128], pl.BF16] = pl.tile.store(t__tile_7, [cache_row__ssa_v0, 0], k_cache__co_l1_iter_v7)
            t__tile_8: pl.Tile[[1, 128], pl.BF16, pl.Mem.Vec] = pl.tile.load(v_proj_tile__co_l0_rv_v1, [ti__idx_v0, kv_col__ssa_v0], [1, 128], [1, 128], target_memory=pl.Mem.Vec, transpose=False)
            v_cache__tile: pl.Tensor[[524288, 128], pl.BF16] = pl.tile.store(t__tile_8, [cache_row__ssa_v0, 0], v_cache__co_l1_iter_v7)
            k_cache__co_l1_rv_v7, v_cache__co_l1_rv_v7 = pl.yield_(k_cache__tile, v_cache__tile)
        return k_cache__co_l1_rv_v7, v_cache__co_l1_rv_v7
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_prefill_layer_incore_5(self, attn_row__co_l0_iter_v2: pl.Out[pl.Tensor[[1, 5120], pl.FP32]], b__idx_v0: pl.Scalar[pl.INDEX], cos_hi__ssa_v0: pl.Tensor[[1, 64], pl.FP32], cos_lo__ssa_v0: pl.Tensor[[1, 64], pl.FP32], ctx_blocks__ssa_v0: pl.Scalar[pl.INDEX], ctx_len__ssa_v0: pl.Scalar[pl.INDEX], h__co_idx_v0: pl.Scalar[pl.INDEX], k_cache__co_l0_rv_v7: pl.Tensor[[524288, 128], pl.BF16], q_proj_tile__co_l0_rv_v1: pl.Tensor[[4, 5120], pl.BF16], sin_hi__ssa_v0: pl.Tensor[[1, 64], pl.FP32], sin_lo__ssa_v0: pl.Tensor[[1, 64], pl.FP32], ti__idx_v0: pl.Scalar[pl.INDEX], v_cache__co_l0_rv_v7: pl.Tensor[[524288, 128], pl.BF16]) -> pl.Tensor[[1, 5120], pl.FP32]:
        cos_hi__tile: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(cos_hi__ssa_v0, [0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec, transpose=False)
        cos_lo__tile: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(cos_lo__ssa_v0, [0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec, transpose=False)
        sin_hi__tile: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(sin_hi__ssa_v0, [0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec, transpose=False)
        sin_lo__tile: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(sin_lo__ssa_v0, [0, 0], [1, 64], [1, 64], target_memory=pl.Mem.Vec, transpose=False)
        for h__ci_idx_v0, (attn_row__co_l1_iter_v2,) in pl.parallel(8, init_values=(attn_row__co_l0_iter_v2,)):
            kvh__ssa_v1: pl.Scalar[pl.INDEX] = (0 + (h__co_idx_v0 * 8 + h__ci_idx_v0) * 1) // 8
            q_col__ssa_v0: pl.Scalar[pl.INDEX] = (0 + (h__co_idx_v0 * 8 + h__ci_idx_v0) * 1) * 128
            t__tile: pl.Tile[[1, 128], pl.BF16, pl.Mem.Vec] = pl.tile.load(q_proj_tile__co_l0_rv_v1, [ti__idx_v0, q_col__ssa_v0], [1, 128], [1, 128], target_memory=pl.Mem.Vec, transpose=False)
            q_row__tile: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.FP32, mode='round')
            q_lo__tile: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(q_row__tile, [1, 64], [0, 0])
            q_hi__tile: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.slice(q_row__tile, [1, 64], [0, 64])
            q_rot__tile: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_1: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(q_lo__tile, cos_lo__tile)
            t__tile_2: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(q_hi__tile, sin_lo__tile)
            t__tile_3: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.sub(t__tile_1, t__tile_2)
            q_rot__tile_1: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(q_rot__tile, t__tile_3, [0, 0])
            t__tile_4: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(q_hi__tile, cos_hi__tile)
            t__tile_5: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(q_lo__tile, sin_hi__tile)
            t__tile_6: pl.Tile[[1, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_4, t__tile_5)
            q_rot__tile_2: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(q_rot__tile_1, t__tile_6, [0, 64])
            q_rot_bf16__tile: pl.Tile[[1, 128], pl.BF16, pl.Mem.Vec] = pl.tile.cast(q_rot__tile_2, target_type=pl.BF16, mode='round')
            oi__tile: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            li__tile: pl.Tile[[1, 1], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 1], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            mi__tile: pl.Tile[[1, 1], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 1], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            oi__tile_1: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.muls(oi__tile, 0.0)
            li__tile_1: pl.Tile[[1, 1], pl.FP32, pl.Mem.Vec] = pl.tile.muls(li__tile, 0.0)
            mi__tile_1: pl.Tile[[1, 1], pl.FP32, pl.Mem.Vec] = pl.tile.muls(mi__tile, 0.0)
            for sb__idx_v0, (li__iter_v2, mi__iter_v2, oi__iter_v2) in pl.range(ctx_blocks__ssa_v0, init_values=(li__tile_1, mi__tile_1, oi__tile_1)):
                s0__ssa_v0: pl.Scalar[pl.INDEX] = sb__idx_v0 * 120
                valid_len__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(120, ctx_len__ssa_v0 - s0__ssa_v0)
                cache_row0__ssa_v0: pl.Scalar[pl.INDEX] = b__idx_v0 * 8 * 4096 + kvh__ssa_v1 * 4096 + s0__ssa_v0
                k_tile__tile: pl.Tile[[128, 120], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[128, valid_len__ssa_v0], slayout=pl.TileLayout.col_major)] = pl.tile.load(k_cache__co_l0_rv_v7, [cache_row0__ssa_v0, 0], [128, 120], [128, valid_len__ssa_v0], target_memory=pl.Mem.Mat, transpose=True)
                v_tile__tile: pl.Tile[[120, 128], pl.BF16, pl.Mem.Mat, pl.TileView(valid_shape=[valid_len__ssa_v0, 128], blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(v_cache__co_l0_rv_v7, [cache_row0__ssa_v0, 0], [120, 128], [valid_len__ssa_v0, 128], target_memory=pl.Mem.Mat, transpose=False)
                t__tile_7: pl.Tile[[1, 120], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(q_rot_bf16__tile, k_tile__tile)
                scores__tile: pl.Tile[[1, 120], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(t__tile_7, 0.0883883)
                scores_valid__tile: pl.Tile[[1, 120], pl.FP32, pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_len__ssa_v0])] = pl.tile.slice(scores__tile, [1, 120], [0, 0], [1, valid_len__ssa_v0])
                tmp_tile: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_8: pl.Tile[[1, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.row_max(scores_valid__tile, tmp_tile)
                cur_mi__tile: pl.Tile[[1, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.cast(t__tile_8, target_type=pl.FP32, mode='round')
                t__tile_9: pl.Tile[[1, 120], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_sub(scores_valid__tile, cur_mi__tile)
                exp_scores__tile: pl.Tile[[1, 120], pl.FP32, pl.Mem.Vec] = pl.tile.exp(t__tile_9)
                tmp_tile_1: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 128], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                t__tile_10: pl.Tile[[1, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.row_sum(exp_scores__tile, tmp_tile_1)
                cur_li__tile: pl.Tile[[1, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.cast(t__tile_10, target_type=pl.FP32, mode='round')
                exp_pad__tile: pl.Tile[[1, 120], pl.FP32, pl.Mem.Vec] = pl.tile.create([1, 120], dtype=pl.FP32, target_memory=pl.Mem.Vec)
                exp_pad__tile_1: pl.Tile[[1, 120], pl.FP32, pl.Mem.Vec] = pl.tile.muls(exp_pad__tile, 0.0)
                exp_pad__tile_2: pl.Tile[[1, 120], pl.FP32, pl.Mem.Vec] = pl.tile.assemble(exp_pad__tile_1, exp_scores__tile, [0, 0])
                t__tile_11: pl.Tile[[1, 120], pl.BF16, pl.Mem.Vec] = pl.tile.cast(exp_pad__tile_2, target_type=pl.BF16, mode='round')
                oi_tmp__tile: pl.Tile[[1, 128], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(t__tile_11, v_tile__tile)
                if sb__idx_v0 == 0:
                    oi__ssa_v4: pl.Tile[[1, 128], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = oi_tmp__tile
                    li__ssa_v4: pl.Tile[[1, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = cur_li__tile
                    mi__ssa_v4: pl.Tile[[1, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = cur_mi__tile
                    li__phi_v6, mi__phi_v6, oi__phi_v6 = pl.yield_(li__ssa_v4, mi__ssa_v4, oi__ssa_v4)
                else:
                    mi_new__tile: pl.Tile[[1, 1], pl.FP32, pl.Mem.Vec] = pl.tile.maximum(mi__iter_v2, cur_mi__tile)
                    t__tile_12: pl.Tile[[1, 1], pl.FP32, pl.Mem.Vec] = pl.tile.sub(mi__iter_v2, mi_new__tile)
                    alpha__tile: pl.Tile[[1, 1], pl.FP32, pl.Mem.Vec] = pl.tile.exp(t__tile_12)
                    t__tile_13: pl.Tile[[1, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.sub(cur_mi__tile, mi_new__tile)
                    beta__tile: pl.Tile[[1, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.exp(t__tile_13)
                    t__tile_14: pl.Tile[[1, 1], pl.FP32, pl.Mem.Vec] = pl.tile.mul(alpha__tile, li__iter_v2)
                    t__tile_15: pl.Tile[[1, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.mul(beta__tile, cur_li__tile)
                    li__tile_2: pl.Tile[[1, 1], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_14, t__tile_15)
                    t__tile_16: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(oi__iter_v2, alpha__tile)
                    t__tile_17: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(oi_tmp__tile, beta__tile)
                    oi__tile_2: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.add(t__tile_16, t__tile_17)
                    mi__ssa_v5: pl.Tile[[1, 1], pl.FP32, pl.Mem.Vec] = mi_new__tile
                    li__phi_v6, mi__phi_v6, oi__phi_v6 = pl.yield_(li__tile_2, mi__ssa_v5, oi__tile_2)
                li__rv_v3, mi__rv_v3, oi__rv_v3 = pl.yield_(li__phi_v6, mi__phi_v6, oi__phi_v6)
            ctx__tile: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_div(oi__rv_v3, li__rv_v3)
            attn_row__tile: pl.Tensor[[1, 5120], pl.FP32] = pl.tile.store(ctx__tile, [0, q_col__ssa_v0], attn_row__co_l1_iter_v2)
            attn_row__co_l1_rv_v2: pl.Tensor[[1, 5120], pl.FP32] = pl.yield_(attn_row__tile)
        return attn_row__co_l1_rv_v2
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_prefill_layer_incore_6(self, attn_tile__rv_v3: pl.Tensor[[4, 5120], pl.FP32], b__idx_v0: pl.Scalar[pl.INDEX], hidden_states__ssa_v0: pl.Tensor[[16, 4096, 5120], pl.BF16], ob__co_idx_v0: pl.Scalar[pl.INDEX], p0__ssa_v0: pl.Scalar[pl.INDEX], resid1_tile__co_l0_iter_v1: pl.Out[pl.Tensor[[4, 5120], pl.FP32]], valid_tok__ssa_v0_3: pl.Scalar[pl.INDEX], wo__ssa_v0: pl.Tensor[[5120, 5120], pl.BF16]) -> pl.Tensor[[4, 5120], pl.FP32]:
        for ob__ci_idx_v0, (resid1_tile__co_l1_iter_v1,) in pl.parallel(8, init_values=(resid1_tile__co_l0_iter_v1,)):
            o0__ssa_v0: pl.Scalar[pl.INDEX] = (0 + (ob__co_idx_v0 * 8 + ob__ci_idx_v0) * 1) * 64
            o_acc__tile: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.create([4, 64], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            o_acc__tile_1: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(o_acc__tile, 0.0)
            for kb__idx_v0, (o_acc__iter_v2,) in pl.range(20, init_values=(o_acc__tile_1,)):
                k0__ssa_v3: pl.Scalar[pl.INDEX] = kb__idx_v0 * 256
                t__tile: pl.Tile[[4, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(attn_tile__rv_v3, [0, k0__ssa_v3], [4, 256], [4, 256], target_memory=pl.Mem.Vec, transpose=False)
                a_chunk__tile: pl.Tile[[4, 256], pl.BF16, pl.Mem.Vec] = pl.tile.cast(t__tile, target_type=pl.BF16, mode='round')
                w_chunk__tile: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(wo__ssa_v0, [k0__ssa_v3, o0__ssa_v0], [256, 64], [256, 64], target_memory=pl.Mem.Mat, transpose=False)
                t__tile_1: pl.Tile[[4, 64], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(a_chunk__tile, w_chunk__tile)
                o_acc__tile_2: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(o_acc__iter_v2, t__tile_1)
                o_acc__rv_v3: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.yield_(o_acc__tile_2)
            t__tile_2: pl.Tile[[1, 4, 64], pl.BF16, pl.Mem.Vec, pl.TileView(valid_shape=[1, valid_tok__ssa_v0_3, 64])] = pl.tile.load(hidden_states__ssa_v0, [b__idx_v0, p0__ssa_v0, o0__ssa_v0], [1, 4, 64], [1, valid_tok__ssa_v0_3, 64], target_memory=pl.Mem.Vec, transpose=False)
            t__tile_3: pl.Tile[[1, 4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.cast(t__tile_2, target_type=pl.FP32, mode='round')
            resid__tile: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t__tile_3, [4, 64])
            t__tile_4: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(o_acc__rv_v3, resid__tile)
            resid1_tile__tile: pl.Tensor[[4, 5120], pl.FP32] = pl.tile.store(t__tile_4, [0, o0__ssa_v0], resid1_tile__co_l1_iter_v1)
            resid1_tile__co_l1_rv_v1: pl.Tensor[[4, 5120], pl.FP32] = pl.yield_(resid1_tile__tile)
        return resid1_tile__co_l1_rv_v1
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_prefill_layer_incore_7(self, resid1_tile__co_l0_rv_v1: pl.Tensor[[4, 5120], pl.FP32], sq_sum__ssa_v5: pl.Tensor[[4, 1], pl.FP32], ret0__out: pl.Out[pl.Tensor[[4, 1], pl.FP32]]) -> pl.Tensor[[4, 1], pl.FP32]:
        sq_sum__tile: pl.Tile[[4, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.load(sq_sum__ssa_v5, [0, 0], [4, 1], [4, 1], target_memory=pl.Mem.Vec, transpose=False)
        sq_sum__tile_1: pl.Tile[[4, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(sq_sum__tile, 0.0)
        for kb__idx_v0, (sq_sum__iter_v7,) in pl.range(20, init_values=(sq_sum__tile_1,)):
            k0__ssa_v4: pl.Scalar[pl.INDEX] = kb__idx_v0 * 256
            x_chunk__tile: pl.Tile[[4, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(resid1_tile__co_l0_rv_v1, [0, k0__ssa_v4], [4, 256], [4, 256], target_memory=pl.Mem.Vec, transpose=False)
            t__tile: pl.Tile[[4, 256], pl.FP32, pl.Mem.Vec] = pl.tile.mul(x_chunk__tile, x_chunk__tile)
            tmp_tile: pl.Tile[[4, 256], pl.FP32, pl.Mem.Vec] = pl.tile.create([4, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            t__tile_1: pl.Tile[[4, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.row_sum(t__tile, tmp_tile)
            sq_sum__tile_2: pl.Tile[[4, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.add(sq_sum__iter_v7, t__tile_1)
            sq_sum__rv_v8: pl.Tile[[4, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.yield_(sq_sum__tile_2)
        t__tile_2: pl.Tile[[4, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.muls(sq_sum__rv_v8, 0.000195313)
        t__tile_3: pl.Tile[[4, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.adds(t__tile_2, 1e-06)
        inv_rms__tile: pl.Tile[[4, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.rsqrt(t__tile_3)
        ret0__store: pl.Tensor[[4, 1], pl.FP32] = pl.tile.store(inv_rms__tile, [0, 0], ret0__out)
        return ret0__store
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_prefill_layer_incore_8(self, down_proj_tile__ssa_v0: pl.Tensor[[4, 5120], pl.FP32], inv_rms__ssa_v1: pl.Tensor[[4, 1], pl.FP32], post_norm_tile__ssa_v0: pl.Out[pl.Tensor[[4, 5120], pl.BF16]], post_rms_weight__ssa_v0: pl.Tensor[[1, 5120], pl.FP32], resid1_tile__co_l0_rv_v1: pl.Tensor[[4, 5120], pl.FP32], ret0__out: pl.Out[pl.Tensor[[4, 5120], pl.FP32]]) -> tuple[pl.Tensor[[4, 5120], pl.FP32], pl.Tensor[[4, 5120], pl.BF16]]:
        down_proj_tile__tile: pl.Tile[[4, 5120], pl.FP32, pl.Mem.Vec] = pl.tile.load(down_proj_tile__ssa_v0, [0, 0], [4, 5120], [4, 5120], target_memory=pl.Mem.Vec, transpose=False)
        inv_rms__tile: pl.Tile[[4, 1], pl.FP32, pl.Mem.Vec, pl.TileView(blayout=pl.TileLayout.col_major)] = pl.tile.load(inv_rms__ssa_v1, [0, 0], [4, 1], [4, 1], target_memory=pl.Mem.Vec, transpose=False)
        down_proj_tile__tile_1: pl.Tile[[4, 5120], pl.FP32, pl.Mem.Vec] = pl.tile.muls(down_proj_tile__tile, 0.0)
        for kb__idx_v0, (post_norm_tile__iter_v1,) in pl.range(20, init_values=(post_norm_tile__ssa_v0,)):
            k0__ssa_v5: pl.Scalar[pl.INDEX] = kb__idx_v0 * 256
            x_chunk__tile: pl.Tile[[4, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(resid1_tile__co_l0_rv_v1, [0, k0__ssa_v5], [4, 256], [4, 256], target_memory=pl.Mem.Vec, transpose=False)
            gamma__tile: pl.Tile[[1, 256], pl.FP32, pl.Mem.Vec] = pl.tile.load(post_rms_weight__ssa_v0, [0, k0__ssa_v5], [1, 256], [1, 256], target_memory=pl.Mem.Vec, transpose=False)
            t__tile: pl.Tile[[4, 256], pl.FP32, pl.Mem.Vec] = pl.tile.row_expand_mul(x_chunk__tile, inv_rms__tile)
            normed__tile: pl.Tile[[4, 256], pl.FP32, pl.Mem.Vec] = pl.tile.col_expand_mul(t__tile, gamma__tile)
            t__tile_1: pl.Tile[[4, 256], pl.BF16, pl.Mem.Vec] = pl.tile.cast(normed__tile, target_type=pl.BF16, mode='round')
            post_norm_tile__tile: pl.Tensor[[4, 5120], pl.BF16] = pl.tile.store(t__tile_1, [0, k0__ssa_v5], post_norm_tile__iter_v1)
            post_norm_tile__rv_v2: pl.Tensor[[4, 5120], pl.BF16] = pl.yield_(post_norm_tile__tile)
        ret0__store: pl.Tensor[[4, 5120], pl.FP32] = pl.tile.store(down_proj_tile__tile_1, [0, 0], ret0__out)
        return ret0__store, post_norm_tile__rv_v2
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_prefill_layer_incore_9(self, gate_acc__ssa_v0: pl.Tensor[[4, 64], pl.FP32], o0__ssa_v1: pl.Scalar[pl.INDEX], post_norm_tile__rv_v2: pl.Tensor[[4, 5120], pl.BF16], up_acc__ssa_v0: pl.Tensor[[4, 64], pl.FP32], w_gate__ssa_v0: pl.Tensor[[5120, 25600], pl.BF16], w_up__ssa_v0: pl.Tensor[[5120, 25600], pl.BF16], ret0__out: pl.Out[pl.Tensor[[4, 64], pl.BF16]]) -> pl.Tensor[[4, 64], pl.BF16]:
        gate_acc__tile: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(gate_acc__ssa_v0, [0, 0], [4, 64], [4, 64], target_memory=pl.Mem.Vec, transpose=False)
        up_acc__tile: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.load(up_acc__ssa_v0, [0, 0], [4, 64], [4, 64], target_memory=pl.Mem.Vec, transpose=False)
        gate_acc__tile_1: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(gate_acc__tile, 0.0)
        up_acc__tile_1: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.muls(up_acc__tile, 0.0)
        for kb__idx_v0, (gate_acc__iter_v2, up_acc__iter_v2) in pl.range(20, init_values=(gate_acc__tile_1, up_acc__tile_1)):
            k0__ssa_v6: pl.Scalar[pl.INDEX] = kb__idx_v0 * 256
            post_chunk__tile: pl.Tile[[4, 256], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(post_norm_tile__rv_v2, [0, k0__ssa_v6], [4, 256], [4, 256], target_memory=pl.Mem.Mat, transpose=False)
            wg__tile: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(w_gate__ssa_v0, [k0__ssa_v6, o0__ssa_v1], [256, 64], [256, 64], target_memory=pl.Mem.Mat, transpose=False)
            wu__tile: pl.Tile[[256, 64], pl.BF16, pl.Mem.Mat, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major)] = pl.tile.load(w_up__ssa_v0, [k0__ssa_v6, o0__ssa_v1], [256, 64], [256, 64], target_memory=pl.Mem.Mat, transpose=False)
            t__tile: pl.Tile[[4, 64], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(post_chunk__tile, wg__tile)
            gate_acc__tile_2: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(gate_acc__iter_v2, t__tile)
            t__tile_1: pl.Tile[[4, 64], pl.FP32, pl.Mem.Acc, pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024)] = pl.tile.matmul(post_chunk__tile, wu__tile)
            up_acc__tile_2: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.add(up_acc__iter_v2, t__tile_1)
            gate_acc__rv_v3, up_acc__rv_v3 = pl.yield_(gate_acc__tile_2, up_acc__tile_2)
        t__tile_2: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.neg(gate_acc__rv_v3)
        t__tile_3: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.exp(t__tile_2)
        t__tile_4: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.adds(t__tile_3, 1.0)
        sigmoid__tile: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.recip(t__tile_4)
        t__tile_5: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(gate_acc__rv_v3, sigmoid__tile)
        mlp_chunk__tile: pl.Tile[[4, 64], pl.FP32, pl.Mem.Vec] = pl.tile.mul(t__tile_5, up_acc__rv_v3)
        mlp_chunk_bf16__tile: pl.Tile[[4, 64], pl.BF16, pl.Mem.Vec] = pl.tile.cast(mlp_chunk__tile, target_type=pl.BF16, mode='round')
        ret0__store: pl.Tensor[[4, 64], pl.BF16] = pl.tile.store(mlp_chunk_bf16__tile, [0, 0], ret0__out)
        return ret0__store
    @pl.function(type=pl.FunctionType.Orchestration)
    def qwen3_prefill_layer(self, hidden_states__ssa_v0: pl.Tensor[[16, 4096, 5120], pl.BF16], seq_lens__ssa_v0: pl.Tensor[[16], pl.INT32], rope_cos__ssa_v0: pl.Tensor[[4096, 128], pl.FP32], rope_sin__ssa_v0: pl.Tensor[[4096, 128], pl.FP32], k_cache__ssa_v0: pl.Tensor[[524288, 128], pl.BF16], v_cache__ssa_v0: pl.Tensor[[524288, 128], pl.BF16], input_rms_weight__ssa_v0: pl.Tensor[[1, 5120], pl.FP32], wq__ssa_v0: pl.Tensor[[5120, 5120], pl.BF16], wk__ssa_v0: pl.Tensor[[5120, 1024], pl.BF16], wv__ssa_v0: pl.Tensor[[5120, 1024], pl.BF16], wo__ssa_v0: pl.Tensor[[5120, 5120], pl.BF16], post_rms_weight__ssa_v0: pl.Tensor[[1, 5120], pl.FP32], w_gate__ssa_v0: pl.Tensor[[5120, 25600], pl.BF16], w_up__ssa_v0: pl.Tensor[[5120, 25600], pl.BF16], w_down__ssa_v0: pl.Tensor[[25600, 5120], pl.BF16], out__ssa_v0: pl.Tensor[[16, 4096, 5120], pl.BF16]) -> pl.Tensor[[16, 4096, 5120], pl.BF16]:
        for b__idx_v0, (k_cache__iter_v1, out__iter_v1, v_cache__iter_v1) in pl.parallel(16, init_values=(k_cache__ssa_v0, out__ssa_v0, v_cache__ssa_v0)):
            seq_len_b__ssa_v0: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens__ssa_v0, [b__idx_v0])
            tok_blocks__ssa_v0: pl.Scalar[pl.INDEX] = (pl.cast(seq_len_b__ssa_v0, pl.INDEX) + 4 - 1) // 4
            for p0_idx__idx_v0, (k_cache__iter_v3, out__iter_v3, v_cache__iter_v3) in pl.range(tok_blocks__ssa_v0, init_values=(k_cache__iter_v1, out__iter_v1, v_cache__iter_v1)):
                p0__ssa_v0: pl.Scalar[pl.INDEX] = p0_idx__idx_v0 * 4
                valid_tok__ssa_v0: pl.Scalar[pl.INDEX] = pl.min(4, pl.cast(seq_len_b__ssa_v0, pl.INDEX) - p0__ssa_v0)
                sq_sum__ssa_v0: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.create([4, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                ret0__out: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.create([4, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                inv_rms__ssa_v0: pl.Tensor[[4, 1], pl.FP32] = self.qwen3_prefill_layer_incore_0(b__idx_v0, hidden_states__ssa_v0, p0__ssa_v0, sq_sum__ssa_v0, valid_tok__ssa_v0, ret0__out)
                q_proj_tile__ssa_v0: pl.Tensor[[4, 5120], pl.BF16] = pl.tensor.create([4, 5120], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                k_proj_tile__ssa_v0: pl.Tensor[[4, 1024], pl.BF16] = pl.tensor.create([4, 1024], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                v_proj_tile__ssa_v0: pl.Tensor[[4, 1024], pl.BF16] = pl.tensor.create([4, 1024], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                for ob__co_idx_v0, (q_proj_tile__co_l0_iter_v1,) in pl.parallel(10, init_values=(q_proj_tile__ssa_v0,)):
                    q_proj_tile__co_l1_rv_v1: pl.Tensor[[4, 5120], pl.BF16] = self.qwen3_prefill_layer_incore_1(b__idx_v0, hidden_states__ssa_v0, input_rms_weight__ssa_v0, inv_rms__ssa_v0, ob__co_idx_v0, p0__ssa_v0, q_proj_tile__co_l0_iter_v1, valid_tok__ssa_v0, wq__ssa_v0)
                    q_proj_tile__co_l0_rv_v1: pl.Tensor[[4, 5120], pl.BF16] = pl.yield_(q_proj_tile__co_l1_rv_v1)
                for ob__co_idx_v0_1, (k_proj_tile__co_l0_iter_v1, v_proj_tile__co_l0_iter_v1) in pl.parallel(4, init_values=(k_proj_tile__ssa_v0, v_proj_tile__ssa_v0)):
                    ret__tmp_v0: pl.Tuple[pl.Tensor[[4, 1024], pl.BF16], pl.Tensor[[4, 1024], pl.BF16]] = self.qwen3_prefill_layer_incore_2(b__idx_v0, hidden_states__ssa_v0, input_rms_weight__ssa_v0, inv_rms__ssa_v0, k_proj_tile__co_l0_iter_v1, ob__co_idx_v0_1, p0__ssa_v0, v_proj_tile__co_l0_iter_v1, valid_tok__ssa_v0, wk__ssa_v0, wv__ssa_v0)
                    k_proj_tile__co_l1_rv_v1: pl.Tensor[[4, 1024], pl.BF16] = ret__tmp_v0[0]
                    v_proj_tile__co_l1_rv_v1: pl.Tensor[[4, 1024], pl.BF16] = ret__tmp_v0[1]
                    k_proj_tile__co_l0_rv_v1, v_proj_tile__co_l0_rv_v1 = pl.yield_(k_proj_tile__co_l1_rv_v1, v_proj_tile__co_l1_rv_v1)
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
                    attn_row__ssa_v0: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.create([1, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    ret0__out_1: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.create([1, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    attn_row__ssa_v1: pl.Tensor[[1, 5120], pl.FP32] = self.qwen3_prefill_layer_incore_3(attn_row__ssa_v0, ret0__out_1)
                    for kvh__co_idx_v0, (k_cache__co_l0_iter_v7, v_cache__co_l0_iter_v7) in pl.parallel(2, init_values=(k_cache__iter_v5, v_cache__iter_v5)):
                        ret__tmp_v0_1: pl.Tuple[pl.Tensor[[524288, 128], pl.BF16], pl.Tensor[[524288, 128], pl.BF16]] = self.qwen3_prefill_layer_incore_4(b__idx_v0, cos_hi__ssa_v0, cos_lo__ssa_v0, k_cache__co_l0_iter_v7, k_proj_tile__co_l0_rv_v1, kvh__co_idx_v0, pos__ssa_v0, sin_hi__ssa_v0, sin_lo__ssa_v0, ti__idx_v0, v_cache__co_l0_iter_v7, v_proj_tile__co_l0_rv_v1)
                        k_cache__co_l1_rv_v7: pl.Tensor[[524288, 128], pl.BF16] = ret__tmp_v0_1[0]
                        v_cache__co_l1_rv_v7: pl.Tensor[[524288, 128], pl.BF16] = ret__tmp_v0_1[1]
                        k_cache__co_l0_rv_v7, v_cache__co_l0_rv_v7 = pl.yield_(k_cache__co_l1_rv_v7, v_cache__co_l1_rv_v7)
                    for h__co_idx_v0, (attn_row__co_l0_iter_v2,) in pl.parallel(8, init_values=(attn_row__ssa_v1,)):
                        attn_row__co_l1_rv_v2: pl.Tensor[[1, 5120], pl.FP32] = self.qwen3_prefill_layer_incore_5(attn_row__co_l0_iter_v2, b__idx_v0, cos_hi__ssa_v0, cos_lo__ssa_v0, ctx_blocks__ssa_v0, ctx_len__ssa_v0, h__co_idx_v0, k_cache__co_l0_rv_v7, q_proj_tile__co_l0_rv_v1, sin_hi__ssa_v0, sin_lo__ssa_v0, ti__idx_v0, v_cache__co_l0_rv_v7)
                        attn_row__co_l0_rv_v2: pl.Tensor[[1, 5120], pl.FP32] = pl.yield_(attn_row__co_l1_rv_v2)
                    attn_tile__ssa_v4: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.assemble(attn_tile__iter_v2, attn_row__co_l0_rv_v2, [ti__idx_v0, 0])
                    attn_tile__rv_v3, k_cache__rv_v6, v_cache__rv_v6 = pl.yield_(attn_tile__ssa_v4, k_cache__co_l0_rv_v7, v_cache__co_l0_rv_v7)
                resid1_tile__ssa_v0: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.create([4, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for ob__co_idx_v0_2, (resid1_tile__co_l0_iter_v1,) in pl.parallel(10, init_values=(resid1_tile__ssa_v0,)):
                    resid1_tile__co_l1_rv_v1: pl.Tensor[[4, 5120], pl.FP32] = self.qwen3_prefill_layer_incore_6(attn_tile__rv_v3, b__idx_v0, hidden_states__ssa_v0, ob__co_idx_v0_2, p0__ssa_v0, resid1_tile__co_l0_iter_v1, valid_tok__ssa_v0, wo__ssa_v0)
                    resid1_tile__co_l0_rv_v1: pl.Tensor[[4, 5120], pl.FP32] = pl.yield_(resid1_tile__co_l1_rv_v1)
                sq_sum__ssa_v5: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.create([4, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                ret0__out_2: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.create([4, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                inv_rms__ssa_v1: pl.Tensor[[4, 1], pl.FP32] = self.qwen3_prefill_layer_incore_7(resid1_tile__co_l0_rv_v1, sq_sum__ssa_v5, ret0__out_2)
                post_norm_tile__ssa_v0: pl.Tensor[[4, 5120], pl.BF16] = pl.tensor.create([4, 5120], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                down_proj_tile__ssa_v0: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.create([4, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                ret0__out_3: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.create([4, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                ret__tmp_v0_2: pl.Tuple[pl.Tensor[[4, 5120], pl.FP32], pl.Tensor[[4, 5120], pl.BF16]] = self.qwen3_prefill_layer_incore_8(down_proj_tile__ssa_v0, inv_rms__ssa_v1, post_norm_tile__ssa_v0, post_rms_weight__ssa_v0, resid1_tile__co_l0_rv_v1, ret0__out_3)
                down_proj_tile__ssa_v1: pl.Tensor[[4, 5120], pl.FP32] = ret__tmp_v0_2[0]
                post_norm_tile__rv_v2: pl.Tensor[[4, 5120], pl.BF16] = ret__tmp_v0_2[1]
                for ob__idx_v0, (down_proj_tile__iter_v2, out__iter_v5) in pl.range(400, init_values=(down_proj_tile__ssa_v1, out__iter_v3)):
                    o0__ssa_v1: pl.Scalar[pl.INDEX] = ob__idx_v0 * 64
                    gate_acc__ssa_v0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.create([4, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    up_acc__ssa_v0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.create([4, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    ret0__out_4: pl.Tensor[[4, 64], pl.BF16] = pl.tensor.create([4, 64], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                    mlp_chunk_bf16__ssa_v0: pl.Tensor[[4, 64], pl.BF16] = self.qwen3_prefill_layer_incore_9(gate_acc__ssa_v0, o0__ssa_v1, post_norm_tile__rv_v2, up_acc__ssa_v0, w_gate__ssa_v0, w_up__ssa_v0, ret0__out_4)
                    for dob__co_idx_v0, (down_proj_tile__co_l0_iter_v4, out__co_l0_iter_v7) in pl.parallel(10, init_values=(down_proj_tile__iter_v2, out__iter_v5)):
                        ret__tmp_v0_3: pl.Tuple[pl.Tensor[[4, 5120], pl.FP32], pl.Tensor[[16, 4096, 5120], pl.BF16]] = self.qwen3_prefill_layer_incore_10(b__idx_v0, dob__co_idx_v0, down_proj_tile__co_l0_iter_v4, mlp_chunk_bf16__ssa_v0, o0__ssa_v1, ob__idx_v0, out__co_l0_iter_v7, p0__ssa_v0, resid1_tile__co_l0_rv_v1, w_down__ssa_v0)
                        down_proj_tile__co_l1_rv_v4: pl.Tensor[[4, 5120], pl.FP32] = ret__tmp_v0_3[0]
                        out__co_l1_rv_v7: pl.Tensor[[16, 4096, 5120], pl.BF16] = ret__tmp_v0_3[1]
                        down_proj_tile__co_l0_rv_v4, out__co_l0_rv_v7 = pl.yield_(down_proj_tile__co_l1_rv_v4, out__co_l1_rv_v7)
                    down_proj_tile__rv_v3, out__rv_v6 = pl.yield_(down_proj_tile__co_l0_rv_v4, out__co_l0_rv_v7)
                k_cache__rv_v4, out__rv_v4, v_cache__rv_v4 = pl.yield_(k_cache__rv_v6, out__rv_v6, v_cache__rv_v6)
            k_cache__rv_v2, out__rv_v2, v_cache__rv_v2 = pl.yield_(k_cache__rv_v4, out__rv_v4, v_cache__rv_v4)
        return out__rv_v2